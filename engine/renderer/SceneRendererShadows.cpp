// Shadow passes: the sun cascade atlas, virtual shadow map pages and the local light atlas (see SceneRendererFrame.hpp).
#include "renderer/SceneRenderer.hpp"
#include "renderer/SceneRendererFrame.hpp"

#include "assets/GltfImporter.hpp"

#include "assets/AssetLibrary.hpp"
#include "assets/Material.hpp"
#include "assets/MaterialGraph.hpp"
#include "assets/MeshLod.hpp"

#include "core/FileSystem.hpp"
#include "core/JobSystem.hpp"
#include "core/Log.hpp"
#include "renderer/FrameResources.hpp"
#include "renderer/GpuTypes.hpp"
#include "renderer/LightGrid.hpp"
#include "renderer/LightProbes.hpp"
#include "renderer/RenderContext.hpp"
#include "scene/Landscape.hpp"
#include "scene/Scene.hpp"

#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/packing.hpp>
#include <glm/matrix.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <optional>
#include <span>
#include <string>


namespace lumora::render {

void SceneRenderer::addShadowPasses(SceneFrame& f) {
    using Draw = SceneDraw;
    using LocalTile = LocalShadowTile;
    nvrhi::IDevice* gpu = f.gpu;
    SceneDrawStats& stats = f.stats;
    const RenderSettings& settings = f.settings;
    RenderGraph& graph = f.graph;
    const DrawState stateFor = f.stateFor;
    const bool shadows = f.shadows;
    const bool virtualShadows = f.virtualShadows;
    const bool localShadows = f.localShadows;
    const std::vector<Draw>& casters = f.casters;
    const auto& cascadeMatrices = f.cascadeMatrices;
    const std::uint32_t tile = f.tile;
    const std::uint32_t atlasSize = f.atlasSize;
    const std::uint32_t localAtlasSize = f.localAtlasSize;
    const VsmFrame& vsmFrame = f.vsmFrame;
    const std::vector<LocalTile>& localTiles = f.localTiles;
    auto shadowPipeline = [this](const Draw& d) -> nvrhi::IGraphicsPipeline* {
        if (d.mesh->skinned()) return shadowSkinned_;
        return (d.variant & kMasked) != 0 ? shadowMasked_ : shadow_;
    };

    RGImage shadowMap{};
    if (shadows && !virtualShadows) {
        // Per-cascade caster culling against the cascade's side planes. Casters between the sun
        // and the cascade are kept: the light-facing plane never culls. One cascade per thread.
        std::array<std::vector<Draw>, kShadowCascades> cascadeCasters;
        JobSystem::global().parallelFor(kShadowCascades, 1, [&](std::size_t begin, std::size_t end) {
            for (std::size_t c = begin; c < end; ++c) {
                Frustum box = Frustum::fromViewProjection(cascadeMatrices[c]);
                box.planes[4] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
                for (const Draw& d : casters) {
                    if (box.intersectsAabb(d.center, d.extent)) cascadeCasters[c].push_back(d);
                }
                cascadeCasters[c] = instanced(cascadeCasters[c]);
            }
        });
        std::size_t casterDraws = 0;
        for (const auto& list : cascadeCasters) casterDraws += list.size();
        stats.drawCalls += static_cast<std::uint32_t>(casterDraws);
        const bool parallelShadow = casterDraws >= settings.parallelRecordingMinDraws;
        stats.parallelRecording = stats.parallelRecording || parallelShadow;
        shadowMap = graph.createImage("SunShadowAtlas", {ctx_.depthFormat(), atlasSize, atlasSize});
        graph.addRasterPass(
            "SunShadow",
            [&](PassBuilder& b) {
                b.writeDepth(shadowMap, LoadOp::Clear, 1.0f);
                b.recordInParallel(parallelShadow);
            },
            [this, stateFor, shadowPipeline, tile, cascadeCasters = std::move(cascadeCasters)](PassContext& ctx) {
                // One cascade per command list when recorded in parallel.
                ctx.recordParallel(kShadowCascades, 1, [&](nvrhi::ICommandList* cmd, std::size_t begin, std::size_t end) {
                    for (std::size_t c = begin; c < end; ++c) {
                        const float x = static_cast<float>((c % 2) * tile);
                        const float y = static_cast<float>((c / 2) * tile);
                        nvrhi::GraphicsState state = stateFor(shadow_, ctx);
                        state.viewport = nvrhi::ViewportState().addViewportAndScissorRect(
                            nvrhi::Viewport(x, x + static_cast<float>(tile), y, y + static_cast<float>(tile), 0.0f, 1.0f));
                        const GpuMesh* mesh = nullptr;
                        std::uint8_t variant = 0xff;
                        for (const Draw& d : cascadeCasters[c]) {
                            if (d.mesh != mesh || d.variant != variant) {
                                variant = d.variant;
                                mesh = d.mesh;
                                state.pipeline = shadowPipeline(d);
                                bindMesh(state, *mesh);
                                cmd->setGraphicsState(state);
                            }
                            PushConstants pc;
                            pc.index0 = d.object;
                            pc.index1 = static_cast<std::uint32_t>(c);
                            push(cmd, pc);
                            cmd->drawIndexed(nvrhi::DrawArguments().setVertexCount(mesh->indexCount).setInstanceCount(d.count));
                        }
                    }
                });
            });
    }

    RGImage vsmPool{};
    RGImage vsmPageTable{};
    if (virtualShadows) {
        if (!vsmPool_) {
            nvrhi::TextureDesc desc;
            desc.width = desc.height = VirtualShadowMap::poolSize();
            desc.format = ctx_.depthFormat();
            desc.debugName = "VirtualShadowPool";
            desc.isRenderTarget = true;
            desc.setClearValue(nvrhi::Color(1.0f));
            desc.initialState = nvrhi::ResourceStates::ShaderResource;
            desc.keepInitialState = true;
            vsmPool_ = gpu->createTexture(desc);
            nvrhi::TextureDesc table;
            table.width = VirtualShadowMap::kPagesPerLevel;
            table.height = VirtualShadowMap::kPagesPerLevel * VirtualShadowMap::kMaxLevels;
            table.format = nvrhi::Format::RGBA32_FLOAT;
            table.debugName = "VirtualShadowPageTable";
            table.initialState = nvrhi::ResourceStates::ShaderResource;
            table.keepInitialState = true;
            vsmPageTable_ = gpu->createTexture(table);
            if (!vsmPool_ || !vsmPageTable_) throw rhi::VulkanError("could not create the virtual shadow map", VK_ERROR_OUT_OF_DEVICE_MEMORY);
            vsmPoolIndex_ = ctx_.bindless().registerTexture(vsmPool_);
            vsmPageTableIndex_ = ctx_.bindless().registerTexture(vsmPageTable_);
            // Created once, before any page was drawn: this frame's pages are the first in it.
        }
        vsmPool = graph.importImage("VirtualShadowPool", {vsmPool_, vsmPoolIndex_});
        vsmPageTable = graph.importImage("VirtualShadowPageTable", {vsmPageTable_, vsmPageTableIndex_});
        // The page table, padded to every level the texture has room for.
        std::vector<float> table(static_cast<std::size_t>(VirtualShadowMap::kPagesPerLevel * VirtualShadowMap::kPagesPerLevel *
                                                          VirtualShadowMap::kMaxLevels) * 4, 0.0f);
        std::copy(vsmFrame.pageTable.begin(), vsmFrame.pageTable.end(), table.begin());
        graph.addPass(
            "VirtualShadowPageTable", [&](PassBuilder& b) { b.copyTo(vsmPageTable); },
            [texture = vsmPageTable_, table = std::move(table)](PassContext& ctx) {
                ctx.cmd->writeTexture(texture, 0, 0, table.data(), VirtualShadowMap::kPagesPerLevel * 4 * sizeof(float));
            });
        if (!vsmFrame.render.empty()) {
            // Casters per page, by their light-space rectangle.
            struct LightRect {
                glm::vec2 min, max;
            };
            const glm::vec3 right(glm::row(vsmFrame.lightView, 0));
            const glm::vec3 up(glm::row(vsmFrame.lightView, 1));
            std::vector<LightRect> rects(casters.size());
            for (std::size_t i = 0; i < casters.size(); ++i) {
                const glm::vec2 c(glm::dot(right, casters[i].center), glm::dot(up, casters[i].center));
                const glm::vec2 e(glm::dot(glm::abs(right), casters[i].extent), glm::dot(glm::abs(up), casters[i].extent));
                rects[i] = {c - e, c + e};
            }
            std::vector<std::vector<Draw>> pageCasters(vsmFrame.render.size());
            JobSystem::global().parallelFor(vsmFrame.render.size(), 16, [&](std::size_t begin, std::size_t end) {
                for (std::size_t p = begin; p < end; ++p) {
                    const VsmPageDraw& page = vsmFrame.render[p];
                    for (std::size_t i = 0; i < casters.size(); ++i) {
                        const LightRect& r = rects[i];
                        if (r.max.x < page.rectMin.x || r.min.x > page.rectMax.x || r.max.y < page.rectMin.y || r.min.y > page.rectMax.y) {
                            continue;
                        }
                        pageCasters[p].push_back(casters[i]);
                    }
                    pageCasters[p] = instanced(pageCasters[p]);
                }
            });
            std::size_t pageDraws = 0;
            for (const auto& list : pageCasters) pageDraws += list.size() + 1;
            stats.drawCalls += static_cast<std::uint32_t>(pageDraws);
            const bool parallelPages = pageDraws >= settings.parallelRecordingMinDraws;
            stats.parallelRecording = stats.parallelRecording || parallelPages;
            graph.addRasterPass(
                "VirtualShadowPages",
                [&](PassBuilder& b) {
                    b.writeDepth(vsmPool, LoadOp::Load);  // cached pages stay
                    b.recordInParallel(parallelPages);
                },
                [this, stateFor, shadowPipeline, pages = vsmFrame.render, pageCasters = std::move(pageCasters)](PassContext& ctx) {
                    ctx.recordParallel(pages.size(), 8, [&](nvrhi::ICommandList* cmd, std::size_t begin, std::size_t end) {
                        constexpr float texels = static_cast<float>(VirtualShadowMap::kPageTexels);
                        for (std::size_t p = begin; p < end; ++p) {
                            const VsmPageDraw& page = pages[p];
                            const float x = static_cast<float>(page.physical % VirtualShadowMap::kPoolPages) * texels;
                            const float y = static_cast<float>(page.physical / VirtualShadowMap::kPoolPages) * texels;
                            nvrhi::GraphicsState state = stateFor(vsmClear_, ctx);
                            state.viewport = nvrhi::ViewportState().addViewportAndScissorRect(
                                nvrhi::Viewport(x, x + texels, y, y + texels, 0.0f, 1.0f));
                            cmd->setGraphicsState(state);
                            push(cmd, PushConstants{});
                            cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
                            const GpuMesh* mesh = nullptr;
                            std::uint8_t variant = 0xff;
                            for (const Draw& d : pageCasters[p]) {
                                if (d.mesh != mesh || d.variant != variant) {
                                    variant = d.variant;
                                    mesh = d.mesh;
                                    state.pipeline = shadowPipeline(d);
                                    bindMesh(state, *mesh);
                                    cmd->setGraphicsState(state);
                                }
                                PushConstants pc;
                                pc.index0 = d.object;
                                pc.flags = 2u;  // a virtual shadow map page
                                pc.param0 = page.center.x;
                                pc.param1 = page.center.y;
                                pc.param2 = page.invHalf;
                                push(cmd, pc);
                                cmd->drawIndexed(nvrhi::DrawArguments().setVertexCount(mesh->indexCount).setInstanceCount(d.count));
                            }
                        }
                    });
                });
        }
    }

    RGImage localAtlas{};
    if (localShadows) {
        ensureLocalShadowAtlas(localAtlasSize);
        localAtlas = graph.importImage("LocalShadowAtlas", {localAtlas_, localAtlasIndex_});
        // Casters per tile: inside the light's range and the tile's frustum; tiles whose light and
        // casters are as they were keep their contents.
        std::vector<std::vector<Draw>> tileCasters(localTiles.size());
        std::vector<std::uint64_t> contents(localTiles.size());
        JobSystem::global().parallelFor(localTiles.size(), 1, [&](std::size_t begin, std::size_t end) {
            for (std::size_t t = begin; t < end; ++t) {
                const LocalTile& localTile = localTiles[t];
                const Frustum frustumOfTile = Frustum::fromViewProjection(localTile.viewProjection);
                std::uint64_t h = 1469598103934665603ull;
                for (int c = 0; c < 4; ++c) {
                    for (int r = 0; r < 4; ++r) h = h * 1099511628211ull ^ std::bit_cast<std::uint32_t>(localTile.viewProjection[c][r]);
                }
                for (const Draw& d : casters) {
                    if (glm::length(d.center - localTile.lightPosition) - glm::length(d.extent) > localTile.range) continue;
                    if (!frustumOfTile.intersectsAabb(d.center, d.extent)) continue;
                    tileCasters[t].push_back(d);
                    h = (h ^ d.state) * 1099511628211ull;
                    h = (h ^ d.key) * 1099511628211ull;
                    if (d.mesh->skinned()) h ^= static_cast<std::uint64_t>(frameCounter_) * 0x9E3779B97F4A7C15ull;  // animated
                }
                contents[t] = h == 0 ? 1 : h;
                tileCasters[t] = instanced(tileCasters[t]);
            }
        });
        std::vector<std::size_t> redraw;
        for (std::size_t t = 0; t < localTiles.size(); ++t) {
            std::uint64_t& held = localTileContents_[localTiles[t].index];
            if (held == contents[t]) continue;
            held = contents[t];
            redraw.push_back(t);
            stats.drawCalls += static_cast<std::uint32_t>(tileCasters[t].size()) + 1;
        }
        stats.localShadowTiles = static_cast<std::uint32_t>(localTiles.size());
        stats.localShadowTilesDrawn = static_cast<std::uint32_t>(redraw.size());
        if (!redraw.empty()) {
            const std::uint32_t tileSize = localAtlasSize / kLocalShadowGrid;
            std::vector<std::pair<std::uint32_t, std::vector<Draw>>> work;
            for (const std::size_t t : redraw) work.emplace_back(localTiles[t].index, std::move(tileCasters[t]));
            graph.addRasterPass(
                "LocalShadows", [&](PassBuilder& b) { b.writeDepth(localAtlas, LoadOp::Load); },  // cached tiles stay
                [this, stateFor, shadowPipeline, tileSize, work = std::move(work)](PassContext& ctx) {
                    for (const auto& [index, tileDraws] : work) {
                        const float x = static_cast<float>((index % kLocalShadowGrid) * tileSize);
                        const float y = static_cast<float>((index / kLocalShadowGrid) * tileSize);
                        nvrhi::GraphicsState state = stateFor(vsmClear_, ctx);  // clear the tile to the far depth
                        state.viewport = nvrhi::ViewportState().addViewportAndScissorRect(
                            nvrhi::Viewport(x, x + static_cast<float>(tileSize), y, y + static_cast<float>(tileSize), 0.0f, 1.0f));
                        ctx.cmd->setGraphicsState(state);
                        push(ctx.cmd, PushConstants{});
                        ctx.cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
                        const GpuMesh* mesh = nullptr;
                        std::uint8_t variant = 0xff;
                        for (const Draw& d : tileDraws) {
                            if (d.mesh != mesh || d.variant != variant) {
                                variant = d.variant;
                                mesh = d.mesh;
                                state.pipeline = shadowPipeline(d);
                                bindMesh(state, *mesh);
                                ctx.cmd->setGraphicsState(state);
                            }
                            PushConstants pc;
                            pc.index0 = d.object;
                            pc.index1 = index;
                            pc.flags = 1u;  // a local shadow tile
                            push(ctx.cmd, pc);
                            ctx.cmd->drawIndexed(nvrhi::DrawArguments().setVertexCount(mesh->indexCount).setInstanceCount(d.count));
                        }
                    }
                });
        }
    }
    f.shadowMap = shadowMap;
    f.vsmPool = vsmPool;
    f.vsmPageTable = vsmPageTable;
    f.localAtlas = localAtlas;
}

} // namespace lumora::render
