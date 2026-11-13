// Passes on the final image: the gizmo overlay with behaviours' debug shapes, sprites and the HUD (see SceneRendererFrame.hpp).
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

void SceneRenderer::addOverlays(SceneFrame& f) {
    nvrhi::IDevice* gpu = f.gpu;
    SceneDrawStats& stats = f.stats;
    const CameraView& camera = f.camera;
    const RenderSettings& settings = f.settings;
    RenderGraph& graph = f.graph;
    FrameResources& frame = f.frame;
    const DrawState stateFor = f.stateFor;
    const VkExtent2D extent = f.extent;
    const RGImage output = f.output;
    const nvrhi::Format outputFormat = f.outputFormat;
    const std::uint32_t sampler = ctx_.linearClampSampler();

    // ---- Overlay (gizmos): unlit, alpha blended, on top of the final image.
    // Debug shapes from behaviours join the overlay: thin solids, about 2 px wide on screen.
    OverlayGeometry debugGeometry;
    if (settings.debug != nullptr && !settings.debug->empty()) {
        const glm::vec3 forward(-camera.view[0][2], -camera.view[1][2], -camera.view[2][2]);
        OverlayBuilder builder(debugGeometry, forward);
        const float pixel = 2.0f * std::tan(camera.fovYRadians * 0.5f) / std::max(1.0f, static_cast<float>(extent.height));
        auto width = [&](const glm::vec3& at) { return std::max(0.002f, glm::length(at - camera.position) * pixel); };
        for (const DebugDraw::Line& l : settings.debug->lines) {
            if (glm::length(l.to - l.from) > 1e-5f) builder.cylinder(l.from, l.to, width((l.from + l.to) * 0.5f), l.color, 6);
        }
        for (const DebugDraw::Box& b : settings.debug->boxes) {
            const glm::vec3 axes[3] = {{b.halfExtent.x, 0, 0}, {0, b.halfExtent.y, 0}, {0, 0, b.halfExtent.z}};
            builder.wireBox(b.center, axes, width(b.center), b.color);
        }
        for (const DebugDraw::Sphere& s : settings.debug->spheres) {
            for (const glm::vec3 axis : {glm::vec3(1, 0, 0), glm::vec3(0, 1, 0), glm::vec3(0, 0, 1)}) {
                builder.torus(s.center, axis, s.radius, width(s.center), s.color, 40, 4);
            }
        }
    }
    std::vector<OverlayVertex> combinedOverlay;
    if (!debugGeometry.empty()) {
        if (settings.overlay != nullptr) combinedOverlay = settings.overlay->vertices;
        combinedOverlay.insert(combinedOverlay.end(), debugGeometry.vertices.begin(), debugGeometry.vertices.end());
    }
    const std::vector<OverlayVertex>* overlayVertices =
        !combinedOverlay.empty() ? &combinedOverlay : settings.overlay != nullptr ? &settings.overlay->vertices : nullptr;
    if (overlayVertices != nullptr && !overlayVertices->empty()) {
        const auto& vertices = *overlayVertices;
        nvrhi::IBuffer* vb = frame.overlayVertices.write(gpu, std::as_bytes(std::span(vertices)), DynamicBuffer::Kind::Vertex,
                                                         sizeof(OverlayVertex), "OverlayVertices");
        PipelineDesc overlayDesc;
        overlayDesc.name = "Overlay";
        overlayDesc.vertexShader = "overlay.vert";
        overlayDesc.pixelShader = "overlay.frag";
        overlayDesc.input = VertexInput::Overlay;
        overlayDesc.blend = BlendMode::Alpha;
        nvrhi::IGraphicsPipeline* overlay = outputPipeline(overlay_, overlayDesc, outputFormat);
        const auto count = static_cast<std::uint32_t>(vertices.size());
        graph.addRasterPass(
            "Overlay", [&](PassBuilder& b) { b.writeColor(output, LoadOp::Load); },
            [=](PassContext& ctx) {
                nvrhi::GraphicsState state = stateFor(overlay, ctx);
                state.vertexBuffers = {nvrhi::VertexBufferBinding().setBuffer(vb).setSlot(0)};
                ctx.cmd->setGraphicsState(state);
                PushConstants pc;
                pc.flags = isSrgbFormat(outputFormat) ? 0u : 1u;  // the overlay draws into the output itself
                push(ctx.cmd, pc);
                ctx.cmd->draw(nvrhi::DrawArguments().setVertexCount(count));
            });
        ++stats.drawCalls;
    }
    // ---- Sprites: pictures at points in the world (editor icons), facing the viewer.
    if (settings.sprites != nullptr && !settings.sprites->empty()) {
        const SpriteAtlas& atlas = *settings.sprites->atlas;
        if (spriteAtlasSource_ != &atlas || spriteAtlasVersion_ != atlas.version || !spriteAtlas_) {
            if (spriteAtlasIndex_ != UINT32_MAX) ctx_.bindless().releaseTexture(spriteAtlasIndex_);
            const std::span<const std::byte> bytes = std::as_bytes(std::span(atlas.rgba));
            spriteAtlas_ = ctx_.createTexture(std::span(&bytes, 1), atlas.width, atlas.height, nvrhi::Format::RGBA8_UNORM,
                                              "SpriteAtlas");
            spriteAtlasIndex_ = ctx_.bindless().registerTexture(spriteAtlas_);
            spriteAtlasSource_ = &atlas;
            spriteAtlasVersion_ = atlas.version;
        }
        const glm::vec2 target(static_cast<float>(extent.width), static_cast<float>(extent.height));
        std::vector<HudVertex> vertices;
        for (const SpriteQuad& q : layoutSprites(*settings.sprites, camera, target)) {
            const Sprite& s = settings.sprites->sprites[q.index];
            const HudVertex tl{q.min, {s.uv.x, s.uv.y}, s.tint};
            const HudVertex tr{{q.max.x, q.min.y}, {s.uv.z, s.uv.y}, s.tint};
            const HudVertex bl{{q.min.x, q.max.y}, {s.uv.x, s.uv.w}, s.tint};
            const HudVertex br{q.max, {s.uv.z, s.uv.w}, s.tint};
            vertices.insert(vertices.end(), {tl, tr, br, tl, br, bl});
        }
        if (!vertices.empty()) {
            nvrhi::IBuffer* vb = frame.spriteVertices.write(gpu, std::as_bytes(std::span(vertices)),
                                                            DynamicBuffer::Kind::Vertex, sizeof(HudVertex), "SpriteVertices");
            PipelineDesc spriteDesc;
            spriteDesc.name = "HUD";
            spriteDesc.vertexShader = "hud.vert";
            spriteDesc.pixelShader = "hud.frag";
            spriteDesc.input = VertexInput::Hud;
            spriteDesc.blend = BlendMode::Alpha;
            nvrhi::IGraphicsPipeline* pipeline = outputPipeline(hud_, spriteDesc, outputFormat);
            const auto count = static_cast<std::uint32_t>(vertices.size());
            const std::uint32_t texture = spriteAtlasIndex_;
            const bool unorm = !isSrgbFormat(outputFormat);
            graph.addRasterPass(
                "Sprites", [&](PassBuilder& b) { b.writeColor(output, LoadOp::Load); },
                [=](PassContext& ctx) {
                    nvrhi::GraphicsState state = stateFor(pipeline, ctx);
                    state.vertexBuffers = {nvrhi::VertexBufferBinding().setBuffer(vb).setSlot(0)};
                    ctx.cmd->setGraphicsState(state);
                    PushConstants pc;
                    pc.index0 = texture;
                    pc.index1 = sampler;
                    pc.flags = (unorm ? 1u : 0u) | 2u;  // bit1: a colour picture, not glyph coverage
                    push(ctx.cmd, pc);
                    ctx.cmd->draw(nvrhi::DrawArguments().setVertexCount(count));
                });
            ++stats.drawCalls;
        }
    }
    // ---- HUD: the game's text and rectangles, on top of everything.
    if (settings.hud != nullptr && !settings.hud->empty()) {
        if (!hudFont_) hudFont_ = std::make_unique<HudFont>();
        std::vector<HudVertex> vertices;
        std::vector<HudFont::Batch> batches;
        AssetLibrary* assets = settings.assets;
        const bool block = settings.blockOnAssets;
        auto image = [&](const std::string& key) -> std::uint32_t {
            if (assets == nullptr || key.empty()) return UINT32_MAX;
            const std::uint32_t index = ctx_.textures().index(*assets, key, TextureKind::Color, block);
            return index == kNoTexture ? UINT32_MAX : index;
        };
        auto font = [&](const std::string& key) -> int {
            if (const int id = hudFont_->findFont(key); id >= 0) return id;
            std::vector<std::byte> bytes;  // empty: not a font, the built-in one (remembered)
            if (assets != nullptr) {
                if (auto path = assets->filePath(key)) {
                    if (auto read = fs::readBinaryFile(*path)) bytes = std::move(*read);
                }
            }
            if (bytes.empty()) log::warn("Renderer", "HUD font '{}' could not be read; using the built-in one", key);
            return hudFont_->addFont(key, std::move(bytes));
        };
        hudFont_->layout(*settings.hud, vertices, batches, image, font);
        if (hudFont_->takeDirty() || !hudAtlas_) {
            // New glyphs: a fresh atlas texture (the old one may still be in flight).
            if (hudAtlas_) retiredHudAtlases_.push_back({hudAtlas_, hudAtlasIndex_, frameCounter_});
            const auto& atlas = hudFont_->atlas();
            const std::span<const std::byte> bytes = std::as_bytes(std::span(atlas));
            hudAtlas_ = ctx_.createTexture(std::span(&bytes, 1), hudFont_->atlasWidth(), hudFont_->atlasHeight(),
                                           nvrhi::Format::R8_UNORM, "HudGlyphAtlas");
            hudAtlasIndex_ = ctx_.bindless().registerTexture(hudAtlas_);
        }
        std::erase_if(retiredHudAtlases_, [&](const RetiredAtlas& r) {
            if (r.frame + 8 > frameCounter_) return false;
            ctx_.bindless().releaseTexture(r.index);
            return true;
        });
        // The canvas is laid out for its screen size; scale it to this target.
        const glm::vec2 scale(static_cast<float>(extent.width) / std::max(settings.hud->screenSize.x, 1.0f),
                              static_cast<float>(extent.height) / std::max(settings.hud->screenSize.y, 1.0f));
        for (HudVertex& v : vertices) v.position *= scale;
        if (!vertices.empty()) {
            nvrhi::IBuffer* vb = frame.hudVertices.write(gpu, std::as_bytes(std::span(vertices)), DynamicBuffer::Kind::Vertex,
                                                         sizeof(HudVertex), "HudVertices");
            PipelineDesc hudDesc;
            hudDesc.name = "HUD";
            hudDesc.vertexShader = "hud.vert";
            hudDesc.pixelShader = "hud.frag";
            hudDesc.input = VertexInput::Hud;
            hudDesc.blend = BlendMode::Alpha;
            nvrhi::IGraphicsPipeline* hud = outputPipeline(hud_, hudDesc, outputFormat);
            const std::uint32_t atlas = hudAtlasIndex_;
            const bool unorm = !isSrgbFormat(outputFormat);
            graph.addRasterPass(
                "HUD", [&](PassBuilder& b) { b.writeColor(output, LoadOp::Load); },
                [=](PassContext& ctx) {
                    nvrhi::GraphicsState state = stateFor(hud, ctx);
                    state.vertexBuffers = {nvrhi::VertexBufferBinding().setBuffer(vb).setSlot(0)};
                    ctx.cmd->setGraphicsState(state);
                    for (const HudFont::Batch& batch : batches) {  // in order: later ones on top
                        PushConstants pc;
                        pc.index0 = batch.picture ? batch.texture : atlas;
                        pc.index1 = sampler;
                        pc.flags = (unorm ? 1u : 0u) | (batch.picture ? 2u : 0u);
                        push(ctx.cmd, pc);
                        ctx.cmd->draw(nvrhi::DrawArguments().setVertexCount(batch.count).setStartVertexLocation(batch.first));
                    }
                });
            stats.drawCalls += static_cast<std::uint32_t>(batches.size());
        }
    }
}

} // namespace lumora::render
