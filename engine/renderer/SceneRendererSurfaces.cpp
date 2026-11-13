// Surface passes: depth prepass, ambient occlusion, contact shadows, fog, sky, opaque and see-through surfaces, decals, reflections, water and the grid (see SceneRendererFrame.hpp).
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

void SceneRenderer::addDepthPasses(SceneFrame& f) {
    using Draw = SceneDraw;
    nvrhi::IDevice* gpu = f.gpu;
    SceneDrawStats& stats = f.stats;
    const RenderSettings& settings = f.settings;
    RenderGraph& graph = f.graph;
    const DrawState stateFor = f.stateFor;
    const VkExtent2D extent = f.renderExtent;
    const FrameConstants& fc = f.fc;
    const std::vector<Draw>& draws = f.draws;
    const RGImage depth = f.depth;
    const RGImage hdr = f.hdr;
    const bool shadows = f.shadows;
    const bool virtualShadows = f.virtualShadows;
    const bool localShadows = f.localShadows;
    const bool haveSun = f.haveSun;
    const bool volumetricFog = f.volumetricFog;
    const glm::uvec3 froxels = f.froxels;
    const RGImage shadowMap = f.shadowMap;
    const RGImage vsmPool = f.vsmPool;
    const RGImage vsmPageTable = f.vsmPageTable;
    const RGImage localAtlas = f.localAtlas;
    const VsmFrame& vsmFrame = f.vsmFrame;
    auto prepassPipeline = [this](const Draw& d) -> nvrhi::IGraphicsPipeline* {
        if (d.mesh->skinned()) return depthPrepassSkinned_;
        switch (d.variant) {
            case kMasked: return depthPrepassMasked_;
            case kTwoSided: return depthPrepassTwoSided_;
            case kMasked | kTwoSided: return depthPrepassMaskedTwoSided_;
            default: return depthPrepass_;
        }
    };

    // ---- Depth prepass, then ambient occlusion from it ---------------------------------------
    const bool parallelPrepass = draws.size() >= settings.parallelRecordingMinDraws;
    const RGImage motion = graph.createImage("Motion", {kMotionFormat, extent.width, extent.height});
    graph.addRasterPass(
        "DepthPrepass",
        [&](PassBuilder& b) {
            b.writeColor(motion, LoadOp::Clear);  // no motion where nothing is drawn (TAA uses the camera)
            b.writeDepth(depth, LoadOp::Clear, 0.0f);  // reversed-Z: 0 is the far plane
            b.recordInParallel(parallelPrepass);
        },
        [this, stateFor, prepassPipeline, draws](PassContext& ctx) {  // a copy: runs after build() returns
            if (draws.empty()) return;
            ctx.recordParallel(draws.size(), 64, [&](nvrhi::ICommandList* cmd, std::size_t begin, std::size_t end) {
                nvrhi::GraphicsState state = stateFor(depthPrepass_, ctx);
                const GpuMesh* mesh = nullptr;
                std::uint8_t variant = 0xff;
                for (std::size_t i = begin; i < end; ++i) {
                    const Draw& d = draws[i];
                    if (d.mesh != mesh || d.variant != variant) {
                        variant = d.variant;
                        mesh = d.mesh;
                        state.pipeline = prepassPipeline(d);
                        bindMesh(state, *mesh);
                        cmd->setGraphicsState(state);
                    }
                    PushConstants pc;
                    pc.index0 = d.object;
                    push(cmd, pc);
                    cmd->drawIndexed(nvrhi::DrawArguments().setVertexCount(mesh->indexCount).setInstanceCount(d.count));
                }
            });
        });

    RGImage ao{};
    const bool ambientOcclusion = settings.ambientOcclusion && !draws.empty() && extent.width >= 8 && extent.height >= 8;
    // Virtual shadow map pages the visible surfaces sample, for a later frame to keep resident.
    if (virtualShadows && vsmMark_ && settings.virtualShadowGpuRequests) {
        VsmRequests& requests = vsmRequests_[vsmRequestFrame_ % vsmRequests_.size()];
        if (!requests.staging) {
            nvrhi::TextureDesc desc;
            desc.width = VirtualShadowMap::kPagesPerLevel;
            desc.height = VirtualShadowMap::kPagesPerLevel * VirtualShadowMap::kMaxLevels;
            desc.format = nvrhi::Format::R32_UINT;
            desc.debugName = "VsmRequestsReadback";
            requests.staging = gpu->createStagingTexture(desc, nvrhi::CpuAccessMode::Read);
        }
        const auto rows = static_cast<std::uint32_t>(VirtualShadowMap::kPagesPerLevel * vsmFrame.levels);
        const RGImage marks = graph.createImage("VsmRequests", {nvrhi::Format::R32_UINT, VirtualShadowMap::kPagesPerLevel, rows, true});
        nvrhi::IDescriptorTable* bindlessTable = ctx_.bindless().table();
        graph.addComputePass(
            "VsmMarkPages",
            [&](PassBuilder& b) {
                b.writeStorage(marks);
                b.sample(depth);
            },
            [this, marks, depth, bindlessTable, extent](PassContext& ctx) {
                ctx.cmd->clearTextureUInt(ctx.texture(marks), nvrhi::AllSubresources, 0u);
                ctx.cmd->setComputeState(nvrhi::ComputeState().setPipeline(vsmMark_).addBindingSet(ctx.frameBindings())
                                             .addBindingSet(bindlessTable).addBindingSet(ctx.storageBindings()));
                PushConstants pc;
                pc.index0 = ctx.bindlessIndex(depth);
                push(ctx.cmd, pc);
                ctx.cmd->dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
            });
        graph.addPass(
            "VsmRequestsReadback", [&](PassBuilder& b) { b.copyFrom(marks); },
            [marks, staging = requests.staging, rows](PassContext& ctx) {
                nvrhi::TextureSlice slice;
                slice.width = VirtualShadowMap::kPagesPerLevel;
                slice.height = rows;
                ctx.cmd->copyTexture(staging, slice, ctx.texture(marks), slice);
            });
        requests.pending = true;
        requests.levels = vsmFrame.levels;
        requests.windowMin = vsmFrame.windowMin;
        requests.generation = vsmFrame.generation;
        requests.viewProjection = fc.viewProjection;
        stats.virtualPagesFromSurfaces = vsmFrame.fromFeedback;
    }
    ++vsmRequestFrame_;

    if (ambientOcclusion) {
        const RGImage aoRaw = graph.createImage("SSAORaw", {kAoFormat, std::max(extent.width / 2, 1u),
                                                              std::max(extent.height / 2, 1u)});
        ao = graph.createImage("SSAO", {kAoFormat, extent.width, extent.height});
        const std::uint32_t linear = ctx_.linearClampSampler();
        graph.addRasterPass(
            "SSAO",
            [&](PassBuilder& b) {
                b.writeColor(aoRaw, LoadOp::DontCare);
                b.sample(depth);
            },
            [this, stateFor, depth, radius = settings.aoRadius, intensity = settings.aoIntensity](PassContext& ctx) {
                PushConstants pc;
                pc.index0 = ctx.bindlessIndex(depth);
                pc.param0 = radius;
                pc.param1 = intensity;
                pc.param2 = 0.02f;
                fullscreen(ctx.cmd, stateFor(ssao_, ctx), pc);
            });
        graph.addRasterPass(
            "SSAOBlur",
            [&](PassBuilder& b) {
                b.writeColor(ao, LoadOp::DontCare);
                b.sample(aoRaw);
                b.sample(depth);
            },
            [this, stateFor, depth, aoRaw, linear](PassContext& ctx) {
                PushConstants pc;
                pc.index0 = ctx.bindlessIndex(depth);
                pc.index1 = ctx.bindlessIndex(aoRaw);
                pc.index2 = linear;
                fullscreen(ctx.cmd, stateFor(ssaoBlur_, ctx), pc);
            });
        stats.ambientOcclusion = true;
    }

    // Contact shadows towards the sun (contact_shadows.hlsl), read by the opaque pass.
    RGImage contact{};
    if (settings.contactShadows && haveSun && contactShadows_ && !draws.empty() && extent.width >= 8 && extent.height >= 8) {
        contact = graph.createImage("ContactShadows", {kAoFormat, extent.width, extent.height, true});
        nvrhi::IDescriptorTable* bindlessTable = ctx_.bindless().table();
        graph.addComputePass(
            "ContactShadows",
            [&](PassBuilder& b) {
                b.writeStorage(contact);
                b.sample(depth);
            },
            // The jitter changes between frames only when TAA blends them (captures stay repeatable).
            [this, contact, depth, bindlessTable, extent, length = std::max(settings.contactShadowLength, 0.01f),
             frame = settings.temporalAntiAliasing ? taaFrame_ : 0u](PassContext& ctx) {
                ctx.cmd->setComputeState(nvrhi::ComputeState().setPipeline(contactShadows_).addBindingSet(ctx.frameBindings())
                                             .addBindingSet(bindlessTable).addBindingSet(ctx.storageBindings()));
                PushConstants pc;
                pc.index0 = ctx.bindlessIndex(depth);
                pc.param0 = length;
                pc.param1 = static_cast<float>(frame % 1024u);
                push(ctx.cmd, pc);
                ctx.cmd->dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
            });
        stats.contactShadows = true;
    }

    RGImage fogVolume{};  // integrated volumetric fog, read by the passes that apply fog
    if (volumetricFog) {
        const RGImage injected = graph.createImage("FogInjected", {kSurfaceFormat, froxels.x, froxels.y, true, froxels.z});
        fogVolume = graph.createImage("FogIntegrated", {kSurfaceFormat, froxels.x, froxels.y, true, froxels.z});
        nvrhi::IDescriptorTable* bindlessTable = ctx_.bindless().table();
        graph.addComputePass(
            "FogInject",
            [&](PassBuilder& b) {
                b.writeStorage(injected);
                if (shadows && !virtualShadows) b.sample(shadowMap);
                if (virtualShadows) {
                    b.sample(vsmPool);
                    b.sample(vsmPageTable);
                }
                if (localShadows) b.sample(localAtlas);
            },
            [this, bindlessTable, froxels, shadows, shadowMap, virtualShadows, vsmPool, vsmPageTable, localShadows, localAtlas](PassContext& ctx) {
                ctx.cmd->setComputeState(nvrhi::ComputeState().setPipeline(fogInject_).addBindingSet(ctx.frameBindings())
                                             .addBindingSet(bindlessTable).addBindingSet(ctx.storageBindings()));
                PushConstants pc;
                pc.index1 = virtualShadows ? ctx.bindlessIndex(vsmPool) : shadows ? ctx.bindlessIndex(shadowMap) : 0u;
                pc.param0 = std::bit_cast<float>(localShadows ? ctx.bindlessIndex(localAtlas) : 0u);
                pc.param1 = std::bit_cast<float>(virtualShadows ? ctx.bindlessIndex(vsmPageTable) : 0u);
                pc.flags = shadows ? 1u : 0u;
                push(ctx.cmd, pc);
                ctx.cmd->dispatch((froxels.x + 3) / 4, (froxels.y + 3) / 4, (froxels.z + 3) / 4);
            });
        graph.addComputePass(
            "FogIntegrate",
            [&](PassBuilder& b) {
                b.writeStorage(fogVolume);
                b.writeStorage(injected);  // u1, read
            },
            [this, bindlessTable, froxels](PassContext& ctx) {
                ctx.cmd->setComputeState(nvrhi::ComputeState().setPipeline(fogIntegrate_).addBindingSet(ctx.frameBindings())
                                             .addBindingSet(bindlessTable).addBindingSet(ctx.storageBindings()));
                push(ctx.cmd, PushConstants{});
                ctx.cmd->dispatch((froxels.x + 7) / 8, (froxels.y + 7) / 8, 1);
            });
    }
    graph.addRasterPass(
        "Sky",
        [&](PassBuilder& b) {
            b.writeColor(hdr, LoadOp::DontCare);
            if (fogVolume.valid()) b.sample(fogVolume);
        },
        [this, stateFor, fogVolume](PassContext& ctx) {
            PushConstants pc;
            if (fogVolume.valid()) pc.param2 = std::bit_cast<float>(ctx.bindlessIndex(fogVolume));
            fullscreen(ctx.cmd, stateFor(sky_, ctx), pc);
        });
    f.motion = motion;
    f.ao = ao;
    f.ambientOcclusion = ambientOcclusion;
    f.contact = contact;
    f.fogVolume = fogVolume;
}

void SceneRenderer::addSurfacePasses(SceneFrame& f) {
    using Draw = SceneDraw;
    SceneDrawStats& stats = f.stats;
    const RenderSettings& settings = f.settings;
    RenderGraph& graph = f.graph;
    const DrawState stateFor = f.stateFor;
    const VkExtent2D extent = f.renderExtent;
    std::vector<Draw>& draws = f.draws;  // moved into the opaque pass
    RGImage& hdr = f.hdr;
    const RGImage depth = f.depth;
    const bool reflections = f.reflections;
    const bool temporal = f.temporal;
    const bool shadows = f.shadows;
    const bool virtualShadows = f.virtualShadows;
    const bool localShadows = f.localShadows;
    const bool ambientOcclusion = f.ambientOcclusion;
    const bool transmissiveDraws = f.transmissiveDraws;
    const RGImage shadowMap = f.shadowMap;
    const RGImage vsmPool = f.vsmPool;
    const RGImage vsmPageTable = f.vsmPageTable;
    const RGImage localAtlas = f.localAtlas;
    const RGImage ao = f.ao;
    const RGImage fogVolume = f.fogVolume;
    const RGImage contact = f.contact;
    const std::vector<std::uint32_t>& decals = f.decals;
    const std::vector<std::uint32_t>& waters = f.waters;
    const GpuMesh* waterGrid = f.waterGrid;
    const std::vector<Draw>& transparents = f.transparents;
    auto opaquePipeline = [this, reflections](const Draw& d) -> nvrhi::IGraphicsPipeline* {
        if (reflections) {
            if (d.mesh->skinned()) return opaqueReflectionsSkinned_;
            return (d.variant & kTwoSided) != 0 ? opaqueReflectionsTwoSided_ : opaqueReflections_;
        }
        if (d.mesh->skinned()) return opaqueSkinned_;
        return (d.variant & kTwoSided) != 0 ? opaqueTwoSided_ : opaque_;
    };

    const bool parallelOpaque = draws.size() >= settings.parallelRecordingMinDraws;
    stats.parallelRecording = stats.parallelRecording || parallelOpaque;
    RGImage surfaceNormal{}, surfaceSpecular{};
    if (reflections) {
        surfaceNormal = graph.createImage("SurfaceNormalRoughness", {kSurfaceFormat, extent.width, extent.height});
        surfaceSpecular = graph.createImage("SurfaceSpecular", {kSurfaceFormat, extent.width, extent.height});
    }
    graph.addRasterPass(
        "Opaque",
        [&](PassBuilder& b) {
            b.writeColor(hdr, LoadOp::Load);
            if (reflections) {
                b.writeColor(surfaceNormal, LoadOp::Clear);
                b.writeColor(surfaceSpecular, LoadOp::Clear);
            }
            b.readDepth(depth);  // from the prepass; the opaque pass only tests against it
            if (shadows && !virtualShadows) b.sample(shadowMap);
            if (virtualShadows) {
                b.sample(vsmPool);
                b.sample(vsmPageTable);
            }
            if (localShadows) b.sample(localAtlas);
            if (ambientOcclusion) b.sample(ao);
            if (fogVolume.valid()) b.sample(fogVolume);
            if (contact.valid()) b.sample(contact);
            b.recordInParallel(parallelOpaque);
        },
        [this, stateFor, opaquePipeline, shadows, shadowMap, virtualShadows, vsmPool, vsmPageTable, localShadows, localAtlas, ambientOcclusion, ao,
         fogVolume, contact, debugCascades = settings.debugCascades,
         draws = std::move(draws)](PassContext& ctx) {
            if (draws.empty()) return;
            const std::uint32_t shadowIndex = virtualShadows ? ctx.bindlessIndex(vsmPool) : shadows ? ctx.bindlessIndex(shadowMap) : 0u;
            const float pageTableIndex = std::bit_cast<float>(virtualShadows ? ctx.bindlessIndex(vsmPageTable) : 0u);
            const std::uint32_t aoIndex = ambientOcclusion ? ctx.bindlessIndex(ao) : 0u;
            const float localAtlasIndex = std::bit_cast<float>(localShadows ? ctx.bindlessIndex(localAtlas) : 0u);
            const float fogIndex = std::bit_cast<float>(fogVolume.valid() ? ctx.bindlessIndex(fogVolume) : 0u);
            const float contactIndex = std::bit_cast<float>(contact.valid() ? ctx.bindlessIndex(contact) : 0u);
            const std::uint32_t flags = (shadows ? 1u : 0u) | (shadows && debugCascades ? 2u : 0u) | (ambientOcclusion ? 4u : 0u) |
                                        (contact.valid() ? 8u : 0u);
            // Draws are sorted by mesh, so each chunk rebinds buffers only where meshes change.
            ctx.recordParallel(draws.size(), 64, [&](nvrhi::ICommandList* cmd, std::size_t begin, std::size_t end) {
                nvrhi::GraphicsState state = stateFor(opaque_, ctx);
                const GpuMesh* mesh = nullptr;
                std::uint8_t variant = 0xff;
                for (std::size_t i = begin; i < end; ++i) {
                    const Draw& d = draws[i];
                    if (d.mesh != mesh || d.variant != variant) {
                        variant = d.variant;
                        mesh = d.mesh;
                        state.pipeline = opaquePipeline(d);
                        bindMesh(state, *mesh);
                        cmd->setGraphicsState(state);
                    }
                    PushConstants pc;
                    pc.index0 = d.object;
                    pc.index1 = shadowIndex;
                    pc.index2 = aoIndex;
                    pc.flags = flags;
                    pc.param0 = localAtlasIndex;
                    pc.param1 = pageTableIndex;
                    pc.param2 = fogIndex;
                    pc.param3 = contactIndex;
                    push(cmd, pc);
                    cmd->drawIndexed(nvrhi::DrawArguments().setVertexCount(mesh->indexCount).setInstanceCount(d.count));
                }
            });
        });

    if (!decals.empty()) {
        const GpuMesh* box = &ctx_.meshes().primitive(MeshPrimitive::Cube);
        const std::uint32_t sampler = ctx_.linearClampSampler();
        graph.addRasterPass(
            "Decals",
            [&](PassBuilder& b) {
                b.writeColor(hdr, LoadOp::Load);
                b.sample(depth);
            },
            [this, stateFor, box, depth, sampler, decals](PassContext& ctx) {
                nvrhi::GraphicsState state = stateFor(decal_, ctx);
                bindMesh(state, *box);
                ctx.cmd->setGraphicsState(state);
                for (const std::uint32_t object : decals) {
                    PushConstants pc;
                    pc.index0 = object;
                    pc.index1 = ctx.bindlessIndex(depth);
                    pc.index2 = sampler;
                    push(ctx.cmd, pc);
                    ctx.cmd->drawIndexed(nvrhi::DrawArguments().setVertexCount(box->indexCount));
                }
            });
        stats.drawCalls += static_cast<std::uint32_t>(decals.size());
    }

    RGImage opaqueColor{};  // the shaded opaque scene, before SSR (water reflects it)
    if (reflections) {
        opaqueColor = hdr;
        // Screen-space reflections: one ray per pixel, then shared between neighbours and put
        // into a new colour image, which the later passes draw over.
        const RGImage rays = graph.createImage("ReflectionRays", {kSurfaceFormat, extent.width, extent.height, true});
        const RGImage reflected = graph.createImage("SceneColorReflections", {ctx_.hdrFormat(), extent.width, extent.height, true});
        nvrhi::IDescriptorTable* bindlessTable = ctx_.bindless().table();
        PushConstants pc;
        pc.param1 = std::max(settings.reflectionDistance, 0.1f);
        pc.param2 = std::clamp(settings.reflectionRoughness, 0.05f, 1.0f);
        pc.param3 = static_cast<float>(taaFrame_ % 1024u);
        pc.flags = temporal ? 1u : 0u;  // jittered rays only when TAA blends them
        graph.addComputePass(
            "SSRTrace",
            [&](PassBuilder& b) {
                b.writeStorage(rays);
                b.sample(hdr);
                b.sample(depth);
                b.sample(surfaceNormal);
            },
            [this, hdr, depth, surfaceNormal, bindlessTable, extent, pc](PassContext& ctx) mutable {
                ctx.cmd->setComputeState(nvrhi::ComputeState().setPipeline(ssrTrace_).addBindingSet(ctx.frameBindings())
                                             .addBindingSet(bindlessTable).addBindingSet(ctx.storageBindings()));
                pc.index0 = ctx.bindlessIndex(hdr);
                pc.index1 = ctx.bindlessIndex(depth);
                pc.index2 = ctx.bindlessIndex(surfaceNormal);
                push(ctx.cmd, pc);
                ctx.cmd->dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
            });
        graph.addComputePass(
            "SSRResolve",
            [&](PassBuilder& b) {
                b.writeStorage(reflected);
                b.sample(hdr);
                b.sample(depth);
                b.sample(surfaceNormal);
                b.sample(surfaceSpecular);
                b.sample(rays);
            },
            [this, hdr, depth, surfaceNormal, surfaceSpecular, rays, bindlessTable, extent, pc](PassContext& ctx) mutable {
                ctx.cmd->setComputeState(nvrhi::ComputeState().setPipeline(ssrResolve_).addBindingSet(ctx.frameBindings())
                                             .addBindingSet(bindlessTable).addBindingSet(ctx.storageBindings()));
                pc.index0 = ctx.bindlessIndex(hdr);
                pc.index1 = ctx.bindlessIndex(depth);
                pc.index2 = ctx.bindlessIndex(surfaceNormal);
                pc.param0 = std::bit_cast<float>(ctx.bindlessIndex(rays));
                pc.param1 = std::bit_cast<float>(ctx.bindlessIndex(surfaceSpecular));
                push(ctx.cmd, pc);
                ctx.cmd->dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
            });
        hdr = reflected;
        stats.screenSpaceReflections = true;
    }

    if (!waters.empty()) {
        const std::uint32_t sampler = ctx_.linearClampSampler();
        graph.addRasterPass(
            "Water",
            [&](PassBuilder& b) {
                b.writeColor(hdr, LoadOp::Load);
                b.sample(depth);
                if (opaqueColor.valid()) b.sample(opaqueColor);
                if (fogVolume.valid()) b.sample(fogVolume);
            },
            [this, stateFor, waterGrid, depth, sampler, waters, opaqueColor, fogVolume](PassContext& ctx) {
                nvrhi::GraphicsState state = stateFor(water_, ctx);
                bindMesh(state, *waterGrid);
                ctx.cmd->setGraphicsState(state);
                for (const std::uint32_t object : waters) {
                    PushConstants pc;
                    pc.index0 = object;
                    pc.index1 = ctx.bindlessIndex(depth);
                    pc.index2 = sampler;
                    if (fogVolume.valid()) pc.param2 = std::bit_cast<float>(ctx.bindlessIndex(fogVolume));
                    if (opaqueColor.valid()) {  // the scene before SSR, to reflect on screen
                        pc.param0 = std::bit_cast<float>(ctx.bindlessIndex(opaqueColor));
                        pc.flags = 1u;
                    }
                    push(ctx.cmd, pc);
                    ctx.cmd->drawIndexed(nvrhi::DrawArguments().setVertexCount(waterGrid->indexCount));
                }
            });
        stats.drawCalls += static_cast<std::uint32_t>(waters.size());
    }

    if (!transparents.empty()) {
        // Transmissive surfaces (glass) see the scene behind them through a copy of it.
        RGImage background{};
        if (transmissiveDraws) {
            background = graph.createImage("TransmissionBackground", {ctx_.hdrFormat(), extent.width, extent.height});
            graph.addPass(
                "TransmissionCopy",
                [&](PassBuilder& b) {
                    b.copyFrom(hdr);
                    b.copyTo(background);
                },
                [hdr, background](PassContext& ctx) {
                    ctx.cmd->copyTexture(ctx.texture(background), nvrhi::TextureSlice(), ctx.texture(hdr), nvrhi::TextureSlice());
                });
        }
        const std::uint32_t clampSampler = ctx_.linearClampSampler();
        graph.addRasterPass(
            "Transparent",
            [&](PassBuilder& b) {
                b.writeColor(hdr, LoadOp::Load);
                if (background.valid()) b.sample(background);
                b.readDepth(depth);  // hidden behind opaque surfaces, but writes no depth
                if (shadows && !virtualShadows) b.sample(shadowMap);
                if (virtualShadows) {
                    b.sample(vsmPool);
                    b.sample(vsmPageTable);
                }
                if (localShadows) b.sample(localAtlas);
                if (fogVolume.valid()) b.sample(fogVolume);
            },
            [this, stateFor, shadows, shadowMap, virtualShadows, vsmPool, vsmPageTable, localShadows, localAtlas, fogVolume, background,
             clampSampler, debugCascades = settings.debugCascades, draws = transparents](PassContext& ctx) {
                const std::uint32_t shadowIndex = virtualShadows ? ctx.bindlessIndex(vsmPool) : shadows ? ctx.bindlessIndex(shadowMap) : 0u;
                const float pageTableIndex = std::bit_cast<float>(virtualShadows ? ctx.bindlessIndex(vsmPageTable) : 0u);
                const float localAtlasIndex = std::bit_cast<float>(localShadows ? ctx.bindlessIndex(localAtlas) : 0u);
                // No ambient occlusion: it was computed for the opaque surfaces behind.
                const std::uint32_t flags = (shadows ? 1u : 0u) | (shadows && debugCascades ? 2u : 0u);
                nvrhi::GraphicsState state = stateFor(transparent_, ctx);
                nvrhi::IGraphicsPipeline* bound = nullptr;
                const GpuMesh* mesh = nullptr;
                for (const Draw& d : draws) {  // in order: back to front
                    nvrhi::IGraphicsPipeline* pipeline = d.mesh->skinned()                ? transparentSkinned_.Get()
                                                         : (d.variant & kTwoSided) != 0 ? transparentTwoSided_.Get()
                                                                                         : transparent_.Get();
                    if (pipeline != bound || d.mesh != mesh) {
                        bound = pipeline;
                        mesh = d.mesh;
                        state.pipeline = pipeline;
                        bindMesh(state, *mesh);
                        ctx.cmd->setGraphicsState(state);
                    }
                    PushConstants pc;
                    pc.index0 = d.object;
                    pc.index1 = shadowIndex;
                    pc.flags = flags;
                    pc.param0 = localAtlasIndex;
                    pc.param1 = pageTableIndex;
                    pc.param2 = std::bit_cast<float>(fogVolume.valid() ? ctx.bindlessIndex(fogVolume) : 0u);
                    if (background.valid()) {
                        pc.flags |= 16u;
                        pc.param3 = std::bit_cast<float>(ctx.bindlessIndex(background));
                        pc.index2 = clampSampler;
                    }
                    push(ctx.cmd, pc);
                    ctx.cmd->drawIndexed(nvrhi::DrawArguments().setVertexCount(mesh->indexCount));
                }
            });
        stats.drawCalls += static_cast<std::uint32_t>(transparents.size());
        stats.transparentObjects = static_cast<std::uint32_t>(transparents.size());
    }

    if (settings.showGrid) {
        graph.addRasterPass(
            "Grid",
            [&](PassBuilder& b) {
                b.writeColor(hdr, LoadOp::Load);
                b.readDepth(depth);
            },
            [this, stateFor](PassContext& ctx) { fullscreen(ctx.cmd, stateFor(grid_, ctx), PushConstants{}); });
    }
}

} // namespace lumora::render
