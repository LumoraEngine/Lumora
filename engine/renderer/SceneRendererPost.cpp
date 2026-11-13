// Particles and post-processing: sun shafts, TAA, depth of field, motion blur, exposure, bloom, tone mapping and FXAA (see SceneRendererFrame.hpp).
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

void SceneRenderer::addParticles(SceneFrame& f) {
    nvrhi::IDevice* gpu = f.gpu;
    SceneDrawStats& stats = f.stats;
    const auto& reg = f.reg;
    const CameraView& camera = f.camera;
    RenderGraph& graph = f.graph;
    FrameResources& frame = f.frame;
    const DrawState stateFor = f.stateFor;
    const RGImage hdr = f.hdr;
    const RGImage depth = f.depth;

    // ---- Particles: camera-facing quads, alpha-blended ones far to near, then additive ones.
    {
        struct Quad {
            glm::vec3 center;
            float size;
            glm::vec4 color;
            float depth;
        };
        std::vector<Quad> alpha;
        std::vector<Quad> additive;
        const glm::vec3 right(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
        const glm::vec3 up(camera.view[0][1], camera.view[1][1], camera.view[2][1]);
        const glm::vec3 forward(-camera.view[0][2], -camera.view[1][2], -camera.view[2][2]);
        auto linear = [](const glm::vec3& srgb) { return glm::pow(glm::max(srgb, glm::vec3(0.0f)), glm::vec3(2.2f)); };
        for (auto [e, emitter, state] : reg.view<ParticleEmitterComponent, ParticleStateComponent>().each()) {
            if (state.particles.empty()) continue;
            const auto* worldTransform = reg.try_get<WorldTransformComponent>(e);
            const glm::mat4 world = worldTransform != nullptr ? worldTransform->matrix : glm::mat4(1.0f);
            const glm::vec3 c0 = linear(emitter.startColor) * emitter.intensity;
            const glm::vec3 c1 = linear(emitter.endColor) * emitter.intensity;
            auto& out = emitter.blend == ParticleBlend::Additive ? additive : alpha;
            for (const auto& p : state.particles) {
                const float t = std::clamp(p.age / p.life, 0.0f, 1.0f);
                const glm::vec3 position = emitter.worldSpace ? p.position : glm::vec3(world * glm::vec4(p.position, 1.0f));
                const float a = glm::mix(emitter.startAlpha, emitter.endAlpha, t);
                if (a <= 0.0f) continue;
                out.push_back({position, glm::mix(emitter.startSize, emitter.endSize, t) * 0.5f,
                               glm::vec4(glm::mix(c0, c1, t), a), glm::dot(position - camera.position, forward)});
            }
        }
        std::ranges::sort(alpha, std::greater<>{}, &Quad::depth);  // far first
        const std::size_t alphaCount = alpha.size();
        const std::size_t total = alphaCount + additive.size();
        stats.particles = static_cast<std::uint32_t>(total);
        if (total > 0) {
            std::vector<ParticleVertex> vertices;
            vertices.reserve(total * 6);
            auto emit = [&](const Quad& q) {
                const glm::vec3 r = right * q.size;
                const glm::vec3 u = up * q.size;
                const ParticleVertex v00{q.center - r - u, {-1.0f, -1.0f}, q.color};
                const ParticleVertex v10{q.center + r - u, {1.0f, -1.0f}, q.color};
                const ParticleVertex v11{q.center + r + u, {1.0f, 1.0f}, q.color};
                const ParticleVertex v01{q.center - r + u, {-1.0f, 1.0f}, q.color};
                vertices.insert(vertices.end(), {v00, v10, v11, v00, v11, v01});
            };
            for (const Quad& q : alpha) emit(q);
            for (const Quad& q : additive) emit(q);
            nvrhi::IBuffer* vb = frame.particleVertices.write(gpu, std::as_bytes(std::span(vertices)),
                                                              DynamicBuffer::Kind::Vertex, sizeof(ParticleVertex),
                                                              "ParticleVertices");
            const auto alphaVertices = static_cast<std::uint32_t>(alphaCount * 6);
            const auto additiveVertices = static_cast<std::uint32_t>((total - alphaCount) * 6);
            graph.addRasterPass(
                "Particles",
                [&](PassBuilder& b) {
                    b.writeColor(hdr, LoadOp::Load);
                    b.readDepth(depth);
                },
                [this, stateFor, vb, alphaVertices, additiveVertices](PassContext& ctx) {
                    nvrhi::GraphicsState state = stateFor(particlesAlpha_, ctx);
                    state.vertexBuffers = {nvrhi::VertexBufferBinding().setBuffer(vb).setSlot(0)};
                    PushConstants pc;
                    if (alphaVertices > 0) {
                        ctx.cmd->setGraphicsState(state);
                        pc.flags = 0u;
                        push(ctx.cmd, pc);
                        ctx.cmd->draw(nvrhi::DrawArguments().setVertexCount(alphaVertices));
                    }
                    if (additiveVertices > 0) {
                        state.pipeline = particlesAdditive_;
                        ctx.cmd->setGraphicsState(state);
                        pc.flags = 1u;
                        push(ctx.cmd, pc);
                        ctx.cmd->draw(nvrhi::DrawArguments().setVertexCount(additiveVertices).setStartVertexLocation(alphaVertices));
                    }
                });
        }
    }
}

void SceneRenderer::addPostProcessing(SceneFrame& f) {
    nvrhi::IDevice* gpu = f.gpu;
    SceneDrawStats& stats = f.stats;
    const EnvironmentSettings& env = f.env;
    const RenderSettings& settings = f.settings;
    RenderGraph& graph = f.graph;
    const DrawState stateFor = f.stateFor;
    const VkExtent2D extent = f.extent;              // after TAA
    const VkExtent2D renderExtent = f.renderExtent;  // before it
    const double timeSeconds = f.timeSeconds;
    const FrameConstants& fc = f.fc;
    const bool temporal = f.temporal;
    const RGImage hdr = f.hdr;
    const RGImage depth = f.depth;
    const RGImage motion = f.motion;
    const RGImage output = f.output;
    const nvrhi::Format outputFormat = f.outputFormat;

    // ---- Sun shafts (half resolution, added onto the HDR image), then depth of field.
    const std::uint32_t linearSampler = ctx_.linearClampSampler();
    if (fc.sunShafts.z > 0.0f && fc.sunShafts.w > 0.0f) {
        const RGImage shaftsImage = graph.createImage("SunShafts", {ctx_.hdrFormat(), std::max(renderExtent.width / 2, 1u),
                                                                    std::max(renderExtent.height / 2, 1u)});
        graph.addRasterPass(
            "SunShafts",
            [&](PassBuilder& b) {
                b.writeColor(shaftsImage, LoadOp::DontCare);
                b.sample(hdr);
                b.sample(depth);
            },
            [this, stateFor, hdr, depth, linearSampler](PassContext& ctx) {
                PushConstants pc;
                pc.index0 = ctx.bindlessIndex(hdr);
                pc.index1 = ctx.bindlessIndex(depth);
                pc.index2 = linearSampler;
                fullscreen(ctx.cmd, stateFor(shafts_, ctx), pc);
            });
        graph.addRasterPass(
            "SunShaftsAdd",
            [&](PassBuilder& b) {
                b.writeColor(hdr, LoadOp::Load);
                b.sample(shaftsImage);
            },
            [this, stateFor, shaftsImage, linearSampler](PassContext& ctx) {
                PushConstants pc;
                pc.index0 = ctx.bindlessIndex(shaftsImage);
                pc.index2 = linearSampler;
                fullscreen(ctx.cmd, stateFor(shaftsAdd_, ctx), pc);
            });
    }
    // ---- Temporal anti-aliasing: the jittered frame blended into the history, which the rest
    // of the frame then reads.
    RGImage resolved = hdr;
    if (temporal) {
        if (extent.width != taaWidth_ || extent.height != taaHeight_ || !taaHistory_[0]) {
            for (std::size_t i = 0; i < 2; ++i) {
                if (taaHistoryIndex_[i] != UINT32_MAX) ctx_.bindless().releaseTexture(taaHistoryIndex_[i]);
                nvrhi::TextureDesc desc;
                desc.width = extent.width;
                desc.height = extent.height;
                desc.format = ctx_.hdrFormat();
                desc.debugName = i == 0 ? "TaaHistoryA" : "TaaHistoryB";
                desc.isUAV = true;
                desc.initialState = nvrhi::ResourceStates::ShaderResource;
                desc.keepInitialState = true;
                taaHistory_[i] = gpu->createTexture(desc);
                if (!taaHistory_[i]) throw rhi::VulkanError("could not create the TAA history", VK_ERROR_OUT_OF_DEVICE_MEMORY);
                taaHistoryIndex_[i] = ctx_.bindless().registerTexture(taaHistory_[i]);
            }
            taaWidth_ = extent.width;
            taaHeight_ = extent.height;
            taaHistoryValid_ = false;
        }
        const std::size_t write = taaFrame_ % 2;
        const RGImage previousHistory = graph.importImage("TaaHistoryPrevious", {taaHistory_[1 - write], taaHistoryIndex_[1 - write]});
        resolved = graph.importImage("TaaHistory", {taaHistory_[write], taaHistoryIndex_[write]});
        const bool historyValid = taaHistoryValid_;
        nvrhi::IDescriptorTable* bindlessTable = ctx_.bindless().table();
        graph.addComputePass(
            "TAA",
            [&](PassBuilder& b) {
                b.writeStorage(resolved);
                b.sample(hdr);
                b.sample(depth);
                b.sample(motion);
                b.sample(previousHistory);
            },
            [this, hdr, depth, motion, previousHistory, historyValid, bindlessTable, extent,
             upscale = renderExtent.width != extent.width || renderExtent.height != extent.height](PassContext& ctx) {
                nvrhi::ComputeState state;
                state.pipeline = taa_;
                state.bindings = {ctx.frameBindings(), bindlessTable, ctx.storageBindings()};
                ctx.cmd->setComputeState(state);
                PushConstants pc;
                pc.index0 = ctx.bindlessIndex(hdr);
                pc.index1 = ctx.bindlessIndex(depth);
                pc.index2 = ctx.bindlessIndex(motion);
                pc.param0 = std::bit_cast<float>(ctx.bindlessIndex(previousHistory));
                pc.param1 = 0.1f;  // weight of the new frame: about the last 10 frames are averaged
                pc.flags = (historyValid ? 1u : 0u) | (upscale ? 2u : 0u);
                push(ctx.cmd, pc);
                ctx.cmd->dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
            });
        taaHistoryValid_ = true;
        stats.temporalAntiAliasing = true;
    }
    RGImage sceneColor = resolved;
    if (fc.depthOfField.z > 0.5f) {
        sceneColor = graph.createImage("DepthOfField", {ctx_.hdrFormat(), extent.width, extent.height});
        graph.addRasterPass(
            "DepthOfField",
            [&](PassBuilder& b) {
                b.writeColor(sceneColor, LoadOp::DontCare);
                b.sample(resolved);
                b.sample(depth);
            },
            [this, stateFor, resolved, depth, linearSampler](PassContext& ctx) {
                PushConstants pc;
                pc.index0 = ctx.bindlessIndex(resolved);
                pc.index1 = ctx.bindlessIndex(depth);
                pc.index2 = linearSampler;
                fullscreen(ctx.cmd, stateFor(depthOfField_, ctx), pc);
            });
    }

    // ---- Motion blur along each pixel's motion (motion_blur.hlsl).
    if (env.motionBlur > 0.0f && motionBlur_ && extent.width >= 8 && extent.height >= 8) {
        const RGImage blurred = graph.createImage("MotionBlur", {ctx_.hdrFormat(), extent.width, extent.height, true});
        nvrhi::IDescriptorTable* bindlessTable = ctx_.bindless().table();
        graph.addComputePass(
            "MotionBlur",
            [&](PassBuilder& b) {
                b.writeStorage(blurred);
                b.sample(sceneColor);
                b.sample(depth);
                b.sample(motion);
            },
            [this, sceneColor, depth, motion, bindlessTable, extent, shutter = std::min(env.motionBlur, 2.0f),
             longest = std::clamp(env.motionBlurMax, 1.0f, 256.0f), frame = taaFrame_](PassContext& ctx) {
                ctx.cmd->setComputeState(nvrhi::ComputeState().setPipeline(motionBlur_).addBindingSet(ctx.frameBindings())
                                             .addBindingSet(bindlessTable).addBindingSet(ctx.storageBindings()));
                PushConstants pc;
                pc.index0 = ctx.bindlessIndex(sceneColor);
                pc.index1 = ctx.bindlessIndex(depth);
                pc.index2 = ctx.bindlessIndex(motion);
                pc.param0 = shutter;
                pc.param1 = longest;
                pc.param2 = static_cast<float>(frame % 1024u);
                push(ctx.cmd, pc);
                ctx.cmd->dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
            });
        sceneColor = blurred;
        stats.motionBlur = true;
    }

    // ---- Automatic exposure: histogram of the image, then adaptation (see exposure_*.hlsl).
    RGImage exposureImage{};
    if (env.autoExposure && exposureHistogram_ && exposureAdapt_ && !settings.linearOutput) {
        if (!exposure_) {
            nvrhi::TextureDesc desc;
            desc.width = desc.height = 1;
            desc.format = nvrhi::Format::R32_FLOAT;
            desc.isUAV = true;
            desc.debugName = "Exposure";
            desc.initialState = nvrhi::ResourceStates::ShaderResource;
            desc.keepInitialState = true;
            exposure_ = ctx_.gpu()->createTexture(desc);
            if (!exposure_) throw rhi::VulkanError("could not create the exposure texture", VK_ERROR_OUT_OF_DEVICE_MEMORY);
            exposureIndex_ = ctx_.bindless().registerTexture(exposure_);
        }
        exposureImage = graph.importImage("Exposure", {exposure_, exposureIndex_});
        const RGImage histogram = graph.createImage("ExposureHistogram", {nvrhi::Format::R32_UINT, 64, 1, true});
        // A live view adapts over time; a single capture (or a jump in time) takes the target.
        const double dt = exposureTime_ < 0.0 ? 1.0e9 : timeSeconds - exposureTime_;
        exposureTime_ = timeSeconds;
        const bool live = settings.temporalAntiAliasing && dt >= 0.0 && dt < 1.0;
        const float adaptation = live ? 1.0f - std::exp(-static_cast<float>(dt) * std::max(env.adaptationSpeed, 0.01f)) : 1.0f;
        constexpr float kLowEv = -10.0f, kRangeEv = 24.0f;
        nvrhi::IDescriptorTable* bindlessTable = ctx_.bindless().table();
        graph.addComputePass(
            "ExposureHistogram",
            [&](PassBuilder& b) {
                b.writeStorage(histogram);
                b.sample(sceneColor);
            },
            [this, histogram, sceneColor, bindlessTable, extent, kLowEv, kRangeEv](PassContext& ctx) {
                ctx.cmd->clearTextureUInt(ctx.texture(histogram), nvrhi::AllSubresources, 0u);
                ctx.cmd->setComputeState(nvrhi::ComputeState().setPipeline(exposureHistogram_).addBindingSet(ctx.frameBindings())
                                             .addBindingSet(bindlessTable).addBindingSet(ctx.storageBindings()));
                PushConstants pc;
                pc.index0 = ctx.bindlessIndex(sceneColor);
                pc.param0 = kLowEv;
                pc.param1 = kRangeEv;
                push(ctx.cmd, pc);
                ctx.cmd->dispatch((extent.width / 2 + 7) / 8 + 1, (extent.height / 2 + 7) / 8 + 1, 1);
            });
        const float minimum = std::max(env.autoExposureMin, 1e-4f);
        const float maximum = std::max(env.autoExposureMax, minimum);
        graph.addComputePass(
            "ExposureAdapt",
            [&](PassBuilder& b) {
                b.writeStorage(exposureImage);
                b.writeStorage(histogram);  // u1, read
            },
            [this, bindlessTable, adaptation, minimum, maximum, compensation = std::max(env.exposure, 0.0f), kLowEv, kRangeEv](PassContext& ctx) {
                ctx.cmd->setComputeState(nvrhi::ComputeState().setPipeline(exposureAdapt_).addBindingSet(ctx.frameBindings())
                                             .addBindingSet(bindlessTable).addBindingSet(ctx.storageBindings()));
                PushConstants pc;
                pc.index0 = std::bit_cast<std::uint32_t>(minimum);
                pc.index1 = std::bit_cast<std::uint32_t>(maximum);
                pc.param0 = kLowEv;
                pc.param1 = kRangeEv;
                pc.param2 = adaptation;
                pc.param3 = compensation;
                push(ctx.cmd, pc);
                ctx.cmd->dispatch(1, 1, 1);
            });
        stats.autoExposure = true;
    } else {
        exposureTime_ = -1.0;  // start from the target when it comes back
    }

    // ---- Bloom: downsample chain, then additive tent upsample back to the half-res level.
    const std::uint32_t sampler = ctx_.linearClampSampler();
    std::vector<RGImage> bloomChain;
    if (settings.bloom && env.bloomIntensity > 0.0f) {
        std::uint32_t w = extent.width, h = extent.height;
        while (bloomChain.size() < 6 && w >= 16 && h >= 16) {
            w /= 2;
            h /= 2;
            bloomChain.push_back(graph.createImage("Bloom" + std::to_string(bloomChain.size()), {ctx_.hdrFormat(), w, h}));
        }
    }
    nvrhi::IGraphicsPipeline* bloomDown = bloomDown_;
    nvrhi::IGraphicsPipeline* bloomUp = bloomUp_;
    for (std::size_t i = 0; i < bloomChain.size(); ++i) {
        const RGImage source = i == 0 ? sceneColor : bloomChain[i - 1];
        const RGImage target = bloomChain[i];
        graph.addRasterPass(
            "BloomDown" + std::to_string(i),
            [&](PassBuilder& b) {
                b.sample(source);
                b.writeColor(target, LoadOp::DontCare);
            },
            [=](PassContext& ctx) {
                PushConstants pc;
                pc.index0 = ctx.bindlessIndex(source);
                pc.index1 = sampler;
                pc.flags = i == 0 ? 1u : 0u;
                fullscreen(ctx.cmd, stateFor(bloomDown, ctx), pc);
            });
    }
    for (std::size_t i = bloomChain.size(); i-- > 1;) {
        const RGImage source = bloomChain[i];
        const RGImage target = bloomChain[i - 1];
        const float radius = 1.0f / static_cast<float>(std::max(extent.width >> (i + 1), 1u));
        graph.addRasterPass(
            "BloomUp" + std::to_string(i),
            [&](PassBuilder& b) {
                b.sample(source);
                b.writeColor(target, LoadOp::Load);
            },
            [=](PassContext& ctx) {
                PushConstants pc;
                pc.index0 = ctx.bindlessIndex(source);
                pc.index1 = sampler;
                pc.param0 = radius;
                fullscreen(ctx.cmd, stateFor(bloomUp, ctx), pc);
            });
    }
    stats.drawCalls += static_cast<std::uint32_t>(bloomChain.empty() ? 0 : bloomChain.size() * 2 - 1);

    // With FXAA the tonemapper writes a gamma-encoded LDR image (luma in alpha) that FXAA then
    // resolves into the output; without, it writes the output directly.
    const bool fxaa = settings.antiAliasing && !temporal && !settings.linearOutput;
    // Half floats: the gamma-encoded image is quantised only once, when FXAA writes the output.
    constexpr nvrhi::Format kLdrFormat = nvrhi::Format::RGBA16_FLOAT;
    const RGImage ldr = fxaa ? graph.createImage("LDR", {kLdrFormat, extent.width, extent.height}) : output;
    const nvrhi::Format tonemapFormat = fxaa ? kLdrFormat : outputFormat;
    PipelineDesc fullscreenDesc;
    fullscreenDesc.vertexShader = "tonemap.vert";
    fullscreenDesc.pixelShader = "tonemap.frag";
    fullscreenDesc.name = "Tonemap";
    nvrhi::IGraphicsPipeline* tonemap = outputPipeline(tonemap_, fullscreenDesc, tonemapFormat);
    const bool shaderSrgb = !isSrgbFormat(tonemapFormat);
    const bool haveBloom = !bloomChain.empty();
    const RGImage bloomResult = haveBloom ? bloomChain.front() : RGImage{};
    // Bloom is a lerp towards the blurred image; normalise for the summed chain levels.
    const float bloomStrength = haveBloom ? std::clamp(env.bloomIntensity, 0.0f, 1.0f) : 0.0f;
    const float bloomNormalisation = haveBloom ? 1.0f / static_cast<float>(bloomChain.size()) : 1.0f;
    const float exposure = env.exposure;
    const float vignette = std::clamp(env.vignette, 0.0f, 1.0f);
    const bool linear = settings.linearOutput;
    graph.addRasterPass(
        "Tonemap",
        [&](PassBuilder& b) {
            b.sample(sceneColor);
            if (haveBloom) b.sample(bloomResult);
            if (exposureImage.valid()) b.sample(exposureImage);
            b.writeColor(ldr, LoadOp::DontCare);
        },
        [=](PassContext& ctx) {
            PushConstants pc;
            pc.index0 = ctx.bindlessIndex(sceneColor);
            pc.index1 = sampler;
            pc.index2 = haveBloom ? ctx.bindlessIndex(bloomResult) : 0u;
            pc.flags = (shaderSrgb ? 1u : 0u) | (haveBloom ? 2u : 0u) | (fxaa ? 4u : 0u) | (linear ? 8u : 0u) |
                       (exposureImage.valid() ? 16u : 0u);
            pc.param0 = exposureImage.valid() ? std::bit_cast<float>(ctx.bindlessIndex(exposureImage)) : exposure;
            pc.param1 = bloomStrength;
            pc.param2 = vignette;
            pc.param3 = bloomNormalisation;
            fullscreen(ctx.cmd, stateFor(tonemap, ctx), pc);
        });
    if (fxaa) {
        PipelineDesc fxaaDesc;
        fxaaDesc.vertexShader = "fxaa.vert";
        fxaaDesc.pixelShader = "fxaa.frag";
        fxaaDesc.name = "FXAA";
        nvrhi::IGraphicsPipeline* resolve = outputPipeline(fxaa_, fxaaDesc, outputFormat);
        const bool outputSrgb = isSrgbFormat(outputFormat);
        graph.addRasterPass(
            "FXAA",
            [&](PassBuilder& b) {
                b.sample(ldr);
                b.writeColor(output, LoadOp::DontCare);
            },
            [=](PassContext& ctx) {
                PushConstants pc;
                pc.index0 = ctx.bindlessIndex(ldr);
                pc.index1 = sampler;
                pc.flags = outputSrgb ? 1u : 0u;
                fullscreen(ctx.cmd, stateFor(resolve, ctx), pc);
            });
        stats.drawCalls += 1;
    }
}

} // namespace lumora::render
