// Frame constants: camera and TAA jitter, sky, probes, lights and the shadow setup (see SceneRendererFrame.hpp).
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

namespace {

/// Halton(2, 3) point i (1-based) in [-0.5, 0.5]^2: the sub-pixel jitter sequence of TAA.
glm::vec2 halton23(std::uint64_t i) {
    auto radical = [](std::uint64_t n, std::uint64_t base) {
        float f = 1.0f, r = 0.0f;
        while (n > 0) {
            f /= static_cast<float>(base);
            r += f * static_cast<float>(n % base);
            n /= base;
        }
        return r;
    };
    return {radical(i, 2) - 0.5f, radical(i, 3) - 0.5f};
}

} // namespace

void SceneRenderer::prepareFrame(SceneFrame& f) {
    using LocalTile = LocalShadowTile;
    nvrhi::IDevice* gpu = f.gpu;
    SceneDrawStats& stats = f.stats;
    const auto& reg = f.reg;
    const EnvironmentSettings& env = f.env;
    const CameraView& camera = f.camera;
    const RenderSettings& settings = f.settings;
    // TAA upscaling: everything up to TAA renders smaller (see RenderSettings::renderScale).
    const VkExtent2D outputExtent = f.extent;
    {
        const bool upscale = settings.temporalAntiAliasing && taa_ && outputExtent.width >= 16 && outputExtent.height >= 16;
        const float scale = upscale ? std::clamp(settings.renderScale, 0.5f, 1.0f) : 1.0f;
        f.renderExtent = outputExtent;
        if (scale < 0.999f) {
            f.renderExtent.width = std::max(8u, static_cast<std::uint32_t>(std::lround(static_cast<float>(outputExtent.width) * scale)));
            f.renderExtent.height = std::max(8u, static_cast<std::uint32_t>(std::lround(static_cast<float>(outputExtent.height) * scale)));
        }
    }
    const VkExtent2D extent = f.renderExtent;
    const double timeSeconds = f.timeSeconds;
    FrameResources& frame = f.frame;

    // ---- Frame constants -------------------------------------------------------------------
    FrameConstants fc;
    fc.volumeCount = glm::uvec4(0u, kNoTexture, 0u, 0u);
    fc.localShadowParams.w = std::bit_cast<float>(kNoTexture);
    fc.view = camera.view;
    fc.projection = camera.projection;
    fc.viewProjection = camera.projection * camera.view;
    fc.inverseViewProjection = glm::inverse(fc.viewProjection);
    // Temporal anti-aliasing: a sub-pixel jitter on the projection (Halton 2,3 over 8 frames)
    // and last frame's matrices for the motion vectors. A cut (teleport, sharp turn, resize)
    // starts the history over.
    const bool temporal = settings.temporalAntiAliasing && taa_ && extent.width >= 8 && extent.height >= 8;
    const glm::mat4 unjittered = fc.viewProjection;
    const glm::vec3 cameraForward = -glm::normalize(glm::vec3(glm::inverse(camera.view)[2]));
    glm::vec2 jitter(0.0f);
    if (temporal) {
        // More jitter positions when upscaling: an output pixel needs about 8 samples of its own.
        const float density = static_cast<float>(outputExtent.width) / static_cast<float>(extent.width);
        const auto phases = static_cast<std::uint64_t>(std::clamp(std::lround(8.0f * density * density), 8l, 32l));
        jitter = halton23(taaFrame_ % phases + 1) * glm::vec2(2.0f / static_cast<float>(extent.width), 2.0f / static_cast<float>(extent.height));
        glm::mat4 projection = camera.projection;
        projection[2][0] -= jitter.x;  // clip.x / w gains jitter.x (w = -view z)
        projection[2][1] -= jitter.y;
        fc.projection = projection;
        fc.viewProjection = projection * camera.view;
        fc.inverseViewProjection = glm::inverse(fc.viewProjection);
        const bool cut = glm::distance(camera.position, previousCameraPosition_) > 5.0f ||
                         glm::dot(cameraForward, previousCameraForward_) < 0.7f || outputExtent.width != taaWidth_ ||
                         outputExtent.height != taaHeight_;
        if (cut) taaHistoryValid_ = false;
    } else {
        taaHistoryValid_ = false;
    }
    fc.previousViewProjection = taaHistoryValid_ ? previousViewProjection_ : unjittered;
    fc.jitter = glm::vec4(jitter, taaHistoryValid_ ? previousJitter_ : jitter);
    fc.cameraPositionTime = glm::vec4(camera.position, static_cast<float>(timeSeconds));
    fc.viewportSize = glm::vec4(static_cast<float>(extent.width), static_cast<float>(extent.height),
                                1.0f / static_cast<float>(extent.width), 1.0f / static_cast<float>(extent.height));
    fc.outputSize = glm::vec4(static_cast<float>(outputExtent.width), static_cast<float>(outputExtent.height),
                              1.0f / static_cast<float>(outputExtent.width), 1.0f / static_cast<float>(outputExtent.height));
    fc.skyZenith = glm::vec4(env.skyZenithColor, env.ambientIntensity);
    fc.skyHorizon = glm::vec4(env.skyHorizonColor, env.exposure);
    fc.groundColor = glm::vec4(env.groundColor, 0.0f);
    fc.gridParams = glm::vec4(settings.showGrid ? 1.0f : 0.0f, settings.gridCellSize, settings.gridFadeDistance, 0.0f);
    fc.fogParams = glm::vec4(std::max(env.fogDensity, 0.0f), std::max(env.fogHeightFalloff, 1e-4f), 0.97f, 0.0f);
    // Volumetric fog: froxels of 8 x 8 pixels, 64 slices (see fog.hlsl).
    const bool volumetricFog = env.volumetricFog && env.fogDensity > 0.0f && fogInject_ && fogIntegrate_ &&
                               extent.width >= 8 && extent.height >= 8;
    const glm::uvec3 froxels((extent.width + 7) / 8, (extent.height + 7) / 8, 64u);
    if (volumetricFog) {
        fc.fogVolume = glm::uvec4(1u, froxels);
        // Slices jittered along the ray from frame to frame when TAA blends the frames.
        const float sliceJitter = settings.temporalAntiAliasing ? std::fmod(static_cast<float>(taaFrame_) * 0.618034f, 1.0f) : 0.5f;
        fc.fogVolumeParams = glm::vec4(std::clamp(env.volumetricFogDistance, 5.0f, 2000.0f),
                                       std::clamp(env.fogAnisotropy, 0.0f, 0.95f), sliceJitter, 0.0f);
        fc.fogAlbedo = glm::vec4(glm::max(env.fogColor, glm::vec3(0.0f)), 1.0f);
        stats.volumetricFog = true;
    }
    if (env.cloudCoverage > 0.0f) {
        fc.clouds = glm::vec4(std::clamp(env.cloudCoverage, 0.0f, 1.0f), env.cloudHeight, std::max(env.cloudThickness, 1.0f),
                              std::max(env.cloudDensity, 0.0f));
        fc.cloudWind = glm::vec4(static_cast<float>(std::fmod(timeSeconds * env.cloudWindSpeed, 1.0e6)), 0.0f, 0.0f, 0.0f);
    }
    fc.grading = glm::vec4(std::max(env.contrast, 0.0f), std::max(env.saturation, 0.0f), std::clamp(env.temperature, -1.0f, 1.0f),
                           std::clamp(env.tint, -1.0f, 1.0f));
    fc.depthOfField = glm::vec4(std::max(env.focusDistance, 0.01f), std::max(env.focusRange, 0.01f),
                                std::clamp(env.dofBlur, 0.0f, 32.0f), camera.nearPlane);
    // Image-based lighting from the scene's HDR sky (procedural sky until it has loaded).
    if (!env.skyTexture.empty() && settings.assets != nullptr) {
        if (const auto* sky = ctx_.environments().get(*settings.assets, env.skyTexture, settings.blockOnAssets)) {
            fc.environmentParams = glm::vec4(1.0f, std::max(env.skyIntensity, 0.0f), glm::radians(env.skyRotationDeg),
                                             static_cast<float>(sky->specularLevels - 1));
            fc.environmentTextures = glm::uvec4(sky->skyIndex, sky->specularIndex, ctx_.materialSampler(), 0u);
            for (std::size_t i = 0; i < sky->irradiance.size(); ++i) fc.irradianceSh[i] = glm::vec4(sky->irradiance[i], 0.0f);
            stats.environmentLit = true;
        }
    }

    // Baked reflection probes: the nearest kMaxReflectionProbes to the camera, then the smallest
    // boxes first (a room's probe wins over the courtyard's around it).
    if (settings.reflectionProbes && settings.assets != nullptr) {
        struct Probe {
            GpuReflectionProbe gpu;
            float distance;
            float volume;
        };
        std::vector<Probe> probes;
        for (auto [e, probe, world] : reg.view<ReflectionProbeComponent, WorldTransformComponent>().each()) {
            if (probe.baked.empty()) continue;
            const auto* baked = ctx_.environments().get(*settings.assets, probe.baked, settings.blockOnAssets);
            if (baked == nullptr) continue;  // loading
            const glm::vec3 center(world.matrix[3]);
            const glm::vec3 half = glm::max(probe.size, glm::vec3(0.01f)) * 0.5f;
            Probe p;
            p.gpu.boxMin = glm::vec4(center - half, std::max(probe.blendDistance, 0.0f));
            p.gpu.boxMax = glm::vec4(center + half, std::max(probe.intensity, 0.0f));
            p.gpu.position = glm::vec4(center, probe.boxProjection ? 1.0f : 0.0f);
            p.gpu.textures = glm::uvec4(baked->specularIndex, baked->specularLevels - 1, 0u, 0u);
            const glm::vec3 outside = glm::max(glm::abs(camera.position - center) - half, glm::vec3(0.0f));
            p.distance = glm::length(outside);
            p.volume = half.x * half.y * half.z;
            probes.push_back(p);
        }
        std::ranges::sort(probes, std::less<>{}, &Probe::distance);
        if (probes.size() > kMaxReflectionProbes) probes.resize(kMaxReflectionProbes);
        std::ranges::sort(probes, std::less<>{}, &Probe::volume);
        for (std::size_t i = 0; i < probes.size(); ++i) fc.probes[i] = probes[i].gpu;
        fc.probeCount.x = static_cast<std::uint32_t>(probes.size());
        stats.reflectionProbes = fc.probeCount.x;
    }

    // Baked light probe volumes, like the reflection probes: nearest first, then smallest first.
    if (settings.lightProbeVolumes && settings.assets != nullptr) {
        struct Volume {
            GpuProbeVolume gpu;
            float distance;
            float volume;
        };
        std::vector<Volume> volumes;
        for (auto [e, volume, world] : reg.view<LightProbeVolumeComponent, WorldTransformComponent>().each()) {
            if (volume.baked.empty()) continue;
            const ProbeVolumeGpu* baked = probeVolume(*settings.assets, volume.baked);
            if (baked == nullptr) continue;
            const glm::vec3 center(world.matrix[3]);
            const glm::vec3 half = glm::max(volume.size, glm::vec3(0.01f)) * 0.5f;
            const glm::vec3 cell = half * 2.0f / glm::vec3(baked->counts);
            Volume v;
            v.gpu.boxMin = glm::vec4(center - half, std::max(volume.blendDistance, 0.0f));
            v.gpu.boxMax = glm::vec4(center + half, std::max(volume.intensity, 0.0f));
            v.gpu.grid = glm::uvec4(baked->index, baked->counts);
            v.gpu.params = glm::vec4(0.3f * std::min({cell.x, cell.y, cell.z}), 0.0f, 0.0f, 0.0f);
            v.distance = glm::length(glm::max(glm::abs(camera.position - center) - half, glm::vec3(0.0f)));
            v.volume = half.x * half.y * half.z;
            volumes.push_back(v);
        }
        std::ranges::sort(volumes, std::less<>{}, &Volume::distance);
        if (volumes.size() > kMaxProbeVolumes) volumes.resize(kMaxProbeVolumes);
        std::ranges::sort(volumes, std::less<>{}, &Volume::volume);
        for (std::size_t i = 0; i < volumes.size(); ++i) fc.volumes[i] = volumes[i].gpu;
        fc.volumeCount.x = static_cast<std::uint32_t>(volumes.size());
        stats.lightProbeVolumes = fc.volumeCount.x;
    }
    // Volumes not drawn for a while (re-baked: a new file each time) are dropped.
    std::erase_if(probeVolumes_, [&](auto& item) {
        if (item.second.lastFrame + 600 >= frameCounter_) return false;
        ctx_.bindless().releaseTexture(item.second.index);
        return true;
    });

    // Directional lights go into the constants (few, they light everything); point lights into
    // a storage buffer, binned into screen tiles so each pixel only visits the ones near it.
    std::uint32_t lightCount = 0;
    std::vector<GpuLight> pointLights;
    struct ShadowCandidate {
        std::uint32_t light;  // index into pointLights
        LightType type;
        float outerAngle;     // spot: half-angle (radians)
        float distance;       // to the camera
    };
    std::vector<ShadowCandidate> shadowCandidates;
    bool haveSun = false;
    std::uint32_t sunIndex = UINT32_MAX;
    glm::vec3 sunTravel(0.0f, -1.0f, 0.0f);
    for (auto [e, light, world] : reg.view<LightComponent, WorldTransformComponent>().each()) {
        GpuLight gl;
        const glm::vec3 position(world.matrix[3]);
        const glm::vec3 direction = glm::normalize(glm::vec3(world.matrix * glm::vec4(0, 0, -1, 0)));
        gl.positionRange = glm::vec4(position, light.range);
        gl.colorIntensity = glm::vec4(light.color, light.intensity);
        gl.directionType = glm::vec4(direction, static_cast<float>(light.type));
        if (light.type == LightType::Spot) {
            const float outer = glm::radians(std::clamp(light.spotAngleDeg, 1.0f, 89.0f));
            const float inner = outer * (1.0f - std::clamp(light.spotSoftness, 0.0f, 1.0f));
            gl.spot = glm::vec4(std::cos(outer), std::cos(inner) + 1e-4f, -1.0f, 0.0f);
        }
        if (light.type != LightType::Directional) {
            if (pointLights.size() < kMaxPointLights && light.intensity > 0.0f) {
                if (light.castShadows && settings.shadows) {
                    shadowCandidates.push_back({static_cast<std::uint32_t>(pointLights.size()), light.type,
                                                glm::radians(std::clamp(light.spotAngleDeg, 1.0f, 89.0f)),
                                                glm::length(position - camera.position)});
                }
                pointLights.push_back(gl);
            }
            continue;
        }
        if (lightCount >= kMaxLights) continue;
        fc.lights[lightCount++] = gl;
        if (!haveSun) {
            haveSun = true;
            sunIndex = lightCount - 1;
            sunTravel = direction;
            fc.sunDirection = glm::vec4(-direction, 1.0f);
            fc.sunColor = glm::vec4(light.color * light.intensity, 1.0f);
        }
    }
    const bool shadows = settings.shadows && haveSun && settings.shadowResolution > 0 && settings.shadowDistance > 0.0f;
    const bool virtualShadows = shadows && settings.virtualShadows && env.virtualShadowMaps;
    // Sun shafts: where the sun is on screen, fading out as it leaves the view.
    if (haveSun && env.sunShafts > 0.0f) {
        const glm::vec4 clip = fc.viewProjection * glm::vec4(camera.position - sunTravel * 1000.0f, 1.0f);
        if (clip.w > 0.0f) {
            const glm::vec2 ndc = glm::vec2(clip) / clip.w;
            const float edge = std::max(std::abs(ndc.x), std::abs(ndc.y));
            const float visibility = std::clamp(1.6f - edge, 0.0f, 1.0f);
            fc.sunShafts = glm::vec4(ndc * 0.5f + 0.5f, env.sunShafts, visibility);
        }
    }
    // Spot and point light shadows: tiles of the local atlas to the lights nearest the camera. A
    // light keeps the tiles it had (their contents may still be valid); new ones take free runs.
    std::vector<LocalTile> localTiles;
    std::ranges::sort(shadowCandidates, std::less<>{}, &ShadowCandidate::distance);
    {
        auto lightKey = [&](const ShadowCandidate& c) {
            const GpuLight& gl = pointLights[c.light];
            std::uint64_t h = static_cast<std::uint64_t>(c.type) + 1;
            for (const float v : {gl.positionRange.x, gl.positionRange.y, gl.positionRange.z, gl.positionRange.w, gl.directionType.x,
                                  gl.directionType.y, gl.directionType.z, c.outerAngle}) {
                h = h * 1099511628211ull ^ std::bit_cast<std::uint32_t>(v);
            }
            return h;
        };
        std::vector<LocalAllocation> kept;
        std::vector<std::pair<std::size_t, std::uint32_t>> chosen;  // candidate, first tile
        std::uint64_t used = 0;  // tiles taken (bit per tile)
        std::uint32_t total = 0;
        std::vector<std::size_t> waiting;
        for (std::size_t i = 0; i < shadowCandidates.size(); ++i) {
            const std::uint32_t needed = shadowCandidates[i].type == LightType::Spot ? 1u : 6u;
            if (total + needed > kLocalShadowTiles) continue;
            total += needed;
            const std::uint64_t key = lightKey(shadowCandidates[i]);
            const auto it = std::ranges::find_if(localAllocations_, [&](const LocalAllocation& a) { return a.light == key && a.count == needed; });
            if (it != localAllocations_.end()) {
                kept.push_back(*it);
                chosen.emplace_back(i, it->first);
                used |= (needed == 64 ? ~0ull : ((1ull << needed) - 1)) << it->first;
            } else {
                waiting.push_back(i);
            }
        }
        for (const std::size_t i : waiting) {
            const std::uint32_t needed = shadowCandidates[i].type == LightType::Spot ? 1u : 6u;
            const std::uint64_t run = (1ull << needed) - 1;
            for (std::uint32_t first = 0; first + needed <= kLocalShadowTiles; ++first) {
                if ((used & (run << first)) != 0) continue;
                used |= run << first;
                kept.push_back({lightKey(shadowCandidates[i]), first, needed});
                chosen.emplace_back(i, first);
                break;
            }
        }
        localAllocations_ = std::move(kept);
        for (const auto& [i, first] : chosen) {
            const ShadowCandidate& c = shadowCandidates[i];
            GpuLight& gl = pointLights[c.light];
            const glm::vec3 position(gl.positionRange);
            const float range = std::max(gl.positionRange.w, 0.1f);
            const float nearPlane = std::clamp(range * 0.005f, 0.02f, 0.5f);
            gl.spot.z = static_cast<float>(first);
            std::uint32_t index = first;
            auto add = [&](const glm::vec3& forward, const glm::vec3& up, float fov) {
                const glm::mat4 view = glm::lookAt(position, position + forward, up);
                const glm::mat4 projection = glm::perspective(fov, 1.0f, nearPlane, range);
                fc.localShadowMatrices[index] = projection * view;
                localTiles.push_back({projection * view, position, range, index});
                ++index;
            };
            if (c.type == LightType::Spot) {
                const glm::vec3 dir = glm::normalize(glm::vec3(gl.directionType));
                add(dir, std::abs(dir.y) > 0.99f ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0),
                    std::min(2.0f * c.outerAngle + glm::radians(4.0f), glm::radians(170.0f)));
            } else {
                const float quarter = glm::half_pi<float>();
                add({1, 0, 0}, {0, 1, 0}, quarter);
                add({-1, 0, 0}, {0, 1, 0}, quarter);
                add({0, 1, 0}, {0, 0, 1}, quarter);
                add({0, -1, 0}, {0, 0, 1}, quarter);
                add({0, 0, 1}, {0, 1, 0}, quarter);
                add({0, 0, -1}, {0, 1, 0}, quarter);
            }
        }
    }
    const bool localShadows = !localTiles.empty();
    const std::uint32_t localAtlasSize = std::min(4096u, ctx_.caps().maxImageDimension2D) / kLocalShadowGrid * kLocalShadowGrid;
    if (localShadows) {
        fc.localShadowParams = glm::vec4(1.0f, 1.0f / static_cast<float>(localAtlasSize),
                                         static_cast<float>(kLocalShadowGrid), std::bit_cast<float>(kNoTexture));
    }
    fc.shadowParams.y = static_cast<float>(ctx_.shadowCompareSampler());  // local shadows use it without a sun
    const bool anyShadows = shadows || localShadows;
    fc.counts = glm::uvec4(lightCount, shadows ? sunIndex : UINT32_MAX, ctx_.materialSampler(), 0);
    stats.lights = lightCount + static_cast<std::uint32_t>(pointLights.size());
    {
        const LightGrid grid = buildLightGrid(pointLights, camera.view, camera.projection, extent.width, extent.height);
        fc.lightGrid = glm::uvec4(grid.tilesX, grid.tilesY, grid.tileSize, static_cast<std::uint32_t>(pointLights.size()));
        if (pointLights.empty()) pointLights.push_back(GpuLight{});  // storage buffers may not be empty
        frame.pointLights.write(gpu, std::as_bytes(std::span(pointLights)), DynamicBuffer::Kind::Structured,
                                sizeof(GpuLight), "FramePointLights");
        std::vector<std::uint32_t> gridData = grid.data;
        if (gridData.empty()) gridData.push_back(0);
        frame.lightGrid.write(gpu, std::as_bytes(std::span(gridData)), DynamicBuffer::Kind::Structured,
                              sizeof(std::uint32_t), "FrameLightGrid");
    }
    // Cascades are the 2x2 tiles of one atlas; each covers the view frustum from the near plane
    // to its split (nested), so every visible point within the shadow distance has a cascade.
    const std::uint32_t tile =
        std::min(settings.shadowResolution, ctx_.caps().maxImageDimension2D / 2);
    const std::uint32_t atlasSize = tile * 2;
    std::array<glm::mat4, kShadowCascades> cascadeMatrices{};
    if (virtualShadows) {
        constexpr float pool = static_cast<float>(VirtualShadowMap::poolSize());
        fc.shadowParams = glm::vec4(2.0f, static_cast<float>(ctx_.shadowCompareSampler()), 1.0f / pool, pool);
        fc.shadowFade = glm::vec4(settings.shadowDistance * 0.85f, settings.shadowDistance, 0.0f, 0.0f);
        stats.shadowsRendered = true;
        stats.virtualShadows = true;
    } else if (shadows) {
        const auto splits = cascadeSplits(camera.nearPlane, settings.shadowDistance, settings.cascadeSplitLambda);
        for (std::uint32_t c = 0; c < kShadowCascades; ++c) {
            const ShadowFit fit = fitSunShadow(camera, sunTravel, splits[c], tile);
            cascadeMatrices[c] = fit.viewProjection;
            fc.cascadeViewProjection[c] = fit.viewProjection;
            fc.cascadeTexelWorld[static_cast<int>(c)] = 2.0f * fit.radius / static_cast<float>(tile);
        }
        fc.shadowParams = glm::vec4(1.0f, static_cast<float>(ctx_.shadowCompareSampler()),
                                    1.0f / static_cast<float>(atlasSize), static_cast<float>(atlasSize));
        fc.shadowFade = glm::vec4(settings.shadowDistance * 0.85f, settings.shadowDistance, 0.0f, 0.0f);
        stats.shadowsRendered = true;
    }

    f.fc = fc;
    f.temporal = temporal;
    f.unjittered = unjittered;
    f.cameraForward = cameraForward;
    f.jitter = jitter;
    f.volumetricFog = volumetricFog;
    f.froxels = froxels;
    f.haveSun = haveSun;
    f.sunTravel = sunTravel;
    f.shadows = shadows;
    f.virtualShadows = virtualShadows;
    f.localTiles = std::move(localTiles);
    f.localShadows = localShadows;
    f.localAtlasSize = localAtlasSize;
    f.anyShadows = anyShadows;
    f.tile = tile;
    f.atlasSize = atlasSize;
    f.cascadeMatrices = cascadeMatrices;
    f.reflections = settings.screenSpaceReflections && ssrTrace_ && ssrResolve_ && extent.width >= 8 && extent.height >= 8;
}

} // namespace lumora::render
