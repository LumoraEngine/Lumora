#pragma once

#include "renderer/GpuTypes.hpp"
#include "renderer/RenderGraph.hpp"

#include <nvrhi/nvrhi.h>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace lumora {
class Scene;
struct EnvironmentSettings;
} // namespace lumora

namespace lumora::render {

class RenderContext;

/// Axis-aligned world-space proxy box used to build a scene's signed distance field
/// and surfel surface properties.
struct SdfBoxProxy {
    glm::vec3 min{0.0f};
    glm::vec3 max{0.0f};
    glm::vec3 albedo{0.8f};
    glm::vec3 emissive{0.0f};
    float roughness = 0.5f;
};

/// Configuration for voxelizing an SDF & surfel cache volume and sphere-tracing GI.
struct SdfGridConfig {
    glm::vec3 boundsMin{-16.0f, -4.0f, -16.0f};
    glm::vec3 boundsMax{16.0f, 12.0f, 16.0f};
    glm::uvec3 resolution{64u, 64u, 64u};
    std::uint32_t maxSteps = 64u;
    float maxDistance = 32.0f;
    float bounceStrength = 1.0f;
    float skyIntensity = 1.0f;
    std::uint32_t raysPerPoint = 16u;
    float hitThreshold = 0.0f;  // <= 0 derives threshold from minimum voxel size

    [[nodiscard]] glm::vec3 extent() const noexcept {
        return glm::max(boundsMax - boundsMin, glm::vec3(1e-4f));
    }

    [[nodiscard]] glm::uvec3 safeResolution() const noexcept {
        return glm::max(resolution, glm::uvec3(1u));
    }

    [[nodiscard]] glm::vec3 voxelSize() const noexcept {
        return extent() / glm::vec3(safeResolution());
    }

    [[nodiscard]] std::size_t voxelCount() const noexcept {
        const glm::uvec3 res = safeResolution();
        return static_cast<std::size_t>(res.x) * res.y * res.z;
    }
};

/// CPU-side 3D signed distance field and surfel radiance cache (layout: (z * ny + y) * nx + x).
struct SdfVolumeData {
    SdfGridConfig config{};
    /// Signed distance in world units (negative inside geometry, 0 on surface, positive outside).
    std::vector<float> distanceField;
    /// Lit surfel radiance (rgb) and surface roughness / hit validity (a > 0 valid, 0 empty space).
    std::vector<glm::vec4> surfelCache;
    /// Nearest-surface diffuse albedo (rgb) and roughness (a), retained for re-lighting surfels.
    std::vector<glm::vec4> surfelAlbedo;
    /// Nearest-surface emissive radiance (rgb) and surface-band validity mask (a: 1 valid, 0 empty).
    std::vector<glm::vec4> surfelEmissive;
    /// Closest-surface outward normals for sharp direct-lighting evaluation.
    std::vector<glm::vec3> surfelNormals;

    [[nodiscard]] std::size_t voxelIndex(std::uint32_t x, std::uint32_t y, std::uint32_t z) const noexcept {
        const glm::uvec3 res = config.safeResolution();
        return (static_cast<std::size_t>(z) * res.y + y) * res.x + x;
    }

    [[nodiscard]] glm::vec3 voxelWorldPosition(std::uint32_t x, std::uint32_t y, std::uint32_t z) const noexcept {
        const glm::uvec3 res = config.safeResolution();
        const glm::vec3 uvw = (glm::vec3(x, y, z) + 0.5f) / glm::vec3(res);
        return config.boundsMin + uvw * config.extent();
    }
};

/// Hit result returned by CPU reference SDF sphere-tracing.
struct SdfRayHit {
    bool hit = false;
    float distance = 0.0f;
    glm::vec3 position{0.0f};
    glm::vec3 normal{0.0f};
    glm::vec3 radiance{0.0f};
    float roughness = 0.5f;
    std::uint32_t steps = 0u;
};

/// Exact signed distance from `point` to an axis-aligned box [boxMin, boxMax]
/// (negative inside, 0 on surface, positive outside).
[[nodiscard]] float signedDistanceBox(const glm::vec3& point, const glm::vec3& boxMin,
                                      const glm::vec3& boxMax) noexcept;

/// Exact outward surface normal of the nearest face/edge/corner of [boxMin, boxMax].
[[nodiscard]] glm::vec3 boxOutwardNormal(const glm::vec3& point, const glm::vec3& boxMin,
                                         const glm::vec3& boxMax) noexcept;

/// Trilinearly samples the 3D SDF distance at `worldPos` (extrapolated conservatively outside bounds).
[[nodiscard]] float sampleSdfDistance(const SdfVolumeData& volume, const glm::vec3& worldPos) noexcept;

/// Trilinearly samples the 3D surfel radiance cache (rgb = radiance, a = roughness / validity) at `worldPos`.
[[nodiscard]] glm::vec4 sampleSdfSurfel(const SdfVolumeData& volume, const glm::vec3& worldPos) noexcept;

/// Estimates the outward unit surface normal via central finite differences on the SDF.
[[nodiscard]] glm::vec3 computeSdfNormal(const SdfVolumeData& volume, const glm::vec3& worldPos) noexcept;

/// Voxelizes a span of world-space box proxies into `SdfVolumeData` (distance field + surface band).
[[nodiscard]] SdfVolumeData buildSdfVolumeFromBoxes(const SdfGridConfig& cfg, std::span<const SdfBoxProxy> boxes);

/// Injects direct directional sunlight (`sunDir` points TOWARDS the sun) and optional local point/spot
/// lights into `volume.surfelCache` using SDF shadow rays so bounced light carries lit surface colors.
void injectDirectLightingToSurfels(SdfVolumeData& volume, const glm::vec3& sunDir, const glm::vec3& sunColor,
                                   std::span<const GpuLight> localLights = {},
                                   const glm::vec3& ambientColor = glm::vec3(0.0f));

/// Propagates indirect light through the surfel cache for multi-bounce global illumination on CPU.
void propagateSdfMultiBounce(SdfVolumeData& volume, std::uint32_t bounces = 1u, std::uint32_t raysPerVoxel = 8u);

/// Extracts `MeshRendererComponent` + `WorldTransformComponent` bounds and `LightComponent` illumination
/// from `scene` to build and light an `SdfVolumeData`.
[[nodiscard]] SdfVolumeData buildSdfVolumeFromScene(const Scene& scene, const SdfGridConfig& cfg);

/// Sphere-traces a ray through `volume` on the CPU, returning hit distance, surface normal, and surfel radiance.
[[nodiscard]] SdfRayHit traceSdfRay(const SdfVolumeData& volume, const glm::vec3& origin,
                                    const glm::vec3& direction, float maxDistance = -1.0f);

/// Evaluates indirect irradiance at `(position, normal)` via deterministic cosine-weighted hemisphere sampling.
[[nodiscard]] glm::vec3 evaluateSdfGiPoint(const SdfVolumeData& volume, const glm::vec3& position,
                                           const glm::vec3& normal,
                                           const glm::vec3& skyColor = glm::vec3(0.0f),
                                           std::uint32_t sampleCount = 0u, std::uint32_t seed = 1u);

/// Evaluates indirect irradiance using the procedural sky gradient from `EnvironmentSettings` on missed rays.
[[nodiscard]] glm::vec3 evaluateSdfGiPoint(const SdfVolumeData& volume, const glm::vec3& position,
                                           const glm::vec3& normal, const EnvironmentSettings& env,
                                           std::uint32_t sampleCount = 0u, std::uint32_t seed = 1u);

/// GPU helper managing 3D SDF & surfel radiance textures, bindless indices, compute pipeline,
/// and RenderGraph dispatch for `shaders/rtgi_sdf.hlsl` (`rtgi.sdf.comp`).
class SdfGiPass {
public:
    /// PushConstants layout shared with `shaders/rtgi_sdf.hlsl`:
    /// - index0: depth texture bindless index
    /// - index1: 3D SDF distance texture bindless index
    /// - index2: 3D surfel radiance texture bindless index
    /// - flags: bits 0..14 normal texture bindless index (`kNoNormalTexture` = reconstruct from depth),
    ///          bit 15 = config packed flag, bits 16..23 = maxSteps, bits 24..27 = raysPerPixel - 1,
    ///          bits 28..31 = temporal frame phase
    /// - param0..2: half-float packed boundsMin (x,y,z) and boundsMax (x,y,z)
    /// - param3: half-float packed (bounceStrength, skyIntensity)
    static constexpr std::uint32_t kNoNormalTexture = 0x7FFFu;

    explicit SdfGiPass(RenderContext& ctx, bool loadPipeline = true);
    ~SdfGiPass();

    SdfGiPass(const SdfGiPass&) = delete;
    SdfGiPass& operator=(const SdfGiPass&) = delete;
    SdfGiPass(SdfGiPass&&) = delete;
    SdfGiPass& operator=(SdfGiPass&&) = delete;

    /// Uploads `volume` into `R16_FLOAT` SDF and `RGBA16_FLOAT` surfel 3D GPU textures and registers
    /// them in the bindless descriptor table.
    void uploadVolume(const SdfVolumeData& volume);

    /// Creates a transient `RGBA16_FLOAT` indirect irradiance image, dispatches `rtgi.sdf.comp`, and returns it.
    [[nodiscard]] RGImage addPass(RenderGraph& graph, RGImage depth, std::uint32_t width, std::uint32_t height,
                                  RGImage normalRoughness = {}, std::uint32_t frameIndex = 0u);

    /// Dispatches `rtgi.sdf.comp` into a caller-provided storage `RGImage` (`RGBA16_FLOAT`).
    void addDispatchPass(RenderGraph& graph, RGImage giOut, RGImage depth, std::uint32_t width,
                         std::uint32_t height, RGImage normalRoughness = {}, std::uint32_t frameIndex = 0u);

    /// Packs `SdfGridConfig` and bindless indices into 32-byte `PushConstants` for `rtgi_sdf.hlsl`.
    [[nodiscard]] static PushConstants makePushConstants(std::uint32_t depthIndex, std::uint32_t sdfIndex,
                                                         std::uint32_t surfelIndex, const SdfGridConfig& cfg,
                                                         std::uint32_t normalIndex = kNoTexture,
                                                         std::uint32_t frameIndex = 0u) noexcept;

    [[nodiscard]] nvrhi::ITexture* sdfTexture() const noexcept { return sdfTexture_; }
    [[nodiscard]] nvrhi::ITexture* surfelTexture() const noexcept { return surfelTexture_; }
    [[nodiscard]] std::uint32_t sdfBindlessIndex() const noexcept { return sdfIndex_; }
    [[nodiscard]] std::uint32_t surfelBindlessIndex() const noexcept { return surfelIndex_; }
    [[nodiscard]] nvrhi::IComputePipeline* pipeline() const noexcept { return pipeline_; }
    [[nodiscard]] const SdfGridConfig& config() const noexcept { return config_; }
    [[nodiscard]] bool hasVolume() const noexcept { return sdfIndex_ != UINT32_MAX && surfelIndex_ != UINT32_MAX; }
    [[nodiscard]] bool ready() const noexcept { return pipeline_ != nullptr && hasVolume(); }

private:
    void ensurePipeline();
    void releaseTextures();

    RenderContext& ctx_;
    nvrhi::ComputePipelineHandle pipeline_;
    nvrhi::TextureHandle sdfTexture_;
    nvrhi::TextureHandle surfelTexture_;
    std::uint32_t sdfIndex_ = UINT32_MAX;
    std::uint32_t surfelIndex_ = UINT32_MAX;
    SdfGridConfig config_{};
};

} // namespace lumora::render
