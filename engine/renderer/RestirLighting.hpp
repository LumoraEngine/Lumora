#pragma once

#include "renderer/GpuTypes.hpp"
#include "renderer/RenderGraph.hpp"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <nvrhi/nvrhi.h>

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace lumora::render {

class RenderContext;

/// 128-bit ReSTIR DI reservoir packed into RGBA32_UINT on the GPU.
struct GpuReservoir {
    std::uint32_t lightIndex = 0;
    float weightSum = 0.0f;
    std::uint32_t sampleCountM = 0;
    float W = 0.0f;
};
static_assert(sizeof(GpuReservoir) == 16);

/// Configuration parameters for spatio-temporal ReSTIR DI resampling.
struct RestirConfig {
    std::uint32_t initialCandidates = 32;
    std::uint32_t spatialNeighbors = 5;
    float spatialRadiusPx = 16.0f;
    std::uint32_t maxTemporalM = 20;
    float depthThreshold = 0.05f;
    float normalThreshold = 0.9f;
};

/// Surface sample at one pixel for CPU reference evaluation and testing.
struct RestirSurfaceSample {
    glm::vec3 position{0.0f};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    glm::vec3 albedo{1.0f};
    float depth = 1.0f;      // > 0 for valid surfaces, <= 0 for sky / empty
    glm::vec2 motion{0.0f};  // NDC delta (current - previous), matching depth.hlsl
    bool valid = true;
};

/// Per-dispatch inputs for direct NVRHI ReSTIR DI command-list execution.
struct RestirDispatchParams {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depthBindlessIndex = kNoTexture;
    std::uint32_t motionBindlessIndex = kNoTexture;
    std::uint32_t normalBindlessIndex = kNoTexture;
    std::uint32_t albedoBindlessIndex = kNoTexture;
    std::uint32_t prevDepthBindlessIndex = kNoTexture;
    std::uint32_t prevNormalBindlessIndex = kNoTexture;
    nvrhi::ITexture* outputTexture = nullptr;  // RGBA16_FLOAT UAV; uses internal hdrOutput() when null
    bool usePreviousViewProjection = false;
    bool resetHistory = false;
    RestirConfig config{};
};

/// Render-graph inputs for scheduling ReSTIR DI compute passes.
struct RestirGraphInputs {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    RGImage depth{};
    RGImage motion{};
    RGImage normalRoughness{};
    RGImage albedo{};
    RGImage prevDepth{};
    RGImage prevNormal{};
    RGImage outputHdr{};  // optional target; created as a transient RGBA16_FLOAT storage image if invalid
    bool usePreviousViewProjection = true;
    bool resetHistory = false;
    RestirConfig config{};
};

// ---- Deterministic RNG helpers (Wang hash + PCG) -----------------------------------------------

[[nodiscard]] std::uint32_t wangHash(std::uint32_t seed) noexcept;
[[nodiscard]] std::uint32_t pcgNext(std::uint32_t& state) noexcept;
[[nodiscard]] float nextRandom01(std::uint32_t& state) noexcept;

// ---- CPU reference lighting & reservoir engine -------------------------------------------------

/// Smooth inverse-square falloff matching shaders/lighting.hlsli pointAttenuation().
[[nodiscard]] float computePointAttenuation(float distance, float range) noexcept;

/// Unshadowed diffuse direct light contribution f(light, surface) for point and spot lights.
[[nodiscard]] glm::vec3 evaluateLightContribution(const GpuLight& light,
                                                   const RestirSurfaceSample& surface) noexcept;

/// Target sampling PDF pHat(light, surface) = luminance(f(light, surface)).
[[nodiscard]] float evaluateTargetPdf(const GpuLight& light, const RestirSurfaceSample& surface) noexcept;

/// Exact analytical direct lighting sum over all point/spot lights at `surface`.
[[nodiscard]] glm::vec3 evaluateDirectLightingCpu(std::span<const GpuLight> lights,
                                                  const RestirSurfaceSample& surface) noexcept;

/// Weighted Reservoir Sampling (WRS) streaming update.
bool updateReservoir(GpuReservoir& reservoir, std::uint32_t lightIndex, float weight, float random01,
                     std::uint32_t sampleCount = 1) noexcept;

/// Finalizes reservoir unbiased contribution weight W = weightSum / (sampleCountM * targetPdf).
void finalizeReservoirWeight(GpuReservoir& reservoir, float targetPdf) noexcept;

/// Checks depth and normal similarity between two surface samples for edge-stopping reuse.
[[nodiscard]] bool isSurfaceCompatible(const RestirSurfaceSample& a, const RestirSurfaceSample& b,
                                       const RestirConfig& config) noexcept;

/// Streaming pairwise merge of `src` (with M clamped to `maxSrcM`) into `dst`.
/// Assumes `dst.weightSum` stores `dst`'s current accumulated weight; caller finalizes `dst`
/// via `finalizeReservoirWeight`.
bool combineReservoirs(GpuReservoir& dst, const GpuReservoir& src, float targetPdfAtDst, float random01,
                       std::uint32_t maxSrcM = (std::numeric_limits<std::uint32_t>::max)()) noexcept;

/// Unbiased Balance-Heuristic MIS combination of finalized `reservoirs` evaluated at `surfaces[0]`.
/// `reservoirs.size()` must equal `surfaces.size()` and be non-empty.
[[nodiscard]] GpuReservoir combineReservoirs(std::span<const GpuReservoir> reservoirs,
                                             std::span<const RestirSurfaceSample> surfaces,
                                             std::span<const GpuLight> lights,
                                             std::uint32_t& rngState) noexcept;

/// Evaluates one frame of ReSTIR DI (initial WRS + temporal reuse + spatial reuse + resolve) on CPU.
void evaluateRestirFrameCpu(std::uint32_t width, std::uint32_t height,
                            std::span<const RestirSurfaceSample> surfaces,
                            std::span<const GpuLight> lights, const RestirConfig& config,
                            std::uint32_t frameIndex, std::vector<GpuReservoir>& outReservoirs,
                            std::vector<glm::vec3>& outHdrLighting,
                            std::span<const GpuReservoir> prevReservoirs = {},
                            std::span<const RestirSurfaceSample> prevSurfaces = {}) noexcept;

/// Push-constant packing helpers matching shaders/restir_di.hlsl.
[[nodiscard]] PushConstants packRestirInitialPushConstants(const RestirDispatchParams& params,
                                                             bool historyValid,
                                                             std::uint64_t frameIndex) noexcept;
[[nodiscard]] PushConstants packRestirSpatialPushConstants(const RestirDispatchParams& params,
                                                             std::uint64_t frameIndex) noexcept;

// ---- NVRHI GPU pass helper ---------------------------------------------------------------------

/// Manages ping-pong RGBA32_UINT reservoir textures, compute pipelines (`restir.initial.comp` entry
/// `CSInitialTemporal` and `restir.spatial.comp` entry `CSSpatialResolve`), and dispatch helpers.
class RestirPass {
public:
    explicit RestirPass(RenderContext& context);
    ~RestirPass();
    RestirPass(const RestirPass&) = delete;
    RestirPass& operator=(const RestirPass&) = delete;

    /// Ensures ping-pong RGBA32_UINT reservoirs and internal RGBA16_FLOAT output match `(width, height)`.
    void ensureResources(std::uint32_t width, std::uint32_t height);

    /// Invalidates temporal history so the next frame does not reuse stale reservoirs.
    void resetHistory() noexcept { historyValid_ = false; }

    /// Records CSInitialTemporal followed by CSSpatialResolve onto `cmd`.
    void dispatch(nvrhi::ICommandList* cmd, nvrhi::IBindingSet* frameBindings,
                  const RestirDispatchParams& params);

    /// Schedules RestirInitialTemporal and RestirSpatialResolve in `graph`, returning resolved HDR image.
    [[nodiscard]] RGImage addPasses(RenderGraph& graph, const RestirGraphInputs& inputs);

    [[nodiscard]] nvrhi::IComputePipeline* initialPipeline() const noexcept { return initialPipeline_; }
    [[nodiscard]] nvrhi::IComputePipeline* spatialPipeline() const noexcept { return spatialPipeline_; }
    [[nodiscard]] nvrhi::ITexture* currentReservoirTexture() const noexcept {
        const std::size_t idx = frameIndex_ > 0 ? static_cast<std::size_t>((frameIndex_ - 1u) & 1u) : 0u;
        return reservoirs_[idx];
    }
    [[nodiscard]] nvrhi::ITexture* previousReservoirTexture() const noexcept {
        const std::size_t idx = frameIndex_ > 0 ? static_cast<std::size_t>(frameIndex_ & 1u) : 1u;
        return reservoirs_[idx];
    }
    [[nodiscard]] nvrhi::ITexture* hdrOutput() const noexcept { return hdrOutput_; }
    [[nodiscard]] std::uint32_t width() const noexcept { return width_; }
    [[nodiscard]] std::uint32_t height() const noexcept { return height_; }
    [[nodiscard]] std::uint64_t frameIndex() const noexcept { return frameIndex_; }
    [[nodiscard]] bool historyValid() const noexcept { return historyValid_; }

private:
    nvrhi::IBindingSet* getOrCreateStorageSet(std::uint32_t writeIdx, nvrhi::ITexture* hdrOut);
    void releaseTextures();

    RenderContext& ctx_;
    nvrhi::ComputePipelineHandle initialPipeline_;
    nvrhi::ComputePipelineHandle spatialPipeline_;
    std::array<nvrhi::TextureHandle, 2> reservoirs_;
    std::array<std::uint32_t, 2> reservoirBindless_{kNoTexture, kNoTexture};
    nvrhi::TextureHandle hdrOutput_;
    std::uint32_t hdrOutputBindless_ = kNoTexture;
    std::array<nvrhi::BindingSetHandle, 2> storageSets_;
    std::array<nvrhi::ITexture*, 2> cachedHdrTargets_{nullptr, nullptr};
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::uint64_t frameIndex_ = 0;
    bool historyValid_ = false;
};

} // namespace lumora::render
