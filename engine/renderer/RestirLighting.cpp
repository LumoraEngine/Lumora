#include "renderer/RestirLighting.hpp"

#include "renderer/RenderContext.hpp"
#include "renderer/vulkan/VulkanCommon.hpp"

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>

namespace lumora::render {

namespace {

constexpr std::uint32_t kMaxSpatialNeighborsCap = 64u;
constexpr float kInvPi = 0.31830988618379067154f;

std::uint32_t initPixelRng(std::uint32_t x, std::uint32_t y, std::uint32_t frameIndex,
                           std::uint32_t salt) noexcept {
    const std::uint32_t seed = x * 1973u + y * 9277u + frameIndex * 26699u + salt * 31847u;
    return wangHash(seed) | 1u;
}

} // namespace

std::uint32_t wangHash(std::uint32_t seed) noexcept {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return seed;
}

std::uint32_t pcgNext(std::uint32_t& state) noexcept {
    state = state * 747796405u + 2891336453u;
    const std::uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float nextRandom01(std::uint32_t& state) noexcept {
    // Upper 24 bits mapped to [0.0f, 1.0f) in IEEE-754 float32.
    return static_cast<float>(pcgNext(state) >> 8u) * (1.0f / 16777216.0f);
}

float computePointAttenuation(float distance, float range) noexcept {
    if (!std::isfinite(distance) || !std::isfinite(range) || distance < 0.0f) {
        return 0.0f;
    }
    const float ratio = distance / std::max(range, 1e-4f);
    const float r2 = ratio * ratio;
    const float window = std::clamp(1.0f - r2 * r2, 0.0f, 1.0f);
    return (window * window) / (distance * distance + 1.0f);
}

glm::vec3 evaluateLightContribution(const GpuLight& light, const RestirSurfaceSample& surface) noexcept {
    if (!surface.valid || surface.depth <= 0.0f) {
        return glm::vec3(0.0f);
    }
    const glm::vec3 toLight = glm::vec3(light.positionRange) - surface.position;
    const float distSq = glm::dot(toLight, toLight);
    if (!std::isfinite(distSq) || distSq <= 1e-12f) {
        return glm::vec3(0.0f);
    }
    const float dist = std::sqrt(distSq);
    const glm::vec3 dirToLight = toLight / dist;

    const float normalLenSq = glm::dot(surface.normal, surface.normal);
    if (!std::isfinite(normalLenSq) || normalLenSq <= 1e-8f) {
        return glm::vec3(0.0f);
    }
    const glm::vec3 n = surface.normal / std::sqrt(normalLenSq);
    const float ndotL = std::max(glm::dot(n, dirToLight), 0.0f);
    if (ndotL <= 0.0f) {
        return glm::vec3(0.0f);
    }

    float attenuation = computePointAttenuation(dist, light.positionRange.w);
    if (static_cast<int>(light.directionType.w) == 2) {  // LIGHT_SPOT
        const glm::vec3 rawSpotDir(light.directionType);
        const float dirLenSq = glm::dot(rawSpotDir, rawSpotDir);
        if (dirLenSq > 1e-8f) {
            const glm::vec3 spotDir = rawSpotDir / std::sqrt(dirLenSq);
            const float cosAngle = glm::dot(-dirToLight, spotDir);
            const float edge0 = light.spot.x;
            const float edge1 = std::max(light.spot.y, edge0 + 1e-5f);
            const float t = std::clamp((cosAngle - edge0) / (edge1 - edge0), 0.0f, 1.0f);
            attenuation *= t * t * (3.0f - 2.0f * t);
        }
    }
    if (attenuation <= 0.0f) {
        return glm::vec3(0.0f);
    }

    const glm::vec3 color = glm::max(glm::vec3(light.colorIntensity), glm::vec3(0.0f));
    const float intensity = std::max(light.colorIntensity.w, 0.0f);
    const glm::vec3 albedo = glm::max(surface.albedo, glm::vec3(0.0f));
    return albedo * kInvPi * color * (intensity * attenuation * ndotL);
}

float evaluateTargetPdf(const GpuLight& light, const RestirSurfaceSample& surface) noexcept {
    const glm::vec3 contrib = evaluateLightContribution(light, surface);
    return glm::dot(contrib, glm::vec3(0.2126f, 0.7152f, 0.0722f));
}

glm::vec3 evaluateDirectLightingCpu(std::span<const GpuLight> lights,
                                    const RestirSurfaceSample& surface) noexcept {
    glm::vec3 total(0.0f);
    for (const GpuLight& light : lights) {
        total += evaluateLightContribution(light, surface);
    }
    return total;
}

bool updateReservoir(GpuReservoir& reservoir, std::uint32_t lightIndex, float weight, float random01,
                     std::uint32_t sampleCount) noexcept {
    const float safeWeight = (std::isfinite(weight) && weight > 0.0f) ? weight : 0.0f;
    reservoir.weightSum += safeWeight;
    reservoir.sampleCountM += sampleCount;
    if (safeWeight > 0.0f &&
        (reservoir.weightSum == safeWeight || random01 * reservoir.weightSum < safeWeight)) {
        reservoir.lightIndex = lightIndex;
        return true;
    }
    return false;
}

void finalizeReservoirWeight(GpuReservoir& reservoir, float targetPdf) noexcept {
    if (reservoir.sampleCountM == 0u || !std::isfinite(targetPdf) || targetPdf <= 0.0f ||
        !std::isfinite(reservoir.weightSum) || reservoir.weightSum <= 0.0f) {
        reservoir.W = 0.0f;
        return;
    }
    reservoir.W = reservoir.weightSum / (static_cast<float>(reservoir.sampleCountM) * targetPdf);
}

bool isSurfaceCompatible(const RestirSurfaceSample& a, const RestirSurfaceSample& b,
                         const RestirConfig& config) noexcept {
    if (!a.valid || !b.valid || a.depth <= 0.0f || b.depth <= 0.0f) {
        return false;
    }
    const float relDepthDiff = std::abs(a.depth - b.depth) / std::max(b.depth, 1e-6f);
    if (relDepthDiff > std::max(config.depthThreshold, 0.0f)) {
        return false;
    }
    const float lenSqA = glm::dot(a.normal, a.normal);
    const float lenSqB = glm::dot(b.normal, b.normal);
    if (lenSqA <= 1e-8f || lenSqB <= 1e-8f) {
        return false;
    }
    const glm::vec3 nA = a.normal / std::sqrt(lenSqA);
    const glm::vec3 nB = b.normal / std::sqrt(lenSqB);
    return glm::dot(nA, nB) >= config.normalThreshold;
}

bool combineReservoirs(GpuReservoir& dst, const GpuReservoir& src, float targetPdfAtDst, float random01,
                       std::uint32_t maxSrcM) noexcept {
    const std::uint32_t clampedM = std::min(src.sampleCountM, maxSrcM);
    if (clampedM == 0u) {
        return false;
    }
    const float safePdf = (std::isfinite(targetPdfAtDst) && targetPdfAtDst > 0.0f) ? targetPdfAtDst : 0.0f;
    const float safeW = (std::isfinite(src.W) && src.W > 0.0f) ? src.W : 0.0f;
    const float weight = safePdf * safeW * static_cast<float>(clampedM);
    return updateReservoir(dst, src.lightIndex, weight, random01, clampedM);
}

GpuReservoir combineReservoirs(std::span<const GpuReservoir> reservoirs,
                                 std::span<const RestirSurfaceSample> surfaces,
                                 std::span<const GpuLight> lights,
                                 std::uint32_t& rngState) noexcept {
    GpuReservoir combined{};
    if (reservoirs.empty() || reservoirs.size() != surfaces.size() || lights.empty()) {
        return combined;
    }
    const RestirSurfaceSample& target = surfaces[0];
    if (!target.valid || target.depth <= 0.0f) {
        return combined;
    }

    std::uint32_t totalM = 0u;
    for (const GpuReservoir& r : reservoirs) {
        totalM += r.sampleCountM;
    }
    if (totalM == 0u) {
        return combined;
    }

    for (std::size_t j = 0; j < reservoirs.size(); ++j) {
        const GpuReservoir& src = reservoirs[j];
        if (src.sampleCountM == 0u) {
            continue;
        }
        if (!std::isfinite(src.W) || src.W <= 0.0f || src.lightIndex >= lights.size()) {
            combined.sampleCountM += src.sampleCountM;
            continue;
        }
        const GpuLight& light = lights[src.lightIndex];
        const float pAtTarget = evaluateTargetPdf(light, target);
        float risWeight = 0.0f;
        if (pAtTarget > 0.0f) {
            float denom = 0.0f;
            float num = 0.0f;
            for (std::size_t k = 0; k < reservoirs.size(); ++k) {
                if (reservoirs[k].sampleCountM == 0u) {
                    continue;
                }
                const float pK = evaluateTargetPdf(light, surfaces[k]);
                const float term = static_cast<float>(reservoirs[k].sampleCountM) * pK;
                denom += term;
                if (k == j) {
                    num = term;
                }
            }
            const float misWeight = denom > 0.0f ? (num / denom) : 0.0f;
            risWeight = misWeight * static_cast<float>(totalM) * pAtTarget * src.W;
        }
        updateReservoir(combined, src.lightIndex, risWeight, nextRandom01(rngState), src.sampleCountM);
    }

    if (combined.sampleCountM > 0u && combined.weightSum > 0.0f && combined.lightIndex < lights.size()) {
        const float pFinal = evaluateTargetPdf(lights[combined.lightIndex], target);
        finalizeReservoirWeight(combined, pFinal);
    } else {
        combined.W = 0.0f;
    }
    return combined;
}

void evaluateRestirFrameCpu(std::uint32_t width, std::uint32_t height,
                            std::span<const RestirSurfaceSample> surfaces,
                            std::span<const GpuLight> lights, const RestirConfig& config,
                            std::uint32_t frameIndex, std::vector<GpuReservoir>& outReservoirs,
                            std::vector<glm::vec3>& outHdrLighting,
                            std::span<const GpuReservoir> prevReservoirs,
                            std::span<const RestirSurfaceSample> prevSurfaces) noexcept {
    const std::size_t totalPixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (width == 0u || height == 0u || surfaces.size() != totalPixels) {
        outReservoirs.clear();
        outHdrLighting.clear();
        return;
    }

    outReservoirs.assign(totalPixels, GpuReservoir{});
    outHdrLighting.assign(totalPixels, glm::vec3(0.0f));
    if (lights.empty()) {
        return;
    }

    const std::uint32_t lightCount = static_cast<std::uint32_t>(lights.size());
    const float sourcePdf = 1.0f / static_cast<float>(lightCount);
    const bool hasPrevReservoirs = prevReservoirs.size() == totalPixels;
    const bool hasPrevSurfaces = prevSurfaces.size() == totalPixels;

    // Pass 1: Initial candidate sampling (WRS) + temporal reuse with M clamping.
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t idx = static_cast<std::size_t>(y) * width + x;
            const RestirSurfaceSample& surface = surfaces[idx];
            if (!surface.valid || surface.depth <= 0.0f) {
                continue;
            }

            std::uint32_t rng = initPixelRng(x, y, frameIndex, 1u);
            GpuReservoir initRes{};
            for (std::uint32_t c = 0; c < config.initialCandidates; ++c) {
                const std::uint32_t candidate =
                    std::min(static_cast<std::uint32_t>(nextRandom01(rng) * static_cast<float>(lightCount)),
                             lightCount - 1u);
                const float pHat = evaluateTargetPdf(lights[candidate], surface);
                const float weight = pHat / sourcePdf;
                updateReservoir(initRes, candidate, weight, nextRandom01(rng), 1u);
            }
            if (initRes.sampleCountM > 0u && initRes.weightSum > 0.0f && initRes.lightIndex < lightCount) {
                finalizeReservoirWeight(initRes, evaluateTargetPdf(lights[initRes.lightIndex], surface));
            }

            GpuReservoir temporalRes = initRes;
            if (hasPrevReservoirs && config.maxTemporalM > 0u) {
                const glm::vec2 uv =
                    (glm::vec2(static_cast<float>(x), static_cast<float>(y)) + 0.5f) /
                    glm::vec2(static_cast<float>(width), static_cast<float>(height));
                const glm::vec2 prevUv = uv - surface.motion * 0.5f;
                if (prevUv.x > 0.0f && prevUv.x < 1.0f && prevUv.y > 0.0f && prevUv.y < 1.0f) {
                    const auto px = static_cast<std::uint32_t>(std::clamp(
                        static_cast<int>(std::floor(prevUv.x * static_cast<float>(width))), 0,
                        static_cast<int>(width) - 1));
                    const auto py = static_cast<std::uint32_t>(std::clamp(
                        static_cast<int>(std::floor(prevUv.y * static_cast<float>(height))), 0,
                        static_cast<int>(height) - 1));
                    const std::size_t prevIdx = static_cast<std::size_t>(py) * width + px;
                    const RestirSurfaceSample& prevSurf = hasPrevSurfaces ? prevSurfaces[prevIdx] : surfaces[prevIdx];
                    if (isSurfaceCompatible(surface, prevSurf, config)) {
                        GpuReservoir prevRes = prevReservoirs[prevIdx];
                        prevRes.sampleCountM = std::min(prevRes.sampleCountM, config.maxTemporalM);
                        if (prevRes.sampleCountM > 0u) {
                            const std::array<GpuReservoir, 2> pairRes{initRes, prevRes};
                            const std::array<RestirSurfaceSample, 2> pairSurf{surface, prevSurf};
                            temporalRes = combineReservoirs(pairRes, pairSurf, lights, rng);
                        }
                    }
                }
            }
            outReservoirs[idx] = temporalRes;
        }
    }

    // Pass 2: Spatial neighbor reuse with edge-stopping and unbiased MIS normalization + resolve.
    const std::uint32_t neighborCap = std::min(config.spatialNeighbors, kMaxSpatialNeighborsCap);
    const float radiusPx = std::max(config.spatialRadiusPx, 0.0f);
    std::vector<GpuReservoir> taps;
    std::vector<RestirSurfaceSample> tapSurfaces;
    taps.reserve(1u + neighborCap);
    tapSurfaces.reserve(1u + neighborCap);

    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t idx = static_cast<std::size_t>(y) * width + x;
            const RestirSurfaceSample& surface = surfaces[idx];
            if (!surface.valid || surface.depth <= 0.0f) {
                continue;
            }

            std::uint32_t spatialRng = initPixelRng(x, y, frameIndex, 2u);
            taps.clear();
            tapSurfaces.clear();
            taps.push_back(outReservoirs[idx]);
            tapSurfaces.push_back(surface);

            for (std::uint32_t n = 0; n < neighborCap; ++n) {
                const float angle = glm::two_pi<float>() * nextRandom01(spatialRng);
                const float r = radiusPx * std::sqrt(nextRandom01(spatialRng));
                const int nx = static_cast<int>(std::lround(static_cast<float>(x) + std::cos(angle) * r));
                const int ny = static_cast<int>(std::lround(static_cast<float>(y) + std::sin(angle) * r));
                if (nx < 0 || ny < 0 || nx >= static_cast<int>(width) || ny >= static_cast<int>(height)) {
                    continue;
                }
                if (nx == static_cast<int>(x) && ny == static_cast<int>(y)) {
                    continue;
                }
                const std::size_t nIdx =
                    static_cast<std::size_t>(ny) * width + static_cast<std::size_t>(nx);
                const RestirSurfaceSample& neighborSurf = surfaces[nIdx];
                if (!isSurfaceCompatible(surface, neighborSurf, config)) {
                    continue;
                }
                taps.push_back(outReservoirs[nIdx]);
                tapSurfaces.push_back(neighborSurf);
            }

            const GpuReservoir resolved = combineReservoirs(taps, tapSurfaces, lights, spatialRng);
            if (resolved.W > 0.0f && resolved.lightIndex < lightCount) {
                outHdrLighting[idx] = evaluateLightContribution(lights[resolved.lightIndex], surface) * resolved.W;
            }
        }
    }
}

PushConstants packRestirInitialPushConstants(const RestirDispatchParams& params, bool historyValid,
                                             std::uint64_t frameIndex) noexcept {
    PushConstants pc;
    pc.index0 = params.depthBindlessIndex;
    pc.index1 = params.motionBindlessIndex;
    pc.index2 = params.normalBindlessIndex;

    const std::uint32_t initCand = std::min(params.config.initialCandidates, 1023u);
    const std::uint32_t maxTempM = std::min(params.config.maxTemporalM, 1023u);
    const std::uint32_t frameBits = static_cast<std::uint32_t>(frameIndex & 0x3FFu);
    pc.flags = (historyValid ? 1u : 0u) | (params.usePreviousViewProjection ? 2u : 0u) | (initCand << 2u) |
               (maxTempM << 12u) | (frameBits << 22u);

    pc.param0 = std::bit_cast<float>(params.prevDepthBindlessIndex);
    pc.param1 = std::bit_cast<float>(params.prevNormalBindlessIndex);
    pc.param2 = params.config.depthThreshold;
    pc.param3 = params.config.normalThreshold;
    return pc;
}

PushConstants packRestirSpatialPushConstants(const RestirDispatchParams& params,
                                             std::uint64_t frameIndex) noexcept {
    PushConstants pc;
    pc.index0 = params.depthBindlessIndex;
    pc.index1 = params.albedoBindlessIndex;
    pc.index2 = params.normalBindlessIndex;

    const std::uint32_t neighbors = std::min(params.config.spatialNeighbors, 16u);
    pc.flags = (neighbors << 1u);

    pc.param0 = params.config.spatialRadiusPx;
    pc.param1 = std::bit_cast<float>(static_cast<std::uint32_t>(frameIndex));
    pc.param2 = params.config.depthThreshold;
    pc.param3 = params.config.normalThreshold;
    return pc;
}

// ---- RestirPass --------------------------------------------------------------------------------

RestirPass::RestirPass(RenderContext& context) : ctx_(context) {
    initialPipeline_ = ctx_.createComputePipeline("restir.initial.comp", "CSInitialTemporal");
    spatialPipeline_ = ctx_.createComputePipeline("restir.spatial.comp", "CSSpatialResolve");
}

RestirPass::~RestirPass() { releaseTextures(); }

void RestirPass::releaseTextures() {
    for (std::size_t i = 0; i < reservoirs_.size(); ++i) {
        if (reservoirBindless_[i] != kNoTexture) {
            ctx_.bindless().releaseTexture(reservoirBindless_[i]);
            reservoirBindless_[i] = kNoTexture;
        }
        reservoirs_[i] = nullptr;
    }
    if (hdrOutputBindless_ != kNoTexture) {
        ctx_.bindless().releaseTexture(hdrOutputBindless_);
        hdrOutputBindless_ = kNoTexture;
    }
    hdrOutput_ = nullptr;
    storageSets_ = {};
    cachedHdrTargets_ = {nullptr, nullptr};
}

void RestirPass::ensureResources(std::uint32_t width, std::uint32_t height) {
    if (width == 0u || height == 0u) {
        return;
    }
    if (width == width_ && height == height_ && reservoirs_[0] && reservoirs_[1] && hdrOutput_) {
        return;
    }

    releaseTextures();
    nvrhi::IDevice* gpu = ctx_.gpu();
    for (std::size_t i = 0; i < reservoirs_.size(); ++i) {
        nvrhi::TextureDesc desc;
        desc.width = width;
        desc.height = height;
        desc.format = nvrhi::Format::RGBA32_UINT;
        desc.isUAV = true;
        desc.debugName = i == 0u ? "RestirReservoirsA" : "RestirReservoirsB";
        desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        desc.keepInitialState = true;
        reservoirs_[i] = gpu->createTexture(desc);
        if (!reservoirs_[i]) {
            throw rhi::VulkanError("could not create ReSTIR reservoir texture", VK_ERROR_OUT_OF_DEVICE_MEMORY);
        }
        reservoirBindless_[i] = ctx_.bindless().registerTexture(reservoirs_[i]);
    }

    nvrhi::TextureDesc hdrDesc;
    hdrDesc.width = width;
    hdrDesc.height = height;
    hdrDesc.format = nvrhi::Format::RGBA16_FLOAT;
    hdrDesc.isShaderResource = true;
    hdrDesc.isUAV = true;
    hdrDesc.debugName = "RestirDirectLighting";
    hdrDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    hdrDesc.keepInitialState = true;
    hdrOutput_ = gpu->createTexture(hdrDesc);
    if (!hdrOutput_) {
        throw rhi::VulkanError("could not create ReSTIR output HDR texture", VK_ERROR_OUT_OF_DEVICE_MEMORY);
    }
    hdrOutputBindless_ = ctx_.bindless().registerTexture(hdrOutput_);

    width_ = width;
    height_ = height;
    historyValid_ = false;
}

nvrhi::IBindingSet* RestirPass::getOrCreateStorageSet(std::uint32_t writeIdx, nvrhi::ITexture* hdrOut) {
    const std::size_t slotIdx = static_cast<std::size_t>(writeIdx & 1u);
    if (storageSets_[slotIdx] && cachedHdrTargets_[slotIdx] == hdrOut) {
        return storageSets_[slotIdx];
    }

    const std::size_t readIdx = 1u - slotIdx;
    nvrhi::BindingSetDesc setDesc;
    setDesc.addItem(nvrhi::BindingSetItem::Texture_UAV(0, reservoirs_[slotIdx]));
    setDesc.addItem(nvrhi::BindingSetItem::Texture_UAV(1, hdrOut));
    setDesc.addItem(nvrhi::BindingSetItem::Texture_UAV(2, reservoirs_[readIdx]));
    setDesc.addItem(nvrhi::BindingSetItem::Texture_UAV(3, ctx_.storageFallback()));

    nvrhi::BindingSetHandle handle = ctx_.gpu()->createBindingSet(setDesc, ctx_.storageLayout());
    if (!handle) {
        throw rhi::VulkanError("could not create ReSTIR storage binding set", VK_ERROR_INITIALIZATION_FAILED);
    }
    storageSets_[slotIdx] = handle;
    cachedHdrTargets_[slotIdx] = hdrOut;
    return storageSets_[slotIdx];
}

void RestirPass::dispatch(nvrhi::ICommandList* cmd, nvrhi::IBindingSet* frameBindings,
                          const RestirDispatchParams& params) {
    if (cmd == nullptr || frameBindings == nullptr || params.width == 0u || params.height == 0u) {
        return;
    }
    if (params.depthBindlessIndex == kNoTexture || params.normalBindlessIndex == kNoTexture) {
        return;
    }

    ensureResources(params.width, params.height);
    if (params.resetHistory) {
        historyValid_ = false;
    }

    const std::uint32_t writeIdx = static_cast<std::uint32_t>(frameIndex_ & 1u);
    const std::uint32_t readIdx = 1u - writeIdx;
    nvrhi::ITexture* hdrOut = params.outputTexture != nullptr ? params.outputTexture : hdrOutput_.Get();
    nvrhi::IBindingSet* storageSet = getOrCreateStorageSet(writeIdx, hdrOut);
    nvrhi::IDescriptorTable* bindlessTable = ctx_.bindless().table();

    const std::uint32_t groupsX = (params.width + 7u) / 8u;
    const std::uint32_t groupsY = (params.height + 7u) / 8u;

    // Ensure any previous writes on the same command list are visible before reading prev reservoirs.
    cmd->setTextureState(reservoirs_[readIdx], nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    cmd->setTextureState(reservoirs_[writeIdx], nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    cmd->commitBarriers();

    // Pass 1: Initial candidate sampling + temporal reprojection & reuse.
    {
        nvrhi::ComputeState state;
        state.pipeline = initialPipeline_;
        state.bindings = {frameBindings, bindlessTable, storageSet};
        cmd->setComputeState(state);

        const PushConstants pc = packRestirInitialPushConstants(params, historyValid_, frameIndex_);
        cmd->setPushConstants(&pc, sizeof(pc));
        cmd->dispatch(groupsX, groupsY, 1u);
    }

    // Explicit UAV barrier: NVRHI skips binding-set barriers when bindings match the previous dispatch.
    cmd->setTextureState(reservoirs_[writeIdx], nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    cmd->commitBarriers();

    // Pass 2: Spatial neighbor reuse + final light resolve to HDR target.
    {
        nvrhi::ComputeState state;
        state.pipeline = spatialPipeline_;
        state.bindings = {frameBindings, bindlessTable, storageSet};
        cmd->setComputeState(state);

        const PushConstants pc = packRestirSpatialPushConstants(params, frameIndex_);
        cmd->setPushConstants(&pc, sizeof(pc));
        cmd->dispatch(groupsX, groupsY, 1u);
    }

    historyValid_ = true;
    ++frameIndex_;
}

RGImage RestirPass::addPasses(RenderGraph& graph, const RestirGraphInputs& inputs) {
    if (inputs.width == 0u || inputs.height == 0u || !inputs.depth.valid() || !inputs.normalRoughness.valid()) {
        return {};
    }

    ensureResources(inputs.width, inputs.height);
    if (inputs.resetHistory) {
        historyValid_ = false;
    }

    const std::uint32_t writeIdx = static_cast<std::uint32_t>(frameIndex_ & 1u);
    const std::uint32_t readIdx = 1u - writeIdx;

    const RGImage currReservoirs =
        graph.importImage("RestirReservoirsCurrent", {reservoirs_[writeIdx], reservoirBindless_[writeIdx]});
    const RGImage prevReservoirs =
        graph.importImage("RestirReservoirsPrevious", {reservoirs_[readIdx], reservoirBindless_[readIdx]});
    const RGImage hdrOut = inputs.outputHdr.valid()
                               ? inputs.outputHdr
                               : graph.createImage("RestirDirectLighting",
                                                   {nvrhi::Format::RGBA16_FLOAT, inputs.width, inputs.height, true});

    const bool historyValid = historyValid_;
    const std::uint64_t frame = frameIndex_;
    const std::uint32_t groupsX = (inputs.width + 7u) / 8u;
    const std::uint32_t groupsY = (inputs.height + 7u) / 8u;
    nvrhi::IDescriptorTable* bindlessTable = ctx_.bindless().table();

    graph.addComputePass(
        "RestirInitialTemporal",
        [&](PassBuilder& b) {
            b.writeStorage(currReservoirs);  // u0
            b.writeStorage(hdrOut);          // u1 (format_rgba16f match)
            b.writeStorage(prevReservoirs);  // u2
            b.sample(inputs.depth);
            if (inputs.motion.valid()) b.sample(inputs.motion);
            b.sample(inputs.normalRoughness);
            if (inputs.prevDepth.valid()) b.sample(inputs.prevDepth);
            if (inputs.prevNormal.valid()) b.sample(inputs.prevNormal);
        },
        [this, inputs, historyValid, frame, groupsX, groupsY, bindlessTable](PassContext& ctx) {
            RestirDispatchParams dp;
            dp.width = inputs.width;
            dp.height = inputs.height;
            dp.depthBindlessIndex = ctx.bindlessIndex(inputs.depth);
            dp.motionBindlessIndex = inputs.motion.valid() ? ctx.bindlessIndex(inputs.motion) : kNoTexture;
            dp.normalBindlessIndex = ctx.bindlessIndex(inputs.normalRoughness);
            dp.prevDepthBindlessIndex = inputs.prevDepth.valid() ? ctx.bindlessIndex(inputs.prevDepth) : kNoTexture;
            dp.prevNormalBindlessIndex = inputs.prevNormal.valid() ? ctx.bindlessIndex(inputs.prevNormal) : kNoTexture;
            dp.usePreviousViewProjection = inputs.usePreviousViewProjection;
            dp.config = inputs.config;

            nvrhi::ComputeState state;
            state.pipeline = initialPipeline_;
            state.bindings = {ctx.frameBindings(), bindlessTable, ctx.storageBindings()};
            ctx.cmd->setComputeState(state);
            const PushConstants pc = packRestirInitialPushConstants(dp, historyValid, frame);
            ctx.cmd->setPushConstants(&pc, sizeof(pc));
            ctx.cmd->dispatch(groupsX, groupsY, 1u);
        });

    graph.addComputePass(
        "RestirSpatialResolve",
        [&](PassBuilder& b) {
            b.writeStorage(currReservoirs);  // u0
            b.writeStorage(hdrOut);          // u1
            b.sample(inputs.depth);
            b.sample(inputs.normalRoughness);
            if (inputs.albedo.valid()) b.sample(inputs.albedo);
            if (inputs.localShadowAtlas.valid()) b.sample(inputs.localShadowAtlas);
        },
        [this, inputs, frame, groupsX, groupsY, bindlessTable](PassContext& ctx) {
            RestirDispatchParams dp;
            dp.width = inputs.width;
            dp.height = inputs.height;
            dp.depthBindlessIndex = ctx.bindlessIndex(inputs.depth);
            dp.albedoBindlessIndex = inputs.albedo.valid() ? ctx.bindlessIndex(inputs.albedo) : kNoTexture;
            dp.normalBindlessIndex = ctx.bindlessIndex(inputs.normalRoughness);
            dp.config = inputs.config;

            nvrhi::ComputeState state;
            state.pipeline = spatialPipeline_;
            state.bindings = {ctx.frameBindings(), bindlessTable, ctx.storageBindings()};
            ctx.cmd->setComputeState(state);
            const PushConstants pc = packRestirSpatialPushConstants(dp, frame);
            ctx.cmd->setPushConstants(&pc, sizeof(pc));
            ctx.cmd->dispatch(groupsX, groupsY, 1u);
        });

    historyValid_ = true;
    ++frameIndex_;
    return hdrOut;
}

} // namespace lumora::render
