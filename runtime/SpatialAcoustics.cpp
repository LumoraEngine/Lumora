#include "runtime/SpatialAcoustics.hpp"

#include <glm/geometric.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <vector>

namespace lumora {

namespace {

constexpr float kSqrtHalf = 0.70710678118654752440f;
constexpr float kGoldenAngleRad = 2.39996322972865332f;  // pi * (3 - sqrt(5))
constexpr float kDenormalThreshold = 1e-24f;

[[nodiscard]] inline float flushDenormal(float value) noexcept {
    return (std::abs(value) < kDenormalThreshold) ? 0.0f : value;
}

[[nodiscard]] inline float smoothstep01(float t) noexcept {
    const float x = std::clamp(t, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

void buildOrthonormalBasis(const glm::vec3& dir, glm::vec3& outTangent, glm::vec3& outBitangent) noexcept {
    const glm::vec3 reference = (std::abs(dir.y) < 0.99f) ? glm::vec3(0.0f, 1.0f, 0.0f)
                                                          : glm::vec3(1.0f, 0.0f, 0.0f);
    outTangent = glm::normalize(glm::cross(reference, dir));
    outBitangent = glm::cross(dir, outTangent);
}

constexpr std::array<float, 4> kCombDelaysSecLeft = {0.0297f, 0.0349f, 0.0397f, 0.0437f};
constexpr std::array<float, 4> kCombDelaysSecRight = {0.0302f, 0.0354f, 0.0402f, 0.0442f};
constexpr std::array<float, 2> kAllPassDelaysSecLeft = {0.0051f, 0.00173f};
constexpr std::array<float, 2> kAllPassDelaysSecRight = {0.0054f, 0.00191f};

} // namespace

SpatialAcousticsSolver::SpatialAcousticsSolver(const SpatialAcousticsConfig& config) noexcept
    : config_(config) {}

OcclusionResult SpatialAcousticsSolver::evaluateOcclusion(
    const glm::vec3& sourcePos,
    const glm::vec3& listenerPos,
    const AcousticRaycastFn& raycast) const {
    const float minHz = std::clamp(config_.minLpfCutoffHz, 20.0f, 20000.0f);
    const float maxHz = std::max(minHz, config_.maxLpfCutoffHz);

    if (!raycast) {
        return OcclusionResult{0.0f, 1.0f, maxHz};
    }

    const glm::vec3 delta = listenerPos - sourcePos;
    const float distance = glm::length(delta);
    if (!std::isfinite(distance) || distance <= 1e-4f) {
        return OcclusionResult{0.0f, 1.0f, maxHz};
    }

    const glm::vec3 dir = delta / distance;
    const float maxRayDist = std::max(distance - 1e-3f, 1e-4f);
    const float probeRadius = std::min(std::max(config_.occlusionProbeRadius, 0.0f), distance * 0.2f);
    const float logMaxHz = std::log(maxHz);

    auto traceWeightedRay = [&](const glm::vec3& originOffset, float& outOcc, float& outTrans, float& outLogCut) {
        const auto hit = raycast(sourcePos + originOffset, dir, maxRayDist);
        if (hit.has_value() && std::isfinite(hit->distance) && hit->distance >= 0.0f && hit->distance <= maxRayDist) {
            const float trans = std::clamp(hit->material.transmission, 0.0f, 1.0f);
            const float occ = 1.0f - trans;
            const float matMinHz = std::clamp(hit->material.lpfCutoffMinHz, minHz, maxHz);
            outOcc = occ;
            outTrans = trans;
            outLogCut = std::lerp(logMaxHz, std::log(matMinHz), occ);
        } else {
            outOcc = 0.0f;
            outTrans = 1.0f;
            outLogCut = logMaxHz;
        }
    };

    if (probeRadius <= 1e-5f) {
        float occ = 0.0f;
        float trans = 1.0f;
        float logCut = logMaxHz;
        traceWeightedRay(glm::vec3(0.0f), occ, trans, logCut);
        return OcclusionResult{occ, trans, std::clamp(std::exp(logCut), minHz, maxHz)};
    }

    glm::vec3 tangent{1.0f, 0.0f, 0.0f};
    glm::vec3 bitangent{0.0f, 1.0f, 0.0f};
    buildOrthonormalBasis(dir, tangent, bitangent);

    const float halfR = probeRadius * 0.5f;
    const float diagR = probeRadius * kSqrtHalf;

    struct RaySample {
        glm::vec3 offset;
        float weight;
    };

    const std::array<RaySample, 9> samples = {{
        {glm::vec3(0.0f), 0.36f},
        {tangent * halfR, 0.09f},
        {-tangent * halfR, 0.09f},
        {bitangent * halfR, 0.09f},
        {-bitangent * halfR, 0.09f},
        {(tangent + bitangent) * diagR, 0.07f},
        {(tangent - bitangent) * diagR, 0.07f},
        {(-tangent + bitangent) * diagR, 0.07f},
        {(-tangent - bitangent) * diagR, 0.07f},
    }};

    float totalOcclusion = 0.0f;
    float totalTransmission = 0.0f;
    float totalLogCutoff = 0.0f;

    for (const RaySample& sample : samples) {
        float occ = 0.0f;
        float trans = 1.0f;
        float logCut = logMaxHz;
        traceWeightedRay(sample.offset, occ, trans, logCut);
        totalOcclusion += sample.weight * occ;
        totalTransmission += sample.weight * trans;
        totalLogCutoff += sample.weight * logCut;
    }

    return OcclusionResult{
        std::clamp(totalOcclusion, 0.0f, 1.0f),
        std::clamp(totalTransmission, 0.0f, 1.0f),
        std::clamp(std::exp(totalLogCutoff), minHz, maxHz),
    };
}

EnvironmentReverbResult SpatialAcousticsSolver::estimateEnvironmentReverb(
    const glm::vec3& listenerPos,
    const AcousticRaycastFn& raycast,
    std::span<const AcousticZone> zones) const {
    const float minRt60 = std::max(config_.minRt60Sec, 0.05f);
    const float maxRt60 = std::max(minRt60, config_.maxRt60Sec);
    const float maxWet = std::clamp(config_.maxReverbWet, 0.0f, 1.0f);

    EnvironmentReverbResult result{};
    result.rt60Sec = minRt60;
    result.wetMix = 0.0f;

    if (raycast) {
        const std::uint32_t rayCount = std::max<std::uint32_t>(config_.reverbRayCount, 8u);
        const float maxRayDist = std::max(config_.reverbMaxRayDistance, 1.0f);

        std::uint32_t hitCount = 0;
        float hitDistSum = 0.0f;
        float hitAbsorptionSum = 0.0f;

        for (std::uint32_t i = 0; i < rayCount; ++i) {
            const float y = 1.0f - (2.0f * (static_cast<float>(i) + 0.5f)) / static_cast<float>(rayCount);
            const float r = std::sqrt(std::max(0.0f, 1.0f - y * y));
            const float theta = kGoldenAngleRad * static_cast<float>(i);
            const glm::vec3 dir{std::cos(theta) * r, y, std::sin(theta) * r};

            const auto hit = raycast(listenerPos, dir, maxRayDist);
            if (hit.has_value() && std::isfinite(hit->distance) && hit->distance >= 0.0f &&
                hit->distance <= maxRayDist) {
                if (hit->material.transmission < 0.999f || hit->material.absorption > 0.001f) {
                    ++hitCount;
                    hitDistSum += hit->distance;
                    hitAbsorptionSum += std::clamp(hit->material.absorption, 0.0f, 1.0f);
                }
            }
        }

        const float enclosureFactor = static_cast<float>(hitCount) / static_cast<float>(rayCount);
        result.enclosureFactor = enclosureFactor;

        if (hitCount > 0) {
            const float meanHitDist = hitDistSum / static_cast<float>(hitCount);
            const float meanWallAbsorption = hitAbsorptionSum / static_cast<float>(hitCount);
            const float meanFreePath = meanHitDist * 1.4f;

            // Escaped sky rays act as complete energy loss (absorption = 1.0).
            const float totalAbsorption =
                (hitAbsorptionSum + static_cast<float>(rayCount - hitCount)) / static_cast<float>(rayCount);
            const float eyringAlpha = -std::log(1.0f - std::clamp(totalAbsorption, 0.0f, 0.985f));
            const float airAndLeakage = 0.06f + 0.002f * meanFreePath;
            const float effectiveAlpha = std::max(eyringAlpha, 0.01f) + airAndLeakage;

            // Sabine/Eyring RT60 in terms of mean free path lambda = 4V/S: RT60 = (0.161 / 4) * lambda / alpha.
            const float rawRt60 = (0.04025f * meanFreePath) / effectiveAlpha;

            const float reflectivity = 1.0f - meanWallAbsorption;
            const float sizeFactor = std::clamp(meanFreePath / 20.0f, 0.0f, 1.0f);
            const float enclosureWeight = enclosureFactor * enclosureFactor;
            const float rawWet = (0.18f + 0.42f * sizeFactor) * reflectivity * enclosureWeight;

            result.rt60Sec = std::clamp(rawRt60, minRt60, maxRt60);
            result.wetMix = std::clamp(rawWet, 0.0f, maxWet);
            result.meanFreePathMeters = meanFreePath;
            result.meanAbsorption = meanWallAbsorption;
        }
    }

    if (!zones.empty()) {
        struct ActiveZone {
            float volume;
            float weight;
            float rt60Sec;
            float wetMix;
        };

        std::array<ActiveZone, 8> stackZones{};
        std::size_t stackCount = 0;
        std::vector<ActiveZone> heapZones;

        const float blendDist = std::max(config_.zoneBlendDistance, 1e-4f);

        for (const AcousticZone& zone : zones) {
            const glm::vec3 bMin = glm::min(zone.boundsMin, zone.boundsMax);
            const glm::vec3 bMax = glm::max(zone.boundsMin, zone.boundsMax);
            const glm::vec3 clamped = glm::clamp(listenerPos, bMin, bMax);
            const float outsideDist = glm::length(listenerPos - clamped);

            float weight = 0.0f;
            if (outsideDist <= 0.0f) {
                weight = 1.0f;
            } else if (outsideDist < blendDist) {
                weight = smoothstep01(1.0f - outsideDist / blendDist);
            }

            if (weight > 0.0f) {
                const glm::vec3 ext = glm::max(bMax - bMin, glm::vec3(1e-2f));
                const ActiveZone active{
                    ext.x * ext.y * ext.z,
                    weight,
                    std::clamp(zone.rt60Sec, minRt60, maxRt60),
                    std::clamp(zone.wetMix, 0.0f, 1.0f),
                };
                if (heapZones.empty() && stackCount < stackZones.size()) {
                    stackZones[stackCount++] = active;
                } else {
                    if (heapZones.empty()) {
                        heapZones.assign(stackZones.begin(), stackZones.end());
                    }
                    heapZones.push_back(active);
                }
            }
        }

        auto applyZones = [&](ActiveZone* begin, ActiveZone* end) {
            // Composite from largest enclosing volume to smallest nested volume so inner zones
            // smoothly override outer zones without abrupt snapping at boundaries.
            std::sort(begin, end, [](const ActiveZone& a, const ActiveZone& b) {
                return a.volume > b.volume;
            });
            for (const ActiveZone* it = begin; it != end; ++it) {
                result.rt60Sec = std::lerp(result.rt60Sec, it->rt60Sec, it->weight);
                result.wetMix = std::lerp(result.wetMix, it->wetMix, it->weight);
            }
        };

        if (!heapZones.empty()) {
            applyZones(heapZones.data(), heapZones.data() + heapZones.size());
        } else if (stackCount > 0) {
            applyZones(stackZones.data(), stackZones.data() + stackCount);
        }
    }

    return result;
}

float SpatialAcousticsSolver::computeDopplerPitch(
    const glm::vec3& sourcePos,
    const glm::vec3& sourceVel,
    const glm::vec3& listenerPos,
    const glm::vec3& listenerVel,
    float speedOfSound,
    float scale) noexcept {
    if (!std::isfinite(speedOfSound) || speedOfSound <= 1e-3f || !std::isfinite(scale) || scale <= 0.0f) {
        return 1.0f;
    }

    const glm::vec3 toSource = sourcePos - listenerPos;
    const float dist = glm::length(toSource);
    if (!std::isfinite(dist) || dist <= 1e-4f) {
        return 1.0f;
    }

    const glm::vec3 dirListenerToSource = toSource / dist;
    const float listenerRadialVel = glm::dot(listenerVel, dirListenerToSource) * scale;
    const float sourceRadialVel = glm::dot(sourceVel, dirListenerToSource) * scale;

    if (!std::isfinite(listenerRadialVel) || !std::isfinite(sourceRadialVel)) {
        return 1.0f;
    }

    // Clamp radial speeds below Mach 1 singularity before division, then clamp output pitch to [0.5, 2.0].
    const float numerator = std::max(0.0f, speedOfSound + listenerRadialVel);
    const float denominator = std::max(1e-3f, speedOfSound + sourceRadialVel);
    return std::clamp(numerator / denominator, 0.5f, 2.0f);
}

AcousticSourceResult SpatialAcousticsSolver::evaluateSource(
    const glm::vec3& sourcePos,
    const glm::vec3& sourceVel,
    const glm::vec3& listenerPos,
    const glm::vec3& listenerVel,
    const AcousticRaycastFn& raycast,
    std::span<const AcousticZone> zones,
    float minDistance,
    float maxDistance) const {
    const EnvironmentReverbResult env = estimateEnvironmentReverb(listenerPos, raycast, zones);
    return evaluateSource(sourcePos, sourceVel, listenerPos, listenerVel, raycast, env, minDistance, maxDistance);
}

AcousticSourceResult SpatialAcousticsSolver::evaluateSource(
    const glm::vec3& sourcePos,
    const glm::vec3& sourceVel,
    const glm::vec3& listenerPos,
    const glm::vec3& listenerVel,
    const AcousticRaycastFn& raycast,
    const EnvironmentReverbResult& cachedEnvironment,
    float minDistance,
    float maxDistance) const {
    const OcclusionResult occ = evaluateOcclusion(sourcePos, listenerPos, raycast);

    const float dist = glm::length(listenerPos - sourcePos);
    const float safeMin = std::max(minDistance, 0.01f);
    const float safeMax = std::max(maxDistance, safeMin + 0.01f);
    const float clampedDist = std::isfinite(dist) ? std::clamp(dist, safeMin, safeMax) : safeMin;
    const float distanceGain = safeMin / clampedDist;

    AcousticSourceResult out{};
    out.occlusion = occ.occlusion;
    out.lpfCutoffHz = occ.lpfCutoffHz;
    out.attenuationGain = std::clamp(distanceGain * occ.transmissionGain, 0.0f, 1.0f);
    out.reverbWet = cachedEnvironment.wetMix;
    out.reverbRt60Sec = cachedEnvironment.rt60Sec;
    out.dopplerPitch = computeDopplerPitch(
        sourcePos, sourceVel, listenerPos, listenerVel, config_.dopplerSpeedOfSound, config_.dopplerScale);
    return out;
}

AcousticSourceResult SpatialAcousticsSolver::evaluateSource(
    const glm::vec3& sourcePos,
    const glm::vec3& listenerPos,
    const AcousticRaycastFn& raycast,
    std::span<const AcousticZone> zones,
    float minDistance,
    float maxDistance) const {
    return evaluateSource(
        sourcePos, glm::vec3(0.0f), listenerPos, glm::vec3(0.0f), raycast, zones, minDistance, maxDistance);
}

// -----------------------------------------------------------------------------
// BiquadLowPassFilter (Zero-Alloc, Lock-Free Real-Time DSP)
// -----------------------------------------------------------------------------

static_assert(std::atomic<float>::is_always_lock_free, "Audio DSP atomics must be lock-free");

BiquadLowPassFilter::BiquadLowPassFilter(const BiquadLowPassFilter& other) noexcept
    : targetSampleRate_(other.targetSampleRate_.load(std::memory_order_relaxed)),
      targetCutoffHz_(other.targetCutoffHz_.load(std::memory_order_relaxed)),
      activeSampleRate_(other.activeSampleRate_),
      activeCutoffHz_(other.activeCutoffHz_),
      b0_(other.b0_),
      b1_(other.b1_),
      b2_(other.b2_),
      a1_(other.a1_),
      a2_(other.a2_),
      z1_(other.z1_),
      z2_(other.z2_) {}

BiquadLowPassFilter& BiquadLowPassFilter::operator=(const BiquadLowPassFilter& other) noexcept {
    if (this != &other) {
        targetSampleRate_.store(other.targetSampleRate_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        targetCutoffHz_.store(other.targetCutoffHz_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        activeSampleRate_ = other.activeSampleRate_;
        activeCutoffHz_ = other.activeCutoffHz_;
        b0_ = other.b0_;
        b1_ = other.b1_;
        b2_ = other.b2_;
        a1_ = other.a1_;
        a2_ = other.a2_;
        z1_ = other.z1_;
        z2_ = other.z2_;
    }
    return *this;
}

BiquadLowPassFilter::BiquadLowPassFilter(BiquadLowPassFilter&& other) noexcept
    : BiquadLowPassFilter(static_cast<const BiquadLowPassFilter&>(other)) {}

BiquadLowPassFilter& BiquadLowPassFilter::operator=(BiquadLowPassFilter&& other) noexcept {
    return *this = static_cast<const BiquadLowPassFilter&>(other);
}

void BiquadLowPassFilter::prepare(float sampleRate) noexcept {
    const float safeRate = std::isfinite(sampleRate) ? std::max(sampleRate, 1000.0f) : 48000.0f;
    targetSampleRate_.store(safeRate, std::memory_order_relaxed);
    updateCoefficients(safeRate, targetCutoffHz_.load(std::memory_order_relaxed));
    reset();
}

void BiquadLowPassFilter::setCutoff(float cutoffHz) noexcept {
    const float safeCutoff = std::isfinite(cutoffHz) ? std::clamp(cutoffHz, 20.0f, 20000.0f) : 20000.0f;
    targetCutoffHz_.store(safeCutoff, std::memory_order_relaxed);
}

void BiquadLowPassFilter::reset() noexcept {
    z1_.fill(0.0f);
    z2_.fill(0.0f);
}

void BiquadLowPassFilter::updateCoefficients(float sampleRate, float cutoffHz) noexcept {
    const float maxSafeCutoff = std::max(20.0f, sampleRate * 0.45f);
    const float fc = std::clamp(cutoffHz, 20.0f, maxSafeCutoff);

    // RBJ Audio EQ Cookbook 2nd-order Butterworth LPF (Q = 1 / sqrt(2)).
    const float omega0 = 2.0f * std::numbers::pi_v<float> * fc / sampleRate;
    const float cosW0 = std::cos(omega0);
    const float sinW0 = std::sin(omega0);
    const float alpha = sinW0 * kSqrtHalf;

    const float invA0 = 1.0f / (1.0f + alpha);
    const float oneMinusCos = 1.0f - cosW0;

    b0_ = (0.5f * oneMinusCos) * invA0;
    b1_ = oneMinusCos * invA0;
    b2_ = b0_;
    a1_ = (-2.0f * cosW0) * invA0;
    a2_ = (1.0f - alpha) * invA0;

    activeSampleRate_ = sampleRate;
    activeCutoffHz_ = cutoffHz;
}

void BiquadLowPassFilter::processInterleavedStereo(float* samples, std::uint32_t frameCount) noexcept {
    if (samples == nullptr || frameCount == 0) {
        return;
    }

    const float desiredRate = targetSampleRate_.load(std::memory_order_relaxed);
    const float desiredCutoff = targetCutoffHz_.load(std::memory_order_relaxed);
    if (desiredRate != activeSampleRate_ || desiredCutoff != activeCutoffHz_) {
        updateCoefficients(desiredRate, desiredCutoff);
    }

    const float b0 = b0_;
    const float b1 = b1_;
    const float b2 = b2_;
    const float a1 = a1_;
    const float a2 = a2_;

    float z1L = z1_[0];
    float z2L = z2_[0];
    float z1R = z1_[1];
    float z2R = z2_[1];

    for (std::uint32_t i = 0; i < frameCount; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i) * 2u;

        const float xL = samples[idx];
        const float yL = b0 * xL + z1L;
        z1L = flushDenormal(b1 * xL - a1 * yL + z2L);
        z2L = flushDenormal(b2 * xL - a2 * yL);
        samples[idx] = yL;

        const float xR = samples[idx + 1u];
        const float yR = b0 * xR + z1R;
        z1R = flushDenormal(b1 * xR - a1 * yR + z2R);
        z2R = flushDenormal(b2 * xR - a2 * yR);
        samples[idx + 1u] = yR;
    }

    z1_[0] = z1L;
    z2_[0] = z2L;
    z1_[1] = z1R;
    z2_[1] = z2R;
}

// -----------------------------------------------------------------------------
// SchroederReverbDsp (Zero-Alloc, Lock-Free Real-Time DSP)
// -----------------------------------------------------------------------------

void SchroederReverbDsp::CombFilter::reset() noexcept {
    buffer.fill(0.0f);
    index = 0;
    filterState = 0.0f;
}

float SchroederReverbDsp::CombFilter::process(float input) noexcept {
    const float delayed = buffer[index];
    filterState = flushDenormal(delayed * (1.0f - damping) + filterState * damping);
    buffer[index] = flushDenormal(input + filterState * feedback);
    if (++index >= delaySamples) {
        index = 0;
    }
    return delayed;
}

void SchroederReverbDsp::AllPassFilter::reset() noexcept {
    buffer.fill(0.0f);
    index = 0;
}

float SchroederReverbDsp::AllPassFilter::process(float input) noexcept {
    const float delayed = buffer[index];
    const float w = flushDenormal(input + feedback * delayed);
    buffer[index] = w;
    const float output = delayed - feedback * w;
    if (++index >= delaySamples) {
        index = 0;
    }
    return output;
}

SchroederReverbDsp::SchroederReverbDsp() noexcept {
    configureDelays(48000.0f);
    updateCombFeedbacks(48000.0f, 0.4f);
}

SchroederReverbDsp::SchroederReverbDsp(const SchroederReverbDsp& other) noexcept
    : targetSampleRate_(other.targetSampleRate_.load(std::memory_order_relaxed)),
      targetRt60Sec_(other.targetRt60Sec_.load(std::memory_order_relaxed)),
      targetWetMix_(other.targetWetMix_.load(std::memory_order_relaxed)),
      activeSampleRate_(other.activeSampleRate_),
      activeRt60Sec_(other.activeRt60Sec_),
      combs_(other.combs_),
      allPasses_(other.allPasses_) {}

SchroederReverbDsp& SchroederReverbDsp::operator=(const SchroederReverbDsp& other) noexcept {
    if (this != &other) {
        targetSampleRate_.store(other.targetSampleRate_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        targetRt60Sec_.store(other.targetRt60Sec_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        targetWetMix_.store(other.targetWetMix_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        activeSampleRate_ = other.activeSampleRate_;
        activeRt60Sec_ = other.activeRt60Sec_;
        combs_ = other.combs_;
        allPasses_ = other.allPasses_;
    }
    return *this;
}

SchroederReverbDsp::SchroederReverbDsp(SchroederReverbDsp&& other) noexcept
    : SchroederReverbDsp(static_cast<const SchroederReverbDsp&>(other)) {}

SchroederReverbDsp& SchroederReverbDsp::operator=(SchroederReverbDsp&& other) noexcept {
    return *this = static_cast<const SchroederReverbDsp&>(other);
}

void SchroederReverbDsp::configureDelays(float sampleRate) noexcept {
    auto secToSamples = [sampleRate](float sec) noexcept -> std::size_t {
        const auto raw = static_cast<std::size_t>(std::lround(sec * sampleRate));
        return std::clamp<std::size_t>(raw, 1u, kMaxDelaySamples);
    };

    for (std::size_t i = 0; i < kNumCombs; ++i) {
        combs_[0][i].delaySamples = secToSamples(kCombDelaysSecLeft[i]);
        combs_[1][i].delaySamples = secToSamples(kCombDelaysSecRight[i]);
        combs_[0][i].damping = 0.18f;
        combs_[1][i].damping = 0.18f;
    }

    for (std::size_t i = 0; i < kNumAllPasses; ++i) {
        allPasses_[0][i].delaySamples = secToSamples(kAllPassDelaysSecLeft[i]);
        allPasses_[1][i].delaySamples = secToSamples(kAllPassDelaysSecRight[i]);
        allPasses_[0][i].feedback = 0.5f;
        allPasses_[1][i].feedback = 0.5f;
    }

    activeSampleRate_ = sampleRate;
}

void SchroederReverbDsp::updateCombFeedbacks(float sampleRate, float rt60Sec) noexcept {
    const float safeRt60 = std::clamp(rt60Sec, 0.05f, 20.0f);
    for (std::size_t ch = 0; ch < 2; ++ch) {
        for (std::size_t i = 0; i < kNumCombs; ++i) {
            const float delaySec = static_cast<float>(combs_[ch][i].delaySamples) / sampleRate;
            const float gain = std::pow(10.0f, (-3.0f * delaySec) / safeRt60);
            combs_[ch][i].feedback = std::clamp(gain, 0.0f, 0.985f);
        }
    }
    activeRt60Sec_ = rt60Sec;
}

void SchroederReverbDsp::prepare(float sampleRate) noexcept {
    const float safeRate = std::isfinite(sampleRate) ? std::max(sampleRate, 1000.0f) : 48000.0f;
    targetSampleRate_.store(safeRate, std::memory_order_relaxed);
    configureDelays(safeRate);
    updateCombFeedbacks(safeRate, targetRt60Sec_.load(std::memory_order_relaxed));
    reset();
}

void SchroederReverbDsp::setParameters(float rt60Sec, float wetMix) noexcept {
    const float safeRt60 = std::isfinite(rt60Sec) ? std::clamp(rt60Sec, 0.05f, 20.0f) : 0.4f;
    const float safeWet = std::isfinite(wetMix) ? std::clamp(wetMix, 0.0f, 1.0f) : 0.0f;
    targetRt60Sec_.store(safeRt60, std::memory_order_relaxed);
    targetWetMix_.store(safeWet, std::memory_order_relaxed);
}

void SchroederReverbDsp::reset() noexcept {
    for (std::size_t ch = 0; ch < 2; ++ch) {
        for (auto& comb : combs_[ch]) {
            comb.reset();
        }
        for (auto& allPass : allPasses_[ch]) {
            allPass.reset();
        }
    }
}

void SchroederReverbDsp::processInterleavedStereo(float* samples, std::uint32_t frameCount) noexcept {
    if (samples == nullptr || frameCount == 0) {
        return;
    }

    const float desiredRate = targetSampleRate_.load(std::memory_order_relaxed);
    const float desiredRt60 = targetRt60Sec_.load(std::memory_order_relaxed);
    if (desiredRate != activeSampleRate_) {
        configureDelays(desiredRate);
        updateCombFeedbacks(desiredRate, desiredRt60);
    } else if (desiredRt60 != activeRt60Sec_) {
        updateCombFeedbacks(desiredRate, desiredRt60);
    }

    const float wet = targetWetMix_.load(std::memory_order_relaxed);
    const float dry = 1.0f - wet;

    for (std::uint32_t i = 0; i < frameCount; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i) * 2u;
        const float inL = samples[idx];
        const float inR = samples[idx + 1u];

        // Feed tank with slightly cross-mixed stereo input for natural diffuse tail.
        const float tankInL = 0.8f * inL + 0.2f * inR;
        const float tankInR = 0.8f * inR + 0.2f * inL;

        float combSumL = 0.0f;
        float combSumR = 0.0f;
        for (std::size_t c = 0; c < kNumCombs; ++c) {
            combSumL += combs_[0][c].process(tankInL);
            combSumR += combs_[1][c].process(tankInR);
        }

        float wetL = combSumL * 0.25f;
        float wetR = combSumR * 0.25f;

        for (std::size_t a = 0; a < kNumAllPasses; ++a) {
            wetL = allPasses_[0][a].process(wetL);
            wetR = allPasses_[1][a].process(wetR);
        }

        samples[idx] = dry * inL + wet * wetL;
        samples[idx + 1u] = dry * inR + wet * wetR;
    }
}

} // namespace lumora
