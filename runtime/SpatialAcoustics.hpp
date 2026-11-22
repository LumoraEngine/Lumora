#pragma once

#include <glm/vec3.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>

namespace lumora {

/// Material classification controlling acoustic absorption, transmission, and low-pass muffling.
enum class AcousticMaterialKind : std::uint8_t {
    Concrete,
    Wood,
    Glass,
    Metal,
    Foliage,
    Air,
};

/// Surface & transmission properties of an acoustic obstacle.
struct AcousticMaterial {
    float absorption = 0.05f;
    float transmission = 1.0f;
    float lpfCutoffMinHz = 20000.0f;
};

/// Returns default acoustic properties for a built-in material kind.
[[nodiscard]] constexpr AcousticMaterial defaultAcousticMaterial(AcousticMaterialKind kind) noexcept {
    switch (kind) {
        case AcousticMaterialKind::Concrete:
            return AcousticMaterial{0.05f, 0.05f, 800.0f};
        case AcousticMaterialKind::Wood:
            return AcousticMaterial{0.15f, 0.22f, 2200.0f};
        case AcousticMaterialKind::Glass:
            return AcousticMaterial{0.08f, 0.25f, 2800.0f};
        case AcousticMaterialKind::Metal:
            return AcousticMaterial{0.03f, 0.10f, 1200.0f};
        case AcousticMaterialKind::Foliage:
            return AcousticMaterial{0.60f, 0.65f, 7500.0f};
        case AcousticMaterialKind::Air:
        default:
            return AcousticMaterial{0.00f, 1.00f, 20000.0f};
    }
}

/// Raycast hit returned by an acoustic geometry query callback.
struct AcousticHit {
    float distance = 0.0f;
    glm::vec3 point{0.0f};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    AcousticMaterial material{};
};

/// Callback querying scene geometry along a unit-direction ray up to `maxDistance` metres.
using AcousticRaycastFn =
    std::function<std::optional<AcousticHit>(const glm::vec3& origin, const glm::vec3& dir, float maxDistance)>;

/// Axis-aligned volume overriding estimated environment reverb parameters smoothly.
struct AcousticZone {
    glm::vec3 boundsMin{0.0f};
    glm::vec3 boundsMax{0.0f};
    float rt60Sec = 1.2f;
    float wetMix = 0.35f;
    std::string name;
};

/// Combined per-source acoustic parameters ready to drive voice DSP and spatialization.
struct AcousticSourceResult {
    float occlusion = 0.0f;
    float lpfCutoffHz = 20000.0f;
    float attenuationGain = 1.0f;
    float reverbWet = 0.0f;
    float reverbRt60Sec = 0.4f;
    float dopplerPitch = 1.0f;
};

/// Output of direct + offset ray occlusion evaluation between a source and listener.
struct OcclusionResult {
    float occlusion = 0.0f;
    float transmissionGain = 1.0f;
    float lpfCutoffHz = 20000.0f;
};

/// Output of Fibonacci-sphere environment reverberation estimation.
struct EnvironmentReverbResult {
    float rt60Sec = 0.4f;
    float wetMix = 0.0f;
    float meanFreePathMeters = 0.0f;
    float enclosureFactor = 0.0f;
    float meanAbsorption = 0.0f;
};

/// Tuning parameters for the spatial acoustics solver.
struct SpatialAcousticsConfig {
    float occlusionProbeRadius = 0.25f;
    float minLpfCutoffHz = 500.0f;
    float maxLpfCutoffHz = 20000.0f;
    std::uint32_t reverbRayCount = 32;
    float reverbMaxRayDistance = 150.0f;
    float zoneBlendDistance = 2.0f;
    float minRt60Sec = 0.15f;
    float maxRt60Sec = 8.0f;
    float maxReverbWet = 0.75f;
    float dopplerSpeedOfSound = 343.0f;
    float dopplerScale = 1.0f;
};

/// Computes ray-traced acoustic occlusion, environment reverberation, and Doppler shift.
class SpatialAcousticsSolver {
public:
    SpatialAcousticsSolver() = default;
    explicit SpatialAcousticsSolver(const SpatialAcousticsConfig& config) noexcept;

    [[nodiscard]] const SpatialAcousticsConfig& config() const noexcept { return config_; }
    void setConfig(const SpatialAcousticsConfig& config) noexcept { config_ = config; }

    /// Casts a direct ray plus perpendicular offset rays around the source-listener line to compute
    /// occlusion [0..1], material transmission gain, and muffled low-pass cutoff [500..20000 Hz].
    [[nodiscard]] OcclusionResult evaluateOcclusion(
        const glm::vec3& sourcePos,
        const glm::vec3& listenerPos,
        const AcousticRaycastFn& raycast) const;

    /// Casts Fibonacci-sphere rays from `listenerPos` to estimate mean free path, enclosure factor,
    /// and Sabine/Eyring RT60 + wet mix, smoothly overridden inside `AcousticZone` AABBs.
    [[nodiscard]] EnvironmentReverbResult estimateEnvironmentReverb(
        const glm::vec3& listenerPos,
        const AcousticRaycastFn& raycast,
        std::span<const AcousticZone> zones = {}) const;

    /// Computes Doppler pitch multiplier safely clamped to [0.5, 2.0].
    [[nodiscard]] static float computeDopplerPitch(
        const glm::vec3& sourcePos,
        const glm::vec3& sourceVel,
        const glm::vec3& listenerPos,
        const glm::vec3& listenerVel,
        float speedOfSound = 343.0f,
        float scale = 1.0f) noexcept;

    /// Evaluates occlusion, environment reverb, distance attenuation, and Doppler for a source.
    [[nodiscard]] AcousticSourceResult evaluateSource(
        const glm::vec3& sourcePos,
        const glm::vec3& sourceVel,
        const glm::vec3& listenerPos,
        const glm::vec3& listenerVel,
        const AcousticRaycastFn& raycast,
        std::span<const AcousticZone> zones = {},
        float minDistance = 1.0f,
        float maxDistance = 50.0f) const;

    /// Overload reusing a precomputed environment reverb estimate for the current listener frame.
    [[nodiscard]] AcousticSourceResult evaluateSource(
        const glm::vec3& sourcePos,
        const glm::vec3& sourceVel,
        const glm::vec3& listenerPos,
        const glm::vec3& listenerVel,
        const AcousticRaycastFn& raycast,
        const EnvironmentReverbResult& cachedEnvironment,
        float minDistance = 1.0f,
        float maxDistance = 50.0f) const;

    /// Convenience overload for stationary sources and listeners.
    [[nodiscard]] AcousticSourceResult evaluateSource(
        const glm::vec3& sourcePos,
        const glm::vec3& listenerPos,
        const AcousticRaycastFn& raycast,
        std::span<const AcousticZone> zones = {},
        float minDistance = 1.0f,
        float maxDistance = 50.0f) const;

private:
    SpatialAcousticsConfig config_{};
};

/// Zero-allocation RBJ Audio EQ Cookbook 2nd-order Butterworth stereo low-pass filter.
/// Safe for real-time audio callbacks (`MA_THREAD_PRIORITY_REALTIME`): no heap allocations or locks.
class BiquadLowPassFilter {
public:
    BiquadLowPassFilter() noexcept = default;
    BiquadLowPassFilter(const BiquadLowPassFilter& other) noexcept;
    BiquadLowPassFilter& operator=(const BiquadLowPassFilter& other) noexcept;
    BiquadLowPassFilter(BiquadLowPassFilter&& other) noexcept;
    BiquadLowPassFilter& operator=(BiquadLowPassFilter&& other) noexcept;

    void prepare(float sampleRate) noexcept;
    void setCutoff(float cutoffHz) noexcept;
    void reset() noexcept;

    void processInterleavedStereo(float* samples, std::uint32_t frameCount) noexcept;

    [[nodiscard]] float sampleRate() const noexcept {
        return targetSampleRate_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] float cutoffHz() const noexcept {
        return targetCutoffHz_.load(std::memory_order_relaxed);
    }

private:
    void updateCoefficients(float sampleRate, float cutoffHz) noexcept;

    std::atomic<float> targetSampleRate_{48000.0f};
    std::atomic<float> targetCutoffHz_{20000.0f};

    float activeSampleRate_ = 48000.0f;
    float activeCutoffHz_ = 20000.0f;
    float b0_ = 1.0f;
    float b1_ = 0.0f;
    float b2_ = 0.0f;
    float a1_ = 0.0f;
    float a2_ = 0.0f;
    std::array<float, 2> z1_{};
    std::array<float, 2> z2_{};
};

/// Zero-allocation Schroeder stereo reverberator (4 parallel comb filters + 2 series all-pass
/// filters per channel) using fixed pre-allocated `std::array` delay buffers.
/// Safe for real-time audio callbacks (`MA_THREAD_PRIORITY_REALTIME`): no heap allocations or locks.
class SchroederReverbDsp {
public:
    static constexpr std::size_t kNumCombs = 4;
    static constexpr std::size_t kNumAllPasses = 2;
    static constexpr std::size_t kMaxDelaySamples = 4800;

    SchroederReverbDsp() noexcept;
    SchroederReverbDsp(const SchroederReverbDsp& other) noexcept;
    SchroederReverbDsp& operator=(const SchroederReverbDsp& other) noexcept;
    SchroederReverbDsp(SchroederReverbDsp&& other) noexcept;
    SchroederReverbDsp& operator=(SchroederReverbDsp&& other) noexcept;

    void prepare(float sampleRate) noexcept;
    void setParameters(float rt60Sec, float wetMix) noexcept;
    void reset() noexcept;

    void processInterleavedStereo(float* samples, std::uint32_t frameCount) noexcept;

    [[nodiscard]] float sampleRate() const noexcept {
        return targetSampleRate_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] float rt60Sec() const noexcept {
        return targetRt60Sec_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] float wetMix() const noexcept {
        return targetWetMix_.load(std::memory_order_relaxed);
    }

private:
    struct CombFilter {
        std::array<float, kMaxDelaySamples> buffer{};
        std::size_t delaySamples = 1;
        std::size_t index = 0;
        float feedback = 0.7f;
        float damping = 0.2f;
        float filterState = 0.0f;

        void reset() noexcept;
        float process(float input) noexcept;
    };

    struct AllPassFilter {
        std::array<float, kMaxDelaySamples> buffer{};
        std::size_t delaySamples = 1;
        std::size_t index = 0;
        float feedback = 0.5f;

        void reset() noexcept;
        float process(float input) noexcept;
    };

    void configureDelays(float sampleRate) noexcept;
    void updateCombFeedbacks(float sampleRate, float rt60Sec) noexcept;

    std::atomic<float> targetSampleRate_{48000.0f};
    std::atomic<float> targetRt60Sec_{0.4f};
    std::atomic<float> targetWetMix_{0.0f};

    float activeSampleRate_ = 48000.0f;
    float activeRt60Sec_ = 0.4f;

    std::array<std::array<CombFilter, kNumCombs>, 2> combs_{};
    std::array<std::array<AllPassFilter, kNumAllPasses>, 2> allPasses_{};
};

} // namespace lumora
