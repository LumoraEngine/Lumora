#include "runtime/SpatialAcoustics.hpp"

#include <gtest/gtest.h>
#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <vector>

using namespace lumora;

namespace {

struct AabbBox {
    glm::vec3 min{0.0f};
    glm::vec3 max{0.0f};
    AcousticMaterial material{};
};

std::optional<AcousticHit> intersectRayBox(
    const glm::vec3& origin,
    const glm::vec3& dir,
    float maxDistance,
    const AabbBox& box) {
    float tMin = 0.0f;
    float tMax = maxDistance;
    glm::vec3 hitNormal{0.0f};

    for (int axis = 0; axis < 3; ++axis) {
        const float d = dir[axis];
        const float o = origin[axis];
        if (std::abs(d) < 1e-7f) {
            if (o < box.min[axis] || o > box.max[axis]) {
                return std::nullopt;
            }
        } else {
            const float invD = 1.0f / d;
            float t1 = (box.min[axis] - o) * invD;
            float t2 = (box.max[axis] - o) * invD;
            glm::vec3 n{0.0f};
            n[axis] = (invD >= 0.0f) ? -1.0f : 1.0f;
            if (t1 > t2) {
                std::swap(t1, t2);
                n = -n;
            }
            if (t1 > tMin) {
                tMin = t1;
                hitNormal = n;
            }
            tMax = std::min(tMax, t2);
            if (tMin > tMax) {
                return std::nullopt;
            }
        }
    }

    if (tMin < 0.0f || tMin > maxDistance) {
        return std::nullopt;
    }

    return AcousticHit{tMin, origin + dir * tMin, hitNormal, box.material};
}

AcousticRaycastFn makeBoxRaycast(std::vector<AabbBox> boxes) {
    return [boxes = std::move(boxes)](const glm::vec3& origin, const glm::vec3& dir, float maxDistance)
        -> std::optional<AcousticHit> {
        std::optional<AcousticHit> closest;
        for (const AabbBox& box : boxes) {
            if (const auto hit = intersectRayBox(origin, dir, maxDistance, box)) {
                if (!closest.has_value() || hit->distance < closest->distance) {
                    closest = hit;
                }
            }
        }
        return closest;
    };
}

AcousticRaycastFn makeInteriorRoomRaycast(const glm::vec3& halfExtents, const AcousticMaterial& mat) {
    return [halfExtents, mat](const glm::vec3& origin, const glm::vec3& dir, float maxDistance)
        -> std::optional<AcousticHit> {
        float bestT = maxDistance + 1.0f;
        glm::vec3 bestNormal{0.0f};
        for (int axis = 0; axis < 3; ++axis) {
            const float d = dir[axis];
            if (std::abs(d) < 1e-7f) {
                continue;
            }
            const float boundary = (d > 0.0f) ? halfExtents[axis] : -halfExtents[axis];
            const float t = (boundary - origin[axis]) / d;
            if (t >= 0.0f && t < bestT) {
                bestT = t;
                bestNormal = glm::vec3(0.0f);
                bestNormal[axis] = (d > 0.0f) ? -1.0f : 1.0f;
            }
        }
        if (bestT <= maxDistance) {
            return AcousticHit{bestT, origin + dir * bestT, bestNormal, mat};
        }
        return std::nullopt;
    };
}

float computeChannelPeak(const std::vector<float>& interleaved, std::size_t channel, std::uint32_t startFrame,
                         std::uint32_t endFrame) {
    float peak = 0.0f;
    for (std::uint32_t f = startFrame; f < endFrame; ++f) {
        peak = std::max(peak, std::abs(interleaved[static_cast<std::size_t>(f) * 2u + channel]));
    }
    return peak;
}

float computeStereoRms(const std::vector<float>& interleaved, std::uint32_t startFrame, std::uint32_t endFrame) {
    if (endFrame <= startFrame) {
        return 0.0f;
    }
    double sumSq = 0.0;
    const std::size_t count = static_cast<std::size_t>(endFrame - startFrame) * 2u;
    for (std::uint32_t f = startFrame; f < endFrame; ++f) {
        const std::size_t idx = static_cast<std::size_t>(f) * 2u;
        const double l = interleaved[idx];
        const double r = interleaved[idx + 1u];
        sumSq += l * l + r * r;
    }
    return static_cast<float>(std::sqrt(sumSq / static_cast<double>(count)));
}

} // namespace

TEST(SpatialAcoustics, DefaultMaterialsMatchAcousticBehavior) {
    const AcousticMaterial air = defaultAcousticMaterial(AcousticMaterialKind::Air);
    const AcousticMaterial concrete = defaultAcousticMaterial(AcousticMaterialKind::Concrete);
    const AcousticMaterial wood = defaultAcousticMaterial(AcousticMaterialKind::Wood);
    const AcousticMaterial glass = defaultAcousticMaterial(AcousticMaterialKind::Glass);
    const AcousticMaterial metal = defaultAcousticMaterial(AcousticMaterialKind::Metal);
    const AcousticMaterial foliage = defaultAcousticMaterial(AcousticMaterialKind::Foliage);

    EXPECT_FLOAT_EQ(air.absorption, 0.0f);
    EXPECT_FLOAT_EQ(air.transmission, 1.0f);
    EXPECT_FLOAT_EQ(air.lpfCutoffMinHz, 20000.0f);

    EXPECT_LT(concrete.transmission, 0.2f);
    EXPECT_LT(concrete.lpfCutoffMinHz, 4000.0f);
    EXPECT_LT(concrete.absorption, foliage.absorption);
    EXPECT_LT(metal.transmission, glass.transmission);
    EXPECT_LT(wood.lpfCutoffMinHz, foliage.lpfCutoffMinHz);
}

TEST(SpatialAcoustics, ClearLineOfSightHasZeroOcclusionAndFullBandwidth) {
    const SpatialAcousticsSolver solver;
    const auto openRaycast = [](const glm::vec3&, const glm::vec3&, float) -> std::optional<AcousticHit> {
        return std::nullopt;
    };

    const OcclusionResult clearRes =
        solver.evaluateOcclusion(glm::vec3(0.0f, 1.5f, -5.0f), glm::vec3(0.0f, 1.5f, 5.0f), openRaycast);
    EXPECT_NEAR(clearRes.occlusion, 0.0f, 1e-4f);
    EXPECT_NEAR(clearRes.transmissionGain, 1.0f, 1e-4f);
    EXPECT_NEAR(clearRes.lpfCutoffHz, 20000.0f, 1.0f);

    // A wall placed behind the listener must not occlude the direct source-listener path.
    const auto wallBehindRaycast = makeBoxRaycast({
        {{-5.0f, -5.0f, 8.0f}, {5.0f, 5.0f, 8.5f}, defaultAcousticMaterial(AcousticMaterialKind::Concrete)},
    });
    const OcclusionResult behindRes =
        solver.evaluateOcclusion(glm::vec3(0.0f, 0.0f, -4.0f), glm::vec3(0.0f, 0.0f, 4.0f), wallBehindRaycast);
    EXPECT_NEAR(behindRes.occlusion, 0.0f, 1e-4f);
    // Also verify zero-distance and null raycast safety.
    const OcclusionResult samePosRes =
        solver.evaluateOcclusion(glm::vec3(1.0f), glm::vec3(1.0f), wallBehindRaycast);
    EXPECT_FLOAT_EQ(samePosRes.occlusion, 0.0f);
    EXPECT_FLOAT_EQ(samePosRes.transmissionGain, 1.0f);

    const OcclusionResult nullRayRes =
        solver.evaluateOcclusion(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 5.0f), AcousticRaycastFn{});
    EXPECT_FLOAT_EQ(nullRayRes.occlusion, 0.0f);
    EXPECT_FLOAT_EQ(nullRayRes.lpfCutoffHz, 20000.0f);
}

TEST(SpatialAcoustics, ConcreteWallCausesHeavyOcclusionAndLowPassMuffling) {
    const SpatialAcousticsSolver solver;
    const auto wallRaycast = makeBoxRaycast({
        {{-5.0f, -3.0f, -0.2f}, {5.0f, 3.0f, 0.2f}, defaultAcousticMaterial(AcousticMaterialKind::Concrete)},
    });

    const OcclusionResult wallRes =
        solver.evaluateOcclusion(glm::vec3(0.0f, 0.0f, -4.0f), glm::vec3(0.0f, 0.0f, 4.0f), wallRaycast);
    EXPECT_GT(wallRes.occlusion, 0.7f);
    EXPECT_LT(wallRes.lpfCutoffHz, 4000.0f);
    EXPECT_LT(wallRes.transmissionGain, 0.25f);

    // Partial edge occlusion around the corner of a concrete wall yields intermediate values.
    const auto halfWallRaycast = makeBoxRaycast({
        {{-5.0f, -3.0f, -0.2f}, {0.0f, 3.0f, 0.2f}, defaultAcousticMaterial(AcousticMaterialKind::Concrete)},
    });
    const OcclusionResult edgeRes =
        solver.evaluateOcclusion(glm::vec3(0.0f, 0.0f, -4.0f), glm::vec3(0.0f, 0.0f, 4.0f), halfWallRaycast);
    EXPECT_GT(edgeRes.occlusion, 0.1f);
    EXPECT_LT(edgeRes.occlusion, wallRes.occlusion);
    EXPECT_GT(edgeRes.lpfCutoffHz, wallRes.lpfCutoffHz);
    EXPECT_LT(edgeRes.lpfCutoffHz, 19500.0f);
}

TEST(SpatialAcoustics, EnvironmentReverbDistinguishesSmallRoomHugeHallAndOpenSky) {
    const SpatialAcousticsSolver solver;
    const AcousticMaterial concrete = defaultAcousticMaterial(AcousticMaterialKind::Concrete);
    const AcousticMaterial foliage = defaultAcousticMaterial(AcousticMaterialKind::Foliage);

    const auto openSkyRaycast = [](const glm::vec3&, const glm::vec3&, float) -> std::optional<AcousticHit> {
        return std::nullopt;
    };
    const auto groundOnlyRaycast = [](const glm::vec3& origin, const glm::vec3& dir, float maxDistance)
        -> std::optional<AcousticHit> {
        if (dir.y < -1e-4f && origin.y > 0.0f) {
            const float t = -origin.y / dir.y;
            if (t >= 0.0f && t <= maxDistance) {
                return AcousticHit{t, origin + dir * t, glm::vec3(0.0f, 1.0f, 0.0f),
                                   defaultAcousticMaterial(AcousticMaterialKind::Wood)};
            }
        }
        return std::nullopt;
    };

    const auto smallRoomRaycast = makeInteriorRoomRaycast(glm::vec3(2.0f, 1.5f, 2.0f), concrete);
    const auto hugeHallRaycast = makeInteriorRoomRaycast(glm::vec3(20.0f, 10.0f, 20.0f), concrete);
    const auto foliageRoomRaycast = makeInteriorRoomRaycast(glm::vec3(2.0f, 1.5f, 2.0f), foliage);

    const glm::vec3 listenerPos{0.0f, 1.0f, 0.0f};
    const EnvironmentReverbResult sky = solver.estimateEnvironmentReverb(glm::vec3(0.0f, 1.7f, 0.0f), openSkyRaycast);
    const EnvironmentReverbResult outdoorGround =
        solver.estimateEnvironmentReverb(glm::vec3(0.0f, 1.7f, 0.0f), groundOnlyRaycast);
    const EnvironmentReverbResult smallRoom = solver.estimateEnvironmentReverb(listenerPos, smallRoomRaycast);
    const EnvironmentReverbResult hugeHall = solver.estimateEnvironmentReverb(listenerPos, hugeHallRaycast);
    const EnvironmentReverbResult foliageRoom = solver.estimateEnvironmentReverb(listenerPos, foliageRoomRaycast);

    EXPECT_NEAR(sky.wetMix, 0.0f, 1e-4f);
    EXPECT_FLOAT_EQ(sky.enclosureFactor, 0.0f);

    EXPECT_LT(outdoorGround.wetMix, smallRoom.wetMix);
    EXPECT_LT(outdoorGround.rt60Sec, smallRoom.rt60Sec);

    EXPECT_GT(smallRoom.enclosureFactor, 0.99f);
    EXPECT_GT(hugeHall.enclosureFactor, 0.99f);
    EXPECT_GT(smallRoom.wetMix, sky.wetMix + 0.1f);
    EXPECT_GT(smallRoom.rt60Sec, sky.rt60Sec + 0.2f);

    EXPECT_GT(hugeHall.rt60Sec, smallRoom.rt60Sec * 2.0f);
    EXPECT_GT(hugeHall.wetMix, smallRoom.wetMix + 0.15f);

    // Absorptive foliage walls produce shorter RT60 and lower wet mix than concrete walls of the same size.
    EXPECT_LT(foliageRoom.rt60Sec, smallRoom.rt60Sec);
    EXPECT_LT(foliageRoom.wetMix, smallRoom.wetMix);
}

TEST(SpatialAcoustics, AcousticZoneOverridesEnvironmentSmoothly) {
    const SpatialAcousticsSolver solver;
    const auto openSkyRaycast = [](const glm::vec3&, const glm::vec3&, float) -> std::optional<AcousticHit> {
        return std::nullopt;
    };

    const std::vector<AcousticZone> zones = {
        AcousticZone{glm::vec3(-4.0f, -4.0f, -4.0f), glm::vec3(4.0f, 4.0f, 4.0f), 2.8f, 0.60f, "Cathedral"},
    };

    const EnvironmentReverbResult inside =
        solver.estimateEnvironmentReverb(glm::vec3(0.0f, 0.0f, 0.0f), openSkyRaycast, zones);
    EXPECT_NEAR(inside.rt60Sec, 2.8f, 1e-3f);
    EXPECT_NEAR(inside.wetMix, 0.60f, 1e-3f);

    // Halfway through the 2.0 m blend margin (at x = 5.0 -> 1.0 m outside boundsMax.x = 4.0).
    const EnvironmentReverbResult edgeBlend =
        solver.estimateEnvironmentReverb(glm::vec3(5.0f, 0.0f, 0.0f), openSkyRaycast, zones);
    EXPECT_GT(edgeBlend.rt60Sec, 0.2f);
    EXPECT_LT(edgeBlend.rt60Sec, inside.rt60Sec);
    EXPECT_GT(edgeBlend.wetMix, 0.05f);
    EXPECT_LT(edgeBlend.wetMix, inside.wetMix);

    // Well outside the blend margin (x = 10.0 -> 6.0 m outside).
    const EnvironmentReverbResult outside =
        solver.estimateEnvironmentReverb(glm::vec3(10.0f, 0.0f, 0.0f), openSkyRaycast, zones);
    EXPECT_NEAR(outside.wetMix, 0.0f, 1e-4f);

    // Nested smaller zone inside larger zone takes priority when listener is inside both.
    const std::vector<AcousticZone> nestedZones = {
        AcousticZone{glm::vec3(-20.0f), glm::vec3(20.0f), 3.0f, 0.55f, "OuterHall"},
        AcousticZone{glm::vec3(-1.5f), glm::vec3(1.5f), 0.5f, 0.20f, "InnerBooth"},
    };
    const EnvironmentReverbResult innerBooth =
        solver.estimateEnvironmentReverb(glm::vec3(0.0f), openSkyRaycast, nestedZones);
    EXPECT_NEAR(innerBooth.rt60Sec, 0.5f, 1e-3f);
    EXPECT_NEAR(innerBooth.wetMix, 0.20f, 1e-3f);

    // 1.0 m outside InnerBooth (x = 2.5) while still inside OuterHall -> smooth 50% blend.
    const EnvironmentReverbResult boothTransition =
        solver.estimateEnvironmentReverb(glm::vec3(2.5f, 0.0f, 0.0f), openSkyRaycast, nestedZones);
    EXPECT_NEAR(boothTransition.rt60Sec, 1.75f, 0.02f);
    EXPECT_NEAR(boothTransition.wetMix, 0.375f, 0.02f);

    // Beyond blend margin of InnerBooth (x = 5.0) -> fully governed by OuterHall.
    const EnvironmentReverbResult outerHallOnly =
        solver.estimateEnvironmentReverb(glm::vec3(5.0f, 0.0f, 0.0f), openSkyRaycast, nestedZones);
    EXPECT_NEAR(outerHallOnly.rt60Sec, 3.0f, 1e-3f);
    EXPECT_NEAR(outerHallOnly.wetMix, 0.55f, 1e-3f);
}

TEST(SpatialAcoustics, DopplerPitchApproachingRecedingStationaryAndClamped) {
    const glm::vec3 listenerPos{0.0f, 0.0f, 0.0f};
    const glm::vec3 sourcePos{20.0f, 0.0f, 0.0f};

    const float stationary = SpatialAcousticsSolver::computeDopplerPitch(
        sourcePos, glm::vec3(0.0f), listenerPos, glm::vec3(0.0f));
    EXPECT_FLOAT_EQ(stationary, 1.0f);

    // Source approaching along -X.
    const float approachingSource = SpatialAcousticsSolver::computeDopplerPitch(
        sourcePos, glm::vec3(-34.3f, 0.0f, 0.0f), listenerPos, glm::vec3(0.0f));
    EXPECT_GT(approachingSource, 1.0f);
    EXPECT_NEAR(approachingSource, 343.0f / (343.0f - 34.3f), 1e-4f);

    // Source receding along +X.
    const float recedingSource = SpatialAcousticsSolver::computeDopplerPitch(
        sourcePos, glm::vec3(34.3f, 0.0f, 0.0f), listenerPos, glm::vec3(0.0f));
    EXPECT_LT(recedingSource, 1.0f);
    EXPECT_NEAR(recedingSource, 343.0f / (343.0f + 34.3f), 1e-4f);

    // Listener approaching stationary source along +X.
    const float approachingListener = SpatialAcousticsSolver::computeDopplerPitch(
        sourcePos, glm::vec3(0.0f), listenerPos, glm::vec3(34.3f, 0.0f, 0.0f));
    EXPECT_GT(approachingListener, 1.0f);

    // Both moving together at identical velocity -> zero relative shift.
    const float tandem = SpatialAcousticsSolver::computeDopplerPitch(
        sourcePos, glm::vec3(25.0f, 0.0f, 0.0f), listenerPos, glm::vec3(25.0f, 0.0f, 0.0f));
    EXPECT_FLOAT_EQ(tandem, 1.0f);

    // Supersonic extreme speeds must clamp cleanly within [0.5, 2.0].
    const float extremeHigh = SpatialAcousticsSolver::computeDopplerPitch(
        sourcePos, glm::vec3(-1000.0f, 0.0f, 0.0f), listenerPos, glm::vec3(1000.0f, 0.0f, 0.0f));
    const float extremeLow = SpatialAcousticsSolver::computeDopplerPitch(
        sourcePos, glm::vec3(1000.0f, 0.0f, 0.0f), listenerPos, glm::vec3(-1000.0f, 0.0f, 0.0f));
    EXPECT_FLOAT_EQ(extremeHigh, 2.0f);
    EXPECT_FLOAT_EQ(extremeLow, 0.5f);
}

TEST(SpatialAcoustics, EvaluateSourceCombinesOcclusionReverbAndDoppler) {
    const SpatialAcousticsSolver solver;
    const auto roomRaycast = makeBoxRaycast({
        {{-5.0f, -3.0f, -0.2f}, {5.0f, 3.0f, 0.2f}, defaultAcousticMaterial(AcousticMaterialKind::Concrete)},
    });
    const std::vector<AcousticZone> zones = {
        AcousticZone{glm::vec3(-10.0f), glm::vec3(10.0f), 1.6f, 0.42f, "Hall"},
    };

    const AcousticSourceResult occludedApproaching = solver.evaluateSource(
        glm::vec3(0.0f, 0.0f, -4.0f), glm::vec3(0.0f, 0.0f, 20.0f),
        glm::vec3(0.0f, 0.0f, 4.0f), glm::vec3(0.0f),
        roomRaycast, zones);

    EXPECT_GT(occludedApproaching.occlusion, 0.7f);
    EXPECT_LT(occludedApproaching.lpfCutoffHz, 4000.0f);
    EXPECT_LT(occludedApproaching.attenuationGain, 0.1f);
    EXPECT_NEAR(occludedApproaching.reverbRt60Sec, 1.6f, 1e-3f);
    EXPECT_NEAR(occludedApproaching.reverbWet, 0.42f, 1e-3f);
    EXPECT_GT(occludedApproaching.dopplerPitch, 1.0f);
}

TEST(SpatialAcoustics, BiquadLowPassFilterPreserves120HzAndAttenuates8000Hz) {
    constexpr float kSampleRate = 48000.0f;
    constexpr std::uint32_t kFrames = 4096;

    BiquadLowPassFilter lpf;
    lpf.prepare(kSampleRate);
    lpf.setCutoff(500.0f);

    auto makeStereoSine = [&](float freqL, float freqR) {
        std::vector<float> buf(static_cast<std::size_t>(kFrames) * 2u, 0.0f);
        for (std::uint32_t i = 0; i < kFrames; ++i) {
            const float t = static_cast<float>(i) / kSampleRate;
            buf[static_cast<std::size_t>(i) * 2u] = std::sin(2.0f * std::numbers::pi_v<float> * freqL * t);
            buf[static_cast<std::size_t>(i) * 2u + 1u] = std::sin(2.0f * std::numbers::pi_v<float> * freqR * t);
        }
        return buf;
    };

    // Left carries 120 Hz, Right carries 8000 Hz simultaneously to also verify independent stereo channels.
    std::vector<float> dualTone = makeStereoSine(120.0f, 8000.0f);
    lpf.processInterleavedStereo(dualTone.data(), kFrames);

    // Measure steady-state peak after filter startup transient (frames 1024..4096).
    const float lowFreqPeak = computeChannelPeak(dualTone, 0u, 1024u, kFrames);
    const float highFreqPeak = computeChannelPeak(dualTone, 1u, 1024u, kFrames);

    EXPECT_GT(lowFreqPeak, 0.8f);
    EXPECT_LT(highFreqPeak, 0.15f);

    // Null / zero frame safety.
    lpf.processInterleavedStereo(nullptr, 128u);
    lpf.processInterleavedStereo(dualTone.data(), 0u);
}

TEST(SpatialAcoustics, SchroederReverbProducesDecayingTailAfterImpulse) {
    constexpr float kSampleRate = 48000.0f;
    constexpr std::uint32_t kFrames = 48000;  // 1.0 second

    SchroederReverbDsp reverb;
    reverb.prepare(kSampleRate);
    reverb.setParameters(1.2f, 0.5f);

    std::vector<float> buffer(static_cast<std::size_t>(kFrames) * 2u, 0.0f);
    buffer[0] = 1.0f;
    buffer[1] = 1.0f;

    reverb.processInterleavedStereo(buffer.data(), kFrames);

    // Dry impulse arrives immediately at frame 0 scaled by (1 - wetMix) = 0.5.
    EXPECT_NEAR(buffer[0], 0.5f, 1e-4f);
    EXPECT_NEAR(buffer[1], 0.5f, 1e-4f);

    for (float sample : buffer) {
        ASSERT_TRUE(std::isfinite(sample));
    }

    const float rmsEarly = computeStereoRms(buffer, 1000u, 12000u);
    const float rmsMid = computeStereoRms(buffer, 12000u, 24000u);
    const float rmsLate = computeStereoRms(buffer, 24000u, 36000u);
    const float rmsEnd = computeStereoRms(buffer, 36000u, 48000u);
    const float peakEarly = computeChannelPeak(buffer, 0u, 1000u, 12000u);

    EXPECT_GT(peakEarly, 0.03f) << "reverb tank should emit clear early reflections after delay time";
    EXPECT_GT(rmsEarly, 0.002f) << "reverb tank should emit significant early tail energy";
    EXPECT_GT(rmsEarly, rmsMid);
    EXPECT_GT(rmsMid, rmsLate);
    EXPECT_GT(rmsLate, rmsEnd);
    EXPECT_GT(rmsEnd, 1e-5f);

    // Shorter RT60 must decay faster than longer RT60.
    SchroederReverbDsp shortReverb;
    shortReverb.prepare(kSampleRate);
    shortReverb.setParameters(0.35f, 0.5f);
    std::vector<float> shortBuf(static_cast<std::size_t>(kFrames) * 2u, 0.0f);
    shortBuf[0] = 1.0f;
    shortBuf[1] = 1.0f;
    shortReverb.processInterleavedStereo(shortBuf.data(), kFrames);
    const float shortLateRms = computeStereoRms(shortBuf, 24000u, 36000u);
    EXPECT_LT(shortLateRms, rmsLate * 0.25f);

    // High sample rate (96 kHz) remains bounded, finite, and produces decaying tail without overflow.
    SchroederReverbDsp highRateReverb;
    highRateReverb.prepare(96000.0f);
    highRateReverb.setParameters(1.0f, 0.5f);
    std::vector<float> highRateBuf(96000u * 2u, 0.0f);
    highRateBuf[0] = 1.0f;
    highRateBuf[1] = 1.0f;
    highRateReverb.processInterleavedStereo(highRateBuf.data(), 96000u);
    for (float sample : highRateBuf) {
        ASSERT_TRUE(std::isfinite(sample));
    }
    EXPECT_GT(computeStereoRms(highRateBuf, 2000u, 24000u), computeStereoRms(highRateBuf, 48000u, 96000u));

    // Zero wetMix preserves dry signal identically.
    SchroederReverbDsp dryReverb;
    dryReverb.prepare(kSampleRate);
    dryReverb.setParameters(1.2f, 0.0f);
    std::vector<float> dryBuf = {0.25f, -0.75f, 0.5f, -0.5f};
    const std::vector<float> expected = dryBuf;
    dryReverb.processInterleavedStereo(dryBuf.data(), 2u);
    EXPECT_FLOAT_EQ(dryBuf[0], expected[0]);
    EXPECT_FLOAT_EQ(dryBuf[1], expected[1]);
    EXPECT_FLOAT_EQ(dryBuf[2], expected[2]);
    EXPECT_FLOAT_EQ(dryBuf[3], expected[3]);
}
