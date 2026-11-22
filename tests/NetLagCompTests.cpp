#include "runtime/net/LagCompensation.hpp"

#include <gtest/gtest.h>

#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/vec3.hpp>

#include <cmath>
#include <limits>
#include <vector>

using namespace lumora::net;

namespace {

std::vector<HitCapsule> makeHumanoidHitboxes() {
    return {
        // Torso / body capsule along local +Y
        HitCapsule{glm::vec3{0.0f, 0.25f, 0.0f}, glm::vec3{0.0f, 1.35f, 0.0f}, 0.25f, HitboxPart::Body},
        // Head sphere capsule (localA == localB)
        HitCapsule{glm::vec3{0.0f, 1.75f, 0.0f}, glm::vec3{0.0f, 1.75f, 0.0f}, 0.16f, HitboxPart::Head},
        // Right limb capsule
        HitCapsule{glm::vec3{0.40f, 0.60f, 0.0f}, glm::vec3{0.40f, 1.25f, 0.0f}, 0.10f, HitboxPart::Limb},
    };
}

} // namespace

TEST(NetLagComp, FastMovingTargetHitsRewoundPoseAndMissesLivePose) {
    LagCompHistory history;
    const entt::entity target = static_cast<entt::entity>(42u);

    const std::vector<HitCapsule> hitboxes = {
        HitCapsule{glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0f, 1.8f, 0.0f}, 0.25f, HitboxPart::Body},
    };

    // Target moves at 10 m/s along +X from t = 0.0 s to t = 1.0 s (x = 10 * t).
    for (int i = 0; i <= 20; ++i) {
        const double t = static_cast<double>(i) * 0.05;
        EntityPoseSnapshot snap;
        snap.timestamp = t;
        snap.position = glm::vec3{static_cast<float>(10.0 * t), 0.0f, 0.0f};
        snap.rotation = glm::quat{1.0f, 0.0f, 0.0f, 0.0f};
        snap.hitboxes = hitboxes;
        history.recordSnapshot(target, snap);
    }

    constexpr double kServerTime = 1.0;
    constexpr double kClientRtt = 0.200; // 200 ms RTT -> 100 ms one-way rewind -> evalTime = 0.9 s (x = 9.0 m)

    const auto liveBefore = history.samplePoseAt(target, kServerTime);
    ASSERT_TRUE(liveBefore.has_value());
    EXPECT_NEAR(liveBefore->position.x, 10.0f, 1e-5f);

    // Shooting where the entity was 100 ms ago (x = 9.0 m) HITS.
    const auto rewoundHit = history.rewindRaycast(
        glm::vec3{9.0f, 0.9f, -5.0f},
        glm::vec3{0.0f, 0.0f, 1.0f},
        20.0f,
        kServerTime,
        kClientRtt);
    ASSERT_TRUE(rewoundHit.has_value());
    EXPECT_EQ(rewoundHit->entity, target);
    EXPECT_EQ(rewoundHit->part, HitboxPart::Body);
    EXPECT_NEAR(rewoundHit->rewoundBySeconds, 0.1, 1e-9);
    EXPECT_NEAR(rewoundHit->point.x, 9.0f, 1e-4f);
    EXPECT_NEAR(rewoundHit->point.y, 0.9f, 1e-4f);
    EXPECT_NEAR(rewoundHit->point.z, -0.25f, 1e-4f);
    EXPECT_NEAR(rewoundHit->distance, 4.75f, 1e-4f);
    EXPECT_NEAR(rewoundHit->normal.z, -1.0f, 1e-4f);
    EXPECT_NEAR(glm::length(rewoundHit->normal), 1.0f, 1e-4f);

    // Shooting at its live server position (x = 10.0 m) MISSES when rewound by 100 ms.
    const auto liveAimMiss = history.rewindRaycast(
        glm::vec3{10.0f, 0.9f, -5.0f},
        glm::vec3{0.0f, 0.0f, 1.0f},
        20.0f,
        kServerTime,
        kClientRtt);
    EXPECT_FALSE(liveAimMiss.has_value());

    // Live stored pose in history remains untouched.
    const auto liveAfter = history.samplePoseAt(target, kServerTime);
    ASSERT_TRUE(liveAfter.has_value());
    EXPECT_NEAR(liveAfter->position.x, 10.0f, 1e-5f);
}

TEST(NetLagComp, FakeHighRttClampsRewindStrictlyToMaxRewindSec) {
    LagCompHistory history(2.0, 0.250);
    const entt::entity target = static_cast<entt::entity>(7u);

    const std::vector<HitCapsule> hitboxes = {
        HitCapsule{glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0f, 1.5f, 0.0f}, 0.25f, HitboxPart::Body},
    };

    // Entity moves at 10 m/s along +X over [0.0, 2.0].
    for (int i = 0; i <= 40; ++i) {
        const double t = static_cast<double>(i) * 0.05;
        EntityPoseSnapshot snap;
        snap.timestamp = t;
        snap.position = glm::vec3{static_cast<float>(10.0 * t), 0.0f, 0.0f};
        snap.hitboxes = hitboxes;
        history.recordSnapshot(target, snap);
    }

    constexpr double kServerTime = 2.0;
    constexpr double kFakeRtt = 2.000; // 2000 ms RTT -> 1.0 s raw one-way, clamped to 0.250 s

    // At 1.0 s ago (unclamped position x = 10.0 m): must MISS because rewind is clamped to 0.250 s!
    const auto unclampedShot = history.rewindRaycast(
        glm::vec3{10.0f, 0.75f, -5.0f},
        glm::vec3{0.0f, 0.0f, 1.0f},
        20.0f,
        kServerTime,
        kFakeRtt);
    EXPECT_FALSE(unclampedShot.has_value());

    // At 0.250 s ago (evalTime = 1.75 s -> x = 17.5 m): must HIT with rewoundBySeconds == 0.250!
    const auto clampedShot = history.rewindRaycast(
        glm::vec3{17.5f, 0.75f, -5.0f},
        glm::vec3{0.0f, 0.0f, 1.0f},
        20.0f,
        kServerTime,
        kFakeRtt);
    ASSERT_TRUE(clampedShot.has_value());
    EXPECT_DOUBLE_EQ(clampedShot->rewoundBySeconds, 0.250);
    EXPECT_NEAR(clampedShot->point.x, 17.5f, 1e-4f);

    // Interp delay adds to one-way RTT and still clamps to maxRewindSec.
    const auto combinedClamp = history.rewindRaycast(
        glm::vec3{17.5f, 0.75f, -5.0f},
        glm::vec3{0.0f, 0.0f, 1.0f},
        20.0f,
        kServerTime,
        0.300,  // 0.150 s one-way
        0.200); // + 0.200 s interp = 0.350 s -> clamped to 0.250 s
    ASSERT_TRUE(combinedClamp.has_value());
    EXPECT_DOUBLE_EQ(combinedClamp->rewoundBySeconds, 0.250);

    // Negative RTT clamps cleanly to 0.0 s (live pose at x = 20.0 m).
    const auto zeroClamp = history.rewindRaycast(
        glm::vec3{20.0f, 0.75f, -5.0f},
        glm::vec3{0.0f, 0.0f, 1.0f},
        20.0f,
        kServerTime,
        -0.500);
    ASSERT_TRUE(zeroClamp.has_value());
    EXPECT_DOUBLE_EQ(zeroClamp->rewoundBySeconds, 0.0);
}

TEST(NetLagComp, HeadshotVsBodyPriorityAndIgnoreEntity) {
    LagCompHistory history;
    const entt::entity shooter = static_cast<entt::entity>(1u);
    const entt::entity victim = static_cast<entt::entity>(2u);

    // Shooter stands directly between the ray origin and the victim.
    history.recordSnapshot(shooter, EntityPoseSnapshot{
        1.0,
        glm::vec3{0.0f, 0.0f, 2.0f},
        glm::quat{1.0f, 0.0f, 0.0f, 0.0f},
        makeHumanoidHitboxes(),
    });

    history.recordSnapshot(victim, EntityPoseSnapshot{
        1.0,
        glm::vec3{0.0f, 0.0f, 6.0f},
        glm::quat{1.0f, 0.0f, 0.0f, 0.0f},
        makeHumanoidHitboxes(),
    });

    // Without ignoreEntity, shooting from z = 0 hits the shooter in front.
    const auto selfHit = history.rewindRaycast(
        glm::vec3{0.0f, 1.75f, 0.0f},
        glm::vec3{0.0f, 0.0f, 1.0f},
        20.0f,
        1.0,
        0.0,
        0.0,
        entt::null);
    ASSERT_TRUE(selfHit.has_value());
    EXPECT_EQ(selfHit->entity, shooter);
    EXPECT_EQ(selfHit->part, HitboxPart::Head);

    // With ignoreEntity = shooter, ray passes through shooter and hits victim's head sphere.
    const auto headHit = history.rewindRaycast(
        glm::vec3{0.0f, 1.75f, 0.0f},
        glm::vec3{0.0f, 0.0f, 1.0f},
        20.0f,
        1.0,
        0.0,
        0.0,
        shooter);
    ASSERT_TRUE(headHit.has_value());
    EXPECT_EQ(headHit->entity, victim);
    EXPECT_EQ(headHit->part, HitboxPart::Head);
    EXPECT_NEAR(headHit->point.z, 6.0f - 0.16f, 1e-4f);

    // Aimed at victim's torso returns Body.
    const auto bodyHit = history.rewindRaycast(
        glm::vec3{0.0f, 0.90f, 0.0f},
        glm::vec3{0.0f, 0.0f, 1.0f},
        20.0f,
        1.0,
        0.0,
        0.0,
        shooter);
    ASSERT_TRUE(bodyHit.has_value());
    EXPECT_EQ(bodyHit->entity, victim);
    EXPECT_EQ(bodyHit->part, HitboxPart::Body);

    // Aimed at victim's right limb returns Limb.
    const auto limbHit = history.rewindRaycast(
        glm::vec3{0.40f, 0.90f, 0.0f},
        glm::vec3{0.0f, 0.0f, 1.0f},
        20.0f,
        1.0,
        0.0,
        0.0,
        shooter);
    ASSERT_TRUE(limbHit.has_value());
    EXPECT_EQ(limbHit->entity, victim);
    EXPECT_EQ(limbHit->part, HitboxPart::Limb);

    // Overlapping head capsule and body capsule with identical front distance prioritize Head
    // regardless of capsule ordering.
    const entt::entity overlapEntity = static_cast<entt::entity>(3u);
    LagCompHistory overlapHistory;
    overlapHistory.recordSnapshot(overlapEntity, EntityPoseSnapshot{
        1.0,
        glm::vec3{0.0f, 0.0f, 5.0f},
        glm::quat{1.0f, 0.0f, 0.0f, 0.0f},
        {
            HitCapsule{glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0f, 1.5f, 0.0f}, 0.25f, HitboxPart::Body},
            HitCapsule{glm::vec3{0.0f, 1.4f, 0.0f}, glm::vec3{0.0f, 1.8f, 0.0f}, 0.25f, HitboxPart::Head},
        },
    });
    const auto tieHit = overlapHistory.rewindRaycast(
        glm::vec3{0.0f, 1.45f, 0.0f},
        glm::vec3{0.0f, 0.0f, 1.0f},
        20.0f,
        1.0,
        0.0);
    ASSERT_TRUE(tieHit.has_value());
    EXPECT_EQ(tieHit->part, HitboxPart::Head);
}

TEST(NetLagComp, PoseInterpolationSlerpAndRotatedHitboxTransform) {
    LagCompHistory history;
    const entt::entity entity = static_cast<entt::entity>(99u);

    // Offset arm capsule in +X local space. Rotating +90 deg around +Y maps local +X to world -Z.
    const std::vector<HitCapsule> hitboxes = {
        HitCapsule{glm::vec3{1.0f, 0.0f, 0.0f}, glm::vec3{1.0f, 1.0f, 0.0f}, 0.15f, HitboxPart::Limb},
    };

    const glm::quat rotStart{1.0f, 0.0f, 0.0f, 0.0f};
    // Negated quaternion of +90 deg Y rotation tests shortest-path slerp across opposite hemispheres.
    const glm::quat rotEnd = -glm::angleAxis(glm::half_pi<float>(), glm::vec3{0.0f, 1.0f, 0.0f});

    history.recordSnapshot(entity, EntityPoseSnapshot{1.0, glm::vec3{0.0f, 0.0f, 0.0f}, rotStart, hitboxes});
    history.recordSnapshot(entity, EntityPoseSnapshot{2.0, glm::vec3{0.0f, 4.0f, 0.0f}, rotEnd, hitboxes});

    // Sample midpoint at t = 1.5 -> pos (0, 2, 0) and 45 deg Y rotation.
    const auto mid = history.samplePoseAt(entity, 1.5);
    ASSERT_TRUE(mid.has_value());
    EXPECT_NEAR(mid->position.x, 0.0f, 1e-5f);
    EXPECT_NEAR(mid->position.y, 2.0f, 1e-5f);
    EXPECT_NEAR(mid->position.z, 0.0f, 1e-5f);

    const glm::vec3 rotatedX = mid->rotation * glm::vec3{1.0f, 0.0f, 0.0f};
    const float expectedComp = std::sqrt(0.5f);
    EXPECT_NEAR(rotatedX.x, expectedComp, 1e-4f);
    EXPECT_NEAR(rotatedX.y, 0.0f, 1e-4f);
    EXPECT_NEAR(rotatedX.z, -expectedComp, 1e-4f);

    // Clamping outside snapshot window returns oldest / newest cleanly.
    const auto clampedOld = history.samplePoseAt(entity, 0.25);
    ASSERT_TRUE(clampedOld.has_value());
    EXPECT_DOUBLE_EQ(clampedOld->timestamp, 1.0);
    EXPECT_NEAR(clampedOld->position.y, 0.0f, 1e-5f);

    const auto clampedNew = history.samplePoseAt(entity, 5.0);
    ASSERT_TRUE(clampedNew.has_value());
    EXPECT_DOUBLE_EQ(clampedNew->timestamp, 2.0);
    EXPECT_NEAR(clampedNew->position.y, 4.0f, 1e-5f);

    // Rewind raycast at t = 1.5 (serverTime = 1.6, RTT = 0.200 -> evalTime = 1.5)
    // hits the rotated offset capsule at (cos 45, 2.5, -sin 45).
    const auto rotatedHit = history.rewindRaycast(
        glm::vec3{expectedComp, 2.5f, -5.0f},
        glm::vec3{0.0f, 0.0f, 1.0f},
        10.0f,
        1.6,
        0.200);
    ASSERT_TRUE(rotatedHit.has_value());
    EXPECT_EQ(rotatedHit->part, HitboxPart::Limb);
    EXPECT_NEAR(rotatedHit->point.x, expectedComp, 1e-4f);
    EXPECT_NEAR(rotatedHit->point.z, -expectedComp - 0.15f, 1e-4f);
}

TEST(NetLagComp, RingBufferPruningAndEntityLifecycle) {
    LagCompHistory history(1.0, 0.250);
    const entt::entity e1 = static_cast<entt::entity>(10u);
    const entt::entity e2 = static_cast<entt::entity>(20u);

    // Null entity and NaN timestamps are ignored.
    history.recordSnapshot(entt::null, EntityPoseSnapshot{0.0});
    history.recordSnapshot(e1, EntityPoseSnapshot{std::numeric_limits<double>::quiet_NaN()});
    EXPECT_EQ(history.trackedEntityCount(), 0u);

    // Record 0.0 .. 2.0 in 0.25 s steps: window [1.0, 2.0] has 5 snapshots.
    for (int i = 0; i <= 8; ++i) {
        const double t = static_cast<double>(i) * 0.25;
        history.recordSnapshot(e1, EntityPoseSnapshot{t, glm::vec3{static_cast<float>(t), 0.0f, 0.0f}});
    }
    EXPECT_EQ(history.trackedEntityCount(), 1u);
    EXPECT_EQ(history.snapshotCount(e1), 5u);

    // Oldest retained snapshot is at t = 1.0.
    const auto oldest = history.samplePoseAt(e1, 0.0);
    ASSERT_TRUE(oldest.has_value());
    EXPECT_DOUBLE_EQ(oldest->timestamp, 1.0);
    EXPECT_NEAR(oldest->position.x, 1.0f, 1e-5f);

    // Out-of-order insert inside window merges cleanly in sorted order.
    history.recordSnapshot(e1, EntityPoseSnapshot{1.125, glm::vec3{1.125f, 0.0f, 0.0f}});
    EXPECT_EQ(history.snapshotCount(e1), 6u);
    const auto inserted = history.samplePoseAt(e1, 1.125);
    ASSERT_TRUE(inserted.has_value());
    EXPECT_NEAR(inserted->position.x, 1.125f, 1e-5f);

    history.recordSnapshot(e2, EntityPoseSnapshot{1.0});
    EXPECT_EQ(history.trackedEntityCount(), 2u);

    history.removeEntity(e1);
    EXPECT_EQ(history.trackedEntityCount(), 1u);
    EXPECT_FALSE(history.samplePoseAt(e1, 1.5).has_value());

    history.clear();
    EXPECT_EQ(history.trackedEntityCount(), 0u);
    EXPECT_FALSE(history.samplePoseAt(e2, 1.0).has_value());
}

TEST(NetLagComp, AnalyticRayCapsuleIntersectionEdgeCases) {
    // 1. Degenerate capsule (sphere) hit with unnormalized direction.
    const auto sphereHit = rayCapsuleIntersect(
        glm::vec3{0.0f, 0.0f, -5.0f},
        glm::vec3{0.0f, 0.0f, 4.0f},
        10.0f,
        glm::vec3{0.0f, 0.0f, 0.0f},
        glm::vec3{0.0f, 0.0f, 0.0f},
        1.0f);
    ASSERT_TRUE(sphereHit.has_value());
    EXPECT_NEAR(sphereHit->distance, 4.0f, 1e-4f);
    EXPECT_NEAR(sphereHit->point.z, -1.0f, 1e-4f);
    EXPECT_NEAR(sphereHit->normal.z, -1.0f, 1e-4f);

    // 2. Ray parallel to capsule axis hits the top hemispherical cap directly.
    const auto axialCapHit = rayCapsuleIntersect(
        glm::vec3{0.0f, 5.0f, 0.0f},
        glm::vec3{0.0f, -1.0f, 0.0f},
        10.0f,
        glm::vec3{0.0f, 0.0f, 0.0f},
        glm::vec3{0.0f, 2.0f, 0.0f},
        0.5f);
    ASSERT_TRUE(axialCapHit.has_value());
    EXPECT_NEAR(axialCapHit->distance, 2.5f, 1e-4f);
    EXPECT_NEAR(axialCapHit->point.y, 2.5f, 1e-4f);
    EXPECT_NEAR(axialCapHit->normal.y, 1.0f, 1e-4f);

    // 3. Ray starting inside capsule returns distance 0 at origin.
    const auto insideHit = rayCapsuleIntersect(
        glm::vec3{0.1f, 1.0f, 0.0f},
        glm::vec3{1.0f, 0.0f, 0.0f},
        10.0f,
        glm::vec3{0.0f, 0.0f, 0.0f},
        glm::vec3{0.0f, 2.0f, 0.0f},
        0.5f);
    ASSERT_TRUE(insideHit.has_value());
    EXPECT_FLOAT_EQ(insideHit->distance, 0.0f);
    EXPECT_NEAR(insideHit->normal.x, 1.0f, 1e-4f);

    // 4. Oblique 45-degree ray hitting the cylinder side (exercises non-unit perpendicular projection).
    const auto obliqueHit = rayCapsuleIntersect(
        glm::vec3{2.0f, 0.0f, 0.0f},
        glm::vec3{-1.0f, 1.0f, 0.0f},
        10.0f,
        glm::vec3{0.0f, 0.0f, 0.0f},
        glm::vec3{0.0f, 2.0f, 0.0f},
        1.0f);
    ASSERT_TRUE(obliqueHit.has_value());
    EXPECT_NEAR(obliqueHit->distance, std::sqrt(2.0f), 1e-4f);
    EXPECT_NEAR(obliqueHit->point.x, 1.0f, 1e-4f);
    EXPECT_NEAR(obliqueHit->point.y, 1.0f, 1e-4f);
    EXPECT_NEAR(obliqueHit->normal.x, 1.0f, 1e-4f);
    EXPECT_NEAR(obliqueHit->normal.y, 0.0f, 1e-4f);

    // 5. Max distance cutoff and zero direction return nullopt.
    EXPECT_FALSE(rayCapsuleIntersect(
        glm::vec3{0.0f, 0.0f, -5.0f},
        glm::vec3{0.0f, 0.0f, 1.0f},
        3.5f,
        glm::vec3{0.0f, 0.0f, 0.0f},
        glm::vec3{0.0f, 2.0f, 0.0f},
        0.5f).has_value());

    EXPECT_FALSE(rayCapsuleIntersect(
        glm::vec3{0.0f},
        glm::vec3{0.0f},
        10.0f,
        glm::vec3{0.0f},
        glm::vec3{0.0f, 1.0f, 0.0f},
        0.5f).has_value());
}

TEST(NetLagComp, ClosestHitAcrossMultipleEntities) {
    LagCompHistory history;
    const entt::entity nearEntity = static_cast<entt::entity>(101u);
    const entt::entity farEntity = static_cast<entt::entity>(102u);

    const std::vector<HitCapsule> cap = {
        HitCapsule{glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0f, 1.0f, 0.0f}, 0.3f, HitboxPart::Body},
    };
    history.recordSnapshot(farEntity, EntityPoseSnapshot{1.0, glm::vec3{0.0f, 0.0f, 12.0f}, glm::quat{1.0f, 0.0f, 0.0f, 0.0f}, cap});
    history.recordSnapshot(nearEntity, EntityPoseSnapshot{1.0, glm::vec3{0.0f, 0.0f, 4.0f}, glm::quat{1.0f, 0.0f, 0.0f, 0.0f}, cap});

    const auto hit = history.rewindRaycast(
        glm::vec3{0.0f, 0.5f, 0.0f},
        glm::vec3{0.0f, 0.0f, 1.0f},
        25.0f,
        1.0,
        0.0);
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(hit->entity, nearEntity);
    EXPECT_NEAR(hit->distance, 3.7f, 1e-4f);
}
