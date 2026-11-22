#include "runtime/ActiveRagdoll.hpp"

#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/mat3x3.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace lumora::test {
namespace {

float quaternionAngleRad(const glm::quat& a, const glm::quat& b) noexcept {
    const glm::quat qa = glm::normalize(a);
    const glm::quat qb = glm::normalize(b);
    const float d = std::clamp(std::abs(glm::dot(qa, qb)), 0.0f, 1.0f);
    return 2.0f * std::acos(d);
}

bool allFinite(const glm::vec3& v) noexcept {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

bool allFinite(const glm::quat& q) noexcept {
    return std::isfinite(q.w) && std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z);
}

} // namespace

TEST(ActiveRagdollPid, ShortestArcSignFlipMatchesDirectRotation) {
    const glm::quat current{1.0f, 0.0f, 0.0f, 0.0f};
    const glm::quat targetPositive = glm::angleAxis(glm::radians(25.0f), glm::vec3{0.0f, 1.0f, 0.0f});
    const glm::quat targetFlipped = -targetPositive; // Same SO(3) orientation, dot < 0

    ASSERT_LT(glm::dot(targetFlipped, current), 0.0f);

    const glm::vec3 errPos = computeQuaternionErrorAxisAngle(current, targetPositive);
    const glm::vec3 errFlip = computeQuaternionErrorAxisAngle(current, targetFlipped);

    EXPECT_NEAR(glm::length(errPos), glm::radians(25.0f), 1e-4f);
    EXPECT_NEAR(errFlip.x, errPos.x, 1e-4f);
    EXPECT_NEAR(errFlip.y, errPos.y, 1e-4f);
    EXPECT_NEAR(errFlip.z, errPos.z, 1e-4f);

    JointPidConfig cfg;
    cfg.kp = 100.0f;
    cfg.kd = 0.0f;
    cfg.maxTorque = 500.0f;

    JointPidState stateA;
    JointPidState stateB;
    const glm::vec3 tauA = computeJointPidTorque(current, targetPositive, glm::vec3{0.0f}, 0.016f, cfg, stateA);
    const glm::vec3 tauB = computeJointPidTorque(current, targetFlipped, glm::vec3{0.0f}, 0.016f, cfg, stateB);

    EXPECT_NEAR(tauA.x, tauB.x, 1e-4f);
    EXPECT_NEAR(tauA.y, tauB.y, 1e-4f);
    EXPECT_NEAR(tauA.z, tauB.z, 1e-4f);
    EXPECT_GT(tauB.y, 0.0f);
}

TEST(ActiveRagdollPid, AntiWindupAndTorqueClampingAndZeroStiffness) {
    JointPidConfig cfg;
    cfg.kp = 300.0f;
    cfg.ki = 120.0f;
    cfg.kd = 10.0f;
    cfg.maxTorque = 50.0f;
    cfg.integralLimit = 0.4f;
    cfg.stiffnessScale = 1.0f;

    JointPidState state;
    const glm::quat current{1.0f, 0.0f, 0.0f, 0.0f};
    const glm::quat target = glm::angleAxis(glm::radians(90.0f), glm::vec3{1.0f, 0.0f, 0.0f});

    // Pin the joint for 2 seconds to saturate integral windup.
    glm::vec3 torque{0.0f};
    for (int i = 0; i < 120; ++i) {
        torque = computeJointPidTorque(current, target, glm::vec3{0.0f}, 1.0f / 60.0f, cfg, state);
        EXPECT_LE(glm::length(state.integralError), cfg.integralLimit + 1e-4f);
        EXPECT_LE(glm::length(torque), cfg.maxTorque + 1e-4f);
    }
    EXPECT_NEAR(glm::length(state.integralError), cfg.integralLimit, 1e-4f);
    EXPECT_NEAR(glm::length(torque), cfg.maxTorque, 1e-4f);

    // Setting stiffnessScale to zero makes the joint fully limp (zero torque, cleared integral).
    cfg.stiffnessScale = 0.0f;
    const glm::vec3 limpTorque = computeJointPidTorque(
        current, target, glm::vec3{2.0f, -1.0f, 0.5f}, 1.0f / 60.0f, cfg, state);
    EXPECT_FLOAT_EQ(limpTorque.x, 0.0f);
    EXPECT_FLOAT_EQ(limpTorque.y, 0.0f);
    EXPECT_FLOAT_EQ(limpTorque.z, 0.0f);
    EXPECT_FLOAT_EQ(glm::length(state.integralError), 0.0f);
}

TEST(ActiveRagdollPid, ClosedLoopRigidBodyConvergesFrom90DegreesWithinHalfSecond) {
    // Inertially simulated rigid body with anisotropic inertia tensor starting 90 deg misaligned.
    const glm::vec3 localInertia{0.30f, 0.25f, 0.35f};
    const glm::vec3 invLocalInertia = 1.0f / localInertia;

    const glm::vec3 axis = glm::normalize(glm::vec3{1.0f, 0.6f, -0.4f});
    glm::quat rot = glm::angleAxis(glm::radians(90.0f), axis);
    const glm::quat targetRot = glm::angleAxis(glm::radians(20.0f), glm::vec3{0.0f, 1.0f, 0.0f});
    const glm::quat targetAntipodal = -targetRot; // Also test antipodal target sign flip mid-loop!
    glm::vec3 angVel{0.0f};

    JointPidConfig cfg;
    cfg.kp = 180.0f;
    cfg.ki = 10.0f;
    cfg.kd = 18.0f;
    cfg.maxTorque = 250.0f;
    cfg.integralLimit = 0.5f;
    cfg.stiffnessScale = 1.0f;
    JointPidState state;

    const float dt = 1.0f / 120.0f;
    const int steps = 60; // 0.5 s exact
    float maxAngSpeed = 0.0f;
    bool passedThroughZero = false;
    float maxOvershootAfterCrossing = 0.0f;

    for (int i = 0; i < steps; ++i) {
        const glm::quat& activeTarget = (i % 2 == 0) ? targetRot : targetAntipodal;
        const glm::vec3 torque = computeJointPidTorque(rot, activeTarget, angVel, dt, cfg, state);
        ASSERT_TRUE(allFinite(torque));

        // World-space inverse inertia: R * diag(invI) * R^T.
        const glm::mat3 R = glm::mat3_cast(rot);
        const glm::vec3 localTorque = glm::transpose(R) * torque;
        const glm::vec3 worldAlpha = R * (invLocalInertia * localTorque);

        angVel += worldAlpha * dt;
        const float speed = glm::length(angVel);
        maxAngSpeed = std::max(maxAngSpeed, speed);
        ASSERT_LT(speed, 40.0f) << "Instability blowup at step " << i;

        if (speed > 1e-7f) {
            rot = glm::normalize(glm::angleAxis(speed * dt, angVel / speed) * rot);
        }
        ASSERT_TRUE(allFinite(rot));

        const float errRad = quaternionAngleRad(rot, targetRot);
        if (errRad < glm::radians(3.0f)) {
            passedThroughZero = true;
        }
        if (passedThroughZero) {
            maxOvershootAfterCrossing = std::max(maxOvershootAfterCrossing, errRad);
        }
    }

    const float finalErrorDeg = glm::degrees(quaternionAngleRad(rot, targetRot));
    EXPECT_LT(finalErrorDeg, 2.0f);
    EXPECT_LT(glm::degrees(maxOvershootAfterCrossing), 5.0f);
    EXPECT_LT(glm::length(angVel), 1.0f);
    EXPECT_GT(maxAngSpeed, 1.0f);
}

TEST(ActiveRagdollDualSkeleton, ComputesWholeBodyMassComVelocityAndAngularMomentum) {
    std::vector<RagdollBoneSample> bones(2);
    // Bone 0 (left): m=2 at (-1, 1, 0), v=(0, 0, +3), w=(0, 2, 0), I=(1, 0.5, 1)
    bones[0].parentIndex = -1;
    bones[0].mass = 2.0f;
    bones[0].physicalPosition = {-1.0f, 1.0f, 0.0f};
    bones[0].linearVelocity = {0.0f, 0.0f, 3.0f};
    bones[0].angularVelocity = {0.0f, 2.0f, 0.0f};
    bones[0].localInertiaDiag = {1.0f, 0.5f, 1.0f};

    // Bone 1 (right): m=4 at (+2, 1, 0), v=(0, 0, 0), w=(0, 2, 0), I=(2, 1.0, 2)
    bones[1].parentIndex = 0;
    bones[1].mass = 4.0f;
    bones[1].physicalPosition = {2.0f, 1.0f, 0.0f};
    bones[1].linearVelocity = {0.0f, 0.0f, 0.0f};
    bones[1].angularVelocity = {0.0f, 2.0f, 0.0f};
    bones[1].localInertiaDiag = {2.0f, 1.0f, 2.0f};

    const RagdollBodySummary summary = DualSkeletonState::computeBodySummary(bones);
    EXPECT_FLOAT_EQ(summary.totalMass, 6.0f);
    // CoM = (2*(-1,1,0) + 4*(2,1,0)) / 6 = (1, 1, 0)
    EXPECT_NEAR(summary.com.x, 1.0f, 1e-5f);
    EXPECT_NEAR(summary.com.y, 1.0f, 1e-5f);
    EXPECT_NEAR(summary.com.z, 0.0f, 1e-5f);
    // CoM vel = (2*(0,0,3) + 4*(0,0,0)) / 6 = (0, 0, 1)
    EXPECT_NEAR(summary.comVelocity.x, 0.0f, 1e-5f);
    EXPECT_NEAR(summary.comVelocity.y, 0.0f, 1e-5f);
    EXPECT_NEAR(summary.comVelocity.z, 1.0f, 1e-5f);

    // Orbital L:
    // r0 = (-2, 0, 0), m0*vRel0 = 2*(0, 0, 2) = (0, 0, 4) -> r0 x p0 = (0, +8, 0)
    // r1 = (+1, 0, 0), m1*vRel1 = 4*(0, 0, -1) = (0, 0, -4) -> r1 x p1 = (0, +4, 0)
    // Spin L:
    // I0*w0 = (0, 0.5*2, 0) = (0, 1, 0)
    // I1*w1 = (0, 1.0*2, 0) = (0, 2, 0)
    // Total L = (0, 15, 0)
    EXPECT_NEAR(summary.angularMomentum.x, 0.0f, 1e-5f);
    EXPECT_NEAR(summary.angularMomentum.y, 15.0f, 1e-5f);
    EXPECT_NEAR(summary.angularMomentum.z, 0.0f, 1e-5f);
}

TEST(ActiveRagdollDualSkeleton, ParentRelativeJointMotorsObeyActionReactionAndScale) {
    DualSkeletonState dual;
    JointPidConfig cfg;
    cfg.kp = 120.0f;
    cfg.kd = 12.0f;
    cfg.maxTorque = 300.0f;
    dual.configure(3, cfg);

    std::vector<RagdollBoneSample> bones(3);
    // Root rotated 30 deg around Y physically, while animated root was identity.
    bones[0].parentIndex = -1;
    bones[0].mass = 15.0f;
    bones[0].physicalRotation = glm::angleAxis(glm::radians(30.0f), glm::vec3{0.0f, 1.0f, 0.0f});
    bones[0].targetRotation = glm::quat{1.0f, 0.0f, 0.0f, 0.0f};

    // Spine (child of root) matches animated relative rotation (0 deg rel -> should equal root phys rot) -> 0 torque!
    bones[1].parentIndex = 0;
    bones[1].mass = 12.0f;
    bones[1].physicalRotation = bones[0].physicalRotation;
    bones[1].targetRotation = glm::quat{1.0f, 0.0f, 0.0f, 0.0f};

    // Head (child of spine) is bent 45 deg forward physically when target relative pose is straight.
    bones[2].parentIndex = 1;
    bones[2].mass = 5.0f;
    bones[2].physicalRotation =
        bones[1].physicalRotation * glm::angleAxis(glm::radians(45.0f), glm::vec3{1.0f, 0.0f, 0.0f});
    bones[2].targetRotation = glm::quat{1.0f, 0.0f, 0.0f, 0.0f};

    const DualSkeletonOutput& outFull = dual.update(bones, 1.0f / 60.0f, 1.0f);
    ASSERT_EQ(outFull.jointMotorTorques.size(), 3u);
    ASSERT_EQ(outFull.netBodyTorques.size(), 3u);

    // Root (driveRoot == false) and aligned spine produce zero motor torque.
    EXPECT_NEAR(glm::length(outFull.jointMotorTorques[0]), 0.0f, 1e-5f);
    EXPECT_NEAR(glm::length(outFull.jointMotorTorques[1]), 0.0f, 1e-5f);
    // Head joint produces strong restoring torque, and net torque on spine is equal & opposite.
    EXPECT_GT(glm::length(outFull.jointMotorTorques[2]), 50.0f);
    EXPECT_NEAR(outFull.netBodyTorques[1].x, -outFull.netBodyTorques[2].x, 1e-4f);
    EXPECT_NEAR(outFull.netBodyTorques[1].y, -outFull.netBodyTorques[2].y, 1e-4f);
    EXPECT_NEAR(outFull.netBodyTorques[1].z, -outFull.netBodyTorques[2].z, 1e-4f);

    // Net internal torque across the whole skeleton sums to zero.
    const glm::vec3 totalNetTorque =
        outFull.netBodyTorques[0] + outFull.netBodyTorques[1] + outFull.netBodyTorques[2];
    EXPECT_NEAR(glm::length(totalNetTorque), 0.0f, 1e-4f);

    // Stiffness multiplier halves torque cleanly.
    const glm::vec3 fullHeadTorque = outFull.jointMotorTorques[2];
    const DualSkeletonOutput& outHalf = dual.update(bones, 1.0f / 60.0f, 0.5f);
    EXPECT_NEAR(glm::length(outHalf.jointMotorTorques[2]), 0.5f * glm::length(fullHeadTorque), 1e-3f);
}

TEST(ActiveRagdollBalance, SupportPolygonConvexHullAndSignedDistanceAllCardinalities) {
    // 0 points -> +inf
    const std::vector<glm::vec2> emptyPoly = computeSupportPolygon({});
    EXPECT_TRUE(emptyPoly.empty());
    EXPECT_TRUE(std::isinf(signedDistanceToSupportPolygon({0.0f, 0.0f}, emptyPoly)));

    // 1 point
    const std::vector<glm::vec2> onePoint = computeSupportPolygon(
        std::vector<glm::vec2>{{1.0f, 2.0f}, {1.0f, 2.0f}});
    ASSERT_EQ(onePoint.size(), 1u);
    EXPECT_NEAR(signedDistanceToSupportPolygon({1.0f, 2.0f}, onePoint), 0.0f, 1e-5f);
    EXPECT_NEAR(signedDistanceToSupportPolygon({4.0f, 6.0f}, onePoint), 5.0f, 1e-5f);

    // 2 points (segment from (-1, 0) to (1, 0))
    const std::vector<glm::vec2> twoPoints = computeSupportPolygon(
        std::vector<glm::vec2>{{-1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}});
    ASSERT_EQ(twoPoints.size(), 2u); // Collinear middle point collapsed
    EXPECT_NEAR(signedDistanceToSupportPolygon({0.25f, 0.0f}, twoPoints), 0.0f, 1e-5f);
    EXPECT_NEAR(signedDistanceToSupportPolygon({0.25f, 0.5f}, twoPoints), 0.5f, 1e-5f);
    EXPECT_NEAR(signedDistanceToSupportPolygon({4.0f, 4.0f}, twoPoints), 5.0f, 1e-5f); // Closest to endpoint (1,0)

    // Arbitrary polygon with interior points -> rectangle [-0.2, 0.2] x [-0.1, 0.2]
    const std::vector<glm::vec2> rawPts{
        {-0.2f, -0.1f},
        {0.2f, -0.1f},
        {0.2f, 0.2f},
        {-0.2f, 0.2f},
        {0.0f, 0.05f},   // Interior
        {0.0f, -0.1f},   // Boundary collinear
        {0.2f, 0.2f},    // Duplicate
    };
    const std::vector<glm::vec2> hull = computeSupportPolygon(rawPts);
    ASSERT_EQ(hull.size(), 4u);

    // Inside: negative distance to nearest edge
    EXPECT_NEAR(signedDistanceToSupportPolygon({0.0f, 0.05f}, hull), -0.15f, 1e-5f);
    // On edge: zero
    EXPECT_NEAR(signedDistanceToSupportPolygon({0.2f, 0.0f}, hull), 0.0f, 1e-5f);
    // Outside edge: positive
    EXPECT_NEAR(signedDistanceToSupportPolygon({0.5f, 0.0f}, hull), 0.3f, 1e-5f);
    // Outside corner: exact Euclidean corner distance sqrt(0.3^2 + 0.4^2) = 0.5
    EXPECT_NEAR(signedDistanceToSupportPolygon({0.5f, 0.6f}, hull), 0.5f, 1e-5f);
}

TEST(ActiveRagdollBalance, UprightStaysBalancedAndForwardPushTriggersRecoveryStep) {
    BalanceController controller;

    BalanceInput input;
    input.com = {0.0f, 1.0f, 0.0f};
    input.comVelocity = {0.0f, 0.0f, 0.0f};
    input.torsoUp = {0.0f, 1.0f, 0.0f};
    input.characterForward = {0.0f, 0.0f, 1.0f};
    input.footContacts = {
        {{-0.15f, 0.0f, -0.10f}, FootSide::Left, true},
        {{-0.15f, 0.0f, 0.15f}, FootSide::Left, true},
        {{0.15f, 0.0f, -0.10f}, FootSide::Right, true},
        {{0.15f, 0.0f, 0.15f}, FootSide::Right, true},
    };

    const BalanceReflexOutput balanced = controller.update(input, 1.0f / 60.0f);
    EXPECT_EQ(balanced.state, BalanceState::Balanced);
    EXPECT_LT(balanced.capturePointSupportDistance, 0.0f);
    EXPECT_EQ(balanced.recoveryFoot, FootSide::None);
    EXPECT_FALSE(balanced.braceActive);
    EXPECT_GE(balanced.stiffnessMultiplier, 0.9f);

    // Apply forward push impulse along +Z -> Capture Point exits toes (z = 0.15)
    input.comVelocity = {0.05f, 0.0f, 1.8f};
    const BalanceReflexOutput recovering = controller.update(input, 1.0f / 60.0f);
    EXPECT_EQ(recovering.state, BalanceState::RecoveringStep);
    EXPECT_GT(recovering.capturePointXZ.y, 0.15f);
    EXPECT_GT(recovering.capturePointSupportDistance, controller.config().stepThreshold);
    EXPECT_NE(recovering.recoveryFoot, FootSide::None);
    EXPECT_GT(recovering.recoveryStepTarget.z, recovering.capturePointXZ.y);
    EXPECT_GT(recovering.fallDirectionXZ.y, 0.9f);
    EXPECT_GE(recovering.stiffnessMultiplier, 1.0f);

    // Lateral push to the left (-X) selects Left foot and places step target to the left.
    controller.reset();
    input.comVelocity = {-1.8f, 0.0f, 0.0f};
    const BalanceReflexOutput leftStep = controller.update(input, 1.0f / 60.0f);
    EXPECT_EQ(leftStep.state, BalanceState::RecoveringStep);
    EXPECT_EQ(leftStep.recoveryFoot, FootSide::Left);
    EXPECT_LT(leftStep.recoveryStepTarget.x, -0.15f);

    // Collapsed / downed pose drops stiffnessMultiplier cleanly.
    input.com = {0.0f, 0.25f, 0.0f};
    input.comVelocity = {0.0f, 0.0f, 0.0f};
    input.torsoUp = {1.0f, 0.0f, 0.0f};
    for (FootContact& fc : input.footContacts) {
        fc.grounded = false;
    }
    const BalanceReflexOutput downed = controller.update(input, 1.0f / 60.0f);
    EXPECT_EQ(downed.state, BalanceState::Downed);
    EXPECT_LT(downed.stiffnessMultiplier, 0.35f);
    EXPECT_LT(downed.stiffnessMultiplier, balanced.stiffnessMultiplier);
}

TEST(ActiveRagdollBalance, CapturePointElevatedGroundClockwisePolyAndAlternatingSteps) {
    // Exact Capture Point check on elevated platform groundY = 10.0
    const glm::vec2 cp = computeCapturePointXZ(
        {1.0f, 11.0f, 2.0f}, {2.0f, 0.0f, -1.0f}, 10.0f, 9.81f, 0.05f);
    const float expectedT = std::sqrt(1.0f / 9.81f);
    EXPECT_NEAR(cp.x, 1.0f + 2.0f * expectedT, 1e-5f);
    EXPECT_NEAR(cp.y, 2.0f - 1.0f * expectedT, 1e-5f);

    // Clockwise polygon orientation works identically to CCW for signedDistanceToSupportPolygon.
    const std::vector<glm::vec2> cwSquare{
        {-1.0f, -1.0f},
        {-1.0f, 1.0f},
        {1.0f, 1.0f},
        {1.0f, -1.0f},
    };
    EXPECT_NEAR(signedDistanceToSupportPolygon({0.0f, 0.0f}, cwSquare), -1.0f, 1e-5f);
    EXPECT_NEAR(signedDistanceToSupportPolygon({1.5f, 0.0f}, cwSquare), 0.5f, 1e-5f);

    // Symmetric straight-forward pushes alternate Left / Right steps when returning through Balanced.
    BalanceController controller;
    BalanceInput input;
    input.com = {0.0f, 1.0f, 0.0f};
    input.footContacts = {
        {{-0.15f, 0.0f, 0.0f}, FootSide::Left, true},
        {{0.15f, 0.0f, 0.0f}, FootSide::Right, true},
    };
    EXPECT_EQ(controller.update(input).state, BalanceState::Balanced);

    input.comVelocity = {0.0f, 0.0f, 1.5f};
    const FootSide firstFoot = controller.update(input).recoveryFoot;
    ASSERT_NE(firstFoot, FootSide::None);

    input.comVelocity = {0.0f, 0.0f, 0.0f};
    EXPECT_EQ(controller.update(input).state, BalanceState::Balanced);

    input.comVelocity = {0.0f, 0.0f, 1.5f};
    const FootSide secondFoot = controller.update(input).recoveryFoot;
    EXPECT_NE(secondFoot, FootSide::None);
    EXPECT_NE(secondFoot, firstFoot);

    // Single grounded foot (Left) chooses free Right foot when pushed off-balance.
    controller.reset();
    input.footContacts = {
        {{-0.15f, 0.0f, 0.0f}, FootSide::Left, true},
        {{0.15f, 0.3f, 0.0f}, FootSide::Right, false},
    };
    input.com = {-0.15f, 1.0f, 0.0f};
    input.comVelocity = {0.0f, 0.0f, 0.0f};
    EXPECT_EQ(controller.update(input).state, BalanceState::Balanced);
    input.comVelocity = {0.0f, 0.0f, 1.2f};
    const BalanceReflexOutput singleFootStep = controller.update(input);
    EXPECT_EQ(singleFootStep.state, BalanceState::RecoveringStep);
    EXPECT_EQ(singleFootStep.recoveryFoot, FootSide::Right);
}

TEST(ActiveRagdollDualSkeleton, ClosedLoopParentChildChainConvergesAndHandlesNonFiniteInputs) {
    DualSkeletonState dual;
    JointPidConfig cfg;
    cfg.kp = 160.0f;
    cfg.ki = 0.0f;
    cfg.kd = 16.0f;
    cfg.maxTorque = 250.0f;
    dual.configure(2, cfg);
    dual.setDriveRoot(true); // Drive root upright while child tracks relative bend

    std::vector<RagdollBoneSample> bones(2);
    bones[0].parentIndex = -1;
    bones[0].mass = 10.0f;
    bones[0].physicalRotation = glm::angleAxis(glm::radians(25.0f), glm::vec3{0.0f, 0.0f, 1.0f});
    bones[0].targetRotation = glm::quat{1.0f, 0.0f, 0.0f, 0.0f};

    bones[1].parentIndex = 0;
    bones[1].mass = 2.0f;
    bones[1].physicalRotation = glm::angleAxis(glm::radians(-65.0f), glm::vec3{1.0f, 0.0f, 0.0f});
    bones[1].targetRotation = glm::angleAxis(glm::radians(30.0f), glm::vec3{0.0f, 1.0f, 0.0f});

    const float invI0 = 1.0f / 0.5f;
    const float invI1 = 1.0f / 0.2f;
    const float dt = 1.0f / 120.0f;

    for (int step = 0; step < 60; ++step) {
        const DualSkeletonOutput& out = dual.update(bones, dt, 1.0f);
        for (std::size_t i = 0; i < 2; ++i) {
            const float invI = (i == 0) ? invI0 : invI1;
            bones[i].angularVelocity += out.netBodyTorques[i] * invI * dt;
            const float speed = glm::length(bones[i].angularVelocity);
            if (speed > 1e-7f) {
                bones[i].physicalRotation = glm::normalize(
                    glm::angleAxis(speed * dt, bones[i].angularVelocity / speed) *
                    bones[i].physicalRotation);
            }
        }
    }

    EXPECT_LT(glm::degrees(quaternionAngleRad(bones[0].physicalRotation, bones[0].targetRotation)), 2.0f);
    EXPECT_LT(glm::degrees(quaternionAngleRad(bones[1].physicalRotation, bones[1].targetRotation)), 2.5f);

    // Non-finite inputs must never produce NaNs.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    JointPidState pidState;
    const glm::vec3 safeTau = computeJointPidTorque(
        glm::quat{nan, 0.0f, 0.0f, 0.0f},
        glm::quat{1.0f, 0.0f, 0.0f, 0.0f},
        glm::vec3{nan, 0.0f, 0.0f},
        dt,
        cfg,
        pidState);
    EXPECT_TRUE(allFinite(safeTau));
    EXPECT_FLOAT_EQ(glm::length(safeTau), 0.0f);
}

TEST(ActiveRagdollBalance, FallBracingEmitsWallRayAndReachableHandTarget) {
    BalanceController controller;

    BalanceInput input;
    input.com = {0.0f, 1.05f, 0.0f};
    input.chestPosition = {0.0f, 1.30f, 0.0f};
    input.hasChestPosition = true;
    input.comVelocity = {0.0f, -0.4f, 4.6f}; // Thrown rapidly toward a wall ahead at +Z
    input.torsoUp = {0.0f, 0.95f, 0.31f};
    input.characterForward = {0.0f, 0.0f, 1.0f};
    input.footContacts = {
        {{-0.15f, 0.0f, 0.0f}, FootSide::Left, false},
        {{0.15f, 0.0f, 0.0f}, FootSide::Right, false},
    };

    const BalanceReflexOutput& out = controller.update(input, 1.0f / 60.0f);
    EXPECT_EQ(out.state, BalanceState::BracingFall);
    EXPECT_TRUE(out.braceActive);
    EXPECT_GT(out.braceRayDir.z, 0.9f);

    // Intersect emitted brace ray with a vertical wall plane at z = 0.55 with normal (0, 0, -1).
    const float wallZ = 0.55f;
    ASSERT_GT(out.braceRayDir.z, 1e-4f);
    const float rayT = (wallZ - out.braceRayOrigin.z) / out.braceRayDir.z;
    ASSERT_GT(rayT, 0.0f);
    ASSERT_LT(rayT, out.braceRayLength);

    const glm::vec3 hitPoint = out.braceRayOrigin + out.braceRayDir * rayT;
    const glm::vec3 hitNormal{0.0f, 0.0f, -1.0f};
    const float armReach = 0.72f;

    const HandBraceTarget handTarget = controller.computeHandBraceTarget(hitPoint, hitNormal, armReach);
    EXPECT_TRUE(handTarget.withinReach);
    EXPECT_TRUE(allFinite(handTarget.position));
    EXPECT_TRUE(allFinite(handTarget.rotation));
    EXPECT_LE(glm::distance(handTarget.position, out.braceRayOrigin), armReach + 1e-4f);
    // Cushioned slightly in front of wall surface along hitNormal
    EXPECT_GT(glm::dot(handTarget.position - hitPoint, hitNormal), 0.0f);
    EXPECT_LE(glm::distance(handTarget.position, hitPoint), 0.10f);
    EXPECT_NEAR(glm::length(handTarget.rotation), 1.0f, 1e-4f);

    // Distant wall beyond arm reach is cleanly clamped to armReach with zero NaN.
    const glm::vec3 farHit{0.0f, 1.2f, 2.0f};
    const HandBraceTarget clampedTarget = controller.computeHandBraceTarget(farHit, hitNormal, armReach);
    EXPECT_FALSE(clampedTarget.withinReach);
    EXPECT_NEAR(glm::distance(clampedTarget.position, out.braceRayOrigin), armReach, 1e-4f);

    // Horizontal surface normal (0, 1, 0) such as ground or car hood has singularity-free rotation.
    const HandBraceTarget groundTarget = BalanceController::computeHandBraceTargetFrom(
        {0.0f, 0.8f, 0.0f}, {0.0f, 0.25f, 0.3f}, {0.0f, 1.0f, 0.0f}, armReach);
    EXPECT_TRUE(groundTarget.withinReach);
    EXPECT_TRUE(allFinite(groundTarget.rotation));
    EXPECT_NEAR(glm::length(groundTarget.rotation), 1.0f, 1e-4f);
}

} // namespace lumora::test
