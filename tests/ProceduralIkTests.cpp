#include "runtime/ProceduralIk.hpp"

#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <span>
#include <vector>

using namespace lumora;

namespace {

bool isAllFinite(const glm::vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

bool isAllFinite(const glm::quat& q) {
    return std::isfinite(q.w) && std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z);
}

float vectorAngleRad(const glm::vec3& a, const glm::vec3& b) {
    const float la = glm::length(a);
    const float lb = glm::length(b);
    if (la < 1e-6f || lb < 1e-6f) {
        return 0.0f;
    }
    return std::acos(std::clamp(glm::dot(a / la, b / lb), -1.0f, 1.0f));
}

} // namespace

TEST(ProceduralIk, TwoBoneReachesRandom3DTargetsAndPreservesLengthsAndPole) {
    std::mt19937 rng(20261122u);
    std::uniform_real_distribution<float> distAngle(0.0f, glm::two_pi<float>());
    std::uniform_real_distribution<float> distCos(-1.0f, 1.0f);
    std::uniform_real_distribution<float> distRadius(0.35f, 1.75f);
    std::uniform_real_distribution<float> distCoord(-2.0f, 2.0f);

    const float len1 = 1.05f;
    const float len2 = 0.85f;
    const float minReach = std::abs(len1 - len2);

    for (int trial = 0; trial < 100; ++trial) {
        const glm::vec3 root{distCoord(rng), distCoord(rng), distCoord(rng)};
        const glm::vec3 mid = root + glm::vec3{0.0f, -len1, 0.0f};
        const glm::vec3 end = mid + glm::vec3{0.2f, -len2 * 0.9f, 0.3f};
        const float actualLen1 = glm::length(mid - root);
        const float actualLen2 = glm::length(end - mid);

        const float phi = distAngle(rng);
        const float z = distCos(rng);
        const float sinTheta = std::sqrt(std::max(0.0f, 1.0f - z * z));
        const glm::vec3 dir{sinTheta * std::cos(phi), sinTheta * std::sin(phi), z};

        const float radius = std::max(minReach + 0.05f, distRadius(rng));
        const glm::vec3 target = root + dir * radius;

        const float polePhi = distAngle(rng);
        const glm::vec3 pole = root + glm::vec3{std::cos(polePhi), 0.3f, std::sin(polePhi)} * 2.0f;

        TwoBoneIkInput in;
        in.rootPos = root;
        in.midPos = mid;
        in.endPos = end;
        in.targetPos = target;
        in.polePos = pole;
        in.weight = 1.0f;
        in.softStretchRatio = 0.999f;

        const TwoBoneIkResult out = solveTwoBoneIk(in);

        ASSERT_TRUE(isAllFinite(out.rootPos));
        ASSERT_TRUE(isAllFinite(out.midPos));
        ASSERT_TRUE(isAllFinite(out.endPos));
        ASSERT_TRUE(isAllFinite(out.rootDelta));
        ASSERT_TRUE(isAllFinite(out.midDelta));

        EXPECT_TRUE(out.targetReached) << "trial " << trial;
        EXPECT_FALSE(out.clampedToMaxReach) << "trial " << trial;
        EXPECT_NEAR(glm::length(out.endPos - target), 0.0f, 1e-3f) << "trial " << trial;
        EXPECT_NEAR(glm::length(out.midPos - out.rootPos), actualLen1, 1e-4f) << "trial " << trial;
        EXPECT_NEAR(glm::length(out.endPos - out.midPos), actualLen2, 1e-4f) << "trial " << trial;

        // Mid-joint must lie on the pole side of the root->target axis (when pole isn't collinear).
        const glm::vec3 polePerp = (pole - root) - dir * glm::dot(pole - root, dir);
        if (glm::length(polePerp) > 1e-3f) {
            EXPECT_GE(glm::dot(out.midPos - out.rootPos, polePerp), -1e-4f) << "trial " << trial;
        }

        // Delta rotations must reproduce the solved bone vectors from the input bone vectors.
        const glm::vec3 rotMid = out.rootPos + out.rootDelta * (mid - root);
        const glm::vec3 rotEnd = rotMid + out.midDelta * (end - mid);
        EXPECT_NEAR(glm::length(rotMid - out.midPos), 0.0f, 1e-4f) << "trial " << trial;
        EXPECT_NEAR(glm::length(rotEnd - out.endPos), 0.0f, 1e-4f) << "trial " << trial;
    }
}

TEST(ProceduralIk, TwoBoneUnreachableTargetExtendsExactlyWithoutNaN) {
    TwoBoneIkInput in;
    in.rootPos = {1.0f, 2.0f, -0.5f};
    in.midPos = {1.0f, 1.0f, -0.5f};   // len1 = 1.0
    in.endPos = {1.0f, 0.2f, 0.1f};    // len2 = sqrt(0.64 + 0.36) = 1.0
    in.targetPos = {7.0f, 2.0f, 7.5f}; // distance = 10.0 >> 2.0
    in.polePos = {1.0f, 4.0f, -0.5f};
    in.weight = 1.0f;

    const float totalLen = glm::length(in.midPos - in.rootPos) + glm::length(in.endPos - in.midPos);
    const TwoBoneIkResult out = solveTwoBoneIk(in);

    ASSERT_TRUE(isAllFinite(out.rootPos));
    ASSERT_TRUE(isAllFinite(out.midPos));
    ASSERT_TRUE(isAllFinite(out.endPos));
    ASSERT_TRUE(isAllFinite(out.rootDelta));
    ASSERT_TRUE(isAllFinite(out.midDelta));

    EXPECT_TRUE(out.clampedToMaxReach);
    EXPECT_FALSE(out.targetReached);
    EXPECT_NEAR(glm::length(out.endPos - out.rootPos), totalLen, 1e-4f);
    EXPECT_NEAR(glm::length(out.midPos - out.rootPos), 1.0f, 1e-4f);
    EXPECT_NEAR(glm::length(out.endPos - out.midPos), 1.0f, 1e-4f);

    const glm::vec3 expectedDir = glm::normalize(in.targetPos - in.rootPos);
    const glm::vec3 actualDir = glm::normalize(out.endPos - out.rootPos);
    EXPECT_NEAR(glm::dot(expectedDir, actualDir), 1.0f, 1e-5f);
}

TEST(ProceduralIk, TwoBoneHandlesDegenerateCollinearZeroLengthAndWeights) {
    // Collinear initial bones and pole along the target line.
    TwoBoneIkInput collinear;
    collinear.rootPos = {0.0f, 0.0f, 0.0f};
    collinear.midPos = {0.0f, 1.0f, 0.0f};
    collinear.endPos = {0.0f, 2.0f, 0.0f};
    collinear.targetPos = {0.0f, 1.0f, 0.0f};
    collinear.polePos = {0.0f, 3.0f, 0.0f};

    const TwoBoneIkResult colOut = solveTwoBoneIk(collinear);
    ASSERT_TRUE(isAllFinite(colOut.midPos));
    ASSERT_TRUE(isAllFinite(colOut.endPos));
    ASSERT_TRUE(isAllFinite(colOut.rootDelta));
    ASSERT_TRUE(isAllFinite(colOut.midDelta));
    EXPECT_NEAR(glm::length(colOut.endPos - collinear.targetPos), 0.0f, 1e-3f);
    EXPECT_NEAR(glm::length(colOut.midPos - colOut.rootPos), 1.0f, 1e-4f);
    EXPECT_NEAR(glm::length(colOut.endPos - colOut.midPos), 1.0f, 1e-4f);

    // Zero-length bone and zero weight must not produce NaN.
    TwoBoneIkInput zeroBone;
    zeroBone.rootPos = {0.0f, 0.0f, 0.0f};
    zeroBone.midPos = {0.0f, 0.0f, 0.0f};
    zeroBone.endPos = {0.0f, 1.0f, 0.0f};
    zeroBone.targetPos = {1.0f, 0.0f, 0.0f};
    zeroBone.polePos = {0.0f, 0.0f, 1.0f};
    const TwoBoneIkResult zeroOut = solveTwoBoneIk(zeroBone);
    ASSERT_TRUE(isAllFinite(zeroOut.midPos));
    ASSERT_TRUE(isAllFinite(zeroOut.endPos));
    EXPECT_NEAR(glm::length(zeroOut.endPos - glm::vec3{1.0f, 0.0f, 0.0f}), 0.0f, 1e-3f);

    // Partial weight keeps bone lengths invariant and weight=0 returns exact input pose.
    TwoBoneIkInput weighted;
    weighted.rootPos = {0.0f, 0.0f, 0.0f};
    weighted.midPos = {0.0f, -1.0f, 0.1f};
    weighted.endPos = {0.0f, -2.0f, 0.0f};
    weighted.targetPos = {1.0f, -1.0f, 0.5f};
    weighted.polePos = {0.0f, -1.0f, 2.0f};
    weighted.weight = 0.0f;
    const TwoBoneIkResult w0 = solveTwoBoneIk(weighted);
    EXPECT_NEAR(glm::length(w0.endPos - weighted.endPos), 0.0f, 1e-6f);

    weighted.weight = 0.5f;
    const TwoBoneIkResult wHalf = solveTwoBoneIk(weighted);
    EXPECT_NEAR(glm::length(wHalf.midPos - wHalf.rootPos), glm::length(weighted.midPos - weighted.rootPos), 1e-4f);
    EXPECT_NEAR(glm::length(wHalf.endPos - wHalf.midPos), glm::length(weighted.endPos - weighted.midPos), 1e-4f);

    // Soft-stretch ratio monotonically approaches max reach smoothly without hyper-extension pop.
    TwoBoneIkInput softIn;
    softIn.rootPos = {0.0f, 0.0f, 0.0f};
    softIn.midPos = {0.0f, 1.0f, 0.0f};
    softIn.endPos = {0.1f, 2.0f, 0.0f};
    softIn.polePos = {0.0f, 1.0f, 1.0f};
    softIn.softStretchRatio = 0.95f;
    float prevLen = 0.0f;
    for (int step = 0; step <= 10; ++step) {
        const float d = 1.85f + 0.03f * static_cast<float>(step);
        softIn.targetPos = {0.0f, d, 0.0f};
        const TwoBoneIkResult softOut = solveTwoBoneIk(softIn);
        const float solvedReach = glm::length(softOut.endPos - softOut.rootPos);
        EXPECT_GE(solvedReach, prevLen - 1e-5f);
        EXPECT_LE(solvedReach, glm::length(softIn.midPos - softIn.rootPos) + glm::length(softIn.endPos - softIn.midPos) + 1e-4f);
        prevLen = solvedReach;
    }

    // NaN inputs fail safely without propagating NaNs.
    TwoBoneIkInput nanIn = weighted;
    nanIn.targetPos.x = std::numeric_limits<float>::quiet_NaN();
    const TwoBoneIkResult nanOut = solveTwoBoneIk(nanIn);
    EXPECT_TRUE(isAllFinite(nanOut.rootPos));
    EXPECT_TRUE(isAllFinite(nanOut.midPos));
    EXPECT_TRUE(isAllFinite(nanOut.endPos));

    // Target closer than |len1 - len2| (inner unreachable sphere) clamps cleanly to minReach.
    TwoBoneIkInput innerIn;
    innerIn.rootPos = {0.0f, 0.0f, 0.0f};
    innerIn.midPos = {0.0f, 2.0f, 0.0f};   // len1 = 2.0
    innerIn.endPos = {0.5f, 2.0f, 0.0f};   // len2 = 0.5 -> minReach = 1.5
    innerIn.targetPos = {0.0f, 0.2f, 0.0f}; // dist = 0.2 < 1.5
    innerIn.polePos = {0.0f, 1.0f, 1.0f};
    const TwoBoneIkResult innerOut = solveTwoBoneIk(innerIn);
    ASSERT_TRUE(isAllFinite(innerOut.midPos));
    ASSERT_TRUE(isAllFinite(innerOut.endPos));
    EXPECT_FALSE(innerOut.targetReached);
    EXPECT_NEAR(glm::length(innerOut.midPos - innerOut.rootPos), 2.0f, 1e-4f);
    EXPECT_NEAR(glm::length(innerOut.endPos - innerOut.midPos), 0.5f, 1e-4f);
    EXPECT_NEAR(glm::length(innerOut.endPos - innerOut.rootPos), 1.5f, 1e-4f);
}

TEST(ProceduralIk, FabrikConvergesOnFiveBoneChainAndPreservesSegmentLengths) {
    // 5-bone chain (6 joints) with varied segment lengths along +Y.
    const std::array<float, 5> expectedLengths{0.8f, 0.9f, 1.0f, 0.7f, 0.6f};
    std::vector<glm::vec3> chain{
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.8f, 0.0f},
        {0.0f, 1.7f, 0.0f},
        {0.0f, 2.7f, 0.0f},
        {0.0f, 3.4f, 0.0f},
        {0.0f, 4.0f, 0.0f},
    };

    const glm::vec3 target{1.8f, 2.2f, -1.1f};
    const bool converged = solveFabrik(chain, target, {}, 16, 1e-4f);
    EXPECT_TRUE(converged);
    EXPECT_NEAR(glm::length(chain.back() - target), 0.0f, 1e-3f);
    EXPECT_NEAR(glm::length(chain.front() - glm::vec3{0.0f, 0.0f, 0.0f}), 0.0f, 1e-6f);

    for (std::size_t i = 0; i < expectedLengths.size(); ++i) {
        ASSERT_TRUE(isAllFinite(chain[i + 1]));
        EXPECT_NEAR(glm::length(chain[i + 1] - chain[i]), expectedLengths[i], 1e-4f) << "segment " << i;
    }

    // Collinear reachable target inside the chain along the initial axis also converges.
    std::vector<glm::vec3> folded{
        {0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 2.0f, 0.0f},
        {0.0f, 3.0f, 0.0f},
        {0.0f, 4.0f, 0.0f},
        {0.0f, 5.0f, 0.0f},
    };
    const glm::vec3 inlineTarget{0.0f, 1.5f, 0.0f};
    EXPECT_TRUE(solveFabrik(folded, inlineTarget, {}, 24, 1e-4f));
    EXPECT_NEAR(glm::length(folded.back() - inlineTarget), 0.0f, 1e-3f);
    for (std::size_t i = 0; i + 1 < folded.size(); ++i) {
        EXPECT_NEAR(glm::length(folded[i + 1] - folded[i]), 1.0f, 1e-4f);
    }
}

TEST(ProceduralIk, FabrikRespectsPerJointConeConstraints) {
    std::vector<glm::vec3> chain{
        {0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 2.0f, 0.0f},
        {0.0f, 3.0f, 0.0f},
        {0.0f, 4.0f, 0.0f},
        {0.0f, 5.0f, 0.0f},
    };
    const float maxCone = glm::radians(25.0f);
    const std::array<float, 5> limits{maxCone, maxCone, maxCone, maxCone, maxCone};

    // Generate a guaranteed-reachable 3D target by forward-posing a 5-bone chain within 20 deg cones.
    const std::array<float, 5> poseStepAngles{
        glm::radians(18.0f), glm::radians(20.0f), glm::radians(18.0f), glm::radians(15.0f), glm::radians(12.0f)};
    const std::array<glm::vec3, 5> bendAxes{
        glm::normalize(glm::vec3{0.0f, 0.0f, -1.0f}),
        glm::normalize(glm::vec3{0.2f, 0.0f, -1.0f}),
        glm::normalize(glm::vec3{-0.1f, 0.0f, -1.0f}),
        glm::normalize(glm::vec3{0.3f, 0.0f, -0.9f}),
        glm::normalize(glm::vec3{0.0f, 0.0f, -1.0f}),
    };
    glm::vec3 fkPos{0.0f, 0.0f, 0.0f};
    glm::vec3 fkDir{0.0f, 1.0f, 0.0f};
    for (std::size_t i = 0; i < 5; ++i) {
        fkDir = glm::normalize(glm::angleAxis(poseStepAngles[i], bendAxes[i]) * fkDir);
        fkPos += fkDir;
    }

    const bool converged = solveFabrik(chain, fkPos, limits, 32, 1e-3f);
    EXPECT_TRUE(converged);
    EXPECT_NEAR(glm::length(chain.back() - fkPos), 0.0f, 1e-3f);

    glm::vec3 prevDir{0.0f, 1.0f, 0.0f};
    for (std::size_t i = 0; i + 1 < chain.size(); ++i) {
        const glm::vec3 seg = chain[i + 1] - chain[i];
        EXPECT_NEAR(glm::length(seg), 1.0f, 1e-4f);
        const float angle = vectorAngleRad(prevDir, seg);
        EXPECT_LE(angle, maxCone + 1e-4f) << "joint " << i;
        prevDir = glm::normalize(seg);
    }

    // Target behind root (unreachable under 25 deg per-joint cones) must still respect cones.
    std::vector<glm::vec3> stiffChain{
        {0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 2.0f, 0.0f},
        {0.0f, 3.0f, 0.0f},
    };
    const std::array<float, 3> stiffLimits{glm::radians(20.0f), glm::radians(20.0f), glm::radians(20.0f)};
    EXPECT_FALSE(solveFabrik(stiffChain, {0.0f, -2.0f, 0.0f}, stiffLimits, 16, 1e-4f));

    glm::vec3 refDir{0.0f, 1.0f, 0.0f};
    for (std::size_t i = 0; i + 1 < stiffChain.size(); ++i) {
        const glm::vec3 seg = stiffChain[i + 1] - stiffChain[i];
        EXPECT_NEAR(glm::length(seg), 1.0f, 1e-4f);
        EXPECT_LE(vectorAngleRad(refDir, seg), glm::radians(20.0f) + 1e-4f);
        refDir = glm::normalize(seg);
    }
}

TEST(ProceduralIk, FootPlacementLowersPelvisOnStairsAndTiltsAnkleCleanly) {
    // Stairs / uneven ground: left foot stays at y=0, right foot steps down to y=-0.28.
    const float pelvisHeight = 1.0f;
    const glm::vec3 leftNominal{-0.15f, 0.0f, 0.0f};
    const glm::vec3 rightNominal{0.15f, 0.0f, 0.0f};

    IkGroundHit leftHit;
    leftHit.hit = true;
    leftHit.hitPoint = {-0.15f, 0.0f, 0.0f};
    leftHit.hitNormal = glm::normalize(glm::vec3{0.0f, 1.0f, 0.0f});

    IkGroundHit rightHit;
    rightHit.hit = true;
    rightHit.hitPoint = {0.15f, -0.28f, 0.0f};
    rightHit.hitNormal = glm::normalize(glm::vec3{0.0f, 0.8660254f, 0.5f}); // 30 deg slope

    const FootPlacementResult res =
        solveFootPlacement(pelvisHeight, leftNominal, rightNominal, leftHit, rightHit, 0.40f, 0.0f);

    EXPECT_TRUE(res.leftPlanted);
    EXPECT_TRUE(res.rightPlanted);
    EXPECT_NEAR(res.pelvisOffsetY, -0.28f, 1e-5f);
    EXPECT_LE(-res.pelvisOffsetY, 0.40f + 1e-5f);
    EXPECT_NEAR(res.solvedPelvisHeight, 0.72f, 1e-5f);
    EXPECT_NEAR(glm::length(res.leftFootTarget - leftHit.hitPoint), 0.0f, 1e-5f);
    EXPECT_NEAR(glm::length(res.rightFootTarget - rightHit.hitPoint), 0.0f, 1e-5f);

    // Ankle rotations must align local +Y with surface normals.
    const glm::vec3 leftAlignedUp = res.leftAnkleRotation * glm::vec3{0.0f, 1.0f, 0.0f};
    const glm::vec3 rightAlignedUp = res.rightAnkleRotation * glm::vec3{0.0f, 1.0f, 0.0f};
    EXPECT_NEAR(glm::length(leftAlignedUp - leftHit.hitNormal), 0.0f, 1e-5f);
    EXPECT_NEAR(glm::length(rightAlignedUp - rightHit.hitNormal), 0.0f, 1e-5f);

    // Drop exceeding maxPelvisDrop is clamped cleanly.
    IkGroundHit deepDropHit;
    deepDropHit.hit = true;
    deepDropHit.hitPoint = {0.15f, -1.2f, 0.0f};
    deepDropHit.hitNormal = {0.0f, 1.0f, 0.0f};
    const FootPlacementResult clamped =
        solveFootPlacement(pelvisHeight, leftNominal, rightNominal, leftHit, deepDropHit, 0.35f, 0.0f);
    EXPECT_NEAR(clamped.pelvisOffsetY, -0.35f, 1e-5f);
    EXPECT_NEAR(clamped.solvedPelvisHeight, 0.65f, 1e-5f);

    // Step UP: right foot is higher (+0.25) while left foot is at 0 -> pelvis must not float off left foot.
    IkGroundHit stepUpHit;
    stepUpHit.hit = true;
    stepUpHit.hitPoint = {0.15f, 0.25f, 0.0f};
    stepUpHit.hitNormal = {0.0f, 1.0f, 0.0f};
    const FootPlacementResult stepUp =
        solveFootPlacement(pelvisHeight, leftNominal, rightNominal, leftHit, stepUpHit, 0.40f, 0.0f);
    EXPECT_NEAR(stepUp.pelvisOffsetY, 0.0f, 1e-5f);
    EXPECT_NEAR(stepUp.rightFootTarget.y, 0.25f, 1e-5f);

    // Steep cliff normal is clamped to maxAnkleTiltRad and sole offset moves along tilted up.
    FootPlacementInput slopeIn;
    slopeIn.pelvisHeight = 1.0f;
    slopeIn.leftFootPos = leftNominal;
    slopeIn.rightFootPos = rightNominal;
    slopeIn.leftHit = {true, {0.0f, 0.0f, 0.0f}, glm::normalize(glm::vec3{1.0f, 0.1f, 0.0f})};
    slopeIn.maxAnkleTiltRad = glm::radians(30.0f);
    slopeIn.footSoleOffset = 0.08f;
    const FootPlacementResult slopeOut = solveFootPlacement(slopeIn);
    const glm::vec3 tiltedUp = slopeOut.leftAnkleRotation * glm::vec3{0.0f, 1.0f, 0.0f};
    EXPECT_NEAR(vectorAngleRad({0.0f, 1.0f, 0.0f}, tiltedUp), glm::radians(30.0f), 1e-4f);
    EXPECT_NEAR(glm::length(slopeOut.leftFootTarget - tiltedUp * 0.08f), 0.0f, 1e-5f);
}

TEST(ProceduralIk, LookAtRotatesWithinYawAndPitchLimits) {
    const glm::vec3 forward{0.0f, 0.0f, -1.0f};
    const glm::vec3 up{0.0f, 1.0f, 0.0f};

    // Reachable look direction within angular limits aligns forward to targetDir.
    const glm::vec3 target = glm::normalize(glm::vec3{0.35f, 0.2f, -1.0f});
    const glm::quat q = solveLookAt(forward, up, target, glm::radians(60.0f), glm::radians(45.0f), 1.0f);
    ASSERT_TRUE(isAllFinite(q));
    const glm::vec3 looked = glm::normalize(q * forward);
    EXPECT_NEAR(glm::length(looked - target), 0.0f, 1e-5f);

    // Extreme target far off to the right and up gets clamped to maxYaw and maxPitch.
    const float maxYaw = glm::radians(35.0f);
    const float maxPitch = glm::radians(20.0f);
    const glm::vec3 extremeTarget = glm::normalize(glm::vec3{1.0f, 1.0f, 0.5f});
    const glm::quat clampedQ = solveLookAt(forward, up, extremeTarget, maxYaw, maxPitch, 1.0f);
    ASSERT_TRUE(isAllFinite(clampedQ));

    const glm::vec3 clampedDir = glm::normalize(clampedQ * forward);
    const glm::vec3 right = glm::cross(forward, up);
    const float actualYaw = std::atan2(-glm::dot(clampedDir, right), glm::dot(clampedDir, forward));
    const float actualPitch = std::asin(std::clamp(glm::dot(clampedDir, up), -1.0f, 1.0f));
    EXPECT_NEAR(actualYaw, -maxYaw, 1e-4f);
    EXPECT_NEAR(actualPitch, maxPitch, 1e-4f);

    // Weight = 0 returns identity; zero targetDir fails safely with identity.
    const glm::quat zeroW = solveLookAt(forward, up, target, maxYaw, maxPitch, 0.0f);
    EXPECT_NEAR(glm::length(zeroW * forward - forward), 0.0f, 1e-6f);

    const glm::quat zeroDir = solveLookAt(forward, up, {0.0f, 0.0f, 0.0f}, maxYaw, maxPitch, 1.0f);
    ASSERT_TRUE(isAllFinite(zeroDir));
    EXPECT_NEAR(glm::length(zeroDir * forward - forward), 0.0f, 1e-6f);

    // Non-canonical basis (+X forward) and backward target clamped cleanly without roll/NaN.
    const glm::vec3 sideFwd{1.0f, 0.0f, 0.0f};
    const glm::quat turnQ = solveLookAt(sideFwd, up, {-1.0f, 0.2f, 0.2f}, glm::radians(45.0f), glm::radians(30.0f), 1.0f);
    ASSERT_TRUE(isAllFinite(turnQ));
    const glm::vec3 turnedFwd = glm::normalize(turnQ * sideFwd);
    EXPECT_NEAR(glm::dot(turnedFwd, sideFwd), std::cos(glm::radians(45.0f)) * std::cos(std::asin(turnedFwd.y)), 1e-4f);

    // Weight = 0.5 produces half rotation.
    const glm::quat halfQ = solveLookAt(forward, up, glm::normalize(glm::vec3{1.0f, 0.0f, -1.0f}),
                                        glm::radians(90.0f), glm::radians(45.0f), 0.5f);
    const glm::vec3 halfDir = glm::normalize(halfQ * forward);
    EXPECT_NEAR(vectorAngleRad(forward, halfDir), glm::radians(22.5f), 1e-4f);
}

TEST(ProceduralIk, FootPlacementHandlesNoHitsAndInvertedNormals) {
    const IkGroundHit miss{};
    const FootPlacementResult noHit = solveFootPlacement(
        1.0f, {-0.15f, 0.0f, 0.0f}, {0.15f, 0.0f, 0.0f}, miss, miss, 0.4f, 0.0f);
    EXPECT_FALSE(noHit.leftPlanted);
    EXPECT_FALSE(noHit.rightPlanted);
    EXPECT_NEAR(noHit.pelvisOffsetY, 0.0f, 1e-6f);
    EXPECT_NEAR(noHit.solvedPelvisHeight, 1.0f, 1e-6f);

    // Inverted normal flips cleanly toward +Y so ankle never flips upside down.
    const IkGroundHit flippedHit{true, {0.0f, 0.0f, 0.0f}, glm::normalize(glm::vec3{0.0f, -0.9659f, -0.2588f})};
    const FootPlacementResult flippedRes = solveFootPlacement(
        1.0f, {-0.15f, 0.0f, 0.0f}, {0.15f, 0.0f, 0.0f}, flippedHit, miss, 0.4f, 0.0f);
    const glm::vec3 flippedUp = flippedRes.leftAnkleRotation * glm::vec3{0.0f, 1.0f, 0.0f};
    EXPECT_GT(flippedUp.y, 0.9f);
}
