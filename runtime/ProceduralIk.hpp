#pragma once

#include <glm/gtc/quaternion.hpp>
#include <glm/vec3.hpp>

#include <cstddef>
#include <span>

namespace lumora {

/// Input configuration for the analytic two-bone IK solver (e.g., arms and legs).
struct TwoBoneIkInput {
    glm::vec3 rootPos{0.0f};
    glm::vec3 midPos{0.0f};
    glm::vec3 endPos{0.0f};
    glm::vec3 targetPos{0.0f};
    /// Hint position controlling the bend plane of the middle joint (knee/elbow).
    glm::vec3 polePos{0.0f};
    /// Blend factor in [0, 1] between the input pose and solved IK pose.
    float weight = 1.0f;
    /// Soft-IK threshold ratio in [0, 1] (e.g., 0.999f) to smooth full-extension transitions
    /// and prevent hyper-extension popping.
    float softStretchRatio = 0.999f;
};

/// Output of the analytic two-bone IK solver.
/// `rootDelta` and `midDelta` are world-space delta rotations mapping the input segment
/// directions (`midPos - rootPos` and `endPos - midPos`) to the solved segment directions.
struct TwoBoneIkResult {
    glm::vec3 rootPos{0.0f};
    glm::vec3 midPos{0.0f};
    glm::vec3 endPos{0.0f};
    glm::quat rootDelta{1.0f, 0.0f, 0.0f, 0.0f};
    glm::quat midDelta{1.0f, 0.0f, 0.0f, 0.0f};
    bool targetReached = false;
    bool clampedToMaxReach = false;
};

/// Solves exact two-bone analytic IK using the law of cosines and a pole vector bend plane.
/// Handles collinear joints, zero-length segments, and unreachable targets without NaN.
[[nodiscard]] TwoBoneIkResult solveTwoBoneIk(const TwoBoneIkInput& input);

/// Solves Forward And Backward Reaching IK (FABRIK) in-place over an arbitrary joint chain,
/// keeping the root joint fixed (`joints.front()`), preserving segment lengths, and clamping
/// per-joint bend angles when `maxBendAngleRad` is non-empty.
/// Returns true if the end-effector reached `target` within `tolerance`.
///
/// `maxBendAngleRad` may be empty (unconstrained), have size `joints.size()` or `joints.size() - 1`
/// (per-joint cone half-angle for joints `0 .. N - 2`), or `joints.size() - 2` (for interior
/// joints `1 .. N - 2`). Values `< 0` or `>= pi` leave that joint unconstrained.
bool solveFabrik(std::span<glm::vec3> joints,
                 const glm::vec3& target,
                 std::span<const float> maxBendAngleRad = {},
                 int maxIterations = 16,
                 float tolerance = 1e-4f);

/// Ground raycast sample used by foot placement IK.
struct IkGroundHit {
    bool hit = false;
    glm::vec3 hitPoint{0.0f};
    glm::vec3 hitNormal{0.0f, 1.0f, 0.0f};
};

/// Input configuration for bipedal foot placement and pelvis height adjustment.
struct FootPlacementInput {
    float pelvisHeight = 0.0f;
    glm::vec3 leftFootPos{0.0f};
    glm::vec3 rightFootPos{0.0f};
    IkGroundHit leftHit{};
    IkGroundHit rightHit{};
    float maxPelvisDrop = 0.4f;
    float maxPelvisLift = 0.25f;
    float footSoleOffset = 0.0f;
    float maxAnkleTiltRad = 0.785398163f; // 45 deg
    glm::vec3 upAxis{0.0f, 1.0f, 0.0f};
};

/// Solved pelvis adjustment, planted foot targets, and surface-aligned ankle rotations.
struct FootPlacementResult {
    float pelvisOffsetY = 0.0f;
    float solvedPelvisHeight = 0.0f;
    glm::vec3 leftFootTarget{0.0f};
    glm::vec3 rightFootTarget{0.0f};
    glm::quat leftAnkleRotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::quat rightAnkleRotation{1.0f, 0.0f, 0.0f, 0.0f};
    bool leftPlanted = false;
    bool rightPlanted = false;
};

/// Computes pelvis vertical offset (`>= -maxPelvisDrop`), planted foot positions, and
/// surface-aligned ankle tilt rotations from left/right ground raycast hits.
[[nodiscard]] FootPlacementResult solveFootPlacement(const FootPlacementInput& input);

/// Convenience overload accepting positional pelvis/foot/hit parameters.
[[nodiscard]] FootPlacementResult solveFootPlacement(
    float pelvisHeight,
    const glm::vec3& leftFootPos,
    const glm::vec3& rightFootPos,
    const IkGroundHit& leftHit,
    const IkGroundHit& rightHit,
    float maxPelvisDrop = 0.4f,
    float footSoleOffset = 0.0f);

/// Computes an angular-clamped world-space delta rotation (`glm::quat`) turning `forward`
/// toward `targetDir` within `[-maxYawRad, maxYawRad]` about `up` and
/// `[-maxPitchRad, maxPitchRad]` about the right axis, blended by `weight` in [0, 1].
[[nodiscard]] glm::quat solveLookAt(
    const glm::vec3& forward,
    const glm::vec3& up,
    const glm::vec3& targetDir,
    float maxYawRad,
    float maxPitchRad,
    float weight = 1.0f);

} // namespace lumora
