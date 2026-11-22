#pragma once

#include <glm/gtc/quaternion.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace lumora {

// ---- 1. Rotational Joint PID Controller --------------------------------------------------------

/// Gains and limits for a single rotational joint PID motor.
struct JointPidConfig {
    float kp = 150.0f;           // Proportional stiffness (N*m / rad)
    float ki = 0.0f;             // Integral gain (N*m / (rad * s))
    float kd = 15.0f;            // Angular velocity damping (N*m * s / rad)
    float maxTorque = 200.0f;    // Maximum torque magnitude (N*m)
    float integralLimit = 1.0f;  // Anti-windup magnitude clamp on integralError (rad * s)
    float stiffnessScale = 1.0f; // Dynamic multiplier applied to gains [0..inf)
};

/// Per-joint persistent PID integrator & diagnostics state.
struct JointPidState {
    glm::vec3 integralError{0.0f};
    glm::vec3 previousError{0.0f};
    glm::vec3 lastTorque{0.0f};
    bool initialized = false;

    void reset() noexcept {
        integralError = glm::vec3{0.0f};
        previousError = glm::vec3{0.0f};
        lastTorque = glm::vec3{0.0f};
        initialized = false;
    }
};

/// Shortest-arc axis-angle orientation error vector (radians) rotating `currentRot` to `targetRot`,
/// expressed in the common reference space (world or parent space).
[[nodiscard]] glm::vec3 computeQuaternionErrorAxisAngle(
    const glm::quat& currentRot,
    const glm::quat& targetRot) noexcept;

/// Computes clamped 3D PID motor torque to drive `currentRot` toward `targetRot` with angular
/// velocity damping (`-kd * currentAngVel`). Applies shortest-arc sign flip and integral anti-windup.
[[nodiscard]] glm::vec3 computeJointPidTorque(
    const glm::quat& currentRot,
    const glm::quat& targetRot,
    const glm::vec3& currentAngVel,
    float dt,
    const JointPidConfig& cfg,
    JointPidState& state) noexcept;

/// Computes clamped 3D PID motor torque using relative angular velocity damping
/// (`-kd * (currentAngVel - targetAngVel)`).
[[nodiscard]] glm::vec3 computeJointPidTorque(
    const glm::quat& currentRot,
    const glm::quat& targetRot,
    const glm::vec3& currentAngVel,
    const glm::vec3& targetAngVel,
    float dt,
    const JointPidConfig& cfg,
    JointPidState& state) noexcept;

// ---- 2. Dual-Skeleton Mapping & Whole-Body Physical Summary ------------------------------------

/// Selects how a bone's PID motor interprets its target orientation.
enum class JointDriveSpace : std::uint8_t {
    Auto = 0,           // ParentRelative when parentIndex >= 0; root follows DualSkeletonState::driveRoot()
    ParentRelative = 1, // Tracks animated orientation relative to physical parent (equal & opposite reaction)
    WorldSpace = 2,     // Tracks animated world orientation directly (no parent reaction)
    Disabled = 3        // Zero torque output
};

/// Snapshot pairing one physical ragdoll rigid body with its animated target bone pose.
struct RagdollBoneSample {
    int parentIndex = -1;                              // -1 for root bone; [0..N-1] index otherwise
    float mass = 1.0f;                                 // Bone mass (kg)
    glm::vec3 physicalPosition{0.0f};                  // World CoM position of physical body
    glm::quat physicalRotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 linearVelocity{0.0f};                    // World linear velocity (m/s)
    glm::vec3 angularVelocity{0.0f};                   // World angular velocity (rad/s)
    glm::vec3 targetPosition{0.0f};                    // Animated target world position
    glm::quat targetRotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 targetAngularVelocity{0.0f};             // Optional animated world angular velocity (rad/s)
    glm::vec3 localInertiaDiag{0.0f};                  // Optional body-local inertia diagonal (kg*m^2)
};

/// Whole-body physical invariants derived from all active ragdoll bodies.
struct RagdollBodySummary {
    float totalMass = 0.0f;
    glm::vec3 com{0.0f};
    glm::vec3 comVelocity{0.0f};
    glm::vec3 angularMomentum{0.0f}; // Total angular momentum about CoM (kg*m^2/s)
    float totalKineticEnergy = 0.0f;
};

/// Outputs computed by DualSkeletonState::update().
struct DualSkeletonOutput {
    std::vector<glm::vec3> jointMotorTorques; // Motor torque generated at each bone/joint (world space, N*m)
    std::vector<glm::vec3> netBodyTorques;    // Net world torque to apply per body (+child, -parent)
    RagdollBodySummary summary;
};

/// Maps animated target bone poses and physical ragdoll bodies into per-joint PID motor torques
/// and whole-body Center-of-Mass / momentum summary.
class DualSkeletonState {
public:
    void configure(std::size_t boneCount, const JointPidConfig& defaultCfg = {});

    void setDefaultConfig(const JointPidConfig& cfg) noexcept { defaultConfig_ = cfg; }
    [[nodiscard]] const JointPidConfig& defaultConfig() const noexcept { return defaultConfig_; }

    void setJointConfig(std::size_t boneIndex, const JointPidConfig& cfg);
    [[nodiscard]] const JointPidConfig& jointConfig(std::size_t boneIndex) const noexcept;

    void setJointDriveSpace(std::size_t boneIndex, JointDriveSpace space);
    [[nodiscard]] JointDriveSpace jointDriveSpace(std::size_t boneIndex) const noexcept;

    /// Whether root bones (parentIndex < 0) in Auto mode receive world-space PID assist torque.
    void setDriveRoot(bool enabled) noexcept { driveRoot_ = enabled; }
    [[nodiscard]] bool driveRoot() const noexcept { return driveRoot_; }

    /// Whether ParentRelative joint torques apply equal-and-opposite reaction torque to parent bodies.
    void setApplyParentReaction(bool enabled) noexcept { applyParentReaction_ = enabled; }
    [[nodiscard]] bool applyParentReaction() const noexcept { return applyParentReaction_; }

    void resetPidStates() noexcept;

    [[nodiscard]] std::size_t boneCount() const noexcept { return states_.size(); }
    [[nodiscard]] const JointPidState& jointState(std::size_t boneIndex) const noexcept;

    [[nodiscard]] const DualSkeletonOutput& update(
        std::span<const RagdollBoneSample> bones,
        float dt,
        float stiffnessMultiplier = 1.0f);

    [[nodiscard]] const DualSkeletonOutput& lastOutput() const noexcept { return output_; }

    [[nodiscard]] static RagdollBodySummary computeBodySummary(
        std::span<const RagdollBoneSample> bones) noexcept;

private:
    void ensureSize(std::size_t count);

    JointPidConfig defaultConfig_{};
    std::vector<JointPidConfig> jointConfigs_;
    std::vector<bool> hasCustomConfig_;
    std::vector<JointDriveSpace> driveSpaces_;
    std::vector<JointPidState> states_;
    DualSkeletonOutput output_{};
    bool driveRoot_ = false;
    bool applyParentReaction_ = true;
};

// ---- 3. Capture Point, Support Polygon & Balance Reflex Controller -----------------------------

enum class BalanceState : std::uint8_t {
    Balanced = 0,
    RecoveringStep = 1,
    BracingFall = 2,
    Downed = 3
};

enum class FootSide : std::uint8_t {
    None = 0,
    Left = 1,
    Right = 2
};

struct FootContact {
    glm::vec3 position{0.0f};
    FootSide side = FootSide::None;
    bool grounded = true;
};

struct BalanceConfig {
    float gravity = 9.81f;
    float minComHeight = 0.05f;
    float stepThreshold = 0.05f;             // Capture-point distance outside support to start step (m)
    float stepExitThreshold = 0.02f;         // Hysteresis margin to return to Balanced (m)
    float stepOvershoot = 0.10f;             // Distance beyond capture point for landing target (m)
    float minFootSeparation = 0.20f;         // Lateral clearance from stance foot (m)
    float maxStepLength = 0.90f;             // Max recovery step reach from CoM XZ (m)
    float braceDownwardSpeed = 2.2f;         // Downward speed (-vy) triggering fall brace (m/s)
    float braceLateralSpeed = 3.5f;          // Horizontal speed when beyond step reach triggering brace (m/s)
    float braceCriticalLateralSpeed = 4.2f;  // High-speed horizontal impact threshold (m/s)
    float braceAirborneSpeed = 2.5f;         // Total speed while airborne triggering fall brace (m/s)
    float braceTiltUpDot = 0.50f;            // dot(torsoUp, +Y) below which falling triggers brace
    float braceRayLength = 1.25f;            // Suggested ray length for obstacle/ground bracing cast (m)
    float defaultArmReach = 0.70f;           // Default arm reach for hand brace queries (m)
    float handSurfaceOffset = 0.05f;         // Cushion offset along surface normal (m)
    float downedComHeight = 0.38f;           // CoM height above ground below which character is downed (m)
    float downedTiltUpDot = 0.40f;           // dot(torsoUp, +Y) threshold for downed horizontal pose
    float downedMaxSpeed = 1.5f;             // Max speed to settle into Downed instead of BracingFall (m/s)
    float balancedStiffness = 1.0f;          // Stiffness multiplier in Balanced state
    float recoveryStiffness = 1.15f;         // Stiffness multiplier in RecoveringStep state
    float bracingStiffness = 0.80f;          // Stiffness multiplier in BracingFall state
    float downedStiffness = 0.15f;           // Relaxed stiffness multiplier in Downed state
    float stiffnessBlendRate = 0.0f;         // 0 = immediate state stiffness; >0 = exp blend rate (1/s)
};

struct BalanceInput {
    glm::vec3 com{0.0f};
    glm::vec3 comVelocity{0.0f};
    glm::vec3 torsoUp{0.0f, 1.0f, 0.0f};
    glm::vec3 characterForward{0.0f, 0.0f, 1.0f};
    glm::vec3 chestPosition{0.0f};
    bool hasChestPosition = false;
    float groundY = 0.0f;
    bool inferGroundYFromFeet = true;
    std::vector<FootContact> footContacts;
};

/// World-space target for a bracing hand against a wall, car hood, or ground surface.
/// Rotation aligns local +Z with `surfaceNormal` (palm faces -Z into surface) and local +Y with fingers.
struct HandBraceTarget {
    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 surfaceNormal{0.0f, 1.0f, 0.0f};
    bool withinReach = true;
};

struct BalanceReflexOutput {
    BalanceState state = BalanceState::Balanced;
    glm::vec2 capturePointXZ{0.0f};
    std::vector<glm::vec2> supportPolygon;
    float capturePointSupportDistance = 0.0f; // < 0 inside, 0 on boundary, > 0 outside
    FootSide recoveryFoot = FootSide::None;
    glm::vec3 recoveryStepTarget{0.0f};
    glm::vec2 fallDirectionXZ{0.0f};
    bool braceActive = false;
    glm::vec3 braceRayOrigin{0.0f};
    glm::vec3 braceRayDir{0.0f, -1.0f, 0.0f};
    float braceRayLength = 1.25f;
    float stiffnessMultiplier = 1.0f;
};

/// Computes Capture Point / Extrapolated CoM in the XZ plane:
/// capturePointXZ = comXZ + comVelXZ * sqrt(comHeight / gravity).
[[nodiscard]] glm::vec2 computeCapturePointXZ(
    const glm::vec3& com,
    const glm::vec3& comVelocity,
    float groundY,
    float gravity = 9.81f,
    float minComHeight = 0.05f) noexcept;

/// Computes the 2D convex hull (CCW order, deduplicated) in XZ from grounded contact points.
/// Cleanly returns 0, 1, 2, or >=3 hull vertices.
[[nodiscard]] std::vector<glm::vec2> computeSupportPolygon(
    std::span<const glm::vec2> contactPointsXZ);

/// Exact signed Euclidean distance from `pointXZ` to convex polygon `polygonXZ`:
/// negative inside, zero on boundary, positive outside (+infinity if polygon is empty).
/// Handles 1-point, 2-point segment, and arbitrary convex polygons (CW or CCW).
[[nodiscard]] float signedDistanceToSupportPolygon(
    const glm::vec2& pointXZ,
    std::span<const glm::vec2> polygonXZ) noexcept;

/// Euphoria-style CoM / Capture-Point balance, recovery-stepping, and fall-bracing controller.
class BalanceController {
public:
    explicit BalanceController(const BalanceConfig& config = {}) noexcept : config_(config) {}

    void setConfig(const BalanceConfig& config) noexcept { config_ = config; }
    [[nodiscard]] const BalanceConfig& config() const noexcept { return config_; }

    void reset() noexcept;

    [[nodiscard]] const BalanceReflexOutput& update(
        const BalanceInput& input,
        float dt = 1.0f / 60.0f);

    [[nodiscard]] const BalanceReflexOutput& lastOutput() const noexcept { return output_; }

    /// Computes a reachable hand cushion target using the controller's last `braceRayOrigin` as
    /// the upper-body anchor (or offsetting along `hitNormal` if called before `update()`).
    [[nodiscard]] HandBraceTarget computeHandBraceTarget(
        const glm::vec3& hitPoint,
        const glm::vec3& hitNormal,
        float armReach) const noexcept;

    /// Computes a reachable hand cushion target anchored at an explicit `shoulderPos`.
    [[nodiscard]] static HandBraceTarget computeHandBraceTargetFrom(
        const glm::vec3& shoulderPos,
        const glm::vec3& hitPoint,
        const glm::vec3& hitNormal,
        float armReach,
        float surfaceOffset = 0.05f,
        const glm::vec3& preferredUp = glm::vec3{0.0f, 1.0f, 0.0f}) noexcept;

private:
    BalanceConfig config_{};
    BalanceReflexOutput output_{};
    FootSide activeRecoveryFoot_ = FootSide::None;
    FootSide lastRecoveryFoot_ = FootSide::Left;
    bool hasLastOrigin_ = false;
};

} // namespace lumora
