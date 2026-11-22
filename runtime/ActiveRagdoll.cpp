#include "runtime/ActiveRagdoll.hpp"

#include <glm/geometric.hpp>
#include <glm/mat3x3.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace lumora {

namespace {

[[nodiscard]] bool isFinite(float v) noexcept {
    return std::isfinite(v);
}

[[nodiscard]] bool isFinite(const glm::vec2& v) noexcept {
    return std::isfinite(v.x) && std::isfinite(v.y);
}

[[nodiscard]] bool isFinite(const glm::vec3& v) noexcept {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

[[nodiscard]] bool isFinite(const glm::quat& q) noexcept {
    return std::isfinite(q.w) && std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z);
}

[[nodiscard]] glm::quat safeNormalize(const glm::quat& q) noexcept {
    if (!isFinite(q)) {
        return glm::quat{1.0f, 0.0f, 0.0f, 0.0f};
    }
    const float len = glm::length(q);
    if (len <= 1e-8f) {
        return glm::quat{1.0f, 0.0f, 0.0f, 0.0f};
    }
    return q / len;
}

[[nodiscard]] glm::vec2 safeNormalize(const glm::vec2& v, const glm::vec2& fallback) noexcept {
    if (!isFinite(v)) {
        return fallback;
    }
    const float len = glm::length(v);
    if (len <= 1e-8f) {
        return fallback;
    }
    return v / len;
}

[[nodiscard]] glm::vec3 safeNormalize(const glm::vec3& v, const glm::vec3& fallback) noexcept {
    if (!isFinite(v)) {
        return fallback;
    }
    const float len = glm::length(v);
    if (len <= 1e-8f) {
        return fallback;
    }
    return v / len;
}

[[nodiscard]] float cross2D(const glm::vec2& a, const glm::vec2& b) noexcept {
    return a.x * b.y - a.y * b.x;
}

[[nodiscard]] float cross2D(const glm::vec2& o, const glm::vec2& a, const glm::vec2& b) noexcept {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

[[nodiscard]] glm::vec2 closestPointOnSegment(
    const glm::vec2& p,
    const glm::vec2& a,
    const glm::vec2& b) noexcept {
    const glm::vec2 ab = b - a;
    const float abLenSq = glm::dot(ab, ab);
    if (abLenSq <= 1e-12f) {
        return a;
    }
    const float t = std::clamp(glm::dot(p - a, ab) / abLenSq, 0.0f, 1.0f);
    return a + ab * t;
}

[[nodiscard]] glm::vec2 closestPointOnPolygonBoundary(
    const glm::vec2& p,
    std::span<const glm::vec2> polygon) noexcept {
    if (polygon.empty()) {
        return p;
    }
    if (polygon.size() == 1) {
        return polygon[0];
    }
    if (polygon.size() == 2) {
        return closestPointOnSegment(p, polygon[0], polygon[1]);
    }

    glm::vec2 best = polygon[0];
    float bestDistSq = std::numeric_limits<float>::infinity();
    const std::size_t count = polygon.size();
    for (std::size_t i = 0; i < count; ++i) {
        const glm::vec2& a = polygon[i];
        const glm::vec2& b = polygon[(i + 1) % count];
        const glm::vec2 candidate = closestPointOnSegment(p, a, b);
        const glm::vec2 diff = p - candidate;
        const float distSq = glm::dot(diff, diff);
        if (distSq < bestDistSq) {
            bestDistSq = distSq;
            best = candidate;
        }
    }
    return best;
}

[[nodiscard]] glm::quat buildSurfaceAlignedHandRotation(
    const glm::vec3& surfaceNormal,
    const glm::vec3& preferredUp) noexcept {
    const glm::vec3 zAxis = safeNormalize(surfaceNormal, glm::vec3{0.0f, 1.0f, 0.0f});
    glm::vec3 refUp = safeNormalize(preferredUp, glm::vec3{0.0f, 1.0f, 0.0f});
    if (std::abs(glm::dot(refUp, zAxis)) > 0.95f) {
        refUp = (std::abs(zAxis.z) < 0.95f) ? glm::vec3{0.0f, 0.0f, 1.0f} : glm::vec3{1.0f, 0.0f, 0.0f};
    }
    const glm::vec3 xAxis = safeNormalize(glm::cross(refUp, zAxis), glm::vec3{1.0f, 0.0f, 0.0f});
    const glm::vec3 yAxis = safeNormalize(glm::cross(zAxis, xAxis), glm::vec3{0.0f, 1.0f, 0.0f});
    const glm::mat3 basis{xAxis, yAxis, zAxis};
    return safeNormalize(glm::quat_cast(basis));
}

} // namespace

// ---- 1. Rotational Joint PID Controller --------------------------------------------------------

glm::vec3 computeQuaternionErrorAxisAngle(
    const glm::quat& currentRot,
    const glm::quat& targetRot) noexcept {
    const glm::quat qCurrent = safeNormalize(currentRot);
    glm::quat qTarget = safeNormalize(targetRot);

    // Ensure shortest-arc rotation between antipodal quaternion hemispheres.
    if (glm::dot(qTarget, qCurrent) < 0.0f) {
        qTarget = -qTarget;
    }

    glm::quat qErr = qTarget * glm::conjugate(qCurrent);
    const float errLen = glm::length(qErr);
    if (errLen <= 1e-8f || !isFinite(qErr)) {
        return glm::vec3{0.0f};
    }
    qErr /= errLen;
    if (qErr.w < 0.0f) {
        qErr = -qErr;
    }

    const glm::vec3 v{qErr.x, qErr.y, qErr.z};
    const float sinHalf = glm::length(v);
    if (sinHalf < 1e-6f) {
        return 2.0f * v;
    }
    const float angle = 2.0f * std::atan2(sinHalf, qErr.w);
    return (angle / sinHalf) * v;
}

glm::vec3 computeJointPidTorque(
    const glm::quat& currentRot,
    const glm::quat& targetRot,
    const glm::vec3& currentAngVel,
    float dt,
    const JointPidConfig& cfg,
    JointPidState& state) noexcept {
    return computeJointPidTorque(
        currentRot, targetRot, currentAngVel, glm::vec3{0.0f}, dt, cfg, state);
}

glm::vec3 computeJointPidTorque(
    const glm::quat& currentRot,
    const glm::quat& targetRot,
    const glm::vec3& currentAngVel,
    const glm::vec3& targetAngVel,
    float dt,
    const JointPidConfig& cfg,
    JointPidState& state) noexcept {
    if (!isFinite(currentAngVel) || !isFinite(targetAngVel) || !isFinite(dt)) {
        state.lastTorque = glm::vec3{0.0f};
        state.initialized = true;
        return glm::vec3{0.0f};
    }

    const float scale = isFinite(cfg.stiffnessScale) ? std::max(0.0f, cfg.stiffnessScale) : 0.0f;
    const float maxMag = isFinite(cfg.maxTorque) ? std::max(0.0f, cfg.maxTorque) : 0.0f;
    if (scale <= 0.0f || maxMag <= 0.0f) {
        state.integralError = glm::vec3{0.0f};
        state.previousError = glm::vec3{0.0f};
        state.lastTorque = glm::vec3{0.0f};
        state.initialized = true;
        return glm::vec3{0.0f};
    }

    const glm::vec3 error = computeQuaternionErrorAxisAngle(currentRot, targetRot);

    const float intLimit = isFinite(cfg.integralLimit) ? std::max(0.0f, cfg.integralLimit) : 0.0f;
    if (dt > 0.0f && intLimit > 0.0f) {
        state.integralError += error * dt;
        const float intLen = glm::length(state.integralError);
        if (intLen > intLimit) {
            state.integralError *= (intLimit / intLen);
        }
    } else if (intLimit <= 0.0f) {
        state.integralError = glm::vec3{0.0f};
    }

    const float kp = isFinite(cfg.kp) ? cfg.kp * scale : 0.0f;
    const float ki = isFinite(cfg.ki) ? cfg.ki * scale : 0.0f;
    const float kd = isFinite(cfg.kd) ? cfg.kd * scale : 0.0f;

    const glm::vec3 relAngVel = currentAngVel - targetAngVel;
    glm::vec3 torque = kp * error + ki * state.integralError - kd * relAngVel;

    if (!isFinite(torque)) {
        torque = glm::vec3{0.0f};
    } else {
        const float torqueMag = glm::length(torque);
        if (torqueMag > maxMag) {
            torque *= (maxMag / torqueMag);
        }
    }

    state.previousError = error;
    state.lastTorque = torque;
    state.initialized = true;
    return torque;
}

// ---- 2. Dual-Skeleton Mapping & Whole-Body Physical Summary ------------------------------------

void DualSkeletonState::configure(std::size_t boneCount, const JointPidConfig& defaultCfg) {
    defaultConfig_ = defaultCfg;
    jointConfigs_.assign(boneCount, defaultCfg);
    hasCustomConfig_.assign(boneCount, false);
    driveSpaces_.assign(boneCount, JointDriveSpace::Auto);
    states_.assign(boneCount, JointPidState{});
    output_.jointMotorTorques.assign(boneCount, glm::vec3{0.0f});
    output_.netBodyTorques.assign(boneCount, glm::vec3{0.0f});
    output_.summary = RagdollBodySummary{};
}

void DualSkeletonState::ensureSize(std::size_t count) {
    const std::size_t oldCount = states_.size();
    if (oldCount == count) {
        return;
    }
    jointConfigs_.resize(count, defaultConfig_);
    hasCustomConfig_.resize(count, false);
    driveSpaces_.resize(count, JointDriveSpace::Auto);
    states_.resize(count);
    output_.jointMotorTorques.resize(count, glm::vec3{0.0f});
    output_.netBodyTorques.resize(count, glm::vec3{0.0f});
}

void DualSkeletonState::setJointConfig(std::size_t boneIndex, const JointPidConfig& cfg) {
    if (boneIndex >= states_.size()) {
        ensureSize(boneIndex + 1);
    }
    jointConfigs_[boneIndex] = cfg;
    hasCustomConfig_[boneIndex] = true;
}

const JointPidConfig& DualSkeletonState::jointConfig(std::size_t boneIndex) const noexcept {
    if (boneIndex < jointConfigs_.size() && hasCustomConfig_[boneIndex]) {
        return jointConfigs_[boneIndex];
    }
    return defaultConfig_;
}

void DualSkeletonState::setJointDriveSpace(std::size_t boneIndex, JointDriveSpace space) {
    if (boneIndex >= states_.size()) {
        ensureSize(boneIndex + 1);
    }
    driveSpaces_[boneIndex] = space;
}

JointDriveSpace DualSkeletonState::jointDriveSpace(std::size_t boneIndex) const noexcept {
    if (boneIndex < driveSpaces_.size()) {
        return driveSpaces_[boneIndex];
    }
    return JointDriveSpace::Auto;
}

void DualSkeletonState::resetPidStates() noexcept {
    for (JointPidState& state : states_) {
        state.reset();
    }
    std::fill(output_.jointMotorTorques.begin(), output_.jointMotorTorques.end(), glm::vec3{0.0f});
    std::fill(output_.netBodyTorques.begin(), output_.netBodyTorques.end(), glm::vec3{0.0f});
}

const JointPidState& DualSkeletonState::jointState(std::size_t boneIndex) const noexcept {
    static const JointPidState kEmptyState{};
    if (boneIndex < states_.size()) {
        return states_[boneIndex];
    }
    return kEmptyState;
}

RagdollBodySummary DualSkeletonState::computeBodySummary(
    std::span<const RagdollBoneSample> bones) noexcept {
    RagdollBodySummary summary{};
    if (bones.empty()) {
        return summary;
    }

    glm::vec3 weightedPos{0.0f};
    glm::vec3 weightedVel{0.0f};
    for (const RagdollBoneSample& b : bones) {
        if (!isFinite(b.mass) || b.mass <= 0.0f) {
            continue;
        }
        const glm::vec3 pos = isFinite(b.physicalPosition) ? b.physicalPosition : glm::vec3{0.0f};
        const glm::vec3 vel = isFinite(b.linearVelocity) ? b.linearVelocity : glm::vec3{0.0f};
        summary.totalMass += b.mass;
        weightedPos += b.mass * pos;
        weightedVel += b.mass * vel;
    }

    if (summary.totalMass <= 1e-8f) {
        summary.totalMass = 0.0f;
        return summary;
    }

    summary.com = weightedPos / summary.totalMass;
    summary.comVelocity = weightedVel / summary.totalMass;

    for (const RagdollBoneSample& b : bones) {
        if (!isFinite(b.mass) || b.mass <= 0.0f) {
            continue;
        }
        const glm::vec3 pos = isFinite(b.physicalPosition) ? b.physicalPosition : glm::vec3{0.0f};
        const glm::vec3 vel = isFinite(b.linearVelocity) ? b.linearVelocity : glm::vec3{0.0f};
        const glm::vec3 omega = isFinite(b.angularVelocity) ? b.angularVelocity : glm::vec3{0.0f};

        const glm::vec3 r = pos - summary.com;
        const glm::vec3 vRel = vel - summary.comVelocity;
        const glm::vec3 orbitalL = glm::cross(r, b.mass * vRel);

        glm::vec3 spinL{0.0f};
        if (isFinite(b.localInertiaDiag) &&
            (b.localInertiaDiag.x > 0.0f || b.localInertiaDiag.y > 0.0f || b.localInertiaDiag.z > 0.0f)) {
            const glm::quat q = safeNormalize(b.physicalRotation);
            const glm::vec3 localOmega = glm::conjugate(q) * omega;
            const glm::vec3 inertia = glm::max(b.localInertiaDiag, glm::vec3{0.0f});
            spinL = q * (inertia * localOmega);
        }

        summary.angularMomentum += orbitalL + spinL;
        summary.totalKineticEnergy +=
            0.5f * b.mass * glm::dot(vel, vel) + 0.5f * glm::dot(omega, spinL);
    }

    return summary;
}

const DualSkeletonOutput& DualSkeletonState::update(
    std::span<const RagdollBoneSample> bones,
    float dt,
    float stiffnessMultiplier) {
    const std::size_t n = bones.size();
    ensureSize(n);

    output_.summary = computeBodySummary(bones);
    std::fill(output_.jointMotorTorques.begin(), output_.jointMotorTorques.end(), glm::vec3{0.0f});
    std::fill(output_.netBodyTorques.begin(), output_.netBodyTorques.end(), glm::vec3{0.0f});

    const float clampedMult = isFinite(stiffnessMultiplier) ? std::max(0.0f, stiffnessMultiplier) : 0.0f;

    for (std::size_t i = 0; i < n; ++i) {
        const RagdollBoneSample& bone = bones[i];
        const int parentIdx = bone.parentIndex;
        const bool hasValidParent =
            (parentIdx >= 0) && (static_cast<std::size_t>(parentIdx) < n) &&
            (static_cast<std::size_t>(parentIdx) != i);

        JointDriveSpace effectiveSpace = driveSpaces_[i];
        if (effectiveSpace == JointDriveSpace::Auto) {
            if (hasValidParent) {
                effectiveSpace = JointDriveSpace::ParentRelative;
            } else {
                effectiveSpace = driveRoot_ ? JointDriveSpace::WorldSpace : JointDriveSpace::Disabled;
            }
        } else if (effectiveSpace == JointDriveSpace::ParentRelative && !hasValidParent) {
            effectiveSpace = driveRoot_ ? JointDriveSpace::WorldSpace : JointDriveSpace::Disabled;
        }

        if (effectiveSpace == JointDriveSpace::Disabled) {
            states_[i].reset();
            continue;
        }

        JointPidConfig cfg = hasCustomConfig_[i] ? jointConfigs_[i] : defaultConfig_;
        cfg.stiffnessScale = std::max(0.0f, cfg.stiffnessScale * clampedMult);

        const glm::quat qChildPhys = safeNormalize(bone.physicalRotation);
        const glm::quat qChildTarget = safeNormalize(bone.targetRotation);
        const glm::vec3 childAngVel =
            isFinite(bone.angularVelocity) ? bone.angularVelocity : glm::vec3{0.0f};
        const glm::vec3 childTargetAngVel =
            isFinite(bone.targetAngularVelocity) ? bone.targetAngularVelocity : glm::vec3{0.0f};

        if (effectiveSpace == JointDriveSpace::ParentRelative) {
            const RagdollBoneSample& parent = bones[static_cast<std::size_t>(parentIdx)];
            const glm::quat qParentPhys = safeNormalize(parent.physicalRotation);
            const glm::quat qParentTarget = safeNormalize(parent.targetRotation);

            const glm::quat qTargetRel = safeNormalize(glm::conjugate(qParentTarget) * qChildTarget);
            const glm::quat qDesiredWorld = safeNormalize(qParentPhys * qTargetRel);

            const glm::vec3 parentAngVel =
                isFinite(parent.angularVelocity) ? parent.angularVelocity : glm::vec3{0.0f};
            const glm::vec3 parentTargetAngVel =
                isFinite(parent.targetAngularVelocity) ? parent.targetAngularVelocity : glm::vec3{0.0f};
            const glm::vec3 relAnimAngVel = childTargetAngVel - parentTargetAngVel;
            const glm::vec3 desiredChildAngVel =
                parentAngVel + qParentPhys * (glm::conjugate(qParentTarget) * relAnimAngVel);

            const glm::vec3 torque = computeJointPidTorque(
                qChildPhys, qDesiredWorld, childAngVel, desiredChildAngVel, dt, cfg, states_[i]);
            output_.jointMotorTorques[i] = torque;
            output_.netBodyTorques[i] += torque;
            if (applyParentReaction_) {
                output_.netBodyTorques[static_cast<std::size_t>(parentIdx)] -= torque;
            }
        } else {
            const glm::vec3 torque = computeJointPidTorque(
                qChildPhys, qChildTarget, childAngVel, childTargetAngVel, dt, cfg, states_[i]);
            output_.jointMotorTorques[i] = torque;
            output_.netBodyTorques[i] += torque;
        }
    }

    return output_;
}

// ---- 3. Capture Point, Support Polygon & Balance Reflex Controller -----------------------------

glm::vec2 computeCapturePointXZ(
    const glm::vec3& com,
    const glm::vec3& comVelocity,
    float groundY,
    float gravity,
    float minComHeight) noexcept {
    const glm::vec2 comXZ = isFinite(com) ? glm::vec2{com.x, com.z} : glm::vec2{0.0f};
    const glm::vec2 velXZ =
        isFinite(comVelocity) ? glm::vec2{comVelocity.x, comVelocity.z} : glm::vec2{0.0f};
    const float safeMinH = isFinite(minComHeight) ? std::max(minComHeight, 1e-4f) : 0.05f;
    const float safeGravity = isFinite(gravity) ? std::max(gravity, 1e-3f) : 9.81f;
    const float safeGroundY = isFinite(groundY) ? groundY : 0.0f;
    const float rawHeight = isFinite(com.y) ? (com.y - safeGroundY) : safeMinH;
    const float comHeight = std::max(rawHeight, safeMinH);
    const float omegaInv = std::sqrt(comHeight / safeGravity);
    return comXZ + velXZ * omegaInv;
}

std::vector<glm::vec2> computeSupportPolygon(std::span<const glm::vec2> contactPointsXZ) {
    std::vector<glm::vec2> pts;
    pts.reserve(contactPointsXZ.size());
    for (const glm::vec2& p : contactPointsXZ) {
        if (isFinite(p)) {
            pts.push_back(p);
        }
    }
    if (pts.empty()) {
        return {};
    }

    std::sort(pts.begin(), pts.end(), [](const glm::vec2& a, const glm::vec2& b) noexcept {
        if (a.x != b.x) {
            return a.x < b.x;
        }
        return a.y < b.y;
    });

    // Deduplicate nearly identical points so degenerate hulls collapse cleanly to 1 or 2 vertices.
    std::vector<glm::vec2> unique;
    unique.reserve(pts.size());
    for (const glm::vec2& p : pts) {
        bool duplicate = false;
        for (const glm::vec2& u : unique) {
            const glm::vec2 d = p - u;
            if (glm::dot(d, d) <= 1e-10f) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            unique.push_back(p);
        }
    }

    if (unique.size() <= 2) {
        return unique;
    }

    std::vector<glm::vec2> lower;
    lower.reserve(unique.size());
    for (const glm::vec2& p : unique) {
        while (lower.size() >= 2 &&
               cross2D(lower[lower.size() - 2], lower[lower.size() - 1], p) <= 1e-6f) {
            lower.pop_back();
        }
        lower.push_back(p);
    }

    std::vector<glm::vec2> upper;
    upper.reserve(unique.size());
    for (std::size_t idx = unique.size(); idx-- > 0;) {
        const glm::vec2& p = unique[idx];
        while (upper.size() >= 2 &&
               cross2D(upper[upper.size() - 2], upper[upper.size() - 1], p) <= 1e-6f) {
            upper.pop_back();
        }
        upper.push_back(p);
    }

    lower.pop_back();
    upper.pop_back();
    lower.insert(lower.end(), upper.begin(), upper.end());
    return lower;
}

float signedDistanceToSupportPolygon(
    const glm::vec2& pointXZ,
    std::span<const glm::vec2> polygonXZ) noexcept {
    if (!isFinite(pointXZ) || polygonXZ.empty()) {
        return std::numeric_limits<float>::infinity();
    }
    if (polygonXZ.size() == 1) {
        return glm::length(pointXZ - polygonXZ[0]);
    }
    if (polygonXZ.size() == 2) {
        const glm::vec2 closest = closestPointOnSegment(pointXZ, polygonXZ[0], polygonXZ[1]);
        return glm::length(pointXZ - closest);
    }

    const std::size_t n = polygonXZ.size();
    float signedArea2 = 0.0f;
    float minDistSq = std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const glm::vec2& a = polygonXZ[i];
        const glm::vec2& b = polygonXZ[(i + 1) % n];
        signedArea2 += cross2D(a, b);
        const glm::vec2 closest = closestPointOnSegment(pointXZ, a, b);
        const glm::vec2 diff = pointXZ - closest;
        minDistSq = std::min(minDistSq, glm::dot(diff, diff));
    }

    const float dist = std::sqrt(std::max(0.0f, minDistSq));
    if (std::abs(signedArea2) <= 1e-7f) {
        return dist;
    }

    const float orient = (signedArea2 > 0.0f) ? 1.0f : -1.0f;
    bool inside = true;
    for (std::size_t i = 0; i < n; ++i) {
        const glm::vec2& a = polygonXZ[i];
        const glm::vec2& b = polygonXZ[(i + 1) % n];
        if (orient * cross2D(b - a, pointXZ - a) < -1e-6f) {
            inside = false;
            break;
        }
    }

    return inside ? -dist : dist;
}

void BalanceController::reset() noexcept {
    output_ = BalanceReflexOutput{};
    output_.braceRayLength = config_.braceRayLength;
    output_.stiffnessMultiplier = config_.balancedStiffness;
    activeRecoveryFoot_ = FootSide::None;
    lastRecoveryFoot_ = FootSide::Left;
    hasLastOrigin_ = false;
}

const BalanceReflexOutput& BalanceController::update(const BalanceInput& input, float dt) {
    std::vector<glm::vec2> groundedXZ;
    groundedXZ.reserve(input.footContacts.size());

    glm::vec2 leftGroundedSum{0.0f};
    int leftGroundedCount = 0;
    glm::vec2 rightGroundedSum{0.0f};
    int rightGroundedCount = 0;
    glm::vec2 leftAllSum{0.0f};
    int leftAllCount = 0;
    glm::vec2 rightAllSum{0.0f};
    int rightAllCount = 0;
    float groundedYSum = 0.0f;
    int groundedCount = 0;

    for (const FootContact& fc : input.footContacts) {
        if (!isFinite(fc.position)) {
            continue;
        }
        const glm::vec2 pXZ{fc.position.x, fc.position.z};
        if (fc.side == FootSide::Left) {
            leftAllSum += pXZ;
            ++leftAllCount;
        } else if (fc.side == FootSide::Right) {
            rightAllSum += pXZ;
            ++rightAllCount;
        }
        if (fc.grounded) {
            groundedXZ.push_back(pXZ);
            groundedYSum += fc.position.y;
            ++groundedCount;
            if (fc.side == FootSide::Left) {
                leftGroundedSum += pXZ;
                ++leftGroundedCount;
            } else if (fc.side == FootSide::Right) {
                rightGroundedSum += pXZ;
                ++rightGroundedCount;
            }
        }
    }

    const float effectiveGroundY =
        (input.inferGroundYFromFeet && groundedCount > 0)
            ? (groundedYSum / static_cast<float>(groundedCount))
            : (isFinite(input.groundY) ? input.groundY : 0.0f);

    const glm::vec3 safeCom = isFinite(input.com) ? input.com : glm::vec3{0.0f};
    const glm::vec3 safeVel = isFinite(input.comVelocity) ? input.comVelocity : glm::vec3{0.0f};
    const glm::vec3 torsoUpUnit = safeNormalize(input.torsoUp, glm::vec3{0.0f, 1.0f, 0.0f});
    const glm::vec2 comXZ{safeCom.x, safeCom.z};
    const glm::vec2 velXZ{safeVel.x, safeVel.z};

    output_.capturePointXZ = computeCapturePointXZ(
        safeCom, safeVel, effectiveGroundY, config_.gravity, config_.minComHeight);
    output_.supportPolygon = computeSupportPolygon(groundedXZ);
    output_.capturePointSupportDistance =
        signedDistanceToSupportPolygon(output_.capturePointXZ, output_.supportPolygon);

    glm::vec2 supportCenterXZ = comXZ;
    if (!output_.supportPolygon.empty()) {
        glm::vec2 sum{0.0f};
        for (const glm::vec2& v : output_.supportPolygon) {
            sum += v;
        }
        supportCenterXZ = sum / static_cast<float>(output_.supportPolygon.size());
    }

    const glm::vec2 closestSupportXZ = closestPointOnPolygonBoundary(
        output_.capturePointXZ, output_.supportPolygon);

    glm::vec2 fallDirXZ{0.0f};
    if (!output_.supportPolygon.empty() && output_.capturePointSupportDistance > 1e-5f) {
        fallDirXZ = safeNormalize(output_.capturePointXZ - closestSupportXZ, glm::vec2{0.0f});
    }
    if (glm::dot(fallDirXZ, fallDirXZ) <= 1e-8f) {
        fallDirXZ = safeNormalize(
            velXZ, safeNormalize(glm::vec2{input.characterForward.x, input.characterForward.z},
                                 glm::vec2{0.0f, 1.0f}));
    }

    const float upDot = torsoUpUnit.y;
    const float rawComHeight = safeCom.y - effectiveGroundY;
    const float horizSpeed = glm::length(velXZ);
    const float totalSpeed = glm::length(safeVel);
    const float downwardSpeed = -safeVel.y;

    const bool heavyDownwardFall = downwardSpeed >= config_.braceDownwardSpeed;
    const bool airborneFast = (groundedCount == 0) && (totalSpeed >= config_.braceAirborneSpeed);
    const bool beyondStepHighSpeed =
        (horizSpeed >= config_.braceLateralSpeed) &&
        (output_.capturePointSupportDistance > config_.maxStepLength);
    const bool criticalLateralImpact = horizSpeed >= config_.braceCriticalLateralSpeed;
    const bool tiltedAndMoving =
        (upDot < config_.braceTiltUpDot) && (rawComHeight > config_.downedComHeight) &&
        (totalSpeed > 0.75f);
    const bool shouldBrace =
        heavyDownwardFall || airborneFast || beyondStepHighSpeed ||
        criticalLateralImpact || tiltedAndMoving;

    const bool lowCom = rawComHeight <= config_.downedComHeight;
    const bool horizontalCollapsed =
        (upDot < config_.downedTiltUpDot) && (rawComHeight <= config_.downedComHeight + 0.15f);
    const bool settledSpeed =
        (totalSpeed <= config_.downedMaxSpeed) && (downwardSpeed < config_.braceDownwardSpeed);
    const bool isDowned = (lowCom || horizontalCollapsed) && settledSpeed;

    BalanceState nextState = BalanceState::Balanced;
    if (isDowned) {
        nextState = BalanceState::Downed;
    } else if (shouldBrace) {
        nextState = BalanceState::BracingFall;
    } else {
        const float exitThreshold = std::min(config_.stepExitThreshold, config_.stepThreshold);
        const float stepThreshold =
            (output_.state == BalanceState::RecoveringStep)
                ? exitThreshold
                : config_.stepThreshold;
        if (groundedCount == 0 || output_.capturePointSupportDistance > stepThreshold) {
            nextState = BalanceState::RecoveringStep;
        } else {
            nextState = BalanceState::Balanced;
        }
    }

    output_.state = nextState;
    output_.fallDirectionXZ =
        (nextState == BalanceState::Balanced) ? glm::vec2{0.0f} : fallDirXZ;

    const glm::vec2 leftAnchorXZ =
        (leftGroundedCount > 0)
            ? (leftGroundedSum / static_cast<float>(leftGroundedCount))
            : ((leftAllCount > 0) ? (leftAllSum / static_cast<float>(leftAllCount)) : comXZ);
    const glm::vec2 rightAnchorXZ =
        (rightGroundedCount > 0)
            ? (rightGroundedSum / static_cast<float>(rightGroundedCount))
            : ((rightAllCount > 0) ? (rightAllSum / static_cast<float>(rightAllCount)) : comXZ);

    if (nextState == BalanceState::RecoveringStep) {
        FootSide chosen = FootSide::None;

        if (leftGroundedCount == 0 && rightGroundedCount > 0) {
            chosen = FootSide::Left;
        } else if (rightGroundedCount == 0 && leftGroundedCount > 0) {
            chosen = FootSide::Right;
        } else if (leftGroundedCount == 1 && rightAllCount == 0) {
            chosen = FootSide::Right;
        } else if (rightGroundedCount == 1 && leftAllCount == 0) {
            chosen = FootSide::Left;
        } else {
            glm::vec2 rightAxisXZ{1.0f, 0.0f};
            if (leftAllCount > 0 && rightAllCount > 0) {
                rightAxisXZ = safeNormalize(rightAnchorXZ - leftAnchorXZ, glm::vec2{1.0f, 0.0f});
            } else {
                const glm::vec2 fwdXZ = safeNormalize(
                    glm::vec2{input.characterForward.x, input.characterForward.z},
                    glm::vec2{0.0f, 1.0f});
                rightAxisXZ = glm::vec2{-fwdXZ.y, fwdXZ.x};
            }

            const float latOffset = glm::dot(output_.capturePointXZ - supportCenterXZ, rightAxisXZ);
            if (activeRecoveryFoot_ != FootSide::None) {
                // Keep active stepping foot unless capture point swung distinctly past the other side.
                if (activeRecoveryFoot_ == FootSide::Left && latOffset > 0.15f) {
                    chosen = FootSide::Right;
                } else if (activeRecoveryFoot_ == FootSide::Right && latOffset < -0.15f) {
                    chosen = FootSide::Left;
                } else {
                    chosen = activeRecoveryFoot_;
                }
            } else {
                if (latOffset > 0.03f) {
                    chosen = FootSide::Right;
                } else if (latOffset < -0.03f) {
                    chosen = FootSide::Left;
                } else if (leftAllCount > 0 && rightAllCount > 0) {
                    const float leftProj = glm::dot(leftAnchorXZ - supportCenterXZ, fallDirXZ);
                    const float rightProj = glm::dot(rightAnchorXZ - supportCenterXZ, fallDirXZ);
                    if (std::abs(leftProj - rightProj) > 0.08f) {
                        chosen = (leftProj < rightProj) ? FootSide::Left : FootSide::Right;
                    } else {
                        chosen = (lastRecoveryFoot_ == FootSide::Right)
                                     ? FootSide::Left
                                     : FootSide::Right;
                    }
                } else {
                    chosen = (lastRecoveryFoot_ == FootSide::Right)
                                 ? FootSide::Left
                                 : FootSide::Right;
                }
            }
        }

        activeRecoveryFoot_ = chosen;
        lastRecoveryFoot_ = chosen;
        output_.recoveryFoot = chosen;

        glm::vec2 stepXZ = output_.capturePointXZ + fallDirXZ * config_.stepOvershoot;
        if (chosen != FootSide::None && leftAllCount > 0 && rightAllCount > 0) {
            const glm::vec2& stanceXZ = (chosen == FootSide::Left) ? rightAnchorXZ : leftAnchorXZ;
            const glm::vec2& stepFootXZ = (chosen == FootSide::Left) ? leftAnchorXZ : rightAnchorXZ;
            glm::vec2 perp{-fallDirXZ.y, fallDirXZ.x};
            const float sepAlongPerp = glm::dot(stepFootXZ - stanceXZ, perp);
            if (std::abs(sepAlongPerp) > 1e-3f) {
                if (sepAlongPerp < 0.0f) {
                    perp = -perp;
                }
                const float currentLat = glm::dot(stepXZ - stanceXZ, perp);
                if (currentLat < config_.minFootSeparation) {
                    stepXZ += perp * (config_.minFootSeparation - currentLat);
                }
            }
        }

        const glm::vec2 fromCom = stepXZ - comXZ;
        const float reach = glm::length(fromCom);
        if (reach > config_.maxStepLength && config_.maxStepLength > 0.0f) {
            stepXZ = comXZ + fromCom * (config_.maxStepLength / reach);
        }
        output_.recoveryStepTarget = glm::vec3{stepXZ.x, effectiveGroundY, stepXZ.y};
    } else {
        activeRecoveryFoot_ = FootSide::None;
        output_.recoveryFoot = FootSide::None;
        output_.recoveryStepTarget = glm::vec3{0.0f};
    }

    const glm::vec3 braceOrigin =
        (input.hasChestPosition && isFinite(input.chestPosition))
            ? input.chestPosition
            : (safeCom + torsoUpUnit * 0.25f);
    glm::vec3 braceDir{0.0f, -1.0f, 0.0f};
    if (totalSpeed > 0.2f) {
        braceDir = safeVel / totalSpeed;
    } else {
        braceDir = safeNormalize(
            glm::vec3{fallDirXZ.x, -0.5f, fallDirXZ.y}, glm::vec3{0.0f, -1.0f, 0.0f});
    }

    output_.braceActive = (nextState == BalanceState::BracingFall);
    output_.braceRayOrigin = braceOrigin;
    output_.braceRayDir = braceDir;
    output_.braceRayLength = config_.braceRayLength;
    hasLastOrigin_ = true;

    float targetStiffness = config_.balancedStiffness;
    switch (nextState) {
        case BalanceState::Balanced:
            targetStiffness = config_.balancedStiffness;
            break;
        case BalanceState::RecoveringStep:
            targetStiffness = config_.recoveryStiffness;
            break;
        case BalanceState::BracingFall:
            targetStiffness = config_.bracingStiffness;
            break;
        case BalanceState::Downed:
            targetStiffness = config_.downedStiffness;
            break;
    }

    if (config_.stiffnessBlendRate > 0.0f && isFinite(dt) && dt > 0.0f) {
        const float alpha = 1.0f - std::exp(-config_.stiffnessBlendRate * dt);
        output_.stiffnessMultiplier += (targetStiffness - output_.stiffnessMultiplier) * alpha;
    } else {
        output_.stiffnessMultiplier = targetStiffness;
    }

    return output_;
}

HandBraceTarget BalanceController::computeHandBraceTarget(
    const glm::vec3& hitPoint,
    const glm::vec3& hitNormal,
    float armReach) const noexcept {
    const float effectiveReach =
        (isFinite(armReach) && armReach > 0.0f) ? armReach : config_.defaultArmReach;
    if (hasLastOrigin_) {
        return computeHandBraceTargetFrom(
            output_.braceRayOrigin,
            hitPoint,
            hitNormal,
            effectiveReach,
            config_.handSurfaceOffset);
    }

    const glm::vec3 safeHit = isFinite(hitPoint) ? hitPoint : glm::vec3{0.0f};
    const glm::vec3 n = safeNormalize(hitNormal, glm::vec3{0.0f, 1.0f, 0.0f});
    const float cushion = std::clamp(config_.handSurfaceOffset, 0.0f, effectiveReach * 0.25f);
    HandBraceTarget target;
    target.surfaceNormal = n;
    target.position = safeHit + n * cushion;
    target.rotation = buildSurfaceAlignedHandRotation(n, glm::vec3{0.0f, 1.0f, 0.0f});
    target.withinReach = true;
    return target;
}

HandBraceTarget BalanceController::computeHandBraceTargetFrom(
    const glm::vec3& shoulderPos,
    const glm::vec3& hitPoint,
    const glm::vec3& hitNormal,
    float armReach,
    float surfaceOffset,
    const glm::vec3& preferredUp) noexcept {
    const glm::vec3 safeShoulder = isFinite(shoulderPos) ? shoulderPos : glm::vec3{0.0f};
    const glm::vec3 safeHit = isFinite(hitPoint) ? hitPoint : safeShoulder;
    const glm::vec3 n = safeNormalize(hitNormal, glm::vec3{0.0f, 1.0f, 0.0f});
    const float reach = isFinite(armReach) ? std::max(0.0f, armReach) : 0.0f;
    const float offset =
        isFinite(surfaceOffset) ? std::clamp(surfaceOffset, 0.0f, reach * 0.25f) : 0.0f;

    const glm::vec3 surfaceTarget = safeHit + n * offset;
    const glm::vec3 delta = surfaceTarget - safeShoulder;
    const float dist = glm::length(delta);

    HandBraceTarget result;
    result.surfaceNormal = n;
    result.rotation = buildSurfaceAlignedHandRotation(n, preferredUp);

    if (dist <= reach + 1e-6f) {
        result.position = surfaceTarget;
        result.withinReach = true;
    } else {
        const glm::vec3 dir = safeNormalize(delta, n);
        result.position = safeShoulder + dir * reach;
        result.withinReach = false;
    }
    return result;
}

} // namespace lumora
