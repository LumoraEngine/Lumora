#include "runtime/ProceduralIk.hpp"

#include <glm/geometric.hpp>
#include <glm/mat3x3.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace lumora {

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kEps = 1e-6f;
constexpr float kEpsSq = kEps * kEps;
constexpr float kReachTolerance = 1e-3f;

[[nodiscard]] bool isFinite(float v) noexcept {
    return std::isfinite(v);
}

[[nodiscard]] bool isFinite(const glm::vec3& v) noexcept {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

[[nodiscard]] glm::vec3 safeNormalize(const glm::vec3& v, const glm::vec3& fallback) noexcept {
    if (!isFinite(v)) {
        return fallback;
    }
    const float lenSq = glm::dot(v, v);
    if (lenSq <= kEpsSq) {
        return fallback;
    }
    return v / std::sqrt(lenSq);
}

[[nodiscard]] glm::vec3 anyPerpendicular(const glm::vec3& unitAxis) noexcept {
    const glm::vec3 candidate =
        std::abs(unitAxis.y) < 0.9f ? glm::vec3{0.0f, 1.0f, 0.0f} : glm::vec3{1.0f, 0.0f, 0.0f};
    return safeNormalize(glm::cross(unitAxis, candidate), glm::vec3{1.0f, 0.0f, 0.0f});
}

[[nodiscard]] glm::quat shortestArc(glm::quat q) noexcept {
    const float lenSq = glm::dot(q, q);
    if (!std::isfinite(lenSq) || lenSq <= kEpsSq) {
        return glm::quat{1.0f, 0.0f, 0.0f, 0.0f};
    }
    q /= std::sqrt(lenSq);
    if (q.w < 0.0f) {
        q = -q;
    }
    return q;
}

[[nodiscard]] glm::quat rotationBetween(const glm::vec3& fromUnit,
                                        const glm::vec3& toUnit,
                                        const glm::vec3& fallbackAxisHint) noexcept {
    const float d = std::clamp(glm::dot(fromUnit, toUnit), -1.0f, 1.0f);
    if (d >= 1.0f - 1e-6f) {
        return glm::quat{1.0f, 0.0f, 0.0f, 0.0f};
    }
    if (d <= -1.0f + 1e-6f) {
        const glm::vec3 ortho = fallbackAxisHint - fromUnit * glm::dot(fallbackAxisHint, fromUnit);
        const glm::vec3 axis = safeNormalize(ortho, anyPerpendicular(fromUnit));
        return shortestArc(glm::angleAxis(kPi, axis));
    }
    const glm::vec3 c = glm::cross(fromUnit, toUnit);
    const float s = std::sqrt((1.0f + d) * 2.0f);
    const float invS = 1.0f / s;
    return shortestArc(glm::quat{0.5f * s, c.x * invS, c.y * invS, c.z * invS});
}

[[nodiscard]] glm::quat frameRotation(const glm::vec3& fromDir,
                                      const glm::vec3& fromNormal,
                                      const glm::vec3& toDir,
                                      const glm::vec3& toNormal) noexcept {
    const glm::vec3 fromSide = safeNormalize(glm::cross(fromDir, fromNormal), anyPerpendicular(fromDir));
    const glm::vec3 toSide = safeNormalize(glm::cross(toDir, toNormal), anyPerpendicular(toDir));
    const glm::mat3 m0{fromDir, fromNormal, fromSide};
    const glm::mat3 m1{toDir, toNormal, toSide};
    return shortestArc(glm::quat_cast(m1 * glm::transpose(m0)));
}

[[nodiscard]] glm::vec3 clampDirectionToCone(const glm::vec3& dir,
                                             const glm::vec3& refAxis,
                                             float maxAngleRad) noexcept {
    if (maxAngleRad >= kPi - 1e-5f) {
        return dir;
    }
    if (maxAngleRad <= 1e-6f) {
        return refAxis;
    }
    const float cosAngle = std::clamp(glm::dot(refAxis, dir), -1.0f, 1.0f);
    const float angle = std::acos(cosAngle);
    if (angle <= maxAngleRad) {
        return dir;
    }
    const glm::vec3 rotAxis = safeNormalize(glm::cross(refAxis, dir), anyPerpendicular(refAxis));
    const glm::vec3 ortho = glm::cross(rotAxis, refAxis);
    return safeNormalize(refAxis * std::cos(maxAngleRad) + ortho * std::sin(maxAngleRad), refAxis);
}

[[nodiscard]] float resolveJointBendLimit(std::span<const float> limits,
                                          std::size_t jointIdx,
                                          std::size_t jointCount) noexcept {
    if (limits.empty() || jointCount < 2) {
        return kPi;
    }
    float raw = kPi;
    if (limits.size() >= jointCount - 1) {
        raw = limits[jointIdx];
    } else if (limits.size() == jointCount - 2) {
        if (jointIdx == 0) {
            return kPi;
        }
        raw = limits[jointIdx - 1];
    } else {
        if (jointIdx >= limits.size()) {
            return kPi;
        }
        raw = limits[jointIdx];
    }
    if (!isFinite(raw) || raw < 0.0f || raw >= kPi) {
        return kPi;
    }
    return raw;
}

} // namespace

TwoBoneIkResult solveTwoBoneIk(const TwoBoneIkInput& input) {
    TwoBoneIkResult result;
    if (!isFinite(input.rootPos) || !isFinite(input.midPos) || !isFinite(input.endPos) ||
        !isFinite(input.targetPos) || !isFinite(input.polePos)) {
        result.rootPos = isFinite(input.rootPos) ? input.rootPos : glm::vec3{0.0f};
        result.midPos = isFinite(input.midPos) ? input.midPos : result.rootPos;
        result.endPos = isFinite(input.endPos) ? input.endPos : result.midPos;
        return result;
    }

    result.rootPos = input.rootPos;
    result.midPos = input.midPos;
    result.endPos = input.endPos;

    const float weight = isFinite(input.weight) ? std::clamp(input.weight, 0.0f, 1.0f) : 0.0f;
    if (weight <= 0.0f) {
        result.targetReached = glm::length(result.endPos - input.targetPos) <= kReachTolerance;
        return result;
    }

    const glm::vec3 uOrig = input.midPos - input.rootPos;
    const glm::vec3 vOrig = input.endPos - input.midPos;
    const float len1 = glm::length(uOrig);
    const float len2 = glm::length(vOrig);

    if (len1 <= kEps && len2 <= kEps) {
        result.midPos = input.rootPos;
        result.endPos = input.rootPos;
        result.targetReached = glm::length(input.targetPos - input.rootPos) <= kReachTolerance;
        return result;
    }

    const glm::quat identity{1.0f, 0.0f, 0.0f, 0.0f};

    // Degenerate single-bone fallbacks when one segment has zero length.
    if (len1 <= kEps) {
        const glm::vec3 toTarget = input.targetPos - input.rootPos;
        const float rawDist = glm::length(toTarget);
        const glm::vec3 v0 = safeNormalize(vOrig, glm::vec3{0.0f, 1.0f, 0.0f});
        const glm::vec3 targetDir = (rawDist > kEps) ? (toTarget / rawDist) : v0;
        const glm::quat fullMidDelta = rotationBetween(v0, targetDir, anyPerpendicular(v0));
        result.midDelta =
            (weight >= 1.0f - kEps) ? fullMidDelta : shortestArc(glm::slerp(identity, fullMidDelta, weight));
        result.midPos = input.rootPos;
        result.endPos = input.rootPos + result.midDelta * vOrig;
        result.clampedToMaxReach = rawDist > len2 + kEps;
        result.targetReached = glm::length(result.endPos - input.targetPos) <= kReachTolerance;
        return result;
    }

    if (len2 <= kEps) {
        const glm::vec3 toTarget = input.targetPos - input.rootPos;
        const float rawDist = glm::length(toTarget);
        const glm::vec3 u0 = uOrig / len1;
        const glm::vec3 targetDir = (rawDist > kEps) ? (toTarget / rawDist) : u0;
        const glm::quat fullRootDelta = rotationBetween(u0, targetDir, anyPerpendicular(u0));
        result.rootDelta =
            (weight >= 1.0f - kEps) ? fullRootDelta : shortestArc(glm::slerp(identity, fullRootDelta, weight));
        result.midPos = input.rootPos + result.rootDelta * uOrig;
        result.endPos = result.midPos;
        result.clampedToMaxReach = rawDist > len1 + kEps;
        result.targetReached = glm::length(result.endPos - input.targetPos) <= kReachTolerance;
        return result;
    }

    const float minReach = std::abs(len1 - len2);
    const float maxReach = len1 + len2;
    const glm::vec3 u0 = uOrig / len1;
    const glm::vec3 v0 = vOrig / len2;

    const glm::vec3 toTarget = input.targetPos - input.rootPos;
    const float rawDist = glm::length(toTarget);
    const glm::vec3 fallbackDir = safeNormalize(input.endPos - input.rootPos, u0);
    const glm::vec3 targetDir = (rawDist > kEps) ? (toTarget / rawDist) : fallbackDir;

    const float softRatio = isFinite(input.softStretchRatio) ? std::clamp(input.softStretchRatio, 0.0f, 1.0f) : 1.0f;
    float effectiveDist = rawDist;
    if (softRatio < 1.0f) {
        const float softStart = minReach + (maxReach - minReach) * softRatio;
        const float softRange = maxReach - softStart;
        if (rawDist > softStart && softRange > kEps) {
            const float excess = rawDist - softStart;
            effectiveDist = softStart + softRange * (1.0f - std::exp(-excess / softRange));
        }
    }
    effectiveDist = std::clamp(effectiveDist, minReach, maxReach);
    result.clampedToMaxReach = rawDist > maxReach + kEps;

    // Bend direction lies in the plane through root->target and polePos, on the polePos side.
    const glm::vec3 toPole = input.polePos - input.rootPos;
    const glm::vec3 poleOrtho = toPole - targetDir * glm::dot(toPole, targetDir);
    glm::vec3 bendDir{0.0f};
    if (glm::dot(poleOrtho, poleOrtho) > kEpsSq) {
        bendDir = glm::normalize(poleOrtho);
    } else {
        const glm::vec3 midOrtho = uOrig - targetDir * glm::dot(uOrig, targetDir);
        bendDir = (glm::dot(midOrtho, midOrtho) > kEpsSq) ? glm::normalize(midOrtho) : anyPerpendicular(targetDir);
    }

    glm::vec3 solvedMid{0.0f};
    glm::vec3 solvedEnd{0.0f};
    if (effectiveDist <= kEps) {
        solvedMid = input.rootPos + bendDir * len1;
        solvedEnd = input.rootPos;
    } else {
        const float cosAlpha = std::clamp(
            (len1 * len1 + effectiveDist * effectiveDist - len2 * len2) / (2.0f * len1 * effectiveDist),
            -1.0f,
            1.0f);
        const float sinAlpha = std::sqrt(std::max(0.0f, 1.0f - cosAlpha * cosAlpha));
        solvedMid = input.rootPos + targetDir * (len1 * cosAlpha) + bendDir * (len1 * sinAlpha);
        solvedEnd = input.rootPos + targetDir * effectiveDist;
    }

    const glm::vec3 u1 = safeNormalize(solvedMid - input.rootPos, targetDir);
    const glm::vec3 v1 = safeNormalize(solvedEnd - solvedMid, targetDir);

    const glm::vec3 cross0 = glm::cross(u0, v0);
    const glm::vec3 cross1 = glm::cross(u1, v1);
    const bool hasPlane0 = glm::dot(cross0, cross0) > kEpsSq;
    const bool hasPlane1 = glm::dot(cross1, cross1) > kEpsSq;
    // Note sign: cross(u1, v1) aligns with cross(bendDir, targetDir) when bending toward bendDir.
    const glm::vec3 poleHingeAxis = safeNormalize(glm::cross(bendDir, targetDir), anyPerpendicular(targetDir));

    glm::quat fullRootDelta = identity;
    glm::quat fullMidDelta = identity;
    if (hasPlane0) {
        const glm::vec3 b0 = glm::normalize(cross0);
        const glm::vec3 b1 = hasPlane1 ? glm::normalize(cross1) : poleHingeAxis;
        fullRootDelta = frameRotation(u0, b0, u1, b1);
        fullMidDelta = frameRotation(v0, b0, v1, b1);
    } else {
        fullRootDelta = rotationBetween(u0, u1, poleHingeAxis);
        fullMidDelta = rotationBetween(v0, v1, poleHingeAxis);
    }

    if (weight >= 1.0f - kEps) {
        result.rootDelta = fullRootDelta;
        result.midDelta = fullMidDelta;
        result.midPos = solvedMid;
        result.endPos = solvedEnd;
    } else {
        result.rootDelta = shortestArc(glm::slerp(identity, fullRootDelta, weight));
        result.midDelta = shortestArc(glm::slerp(identity, fullMidDelta, weight));
        result.midPos = input.rootPos + result.rootDelta * uOrig;
        result.endPos = result.midPos + result.midDelta * vOrig;
    }

    result.targetReached = glm::length(result.endPos - input.targetPos) <= kReachTolerance;
    return result;
}

bool solveFabrik(std::span<glm::vec3> joints,
                 const glm::vec3& target,
                 std::span<const float> maxBendAngleRad,
                 int maxIterations,
                 float tolerance) {
    const std::size_t n = joints.size();
    if (n == 0 || !isFinite(target)) {
        return false;
    }
    for (const glm::vec3& j : joints) {
        if (!isFinite(j)) {
            return false;
        }
    }
    const float tol = isFinite(tolerance) ? std::max(tolerance, 1e-6f) : 1e-4f;
    if (n == 1) {
        return glm::length(joints[0] - target) <= tol;
    }

    const int iterations = std::max(maxIterations, 1);
    const glm::vec3 rootOrigin = joints[0];

    std::vector<float> lengths(n - 1, 0.0f);
    std::vector<glm::vec3> restDirs(n - 1, glm::vec3{0.0f, 1.0f, 0.0f});
    std::vector<float> limits(n - 1, kPi);
    float totalLength = 0.0f;
    for (std::size_t i = 0; i + 1 < n; ++i) {
        const glm::vec3 seg = joints[i + 1] - joints[i];
        const float segLen = glm::length(seg);
        lengths[i] = segLen;
        totalLength += segLen;
        if (segLen > kEps) {
            restDirs[i] = seg / segLen;
        } else if (i > 0) {
            restDirs[i] = restDirs[i - 1];
        }
        limits[i] = resolveJointBendLimit(maxBendAngleRad, i, n);
    }

    if (totalLength <= kEps) {
        for (std::size_t i = 1; i < n; ++i) {
            joints[i] = rootOrigin;
        }
        return glm::length(rootOrigin - target) <= tol;
    }

    // Check if unconstrained and target is beyond total reach -> exact full stretch.
    bool anyConstrained = false;
    for (float lim : limits) {
        if (lim < kPi - 1e-5f) {
            anyConstrained = true;
            break;
        }
    }

    const glm::vec3 toTarget = target - rootOrigin;
    const float distToTarget = glm::length(toTarget);
    if (!anyConstrained && distToTarget >= totalLength) {
        const glm::vec3 reachDir = (distToTarget > kEps) ? (toTarget / distToTarget) : restDirs[0];
        joints[0] = rootOrigin;
        for (std::size_t i = 0; i + 1 < n; ++i) {
            joints[i + 1] = joints[i] + reachDir * lengths[i];
        }
        return glm::length(joints.back() - target) <= tol;
    }

    std::vector<glm::vec3> bestJoints(joints.begin(), joints.end());
    float bestError = std::numeric_limits<float>::max();
    float prevError = glm::length(joints.back() - target);
    const float collinearThreshSq = (1e-4f * totalLength) * (1e-4f * totalLength);
    const glm::vec3 targetAxis = (distToTarget > kEps) ? (toTarget / distToTarget) : restDirs[0];

    for (int iter = 0; iter < iterations; ++iter) {
        // If the chain is stuck on the target line in a collinear fold against a reachable target,
        // apply a small lateral perturbation to interior joints so FABRIK can bend out of singularity.
        if (iter > 0 && prevError > tol && distToTarget < totalLength - tol &&
            std::abs(glm::length(joints.back() - target) - prevError) < 1e-6f * totalLength) {
            bool collinear = true;
            for (std::size_t i = 1; i < n; ++i) {
                const glm::vec3 rel = joints[i] - rootOrigin;
                const glm::vec3 perp = rel - targetAxis * glm::dot(rel, targetAxis);
                if (glm::dot(perp, perp) > collinearThreshSq) {
                    collinear = false;
                    break;
                }
            }
            if (collinear) {
                const glm::vec3 nudge = anyPerpendicular(targetAxis) * (0.01f * totalLength);
                for (std::size_t i = 1; i + 1 < n; ++i) {
                    joints[i] += nudge * static_cast<float>((i % 2 == 1) ? 1.0f : -0.5f);
                }
            }
        }

        // Forward pass: pull from target back to root preserving segment lengths.
        joints[n - 1] = target;
        for (std::size_t k = n - 1; k-- > 0;) {
            const float segLen = lengths[k];
            if (segLen <= kEps) {
                joints[k] = joints[k + 1];
                continue;
            }
            const glm::vec3 backDir = safeNormalize(joints[k] - joints[k + 1], -restDirs[k]);
            joints[k] = joints[k + 1] + backDir * segLen;
        }

        // Backward pass: anchor root and step forward, enforcing segment lengths & cone limits.
        joints[0] = rootOrigin;
        glm::vec3 prevBoneDir = restDirs[0];
        for (std::size_t i = 0; i + 1 < n; ++i) {
            const float segLen = lengths[i];
            if (segLen <= kEps) {
                joints[i + 1] = joints[i];
                continue;
            }
            const glm::vec3 rawDir = safeNormalize(joints[i + 1] - joints[i], prevBoneDir);
            const glm::vec3 refDir = (i == 0) ? restDirs[0] : prevBoneDir;
            const glm::vec3 clampedDir = clampDirectionToCone(rawDir, refDir, limits[i]);
            joints[i + 1] = joints[i] + clampedDir * segLen;
            prevBoneDir = clampedDir;
        }

        const float err = glm::length(joints.back() - target);
        prevError = err;
        if (err < bestError) {
            bestError = err;
            bestJoints.assign(joints.begin(), joints.end());
        }
        if (err <= tol) {
            return true;
        }
    }

    std::copy(bestJoints.begin(), bestJoints.end(), joints.begin());
    return bestError <= tol;
}

FootPlacementResult solveFootPlacement(const FootPlacementInput& input) {
    FootPlacementResult result;

    const glm::vec3 up = safeNormalize(input.upAxis, glm::vec3{0.0f, 1.0f, 0.0f});
    const float pelvisH = isFinite(input.pelvisHeight) ? input.pelvisHeight : 0.0f;
    const float maxDrop = isFinite(input.maxPelvisDrop) ? std::max(0.0f, input.maxPelvisDrop) : 0.0f;
    const float maxLift = isFinite(input.maxPelvisLift) ? std::max(0.0f, input.maxPelvisLift) : 0.0f;
    const float soleOffset = isFinite(input.footSoleOffset) ? input.footSoleOffset : 0.0f;
    const float maxTilt = isFinite(input.maxAnkleTiltRad) ? std::clamp(input.maxAnkleTiltRad, 0.0f, kPi * 0.5f) : kPi * 0.25f;

    const glm::vec3 nominalLeft = isFinite(input.leftFootPos) ? input.leftFootPos : glm::vec3{0.0f};
    const glm::vec3 nominalRight = isFinite(input.rightFootPos) ? input.rightFootPos : glm::vec3{0.0f};

    result.leftFootTarget = nominalLeft;
    result.rightFootTarget = nominalRight;
    result.solvedPelvisHeight = pelvisH;

    auto solveAnkleTilt = [&](const glm::vec3& rawNormal) -> glm::quat {
        glm::vec3 n = safeNormalize(rawNormal, up);
        if (glm::dot(n, up) < 0.0f) {
            n = -n;
        }
        const float cosTheta = std::clamp(glm::dot(up, n), -1.0f, 1.0f);
        const float theta = std::min(std::acos(cosTheta), maxTilt);
        if (theta <= 1e-6f) {
            return glm::quat{1.0f, 0.0f, 0.0f, 0.0f};
        }
        const glm::vec3 axis = safeNormalize(glm::cross(up, n), anyPerpendicular(up));
        return shortestArc(glm::angleAxis(theta, axis));
    };

    bool hasLeftDelta = false;
    bool hasRightDelta = false;
    float leftDelta = 0.0f;
    float rightDelta = 0.0f;

    if (input.leftHit.hit && isFinite(input.leftHit.hitPoint)) {
        result.leftPlanted = true;
        result.leftAnkleRotation = solveAnkleTilt(input.leftHit.hitNormal);
        const glm::vec3 plantedUp = result.leftAnkleRotation * up;
        result.leftFootTarget = input.leftHit.hitPoint + plantedUp * soleOffset;
        leftDelta = glm::dot(result.leftFootTarget - nominalLeft, up);
        hasLeftDelta = true;
    }

    if (input.rightHit.hit && isFinite(input.rightHit.hitPoint)) {
        result.rightPlanted = true;
        result.rightAnkleRotation = solveAnkleTilt(input.rightHit.hitNormal);
        const glm::vec3 plantedUp = result.rightAnkleRotation * up;
        result.rightFootTarget = input.rightHit.hitPoint + plantedUp * soleOffset;
        rightDelta = glm::dot(result.rightFootTarget - nominalRight, up);
        hasRightDelta = true;
    }

    float desiredOffset = 0.0f;
    if (hasLeftDelta && hasRightDelta) {
        desiredOffset = std::min(leftDelta, rightDelta);
    } else if (hasLeftDelta) {
        desiredOffset = leftDelta;
    } else if (hasRightDelta) {
        desiredOffset = rightDelta;
    }

    result.pelvisOffsetY = std::clamp(desiredOffset, -maxDrop, maxLift);
    result.solvedPelvisHeight = pelvisH + result.pelvisOffsetY;
    return result;
}

FootPlacementResult solveFootPlacement(float pelvisHeight,
                                       const glm::vec3& leftFootPos,
                                       const glm::vec3& rightFootPos,
                                       const IkGroundHit& leftHit,
                                       const IkGroundHit& rightHit,
                                       float maxPelvisDrop,
                                       float footSoleOffset) {
    FootPlacementInput input;
    input.pelvisHeight = pelvisHeight;
    input.leftFootPos = leftFootPos;
    input.rightFootPos = rightFootPos;
    input.leftHit = leftHit;
    input.rightHit = rightHit;
    input.maxPelvisDrop = maxPelvisDrop;
    input.footSoleOffset = footSoleOffset;
    return solveFootPlacement(input);
}

glm::quat solveLookAt(const glm::vec3& forward,
                      const glm::vec3& up,
                      const glm::vec3& targetDir,
                      float maxYawRad,
                      float maxPitchRad,
                      float weight) {
    const glm::quat identity{1.0f, 0.0f, 0.0f, 0.0f};
    if (!isFinite(forward) || !isFinite(up) || !isFinite(targetDir)) {
        return identity;
    }

    const float clampedWeight = isFinite(weight) ? std::clamp(weight, 0.0f, 1.0f) : 0.0f;
    if (clampedWeight <= 0.0f) {
        return identity;
    }

    const glm::vec3 fwd = safeNormalize(forward, glm::vec3{0.0f});
    if (glm::dot(fwd, fwd) <= kEpsSq) {
        return identity;
    }
    const glm::vec3 tgt = safeNormalize(targetDir, glm::vec3{0.0f});
    if (glm::dot(tgt, tgt) <= kEpsSq) {
        return identity;
    }

    const glm::vec3 upOrtho = up - fwd * glm::dot(up, fwd);
    const glm::vec3 upApprox =
        (glm::dot(upOrtho, upOrtho) > kEpsSq) ? glm::normalize(upOrtho) : anyPerpendicular(fwd);
    const glm::vec3 right = safeNormalize(glm::cross(fwd, upApprox), anyPerpendicular(fwd));
    const glm::vec3 upUnit = safeNormalize(glm::cross(right, fwd), upApprox);

    const float x = glm::dot(tgt, right);
    const float y = glm::dot(tgt, upUnit);
    const float z = glm::dot(tgt, fwd);
    const float horiz = std::sqrt(x * x + z * z);

    const float rawYaw = (horiz > kEps) ? std::atan2(-x, z) : 0.0f;
    const float rawPitch = std::atan2(y, horiz);

    const float yawLimit = isFinite(maxYawRad) ? std::max(0.0f, maxYawRad) : 0.0f;
    const float pitchLimit = isFinite(maxPitchRad) ? std::max(0.0f, maxPitchRad) : 0.0f;

    const float yaw = std::clamp(rawYaw, -yawLimit, yawLimit);
    const float pitch = std::clamp(rawPitch, -pitchLimit, pitchLimit);

    const glm::quat qYaw = glm::angleAxis(yaw, upUnit);
    const glm::quat qPitch = glm::angleAxis(pitch, right);
    const glm::quat fullRot = shortestArc(qYaw * qPitch);

    if (clampedWeight >= 1.0f - kEps) {
        return fullRot;
    }
    return shortestArc(glm::slerp(identity, fullRot, clampedWeight));
}

} // namespace lumora
