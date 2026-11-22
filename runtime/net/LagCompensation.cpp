#include "runtime/net/LagCompensation.hpp"

#include <glm/common.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

namespace lumora::net {

namespace {

constexpr double kTimestampEpsilon = 1e-9;
constexpr double kSegmentLenSqEpsilon = 1e-12;
constexpr float kHitTieEpsilon = 1e-4f;

[[nodiscard]] bool isFiniteVec3(const glm::vec3& v) noexcept {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

[[nodiscard]] bool isFiniteQuat(const glm::quat& q) noexcept {
    return std::isfinite(q.w) && std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z);
}

[[nodiscard]] glm::quat normalizeSafe(const glm::quat& q) noexcept {
    if (!isFiniteQuat(q)) {
        return glm::quat{1.0f, 0.0f, 0.0f, 0.0f};
    }
    const float lenSq = glm::dot(q, q);
    if (lenSq <= 1e-12f) {
        return glm::quat{1.0f, 0.0f, 0.0f, 0.0f};
    }
    return glm::normalize(q);
}

[[nodiscard]] glm::dvec3 safeNormal(const glm::dvec3& v, const glm::dvec3& fallback) noexcept {
    const double lenSq = glm::dot(v, v);
    if (lenSq <= 1e-24) {
        return fallback;
    }
    return v / std::sqrt(lenSq);
}

[[nodiscard]] constexpr int partPriority(HitboxPart part) noexcept {
    switch (part) {
    case HitboxPart::Head:
        return 3;
    case HitboxPart::Body:
        return 2;
    case HitboxPart::Limb:
        return 1;
    }
    return 0;
}

struct DoubleHit {
    double t = 0.0;
    glm::dvec3 normal{0.0};
};

[[nodiscard]] std::optional<DoubleHit> intersectSphereEntry(
    const glm::dvec3& origin,
    const glm::dvec3& unitDir,
    double maxDistance,
    const glm::dvec3& center,
    double radiusSq) noexcept {
    const glm::dvec3 offset = origin - center;
    const double c = glm::dot(offset, offset) - radiusSq;
    if (c <= 0.0) {
        return DoubleHit{0.0, safeNormal(offset, -unitDir)};
    }
    const double b = glm::dot(offset, unitDir);
    if (b >= 0.0) {
        return std::nullopt;
    }
    const double disc = b * b - c;
    if (disc < 0.0) {
        return std::nullopt;
    }
    const double q = -b + std::sqrt(disc);
    const double t = c / q;
    if (t < 0.0 || t > maxDistance) {
        return std::nullopt;
    }
    const glm::dvec3 hitPoint = origin + t * unitDir;
    return DoubleHit{t, safeNormal(hitPoint - center, -unitDir)};
}

} // namespace

std::optional<CapsuleRayHit> rayCapsuleIntersect(
    const glm::vec3& origin,
    const glm::vec3& direction,
    float maxDistance,
    const glm::vec3& capsuleA,
    const glm::vec3& capsuleB,
    float radius) noexcept {
    if (!isFiniteVec3(origin) || !isFiniteVec3(direction) ||
        !isFiniteVec3(capsuleA) || !isFiniteVec3(capsuleB) ||
        !std::isfinite(maxDistance) || !std::isfinite(radius) ||
        maxDistance <= 0.0f || radius <= 0.0f) {
        return std::nullopt;
    }

    const glm::dvec3 o{origin};
    const glm::dvec3 rawDir{direction};
    const double dirLenSq = glm::dot(rawDir, rawDir);
    if (dirLenSq <= 1e-12) {
        return std::nullopt;
    }
    const glm::dvec3 d = rawDir / std::sqrt(dirLenSq);
    const double maxDist = static_cast<double>(maxDistance);
    const double r = static_cast<double>(radius);
    const double rSq = r * r;

    const glm::dvec3 a{capsuleA};
    const glm::dvec3 b{capsuleB};
    const glm::dvec3 seg = b - a;
    const double segLenSq = glm::dot(seg, seg);

    // Degenerate capsule reduces to a sphere centered at the midpoint.
    if (segLenSq <= kSegmentLenSqEpsilon) {
        const auto hit = intersectSphereEntry(o, d, maxDist, 0.5 * (a + b), rSq);
        if (!hit) {
            return std::nullopt;
        }
        return CapsuleRayHit{
            static_cast<float>(hit->t),
            glm::vec3(o + hit->t * d),
            glm::vec3(hit->normal),
        };
    }

    const double segLen = std::sqrt(segLenSq);
    const glm::dvec3 axis = seg / segLen;
    const glm::dvec3 relOrigin = o - a;
    const double originAxial = glm::dot(relOrigin, axis);

    // Early out when the ray origin starts inside or on the capsule surface.
    const double clampedAxial = std::clamp(originAxial, 0.0, segLen);
    const glm::dvec3 closestOnSeg = a + clampedAxial * axis;
    const glm::dvec3 radialOrigin = o - closestOnSeg;
    if (glm::dot(radialOrigin, radialOrigin) <= rSq) {
        return CapsuleRayHit{
            0.0f,
            origin,
            glm::vec3(safeNormal(radialOrigin, -d)),
        };
    }

    // Outside the solid capsule (the union of Sphere(a, r), Sphere(b, r), and FiniteCylinder(a, b, r)),
    // the earliest entry intersection across those three shapes is the exact capsule entry point.
    bool hasHit = false;
    double bestT = maxDist;
    glm::dvec3 bestNormal{0.0};

    if (const auto hitA = intersectSphereEntry(o, d, bestT, a, rSq)) {
        hasHit = true;
        bestT = hitA->t;
        bestNormal = hitA->normal;
    }
    if (const auto hitB = intersectSphereEntry(o, d, bestT, b, rSq)) {
        if (!hasHit || hitB->t < bestT) {
            hasHit = true;
            bestT = hitB->t;
            bestNormal = hitB->normal;
        }
    }

    const double dirAxial = glm::dot(d, axis);
    const glm::dvec3 dirPerp = d - dirAxial * axis;
    const glm::dvec3 originPerp = relOrigin - originAxial * axis;
    const double quadA = glm::dot(dirPerp, dirPerp);
    if (quadA > kSegmentLenSqEpsilon) {
        const double quadB = glm::dot(originPerp, dirPerp);
        const double quadC = glm::dot(originPerp, originPerp) - rSq;
        // Rays starting inside the infinite cylinder (quadC <= 0) while outside the capsule must
        // enter through Sphere(a, r) or Sphere(b, r), already handled above.
        if (quadC > 0.0 && quadB < 0.0) {
            const double disc = quadB * quadB - quadA * quadC;
            if (disc >= 0.0) {
                const double q = -quadB + std::sqrt(disc);
                const double tCyl = quadC / q;
                if (tCyl >= 0.0 && (!hasHit ? (tCyl <= bestT) : (tCyl < bestT))) {
                    const double hitAxial = originAxial + tCyl * dirAxial;
                    if (hitAxial >= -kTimestampEpsilon && hitAxial <= segLen + kTimestampEpsilon) {
                        const double clampedHitAxial = std::clamp(hitAxial, 0.0, segLen);
                        const glm::dvec3 hitPoint = o + tCyl * d;
                        const glm::dvec3 segPoint = a + clampedHitAxial * axis;
                        hasHit = true;
                        bestT = tCyl;
                        bestNormal = safeNormal(hitPoint - segPoint, -d);
                    }
                }
            }
        }
    }

    if (!hasHit) {
        return std::nullopt;
    }
    return CapsuleRayHit{
        static_cast<float>(bestT),
        glm::vec3(o + bestT * d),
        glm::vec3(bestNormal),
    };
}

LagCompHistory::LagCompHistory(double historyDurationSec, double maxRewindSec) noexcept
    : historyDurationSec_(std::isfinite(historyDurationSec) ? std::max(0.0, historyDurationSec)
                                                            : kDefaultHistoryDurationSec),
      maxRewindSec_(std::isfinite(maxRewindSec) ? std::max(0.0, maxRewindSec)
                                                : kDefaultMaxRewindSec) {}

void LagCompHistory::setHistoryDuration(double seconds) noexcept {
    if (!std::isfinite(seconds)) {
        return;
    }
    historyDurationSec_ = std::max(0.0, seconds);
    for (auto& [entity, ring] : histories_) {
        (void)entity;
        pruneEntityHistory(ring);
    }
}

void LagCompHistory::setMaxRewind(double seconds) noexcept {
    if (!std::isfinite(seconds)) {
        return;
    }
    maxRewindSec_ = std::max(0.0, seconds);
}

void LagCompHistory::recordSnapshot(entt::entity entity, const EntityPoseSnapshot& snapshot) {
    if (entity == entt::null || !std::isfinite(snapshot.timestamp)) {
        return;
    }

    EntityPoseSnapshot entry = snapshot;
    entry.rotation = normalizeSafe(snapshot.rotation);

    auto& ring = histories_[entity];
    if (ring.empty() || entry.timestamp > ring.back().timestamp + kTimestampEpsilon) {
        ring.push_back(std::move(entry));
    } else if (std::abs(entry.timestamp - ring.back().timestamp) <= kTimestampEpsilon) {
        ring.back() = std::move(entry);
    } else {
        const double cutoff = ring.back().timestamp - historyDurationSec_ - kTimestampEpsilon;
        if (entry.timestamp < cutoff) {
            return;
        }
        auto it = std::lower_bound(
            ring.begin(), ring.end(), entry.timestamp,
            [](const EntityPoseSnapshot& s, double t) { return s.timestamp < t; });
        if (it != ring.end() && std::abs(it->timestamp - entry.timestamp) <= kTimestampEpsilon) {
            *it = std::move(entry);
        } else {
            ring.insert(it, std::move(entry));
        }
    }

    pruneEntityHistory(ring);
}

void LagCompHistory::removeEntity(entt::entity entity) {
    histories_.erase(entity);
}

void LagCompHistory::clear() noexcept {
    histories_.clear();
}

std::size_t LagCompHistory::snapshotCount(entt::entity entity) const noexcept {
    const auto it = histories_.find(entity);
    return it == histories_.end() ? 0u : it->second.size();
}

std::optional<EntityPoseSnapshot> LagCompHistory::samplePoseAt(entt::entity entity, double targetTime) const {
    if (entity == entt::null || !std::isfinite(targetTime)) {
        return std::nullopt;
    }
    const auto mapIt = histories_.find(entity);
    if (mapIt == histories_.end() || mapIt->second.empty()) {
        return std::nullopt;
    }

    const auto& ring = mapIt->second;
    if (ring.size() == 1 || targetTime <= ring.front().timestamp + kTimestampEpsilon) {
        return ring.front();
    }
    if (targetTime >= ring.back().timestamp - kTimestampEpsilon) {
        return ring.back();
    }

    const auto upper = std::lower_bound(
        ring.begin(), ring.end(), targetTime,
        [](const EntityPoseSnapshot& s, double t) { return s.timestamp < t; });
    if (upper == ring.begin()) {
        return ring.front();
    }
    if (upper == ring.end()) {
        return ring.back();
    }

    const EntityPoseSnapshot& next = *upper;
    if (std::abs(next.timestamp - targetTime) <= kTimestampEpsilon) {
        return next;
    }
    const EntityPoseSnapshot& prev = *std::prev(upper);
    const double span = next.timestamp - prev.timestamp;
    if (span <= kTimestampEpsilon) {
        return next;
    }

    const float alpha = static_cast<float>(std::clamp((targetTime - prev.timestamp) / span, 0.0, 1.0));

    EntityPoseSnapshot sampled;
    sampled.timestamp = targetTime;
    sampled.position = glm::mix(prev.position, next.position, alpha);
    sampled.rotation = normalizeSafe(glm::slerp(normalizeSafe(prev.rotation), normalizeSafe(next.rotation), alpha));

    if (prev.hitboxes.size() == next.hitboxes.size()) {
        bool partsMatch = true;
        for (std::size_t i = 0; i < prev.hitboxes.size(); ++i) {
            if (prev.hitboxes[i].part != next.hitboxes[i].part) {
                partsMatch = false;
                break;
            }
        }
        if (partsMatch) {
            sampled.hitboxes.reserve(prev.hitboxes.size());
            for (std::size_t i = 0; i < prev.hitboxes.size(); ++i) {
                const HitCapsule& a = prev.hitboxes[i];
                const HitCapsule& b = next.hitboxes[i];
                sampled.hitboxes.push_back(HitCapsule{
                    glm::mix(a.localA, b.localA, alpha),
                    glm::mix(a.localB, b.localB, alpha),
                    std::lerp(a.radius, b.radius, alpha),
                    a.part,
                });
            }
        } else {
            sampled.hitboxes = alpha < 0.5f ? prev.hitboxes : next.hitboxes;
        }
    } else {
        sampled.hitboxes = alpha < 0.5f ? prev.hitboxes : next.hitboxes;
    }

    return sampled;
}

std::optional<LagCompHit> LagCompHistory::rewindRaycast(
    const glm::vec3& origin,
    const glm::vec3& direction,
    float maxDistance,
    double currentServerTime,
    double clientRttSeconds,
    double interpDelaySeconds,
    entt::entity ignoreEntity) const {
    if (!std::isfinite(currentServerTime) || !isFiniteVec3(origin) || !isFiniteVec3(direction) ||
        !std::isfinite(maxDistance) || maxDistance <= 0.0f) {
        return std::nullopt;
    }

    const float dirLenSq = glm::dot(direction, direction);
    if (dirLenSq <= 1e-12f) {
        return std::nullopt;
    }
    const glm::vec3 unitDir = direction / std::sqrt(dirLenSq);

    const double safeRtt = std::isfinite(clientRttSeconds) ? clientRttSeconds : 0.0;
    const double safeInterp = std::isfinite(interpDelaySeconds) ? interpDelaySeconds : 0.0;
    const double rewindDelta = std::clamp(safeRtt * 0.5 + safeInterp, 0.0, maxRewindSec_);
    const double evalTime = currentServerTime - rewindDelta;

    std::optional<LagCompHit> best;
    for (const auto& [entity, ring] : histories_) {
        (void)ring;
        if (entity == ignoreEntity || entity == entt::null) {
            continue;
        }

        const auto sampledPose = samplePoseAt(entity, evalTime);
        if (!sampledPose || sampledPose->hitboxes.empty()) {
            continue;
        }

        const glm::quat worldRot = normalizeSafe(sampledPose->rotation);
        const glm::vec3& worldPos = sampledPose->position;

        for (const HitCapsule& cap : sampledPose->hitboxes) {
            if (!std::isfinite(cap.radius) || cap.radius <= 0.0f) {
                continue;
            }
            const glm::vec3 worldA = worldPos + (worldRot * cap.localA);
            const glm::vec3 worldB = worldPos + (worldRot * cap.localB);

            const float searchDistance = best ? std::min(maxDistance, best->distance + kHitTieEpsilon) : maxDistance;
            const auto hit = rayCapsuleIntersect(origin, unitDir, searchDistance, worldA, worldB, cap.radius);
            if (!hit) {
                continue;
            }

            const bool isCloser = !best || hit->distance < best->distance - kHitTieEpsilon;
            const bool isPriorityTie =
                best && std::abs(hit->distance - best->distance) <= kHitTieEpsilon &&
                partPriority(cap.part) > partPriority(best->part);
            if (isCloser || isPriorityTie) {
                best = LagCompHit{
                    entity,
                    cap.part,
                    hit->point,
                    hit->normal,
                    hit->distance,
                    rewindDelta,
                };
            }
        }
    }

    return best;
}

std::optional<CapsuleRayHit> LagCompHistory::rayCapsuleIntersect(
    const glm::vec3& origin,
    const glm::vec3& direction,
    float maxDistance,
    const glm::vec3& capsuleA,
    const glm::vec3& capsuleB,
    float radius) noexcept {
    return lumora::net::rayCapsuleIntersect(origin, direction, maxDistance, capsuleA, capsuleB, radius);
}

void LagCompHistory::pruneEntityHistory(std::deque<EntityPoseSnapshot>& ring) const noexcept {
    if (ring.size() <= 1) {
        return;
    }
    const double cutoff = ring.back().timestamp - historyDurationSec_ - kTimestampEpsilon;
    while (ring.size() > 1 && ring.front().timestamp < cutoff) {
        ring.pop_front();
    }
}

} // namespace lumora::net
