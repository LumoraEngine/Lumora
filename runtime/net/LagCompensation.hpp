#pragma once

#include <entt/entity/entity.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/vec3.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <unordered_map>
#include <vector>

namespace lumora::net {

/// Semantic region for a capsule hitbox on an entity.
enum class HitboxPart : std::uint8_t {
    Body = 0,
    Head = 1,
    Limb = 2,
};

/// Local-space capsule hitbox attached to an entity pose snapshot.
/// Setting `localA == localB` represents a sphere of `radius`.
struct HitCapsule {
    glm::vec3 localA{0.0f};
    glm::vec3 localB{0.0f, 1.0f, 0.0f};
    float radius = 0.25f;
    HitboxPart part = HitboxPart::Body;
};

/// Authoritative server pose and hitbox configuration captured at a server timestamp.
struct EntityPoseSnapshot {
    double timestamp = 0.0;
    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<HitCapsule> hitboxes;
};

/// Closest hit produced by a lag-compensated rewind raycast.
struct LagCompHit {
    entt::entity entity = entt::null;
    HitboxPart part = HitboxPart::Body;
    glm::vec3 point{0.0f};
    glm::vec3 normal{0.0f};
    float distance = 0.0f;
    double rewoundBySeconds = 0.0;
};

/// Geometric result of an exact analytic ray-vs-capsule intersection.
struct CapsuleRayHit {
    float distance = 0.0f;
    glm::vec3 point{0.0f};
    glm::vec3 normal{0.0f};
};

/// Exact analytic ray-vs-capsule intersection (`direction` need not be normalized).
/// Supports degenerate sphere capsules (`capsuleA == capsuleB`) as well as arbitrary
/// finite capsules, returning the closest entry hit within `[0, maxDistance]`.
[[nodiscard]] std::optional<CapsuleRayHit> rayCapsuleIntersect(
    const glm::vec3& origin,
    const glm::vec3& direction,
    float maxDistance,
    const glm::vec3& capsuleA,
    const glm::vec3& capsuleB,
    float radius) noexcept;

/// Authoritative server lag-compensation history storing per-entity pose ring buffers,
/// slerp/linear pose sampling, and clamped rewind raycasts against entity hitboxes
/// without mutating live scene transforms.
class LagCompHistory {
public:
    static constexpr double kDefaultHistoryDurationSec = 1.0;
    static constexpr double kDefaultMaxRewindSec = 0.250;

    explicit LagCompHistory(double historyDurationSec = kDefaultHistoryDurationSec,
                            double maxRewindSec = kDefaultMaxRewindSec) noexcept;

    /// Sets how long snapshots are retained relative to each entity's newest timestamp.
    void setHistoryDuration(double seconds) noexcept;
    /// Sets the anti-exploit upper bound on rewind duration in seconds.
    void setMaxRewind(double seconds) noexcept;

    [[nodiscard]] double historyDuration() const noexcept { return historyDurationSec_; }
    [[nodiscard]] double maxRewind() const noexcept { return maxRewindSec_; }

    /// Appends a snapshot to the entity's ring buffer and prunes entries older than
    /// `newestTimestamp - historyDuration`.
    void recordSnapshot(entt::entity entity, const EntityPoseSnapshot& snapshot);

    /// Removes all history for a single entity.
    void removeEntity(entt::entity entity);
    /// Clears all tracked entities and snapshot history.
    void clear() noexcept;

    [[nodiscard]] std::size_t trackedEntityCount() const noexcept { return histories_.size(); }
    [[nodiscard]] std::size_t snapshotCount(entt::entity entity) const noexcept;

    /// Samples an entity's pose at `targetTime` using linear position interpolation and
    /// shortest-path quaternion `glm::slerp`, clamping cleanly to oldest/newest if outside.
    [[nodiscard]] std::optional<EntityPoseSnapshot> samplePoseAt(entt::entity entity, double targetTime) const;

    /// Performs an authoritative lag-compensated raycast at `currentServerTime - rewindDelta`,
    /// where `rewindDelta = std::clamp(clientRttSeconds * 0.5 + interpDelaySeconds, 0.0, maxRewindSec)`.
    [[nodiscard]] std::optional<LagCompHit> rewindRaycast(
        const glm::vec3& origin,
        const glm::vec3& direction,
        float maxDistance,
        double currentServerTime,
        double clientRttSeconds,
        double interpDelaySeconds = 0.0,
        entt::entity ignoreEntity = entt::null) const;

    /// Exact analytic ray-vs-capsule helper forwarding to `lumora::net::rayCapsuleIntersect`.
    [[nodiscard]] static std::optional<CapsuleRayHit> rayCapsuleIntersect(
        const glm::vec3& origin,
        const glm::vec3& direction,
        float maxDistance,
        const glm::vec3& capsuleA,
        const glm::vec3& capsuleB,
        float radius) noexcept;

private:
    void pruneEntityHistory(std::deque<EntityPoseSnapshot>& ring) const noexcept;

    double historyDurationSec_ = kDefaultHistoryDurationSec;
    double maxRewindSec_ = kDefaultMaxRewindSec;
    std::unordered_map<entt::entity, std::deque<EntityPoseSnapshot>> histories_;
};

} // namespace lumora::net
