#include "runtime/RvoCrowd.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace lumora {

namespace {

constexpr float kEpsilon = 1e-5f;
constexpr std::uint32_t kInvalidIndex = std::numeric_limits<std::uint32_t>::max();
constexpr float kHeadOnBiasRad = 0.02f;

[[nodiscard]] inline float dot2(const glm::vec2& a, const glm::vec2& b) noexcept {
    return a.x * b.x + a.y * b.y;
}

[[nodiscard]] inline float det2(const glm::vec2& a, const glm::vec2& b) noexcept {
    return a.x * b.y - a.y * b.x;
}

[[nodiscard]] inline float lengthSq2(const glm::vec2& v) noexcept {
    return v.x * v.x + v.y * v.y;
}

[[nodiscard]] inline glm::vec2 normalizeSafe(const glm::vec2& v, const glm::vec2& fallback) noexcept {
    const float lenSq = lengthSq2(v);
    if (lenSq <= kEpsilon * kEpsilon) {
        return fallback;
    }
    return v / std::sqrt(lenSq);
}

[[nodiscard]] inline std::int32_t cellCoord(float value, float invCellSize) noexcept {
    if (!std::isfinite(value)) {
        return 0;
    }
    const double scaled = std::floor(static_cast<double>(value) * static_cast<double>(invCellSize));
    const double clamped = std::clamp(scaled, -2000000000.0, 2000000000.0);
    return static_cast<std::int32_t>(clamped);
}

[[nodiscard]] inline std::uint64_t packCellKey(std::int32_t cx, std::int32_t cz) noexcept {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(cx)) << 32) |
           static_cast<std::uint64_t>(static_cast<std::uint32_t>(cz));
}

[[nodiscard]] inline std::size_t hashCellKey(std::uint64_t key, std::size_t mask) noexcept {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return static_cast<std::size_t>(key) & mask;
}

[[nodiscard]] std::size_t nextPowerOfTwo(std::size_t n) noexcept {
    std::size_t p = 1;
    while (p < n && p <= (std::numeric_limits<std::size_t>::max() >> 1)) {
        p <<= 1;
    }
    return p;
}

[[nodiscard]] float distSqPointToSegment(const glm::vec2& p, const glm::vec2& a, const glm::vec2& b) noexcept {
    const glm::vec2 ab = b - a;
    const float abLenSq = lengthSq2(ab);
    if (abLenSq <= kEpsilon * kEpsilon) {
        return lengthSq2(p - a);
    }
    const float t = std::clamp(dot2(p - a, ab) / abLenSq, 0.0f, 1.0f);
    const glm::vec2 closest = a + t * ab;
    return lengthSq2(p - closest);
}

[[nodiscard]] float distSqPointToCell(const glm::vec2& pos, std::int32_t cx, std::int32_t cz,
                                      float cellSize) noexcept {
    const float minX = static_cast<float>(cx) * cellSize;
    const float maxX = static_cast<float>(cx + 1) * cellSize;
    const float minZ = static_cast<float>(cz) * cellSize;
    const float maxZ = static_cast<float>(cz + 1) * cellSize;

    float dx = 0.0f;
    if (pos.x < minX) {
        dx = minX - pos.x;
    } else if (pos.x > maxX) {
        dx = pos.x - maxX;
    }

    float dz = 0.0f;
    if (pos.y < minZ) {
        dz = minZ - pos.y;
    } else if (pos.y > maxZ) {
        dz = pos.y - maxZ;
    }

    return dx * dx + dz * dz;
}

} // namespace

RvoCrowd::RvoCrowd(float spatialCellSize) noexcept : requestedCellSize_(spatialCellSize) {}

std::uint32_t RvoCrowd::addAgent(const RvoAgentConfig& config) {
    const std::uint32_t id = nextAgentId_++;
    const std::size_t index = agents_.size();
    agents_.push_back(AgentEntry{id, config});
    idToIndex_.emplace(id, index);
    return id;
}

bool RvoCrowd::removeAgent(std::uint32_t id) {
    const auto it = idToIndex_.find(id);
    if (it == idToIndex_.end()) {
        return false;
    }

    const std::size_t removeIdx = it->second;
    const std::size_t lastIdx = agents_.size() - 1;
    if (removeIdx != lastIdx) {
        agents_[removeIdx] = agents_[lastIdx];
        idToIndex_[agents_[removeIdx].id] = removeIdx;
    }
    agents_.pop_back();
    idToIndex_.erase(it);
    return true;
}

void RvoCrowd::clearAgents() noexcept {
    agents_.clear();
    idToIndex_.clear();
    nextAgentId_ = 0;
}

void RvoCrowd::setPreferredVelocity(std::uint32_t id, glm::vec2 preferredVelocity) {
    const auto it = idToIndex_.find(id);
    if (it == idToIndex_.end()) {
        throw std::out_of_range("RvoCrowd::setPreferredVelocity: invalid agent id");
    }
    agents_[it->second].config.preferredVelocity = preferredVelocity;
}

void RvoCrowd::setPosition(std::uint32_t id, glm::vec2 position) {
    const auto it = idToIndex_.find(id);
    if (it == idToIndex_.end()) {
        throw std::out_of_range("RvoCrowd::setPosition: invalid agent id");
    }
    agents_[it->second].config.position = position;
}

void RvoCrowd::setVelocity(std::uint32_t id, glm::vec2 velocity) {
    const auto it = idToIndex_.find(id);
    if (it == idToIndex_.end()) {
        throw std::out_of_range("RvoCrowd::setVelocity: invalid agent id");
    }
    agents_[it->second].config.velocity = velocity;
}

void RvoCrowd::setAgentConfig(std::uint32_t id, const RvoAgentConfig& config) {
    const auto it = idToIndex_.find(id);
    if (it == idToIndex_.end()) {
        throw std::out_of_range("RvoCrowd::setAgentConfig: invalid agent id");
    }
    agents_[it->second].config = config;
}

bool RvoCrowd::hasAgent(std::uint32_t id) const noexcept {
    return idToIndex_.find(id) != idToIndex_.end();
}

const RvoAgentConfig& RvoCrowd::agent(std::uint32_t id) const {
    const auto it = idToIndex_.find(id);
    if (it == idToIndex_.end()) {
        throw std::out_of_range("RvoCrowd::agent: invalid agent id");
    }
    return agents_[it->second].config;
}

const RvoAgentConfig* RvoCrowd::findAgent(std::uint32_t id) const noexcept {
    const auto it = idToIndex_.find(id);
    if (it == idToIndex_.end()) {
        return nullptr;
    }
    return &agents_[it->second].config;
}

std::size_t RvoCrowd::agentCount() const noexcept {
    return agents_.size();
}

std::uint32_t RvoCrowd::addObstacleSegment(glm::vec2 a, glm::vec2 b) {
    const std::uint32_t index = static_cast<std::uint32_t>(obstacles_.size());
    obstacles_.push_back(RvoObstacleSegment{a, b});
    return index;
}

void RvoCrowd::clearObstacles() noexcept {
    obstacles_.clear();
}

std::size_t RvoCrowd::obstacleCount() const noexcept {
    return obstacles_.size();
}

std::span<const RvoObstacleSegment> RvoCrowd::obstacles() const noexcept {
    return obstacles_;
}

void RvoCrowd::rebuildSpatialGrid(float cellSize) {
    const std::size_t count = agents_.size();
    const std::size_t targetBuckets = nextPowerOfTwo(std::max<std::size_t>(64, count * 2 + 16));
    if (gridBuckets_.size() < targetBuckets) {
        gridBuckets_.assign(targetBuckets, CellBucket{});
        gridStamp_ = 0;
    }

    if (gridStamp_ == std::numeric_limits<std::uint32_t>::max()) {
        std::fill(gridBuckets_.begin(), gridBuckets_.end(), CellBucket{});
        gridStamp_ = 0;
    }
    ++gridStamp_;

    gridNext_.resize(count);
    const std::size_t mask = gridBuckets_.size() - 1;
    const float invCellSize = 1.0f / cellSize;

    for (std::size_t i = 0; i < count; ++i) {
        const glm::vec2& pos = agents_[i].config.position;
        const std::int32_t cx = cellCoord(pos.x, invCellSize);
        const std::int32_t cz = cellCoord(pos.y, invCellSize);
        const std::uint64_t key = packCellKey(cx, cz);

        std::size_t slot = hashCellKey(key, mask);
        while (gridBuckets_[slot].stamp == gridStamp_ && gridBuckets_[slot].key != key) {
            slot = (slot + 1) & mask;
        }

        if (gridBuckets_[slot].stamp != gridStamp_) {
            gridBuckets_[slot].stamp = gridStamp_;
            gridBuckets_[slot].key = key;
            gridBuckets_[slot].head = static_cast<std::uint32_t>(i);
            gridNext_[i] = kInvalidIndex;
        } else {
            gridNext_[i] = gridBuckets_[slot].head;
            gridBuckets_[slot].head = static_cast<std::uint32_t>(i);
        }
    }
}

void RvoCrowd::queryAgentNeighbors(std::size_t agentIndex, float cellSize,
                                   std::vector<NeighborCandidate>& outNeighbors) const {
    outNeighbors.clear();
    const auto& self = agents_[agentIndex];
    const std::uint32_t maxNeighbors = self.config.maxNeighbors;
    const float neighborDist = self.config.neighborDist;
    if (maxNeighbors == 0 || neighborDist <= 0.0f || agents_.size() <= 1) {
        return;
    }

    float currentRangeSq = neighborDist * neighborDist;
    const glm::vec2& pos = self.config.position;

    auto tryInsertCandidate = [&](std::uint32_t candidateIndex) {
        if (candidateIndex == static_cast<std::uint32_t>(agentIndex)) {
            return;
        }
        const auto& other = agents_[candidateIndex];
        const float dSq = lengthSq2(other.config.position - pos);
        if (dSq > currentRangeSq) {
            return;
        }
        if (outNeighbors.size() == maxNeighbors && dSq == currentRangeSq &&
            other.id >= outNeighbors.back().id) {
            return;
        }

        const NeighborCandidate cand{dSq, candidateIndex, other.id};
        auto posIt = std::lower_bound(
            outNeighbors.begin(), outNeighbors.end(), cand,
            [](const NeighborCandidate& lhs, const NeighborCandidate& rhs) {
                if (lhs.distSq != rhs.distSq) {
                    return lhs.distSq < rhs.distSq;
                }
                return lhs.id < rhs.id;
            });

        if (outNeighbors.size() < maxNeighbors) {
            outNeighbors.insert(posIt, cand);
        } else {
            outNeighbors.insert(posIt, cand);
            outNeighbors.pop_back();
        }

        if (outNeighbors.size() == maxNeighbors) {
            currentRangeSq = outNeighbors.back().distSq;
        }
    };

    const float invCellSize = 1.0f / cellSize;
    const double rawSpan = std::ceil(static_cast<double>(neighborDist) * static_cast<double>(invCellSize));
    const std::int64_t clampedSpan = static_cast<std::int64_t>(std::min(rawSpan, 1000000.0));
    const std::int64_t gridSide = 2 * clampedSpan + 1;

    // Fall back to direct scan only for outlier huge neighborDist radii covering far more cells than agents.
    if (gridSide * gridSide > static_cast<std::int64_t>(agents_.size()) * 4 + 64) {
        for (std::size_t i = 0; i < agents_.size(); ++i) {
            tryInsertCandidate(static_cast<std::uint32_t>(i));
        }
        return;
    }

    const std::int32_t cx = cellCoord(pos.x, invCellSize);
    const std::int32_t cz = cellCoord(pos.y, invCellSize);
    const std::int32_t maxRing = static_cast<std::int32_t>(clampedSpan);
    const std::size_t mask = gridBuckets_.size() - 1;

    auto visitCell = [&](std::int32_t nx, std::int32_t nz) {
        if (distSqPointToCell(pos, nx, nz, cellSize) > currentRangeSq) {
            return;
        }
        const std::uint64_t key = packCellKey(nx, nz);
        std::size_t slot = hashCellKey(key, mask);
        while (gridBuckets_[slot].stamp == gridStamp_ && gridBuckets_[slot].key != key) {
            slot = (slot + 1) & mask;
        }
        if (gridBuckets_[slot].stamp != gridStamp_) {
            return;
        }

        std::uint32_t idx = gridBuckets_[slot].head;
        while (idx != kInvalidIndex) {
            tryInsertCandidate(idx);
            idx = gridNext_[idx];
        }
    };

    visitCell(cx, cz);

    for (std::int32_t ring = 1; ring <= maxRing; ++ring) {
        if (outNeighbors.size() == maxNeighbors) {
            const float xMin = static_cast<float>(cx - ring + 1) * cellSize;
            const float xMax = static_cast<float>(cx + ring) * cellSize;
            const float zMin = static_cast<float>(cz - ring + 1) * cellSize;
            const float zMax = static_cast<float>(cz + ring) * cellSize;
            const float minEdgeDist = std::min({pos.x - xMin, xMax - pos.x, pos.y - zMin, zMax - pos.y});
            if (minEdgeDist > 0.0f && minEdgeDist * minEdgeDist > currentRangeSq) {
                break;
            }
        }

        for (std::int32_t dx = -ring; dx <= ring; ++dx) {
            visitCell(cx + dx, cz - ring);
            visitCell(cx + dx, cz + ring);
        }
        for (std::int32_t dz = -ring + 1; dz <= ring - 1; ++dz) {
            visitCell(cx - ring, cz + dz);
            visitCell(cx + ring, cz + dz);
        }
    }
}

void RvoCrowd::queryNearbyObstacles(std::size_t agentIndex, float rangeSq,
                                    std::vector<ObstacleCandidate>& outObstacles) const {
    outObstacles.clear();
    if (obstacles_.empty()) {
        return;
    }

    const glm::vec2& pos = agents_[agentIndex].config.position;
    for (std::size_t i = 0; i < obstacles_.size(); ++i) {
        const auto& seg = obstacles_[i];
        const float dSq = distSqPointToSegment(pos, seg.a, seg.b);
        if (dSq <= rangeSq) {
            outObstacles.push_back(ObstacleCandidate{dSq, i});
        }
    }

    std::sort(outObstacles.begin(), outObstacles.end(),
              [](const ObstacleCandidate& lhs, const ObstacleCandidate& rhs) {
                  if (lhs.distSq != rhs.distSq) {
                      return lhs.distSq < rhs.distSq;
                  }
                  return lhs.index < rhs.index;
              });
}

void RvoCrowd::buildObstacleOrcaLine(const RvoAgentConfig& selfConfig, const RvoObstacleSegment& segment,
                                     float invDt, OrcaLine& outLine) const {
    const float radius = std::max(0.0f, selfConfig.radius);
    const float radiusSq = radius * radius;
    const float invTau = 1.0f / std::max(selfConfig.timeHorizonObst, 1e-4f);

    const glm::vec2 segVec = segment.b - segment.a;
    const float segLenSq = lengthSq2(segVec);

    // Degenerate zero-length segment treated as a static point obstacle of radius `radius`.
    if (segLenSq <= kEpsilon * kEpsilon) {
        const glm::vec2 relPos = segment.a - selfConfig.position;
        const float distSq = lengthSq2(relPos);
        if (distSq <= radiusSq) {
            if (distSq > kEpsilon * kEpsilon) {
                const float dist = std::sqrt(distSq);
                const glm::vec2 normal = -relPos / dist;
                outLine.direction = glm::vec2(normal.y, -normal.x);
                outLine.point = ((radius - dist) * invDt) * normal;
            } else {
                outLine.direction = glm::vec2(0.0f, 1.0f);
                outLine.point = glm::vec2(-radius * invDt, 0.0f);
            }
            return;
        }

        const glm::vec2 w = selfConfig.velocity - invTau * relPos;
        const float wLenSq = lengthSq2(w);
        const float dot1 = dot2(w, relPos);

        if (dot1 < 0.0f && dot1 * dot1 > radiusSq * wLenSq) {
            const float wLen = std::sqrt(wLenSq);
            glm::vec2 unitW = w / wLen;
            const float dist = std::sqrt(distSq);
            if (std::abs(det2(relPos, unitW)) < 1e-4f * dist && dot2(relPos, unitW) < 0.0f) {
                unitW = normalizeSafe(unitW + kHeadOnBiasRad * glm::vec2(-unitW.y, unitW.x), unitW);
            }
            outLine.direction = glm::vec2(unitW.y, -unitW.x);
            outLine.point = invTau * relPos + (radius * invTau) * unitW;
        } else {
            const float leg = std::sqrt(std::max(0.0f, distSq - radiusSq));
            if (det2(relPos, w) > 0.0f) {
                outLine.direction = glm::vec2(relPos.x * leg - relPos.y * radius,
                                              relPos.x * radius + relPos.y * leg) /
                                    distSq;
            } else {
                outLine.direction = -glm::vec2(relPos.x * leg + relPos.y * radius,
                                               -relPos.x * radius + relPos.y * leg) /
                                    distSq;
            }
            const float dot2Val = dot2(selfConfig.velocity, outLine.direction);
            outLine.point = dot2Val * outLine.direction;
        }
        return;
    }

    // Orient segment (p1 -> p2) so the agent lies on the right side (two-sided obstacle segment).
    glm::vec2 p1 = segment.a;
    glm::vec2 p2 = segment.b;
    const float side = det2(segVec, selfConfig.position - segment.a);
    if (side > 0.0f) {
        p1 = segment.b;
        p2 = segment.a;
    } else if (side == 0.0f) {
        const glm::vec2 leftNormal(-segVec.y, segVec.x);
        const float prefProj = dot2(leftNormal, selfConfig.preferredVelocity);
        const float velProj = dot2(leftNormal, selfConfig.velocity);
        if (prefProj > 0.0f || (prefProj == 0.0f && velProj > 0.0f)) {
            p1 = segment.b;
            p2 = segment.a;
        }
    }

    const glm::vec2 q1 = p1 - selfConfig.position;
    const glm::vec2 q2 = p2 - selfConfig.position;
    const glm::vec2 obsVec = q2 - q1;
    const float obsLenSq = lengthSq2(obsVec);
    const float obsLen = std::sqrt(obsLenSq);
    const glm::vec2 obsUnit = obsVec / obsLen;
    const glm::vec2 rightNormal(obsUnit.y, -obsUnit.x); // Points from segment toward agent's side.

    const float distSq1 = lengthSq2(q1);
    const float distSq2 = lengthSq2(q2);
    const float s = dot2(-q1, obsVec) / obsLenSq;
    const float signedPerp = det2(obsUnit, -q1);

    // Current overlap resolution.
    if (s < 0.0f && distSq1 <= radiusSq) {
        if (distSq1 > kEpsilon * kEpsilon) {
            const float dist = std::sqrt(distSq1);
            const glm::vec2 normal = -q1 / dist;
            outLine.direction = glm::vec2(normal.y, -normal.x);
            outLine.point = ((radius - dist) * invDt) * normal;
        } else {
            outLine.direction = -obsUnit;
            outLine.point = (radius * invDt) * rightNormal;
        }
        return;
    }

    if (s > 1.0f && distSq2 <= radiusSq) {
        if (distSq2 > kEpsilon * kEpsilon) {
            const float dist = std::sqrt(distSq2);
            const glm::vec2 normal = -q2 / dist;
            outLine.direction = glm::vec2(normal.y, -normal.x);
            outLine.point = ((radius - dist) * invDt) * normal;
        } else {
            outLine.direction = -obsUnit;
            outLine.point = (radius * invDt) * rightNormal;
        }
        return;
    }

    if (s >= 0.0f && s <= 1.0f && signedPerp * signedPerp <= radiusSq) {
        const float dist = std::abs(signedPerp);
        outLine.direction = -obsUnit;
        outLine.point = ((radius - dist) * invDt) * rightNormal;
        return;
    }

    // Non-overlapping obstacle Velocity Obstacle construction.
    const float leg1 = std::sqrt(std::max(0.0f, distSq1 - radiusSq));
    const glm::vec2 leftLegDir(
        (q1.x * leg1 - q1.y * radius) / distSq1,
        (q1.x * radius + q1.y * leg1) / distSq1);

    const float leg2 = std::sqrt(std::max(0.0f, distSq2 - radiusSq));
    const glm::vec2 rightLegDir(
        (q2.x * leg2 + q2.y * radius) / distSq2,
        (-q2.x * radius + q2.y * leg2) / distSq2);

    const glm::vec2 leftCutoff = invTau * q1;
    const glm::vec2 rightCutoff = invTau * q2;
    const glm::vec2 cutoffVec = rightCutoff - leftCutoff;
    const float cutoffLenSq = lengthSq2(cutoffVec);

    const float t = (cutoffLenSq > kEpsilon * kEpsilon)
        ? dot2(selfConfig.velocity - leftCutoff, cutoffVec) / cutoffLenSq
        : 0.5f;
    const float tLeft = dot2(selfConfig.velocity - leftCutoff, leftLegDir);
    const float tRight = dot2(selfConfig.velocity - rightCutoff, rightLegDir);

    if (t < 0.0f && tLeft < 0.0f) {
        const glm::vec2 w = selfConfig.velocity - leftCutoff;
        const float wLen = std::sqrt(lengthSq2(w));
        glm::vec2 unitW = (wLen > kEpsilon) ? (w / wLen) : rightNormal;
        if (std::abs(det2(q1, unitW)) < 1e-4f * std::sqrt(distSq1) && dot2(q1, unitW) < 0.0f) {
            unitW = normalizeSafe(unitW + kHeadOnBiasRad * rightNormal, unitW);
        }
        outLine.direction = glm::vec2(unitW.y, -unitW.x);
        outLine.point = leftCutoff + (radius * invTau) * unitW;
        return;
    }

    if (t > 1.0f && tRight < 0.0f) {
        const glm::vec2 w = selfConfig.velocity - rightCutoff;
        const float wLen = std::sqrt(lengthSq2(w));
        glm::vec2 unitW = (wLen > kEpsilon) ? (w / wLen) : rightNormal;
        if (std::abs(det2(q2, unitW)) < 1e-4f * std::sqrt(distSq2) && dot2(q2, unitW) < 0.0f) {
            unitW = normalizeSafe(unitW + kHeadOnBiasRad * rightNormal, unitW);
        }
        outLine.direction = glm::vec2(unitW.y, -unitW.x);
        outLine.point = rightCutoff + (radius * invTau) * unitW;
        return;
    }

    const float distSqCutoff = (t < 0.0f || t > 1.0f)
        ? std::numeric_limits<float>::infinity()
        : lengthSq2(selfConfig.velocity - (leftCutoff + t * cutoffVec));
    const float distSqLeft = (tLeft < 0.0f)
        ? std::numeric_limits<float>::infinity()
        : lengthSq2(selfConfig.velocity - (leftCutoff + tLeft * leftLegDir));
    const float distSqRight = (tRight < 0.0f)
        ? std::numeric_limits<float>::infinity()
        : lengthSq2(selfConfig.velocity - (rightCutoff + tRight * rightLegDir));

    if (distSqCutoff <= distSqLeft && distSqCutoff <= distSqRight) {
        outLine.direction = -obsUnit;
        outLine.point = leftCutoff + (radius * invTau) * rightNormal;
    } else if (distSqLeft <= distSqRight) {
        outLine.direction = leftLegDir;
        outLine.point = leftCutoff + (radius * invTau) * glm::vec2(-leftLegDir.y, leftLegDir.x);
    } else {
        outLine.direction = -rightLegDir;
        outLine.point = rightCutoff + (radius * invTau) * glm::vec2(-outLine.direction.y, outLine.direction.x);
    }
}

void RvoCrowd::buildAgentOrcaLine(const AgentEntry& self, const AgentEntry& other,
                                  float invDt, OrcaLine& outLine) const {
    const auto& a = self.config;
    const auto& b = other.config;

    const glm::vec2 relativePosition = b.position - a.position;
    const glm::vec2 relativeVelocity = a.velocity - b.velocity;
    const float distSq = lengthSq2(relativePosition);
    const float combinedRadius = std::max(0.0f, a.radius) + std::max(0.0f, b.radius);
    const float combinedRadiusSq = combinedRadius * combinedRadius;

    glm::vec2 u{0.0f, 0.0f};
    glm::vec2 direction{0.0f, 0.0f};

    if (distSq > combinedRadiusSq) {
        const float invTimeHorizon = 1.0f / std::max(a.timeHorizon, 1e-4f);
        const glm::vec2 w = relativeVelocity - invTimeHorizon * relativePosition;
        const float wLengthSq = lengthSq2(w);
        const float dotProduct1 = dot2(w, relativePosition);

        if (dotProduct1 < 0.0f && dotProduct1 * dotProduct1 > combinedRadiusSq * wLengthSq) {
            const float wLength = std::sqrt(wLengthSq);
            glm::vec2 unitW = w / wLength;

            // Break exact collinear head-on cutoff-circle saddle symmetry with a right-hand bias
            // consistent with the tangent leg convention.
            const float dist = std::sqrt(distSq);
            if (std::abs(det2(relativePosition, unitW)) < 1e-3f * dist && dot2(relativePosition, unitW) < 0.0f) {
                const glm::vec2 rightStep(-unitW.y, unitW.x);
                unitW = normalizeSafe(unitW + kHeadOnBiasRad * rightStep, unitW);
            }

            direction = glm::vec2(unitW.y, -unitW.x);
            u = (combinedRadius * invTimeHorizon - wLength) * unitW;
        } else {
            const float leg = std::sqrt(std::max(0.0f, distSq - combinedRadiusSq));
            if (det2(relativePosition, w) > 0.0f) {
                direction = glm::vec2(
                    relativePosition.x * leg - relativePosition.y * combinedRadius,
                    relativePosition.x * combinedRadius + relativePosition.y * leg) /
                    distSq;
            } else {
                direction = -glm::vec2(
                    relativePosition.x * leg + relativePosition.y * combinedRadius,
                    -relativePosition.x * combinedRadius + relativePosition.y * leg) /
                    distSq;
            }
            const float dotProduct2 = dot2(relativeVelocity, direction);
            u = dotProduct2 * direction - relativeVelocity;
        }
    } else {
        // Agents already overlap; separate over one step using `invDt`.
        const glm::vec2 w = relativeVelocity - invDt * relativePosition;
        const float wLength = std::sqrt(lengthSq2(w));
        glm::vec2 unitW{0.0f, 0.0f};
        if (wLength > kEpsilon) {
            unitW = w / wLength;
            if (distSq > kEpsilon * kEpsilon) {
                const float dist = std::sqrt(distSq);
                if (std::abs(det2(relativePosition, unitW)) < 1e-3f * dist && dot2(relativePosition, unitW) < 0.0f) {
                    const glm::vec2 rightStep(-unitW.y, unitW.x);
                    unitW = normalizeSafe(unitW + kHeadOnBiasRad * rightStep, unitW);
                }
            }
        } else if (distSq > kEpsilon * kEpsilon) {
            unitW = -relativePosition / std::sqrt(distSq);
        } else {
            unitW = (self.id < other.id) ? glm::vec2(-1.0f, 0.0f) : glm::vec2(1.0f, 0.0f);
        }

        direction = glm::vec2(unitW.y, -unitW.x);
        u = (combinedRadius * invDt - wLength) * unitW;
    }

    outLine.direction = direction;
    outLine.point = a.velocity + 0.5f * u;
}

void RvoCrowd::step(float dt) {
    if (dt <= 0.0f || !std::isfinite(dt) || agents_.empty()) {
        return;
    }

    const std::size_t count = agents_.size();
    const float invDt = 1.0f / dt;

    float cellSize = requestedCellSize_;
    if (cellSize <= 0.0f) {
        double sumNeighborDist = 0.0;
        for (const auto& entry : agents_) {
            sumNeighborDist += static_cast<double>(std::max(0.0f, entry.config.neighborDist));
        }
        const float avgNeighborDist = static_cast<float>(sumNeighborDist / static_cast<double>(count));
        cellSize = std::clamp(avgNeighborDist * 0.5f, 2.0f, 10.0f);
    }

    rebuildSpatialGrid(cellSize);
    nextVelocities_.resize(count);

    for (std::size_t i = 0; i < count; ++i) {
        const auto& self = agents_[i];
        const float maxSpeed = std::max(0.0f, self.config.maxSpeed);

        orcaLinesScratch_.clear();

        if (!obstacles_.empty()) {
            const float currentSpeed = std::sqrt(lengthSq2(self.config.velocity));
            const float obsRange = std::max(0.0f, self.config.timeHorizonObst) *
                                       std::max(maxSpeed, currentSpeed) +
                                   std::max(0.0f, self.config.radius);
            queryNearbyObstacles(i, obsRange * obsRange, obstaclesScratch_);
            for (const auto& obsCand : obstaclesScratch_) {
                OrcaLine line{};
                buildObstacleOrcaLine(self.config, obstacles_[obsCand.index], invDt, line);
                orcaLinesScratch_.push_back(line);
            }
        }

        const std::size_t numObstLines = orcaLinesScratch_.size();

        queryAgentNeighbors(i, cellSize, neighborsScratch_);
        for (const auto& neighCand : neighborsScratch_) {
            OrcaLine line{};
            buildAgentOrcaLine(self, agents_[neighCand.index], invDt, line);
            orcaLinesScratch_.push_back(line);
        }

        glm::vec2 newVel{0.0f, 0.0f};
        const std::size_t lineFail = linearProgram2(
            orcaLinesScratch_, maxSpeed, self.config.preferredVelocity, false, newVel);
        if (lineFail < orcaLinesScratch_.size()) {
            linearProgram3(orcaLinesScratch_, numObstLines, lineFail, maxSpeed, newVel);
        }

        nextVelocities_[i] = newVel;
    }

    for (std::size_t i = 0; i < count; ++i) {
        agents_[i].config.velocity = nextVelocities_[i];
        agents_[i].config.position += nextVelocities_[i] * dt;
    }
}

bool RvoCrowd::linearProgram1(std::span<const OrcaLine> lines, std::size_t lineNo, float radius,
                              const glm::vec2& optVelocity, bool directionOpt, glm::vec2& result) {
    const float clampedRadius = std::max(0.0f, radius);
    const glm::vec2& linePoint = lines[lineNo].point;
    const glm::vec2& lineDir = lines[lineNo].direction;

    const float dotProduct = dot2(linePoint, lineDir);
    const float discriminant = dotProduct * dotProduct + clampedRadius * clampedRadius - lengthSq2(linePoint);

    if (discriminant < 0.0f) {
        return false;
    }

    const float sqrtDiscriminant = std::sqrt(discriminant);
    float tLeft = -dotProduct - sqrtDiscriminant;
    float tRight = -dotProduct + sqrtDiscriminant;

    for (std::size_t i = 0; i < lineNo; ++i) {
        const float denominator = det2(lineDir, lines[i].direction);
        const float numerator = det2(lines[i].direction, linePoint - lines[i].point);

        if (std::abs(denominator) <= kEpsilon) {
            if (numerator < 0.0f) {
                return false;
            }
            continue;
        }

        const float t = numerator / denominator;
        if (denominator >= 0.0f) {
            tRight = std::min(tRight, t);
        } else {
            tLeft = std::max(tLeft, t);
        }

        if (tLeft > tRight) {
            return false;
        }
    }

    if (directionOpt) {
        if (dot2(optVelocity, lineDir) > 0.0f) {
            result = linePoint + tRight * lineDir;
        } else {
            result = linePoint + tLeft * lineDir;
        }
    } else {
        const float t = dot2(lineDir, optVelocity - linePoint);
        if (t < tLeft) {
            result = linePoint + tLeft * lineDir;
        } else if (t > tRight) {
            result = linePoint + tRight * lineDir;
        } else {
            result = linePoint + t * lineDir;
        }
    }

    return true;
}

std::size_t RvoCrowd::linearProgram2(std::span<const OrcaLine> lines, float radius,
                                     const glm::vec2& optVelocity, bool directionOpt, glm::vec2& result) {
    const float clampedRadius = std::max(0.0f, radius);

    if (directionOpt) {
        const float lenSq = lengthSq2(optVelocity);
        if (lenSq > kEpsilon * kEpsilon) {
            result = (optVelocity / std::sqrt(lenSq)) * clampedRadius;
        } else {
            result = glm::vec2(0.0f, 0.0f);
        }
    } else if (lengthSq2(optVelocity) > clampedRadius * clampedRadius) {
        result = normalizeSafe(optVelocity, glm::vec2(0.0f, 0.0f)) * clampedRadius;
    } else {
        result = optVelocity;
    }

    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (det2(lines[i].direction, lines[i].point - result) > 0.0f) {
            const glm::vec2 tempResult = result;
            if (!linearProgram1(lines, i, clampedRadius, optVelocity, directionOpt, result)) {
                result = tempResult;
                return i;
            }
        }
    }

    return lines.size();
}

void RvoCrowd::linearProgram3(std::span<const OrcaLine> lines, std::size_t numObstLines,
                              std::size_t beginLine, float radius, glm::vec2& result) {
    const float clampedRadius = std::max(0.0f, radius);
    const std::size_t clampedObst = std::min(numObstLines, lines.size());
    const std::size_t hardObstLines = (beginLine >= clampedObst) ? clampedObst : 0;

    float distance = 0.0f;
    std::vector<OrcaLine> projLines;
    projLines.reserve(lines.size());

    for (std::size_t i = beginLine; i < lines.size(); ++i) {
        if (det2(lines[i].direction, lines[i].point - result) > distance) {
            projLines.assign(lines.begin(), lines.begin() + static_cast<std::ptrdiff_t>(hardObstLines));

            for (std::size_t j = hardObstLines; j < i; ++j) {
                OrcaLine line{};
                const float determinant = det2(lines[i].direction, lines[j].direction);

                if (std::abs(determinant) <= kEpsilon) {
                    if (dot2(lines[i].direction, lines[j].direction) > 0.0f) {
                        continue;
                    }
                    line.point = 0.5f * (lines[i].point + lines[j].point);
                } else {
                    line.point = lines[i].point +
                        (det2(lines[j].direction, lines[i].point - lines[j].point) / determinant) *
                        lines[i].direction;
                }

                const glm::vec2 dirDiff = lines[j].direction - lines[i].direction;
                const float dirDiffLenSq = lengthSq2(dirDiff);
                if (dirDiffLenSq <= kEpsilon * kEpsilon) {
                    continue;
                }
                line.direction = dirDiff / std::sqrt(dirDiffLenSq);
                projLines.push_back(line);
            }

            const glm::vec2 tempResult = result;
            const glm::vec2 optDir(-lines[i].direction.y, lines[i].direction.x);
            if (linearProgram2(projLines, clampedRadius, optDir, true, result) < projLines.size()) {
                result = tempResult;
            }

            distance = det2(lines[i].direction, lines[i].point - result);
        }
    }
}

} // namespace lumora
