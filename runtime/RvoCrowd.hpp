#pragma once

#include <glm/vec2.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace lumora {

/// Directed half-plane constraint in 2D velocity space:
/// feasible velocities lie on or to the left of `direction` passing through `point`.
struct OrcaLine {
    glm::vec2 point{0.0f, 0.0f};
    glm::vec2 direction{0.0f, 0.0f};
};

/// Configuration and state of a single RVO2 / ORCA crowd agent on the 2D XZ plane.
struct RvoAgentConfig {
    glm::vec2 position{0.0f, 0.0f};
    glm::vec2 velocity{0.0f, 0.0f};
    glm::vec2 preferredVelocity{0.0f, 0.0f};
    float radius = 0.5f;
    float maxSpeed = 2.0f;
    float neighborDist = 15.0f;
    std::uint32_t maxNeighbors = 10;
    float timeHorizon = 5.0f;
    float timeHorizonObst = 5.0f;
};

/// Static 2D line-segment obstacle on the XZ plane.
struct RvoObstacleSegment {
    glm::vec2 a{0.0f, 0.0f};
    glm::vec2 b{0.0f, 0.0f};
};

/// 2D XZ-plane RVO2 / ORCA local collision-avoidance crowd solver.
class RvoCrowd {
public:
    explicit RvoCrowd(float spatialCellSize = 0.0f) noexcept;

    std::uint32_t addAgent(const RvoAgentConfig& config);
    bool removeAgent(std::uint32_t id);
    void clearAgents() noexcept;

    void setPreferredVelocity(std::uint32_t id, glm::vec2 preferredVelocity);
    void setPosition(std::uint32_t id, glm::vec2 position);
    void setVelocity(std::uint32_t id, glm::vec2 velocity);
    void setAgentConfig(std::uint32_t id, const RvoAgentConfig& config);

    [[nodiscard]] bool hasAgent(std::uint32_t id) const noexcept;
    [[nodiscard]] const RvoAgentConfig& agent(std::uint32_t id) const;
    [[nodiscard]] const RvoAgentConfig* findAgent(std::uint32_t id) const noexcept;
    [[nodiscard]] std::size_t agentCount() const noexcept;

    std::uint32_t addObstacleSegment(glm::vec2 a, glm::vec2 b);
    void clearObstacles() noexcept;
    [[nodiscard]] std::size_t obstacleCount() const noexcept;
    [[nodiscard]] std::span<const RvoObstacleSegment> obstacles() const noexcept;

    /// Updates agent velocities via ORCA and advances positions concurrently (`pos += vel * dt`).
    void step(float dt);

    /// 1D linear program along line `lineNo` within disk of `radius` subject to prior `lines[0..lineNo-1]`.
    static bool linearProgram1(std::span<const OrcaLine> lines, std::size_t lineNo, float radius,
                               const glm::vec2& optVelocity, bool directionOpt, glm::vec2& result);
    /// Incremental 2D linear program within disk of `radius`; returns the index of the first infeasible line.
    static std::size_t linearProgram2(std::span<const OrcaLine> lines, float radius,
                                      const glm::vec2& optVelocity, bool directionOpt, glm::vec2& result);
    /// Infeasible fallback solver minimizing maximum penetration distance across violated agent constraints.
    static void linearProgram3(std::span<const OrcaLine> lines, std::size_t numObstLines,
                               std::size_t beginLine, float radius, glm::vec2& result);

private:
    struct AgentEntry {
        std::uint32_t id = 0;
        RvoAgentConfig config{};
    };

    struct CellBucket {
        std::uint64_t key = 0;
        std::uint32_t head = 0;
        std::uint32_t stamp = 0;
    };

    struct NeighborCandidate {
        float distSq = 0.0f;
        std::uint32_t index = 0;
        std::uint32_t id = 0;
    };

    struct ObstacleCandidate {
        float distSq = 0.0f;
        std::size_t index = 0;
    };

    void rebuildSpatialGrid(float cellSize);
    void queryAgentNeighbors(std::size_t agentIndex, float cellSize,
                             std::vector<NeighborCandidate>& outNeighbors) const;
    void queryNearbyObstacles(std::size_t agentIndex, float rangeSq,
                              std::vector<ObstacleCandidate>& outObstacles) const;
    void buildObstacleOrcaLine(const RvoAgentConfig& selfConfig, const RvoObstacleSegment& segment,
                               float invDt, OrcaLine& outLine) const;
    void buildAgentOrcaLine(const AgentEntry& self, const AgentEntry& other,
                            float invDt, OrcaLine& outLine) const;

    float requestedCellSize_ = 0.0f;
    std::uint32_t nextAgentId_ = 0;
    std::vector<AgentEntry> agents_;
    std::unordered_map<std::uint32_t, std::size_t> idToIndex_;
    std::vector<RvoObstacleSegment> obstacles_;

    std::vector<CellBucket> gridBuckets_;
    std::vector<std::uint32_t> gridNext_;
    std::uint32_t gridStamp_ = 0;

    std::vector<glm::vec2> nextVelocities_;
    std::vector<OrcaLine> orcaLinesScratch_;
    std::vector<NeighborCandidate> neighborsScratch_;
    std::vector<ObstacleCandidate> obstaclesScratch_;
};

} // namespace lumora
