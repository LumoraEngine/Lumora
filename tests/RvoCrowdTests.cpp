#include "runtime/RvoCrowd.hpp"

#include <gtest/gtest.h>

#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

using namespace lumora;

namespace {

constexpr float kPi = 3.14159265358979323846f;

float pointSegmentDistance(glm::vec2 p, glm::vec2 a, glm::vec2 b) {
    const glm::vec2 ab = b - a;
    const float lenSq = glm::dot(ab, ab);
    if (lenSq <= 1e-10f) {
        return glm::distance(p, a);
    }
    const float t = std::clamp(glm::dot(p - a, ab) / lenSq, 0.0f, 1.0f);
    return glm::distance(p, a + t * ab);
}

glm::vec2 desiredVelocityToGoal(glm::vec2 pos, glm::vec2 goal, float maxSpeed, float slowRadius = 0.25f) {
    const glm::vec2 toGoal = goal - pos;
    const float dist = glm::length(toGoal);
    if (dist <= 1e-5f) {
        return glm::vec2(0.0f, 0.0f);
    }
    const float speed = (dist < slowRadius) ? maxSpeed * (dist / slowRadius) : maxSpeed;
    return (toGoal / dist) * speed;
}

} // namespace

TEST(RvoCrowd, AgentAndObstacleManagement) {
    RvoCrowd crowd;
    EXPECT_EQ(crowd.agentCount(), 0u);
    EXPECT_EQ(crowd.obstacleCount(), 0u);

    const std::uint32_t a0 = crowd.addAgent({.position = {1.0f, 2.0f}, .radius = 0.4f, .maxSpeed = 3.0f});
    const std::uint32_t a1 = crowd.addAgent({.position = {3.0f, 4.0f}, .radius = 0.6f, .maxSpeed = 2.5f});
    const std::uint32_t a2 = crowd.addAgent({.position = {5.0f, 6.0f}, .radius = 0.5f, .maxSpeed = 1.5f});
    EXPECT_EQ(crowd.agentCount(), 3u);
    EXPECT_TRUE(crowd.hasAgent(a0));
    EXPECT_TRUE(crowd.hasAgent(a1));
    EXPECT_TRUE(crowd.hasAgent(a2));

    crowd.setPreferredVelocity(a1, {1.0f, -0.5f});
    crowd.setPosition(a2, {-2.0f, 7.0f});
    EXPECT_FLOAT_EQ(crowd.agent(a1).preferredVelocity.x, 1.0f);
    EXPECT_FLOAT_EQ(crowd.agent(a1).preferredVelocity.y, -0.5f);
    EXPECT_FLOAT_EQ(crowd.agent(a2).position.x, -2.0f);
    EXPECT_FLOAT_EQ(crowd.agent(a2).position.y, 7.0f);

    // Removing the first agent swaps-and-pops internally while preserving remaining IDs.
    EXPECT_TRUE(crowd.removeAgent(a0));
    EXPECT_FALSE(crowd.removeAgent(a0));
    EXPECT_EQ(crowd.agentCount(), 2u);
    EXPECT_FALSE(crowd.hasAgent(a0));
    EXPECT_EQ(crowd.findAgent(a0), nullptr);
    ASSERT_NE(crowd.findAgent(a1), nullptr);
    ASSERT_NE(crowd.findAgent(a2), nullptr);
    EXPECT_FLOAT_EQ(crowd.agent(a1).radius, 0.6f);
    EXPECT_FLOAT_EQ(crowd.agent(a2).position.x, -2.0f);
    EXPECT_THROW((void)crowd.agent(a0), std::out_of_range);

    crowd.addObstacleSegment({0.0f, 0.0f}, {4.0f, 0.0f});
    crowd.addObstacleSegment({4.0f, 0.0f}, {4.0f, 4.0f});
    EXPECT_EQ(crowd.obstacleCount(), 2u);
    EXPECT_EQ(crowd.obstacles().size(), 2u);
    crowd.clearObstacles();
    EXPECT_EQ(crowd.obstacleCount(), 0u);

    crowd.clearAgents();
    EXPECT_EQ(crowd.agentCount(), 0u);
}

TEST(RvoCrowd, LinearProgrammingFeasibleAndInfeasibleFallback) {
    // Feasible intersection: x >= 0.5, y >= 0.5 within disk radius 2.0.
    const std::vector<OrcaLine> feasibleLines = {
        {.point = {0.5f, 0.0f}, .direction = {0.0f, -1.0f}}, // left is +X => x >= 0.5
        {.point = {0.0f, 0.5f}, .direction = {1.0f, 0.0f}},  // left is +Y => y >= 0.5
    };
    glm::vec2 result{0.0f, 0.0f};
    const std::size_t fail = RvoCrowd::linearProgram2(feasibleLines, 2.0f, {0.0f, 0.0f}, false, result);
    EXPECT_EQ(fail, feasibleLines.size());
    EXPECT_NEAR(result.x, 0.5f, 1e-4f);
    EXPECT_NEAR(result.y, 0.5f, 1e-4f);

    // Infeasible agent lines: hard obstacle constraint y >= 0.0, plus conflicting agent lines
    // x <= -1.0 and x >= 1.0. LP3 should keep y >= 0.0 satisfied and balance x = 0.0.
    const std::vector<OrcaLine> infeasibleLines = {
        {.point = {0.0f, 0.0f}, .direction = {1.0f, 0.0f}},   // obstacle: y >= 0
        {.point = {-1.0f, 0.0f}, .direction = {0.0f, 1.0f}},  // agent 1: x <= -1
        {.point = {1.0f, 0.0f}, .direction = {0.0f, -1.0f}},  // agent 2: x >= 1
    };
    const std::size_t lineFail = RvoCrowd::linearProgram2(infeasibleLines, 2.0f, {0.0f, 1.0f}, false, result);
    EXPECT_EQ(lineFail, 2u);
    RvoCrowd::linearProgram3(infeasibleLines, 1u, lineFail, 2.0f, result);
    EXPECT_GE(result.y, -1e-4f);
    EXPECT_NEAR(result.x, 0.0f, 1e-4f);
    EXPECT_LE(glm::length(result), 2.0f + 1e-4f);

    // Infeasible obstacle lines themselves (beginLine < numObstLines): should balance violation cleanly.
    const std::vector<OrcaLine> infeasibleObstacles = {
        {.point = {-0.8f, 0.0f}, .direction = {0.0f, 1.0f}},   // x <= -0.8
        {.point = {0.8f, 0.0f}, .direction = {0.0f, -1.0f}},   // x >= 0.8
    };
    const std::size_t obstFail = RvoCrowd::linearProgram2(infeasibleObstacles, 2.0f, {0.0f, 0.0f}, false, result);
    EXPECT_EQ(obstFail, 1u);
    RvoCrowd::linearProgram3(infeasibleObstacles, 2u, obstFail, 2.0f, result);
    EXPECT_NEAR(result.x, 0.0f, 1e-4f);
    EXPECT_LE(glm::length(result), 2.0f + 1e-4f);
}

TEST(RvoCrowd, DegenerateStepAndPointObstacleHandledGracefully) {
    RvoCrowd crowd;
    crowd.addObstacleSegment({0.0f, 0.0f}, {0.0f, 0.0f}); // degenerate point obstacle
    const std::uint32_t id = crowd.addAgent({
        .position = {-3.0f, 0.0f},
        .velocity = {0.0f, 0.0f},
        .preferredVelocity = {2.0f, 0.0f},
        .radius = 0.5f,
        .maxSpeed = 2.0f,
        .timeHorizonObst = 3.0f,
    });

    // Zero or negative dt is a safe no-op.
    crowd.step(0.0f);
    crowd.step(-0.1f);
    EXPECT_FLOAT_EQ(crowd.agent(id).position.x, -3.0f);
    EXPECT_FLOAT_EQ(crowd.agent(id).position.y, 0.0f);

    for (int i = 0; i < 60 * 5; ++i) {
        const glm::vec2 pos = crowd.agent(id).position;
        crowd.setPreferredVelocity(id, desiredVelocityToGoal(pos, {3.0f, 0.0f}, 2.0f));
        crowd.step(1.0f / 60.0f);
        EXPECT_TRUE(std::isfinite(crowd.agent(id).position.x));
        EXPECT_TRUE(std::isfinite(crowd.agent(id).position.y));
        EXPECT_GE(glm::length(crowd.agent(id).position), 0.5f - 0.02f);
    }
    EXPECT_LT(glm::distance(crowd.agent(id).position, glm::vec2{3.0f, 0.0f}), 0.2f);
}

TEST(RvoCrowd, TwoAgentsHeadOnSwapCleanlyWithoutSticking) {
    for (const bool startFromRest : {true, false}) {
        RvoCrowd crowd;
        constexpr float r1 = 0.5f;
        constexpr float r2 = 0.5f;
        constexpr float maxSpeed = 2.0f;
        const glm::vec2 startA{-5.0f, 0.0f};
        const glm::vec2 startB{5.0f, 0.0f};
        const glm::vec2 goalA = startB;
        const glm::vec2 goalB = startA;

        const std::uint32_t idA = crowd.addAgent({
            .position = startA,
            .velocity = startFromRest ? glm::vec2{0.0f, 0.0f} : glm::vec2{maxSpeed, 0.0f},
            .preferredVelocity = {maxSpeed, 0.0f},
            .radius = r1,
            .maxSpeed = maxSpeed,
            .neighborDist = 15.0f,
            .maxNeighbors = 10,
            .timeHorizon = 5.0f,
            .timeHorizonObst = 5.0f,
        });
        const std::uint32_t idB = crowd.addAgent({
            .position = startB,
            .velocity = startFromRest ? glm::vec2{0.0f, 0.0f} : glm::vec2{-maxSpeed, 0.0f},
            .preferredVelocity = {-maxSpeed, 0.0f},
            .radius = r2,
            .maxSpeed = maxSpeed,
            .neighborDist = 15.0f,
            .maxNeighbors = 10,
            .timeHorizon = 5.0f,
            .timeHorizonObst = 5.0f,
        });

        constexpr float dt = 1.0f / 60.0f;
        float minDist = std::numeric_limits<float>::infinity();
        bool passedOrigin = false;

        for (int step = 0; step < 60 * 15; ++step) {
            const glm::vec2 posA = crowd.agent(idA).position;
            const glm::vec2 posB = crowd.agent(idB).position;
            crowd.setPreferredVelocity(idA, desiredVelocityToGoal(posA, goalA, maxSpeed, 0.1f));
            crowd.setPreferredVelocity(idB, desiredVelocityToGoal(posB, goalB, maxSpeed, 0.1f));
            crowd.step(dt);

            const glm::vec2 newPosA = crowd.agent(idA).position;
            const glm::vec2 newPosB = crowd.agent(idB).position;
            minDist = std::min(minDist, glm::distance(newPosA, newPosB));
            if (newPosA.x > 1.0f && newPosB.x < -1.0f) {
                passedOrigin = true;
            }
            if (glm::distance(newPosA, goalA) < 0.05f && glm::distance(newPosB, goalB) < 0.05f) {
                break;
            }
        }

        EXPECT_TRUE(passedOrigin) << "startFromRest=" << startFromRest;
        EXPECT_GE(minDist, r1 + r2 - 0.02f) << "startFromRest=" << startFromRest;
        EXPECT_LT(glm::distance(crowd.agent(idA).position, goalA), 0.15f) << "startFromRest=" << startFromRest;
        EXPECT_LT(glm::distance(crowd.agent(idB).position, goalB), 0.15f) << "startFromRest=" << startFromRest;
    }
}

TEST(RvoCrowd, SixteenAgentAntipodalCircleCrossing) {
    RvoCrowd crowd;
    constexpr std::size_t kAgentCount = 16;
    constexpr float kCircleRadius = 10.0f;
    constexpr float kAgentRadius = 0.45f;
    constexpr float kMaxSpeed = 2.0f;

    std::vector<std::uint32_t> ids;
    std::vector<glm::vec2> goals;
    ids.reserve(kAgentCount);
    goals.reserve(kAgentCount);

    for (std::size_t i = 0; i < kAgentCount; ++i) {
        const float angle = (2.0f * kPi * static_cast<float>(i)) / static_cast<float>(kAgentCount);
        const glm::vec2 start{kCircleRadius * std::cos(angle), kCircleRadius * std::sin(angle)};
        const glm::vec2 goal = -start;
        goals.push_back(goal);
        ids.push_back(crowd.addAgent({
            .position = start,
            .velocity = {0.0f, 0.0f},
            .preferredVelocity = desiredVelocityToGoal(start, goal, kMaxSpeed),
            .radius = kAgentRadius,
            .maxSpeed = kMaxSpeed,
            .neighborDist = 15.0f,
            .maxNeighbors = 15,
            .timeHorizon = 5.0f,
            .timeHorizonObst = 5.0f,
        }));
    }

    constexpr float dt = 1.0f / 60.0f;
    float minPairDist = std::numeric_limits<float>::infinity();

    for (int step = 0; step < 60 * 25; ++step) {
        bool allClose = true;
        for (std::size_t i = 0; i < kAgentCount; ++i) {
            const glm::vec2 pos = crowd.agent(ids[i]).position;
            const float distToGoal = glm::distance(pos, goals[i]);
            if (distToGoal > 0.15f) {
                allClose = false;
            }
            crowd.setPreferredVelocity(ids[i], desiredVelocityToGoal(pos, goals[i], kMaxSpeed, 0.3f));
        }
        if (allClose) {
            break;
        }

        crowd.step(dt);

        for (std::size_t i = 0; i < kAgentCount; ++i) {
            for (std::size_t j = i + 1; j < kAgentCount; ++j) {
                const float d = glm::distance(crowd.agent(ids[i]).position, crowd.agent(ids[j]).position);
                minPairDist = std::min(minPairDist, d);
            }
        }
    }

    EXPECT_GE(minPairDist, 2.0f * kAgentRadius - 1e-3f) << "Zero inter-agent penetrations expected";
    for (std::size_t i = 0; i < kAgentCount; ++i) {
        EXPECT_LT(glm::distance(crowd.agent(ids[i]).position, goals[i]), 0.3f) << "Agent " << i;
    }
}

TEST(RvoCrowd, AgentsSteerSmoothlyAroundWallAndCorridorObstacles) {
    // Part 1: Single agent steers smoothly around a blocking wall segment toward its goal.
    for (const auto& [wallA, wallB, start, goal] : std::vector<std::tuple<glm::vec2, glm::vec2, glm::vec2, glm::vec2>>{
             {{0.0f, -2.5f}, {0.0f, 1.0f}, {-4.0f, -0.5f}, {4.5f, 1.8f}},
             {{-1.5f, -1.5f}, {1.5f, 1.5f}, {-4.0f, 0.5f}, {4.0f, 0.0f}},
         }) {
        RvoCrowd crowd;
        constexpr float kRadius = 0.5f;
        constexpr float kMaxSpeed = 2.5f;
        crowd.addObstacleSegment(wallA, wallB);

        const std::uint32_t agentId = crowd.addAgent({
            .position = start,
            .velocity = {0.0f, 0.0f},
            .preferredVelocity = desiredVelocityToGoal(start, goal, kMaxSpeed),
            .radius = kRadius,
            .maxSpeed = kMaxSpeed,
            .neighborDist = 10.0f,
            .maxNeighbors = 8,
            .timeHorizon = 4.0f,
            .timeHorizonObst = 4.0f,
        });

        constexpr float dt = 1.0f / 60.0f;
        float minWallDist = std::numeric_limits<float>::infinity();
        for (int step = 0; step < 60 * 15; ++step) {
            const glm::vec2 pos = crowd.agent(agentId).position;
            if (glm::distance(pos, goal) < 0.1f) {
                break;
            }
            crowd.setPreferredVelocity(agentId, desiredVelocityToGoal(pos, goal, kMaxSpeed, 0.25f));
            crowd.step(dt);
            minWallDist = std::min(minWallDist, pointSegmentDistance(crowd.agent(agentId).position, wallA, wallB));
        }

        EXPECT_GE(minWallDist, kRadius - 0.02f);
        EXPECT_LT(glm::distance(crowd.agent(agentId).position, goal), 0.2f);
    }

    // Part 2: Narrowing corridor funnel + central diamond obstacle with opposing agents crossing.
    {
        RvoCrowd crowd;
        constexpr float kRadius = 0.4f;
        constexpr float kMaxSpeed = 2.0f;

        // Corridor outer walls with a narrowed middle section: Z in [-3, 3] narrowing to [-2.2, 2.2].
        crowd.addObstacleSegment({-8.0f, 3.0f}, {-2.0f, 2.2f});
        crowd.addObstacleSegment({-2.0f, 2.2f}, {2.0f, 2.2f});
        crowd.addObstacleSegment({2.0f, 2.2f}, {8.0f, 3.0f});

        crowd.addObstacleSegment({-8.0f, -3.0f}, {-2.0f, -2.2f});
        crowd.addObstacleSegment({-2.0f, -2.2f}, {2.0f, -2.2f});
        crowd.addObstacleSegment({2.0f, -2.2f}, {8.0f, -3.0f});

        // Central diamond obstacle in the corridor creating upper and lower passing lanes.
        crowd.addObstacleSegment({-1.2f, 0.0f}, {0.0f, 0.7f});
        crowd.addObstacleSegment({0.0f, 0.7f}, {1.2f, 0.0f});
        crowd.addObstacleSegment({1.2f, 0.0f}, {0.0f, -0.7f});
        crowd.addObstacleSegment({0.0f, -0.7f}, {-1.2f, 0.0f});

        struct AgentGoal {
            std::uint32_t id = 0;
            glm::vec2 goal{0.0f, 0.0f};
        };
        std::vector<AgentGoal> entries;
        const std::vector<std::pair<glm::vec2, glm::vec2>> startGoalPairs = {
            {{-6.5f, 0.6f}, {6.5f, 1.3f}},
            {{-6.5f, -0.6f}, {6.5f, -1.3f}},
            {{6.5f, 1.3f}, {-6.5f, 1.2f}},
            {{6.5f, -1.3f}, {-6.5f, -1.2f}},
        };

        for (const auto& [start, goal] : startGoalPairs) {
            const std::uint32_t id = crowd.addAgent({
                .position = start,
                .velocity = {0.0f, 0.0f},
                .preferredVelocity = desiredVelocityToGoal(start, goal, kMaxSpeed),
                .radius = kRadius,
                .maxSpeed = kMaxSpeed,
                .neighborDist = 10.0f,
                .maxNeighbors = 8,
                .timeHorizon = 4.0f,
                .timeHorizonObst = 3.0f,
            });
            entries.push_back({id, goal});
        }

        constexpr float dt = 1.0f / 60.0f;
        float minObstacleDist = std::numeric_limits<float>::infinity();
        float minAgentDist = std::numeric_limits<float>::infinity();

        for (int step = 0; step < 60 * 25; ++step) {
            bool allArrived = true;
            for (const auto& entry : entries) {
                const glm::vec2 pos = crowd.agent(entry.id).position;
                if (glm::distance(pos, entry.goal) > 0.2f) {
                    allArrived = false;
                }
                crowd.setPreferredVelocity(entry.id, desiredVelocityToGoal(pos, entry.goal, kMaxSpeed, 0.3f));
            }
            if (allArrived) {
                break;
            }

            crowd.step(dt);

            for (std::size_t i = 0; i < entries.size(); ++i) {
                const glm::vec2 pi = crowd.agent(entries[i].id).position;
                for (const auto& seg : crowd.obstacles()) {
                    minObstacleDist = std::min(minObstacleDist, pointSegmentDistance(pi, seg.a, seg.b));
                }
                for (std::size_t j = i + 1; j < entries.size(); ++j) {
                    const glm::vec2 pj = crowd.agent(entries[j].id).position;
                    minAgentDist = std::min(minAgentDist, glm::distance(pi, pj));
                }
            }
        }

        EXPECT_GE(minObstacleDist, kRadius - 0.02f);
        EXPECT_GE(minAgentDist, 2.0f * kRadius - 0.02f);
        for (const auto& entry : entries) {
            EXPECT_LT(glm::distance(crowd.agent(entry.id).position, entry.goal), 0.3f);
        }
    }

    // Part 3: Two-sided obstacle segment blocks penetration symmetrically from either side.
    {
        RvoCrowd crowd;
        constexpr float kRadius = 0.5f;
        const glm::vec2 wallA{0.0f, -3.0f};
        const glm::vec2 wallB{0.0f, 3.0f};
        crowd.addObstacleSegment(wallA, wallB);

        const std::uint32_t leftAgent = crowd.addAgent({
            .position = {-3.0f, 0.0f},
            .velocity = {0.0f, 0.0f},
            .preferredVelocity = {2.0f, 0.0f},
            .radius = kRadius,
            .maxSpeed = 2.0f,
            .timeHorizonObst = 3.0f,
        });
        const std::uint32_t rightAgent = crowd.addAgent({
            .position = {3.0f, 0.0f},
            .velocity = {0.0f, 0.0f},
            .preferredVelocity = {-2.0f, 0.0f},
            .radius = kRadius,
            .maxSpeed = 2.0f,
            .timeHorizonObst = 3.0f,
        });

        for (int step = 0; step < 60 * 5; ++step) {
            crowd.step(1.0f / 60.0f);
        }

        EXPECT_LE(crowd.agent(leftAgent).position.x, -kRadius + 0.02f);
        EXPECT_GE(crowd.agent(rightAgent).position.x, kRadius - 0.02f);
    }
}

TEST(RvoCrowd, ConcurrentDoubleBufferedStepIsOrderIndependent) {
    auto runSymmetricScenario = [](bool reverseAddOrder) {
        RvoCrowd crowd;
        const RvoAgentConfig cfgA{
            .position = {-3.0f, 1.0f},
            .velocity = {1.5f, -0.5f},
            .preferredVelocity = {1.5f, -0.5f},
            .radius = 0.5f,
            .maxSpeed = 2.0f,
        };
        const RvoAgentConfig cfgB{
            .position = {3.0f, -1.0f},
            .velocity = {-1.5f, 0.5f},
            .preferredVelocity = {-1.5f, 0.5f},
            .radius = 0.5f,
            .maxSpeed = 2.0f,
        };

        std::uint32_t idA = 0;
        std::uint32_t idB = 0;
        if (!reverseAddOrder) {
            idA = crowd.addAgent(cfgA);
            idB = crowd.addAgent(cfgB);
        } else {
            idB = crowd.addAgent(cfgB);
            idA = crowd.addAgent(cfgA);
        }

        for (int s = 0; s < 30; ++s) {
            crowd.step(1.0f / 60.0f);
        }
        return std::pair<glm::vec2, glm::vec2>{crowd.agent(idA).position, crowd.agent(idB).position};
    };

    const auto [posA1, posB1] = runSymmetricScenario(false);
    const auto [posA2, posB2] = runSymmetricScenario(true);

    EXPECT_NEAR(posA1.x, posA2.x, 1e-5f);
    EXPECT_NEAR(posA1.y, posA2.y, 1e-5f);
    EXPECT_NEAR(posB1.x, posB2.x, 1e-5f);
    EXPECT_NEAR(posB1.y, posB2.y, 1e-5f);
    EXPECT_NEAR(posA1.x, -posB1.x, 1e-5f);
    EXPECT_NEAR(posA1.y, -posB1.y, 1e-5f);
}

TEST(RvoCrowd, SpatialHashHandlesLargeMultiClusterCrowdWithoutOverlap) {
    RvoCrowd crowd(3.0f);
    constexpr float kRadius = 0.4f;
    constexpr float kMaxSpeed = 2.0f;
    constexpr int kRows = 4;
    constexpr int kCols = 8;

    struct Item {
        std::uint32_t id = 0;
        glm::vec2 goal{0.0f, 0.0f};
    };
    std::vector<Item> items;
    items.reserve(kRows * kCols * 2);

    // Two opposing 4x8 grids (64 agents total) crossing horizontally through each other.
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kCols; ++c) {
            const float z = (static_cast<float>(r) - 1.5f) * 2.2f;
            const float xLeft = -12.0f - static_cast<float>(c) * 2.0f;
            const float xRight = 12.0f + static_cast<float>(c) * 2.0f;

            const glm::vec2 goalL{12.0f + static_cast<float>(c) * 2.0f, z};
            const glm::vec2 goalR{-12.0f - static_cast<float>(c) * 2.0f, z};

            items.push_back({
                crowd.addAgent({
                    .position = {xLeft, z},
                    .velocity = {0.0f, 0.0f},
                    .preferredVelocity = desiredVelocityToGoal({xLeft, z}, goalL, kMaxSpeed),
                    .radius = kRadius,
                    .maxSpeed = kMaxSpeed,
                    .neighborDist = 8.0f,
                    .maxNeighbors = 10,
                    .timeHorizon = 4.0f,
                    .timeHorizonObst = 4.0f,
                }),
                goalL,
            });
            items.push_back({
                crowd.addAgent({
                    .position = {xRight, z},
                    .velocity = {0.0f, 0.0f},
                    .preferredVelocity = desiredVelocityToGoal({xRight, z}, goalR, kMaxSpeed),
                    .radius = kRadius,
                    .maxSpeed = kMaxSpeed,
                    .neighborDist = 8.0f,
                    .maxNeighbors = 10,
                    .timeHorizon = 4.0f,
                    .timeHorizonObst = 4.0f,
                }),
                goalR,
            });
        }
    }

    constexpr float dt = 1.0f / 30.0f;
    float minPairDist = std::numeric_limits<float>::infinity();
    for (int step = 0; step < 30 * 35; ++step) {
        bool done = true;
        for (const auto& item : items) {
            const glm::vec2 pos = crowd.agent(item.id).position;
            if (glm::distance(pos, item.goal) > 0.25f) {
                done = false;
            }
            crowd.setPreferredVelocity(item.id, desiredVelocityToGoal(pos, item.goal, kMaxSpeed, 0.4f));
        }
        if (done) {
            break;
        }
        crowd.step(dt);

        for (std::size_t i = 0; i < items.size(); ++i) {
            for (std::size_t j = i + 1; j < items.size(); ++j) {
                minPairDist = std::min(
                    minPairDist,
                    glm::distance(crowd.agent(items[i].id).position, crowd.agent(items[j].id).position));
            }
        }
    }

    EXPECT_GE(minPairDist, 2.0f * kRadius - 0.02f);
    for (const auto& item : items) {
        EXPECT_LT(glm::distance(crowd.agent(item.id).position, item.goal), 0.35f);
    }
}
