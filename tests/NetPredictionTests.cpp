#include "runtime/net/Bytes.hpp"
#include "runtime/net/ClientPrediction.hpp"

#include <gtest/gtest.h>
#include <glm/geometric.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace lumora::net;

namespace {

PlayerInputCmd makeInput(std::uint32_t tick, glm::vec3 moveDir, float yawDeg = 0.0f,
                         std::uint32_t buttons = InputButton::None, float dt = 1.0f / 60.0f) {
    PlayerInputCmd cmd;
    cmd.tick = tick;
    cmd.moveDir = moveDir;
    cmd.yawDeg = yawDeg;
    cmd.pitchDeg = 0.0f;
    cmd.buttons = buttons;
    cmd.dt = dt;
    return cmd;
}

void expectVecNear(const glm::vec3& actual, const glm::vec3& expected, float eps = 1e-5f) {
    EXPECT_NEAR(actual.x, expected.x, eps);
    EXPECT_NEAR(actual.y, expected.y, eps);
    EXPECT_NEAR(actual.z, expected.z, eps);
}

void expectStateNear(const PredictedState& actual, const PredictedState& expected, float eps = 1e-5f) {
    EXPECT_EQ(actual.tick, expected.tick);
    expectVecNear(actual.position, expected.position, eps);
    expectVecNear(actual.velocity, expected.velocity, eps);
    EXPECT_NEAR(actual.yawDeg, expected.yawDeg, eps);
    EXPECT_EQ(actual.grounded, expected.grounded);
}

std::vector<TransformDeltaState> makeEntities(std::uint32_t count) {
    std::vector<TransformDeltaState> states;
    states.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const float f = static_cast<float>(i);
        TransformDeltaState s;
        s.entityNetId = 1000u + i;
        s.position = glm::vec3{f * 1.5f, 0.25f * f, -f * 0.5f};
        s.rotationDeg = glm::vec3{0.0f, f * 3.0f, 0.0f};
        s.velocity = glm::vec3{1.0f, 0.0f, -0.5f};
        states.push_back(s);
    }
    return states;
}

} // namespace

TEST(NetPrediction, MatchingServerTicksTriggerZeroReconciliations) {
    PredictionController client;
    PredictedState serverState{};

    constexpr std::uint32_t kTotalTicks = 48;
    std::vector<PredictedState> serverHistory;
    serverHistory.reserve(kTotalTicks);

    for (std::uint32_t tick = 1; tick <= kTotalTicks; ++tick) {
        const std::uint32_t buttons =
            (tick == 5 || tick == 30) ? InputButton::Jump :
            (tick >= 10 && tick <= 20) ? InputButton::Sprint : InputButton::None;
        const glm::vec3 dir = (tick <= 35) ? glm::vec3{1.0f, 0.0f, 0.5f} : glm::vec3{0.0f};
        const PlayerInputCmd cmd = makeInput(tick, dir, static_cast<float>(tick) * 2.0f, buttons);

        const PredictedState predicted = client.recordAndPredict(cmd);
        serverState = defaultPredictStep(serverState, cmd);
        serverHistory.push_back(serverState);
        expectStateNear(predicted, serverState, 1e-6f);

        // Simulate authoritative snapshots arriving 6 ticks behind the client.
        if (tick > 6) {
            const PredictedState& delayedServer = serverHistory[static_cast<std::size_t>(tick - 7)];
            EXPECT_FALSE(client.reconcile(delayedServer));
        }
    }

    // Reconcile the remaining buffered server ticks.
    for (std::size_t idx = serverHistory.size() - 6; idx < serverHistory.size(); ++idx) {
        EXPECT_FALSE(client.reconcile(serverHistory[idx]));
    }

    EXPECT_EQ(client.reconciliationCount(), 0u);
    expectVecNear(client.visualErrorOffset(), glm::vec3{0.0f}, 1e-6f);
    expectVecNear(client.renderPosition(), client.currentState().position, 1e-6f);
}

TEST(NetPrediction, ReconcileRespectsPositionTolerance) {
    PredictionController client;
    for (std::uint32_t tick = 1; tick <= 10; ++tick) {
        (void)client.recordAndPredict(makeInput(tick, {1.0f, 0.0f, 0.0f}));
    }

    const auto tick6 = client.stateAtTick(6);
    ASSERT_TRUE(tick6.has_value());

    PredictedState withinTol = *tick6;
    withinTol.position.x += 5e-4f;
    EXPECT_FALSE(client.reconcile(withinTol, 1e-3f));
    EXPECT_EQ(client.reconciliationCount(), 0u);

    PredictedState outsideTol = *tick6;
    outsideTol.position.x += 2e-3f;
    EXPECT_TRUE(client.reconcile(outsideTol, 1e-3f));
    EXPECT_EQ(client.reconciliationCount(), 1u);
}

TEST(NetPrediction, ServerDivergenceAtMinus12ReplaysToGroundTruthAndSmoothsVisualError) {
    PredictionController client;
    PredictedState serverState{};

    constexpr std::uint32_t kLatestTick = 30;
    constexpr std::uint32_t kDivergenceTick = kLatestTick - 12; // tick 18

    std::vector<PlayerInputCmd> commands;
    commands.reserve(kLatestTick);
    PredictedState serverDivergedAt18{};

    for (std::uint32_t tick = 1; tick <= kLatestTick; ++tick) {
        const std::uint32_t buttons = (tick == 12 || tick == 22) ? InputButton::Jump : InputButton::None;
        const PlayerInputCmd cmd = makeInput(tick, {1.0f, 0.0f, -0.25f}, static_cast<float>(tick) * 1.5f, buttons);
        commands.push_back(cmd);

        (void)client.recordAndPredict(cmd);
        serverState = defaultPredictStep(serverState, cmd);

        if (tick == kDivergenceTick) {
            // Authoritative server resolves a wall collision + knockback impulse at tick N - 12.
            serverState.position.x -= 1.25f;
            serverState.position.y += 0.40f;
            serverState.velocity = glm::vec3{-3.0f, 4.0f, 1.5f};
            serverState.grounded = false;
            serverDivergedAt18 = serverState;
        }
    }

    const glm::vec3 preReconcilePos = client.currentState().position;
    EXPECT_TRUE(client.reconcile(serverDivergedAt18, 1e-3f));
    EXPECT_EQ(client.reconciliationCount(), 1u);

    // Client state at tick N must match the server ground truth after replaying N-11..N.
    expectStateNear(client.currentState(), serverState, 1e-5f);

    // Visual error offset absorbs the snap so renderPosition() does not pop on the reconciliation frame.
    expectVecNear(client.renderPosition(), preReconcilePos, 1e-5f);
    const float initialErrorLen = glm::length(client.visualErrorOffset());
    EXPECT_GT(initialErrorLen, 0.5f);

    // Decaying visual error shrinks monotonically toward zero and converges renderPosition() to physics position.
    float prevErrorLen = initialErrorLen;
    for (int frame = 0; frame < 60; ++frame) {
        client.decayVisualError(1.0f / 60.0f, 15.0f);
        const float currentErrorLen = glm::length(client.visualErrorOffset());
        EXPECT_LE(currentErrorLen, prevErrorLen);
        if (prevErrorLen > 1e-5f) {
            EXPECT_LT(currentErrorLen, prevErrorLen);
        }
        prevErrorLen = currentErrorLen;
    }

    EXPECT_LT(glm::length(client.visualErrorOffset()), 1e-4f);
    expectVecNear(client.renderPosition(), serverState.position, 1e-4f);
}

TEST(NetPrediction, RingBufferEvictsOutOfWindowTicksAfter256Slots) {
    PredictionController client;
    PredictedState serverState{};

    constexpr std::uint32_t kTotalTicks = 300;
    PredictedState serverAtTick44{};
    PredictedState serverAtTick45{};

    for (std::uint32_t tick = 1; tick <= kTotalTicks; ++tick) {
        const PlayerInputCmd cmd = makeInput(tick, {1.0f, 0.0f, 0.0f});
        (void)client.recordAndPredict(cmd);
        serverState = defaultPredictStep(serverState, cmd);

        if (tick == 44) {
            serverAtTick44 = serverState;
            serverAtTick44.position.x += 5.0f;
        } else if (tick == 45) {
            serverState.position.x -= 2.0f;
            serverAtTick45 = serverState;
        }
    }

    // Tick 44 was overwritten when tick 300 (= 44 + 256) was recorded.
    EXPECT_FALSE(client.stateAtTick(44).has_value());
    EXPECT_FALSE(client.reconcile(serverAtTick44));
    EXPECT_EQ(client.reconciliationCount(), 0u);

    // Future ticks are also out-of-window.
    PredictedState futureState = serverState;
    futureState.tick = 305;
    futureState.position.x += 10.0f;
    EXPECT_FALSE(client.reconcile(futureState));

    // Tick 45 is the oldest valid slot (300 - 45 == 255 < 256) and replays ticks 46..300 cleanly.
    EXPECT_TRUE(client.stateAtTick(45).has_value());
    EXPECT_TRUE(client.reconcile(serverAtTick45));
    EXPECT_EQ(client.reconciliationCount(), 1u);
    expectStateNear(client.currentState(), serverState, 1e-5f);
}

TEST(NetPrediction, DefaultPredictStepAppliesAccelDragJumpAndGravity) {
    PredictedState state{};

    // Standing jump launches upward and leaves ground.
    const PredictedState jumped = defaultPredictStep(
        state, makeInput(1, {1.0f, 0.0f, 0.0f}, 45.0f, InputButton::Jump, 1.0f / 60.0f));
    EXPECT_FALSE(jumped.grounded);
    EXPECT_GT(jumped.position.y, 0.0f);
    EXPECT_GT(jumped.velocity.y, 0.0f);
    EXPECT_GT(jumped.velocity.x, 0.0f);
    EXPECT_FLOAT_EQ(jumped.yawDeg, 45.0f);

    // Holding Jump while airborne must NOT double-jump; releasing lets it land on y = 0.
    PredictedState current = jumped;
    for (std::uint32_t tick = 2; tick <= 30; ++tick) {
        const float prevVelY = current.velocity.y;
        current = defaultPredictStep(current, makeInput(tick, {0.0f, 0.0f, 0.0f}, 45.0f, InputButton::Jump, 1.0f / 60.0f));
        EXPECT_FALSE(current.grounded);
        EXPECT_LT(current.velocity.y, prevVelY);
    }
    for (std::uint32_t tick = 31; tick <= 90; ++tick) {
        current = defaultPredictStep(current, makeInput(tick, {0.0f, 0.0f, 0.0f}, 45.0f, InputButton::None, 1.0f / 60.0f));
    }
    EXPECT_TRUE(current.grounded);
    EXPECT_FLOAT_EQ(current.position.y, 0.0f);
    EXPECT_FLOAT_EQ(current.velocity.y, 0.0f);
    EXPECT_LT(std::abs(current.velocity.x), 0.01f); // Drag decayed horizontal velocity.

    // Once grounded again, Jump launches cleanly.
    const PredictedState secondJump = defaultPredictStep(
        current, makeInput(91, {0.0f, 0.0f, 0.0f}, 45.0f, InputButton::Jump, 1.0f / 60.0f));
    EXPECT_FALSE(secondJump.grounded);
    EXPECT_GT(secondJump.velocity.y, 0.0f);
}

TEST(NetPrediction, HandlesUint32TickWraparoundAcrossRingBuffer) {
    PredictionController client;
    PredictedState serverState{};
    constexpr std::uint32_t kStartTick = 0xFFFFFFF8u;

    PredictedState diverged{};
    for (std::uint32_t i = 0; i < 20; ++i) {
        const std::uint32_t tick = kStartTick + i;
        const PlayerInputCmd cmd = makeInput(tick, {1.0f, 0.0f, 0.0f});
        (void)client.recordAndPredict(cmd);
        serverState = defaultPredictStep(serverState, cmd);
        if (i == 6) { // tick 0xFFFFFFFE, before wraparound
            serverState.position.z += 2.0f;
            diverged = serverState;
        }
    }

    EXPECT_TRUE(client.reconcile(diverged));
    expectStateNear(client.currentState(), serverState, 1e-5f);
}

TEST(NetPrediction, WireSerializationRoundtripsCommandsAndStates) {
    const PlayerInputCmd cmd = makeInput(42, {0.6f, 0.0f, -0.8f}, 135.5f, InputButton::Jump | InputButton::Sprint, 0.016f);
    const std::vector<std::byte> cmdBytes = PredictionController::encodeInputCmd(cmd);
    const auto decodedCmd = PredictionController::decodeInputCmd(cmdBytes);
    ASSERT_TRUE(decodedCmd.has_value());
    EXPECT_EQ(*decodedCmd, cmd);

    PredictedState state;
    state.tick = 42;
    state.position = {12.5f, 3.25f, -7.75f};
    state.velocity = {4.5f, -1.25f, 0.75f};
    state.yawDeg = 135.5f;
    state.grounded = false;

    const std::vector<std::byte> stateBytes = PredictionController::encodePredictedState(state);
    const auto decodedState = PredictionController::decodePredictedState(stateBytes);
    ASSERT_TRUE(decodedState.has_value());
    EXPECT_EQ(*decodedState, state);

    // Combined stream roundtrip via ByteWriter / ByteReader.
    std::vector<std::byte> packet;
    ByteWriter writer{packet};
    encodeInputCmd(writer, cmd);
    encodePredictedState(writer, state);

    ByteReader reader{packet};
    const auto streamCmd = decodeInputCmd(reader);
    const auto streamState = decodePredictedState(reader);
    ASSERT_TRUE(streamCmd.has_value());
    ASSERT_TRUE(streamState.has_value());
    EXPECT_TRUE(reader.done());
    EXPECT_EQ(*streamCmd, cmd);
    EXPECT_EQ(*streamState, state);

    // Truncated packets fail safely.
    EXPECT_FALSE(decodeInputCmd(std::span<const std::byte>(cmdBytes).first(cmdBytes.size() - 1)).has_value());
    EXPECT_FALSE(decodePredictedState(std::span<const std::byte>(stateBytes).first(stateBytes.size() - 1)).has_value());
}

TEST(NetPrediction, DeltaSnapshotShrinksUnchanged100EntitiesToHeaderOnly) {
    const std::vector<TransformDeltaState> baseline = makeEntities(100);
    const std::vector<std::byte> encoded = DeltaSnapshot::encodeDelta(baseline, baseline);

    EXPECT_LT(encoded.size(), 16u);

    const auto decoded = DeltaSnapshot::decodeDelta(baseline, encoded);
    ASSERT_TRUE(decoded.has_value());
    EXPECT_EQ(*decoded, baseline);
}

TEST(NetPrediction, DeltaSnapshotRoundtripsFullAndPartialUpdates) {
    const std::vector<TransformDeltaState> current100 = makeEntities(100);

    // Full snapshot from empty baseline.
    const std::vector<std::byte> fullPayload = DeltaSnapshot::encodeDelta({}, current100);
    const auto fullDecoded = DeltaSnapshot::decodeDelta({}, fullPayload);
    ASSERT_TRUE(fullDecoded.has_value());
    EXPECT_EQ(*fullDecoded, current100);

    // Partial update: modify only specific fields on 3 out of 100 entities.
    std::vector<TransformDeltaState> next100 = current100;
    next100[5].position += glm::vec3{2.0f, -0.5f, 1.0f};
    next100[42].rotationDeg.y += 45.0f;
    next100[91].velocity = glm::vec3{0.0f, 9.5f, 0.0f};
    next100[91].position.y += 0.25f;

    const std::vector<std::byte> partialPayload = DeltaSnapshot::encodeDelta(current100, next100);
    EXPECT_LT(partialPayload.size(), 128u);
    EXPECT_LT(partialPayload.size() * 10u, fullPayload.size());

    const auto partialDecoded = DeltaSnapshot::decodeDelta(current100, partialPayload);
    ASSERT_TRUE(partialDecoded.has_value());
    EXPECT_EQ(*partialDecoded, next100);
}

TEST(NetPrediction, DeltaSnapshotHandlesSpawnsDespawnsReordersAndCorruptInput) {
    const std::vector<TransformDeltaState> baseline = makeEntities(6);

    // Remove entity 1001, spawn entity 2000, mutate entity 1003, and reorder entities.
    std::vector<TransformDeltaState> current = {
        baseline[5],
        baseline[2],
        baseline[3],
        TransformDeltaState{2000u, {9.0f, 1.0f, 2.0f}, {0.0f, 90.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},
        baseline[0],
        baseline[4],
    };
    current[2].position.x += 12.0f;

    const std::vector<std::byte> encoded = DeltaSnapshot::encodeDelta(baseline, current);
    const auto decoded = DeltaSnapshot::decodeDelta(baseline, encoded);
    ASSERT_TRUE(decoded.has_value());
    EXPECT_EQ(*decoded, current);

    // Empty baseline and empty current -> header-only (< 16 bytes) and decodes to empty vector.
    const std::vector<std::byte> emptyEncoded = DeltaSnapshot::encodeDelta({}, {});
    EXPECT_LT(emptyEncoded.size(), 16u);
    const auto emptyDecoded = DeltaSnapshot::decodeDelta({}, emptyEncoded);
    ASSERT_TRUE(emptyDecoded.has_value());
    EXPECT_TRUE(emptyDecoded->empty());

    // Truncated or corrupted payloads return nullopt.
    EXPECT_FALSE(DeltaSnapshot::decodeDelta(baseline, {}).has_value());
    EXPECT_FALSE(DeltaSnapshot::decodeDelta(baseline, std::span<const std::byte>(encoded).first(encoded.size() - 1)).has_value());

    std::vector<std::byte> badVersion = encoded;
    badVersion[0] = std::byte{0x07};
    EXPECT_FALSE(DeltaSnapshot::decodeDelta(baseline, badVersion).has_value());
}
