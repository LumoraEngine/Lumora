#pragma once

#include "runtime/net/Bytes.hpp"

#include <glm/vec3.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace lumora::net {

/// Bitflags for `PlayerInputCmd::buttons`.
namespace InputButton {
inline constexpr std::uint32_t None = 0u;
inline constexpr std::uint32_t Jump = 1u << 0;
inline constexpr std::uint32_t Sprint = 1u << 1;
inline constexpr std::uint32_t Crouch = 1u << 2;
} // namespace InputButton

/// One fixed-tick player input command sent from client to authoritative server and stored
/// locally for prediction replay.
struct PlayerInputCmd {
    std::uint32_t tick = 0;
    glm::vec3 moveDir{0.0f};
    float yawDeg = 0.0f;
    float pitchDeg = 0.0f;
    std::uint32_t buttons = InputButton::None;
    float dt = 1.0f / 60.0f;

    friend bool operator==(const PlayerInputCmd&, const PlayerInputCmd&) = default;
};

/// Authoritative or client-predicted kinematic character state at a simulation tick.
struct PredictedState {
    std::uint32_t tick = 0;
    glm::vec3 position{0.0f};
    glm::vec3 velocity{0.0f};
    float yawDeg = 0.0f;
    bool grounded = true;

    friend bool operator==(const PredictedState&, const PredictedState&) = default;
};

/// Deterministic state transition function shared between client prediction and server simulation.
using InputStepFn = std::function<PredictedState(const PredictedState&, const PlayerInputCmd&)>;

/// Default deterministic kinematic integrator with horizontal acceleration, drag, gravity,
/// jump impulse when grounded, and a flat y = 0 ground plane.
[[nodiscard]] PredictedState defaultPredictStep(const PredictedState& state, const PlayerInputCmd& cmd);

/// Encodes / decodes `PlayerInputCmd` in little-endian wire order via `ByteWriter` / `ByteReader`.
void encodeInputCmd(ByteWriter& writer, const PlayerInputCmd& cmd);
[[nodiscard]] std::vector<std::byte> encodeInputCmd(const PlayerInputCmd& cmd);
[[nodiscard]] std::optional<PlayerInputCmd> decodeInputCmd(ByteReader& reader);
[[nodiscard]] std::optional<PlayerInputCmd> decodeInputCmd(std::span<const std::byte> bytes);

/// Encodes / decodes `PredictedState` in little-endian wire order via `ByteWriter` / `ByteReader`.
void encodePredictedState(ByteWriter& writer, const PredictedState& state);
[[nodiscard]] std::vector<std::byte> encodePredictedState(const PredictedState& state);
[[nodiscard]] std::optional<PredictedState> decodePredictedState(ByteReader& reader);
[[nodiscard]] std::optional<PredictedState> decodePredictedState(std::span<const std::byte> bytes);

/// Client-side prediction & server reconciliation controller backed by fixed 256-slot circular
/// input and state history buffers.
class PredictionController {
public:
    static constexpr std::size_t kHistoryCapacity = 256;

    PredictionController();
    explicit PredictionController(InputStepFn stepFn, PredictedState initialState = {});

    /// Replaces the simulation step function (falls back to `defaultPredictStep` if empty).
    void setStepFunction(InputStepFn stepFn);
    /// Clears all buffered history and resets prediction state and visual error offset.
    void reset(const PredictedState& initialState = {});

    /// Applies `cmd` immediately using `stepFn_`, records `(cmd, state)` in the circular history
    /// buffers, and returns the resulting predicted state.
    PredictedState recordAndPredict(const PlayerInputCmd& cmd);

    /// Compares `authoritativeState` against the recorded prediction at `authoritativeState.tick`.
    /// Returns false if the tick is outside the 256-tick history window or within `posTolerance`.
    /// On mismatch, snaps to `authoritativeState`, deterministically replays stored inputs for
    /// `(authoritativeState.tick + 1 .. latestTick)`, accumulates the position discrepancy into
    /// `visualErrorOffset_` for smooth error decay, increments `reconciliationCount_`, and returns true.
    bool reconcile(const PredictedState& authoritativeState, float posTolerance = 1e-3f);

    /// Exponentially decays `visualErrorOffset_` toward zero over time (`exp(-decayRate * dt)`).
    void decayVisualError(float dt, float decayRate = 15.0f);

    /// Smoothed visual position (`currentState().position + visualErrorOffset()`).
    [[nodiscard]] glm::vec3 renderPosition() const noexcept { return currentState_.position + visualErrorOffset_; }
    [[nodiscard]] const PredictedState& currentState() const noexcept { return currentState_; }
    [[nodiscard]] const glm::vec3& visualErrorOffset() const noexcept { return visualErrorOffset_; }
    [[nodiscard]] std::uint64_t reconciliationCount() const noexcept { return reconciliationCount_; }
    [[nodiscard]] std::uint32_t latestTick() const noexcept { return latestTick_; }
    [[nodiscard]] bool hasHistory() const noexcept { return hasHistory_; }

    [[nodiscard]] std::optional<PredictedState> stateAtTick(std::uint32_t tick) const;
    [[nodiscard]] std::optional<PlayerInputCmd> inputAtTick(std::uint32_t tick) const;

    static void encodeInputCmd(ByteWriter& writer, const PlayerInputCmd& cmd);
    [[nodiscard]] static std::vector<std::byte> encodeInputCmd(const PlayerInputCmd& cmd);
    [[nodiscard]] static std::optional<PlayerInputCmd> decodeInputCmd(ByteReader& reader);
    [[nodiscard]] static std::optional<PlayerInputCmd> decodeInputCmd(std::span<const std::byte> bytes);

    static void encodePredictedState(ByteWriter& writer, const PredictedState& state);
    [[nodiscard]] static std::vector<std::byte> encodePredictedState(const PredictedState& state);
    [[nodiscard]] static std::optional<PredictedState> decodePredictedState(ByteReader& reader);
    [[nodiscard]] static std::optional<PredictedState> decodePredictedState(std::span<const std::byte> bytes);

private:
    [[nodiscard]] static constexpr std::size_t slotFor(std::uint32_t tick) noexcept {
        return static_cast<std::size_t>(tick) % kHistoryCapacity;
    }
    [[nodiscard]] bool isTickInWindow(std::uint32_t tick) const noexcept;

    InputStepFn stepFn_;
    PredictedState currentState_{};
    glm::vec3 visualErrorOffset_{0.0f};
    std::uint64_t reconciliationCount_ = 0;
    std::uint32_t latestTick_ = 0;
    bool hasHistory_ = false;

    std::array<PlayerInputCmd, kHistoryCapacity> inputHistory_{};
    std::array<PredictedState, kHistoryCapacity> stateHistory_{};
    std::array<bool, kHistoryCapacity> occupied_{};
};

/// Replicated transform & velocity state for one networked entity in a delta snapshot.
struct TransformDeltaState {
    std::uint32_t entityNetId = 0;
    glm::vec3 position{0.0f};
    glm::vec3 rotationDeg{0.0f};
    glm::vec3 velocity{0.0f};

    friend bool operator==(const TransformDeltaState&, const TransformDeltaState&) = default;
};

/// Bitmask-compressed delta snapshot codec comparing a baseline frame against the current frame.
/// Unchanged frames emit a compact header-only payload (< 16 bytes).
class DeltaSnapshot {
public:
    static constexpr std::uint8_t kFieldPosition = 1u << 0;
    static constexpr std::uint8_t kFieldRotation = 1u << 1;
    static constexpr std::uint8_t kFieldVelocity = 1u << 2;
    static constexpr std::uint8_t kFieldAll = kFieldPosition | kFieldRotation | kFieldVelocity;

    [[nodiscard]] static std::vector<std::byte> encodeDelta(
        std::span<const TransformDeltaState> baseline,
        std::span<const TransformDeltaState> current);

    [[nodiscard]] static std::optional<std::vector<TransformDeltaState>> decodeDelta(
        std::span<const TransformDeltaState> baseline,
        std::span<const std::byte> bytes);
};

} // namespace lumora::net
