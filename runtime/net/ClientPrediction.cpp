#include "runtime/net/ClientPrediction.hpp"

#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace lumora::net {

namespace {

constexpr float kDefaultAcceleration = 35.0f;
constexpr float kDefaultDrag = 6.0f;
constexpr float kDefaultGravity = 9.81f;
constexpr float kDefaultJumpSpeed = 6.0f;
constexpr float kSprintMultiplier = 1.5f;
constexpr float kCrouchMultiplier = 0.5f;

constexpr std::uint8_t kDeltaFormatVersion = 0x01u;
constexpr std::uint8_t kDeltaVersionMask = 0x0Fu;
constexpr std::uint8_t kFlagHasRemovals = 0x10u;
constexpr std::uint8_t kFlagHasChanges = 0x20u;
constexpr std::uint8_t kFlagHasOrder = 0x40u;
constexpr std::uint8_t kValidHeaderFlags = kDeltaVersionMask | kFlagHasRemovals | kFlagHasChanges | kFlagHasOrder;

constexpr std::size_t kInputCmdWireBytes = 32;
constexpr std::size_t kPredictedStateWireBytes = 33;

[[nodiscard]] bool isFiniteVec(const glm::vec3& v) noexcept {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

[[nodiscard]] bool vecEqual(const glm::vec3& a, const glm::vec3& b) noexcept {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

void writeVec3(ByteWriter& writer, const glm::vec3& v) {
    writer.f32(v.x);
    writer.f32(v.y);
    writer.f32(v.z);
}

glm::vec3 readVec3(ByteReader& reader) {
    glm::vec3 v;
    v.x = reader.f32();
    v.y = reader.f32();
    v.z = reader.f32();
    return v;
}

[[nodiscard]] std::uint8_t computeFieldMask(const TransformDeltaState& base, const TransformDeltaState& cur) noexcept {
    std::uint8_t mask = 0u;
    if (!vecEqual(base.position, cur.position)) {
        mask |= DeltaSnapshot::kFieldPosition;
    }
    if (!vecEqual(base.rotationDeg, cur.rotationDeg)) {
        mask |= DeltaSnapshot::kFieldRotation;
    }
    if (!vecEqual(base.velocity, cur.velocity)) {
        mask |= DeltaSnapshot::kFieldVelocity;
    }
    return mask;
}

struct DeltaEntry {
    TransformDeltaState state;
    std::uint8_t mask = 0u;
};

} // namespace

PredictedState defaultPredictStep(const PredictedState& state, const PlayerInputCmd& cmd) {
    PredictedState next = state;
    next.tick = cmd.tick;
    if (std::isfinite(cmd.yawDeg)) {
        next.yawDeg = cmd.yawDeg;
    }

    const float dt = (std::isfinite(cmd.dt) && cmd.dt > 0.0f) ? cmd.dt : 0.0f;
    if (dt == 0.0f) {
        return next;
    }

    glm::vec3 wishDir{0.0f};
    if (isFiniteVec(cmd.moveDir)) {
        wishDir.x = cmd.moveDir.x;
        wishDir.z = cmd.moveDir.z;
        const float horizLenSq = wishDir.x * wishDir.x + wishDir.z * wishDir.z;
        if (horizLenSq > 1.0f) {
            wishDir /= std::sqrt(horizLenSq);
        } else if (horizLenSq <= 1e-12f) {
            wishDir = glm::vec3{0.0f};
        }
    }

    float accel = kDefaultAcceleration;
    if ((cmd.buttons & InputButton::Sprint) != 0u) {
        accel *= kSprintMultiplier;
    } else if ((cmd.buttons & InputButton::Crouch) != 0u) {
        accel *= kCrouchMultiplier;
    }

    glm::vec3 horizVel{next.velocity.x, 0.0f, next.velocity.z};
    horizVel += wishDir * (accel * dt);
    horizVel /= (1.0f + kDefaultDrag * dt);

    float velY = next.velocity.y;
    if (next.grounded && (cmd.buttons & InputButton::Jump) != 0u) {
        velY = kDefaultJumpSpeed;
        next.grounded = false;
    }
    if (!next.grounded) {
        velY -= kDefaultGravity * dt;
    }

    next.velocity = glm::vec3{horizVel.x, velY, horizVel.z};
    next.position += next.velocity * dt;

    if (next.position.y <= 0.0f) {
        next.position.y = 0.0f;
        if (next.velocity.y < 0.0f) {
            next.velocity.y = 0.0f;
        }
        next.grounded = true;
    } else {
        next.grounded = false;
    }

    return next;
}

void encodeInputCmd(ByteWriter& writer, const PlayerInputCmd& cmd) {
    writer.u32(cmd.tick);
    writeVec3(writer, cmd.moveDir);
    writer.f32(cmd.yawDeg);
    writer.f32(cmd.pitchDeg);
    writer.u32(cmd.buttons);
    writer.f32(cmd.dt);
}

std::vector<std::byte> encodeInputCmd(const PlayerInputCmd& cmd) {
    std::vector<std::byte> out;
    out.reserve(kInputCmdWireBytes);
    ByteWriter writer{out};
    encodeInputCmd(writer, cmd);
    return out;
}

std::optional<PlayerInputCmd> decodeInputCmd(ByteReader& reader) {
    PlayerInputCmd cmd;
    cmd.tick = reader.u32();
    cmd.moveDir = readVec3(reader);
    cmd.yawDeg = reader.f32();
    cmd.pitchDeg = reader.f32();
    cmd.buttons = reader.u32();
    cmd.dt = reader.f32();
    if (!reader.ok) {
        return std::nullopt;
    }
    return cmd;
}

std::optional<PlayerInputCmd> decodeInputCmd(std::span<const std::byte> bytes) {
    ByteReader reader{bytes};
    auto cmd = decodeInputCmd(reader);
    if (!cmd || !reader.done()) {
        return std::nullopt;
    }
    return cmd;
}

void encodePredictedState(ByteWriter& writer, const PredictedState& state) {
    writer.u32(state.tick);
    writeVec3(writer, state.position);
    writeVec3(writer, state.velocity);
    writer.f32(state.yawDeg);
    writer.u8(state.grounded ? 1u : 0u);
}

std::vector<std::byte> encodePredictedState(const PredictedState& state) {
    std::vector<std::byte> out;
    out.reserve(kPredictedStateWireBytes);
    ByteWriter writer{out};
    encodePredictedState(writer, state);
    return out;
}

std::optional<PredictedState> decodePredictedState(ByteReader& reader) {
    PredictedState state;
    state.tick = reader.u32();
    state.position = readVec3(reader);
    state.velocity = readVec3(reader);
    state.yawDeg = reader.f32();
    state.grounded = reader.u8() != 0u;
    if (!reader.ok) {
        return std::nullopt;
    }
    return state;
}

std::optional<PredictedState> decodePredictedState(std::span<const std::byte> bytes) {
    ByteReader reader{bytes};
    auto state = decodePredictedState(reader);
    if (!state || !reader.done()) {
        return std::nullopt;
    }
    return state;
}

PredictionController::PredictionController()
    : stepFn_(defaultPredictStep) {}

PredictionController::PredictionController(InputStepFn stepFn, PredictedState initialState)
    : stepFn_(stepFn ? std::move(stepFn) : InputStepFn(defaultPredictStep)),
      currentState_(initialState),
      latestTick_(initialState.tick) {}

void PredictionController::setStepFunction(InputStepFn stepFn) {
    stepFn_ = stepFn ? std::move(stepFn) : InputStepFn(defaultPredictStep);
}

void PredictionController::reset(const PredictedState& initialState) {
    currentState_ = initialState;
    visualErrorOffset_ = glm::vec3{0.0f};
    reconciliationCount_ = 0;
    latestTick_ = initialState.tick;
    hasHistory_ = false;
    occupied_.fill(false);
}

bool PredictionController::isTickInWindow(std::uint32_t tick) const noexcept {
    if (!hasHistory_) {
        return false;
    }
    const std::uint32_t age = latestTick_ - tick;
    if (age >= kHistoryCapacity) {
        return false;
    }
    const std::size_t slot = slotFor(tick);
    return occupied_[slot] && stateHistory_[slot].tick == tick;
}

PredictedState PredictionController::recordAndPredict(const PlayerInputCmd& cmd) {
    currentState_ = stepFn_(currentState_, cmd);
    currentState_.tick = cmd.tick;

    const std::size_t slot = slotFor(cmd.tick);
    inputHistory_[slot] = cmd;
    stateHistory_[slot] = currentState_;
    occupied_[slot] = true;

    if (!hasHistory_ || (cmd.tick - latestTick_) < 0x80000000u) {
        latestTick_ = cmd.tick;
        hasHistory_ = true;
    }

    return currentState_;
}

bool PredictionController::reconcile(const PredictedState& authoritativeState, float posTolerance) {
    if (!isTickInWindow(authoritativeState.tick) || !isFiniteVec(authoritativeState.position)) {
        return false;
    }

    const std::size_t authSlot = slotFor(authoritativeState.tick);
    const glm::vec3 delta = stateHistory_[authSlot].position - authoritativeState.position;
    const float distSq = glm::dot(delta, delta);
    const float tol = (std::isfinite(posTolerance) && posTolerance > 0.0f) ? posTolerance : 0.0f;
    if (distSq <= tol * tol) {
        return false;
    }

    const glm::vec3 previousPosition = currentState_.position;
    stateHistory_[authSlot] = authoritativeState;

    PredictedState replayed = authoritativeState;
    const std::uint32_t replayCount = latestTick_ - authoritativeState.tick;
    for (std::uint32_t i = 1; i <= replayCount; ++i) {
        const std::uint32_t tick = authoritativeState.tick + i;
        const std::size_t slot = slotFor(tick);
        if (!occupied_[slot] || inputHistory_[slot].tick != tick) {
            continue;
        }
        replayed = stepFn_(replayed, inputHistory_[slot]);
        replayed.tick = tick;
        stateHistory_[slot] = replayed;
    }

    currentState_ = replayed;
    visualErrorOffset_ += (previousPosition - currentState_.position);
    ++reconciliationCount_;
    return true;
}

void PredictionController::decayVisualError(float dt, float decayRate) {
    if (!std::isfinite(dt) || dt <= 0.0f || !std::isfinite(decayRate) || decayRate <= 0.0f) {
        return;
    }
    visualErrorOffset_ *= std::exp(-decayRate * dt);
    if (glm::dot(visualErrorOffset_, visualErrorOffset_) < 1e-12f) {
        visualErrorOffset_ = glm::vec3{0.0f};
    }
}

std::optional<PredictedState> PredictionController::stateAtTick(std::uint32_t tick) const {
    if (!isTickInWindow(tick)) {
        return std::nullopt;
    }
    return stateHistory_[slotFor(tick)];
}

std::optional<PlayerInputCmd> PredictionController::inputAtTick(std::uint32_t tick) const {
    if (!isTickInWindow(tick)) {
        return std::nullopt;
    }
    const std::size_t slot = slotFor(tick);
    if (inputHistory_[slot].tick != tick) {
        return std::nullopt;
    }
    return inputHistory_[slot];
}

void PredictionController::encodeInputCmd(ByteWriter& writer, const PlayerInputCmd& cmd) {
    ::lumora::net::encodeInputCmd(writer, cmd);
}

std::vector<std::byte> PredictionController::encodeInputCmd(const PlayerInputCmd& cmd) {
    return ::lumora::net::encodeInputCmd(cmd);
}

std::optional<PlayerInputCmd> PredictionController::decodeInputCmd(ByteReader& reader) {
    return ::lumora::net::decodeInputCmd(reader);
}

std::optional<PlayerInputCmd> PredictionController::decodeInputCmd(std::span<const std::byte> bytes) {
    return ::lumora::net::decodeInputCmd(bytes);
}

void PredictionController::encodePredictedState(ByteWriter& writer, const PredictedState& state) {
    ::lumora::net::encodePredictedState(writer, state);
}

std::vector<std::byte> PredictionController::encodePredictedState(const PredictedState& state) {
    return ::lumora::net::encodePredictedState(state);
}

std::optional<PredictedState> PredictionController::decodePredictedState(ByteReader& reader) {
    return ::lumora::net::decodePredictedState(reader);
}

std::optional<PredictedState> PredictionController::decodePredictedState(std::span<const std::byte> bytes) {
    return ::lumora::net::decodePredictedState(bytes);
}

std::vector<std::byte> DeltaSnapshot::encodeDelta(
    std::span<const TransformDeltaState> baseline,
    std::span<const TransformDeltaState> current) {

    std::vector<DeltaEntry> changes;
    std::vector<std::uint32_t> removedIds;
    bool needsReorder = false;

    bool sameOrderIds = (baseline.size() == current.size());
    if (sameOrderIds) {
        for (std::size_t i = 0; i < current.size(); ++i) {
            if (baseline[i].entityNetId != current[i].entityNetId) {
                sameOrderIds = false;
                break;
            }
        }
    }

    if (sameOrderIds) {
        for (std::size_t i = 0; i < current.size(); ++i) {
            const std::uint8_t mask = computeFieldMask(baseline[i], current[i]);
            if (mask != 0u) {
                changes.push_back(DeltaEntry{current[i], mask});
            }
        }
    } else {
        std::unordered_map<std::uint32_t, std::size_t> baseIndexById;
        baseIndexById.reserve(baseline.size());
        for (std::size_t i = 0; i < baseline.size(); ++i) {
            baseIndexById.emplace(baseline[i].entityNetId, i);
        }

        std::vector<bool> seenInCurrent(baseline.size(), false);
        std::vector<std::uint32_t> addedIds;
        for (const TransformDeltaState& cur : current) {
            const auto it = baseIndexById.find(cur.entityNetId);
            if (it != baseIndexById.end()) {
                seenInCurrent[it->second] = true;
                const std::uint8_t mask = computeFieldMask(baseline[it->second], cur);
                if (mask != 0u) {
                    changes.push_back(DeltaEntry{cur, mask});
                }
            } else {
                changes.push_back(DeltaEntry{cur, kFieldAll});
                addedIds.push_back(cur.entityNetId);
            }
        }

        for (std::size_t i = 0; i < baseline.size(); ++i) {
            if (!seenInCurrent[i]) {
                removedIds.push_back(baseline[i].entityNetId);
            }
        }

        // Check whether surviving baseline entities followed by newly added entities already
        // match `current`'s order, avoiding an order table in the common append/remove cases.
        std::size_t curIdx = 0;
        for (std::size_t i = 0; i < baseline.size() && !needsReorder; ++i) {
            if (!seenInCurrent[i]) {
                continue;
            }
            if (curIdx >= current.size() || current[curIdx].entityNetId != baseline[i].entityNetId) {
                needsReorder = true;
            }
            ++curIdx;
        }
        for (std::size_t i = 0; i < addedIds.size() && !needsReorder; ++i) {
            if (curIdx >= current.size() || current[curIdx].entityNetId != addedIds[i]) {
                needsReorder = true;
            }
            ++curIdx;
        }
        if (curIdx != current.size()) {
            needsReorder = true;
        }
    }

    std::uint8_t flags = kDeltaFormatVersion;
    if (!removedIds.empty()) {
        flags |= kFlagHasRemovals;
    }
    if (!changes.empty()) {
        flags |= kFlagHasChanges;
    }
    if (needsReorder) {
        flags |= kFlagHasOrder;
    }

    std::vector<std::byte> out;
    out.reserve(5 + (changes.empty() ? 0 : 4 + changes.size() * 17));
    ByteWriter writer{out};
    writer.u8(flags);
    writer.u32(static_cast<std::uint32_t>(current.size()));

    if (!removedIds.empty()) {
        writer.u32(static_cast<std::uint32_t>(removedIds.size()));
        for (const std::uint32_t id : removedIds) {
            writer.u32(id);
        }
    }

    if (!changes.empty()) {
        writer.u32(static_cast<std::uint32_t>(changes.size()));
        for (const DeltaEntry& entry : changes) {
            writer.u32(entry.state.entityNetId);
            writer.u8(entry.mask);
            if ((entry.mask & kFieldPosition) != 0u) {
                writeVec3(writer, entry.state.position);
            }
            if ((entry.mask & kFieldRotation) != 0u) {
                writeVec3(writer, entry.state.rotationDeg);
            }
            if ((entry.mask & kFieldVelocity) != 0u) {
                writeVec3(writer, entry.state.velocity);
            }
        }
    }

    if (needsReorder) {
        for (const TransformDeltaState& cur : current) {
            writer.u32(cur.entityNetId);
        }
    }

    return out;
}

std::optional<std::vector<TransformDeltaState>> DeltaSnapshot::decodeDelta(
    std::span<const TransformDeltaState> baseline,
    std::span<const std::byte> bytes) {

    ByteReader reader{bytes};
    const std::uint8_t flags = reader.u8();
    if (!reader.ok || (flags & kDeltaVersionMask) != kDeltaFormatVersion || (flags & ~kValidHeaderFlags) != 0u) {
        return std::nullopt;
    }

    const std::uint32_t expectedCount = reader.u32();
    if (!reader.ok) {
        return std::nullopt;
    }

    std::vector<TransformDeltaState> result(baseline.begin(), baseline.end());

    if ((flags & kFlagHasRemovals) != 0u) {
        const std::uint32_t removedCount = reader.u32();
        if (!reader.ok || removedCount > baseline.size() ||
            (reader.data.size() - reader.at) / 4 < removedCount) {
            return std::nullopt;
        }
        std::unordered_set<std::uint32_t> removedIds;
        removedIds.reserve(removedCount);
        for (std::uint32_t i = 0; i < removedCount; ++i) {
            removedIds.insert(reader.u32());
        }
        if (!reader.ok) {
            return std::nullopt;
        }
        std::erase_if(result, [&](const TransformDeltaState& s) {
            return removedIds.contains(s.entityNetId);
        });
    }

    if ((flags & kFlagHasChanges) != 0u) {
        const std::uint32_t changedCount = reader.u32();
        if (!reader.ok || (reader.data.size() - reader.at) / 5 < changedCount) {
            return std::nullopt;
        }

        std::unordered_map<std::uint32_t, std::size_t> indexById;
        indexById.reserve(result.size() + changedCount);
        for (std::size_t i = 0; i < result.size(); ++i) {
            indexById.emplace(result[i].entityNetId, i);
        }

        for (std::uint32_t i = 0; i < changedCount; ++i) {
            const std::uint32_t id = reader.u32();
            const std::uint8_t mask = reader.u8();
            if (!reader.ok || mask == 0u || (mask & ~kFieldAll) != 0u) {
                return std::nullopt;
            }

            std::size_t targetIdx = 0;
            const auto it = indexById.find(id);
            if (it != indexById.end()) {
                targetIdx = it->second;
            } else {
                targetIdx = result.size();
                indexById.emplace(id, targetIdx);
                result.push_back(TransformDeltaState{.entityNetId = id});
            }

            TransformDeltaState& target = result[targetIdx];
            if ((mask & kFieldPosition) != 0u) {
                target.position = readVec3(reader);
            }
            if ((mask & kFieldRotation) != 0u) {
                target.rotationDeg = readVec3(reader);
            }
            if ((mask & kFieldVelocity) != 0u) {
                target.velocity = readVec3(reader);
            }
            if (!reader.ok) {
                return std::nullopt;
            }
        }
    }

    if ((flags & kFlagHasOrder) != 0u) {
        if ((reader.data.size() - reader.at) / 4 < expectedCount || result.size() != expectedCount) {
            return std::nullopt;
        }

        std::unordered_map<std::uint32_t, std::size_t> indexById;
        indexById.reserve(result.size());
        for (std::size_t i = 0; i < result.size(); ++i) {
            indexById.emplace(result[i].entityNetId, i);
        }

        std::vector<TransformDeltaState> reordered;
        reordered.reserve(expectedCount);
        std::vector<bool> used(result.size(), false);
        for (std::uint32_t i = 0; i < expectedCount; ++i) {
            const std::uint32_t id = reader.u32();
            const auto it = indexById.find(id);
            if (!reader.ok || it == indexById.end() || used[it->second]) {
                return std::nullopt;
            }
            used[it->second] = true;
            reordered.push_back(result[it->second]);
        }
        result = std::move(reordered);
    }

    if (!reader.ok || !reader.done() || result.size() != expectedCount) {
        return std::nullopt;
    }
    return result;
}

} // namespace lumora::net
