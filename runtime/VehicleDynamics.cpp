#include "runtime/VehicleDynamics.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace lumora {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kDegToRad = kPi / 180.0f;
constexpr float kRpmToRadS = (2.0f * kPi) / 60.0f;
constexpr float kRadSToRpm = 60.0f / (2.0f * kPi);
constexpr float kMsToKmh = 3.6f;
constexpr float kMaxSubStepSec = 1.0f / 240.0f;
constexpr int kMaxSubSteps = 16;

[[nodiscard]] float sanitizeFinite(float value, float fallback = 0.0f) noexcept {
    return std::isfinite(value) ? value : fallback;
}

[[nodiscard]] float sanitizePositive(float value, float fallback) noexcept {
    return (std::isfinite(value) && value > 0.0f) ? value : fallback;
}

void stepAxleWheels(const PowertrainParams& pt, const PacejkaParams& tire,
                    const DifferentialParams& diff, float axleTorqueNm, float brakeTorquePerWheel,
                    float handbrakeTorquePerWheel, float steerRadLeft, float steerRadRight,
                    float forwardSpeedMs, float lateralSpeedMs, const std::array<float, 4>& loads,
                    const std::array<bool, 4>& contacts, const std::array<float, 4>& grips,
                    std::size_t idxLeft, std::size_t idxRight, bool driven, float dt,
                    std::array<float, 4>& omegas, std::array<float, 4>& driveTorquesOut,
                    std::array<float, 4>& lonForceAccum, std::array<float, 4>& latForceAccum,
                    std::array<float, 4>& slipRatioOut, std::array<float, 4>& slipAngleOut) noexcept {
    const float wheelRadius = sanitizePositive(pt.wheelRadius, 0.35f);
    const float wheelInertia = sanitizePositive(pt.wheelInertia, 1.2f);
    const float lonPeakSlope =
        std::max(0.0f, tire.longitudinal.b * tire.longitudinal.c * tire.longitudinal.d);

    DifferentialOutput split{0.0f, 0.0f};
    if (driven) {
        split = splitDifferentialTorque(axleTorqueNm, omegas[idxLeft], omegas[idxRight], diff);
    }

    const std::array<std::size_t, 2> indices{idxLeft, idxRight};
    const std::array<float, 2> steerRads{steerRadLeft, steerRadRight};
    const std::array<float, 2> driveTorques{split.leftTorqueNm, split.rightTorqueNm};

    for (std::size_t slot = 0; slot < 2; ++slot) {
        const std::size_t w = indices[slot];
        const float steerRad = steerRads[slot];
        const float driveTorque = driveTorques[slot];
        driveTorquesOut[w] = driveTorque;

        const float cosS = std::cos(steerRad);
        const float sinS = std::sin(steerRad);
        const float vWheelLon = forwardSpeedMs * cosS + lateralSpeedMs * sinS;
        const float rollingOmega = vWheelLon / wheelRadius;

        const float normalLoad = contacts[w] ? std::max(0.0f, loads[w]) : 0.0f;
        const float grip = std::max(0.0f, grips[w]);

        const float slipRatio = computeSlipRatio(omegas[w], wheelRadius, vWheelLon);
        const float slipAngle = computeSlipAngle(lateralSpeedMs, forwardSpeedMs, steerRad);
        const TireForces forces = computeTireForces(normalLoad, slipRatio, slipAngle, tire, grip);

        slipRatioOut[w] = slipRatio;
        slipAngleOut[w] = slipAngle;
        lonForceAccum[w] += forces.longitudinal;
        latForceAccum[w] += forces.lateral;

        // Implicit linearization around rolling speed keeps wheel angular velocity unconditionally
        // stable even at low vehicle speed or stiff tire slopes, while using secant stiffness in
        // post-peak spin so wheels genuinely break traction when drive torque exceeds peak grip.
        float tireReactionGain = 0.0f;
        if (normalLoad > 0.0f && grip > 0.0f) {
            const float deltaOmega = omegas[w] - rollingOmega;
            const float tireTorque = forces.longitudinal * wheelRadius;
            if (std::abs(deltaOmega) > 0.25f) {
                tireReactionGain = std::abs(tireTorque) / std::abs(deltaOmega);
            } else {
                const float denomSpeed = std::max(std::abs(vWheelLon), 0.5f);
                tireReactionGain =
                    (wheelRadius * wheelRadius * normalLoad * grip * lonPeakSlope) / denomSpeed;
            }
        }

        const float bearingDamping = 0.05f;
        const float denom = wheelInertia + dt * (tireReactionGain + bearingDamping);
        const float omegaNoBrake =
            (wheelInertia * omegas[w] + dt * (driveTorque + tireReactionGain * rollingOmega)) /
            std::max(denom, 1e-6f);

        const float totalBrakeTorque =
            std::max(0.0f, brakeTorquePerWheel) + std::max(0.0f, handbrakeTorquePerWheel);
        const float maxBrakeDelta = (totalBrakeTorque / wheelInertia) * dt;
        if (std::abs(omegaNoBrake) <= maxBrakeDelta) {
            omegas[w] = 0.0f;
        } else {
            omegas[w] = omegaNoBrake - std::copysign(maxBrakeDelta, omegaNoBrake);
        }
    }

    if (driven && diff.type == DiffType::Locked) {
        const float lockedOmega = 0.5f * (omegas[idxLeft] + omegas[idxRight]);
        omegas[idxLeft] = lockedOmega;
        omegas[idxRight] = lockedOmega;
    }
}

void stepSuspensionWheel(const SuspensionParams& params, float loadN, bool contact, float dt,
                         float& compression, float& compressionVel) noexcept {
    const float mass = sanitizePositive(params.cornerMassKg, 350.0f);
    const float targetLoad = contact ? std::max(0.0f, loadN) : 0.0f;

    // Integrate 1D spring-damper toward target normal load using semi-implicit Euler:
    //   m * x'' = loadN + F_hooke(x, x')
    // where F_hooke = -k * x - c(x') * x' (- bumpStop), so at equilibrium k * x == loadN.
    const float hookeForce = computeSuspensionForce(compression, compressionVel, params);
    const float accel = (targetLoad + hookeForce) / mass;
    compressionVel += accel * dt;
    compression += compressionVel * dt;

    if (compression < 0.0f) {
        compression = 0.0f;
        if (compressionVel < 0.0f) {
            compressionVel = 0.0f;
        }
    } else if (params.restLength > 0.0f && compression > params.restLength) {
        compression = params.restLength;
        if (compressionVel > 0.0f) {
            compressionVel = 0.0f;
        }
    }
}

} // namespace

float evaluatePacejka(float slip, float b, float c, float d, float e) noexcept {
    if (!std::isfinite(slip) || !std::isfinite(b) || !std::isfinite(c) || !std::isfinite(d) ||
        !std::isfinite(e)) {
        return 0.0f;
    }

    const float u = b * slip;
    const float atanU = std::atan(u);
    const float phi = u - e * (u - atanU);
    return d * std::sin(c * std::atan(phi));
}

float evaluatePacejka(float slip, const PacejkaCurve& curve) noexcept {
    return evaluatePacejka(slip, curve.b, curve.c, curve.d, curve.e);
}

float estimatePacejkaPeakSlip(const PacejkaCurve& curve) noexcept {
    const float b = sanitizeFinite(curve.b);
    const float c = sanitizeFinite(curve.c);
    const float e = sanitizeFinite(curve.e);
    if (b <= 1e-4f) {
        return 0.12f;
    }
    if (c <= 1.0f) {
        return 2.0f / b;
    }
    const float targetPhi = std::tan((kPi * 0.5f) / std::max(c, 1.001f));
    if (e > 1.0f) {
        return std::clamp(targetPhi / b, 0.03f, 0.5f);
    }

    float u = targetPhi;
    for (int iter = 0; iter < 8; ++iter) {
        const float atanU = std::atan(u);
        const float f = (u - e * (u - atanU)) - targetPhi;
        const float df = 1.0f - e + e / (1.0f + u * u);
        if (std::abs(df) < 1e-6f) {
            break;
        }
        u = std::max(0.0f, u - f / df);
    }
    return std::max(u / b, 1e-3f);
}

float computeSlipRatio(float wheelAngularVelRadS, float wheelRadius, float longitudinalSpeed,
                       float minSpeedEps) noexcept {
    if (!std::isfinite(wheelAngularVelRadS) || !std::isfinite(wheelRadius) ||
        !std::isfinite(longitudinalSpeed) || wheelRadius <= 0.0f) {
        return 0.0f;
    }
    const float minSpeed = std::max(sanitizeFinite(minSpeedEps, 0.5f), 1e-3f);
    const float wheelSpeed = wheelAngularVelRadS * wheelRadius;
    const float denom = std::max(std::abs(longitudinalSpeed), minSpeed);
    return std::clamp((wheelSpeed - longitudinalSpeed) / denom, -2.0f, 2.0f);
}

float computeSlipAngle(float localVelX, float localVelZ, float steerRad,
                       float minSpeedEps) noexcept {
    if (!std::isfinite(localVelX) || !std::isfinite(localVelZ) || !std::isfinite(steerRad)) {
        return 0.0f;
    }
    const float minSpeed = std::max(sanitizeFinite(minSpeedEps, 0.5f), 1e-3f);

    const float cosS = std::cos(steerRad);
    const float sinS = std::sin(steerRad);
    const float vWheelLon = localVelZ * cosS + localVelX * sinS;
    const float vWheelLat = localVelX * cosS - localVelZ * sinS;
    const float denom = std::max(std::abs(vWheelLon), minSpeed);
    return std::atan2(-vWheelLat, denom);
}

TireForces computeTireForces(float normalLoadN, float slipRatio, float slipAngleRad,
                             const PacejkaParams& params, float surfaceGrip) noexcept {
    if (!std::isfinite(normalLoadN) || !std::isfinite(slipRatio) ||
        !std::isfinite(slipAngleRad) || !std::isfinite(surfaceGrip)) {
        return {};
    }
    const float load = normalLoadN;
    const float grip = surfaceGrip;
    const float kappa = slipRatio;
    const float alpha = slipAngleRad;

    if (load <= 0.0f || grip <= 0.0f) {
        return {};
    }

    float loadScale = 1.0f;
    if (params.loadSensitivity > 0.0f && params.referenceLoadN > 0.0f) {
        const float relLoad = (load - params.referenceLoadN) / params.referenceLoadN;
        loadScale = std::clamp(1.0f - params.loadSensitivity * relLoad, 0.25f, 1.5f);
    }

    const float effectiveLoad = load * grip * loadScale;
    const float fxMax = std::max(0.0f, effectiveLoad * std::max(0.0f, sanitizeFinite(params.longitudinal.d)));
    const float fyMax = std::max(0.0f, effectiveLoad * std::max(0.0f, sanitizeFinite(params.lateral.d)));
    if (fxMax <= 0.0f && fyMax <= 0.0f) {
        return {};
    }

    const float kappaPeak = estimatePacejkaPeakSlip(params.longitudinal);
    const float alphaPeak = estimatePacejkaPeakSlip(params.lateral);
    const float sx = kappa / kappaPeak;
    const float sy = alpha / alphaPeak;
    const float combinedSlip = std::hypot(sx, sy);

    if (combinedSlip <= 1e-7f) {
        return {};
    }

    float fx = 0.0f;
    float fy = 0.0f;

    if (std::abs(alpha) <= 1e-7f) {
        fx = effectiveLoad * evaluatePacejka(kappa, params.longitudinal);
    } else if (std::abs(kappa) <= 1e-7f) {
        fy = effectiveLoad * evaluatePacejka(alpha, params.lateral);
    } else {
        const float wx = sx / combinedSlip;
        const float wy = sy / combinedSlip;
        fx = wx * effectiveLoad * evaluatePacejka(combinedSlip * kappaPeak, params.longitudinal);
        fy = wy * effectiveLoad * evaluatePacejka(combinedSlip * alphaPeak, params.lateral);
    }

    if (fxMax <= 0.0f) {
        fx = 0.0f;
    }
    if (fyMax <= 0.0f) {
        fy = 0.0f;
    }

    const float budget = std::max(0.0f, sanitizeFinite(params.maxCombinedGripRatio, 1.0f));
    const float nx = fxMax > 1e-6f ? (fx / fxMax) : 0.0f;
    const float ny = fyMax > 1e-6f ? (fy / fyMax) : 0.0f;
    const float norm = std::hypot(nx, ny);
    if (norm > budget && norm > 1e-6f) {
        const float scale = budget / norm;
        fx *= scale;
        fy *= scale;
    }

    return TireForces{fx, fy};
}

float computeSuspensionForce(float compression, float compressionVel,
                             const SuspensionParams& params) noexcept {
    if (!std::isfinite(compression) || !std::isfinite(compressionVel)) {
        return 0.0f;
    }
    const float x = compression;
    const float v = compressionVel;
    const float k = std::max(0.0f, sanitizeFinite(params.stiffness));
    const float c = v >= 0.0f ? std::max(0.0f, sanitizeFinite(params.bumpDamping))
                              : std::max(0.0f, sanitizeFinite(params.reboundDamping));

    float force = -k * x - c * v;
    const float maxTravel = std::max(0.0f, sanitizeFinite(params.maxTravel));
    const float bumpStop = std::max(0.0f, sanitizeFinite(params.bumpStopStiffness));
    if (x > maxTravel && bumpStop > 0.0f) {
        force -= bumpStop * (x - maxTravel);
    }
    return force;
}

float computeSuspensionSupportLoad(float compression, float compressionVel,
                                   const SuspensionParams& params) noexcept {
    if (!std::isfinite(compression) || !std::isfinite(compressionVel)) {
        return 0.0f;
    }
    const float clampedCompression = std::max(0.0f, compression);
    if (clampedCompression <= 0.0f && compressionVel <= 0.0f) {
        return 0.0f;
    }
    const float restoring = computeSuspensionForce(clampedCompression, compressionVel, params);
    return std::max(0.0f, -restoring);
}

float computeAntiRollForce(float leftCompression, float rightCompression,
                           float antiRollStiffness) noexcept {
    if (!std::isfinite(leftCompression) || !std::isfinite(rightCompression) ||
        !std::isfinite(antiRollStiffness)) {
        return 0.0f;
    }
    return (leftCompression - rightCompression) * std::max(0.0f, antiRollStiffness);
}

float PowertrainParams::sampleEngineTorque(float rpm) const noexcept {
    const float clampedRpm = sanitizeFinite(rpm);
    if (clampedRpm <= 0.0f || torqueCurve.empty()) {
        return 0.0f;
    }
    if (torqueCurve.size() == 1) {
        return std::max(0.0f, sanitizeFinite(torqueCurve.front().torqueNm));
    }
    if (clampedRpm <= torqueCurve.front().rpm) {
        return std::max(0.0f, sanitizeFinite(torqueCurve.front().torqueNm));
    }
    if (clampedRpm >= torqueCurve.back().rpm) {
        return std::max(0.0f, sanitizeFinite(torqueCurve.back().torqueNm));
    }

    for (std::size_t i = 1; i < torqueCurve.size(); ++i) {
        const TorqueCurvePoint& p0 = torqueCurve[i - 1];
        const TorqueCurvePoint& p1 = torqueCurve[i];
        if (clampedRpm <= p1.rpm) {
            const float span = p1.rpm - p0.rpm;
            if (span <= 1e-5f) {
                return std::max(0.0f, sanitizeFinite(p1.torqueNm));
            }
            const float t = std::clamp((clampedRpm - p0.rpm) / span, 0.0f, 1.0f);
            return std::max(0.0f, p0.torqueNm + t * (p1.torqueNm - p0.torqueNm));
        }
    }
    return std::max(0.0f, sanitizeFinite(torqueCurve.back().torqueNm));
}

float PowertrainParams::gearRatio(int gear) const noexcept {
    const int idx = gear + 1;
    if (idx < 0 || idx >= static_cast<int>(gearRatios.size())) {
        return 0.0f;
    }
    return sanitizeFinite(gearRatios[static_cast<std::size_t>(idx)]);
}

DifferentialOutput splitDifferentialTorque(float inputTorqueNm, float leftWheelVelRadS,
                                           float rightWheelVelRadS,
                                           const DifferentialParams& params) noexcept {
    const float totalTorque = sanitizeFinite(inputTorqueNm);
    const float halfTorque = 0.5f * totalTorque;
    if (params.type == DiffType::Open) {
        return DifferentialOutput{halfTorque, halfTorque};
    }

    const float omegaL = sanitizeFinite(leftWheelVelRadS);
    const float omegaR = sanitizeFinite(rightWheelVelRadS);
    const float deltaOmega = omegaL - omegaR;

    if (params.type == DiffType::LimitedSlip) {
        const float eps = std::max(sanitizeFinite(params.smoothingEpsRadS, 0.25f), 1e-4f);
        const float dir = deltaOmega / (std::abs(deltaOmega) + eps);
        const float locking = std::clamp(sanitizeFinite(params.lsdLockingFactor, 0.65f), 0.0f, 1.0f);
        const float preload = std::max(0.0f, sanitizeFinite(params.lsdPreloadNm));
        const float viscous = std::max(0.0f, sanitizeFinite(params.viscousDamping));

        const float torqueBias = 0.5f * std::abs(totalTorque) * locking;
        const float rawTransfer = (torqueBias + preload) * dir + viscous * deltaOmega;
        const float maxTransfer = std::max(0.5f * std::abs(totalTorque), preload);
        const float transfer = std::clamp(rawTransfer, -maxTransfer, maxTransfer);
        return DifferentialOutput{halfTorque - transfer, halfTorque + transfer};
    }

    // DiffType::Locked: strong coupling resisting any half-shaft speed mismatch.
    const float lockGain = std::max(0.0f, sanitizeFinite(params.lockGain, 250.0f));
    const float eps = std::max(sanitizeFinite(params.smoothingEpsRadS, 0.25f) * 0.2f, 1e-3f);
    const float dir = deltaOmega / (std::abs(deltaOmega) + eps);
    const float rawTransfer = 0.5f * std::abs(totalTorque) * dir + lockGain * deltaOmega;
    const float maxTransfer = std::max(0.5f * std::abs(totalTorque), lockGain * 10.0f);
    const float transfer = std::clamp(rawTransfer, -maxTransfer, maxTransfer);
    return DifferentialOutput{halfTorque - transfer, halfTorque + transfer};
}

VehicleDrivetrainSim::VehicleDrivetrainSim() { reset(); }

VehicleDrivetrainSim::VehicleDrivetrainSim(VehicleDynamicsConfig config)
    : config_(std::move(config)) {
    reset();
}

void VehicleDrivetrainSim::reset() noexcept {
    wheelAngularVel_.fill(0.0f);
    suspensionCompression_.fill(0.0f);
    suspensionCompressionVel_.fill(0.0f);
    wheelGrips_.fill(1.0f);
    engineRpm_ = std::max(100.0f, sanitizeFinite(config_.powertrain.idleRpm, 850.0f));
    currentGear_ = 1;
    shiftTimer_ = 0.0f;
    shiftCooldown_ = 0.0f;
    downshiftHoldTimer_ = 0.0f;
    clutchEngagement_ = 0.0f;
    lastTelemetry_ = VehicleTelemetry{};
    lastTelemetry_.engineRpm = engineRpm_;
    lastTelemetry_.currentGear = currentGear_;
}

void VehicleDrivetrainSim::setConfig(VehicleDynamicsConfig config) {
    config_ = std::move(config);
    engineRpm_ = std::clamp(engineRpm_, config_.powertrain.idleRpm,
                            std::max(config_.powertrain.revLimiterRpm, config_.powertrain.idleRpm));
}

void VehicleDrivetrainSim::setDiffType(DiffType type) noexcept {
    config_.frontDifferential.type = type;
    config_.rearDifferential.type = type;
}

bool VehicleDrivetrainSim::setGear(int gear) noexcept {
    if (gear < -1 || gear > 6) {
        return false;
    }
    if (gear != currentGear_) {
        beginShift(gear);
    }
    return true;
}

bool VehicleDrivetrainSim::shiftUp() noexcept {
    if (currentGear_ >= 6) {
        return false;
    }
    beginShift(currentGear_ + 1);
    return true;
}

bool VehicleDrivetrainSim::shiftDown() noexcept {
    if (currentGear_ <= -1) {
        return false;
    }
    beginShift(currentGear_ - 1);
    return true;
}

void VehicleDrivetrainSim::setClutchOverride(float engagement) noexcept {
    clutchOverride_ = std::clamp(sanitizeFinite(engagement), 0.0f, 1.0f);
}

void VehicleDrivetrainSim::setWheelGrip(std::size_t wheelIndex, float grip) noexcept {
    if (wheelIndex < kVehicleWheelCount) {
        wheelGrips_[wheelIndex] = std::max(0.0f, sanitizeFinite(grip, 1.0f));
    }
}

void VehicleDrivetrainSim::setWheelGrips(const std::array<float, kVehicleWheelCount>& grips) noexcept {
    for (std::size_t i = 0; i < kVehicleWheelCount; ++i) {
        wheelGrips_[i] = std::max(0.0f, sanitizeFinite(grips[i], 1.0f));
    }
}

void VehicleDrivetrainSim::beginShift(int targetGear) noexcept {
    const int clampedTarget = std::clamp(targetGear, -1, 6);
    if (clampedTarget > currentGear_) {
        downshiftHoldTimer_ = 0.60f;
    }
    currentGear_ = clampedTarget;
    shiftTimer_ = std::max(0.0f, sanitizeFinite(config_.powertrain.shiftDurationSec));
    shiftCooldown_ = std::max(0.0f, sanitizeFinite(config_.powertrain.shiftCooldownSec));
}

void VehicleDrivetrainSim::updateAutoShift(float drivetrainRpm) noexcept {
    const PowertrainParams& pt = config_.powertrain;
    if (!pt.autoShift || shiftTimer_ > 0.0f || shiftCooldown_ > 0.0f) {
        return;
    }
    if (currentGear_ < 1) {
        return;
    }

    const float upshift = std::max(pt.upshiftRpm, pt.idleRpm + 500.0f);
    const float downshift = std::clamp(pt.downshiftRpm, pt.idleRpm, upshift - 400.0f);

    if (currentGear_ < 6 && engineRpm_ >= upshift && drivetrainRpm >= upshift &&
        clutchEngagement_ >= 0.90f) {
        beginShift(currentGear_ + 1);
    } else if (currentGear_ > 1 && downshiftHoldTimer_ <= 0.0f &&
               drivetrainRpm <= downshift && engineRpm_ <= downshift + 200.0f) {
        beginShift(currentGear_ - 1);
    }
}

VehicleTelemetry VehicleDrivetrainSim::step(
    float throttle, float brake, float handbrake, float steerDeg,
    const std::array<float, kVehicleWheelCount>& wheelLoadsN,
    const std::array<bool, kVehicleWheelCount>& wheelContacts, float forwardSpeedMs,
    float lateralSpeedMs, float dt) noexcept {
    const float cleanDt = sanitizeFinite(dt);
    if (cleanDt <= 0.0f) {
        return lastTelemetry_;
    }

    const float cleanThrottle = std::clamp(sanitizeFinite(throttle), 0.0f, 1.0f);
    const float cleanBrake = std::clamp(sanitizeFinite(brake), 0.0f, 1.0f);
    const float cleanHandbrake = std::clamp(sanitizeFinite(handbrake), 0.0f, 1.0f);
    const float maxSteer = std::clamp(sanitizeFinite(config_.powertrain.maxSteerDeg, 45.0f), 0.0f, 85.0f);
    const float steerRad = std::clamp(sanitizeFinite(steerDeg), -maxSteer, maxSteer) * kDegToRad;
    const float vForward = sanitizeFinite(forwardSpeedMs);
    const float vLateral = sanitizeFinite(lateralSpeedMs);

    std::array<float, kVehicleWheelCount> cleanLoads{};
    for (std::size_t i = 0; i < kVehicleWheelCount; ++i) {
        cleanLoads[i] = std::max(0.0f, sanitizeFinite(wheelLoadsN[i]));
    }

    const PowertrainParams& pt = config_.powertrain;
    const float idleRpm = std::max(100.0f, sanitizeFinite(pt.idleRpm, 850.0f));
    const float redlineRpm = std::max(idleRpm + 500.0f, sanitizeFinite(pt.redlineRpm, 7000.0f));
    const float revLimiterRpm = std::max(redlineRpm + 50.0f, sanitizeFinite(pt.revLimiterRpm, 7200.0f));
    const float engineInertia = sanitizePositive(pt.engineInertia, 0.25f);
    const float finalDrive = sanitizePositive(pt.finalDriveRatio, 3.73f);
    const float efficiency = std::clamp(sanitizeFinite(pt.drivetrainEfficiency, 0.9f), 0.0f, 1.0f);

    if (pt.autoShift && currentGear_ == 0 && cleanThrottle > 0.05f) {
        currentGear_ = 1;
    }

    const int substeps = std::clamp(static_cast<int>(std::ceil(cleanDt / kMaxSubStepSec)), 1, kMaxSubSteps);
    const float h = cleanDt / static_cast<float>(substeps);

    std::array<float, kVehicleWheelCount> lonForceAccum{};
    std::array<float, kVehicleWheelCount> latForceAccum{};
    std::array<float, kVehicleWheelCount> slipRatioLast{};
    std::array<float, kVehicleWheelCount> slipAngleLast{};
    std::array<float, kVehicleWheelCount> driveTorqueLast{};
    float lastEngineTorque = 0.0f;

    const bool frontDriven =
        config_.driveLayout == DriveLayout::Fwd || config_.driveLayout == DriveLayout::Awd;
    const bool rearDriven =
        config_.driveLayout == DriveLayout::Rwd || config_.driveLayout == DriveLayout::Awd;
    const float frontSplit = config_.driveLayout == DriveLayout::Awd
                                 ? std::clamp(sanitizeFinite(config_.frontTorqueBias, 0.5f), 0.0f, 1.0f)
                                 : (config_.driveLayout == DriveLayout::Fwd ? 1.0f : 0.0f);
    const float rearSplit = 1.0f - frontSplit;

    for (int sub = 0; sub < substeps; ++sub) {
        if (shiftTimer_ > 0.0f) {
            shiftTimer_ = std::max(0.0f, shiftTimer_ - h);
        }
        if (shiftCooldown_ > 0.0f) {
            shiftCooldown_ = std::max(0.0f, shiftCooldown_ - h);
        }
        if (downshiftHoldTimer_ > 0.0f) {
            downshiftHoldTimer_ = std::max(0.0f, downshiftHoldTimer_ - h);
        }

        const float gearRatio = pt.gearRatio(currentGear_);
        const float totalRatio = gearRatio * finalDrive;

        // Average driven wheel speed projected back to engine shaft.
        float drivenOmega = 0.0f;
        if (frontDriven && rearDriven) {
            const float frontOmega = 0.5f * (wheelAngularVel_[0] + wheelAngularVel_[1]);
            const float rearOmega = 0.5f * (wheelAngularVel_[2] + wheelAngularVel_[3]);
            drivenOmega = frontSplit * frontOmega + rearSplit * rearOmega;
        } else if (frontDriven) {
            drivenOmega = 0.5f * (wheelAngularVel_[0] + wheelAngularVel_[1]);
        } else {
            drivenOmega = 0.5f * (wheelAngularVel_[2] + wheelAngularVel_[3]);
        }
        const float drivetrainOmega = std::max(0.0f, drivenOmega * totalRatio);
        const float drivetrainRpm = drivetrainOmega * kRadSToRpm;

        // Engine torque with rev limiter and closed-throttle engine braking.
        float rawTorque = pt.sampleEngineTorque(engineRpm_);
        if (engineRpm_ > redlineRpm) {
            const float limiterCut =
                1.0f - std::clamp((engineRpm_ - redlineRpm) / (revLimiterRpm - redlineRpm), 0.0f, 1.0f);
            rawTorque *= limiterCut;
        }
        const float combustionTorque = cleanThrottle * rawTorque;
        const float rpmNorm = std::clamp((engineRpm_ - idleRpm) / (redlineRpm - idleRpm), 0.0f, 1.0f);
        const float engineBrake =
            (1.0f - cleanThrottle) * std::max(0.0f, sanitizeFinite(pt.engineBrakingTorque)) * rpmNorm;
        const float idleAssist =
            engineRpm_ < idleRpm ? (idleRpm - engineRpm_) * 0.25f : 0.0f;
        const float netEngineTorque = combustionTorque - engineBrake + idleAssist;
        lastEngineTorque = combustionTorque - engineBrake;

        float engineOmega = engineRpm_ * kRpmToRadS;
        const float idleOmega = idleRpm * kRpmToRadS;
        float transmittedClutchTorque = 0.0f;

        // Clutch engagement & engine-drivetrain coupling:
        // - Neutral / shift / stopped coasting: clutch disengaged.
        // - Launch slip (drivetrain speed below idle/launch band): clutch slips smoothly toward
        //   target launch RPM without dragging engine below idle.
        // - Engaged driving (drivetrain speed >= launch band): clutch locks engine to drivetrain.
        const bool inDriveGear = currentGear_ != 0 && std::abs(totalRatio) >= 1e-4f;
        if (clutchOverride_ >= 0.0f) {
            clutchEngagement_ = clutchOverride_;
            if (inDriveGear && clutchEngagement_ > 0.0f) {
                const float clutchGain = clutchEngagement_ * 35.0f;
                const float targetOmega = std::max(drivetrainOmega, idleOmega * 0.9f);
                const float newEngineOmega =
                    (engineInertia * engineOmega + h * (netEngineTorque + clutchGain * targetOmega)) /
                    std::max(engineInertia + h * clutchGain, 1e-6f);
                transmittedClutchTorque =
                    std::clamp(clutchGain * (newEngineOmega - drivetrainOmega), -500.0f, 1200.0f);
                engineOmega = newEngineOmega;
            } else {
                engineOmega += (netEngineTorque / engineInertia) * h;
            }
        } else if (!inDriveGear || shiftTimer_ > 0.0f) {
            clutchEngagement_ = 0.0f;
            if (shiftTimer_ > 0.0f && inDriveGear) {
                // Smoothly rev-match engine RPM toward the newly selected gear during shift dip.
                const float targetOmega = std::max(idleOmega, drivetrainOmega);
                engineOmega += (targetOmega - engineOmega) * std::min(1.0f, h * 15.0f);
            } else {
                engineOmega += (netEngineTorque / engineInertia) * h;
            }
        } else if (cleanThrottle < 0.02f && std::abs(vForward) < 0.35f && drivetrainRpm < idleRpm) {
            clutchEngagement_ = 0.0f;
            engineOmega += (netEngineTorque / engineInertia) * h;
        } else {
            const float launchLockRpm = idleRpm + 700.0f;
            if (drivetrainRpm >= launchLockRpm) {
                clutchEngagement_ = 1.0f;
                const float clutchGain = 40.0f;
                const float newEngineOmega =
                    (engineInertia * engineOmega + h * (netEngineTorque + clutchGain * drivetrainOmega)) /
                    std::max(engineInertia + h * clutchGain, 1e-6f);
                transmittedClutchTorque =
                    std::clamp(clutchGain * (newEngineOmega - drivetrainOmega), -500.0f, 1200.0f);
                engineOmega = newEngineOmega;
            } else {
                const float targetLaunchRpm = idleRpm + cleanThrottle * 1400.0f;
                const float targetLaunchOmega = std::max(targetLaunchRpm * kRpmToRadS, drivetrainOmega);
                const float launchSlip =
                    std::clamp((engineRpm_ - idleRpm * 0.90f) / 1000.0f, 0.25f, 1.0f);
                const float speedBlend = std::clamp(drivetrainRpm / launchLockRpm, 0.0f, 1.0f);
                clutchEngagement_ = std::max(launchSlip, speedBlend);

                transmittedClutchTorque =
                    clutchEngagement_ * std::max(combustionTorque, 0.0f) -
                    speedBlend * engineBrake;
                const float launchTorqueLeft = netEngineTorque - transmittedClutchTorque * 0.65f;
                engineOmega += (launchTorqueLeft / engineInertia) * h;
                if (cleanThrottle > 0.05f && engineOmega < targetLaunchOmega) {
                    engineOmega += (targetLaunchOmega - engineOmega) * std::min(1.0f, h * 6.0f);
                }
            }
        }

        engineOmega = std::clamp(engineOmega, idleOmega * 0.7f, revLimiterRpm * kRpmToRadS);
        engineRpm_ = engineOmega * kRadSToRpm;

        const float totalAxleTorque = transmittedClutchTorque * totalRatio * efficiency;
        const float frontAxleTorque = frontDriven ? totalAxleTorque * frontSplit : 0.0f;
        const float rearAxleTorque = rearDriven ? totalAxleTorque * rearSplit : 0.0f;

        const float maxBrake = std::max(0.0f, sanitizeFinite(pt.maxBrakeTorqueNm));
        const float frontBias = std::clamp(sanitizeFinite(pt.frontBrakeBias, 0.65f), 0.0f, 1.0f);
        const float frontBrakePerWheel = cleanBrake * maxBrake * frontBias;
        const float rearBrakePerWheel = cleanBrake * maxBrake * (1.0f - frontBias);
        const float rearHandbrakePerWheel =
            cleanHandbrake * std::max(0.0f, sanitizeFinite(pt.maxHandbrakeTorqueNm));

        stepAxleWheels(pt, config_.tire, config_.frontDifferential, frontAxleTorque,
                       frontBrakePerWheel, 0.0f, steerRad, steerRad, vForward, vLateral, cleanLoads,
                       wheelContacts, wheelGrips_, 0, 1, frontDriven, h, wheelAngularVel_,
                       driveTorqueLast, lonForceAccum, latForceAccum, slipRatioLast, slipAngleLast);

        stepAxleWheels(pt, config_.tire, config_.rearDifferential, rearAxleTorque,
                       rearBrakePerWheel, rearHandbrakePerWheel, 0.0f, 0.0f, vForward, vLateral,
                       cleanLoads, wheelContacts, wheelGrips_, 2, 3, rearDriven, h, wheelAngularVel_,
                       driveTorqueLast, lonForceAccum, latForceAccum, slipRatioLast, slipAngleLast);

        for (std::size_t i = 0; i < kVehicleWheelCount; ++i) {
            stepSuspensionWheel(config_.suspension, cleanLoads[i], wheelContacts[i], h,
                                suspensionCompression_[i], suspensionCompressionVel_[i]);
        }

        if (sub == substeps - 1) {
            updateAutoShift(drivetrainRpm);
        }
    }

    const float invSubsteps = 1.0f / static_cast<float>(substeps);
    VehicleTelemetry telemetry{};
    telemetry.engineRpm = engineRpm_;
    telemetry.currentGear = currentGear_;
    telemetry.speedKmh = std::hypot(vForward, vLateral) * kMsToKmh;
    telemetry.engineTorqueNm = lastEngineTorque;
    telemetry.clutchEngagement = clutchEngagement_;
    telemetry.isShifting = shiftTimer_ > 0.0f;
    telemetry.antiRollForceFrontN = computeAntiRollForce(
        suspensionCompression_[0], suspensionCompression_[1], config_.suspension.antiRollStiffness);
    telemetry.antiRollForceRearN = computeAntiRollForce(
        suspensionCompression_[2], suspensionCompression_[3], config_.suspension.antiRollStiffness);

    for (std::size_t i = 0; i < kVehicleWheelCount; ++i) {
        WheelTelemetry& wt = telemetry.wheels[i];
        wt.slipRatio = slipRatioLast[i];
        wt.slipAngle = slipAngleLast[i];
        wt.lonForce = lonForceAccum[i] * invSubsteps;
        wt.latForce = latForceAccum[i] * invSubsteps;
        wt.compression = suspensionCompression_[i];
        wt.compressionVel = suspensionCompressionVel_[i];
        wt.angularVelRadS = wheelAngularVel_[i];
        wt.driveTorqueNm = driveTorqueLast[i];
        wt.normalLoadN = wheelContacts[i] ? cleanLoads[i] : 0.0f;
        wt.inContact = wheelContacts[i];
    }

    const float speedMs = std::hypot(vForward, vLateral);
    const bool rearGrounded = wheelContacts[2] || wheelContacts[3];
    if (speedMs >= config_.driftSpeedThresholdMs && rearGrounded) {
        const float bodySlip = std::atan2(std::abs(vLateral), std::max(std::abs(vForward), 0.5f));
        const float maxRearSlip =
            std::max(std::abs(telemetry.wheels[2].slipAngle), std::abs(telemetry.wheels[3].slipAngle));
        telemetry.isDrifting =
            (bodySlip >= config_.driftSlipAngleThresholdRad) ||
            (maxRearSlip >= config_.driftSlipAngleThresholdRad);
    }

    lastTelemetry_ = telemetry;
    return telemetry;
}

VehicleTelemetry VehicleDrivetrainSim::step(
    float throttle, float brake, float handbrake, float steerDeg,
    const float (&wheelLoadsN)[kVehicleWheelCount],
    const bool (&wheelContacts)[kVehicleWheelCount], float forwardSpeedMs, float lateralSpeedMs,
    float dt) noexcept {
    const std::array<float, kVehicleWheelCount> loads{
        wheelLoadsN[0], wheelLoadsN[1], wheelLoadsN[2], wheelLoadsN[3]};
    const std::array<bool, kVehicleWheelCount> contacts{
        wheelContacts[0], wheelContacts[1], wheelContacts[2], wheelContacts[3]};
    return step(throttle, brake, handbrake, steerDeg, loads, contacts, forwardSpeedMs,
                lateralSpeedMs, dt);
}

} // namespace lumora
