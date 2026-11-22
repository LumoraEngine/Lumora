#include "runtime/VehicleDynamics.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

using namespace lumora;

namespace {

void expectAllTelemetryFinite(const VehicleTelemetry& t) {
    EXPECT_TRUE(std::isfinite(t.engineRpm));
    EXPECT_TRUE(std::isfinite(t.speedKmh));
    EXPECT_TRUE(std::isfinite(t.engineTorqueNm));
    EXPECT_TRUE(std::isfinite(t.clutchEngagement));
    EXPECT_TRUE(std::isfinite(t.antiRollForceFrontN));
    EXPECT_TRUE(std::isfinite(t.antiRollForceRearN));
    for (std::size_t i = 0; i < kVehicleWheelCount; ++i) {
        const auto& w = t.wheels[i];
        EXPECT_TRUE(std::isfinite(w.slipRatio)) << "wheel " << i;
        EXPECT_TRUE(std::isfinite(w.slipAngle)) << "wheel " << i;
        EXPECT_TRUE(std::isfinite(w.lonForce)) << "wheel " << i;
        EXPECT_TRUE(std::isfinite(w.latForce)) << "wheel " << i;
        EXPECT_TRUE(std::isfinite(w.compression)) << "wheel " << i;
        EXPECT_TRUE(std::isfinite(w.compressionVel)) << "wheel " << i;
        EXPECT_TRUE(std::isfinite(w.angularVelRadS)) << "wheel " << i;
        EXPECT_TRUE(std::isfinite(w.driveTorqueNm)) << "wheel " << i;
        EXPECT_TRUE(std::isfinite(w.normalLoadN)) << "wheel " << i;
    }
}

} // namespace

TEST(VehicleDynamicsTire, PacejkaPeaksAtRealisticSlipAndDropsOffInDriftRegime) {
    PacejkaParams params;

    float peakSlip = 0.0f;
    float peakForce = 0.0f;
    constexpr float normalLoadN = 4000.0f;
    for (int i = 0; i <= 1000; ++i) {
        const float slip = static_cast<float>(i) * 0.001f;  // 0 .. 1.0
        const TireForces f = computeTireForces(normalLoadN, slip, 0.0f, params, 1.0f);
        EXPECT_TRUE(std::isfinite(f.longitudinal));
        EXPECT_NEAR(f.lateral, 0.0f, 1e-4f);
        if (f.longitudinal > peakForce) {
            peakForce = f.longitudinal;
            peakSlip = slip;
        }
    }

    EXPECT_GE(peakSlip, 0.08f) << "longitudinal Pacejka peak should occur at >= 8% slip";
    EXPECT_LE(peakSlip, 0.15f) << "longitudinal Pacejka peak should occur at <= 15% slip";
    EXPECT_NEAR(peakForce, normalLoadN * params.longitudinal.d, 5.0f);

    // Post-peak drop-off in heavy wheelspin / drift regime.
    const TireForces postPeak50 = computeTireForces(normalLoadN, 0.50f, 0.0f, params, 1.0f);
    const TireForces postPeak100 = computeTireForces(normalLoadN, 1.00f, 0.0f, params, 1.0f);
    EXPECT_GT(postPeak50.longitudinal, 0.0f);
    EXPECT_LT(postPeak50.longitudinal, peakForce * 0.90f);
    EXPECT_GT(postPeak100.longitudinal, 0.0f);
    EXPECT_LT(postPeak100.longitudinal, postPeak50.longitudinal);

    // Lateral curve peaks cleanly around ~8-13 deg (0.14-0.23 rad) and drops post-peak.
    float peakAngle = 0.0f;
    float peakLatForce = 0.0f;
    for (int i = 0; i <= 1000; ++i) {
        const float angle = static_cast<float>(i) * 0.001f;  // 0 .. 1.0 rad
        const TireForces f = computeTireForces(normalLoadN, 0.0f, angle, params, 1.0f);
        if (f.lateral > peakLatForce) {
            peakLatForce = f.lateral;
            peakAngle = angle;
        }
    }
    EXPECT_GE(peakAngle, 0.10f);
    EXPECT_LE(peakAngle, 0.25f);
    EXPECT_NEAR(peakLatForce, normalLoadN * params.lateral.d, 5.0f);
    const TireForces highDrift = computeTireForces(normalLoadN, 0.0f, 0.70f, params, 1.0f);
    EXPECT_GT(highDrift.lateral, 0.0f);
    EXPECT_LT(highDrift.lateral, peakLatForce * 0.92f);

    // Odd symmetry and zero at zero slip.
    const TireForces pos = computeTireForces(normalLoadN, 0.11f, 0.0f, params);
    const TireForces neg = computeTireForces(normalLoadN, -0.11f, 0.0f, params);
    EXPECT_NEAR(pos.longitudinal, -neg.longitudinal, 1e-3f);
    const TireForces zero = computeTireForces(normalLoadN, 0.0f, 0.0f, params);
    EXPECT_FLOAT_EQ(zero.longitudinal, 0.0f);
    EXPECT_FLOAT_EQ(zero.lateral, 0.0f);
}

TEST(VehicleDynamicsTire, SlipCalculationsHandleZeroSpeedAndNanSafely) {
    // Exact slip ratio at speed: (22 - 20) / 20 == 0.10.
    EXPECT_NEAR(computeSlipRatio(22.0f / 0.35f, 0.35f, 20.0f), 0.10f, 1e-4f);
    // Locked wheel braking: (0 - 20) / 20 == -1.0.
    EXPECT_NEAR(computeSlipRatio(0.0f, 0.35f, 20.0f), -1.0f, 1e-4f);
    // Zero-speed safety without NaN/explosion.
    EXPECT_FLOAT_EQ(computeSlipRatio(0.0f, 0.35f, 0.0f), 0.0f);
    const float standstillSpin = computeSlipRatio(30.0f, 0.35f, 0.0f);
    EXPECT_TRUE(std::isfinite(standstillSpin));
    EXPECT_GT(standstillSpin, 0.0f);
    EXPECT_LE(standstillSpin, 2.0f);
    EXPECT_FLOAT_EQ(computeSlipRatio(10.0f, 0.0f, 5.0f), 0.0f);

    // Slip angle at standstill and NaN guard.
    EXPECT_FLOAT_EQ(computeSlipAngle(0.0f, 0.0f, 0.0f), 0.0f);
    const float nanVal = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FLOAT_EQ(computeSlipRatio(nanVal, 0.35f, 10.0f), 0.0f);
    EXPECT_FLOAT_EQ(computeSlipAngle(nanVal, 10.0f, 0.1f), 0.0f);

    // Steering right while moving straight forward yields positive slip angle and positive lateral force.
    const float steerRad = 0.15f;
    const float alpha = computeSlipAngle(0.0f, 20.0f, steerRad);
    EXPECT_NEAR(alpha, steerRad, 1e-4f);
    PacejkaParams params;
    const TireForces steerForces = computeTireForces(4000.0f, 0.0f, alpha, params);
    EXPECT_GT(steerForces.lateral, 1000.0f);

    // Sliding straight in reverse does not produce 180-deg slip angle.
    EXPECT_NEAR(computeSlipAngle(0.0f, -10.0f, 0.0f), 0.0f, 1e-5f);
}

TEST(VehicleDynamicsTire, FrictionEllipseBoundsCombinedBrakingAndCornering) {
    PacejkaParams params;
    constexpr float normalLoadN = 4000.0f;
    constexpr float grip = 0.85f;
    const float fxMax = normalLoadN * grip * params.longitudinal.d;
    const float fyMax = normalLoadN * grip * params.lateral.d;

    const TireForces pureBrake = computeTireForces(normalLoadN, -0.12f, 0.0f, params, grip);
    const TireForces pureCorner = computeTireForces(normalLoadN, 0.0f, 0.17f, params, grip);
    const TireForces combined = computeTireForces(normalLoadN, -0.12f, 0.17f, params, grip);

    EXPECT_LT(pureBrake.longitudinal, -0.9f * fxMax);
    EXPECT_GT(pureCorner.lateral, 0.9f * fyMax);

    // Combined braking + cornering sacrifices both compared to pure maneuvers.
    EXPECT_LT(std::abs(combined.longitudinal), std::abs(pureBrake.longitudinal));
    EXPECT_LT(std::abs(combined.lateral), std::abs(pureCorner.lateral));

    // Normalized friction ellipse bound holds across a sweep of slip ratios and angles.
    for (float kappa : {-1.0f, -0.3f, -0.12f, -0.05f, 0.0f, 0.05f, 0.12f, 0.5f}) {
        for (float angle : {-0.6f, -0.17f, -0.05f, 0.0f, 0.05f, 0.17f, 0.6f}) {
            const TireForces f = computeTireForces(normalLoadN, kappa, angle, params, grip);
            const float ellipseNorm = std::hypot(f.longitudinal / fxMax, f.lateral / fyMax);
            EXPECT_LE(ellipseNorm, 1.0f + 1e-4f)
                << "kappa=" << kappa << " angle=" << angle;
        }
    }

    // Zero / negative load or grip gives zero force.
    const TireForces zeroLoad = computeTireForces(0.0f, 0.12f, 0.15f, params, 1.0f);
    EXPECT_FLOAT_EQ(zeroLoad.longitudinal, 0.0f);
    EXPECT_FLOAT_EQ(zeroLoad.lateral, 0.0f);
}

TEST(VehicleDynamicsSuspension, ReachesStaticEquilibriumAndModelsAsymmetricDampingAndAntiRoll) {
    SuspensionParams susp;
    susp.stiffness = 32000.0f;
    susp.bumpDamping = 2400.0f;
    susp.reboundDamping = 4800.0f;
    susp.maxTravel = 0.25f;
    susp.bumpStopStiffness = 160000.0f;

    constexpr float cornerMassKg = 360.0f;
    constexpr float g = 9.81f;
    constexpr float dt = 1.0f / 120.0f;

    float compression = 0.0f;
    float compressionVel = 0.0f;
    for (int step = 0; step < 600; ++step) {
        const float hookeForce = computeSuspensionForce(compression, compressionVel, susp);
        const float accel = g + hookeForce / cornerMassKg;
        compressionVel += accel * dt;
        compression += compressionVel * dt;
        if (compression < 0.0f) {
            compression = 0.0f;
            compressionVel = std::max(0.0f, compressionVel);
        }
    }

    const float expectedCompression = (cornerMassKg * g) / susp.stiffness;
    EXPECT_NEAR(compression, expectedCompression, 1e-3f);
    EXPECT_NEAR(susp.stiffness * compression, cornerMassKg * g, 5.0f);
    EXPECT_NEAR(compressionVel, 0.0f, 1e-2f);

    // Asymmetric bump vs rebound damping: rebound magnitude > bump magnitude at equal |v|.
    const float fBump = computeSuspensionForce(0.10f, 0.5f, susp);
    const float fRebound = computeSuspensionForce(0.10f, -0.5f, susp);
    const float fStatic = computeSuspensionForce(0.10f, 0.0f, susp);
    EXPECT_NEAR(std::abs(fBump - fStatic), susp.bumpDamping * 0.5f, 1e-3f);
    EXPECT_NEAR(std::abs(fRebound - fStatic), susp.reboundDamping * 0.5f, 1e-3f);
    EXPECT_GT(std::abs(fRebound - fStatic), std::abs(fBump - fStatic));

    // Bump stop engages sharply past maxTravel.
    const float fBeforeStop = computeSuspensionForce(0.25f, 0.0f, susp);
    const float fPastStop = computeSuspensionForce(0.27f, 0.0f, susp);
    EXPECT_LT(fPastStop - fBeforeStop, -susp.stiffness * 0.02f - susp.bumpStopStiffness * 0.019f);

    // Anti-roll force opposes roll asymmetry and vanishes on equal compression.
    EXPECT_FLOAT_EQ(computeAntiRollForce(0.12f, 0.12f, susp.antiRollStiffness), 0.0f);
    const float arb = computeAntiRollForce(0.15f, 0.05f, susp.antiRollStiffness);
    EXPECT_NEAR(arb, 0.10f * susp.antiRollStiffness, 1e-3f);
    EXPECT_NEAR(computeAntiRollForce(0.05f, 0.15f, susp.antiRollStiffness), -arb, 1e-3f);
}

TEST(VehicleDynamicsPowertrain, AcceleratesCleanlyFromZeroThroughGears1To3) {
    VehicleDynamicsConfig cfg;
    cfg.driveLayout = DriveLayout::Awd;
    cfg.powertrain.autoShift = true;
    VehicleDrivetrainSim sim(cfg);

    // Verify piecewise-linear torque curve interpolation directly.
    EXPECT_NEAR(sim.sampleEngineTorque(800.0f), 180.0f, 1e-4f);
    EXPECT_NEAR(sim.sampleEngineTorque(1400.0f), 230.0f, 1e-4f);
    EXPECT_NEAR(sim.sampleEngineTorque(5000.0f), 400.0f, 1e-4f);
    EXPECT_FLOAT_EQ(sim.sampleEngineTorque(0.0f), 0.0f);

    constexpr float vehicleMassKg = 1400.0f;
    constexpr float aeroDragCoeff = 0.35f;
    constexpr float dt = 1.0f / 60.0f;
    const std::array<float, 4> loads{3500.0f, 3500.0f, 3500.0f, 3500.0f};
    const std::array<bool, 4> contacts{true, true, true, true};

    float forwardSpeedMs = 0.0f;
    bool sawGear1 = false;
    bool sawGear2 = false;
    bool sawGear3 = false;
    int maxGearReached = 0;

    for (int frame = 0; frame < 900; ++frame) {  // 15 seconds at 60 Hz
        const VehicleTelemetry tel =
            sim.step(1.0f, 0.0f, 0.0f, 0.0f, loads, contacts, forwardSpeedMs, 0.0f, dt);
        expectAllTelemetryFinite(tel);

        EXPECT_GE(tel.engineRpm, cfg.powertrain.idleRpm * 0.7f);
        EXPECT_LE(tel.engineRpm, cfg.powertrain.revLimiterRpm + 1.0f);

        if (tel.currentGear == 1) sawGear1 = true;
        if (tel.currentGear == 2) sawGear2 = true;
        if (tel.currentGear == 3) sawGear3 = true;
        maxGearReached = std::max(maxGearReached, tel.currentGear);

        // Expect monotonic non-decreasing gear progression during continuous full throttle.
        EXPECT_GE(tel.currentGear, maxGearReached) << "shift hysteresis must prevent gear hunting";

        float totalLonForce = 0.0f;
        for (const auto& w : tel.wheels) {
            totalLonForce += w.lonForce;
        }
        const float drag = aeroDragCoeff * forwardSpeedMs * std::abs(forwardSpeedMs);
        forwardSpeedMs += ((totalLonForce - drag) / vehicleMassKg) * dt;
    }

    EXPECT_TRUE(sawGear1);
    EXPECT_TRUE(sawGear2);
    EXPECT_TRUE(sawGear3);
    EXPECT_GE(maxGearReached, 3);
    EXPECT_GT(forwardSpeedMs, 30.0f);

    // Manual shifting & Neutral / Reverse checks.
    sim.reset();
    sim.setAutoShift(false);
    EXPECT_TRUE(sim.setGear(0));
    const VehicleTelemetry neutralTel =
        sim.step(1.0f, 0.0f, 0.0f, 0.0f, loads, contacts, 10.0f, 0.0f, dt);
    expectAllTelemetryFinite(neutralTel);
    EXPECT_EQ(neutralTel.currentGear, 0);
    for (const auto& w : neutralTel.wheels) {
        EXPECT_FLOAT_EQ(w.driveTorqueNm, 0.0f);
    }

    EXPECT_TRUE(sim.setGear(-1));
    VehicleTelemetry reverseTel{};
    for (int i = 0; i < 15; ++i) {
        reverseTel = sim.step(0.8f, 0.0f, 0.0f, 0.0f, loads, contacts, 0.0f, 0.0f, dt);
        expectAllTelemetryFinite(reverseTel);
    }
    EXPECT_EQ(reverseTel.currentGear, -1);
    EXPECT_LT(reverseTel.wheels[2].driveTorqueNm, -50.0f);
    EXPECT_LT(reverseTel.wheels[2].lonForce, -100.0f);
}

TEST(VehicleDynamicsDiff, LsdTransfersMoreTorqueToHighGripWheelThanOpenDiffOnIce) {
    // 1. Direct splitDifferentialTorque unit verification: left wheel spinning fast on ice.
    DifferentialParams openParams;
    openParams.type = DiffType::Open;
    DifferentialParams lsdParams;
    lsdParams.type = DiffType::LimitedSlip;
    lsdParams.lsdPreloadNm = 60.0f;
    lsdParams.lsdLockingFactor = 0.70f;
    lsdParams.viscousDamping = 10.0f;

    const DifferentialOutput openSplit = splitDifferentialTorque(1000.0f, 60.0f, 25.0f, openParams);
    const DifferentialOutput lsdSplit = splitDifferentialTorque(1000.0f, 60.0f, 25.0f, lsdParams);

    EXPECT_FLOAT_EQ(openSplit.leftTorqueNm, 500.0f);
    EXPECT_FLOAT_EQ(openSplit.rightTorqueNm, 500.0f);
    EXPECT_NEAR(lsdSplit.leftTorqueNm + lsdSplit.rightTorqueNm, 1000.0f, 1e-3f);
    EXPECT_GT(lsdSplit.rightTorqueNm, openSplit.rightTorqueNm * 1.5f);
    EXPECT_LT(lsdSplit.leftTorqueNm, openSplit.leftTorqueNm * 0.5f);

    // 2. Closed-loop drivetrain test: Rear-Left on ice (grip = 0.1), Rear-Right on dry tarmac (grip = 1.0).
    auto makeSim = [](DiffType diffType) {
        VehicleDynamicsConfig cfg;
        cfg.driveLayout = DriveLayout::Rwd;
        cfg.powertrain.autoShift = false;
        cfg.rearDifferential.type = diffType;
        cfg.rearDifferential.lsdPreloadNm = 80.0f;
        cfg.rearDifferential.lsdLockingFactor = 0.75f;
        cfg.rearDifferential.viscousDamping = 12.0f;
        VehicleDrivetrainSim sim(cfg);
        sim.setGear(2);  // 2nd gear so dry wheel doesn't saturate immediately in open diff
        sim.setWheelGrips({1.0f, 1.0f, 0.1f, 1.0f});  // RL on ice!
        return sim;
    };

    VehicleDrivetrainSim openSim = makeSim(DiffType::Open);
    VehicleDrivetrainSim lsdSim = makeSim(DiffType::LimitedSlip);
    VehicleDrivetrainSim lockedSim = makeSim(DiffType::Locked);

    const std::array<float, 4> loads{3500.0f, 3500.0f, 3500.0f, 3500.0f};
    const std::array<bool, 4> contacts{true, true, true, true};
    constexpr float dt = 1.0f / 60.0f;

    float openSpeed = 6.0f;
    float lsdSpeed = 6.0f;
    float lockedSpeed = 6.0f;
    VehicleTelemetry openTel{};
    VehicleTelemetry lsdTel{};
    VehicleTelemetry lockedTel{};

    for (int i = 0; i < 60; ++i) {
        openTel = openSim.step(0.55f, 0.0f, 0.0f, 0.0f, loads, contacts, openSpeed, 0.0f, dt);
        lsdTel = lsdSim.step(0.55f, 0.0f, 0.0f, 0.0f, loads, contacts, lsdSpeed, 0.0f, dt);
        lockedTel = lockedSim.step(0.55f, 0.0f, 0.0f, 0.0f, loads, contacts, lockedSpeed, 0.0f, dt);

        expectAllTelemetryFinite(openTel);
        expectAllTelemetryFinite(lsdTel);
        expectAllTelemetryFinite(lockedTel);

        openSpeed += ((openTel.wheels[2].lonForce + openTel.wheels[3].lonForce) / 1400.0f) * dt;
        lsdSpeed += ((lsdTel.wheels[2].lonForce + lsdTel.wheels[3].lonForce) / 1400.0f) * dt;
        lockedSpeed +=
            ((lockedTel.wheels[2].lonForce + lockedTel.wheels[3].lonForce) / 1400.0f) * dt;
    }

    constexpr std::size_t kRL = static_cast<std::size_t>(WheelIndex::RearLeft);
    constexpr std::size_t kRR = static_cast<std::size_t>(WheelIndex::RearRight);

    // Open diff wastes torque spinning the ice wheel; LSD & Locked send much more to high-grip RR.
    EXPECT_GT(openTel.wheels[kRL].angularVelRadS, lsdTel.wheels[kRL].angularVelRadS * 1.2f);
    EXPECT_GT(lsdTel.wheels[kRR].driveTorqueNm, openTel.wheels[kRR].driveTorqueNm * 1.4f);
    EXPECT_GT(lsdTel.wheels[kRR].lonForce, openTel.wheels[kRR].lonForce * 1.3f);
    EXPECT_GT(lsdSpeed, openSpeed);
    EXPECT_NEAR(lockedTel.wheels[kRL].angularVelRadS, lockedTel.wheels[kRR].angularVelRadS, 1e-3f);
    EXPECT_GT(lockedSpeed, openSpeed);
}

TEST(VehicleDynamicsDrivetrain, DriftingHandbrakeAndEdgeCasesRemainStable) {
    VehicleDynamicsConfig cfg;
    cfg.driveLayout = DriveLayout::Rwd;
    VehicleDrivetrainSim sim(cfg);

    const std::array<float, 4> loads{3500.0f, 3500.0f, 3500.0f, 3500.0f};
    const std::array<bool, 4> grounded{true, true, true, true};
    const std::array<bool, 4> airborne{false, false, false, false};
    constexpr float dt = 1.0f / 60.0f;

    // Handbrake + lateral slide at speed should lock rear wheels and report drifting.
    VehicleTelemetry driftTel{};
    for (int i = 0; i < 30; ++i) {
        driftTel = sim.step(0.0f, 0.0f, 1.0f, 20.0f, loads, grounded, 18.0f, 6.0f, dt);
        expectAllTelemetryFinite(driftTel);
    }
    EXPECT_TRUE(driftTel.isDrifting);
    EXPECT_NEAR(driftTel.wheels[2].angularVelRadS, 0.0f, 0.5f);
    EXPECT_NEAR(driftTel.wheels[3].angularVelRadS, 0.0f, 0.5f);
    EXPECT_NEAR(driftTel.wheels[0].compression, loads[0] / cfg.suspension.stiffness, 0.02f);

    // Airborne state zeroes tire forces and relaxes suspension compression cleanly.
    VehicleTelemetry airTel{};
    for (int i = 0; i < 180; ++i) {
        airTel = sim.step(0.0f, 0.0f, 0.0f, 0.0f, loads, airborne, 15.0f, 0.0f, dt);
        expectAllTelemetryFinite(airTel);
    }
    for (const auto& w : airTel.wheels) {
        EXPECT_FLOAT_EQ(w.lonForce, 0.0f);
        EXPECT_FLOAT_EQ(w.latForce, 0.0f);
        EXPECT_NEAR(w.compression, 0.0f, 1e-3f);
    }

    // Zero/negative dt and NaN inputs are harmless.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const VehicleTelemetry badDtTel =
        sim.step(nan, nan, nan, nan, loads, grounded, nan, nan, 0.0f);
    expectAllTelemetryFinite(badDtTel);
    const VehicleTelemetry badInputsTel =
        sim.step(nan, nan, nan, nan, loads, grounded, nan, nan, dt);
    expectAllTelemetryFinite(badInputsTel);
}

TEST(VehicleDynamicsPowertrain, EngineBrakingClutchOverrideDownshiftAndCArrayOverload) {
    VehicleDynamicsConfig cfg;
    cfg.driveLayout = DriveLayout::Rwd;
    VehicleDrivetrainSim sim(cfg);

    const float rawLoads[4]{3500.0f, 3500.0f, 3500.0f, 3500.0f};
    const bool rawContacts[4]{true, true, true, true};
    constexpr float dt = 1.0f / 60.0f;

    // 1. Clutch override = 0 disconnects drive torque while engine revs freely on throttle.
    sim.setClutchOverride(0.0f);
    VehicleTelemetry revTel{};
    for (int i = 0; i < 30; ++i) {
        revTel = sim.step(1.0f, 0.0f, 0.0f, 0.0f, rawLoads, rawContacts, 0.0f, 0.0f, dt);
        expectAllTelemetryFinite(revTel);
    }
    EXPECT_FLOAT_EQ(revTel.clutchEngagement, 0.0f);
    EXPECT_GT(revTel.engineRpm, 4000.0f);
    EXPECT_FLOAT_EQ(revTel.wheels[2].driveTorqueNm, 0.0f);
    EXPECT_FLOAT_EQ(revTel.wheels[3].driveTorqueNm, 0.0f);
    sim.clearClutchOverride();

    // 2. Engine braking at speed in 2nd gear with closed throttle produces braking force.
    sim.reset();
    sim.setAutoShift(false);
    EXPECT_TRUE(sim.setGear(2));
    VehicleTelemetry coastTel{};
    for (int i = 0; i < 30; ++i) {
        coastTel = sim.step(0.0f, 0.0f, 0.0f, 0.0f, rawLoads, rawContacts, 22.0f, 0.0f, dt);
        expectAllTelemetryFinite(coastTel);
    }
    EXPECT_LT(coastTel.wheels[2].driveTorqueNm, -10.0f);
    EXPECT_LT(coastTel.wheels[2].lonForce, -20.0f);

    // 3. Auto-downshift when slowing from 3rd gear toward standstill.
    sim.setAutoShift(true);
    EXPECT_TRUE(sim.setGear(3));
    float speed = 35.0f;
    int minGearSeenAfterSlowdown = 6;
    for (int i = 0; i < 240; ++i) {
        speed = std::max(0.0f, speed - 0.18f);
        const VehicleTelemetry decelTel =
            sim.step(0.0f, 0.4f, 0.0f, 0.0f, rawLoads, rawContacts, speed, 0.0f, dt);
        expectAllTelemetryFinite(decelTel);
        minGearSeenAfterSlowdown = std::min(minGearSeenAfterSlowdown, decelTel.currentGear);
    }
    EXPECT_EQ(minGearSeenAfterSlowdown, 1);

    // 4. Direct evaluatePacejka & computeSuspensionSupportLoad helpers match contract.
    EXPECT_NEAR(evaluatePacejka(0.118f, cfg.tire.longitudinal), cfg.tire.longitudinal.d, 0.01f);
    EXPECT_FLOAT_EQ(computeSuspensionSupportLoad(-0.05f, -0.2f, cfg.suspension), 0.0f);
    EXPECT_NEAR(computeSuspensionSupportLoad(0.10f, 0.0f, cfg.suspension),
                0.10f * cfg.suspension.stiffness, 1e-2f);
}

