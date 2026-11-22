#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace lumora {

/// Wheel slot indices matching Lumora's chassis convention: FL, FR, RL, RR.
enum class WheelIndex : std::size_t {
    FrontLeft = 0,
    FrontRight = 1,
    RearLeft = 2,
    RearRight = 3
};

inline constexpr std::size_t kVehicleWheelCount = 4;

/// Parameters for a single Pacejka Magic Formula curve:
///   y(x) = d * sin(c * atan(b * x - e * (b * x - atan(b * x))))
struct PacejkaCurve {
    float b = 14.0f;  ///< Stiffness factor (initial slope around zero slip).
    float c = 1.65f;  ///< Shape factor (controls peak vs post-peak drop-off; typically 1.2..1.9).
    float d = 1.10f;  ///< Peak normalized friction coefficient (mu_peak).
    float e = 0.40f;  ///< Curvature factor (<= 1.0; shapes transition around the peak).
};

/// Tire model configuration containing longitudinal and lateral Pacejka curves.
struct PacejkaParams {
    PacejkaCurve longitudinal{14.0f, 1.65f, 1.10f, 0.40f};
    PacejkaCurve lateral{11.0f, 1.55f, 1.15f, 0.40f};
    float loadSensitivity = 0.0f;       ///< Optional peak-mu reduction with load (0 = strictly linear in normalLoadN).
    float referenceLoadN = 4000.0f;     ///< Nominal wheel load used when loadSensitivity > 0 (N).
    float maxCombinedGripRatio = 1.0f;  ///< Normalized friction-ellipse budget radius.
};

/// Resulting tire forces in the wheel-local horizontal plane.
struct TireForces {
    float longitudinal = 0.0f;  ///< Force along wheel forward axis (+N drives forward, -N brakes).
    float lateral = 0.0f;       ///< Force along wheel right axis (+N pushes right).
};

/// Evaluates the raw Pacejka Magic Formula y(x) = d * sin(c * atan(b * x - e * (b * x - atan(b * x)))).
[[nodiscard]] float evaluatePacejka(float slip, float b, float c, float d, float e) noexcept;
[[nodiscard]] float evaluatePacejka(float slip, const PacejkaCurve& curve) noexcept;

/// Estimates the positive slip value at which a Pacejka curve reaches its peak.
[[nodiscard]] float estimatePacejkaPeakSlip(const PacejkaCurve& curve) noexcept;

/// Computes longitudinal slip ratio (omega * R - v_lon) / max(|v_lon|, minSpeedEps),
/// safely clamped and guarded at zero/low longitudinal speed.
[[nodiscard]] float computeSlipRatio(float wheelAngularVelRadS, float wheelRadius, float longitudinalSpeed,
                                     float minSpeedEps = 0.5f) noexcept;

/// Computes wheel slip angle in radians in the steered wheel frame (positive when wheel points
/// right relative to velocity vector; localVelX = +right, localVelZ = +forward, steerRad = +right).
[[nodiscard]] float computeSlipAngle(float localVelX, float localVelZ, float steerRad,
                                     float minSpeedEps = 0.5f) noexcept;

/// Computes combined longitudinal and lateral tire forces (N) respecting the friction-ellipse
/// traction budget under simultaneous acceleration/braking and cornering.
[[nodiscard]] TireForces computeTireForces(float normalLoadN, float slipRatio, float slipAngleRad,
                                           const PacejkaParams& params, float surfaceGrip = 1.0f) noexcept;

/// Hooke spring + asymmetric bump/rebound damper + bump stop + anti-roll bar parameters.
struct SuspensionParams {
    float stiffness = 35000.0f;          ///< Spring constant k (N/m).
    float bumpDamping = 2800.0f;         ///< Damping coefficient during compression (compressionVel >= 0, N*s/m).
    float reboundDamping = 4200.0f;      ///< Damping coefficient during extension (compressionVel < 0, N*s/m).
    float restLength = 0.35f;            ///< Uncompressed strut length (m).
    float maxTravel = 0.25f;             ///< Compression threshold where the bump stop engages (m).
    float bumpStopStiffness = 150000.0f; ///< Extra stiffness applied beyond maxTravel (N/m).
    float antiRollStiffness = 12000.0f;  ///< Anti-roll bar stiffness across left/right wheels (N/m).
    float cornerMassKg = 350.0f;         ///< Effective sprung corner mass for internal state stepping (kg).
};

/// Computes Hooke restoring force F = -k * compression - c * compressionVel (with bump vs rebound
/// damping and bump stop stiffness past maxTravel).
[[nodiscard]] float computeSuspensionForce(float compression, float compressionVel,
                                           const SuspensionParams& params) noexcept;

/// Computes non-negative upward support load on the chassis/tire from suspension state.
[[nodiscard]] float computeSuspensionSupportLoad(float compression, float compressionVel,
                                                 const SuspensionParams& params) noexcept;

/// Computes anti-roll bar force delta from left/right suspension compressions:
/// (+force applied to left wheel, -force applied to right wheel).
[[nodiscard]] float computeAntiRollForce(float leftCompression, float rightCompression,
                                         float antiRollStiffness) noexcept;

/// A single sample on the piecewise-linear engine RPM torque curve.
struct TorqueCurvePoint {
    float rpm = 0.0f;
    float torqueNm = 0.0f;
};

/// Engine, gearbox, and brake configuration.
/// Gear ratios array layout: [0] = Reverse, [1] = Neutral, [2..7] = Gears 1..6.
struct PowertrainParams {
    std::vector<TorqueCurvePoint> torqueCurve{
        {800.0f, 180.0f},
        {2000.0f, 280.0f},
        {3500.0f, 360.0f},
        {5000.0f, 400.0f},
        {6200.0f, 370.0f},
        {7000.0f, 300.0f},
    };
    float idleRpm = 850.0f;
    float redlineRpm = 7000.0f;
    float revLimiterRpm = 7200.0f;
    float engineBrakingTorque = 65.0f;  ///< Peak closed-throttle engine braking torque near redline (Nm).
    float engineInertia = 0.25f;        ///< Engine rotational inertia (kg*m^2).
    std::array<float, 8> gearRatios{-3.40f, 0.0f, 3.60f, 2.19f, 1.41f, 1.00f, 0.83f, 0.69f};
    float finalDriveRatio = 3.73f;
    float drivetrainEfficiency = 0.90f;
    float upshiftRpm = 6000.0f;         ///< Auto-shift up threshold (RPM).
    float downshiftRpm = 2400.0f;       ///< Auto-shift down threshold (RPM; provides hysteresis).
    float shiftDurationSec = 0.10f;     ///< Clutch dip duration during gear transitions (s).
    float shiftCooldownSec = 0.25f;     ///< Minimum time between automatic shifts (s).
    float maxBrakeTorqueNm = 3500.0f;   ///< Total per-wheel hydraulic brake torque capacity (Nm).
    float maxHandbrakeTorqueNm = 4500.0f; ///< Rear wheel handbrake torque capacity (Nm).
    float frontBrakeBias = 0.65f;       ///< Front brake distribution fraction [0..1].
    float wheelRadius = 0.35f;          ///< Nominal wheel radius (m).
    float wheelInertia = 1.2f;          ///< Single wheel rotational inertia (kg*m^2).
    float maxSteerDeg = 45.0f;          ///< Maximum steering angle clamp (deg).
    bool autoShift = true;

    /// Samples the piecewise-linear torque curve at the given RPM.
    [[nodiscard]] float sampleEngineTorque(float rpm) const noexcept;

    /// Returns the gear ratio for gear in [-1..6] (-1 = Reverse, 0 = Neutral, 1..6 = Forward).
    [[nodiscard]] float gearRatio(int gear) const noexcept;
};

/// Axle differential locking behavior.
enum class DiffType : std::uint8_t {
    Open = 0,
    Locked = 1,
    LimitedSlip = 2
};

/// Vehicle drive axle configuration.
enum class DriveLayout : std::uint8_t {
    Rwd = 0,
    Fwd = 1,
    Awd = 2
};

/// Differential configuration supporting Open, Locked, and clutch/viscous LSD modes.
struct DifferentialParams {
    DiffType type = DiffType::LimitedSlip;
    float lsdPreloadNm = 50.0f;       ///< Static preload torque resisting wheel speed mismatch (Nm).
    float lsdLockingFactor = 0.65f;   ///< Torque-sensing locking factor [0..1].
    float viscousDamping = 8.0f;      ///< Speed-sensitive viscous coupling gain (Nm / (rad/s)).
    float lockGain = 250.0f;          ///< Coupling stiffness for DiffType::Locked (Nm / (rad/s)).
    float smoothingEpsRadS = 0.25f;   ///< Velocity delta smoothing band near zero slip (rad/s).
};

/// Output torque split across an axle's left and right half-shafts.
struct DifferentialOutput {
    float leftTorqueNm = 0.0f;
    float rightTorqueNm = 0.0f;
};

/// Splits axle drive torque between left and right wheels according to the differential model.
/// Transfers torque from the faster-spinning wheel toward the slower (higher-grip) wheel in LSD/Locked modes.
[[nodiscard]] DifferentialOutput splitDifferentialTorque(float inputTorqueNm, float leftWheelVelRadS,
                                                       float rightWheelVelRadS,
                                                       const DifferentialParams& params) noexcept;

/// Aggregate configuration for VehicleDrivetrainSim.
struct VehicleDynamicsConfig {
    PacejkaParams tire{};
    SuspensionParams suspension{};
    PowertrainParams powertrain{};
    DifferentialParams frontDifferential{};
    DifferentialParams rearDifferential{};
    DriveLayout driveLayout = DriveLayout::Awd;
    float frontTorqueBias = 0.50f;            ///< Front-axle torque fraction when driveLayout == Awd [0..1].
    float driftSpeedThresholdMs = 3.0f;       ///< Minimum speed to classify drifting (m/s).
    float driftSlipAngleThresholdRad = 0.175f;///< Slip angle threshold to classify drifting (~10 deg).
};

/// Per-wheel telemetry snapshot from VehicleDrivetrainSim::step.
struct WheelTelemetry {
    float slipRatio = 0.0f;
    float slipAngle = 0.0f;    ///< Wheel slip angle (rad).
    float lonForce = 0.0f;     ///< Longitudinal tire force (N).
    float latForce = 0.0f;     ///< Lateral tire force (N).
    float compression = 0.0f;  ///< Suspension spring compression (m).
    float compressionVel = 0.0f;
    float angularVelRadS = 0.0f;
    float driveTorqueNm = 0.0f;
    float normalLoadN = 0.0f;
    bool inContact = false;
};

/// Full vehicle telemetry returned by VehicleDrivetrainSim::step.
struct VehicleTelemetry {
    float engineRpm = 0.0f;
    int currentGear = 1;       ///< -1 = Reverse, 0 = Neutral, 1..6 = Forward gears.
    float speedKmh = 0.0f;
    float engineTorqueNm = 0.0f;
    float clutchEngagement = 0.0f;
    float antiRollForceFrontN = 0.0f;
    float antiRollForceRearN = 0.0f;
    std::array<WheelTelemetry, kVehicleWheelCount> wheels{};
    bool isDrifting = false;
    bool isShifting = false;
};

/// Stateful vehicle powertrain, differential, suspension, and Pacejka tire simulator.
/// Coordinate convention: forwardSpeedMs (+forward), lateralSpeedMs (+right), steerDeg (+right).
class VehicleDrivetrainSim {
public:
    VehicleDrivetrainSim();
    explicit VehicleDrivetrainSim(VehicleDynamicsConfig config);

    void reset() noexcept;
    void setConfig(VehicleDynamicsConfig config);
    [[nodiscard]] const VehicleDynamicsConfig& config() const noexcept { return config_; }
    [[nodiscard]] VehicleDynamicsConfig& config() noexcept { return config_; }

    void setAutoShift(bool enabled) noexcept { config_.powertrain.autoShift = enabled; }
    void setDiffType(DiffType type) noexcept;

    /// Sets current gear manually (-1 = Reverse, 0 = Neutral, 1..6 = Forward). Returns false if out of range.
    bool setGear(int gear) noexcept;
    bool shiftUp() noexcept;
    bool shiftDown() noexcept;

    /// Overrides automatic clutch engagement with a fixed value in [0..1].
    void setClutchOverride(float engagement) noexcept;
    void clearClutchOverride() noexcept { clutchOverride_ = -1.0f; }

    /// Sets surface grip multiplier for a single wheel [FL=0, FR=1, RL=2, RR=3] or all four wheels.
    void setWheelGrip(std::size_t wheelIndex, float grip) noexcept;
    void setWheelGrips(const std::array<float, kVehicleWheelCount>& grips) noexcept;

    [[nodiscard]] float sampleEngineTorque(float rpm) const noexcept {
        return config_.powertrain.sampleEngineTorque(rpm);
    }
    [[nodiscard]] int currentGear() const noexcept { return currentGear_; }
    [[nodiscard]] float engineRpm() const noexcept { return engineRpm_; }
    [[nodiscard]] const std::array<float, kVehicleWheelCount>& wheelAngularVelocities() const noexcept {
        return wheelAngularVel_;
    }

    /// Advances powertrain, differential, wheel spin, tire forces, and suspension state by dt seconds.
    [[nodiscard]] VehicleTelemetry step(float throttle, float brake, float handbrake, float steerDeg,
                                        const std::array<float, kVehicleWheelCount>& wheelLoadsN,
                                        const std::array<bool, kVehicleWheelCount>& wheelContacts,
                                        float forwardSpeedMs, float lateralSpeedMs, float dt) noexcept;

    /// Convenience overload accepting raw 4-element C-style arrays.
    [[nodiscard]] VehicleTelemetry step(float throttle, float brake, float handbrake, float steerDeg,
                                        const float (&wheelLoadsN)[kVehicleWheelCount],
                                        const bool (&wheelContacts)[kVehicleWheelCount],
                                        float forwardSpeedMs, float lateralSpeedMs, float dt) noexcept;

private:
    void updateAutoShift(float drivetrainRpm) noexcept;
    void beginShift(int targetGear) noexcept;

    VehicleDynamicsConfig config_{};
    std::array<float, kVehicleWheelCount> wheelAngularVel_{0.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, kVehicleWheelCount> suspensionCompression_{0.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, kVehicleWheelCount> suspensionCompressionVel_{0.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, kVehicleWheelCount> wheelGrips_{1.0f, 1.0f, 1.0f, 1.0f};
    float engineRpm_ = 850.0f;
    int currentGear_ = 1;
    float shiftTimer_ = 0.0f;
    float shiftCooldown_ = 0.0f;
    float downshiftHoldTimer_ = 0.0f;
    float clutchEngagement_ = 0.0f;
    float clutchOverride_ = -1.0f;
    VehicleTelemetry lastTelemetry_{};
};

} // namespace lumora
