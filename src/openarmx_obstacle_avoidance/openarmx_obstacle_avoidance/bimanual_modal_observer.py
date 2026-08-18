#!/usr/bin/env python3

import os

# Tiny matrices are faster and deterministic without a multi-threaded BLAS pool.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from collections import deque
import json
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, Float64MultiArray, String

from .numpy_residual_gru import NumpyResidualGRUEnsemble

try:
    from scipy.linalg import expm
except ImportError:  # pragma: no cover - reported clearly at runtime
    expm = None


DEFAULT_MODAL_JOINTS = (
    "openarmx_left_joint1",
    "openarmx_left_joint2",
    "openarmx_left_joint4",
    "openarmx_right_joint1",
    "openarmx_right_joint2",
    "openarmx_right_joint4",
)


def discretize_modal_model(
    rate_hz: float,
    frequencies_hz: np.ndarray,
    damping_ratios: np.ndarray,
    coupling: np.ndarray,
    disturbance_cutoff_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    if expm is None:
        raise RuntimeError("scipy.linalg.expm is required by the modal observer")
    coupling = np.asarray(coupling, dtype=float)
    if coupling.ndim != 2 or coupling.shape[0] != 2:
        raise ValueError("modal coupling must have shape (2, input_count)")
    input_count = int(coupling.shape[1])
    a_continuous = np.zeros((6, 6), dtype=float)
    b_continuous = np.zeros((6, input_count), dtype=float)
    for mode in range(2):
        omega = 2.0 * math.pi * float(frequencies_hz[mode])
        index = 2 * mode
        a_continuous[index, index + 1] = 1.0
        a_continuous[index + 1, index] = -(omega * omega)
        a_continuous[index + 1, index + 1] = (
            -2.0 * float(damping_ratios[mode]) * omega
        )
        b_continuous[index + 1, :] = coupling[mode, :]
    disturbance_decay = -2.0 * math.pi * disturbance_cutoff_hz
    a_continuous[4, 4] = disturbance_decay
    a_continuous[5, 5] = disturbance_decay

    augmented = np.zeros((6 + input_count, 6 + input_count), dtype=float)
    augmented[:6, :6] = a_continuous
    augmented[:6, 6 : 6 + input_count] = b_continuous
    discrete = expm(augmented / rate_hz)
    return discrete[:6, :6], discrete[:6, 6 : 6 + input_count]


class ModalKalmanFilter:
    def __init__(
        self,
        a_discrete: np.ndarray,
        b_discrete: np.ndarray,
        process_noise: float,
        disturbance_process_noise: float,
        measurement_noise: float,
    ) -> None:
        self.a = np.asarray(a_discrete, dtype=float)
        self.b = np.asarray(b_discrete, dtype=float)
        self.c = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        self.c = np.pad(self.c, ((0, 0), (0, 2)))
        self.c[0, 4] = 1.0
        self.c[1, 5] = 1.0
        self.q = np.diag(
            [
                0.02 * process_noise,
                process_noise,
                0.02 * process_noise,
                process_noise,
                disturbance_process_noise,
                disturbance_process_noise,
            ]
        )
        self.r = np.eye(2, dtype=float) * measurement_noise
        self.state = np.zeros(6, dtype=float)
        self.covariance = np.eye(6, dtype=float) * 0.1
        self.innovation = np.zeros(2, dtype=float)
        self.normalized_innovation_squared = 0.0

    def reset(self) -> None:
        self.state.fill(0.0)
        self.covariance = np.eye(6, dtype=float) * 0.1
        self.innovation.fill(0.0)
        self.normalized_innovation_squared = 0.0

    def update(
        self,
        acceleration: np.ndarray,
        modal_rates: np.ndarray,
        prediction_residual: np.ndarray | None = None,
    ) -> np.ndarray:
        predicted_state = self.a @ self.state + self.b @ acceleration
        if prediction_residual is not None:
            predicted_state += np.asarray(prediction_residual, dtype=float)
        predicted_covariance = self.a @ self.covariance @ self.a.T + self.q
        innovation = modal_rates - self.c @ predicted_state
        innovation_covariance = (
            self.c @ predicted_covariance @ self.c.T + self.r
        )
        self.innovation = innovation.copy()
        self.normalized_innovation_squared = float(
            innovation.T
            @ np.linalg.solve(innovation_covariance, innovation)
        )
        gain = np.linalg.solve(
            innovation_covariance,
            self.c @ predicted_covariance,
        ).T
        self.state = predicted_state + gain @ innovation
        identity = np.eye(6, dtype=float)
        joseph_left = identity - gain @ self.c
        self.covariance = (
            joseph_left @ predicted_covariance @ joseph_left.T
            + gain @ self.r @ gain.T
        )
        return self.state.copy()


class BimanualModalObserver(Node):
    def __init__(self) -> None:
        super().__init__("bimanual_modal_observer")
        self.declare_parameter("imu_topic", "/camera/imu")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("rate_hz", 100.0)
        self.declare_parameter("calibration_duration", 2.0)
        self.declare_parameter("sensor_timeout", 0.15)
        self.declare_parameter("joint_acceleration_window", 0.12)
        self.declare_parameter("joint_acceleration_alpha", 0.25)
        self.declare_parameter("max_joint_acceleration", 15.0)
        self.declare_parameter("modal_frequencies_hz", [1.6846, 2.4155])
        self.declare_parameter("modal_damping_ratios", [0.07918, 0.05082])
        self.declare_parameter("modal_joint_names", list(DEFAULT_MODAL_JOINTS))
        self.declare_parameter("disturbance_cutoff_hz", 1.0)
        self.declare_parameter(
            "modal_coupling_flat",
            [
                0.23935252,
                0.00220909,
                -0.09528608,
                -0.23503855,
                -0.00061910,
                -0.09961852,
                0.00217554,
                0.18233809,
                -0.00085226,
                0.01564604,
                0.17299390,
                0.00756648,
            ],
        )
        self.declare_parameter("process_noise", 3e-5)
        self.declare_parameter("disturbance_process_noise", 1e-4)
        self.declare_parameter("measurement_noise", 1e-4)
        self.declare_parameter(
            "state_topic", "/openarmx/antisway/modal_state"
        )
        self.declare_parameter(
            "diagnostics_topic", "/openarmx/antisway/observer_diagnostics"
        )
        self.declare_parameter(
            "valid_topic", "/openarmx/antisway/observer_valid"
        )
        self.declare_parameter(
            "status_topic", "/openarmx/antisway/observer_status"
        )
        self.declare_parameter("residual_model_enabled", False)
        self.declare_parameter("residual_model_monitor_only", True)
        self.declare_parameter("residual_model_backend", "numpy")
        self.declare_parameter("residual_model_device", "cpu")
        self.declare_parameter("residual_model_path", "")
        self.declare_parameter("residual_model_correction_gain", 1.0)
        self.declare_parameter(
            "residual_prediction_topic",
            "/openarmx/antisway/residual_predicted_state",
        )
        self.declare_parameter(
            "residual_diagnostics_topic",
            "/openarmx/antisway/residual_diagnostics",
        )

        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.joint_states_topic = str(
            self.get_parameter("joint_states_topic").value
        )
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.calibration_duration = max(
            0.2, float(self.get_parameter("calibration_duration").value)
        )
        self.sensor_timeout = max(
            0.02, float(self.get_parameter("sensor_timeout").value)
        )
        self.acceleration_window = max(
            0.06,
            float(self.get_parameter("joint_acceleration_window").value),
        )
        self.acceleration_alpha = float(
            np.clip(
                float(self.get_parameter("joint_acceleration_alpha").value),
                0.0,
                1.0,
            )
        )
        self.max_joint_acceleration = max(
            0.1, float(self.get_parameter("max_joint_acceleration").value)
        )
        frequencies = np.asarray(
            self.get_parameter("modal_frequencies_hz").value,
            dtype=float,
        )
        damping = np.asarray(
            self.get_parameter("modal_damping_ratios").value,
            dtype=float,
        )
        self.modal_joint_names = [
            str(name) for name in self.get_parameter("modal_joint_names").value
        ]
        if not self.modal_joint_names:
            raise RuntimeError("modal_joint_names must not be empty")
        coupling_flat = np.asarray(
            self.get_parameter("modal_coupling_flat").value,
            dtype=float,
        )
        expected_coupling_values = 2 * len(self.modal_joint_names)
        if coupling_flat.size != expected_coupling_values:
            raise RuntimeError(
                "modal_coupling_flat must contain "
                f"{expected_coupling_values} values for "
                f"{len(self.modal_joint_names)} modal joints"
            )
        coupling = coupling_flat.reshape(2, len(self.modal_joint_names))
        disturbance_cutoff_hz = float(
            self.get_parameter("disturbance_cutoff_hz").value
        )
        if frequencies.shape != (2,) or damping.shape != (2,):
            raise RuntimeError("modal frequencies and damping must each have 2 values")
        if self.rate_hz < 20.0 or self.rate_hz > 500.0:
            raise RuntimeError("rate_hz must be in [20, 500]")
        if np.any(frequencies <= 0.0) or np.any(damping <= 0.0):
            raise RuntimeError("modal frequencies and damping must be positive")
        if disturbance_cutoff_hz <= 0.0:
            raise RuntimeError("disturbance_cutoff_hz must be positive")
        self.frequencies = frequencies
        self.damping = damping
        self.coupling = coupling
        self.disturbance_cutoff_hz = disturbance_cutoff_hz
        a_discrete, b_discrete = discretize_modal_model(
            self.rate_hz,
            frequencies,
            damping,
            coupling,
            disturbance_cutoff_hz,
        )
        self.filter = ModalKalmanFilter(
            a_discrete,
            b_discrete,
            max(1e-9, float(self.get_parameter("process_noise").value)),
            max(
                1e-9,
                float(
                    self.get_parameter("disturbance_process_noise").value
                ),
            ),
            max(1e-9, float(self.get_parameter("measurement_noise").value)),
        )

        self.residual_model_enabled = bool(
            self.get_parameter("residual_model_enabled").value
        )
        self.residual_model_monitor_only = bool(
            self.get_parameter("residual_model_monitor_only").value
        )
        self.residual_model_correction_gain = float(
            np.clip(
                float(
                    self.get_parameter("residual_model_correction_gain").value
                ),
                0.0,
                1.0,
            )
        )
        self.residual_model = None
        self.residual_model_backend = str(
            self.get_parameter("residual_model_backend").value
        ).strip().lower()
        self.residual_model_device = str(
            self.get_parameter("residual_model_device").value
        ).strip().lower()
        self.residual_model_a = None
        self.residual_model_b = None
        self.residual_feature_history = None
        self.residual_model_period = None
        self.last_residual_model_time = None
        self.pending_prediction_residual = None
        self.latest_residual_mean = None
        self.latest_residual_std = None
        if self.residual_model_enabled:
            residual_model_path = str(
                self.get_parameter("residual_model_path").value
            ).strip()
            if not residual_model_path:
                raise RuntimeError(
                    "residual_model_path is required when residual_model_enabled=true"
                )
            if self.residual_model_backend == "numpy":
                self.residual_model = NumpyResidualGRUEnsemble(residual_model_path)
            elif self.residual_model_backend == "torchscript":
                from .torchscript_residual_gru import TorchScriptResidualGRUEnsemble

                self.residual_model = TorchScriptResidualGRUEnsemble(
                    residual_model_path,
                    device=self.residual_model_device,
                )
            else:
                raise RuntimeError(
                    "residual_model_backend must be 'numpy' or 'torchscript'"
                )
            if self.residual_model.input_size != 24 or self.residual_model.output_size != 6:
                raise RuntimeError(
                    "Residual model must use the trained 24-input/6-output contract"
                )
            if (
                not self.residual_model_monitor_only
                and abs(self.rate_hz - self.residual_model.rate_hz) > 1e-6
            ):
                raise RuntimeError(
                    "Active residual correction requires observer rate_hz to equal "
                    f"the model rate ({self.residual_model.rate_hz:.1f} Hz)"
                )
            self.residual_model_a, self.residual_model_b = discretize_modal_model(
                self.residual_model.rate_hz,
                frequencies,
                damping,
                coupling,
                disturbance_cutoff_hz,
            )
            self.residual_feature_history = deque(
                maxlen=self.residual_model.history_samples
            )
            self.residual_model_period = 1.0 / self.residual_model.rate_hz

        history_length = max(20, int(math.ceil(self.rate_hz * 0.5)))
        self.joint_history: deque[tuple[float, np.ndarray]] = deque(
            maxlen=history_length
        )
        self.filtered_acceleration = np.zeros(
            len(self.modal_joint_names),
            dtype=float,
        )
        self.filtered_velocity = np.zeros(
            len(self.modal_joint_names), dtype=float
        )
        self.latest_joint_position = None
        self.latest_gyro = None
        self.latest_imu_monotonic = None
        self.latest_joint_monotonic = None
        self.gyro_bias_sum = np.zeros(2, dtype=float)
        self.gyro_bias = np.zeros(2, dtype=float)
        self.gyro_bias_count = 0
        self.calibration_started = None
        self.calibrated = False
        self.last_valid = False
        self.last_status_time = 0.0

        self.create_subscription(
            Imu,
            self.imu_topic,
            self._imu_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            JointState,
            self.joint_states_topic,
            self._joint_cb,
            qos_profile_sensor_data,
        )
        self.state_pub = self.create_publisher(
            Float64MultiArray,
            str(self.get_parameter("state_topic").value),
            10,
        )
        self.diagnostics_pub = self.create_publisher(
            Float64MultiArray,
            str(self.get_parameter("diagnostics_topic").value),
            10,
        )
        self.valid_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("valid_topic").value),
            10,
        )
        self.status_pub = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            10,
        )
        self.residual_prediction_pub = self.create_publisher(
            Float64MultiArray,
            str(self.get_parameter("residual_prediction_topic").value),
            10,
        )
        self.residual_diagnostics_pub = self.create_publisher(
            Float64MultiArray,
            str(self.get_parameter("residual_diagnostics_topic").value),
            10,
        )
        self.timer = self.create_timer(1.0 / self.rate_hz, self._timer_cb)
        self.get_logger().info(
            "read-only modal observer: "
            f"rate={self.rate_hz:.1f}Hz, imu={self.imu_topic}, "
            f"joints={self.modal_joint_names}, "
            f"slow_cutoff={self.disturbance_cutoff_hz:.2f}Hz"
        )
        if self.residual_model is not None:
            self.get_logger().info(
                "residual GRU connected: "
                f"ensemble={self.residual_model.ensemble_size}, "
                f"history={self.residual_model.history_samples} samples, "
                f"model_rate={self.residual_model.rate_hz:.1f}Hz, "
                f"backend={self.residual_model_backend}, "
                f"device={self.residual_model_device}, "
                f"monitor_only={self.residual_model_monitor_only}"
            )
        self.get_logger().info(
            f"Keep the platform still for {self.calibration_duration:.1f}s "
            "while gyro bias is calibrated."
        )

    def _imu_cb(self, msg: Imu) -> None:
        gyro = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.z],
            dtype=float,
        )
        if not np.all(np.isfinite(gyro)):
            return
        now = time.monotonic()
        self.latest_imu_monotonic = now
        if not self.calibrated:
            if self.calibration_started is None:
                self.calibration_started = now
            self.gyro_bias_sum += gyro
            self.gyro_bias_count += 1
            if (
                now - self.calibration_started >= self.calibration_duration
                and self.gyro_bias_count >= 20
            ):
                self.gyro_bias = self.gyro_bias_sum / self.gyro_bias_count
                self.calibrated = True
                self.filter.reset()
                self.get_logger().info(
                    "gyro calibration complete: "
                    f"bias_x={self.gyro_bias[0]:.6f}, "
                    f"bias_z={self.gyro_bias[1]:.6f} rad/s"
                )
        self.latest_gyro = gyro

    def _joint_cb(self, msg: JointState) -> None:
        indices = {name: index for index, name in enumerate(msg.name)}
        if not all(name in indices for name in self.modal_joint_names):
            return
        positions = np.asarray(
            [msg.position[indices[name]] for name in self.modal_joint_names],
            dtype=float,
        )
        if not np.all(np.isfinite(positions)):
            return
        now = time.monotonic()
        self.joint_history.append((now, positions))
        self.latest_joint_position = positions
        self.latest_joint_monotonic = now

    def _estimate_joint_kinematics(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.joint_history) < 7:
            return (
                self.filtered_velocity.copy(),
                self.filtered_acceleration.copy(),
            )
        newest_time = self.joint_history[-1][0]
        samples = [
            sample
            for sample in self.joint_history
            if sample[0] >= newest_time - self.acceleration_window
        ]
        if len(samples) < 7:
            return (
                self.filtered_velocity.copy(),
                self.filtered_acceleration.copy(),
            )
        times = np.asarray([sample[0] - newest_time for sample in samples])
        positions = np.asarray([sample[1] for sample in samples])
        design = np.column_stack(
            (times * times * times, times * times, times, np.ones_like(times))
        )
        coefficients = np.linalg.lstsq(design, positions, rcond=None)[0]
        velocity = coefficients[2, :]
        acceleration = 2.0 * coefficients[1, :]
        acceleration = np.clip(
            acceleration,
            -self.max_joint_acceleration,
            self.max_joint_acceleration,
        )
        self.filtered_acceleration += self.acceleration_alpha * (
            acceleration - self.filtered_acceleration
        )
        self.filtered_velocity += self.acceleration_alpha * (
            velocity - self.filtered_velocity
        )
        return (
            self.filtered_velocity.copy(),
            self.filtered_acceleration.copy(),
        )

    def _run_residual_model(
        self,
        now: float,
        state: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
    ) -> None:
        if self.residual_model is None:
            return
        if (
            self.last_residual_model_time is not None
            and now - self.last_residual_model_time
            < self.residual_model_period * 0.8
        ):
            return
        self.last_residual_model_time = now
        feature = np.concatenate(
            (state, self.latest_joint_position, velocity, acceleration)
        )
        self.residual_feature_history.append(feature)
        if len(self.residual_feature_history) < self.residual_model.history_samples:
            return
        sequence = np.asarray(self.residual_feature_history, dtype=float)
        residual_mean, residual_std = self.residual_model.predict(sequence)
        physics_next = self.residual_model_a @ state
        physics_next += self.residual_model_b @ acceleration
        hybrid_next = physics_next + residual_mean
        self.latest_residual_mean = residual_mean
        self.latest_residual_std = residual_std
        self.pending_prediction_residual = (
            self.residual_model_correction_gain * residual_mean
        )
        self.residual_prediction_pub.publish(
            Float64MultiArray(data=hybrid_next.tolist())
        )
        # Layout: residual mean[6], ensemble std[6], physics next[6], hybrid next[6].
        self.residual_diagnostics_pub.publish(
            Float64MultiArray(
                data=[
                    *residual_mean.tolist(),
                    *residual_std.tolist(),
                    *physics_next.tolist(),
                    *hybrid_next.tolist(),
                ]
            )
        )

    def _sensors_are_fresh(self, now: float) -> bool:
        return bool(
            self.calibrated
            and self.latest_gyro is not None
            and self.latest_imu_monotonic is not None
            and self.latest_joint_monotonic is not None
            and now - self.latest_imu_monotonic <= self.sensor_timeout
            and now - self.latest_joint_monotonic <= self.sensor_timeout
            and len(self.joint_history) >= 7
        )

    def _timer_cb(self) -> None:
        now = time.monotonic()
        valid = self._sensors_are_fresh(now)
        self.valid_pub.publish(Bool(data=valid))
        if not valid:
            if self.last_valid:
                self.get_logger().warning(
                    "Modal observer input became invalid; state publication paused."
                )
            self.last_valid = False
            self._publish_status(
                now,
                valid,
                np.zeros(len(self.modal_joint_names), dtype=float),
            )
            return

        velocity, acceleration = self._estimate_joint_kinematics()
        modal_rates = self.latest_gyro - self.gyro_bias
        prediction_residual = None
        if (
            self.residual_model is not None
            and not self.residual_model_monitor_only
            and self.pending_prediction_residual is not None
        ):
            prediction_residual = self.pending_prediction_residual
            self.pending_prediction_residual = None
        state = self.filter.update(
            acceleration, modal_rates, prediction_residual
        )
        self._run_residual_model(now, state, velocity, acceleration)
        roll_omega = 2.0 * math.pi * self.frequencies[0]
        yaw_omega = 2.0 * math.pi * self.frequencies[1]
        roll_energy = 0.5 * (
            state[1] * state[1] + (roll_omega * state[0]) ** 2
        )
        yaw_energy = 0.5 * (
            state[3] * state[3] + (yaw_omega * state[2]) ** 2
        )
        self.state_pub.publish(Float64MultiArray(data=state.tolist()))
        self.diagnostics_pub.publish(
            Float64MultiArray(
                data=[
                    *acceleration.tolist(),
                    float(modal_rates[0]),
                    float(modal_rates[1]),
                    float(roll_energy),
                    float(yaw_energy),
                    *self.filter.innovation.tolist(),
                    self.filter.normalized_innovation_squared,
                ]
            )
        )
        self.last_valid = True
        self._publish_status(now, valid, acceleration, state, roll_energy, yaw_energy)

    def _publish_status(
        self,
        now: float,
        valid: bool,
        acceleration: np.ndarray,
        state: np.ndarray | None = None,
        roll_energy: float = 0.0,
        yaw_energy: float = 0.0,
    ) -> None:
        if now - self.last_status_time < 1.0:
            return
        self.last_status_time = now
        payload = {
            "valid": valid,
            "calibrated": self.calibrated,
            "gyro_bias_samples": self.gyro_bias_count,
            "joint_acceleration": acceleration.tolist(),
            "modal_joint_names": self.modal_joint_names,
            "state": state.tolist() if state is not None else None,
            "state_order": [
                "roll_angle",
                "roll_rate",
                "yaw_angle",
                "yaw_rate",
                "slow_gyro_x",
                "slow_gyro_z",
            ],
            "roll_energy": roll_energy,
            "yaw_energy": yaw_energy,
            "innovation": self.filter.innovation.tolist() if valid else None,
            "nis": self.filter.normalized_innovation_squared if valid else None,
            "residual_model": {
                "enabled": self.residual_model is not None,
                "backend": self.residual_model_backend,
                "device": self.residual_model_device,
                "monitor_only": self.residual_model_monitor_only,
                "history_ready": bool(
                    self.residual_feature_history is not None
                    and len(self.residual_feature_history)
                    == self.residual_feature_history.maxlen
                ),
                "mean": (
                    self.latest_residual_mean.tolist()
                    if self.latest_residual_mean is not None
                    else None
                ),
                "std": (
                    self.latest_residual_std.tolist()
                    if self.latest_residual_std is not None
                    else None
                ),
            },
        }
        self.status_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BimanualModalObserver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
