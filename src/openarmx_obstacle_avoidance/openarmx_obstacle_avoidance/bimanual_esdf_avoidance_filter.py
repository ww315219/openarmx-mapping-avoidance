from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass

import numpy as np
import pinocchio as pin
import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from nvblox_msgs.srv import EsdfAndGradients
from openarmx_safety_msgs.msg import ProtectedCableArray, SafetyStatus
from rcl_interfaces.srv import GetParameters
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, Float64MultiArray, String
from visualization_msgs.msg import Marker, MarkerArray

try:
    from scipy.optimize import minimize
    from scipy import sparse
    from scipy.signal import iirnotch, lfilter, lfilter_zi
except Exception:  # noqa: BLE001
    minimize = None
    sparse = None
    iirnotch = None
    lfilter = None
    lfilter_zi = None

try:
    import osqp
except Exception:  # noqa: BLE001
    osqp = None

from openarmx_obstacle_avoidance.right_arm_esdf_avoidance_filter import (
    CollisionSphere,
    EsdfGrid,
    SphereState,
    _point_from_array,
    _skew,
    _vector_from_array,
)
from openarmx_obstacle_avoidance.bimanual_modal_observer import discretize_modal_model


LEFT_JOINT_NAMES = [f"openarmx_left_joint{i}" for i in range(1, 8)]
RIGHT_JOINT_NAMES = [f"openarmx_right_joint{i}" for i in range(1, 8)]
BIMANUAL_JOINT_NAMES = LEFT_JOINT_NAMES + RIGHT_JOINT_NAMES


@dataclass
class CableCapsule:
    start: np.ndarray
    end: np.ndarray
    radius: float
    position_std: float = 0.0
    radius_std: float = 0.0
    confidence: float = 1.0
    cable_id: str = ""


@dataclass
class PredictiveAntiswayResult:
    acceleration: np.ndarray
    sequence: np.ndarray
    solve_ms: float
    iterations: int
    status: str


@dataclass
class InputShaperResult:
    positions: np.ndarray
    modal_input: np.ndarray
    modal_output: np.ndarray
    correction: np.ndarray


@dataclass
class TaskPreservingAntiswayResult:
    acceleration_correction: np.ndarray
    predicted_modal_input: np.ndarray
    target_modal_input: np.ndarray
    task_residual: float
    nullity: int
    solve_ms: float
    iterations: int
    status: str


class JerkLimitedCommandGovernor:
    """Causal joint command governor with bounded velocity, acceleration, and jerk."""

    def __init__(
        self,
        *,
        joint_count: int,
        max_velocity: float,
        max_acceleration: float,
        max_deceleration: float,
        max_jerk: float,
        position_gain: float,
        position_tolerance: float,
    ) -> None:
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)
        self.max_deceleration = float(max_deceleration)
        self.max_jerk = float(max_jerk)
        self.position_gain = float(position_gain)
        self.position_tolerance = float(position_tolerance)
        if joint_count <= 0:
            raise ValueError("joint_count must be positive")
        if min(
            self.max_velocity,
            self.max_acceleration,
            self.max_deceleration,
            self.max_jerk,
            self.position_gain,
        ) <= 0.0:
            raise ValueError("command governor limits and gain must be positive")
        self.velocity = np.zeros(joint_count, dtype=float)
        self.acceleration = np.zeros(joint_count, dtype=float)
        self.active = False

    def reset(self) -> None:
        self.velocity.fill(0.0)
        self.acceleration.fill(0.0)
        self.active = False

    def reset_indices(self, indices: slice | np.ndarray) -> None:
        self.velocity[indices] = 0.0
        self.acceleration[indices] = 0.0
        self.active = bool(
            np.any(np.abs(self.velocity) > self.position_tolerance)
            or np.any(np.abs(self.acceleration) > self.position_tolerance)
        )

    def step(
        self,
        current_position: np.ndarray,
        target_position: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        current = np.asarray(current_position, dtype=float)
        target = np.asarray(target_position, dtype=float)
        if current.shape != self.velocity.shape or target.shape != self.velocity.shape:
            raise ValueError("command governor input shape mismatch")
        dt = max(float(dt), 1e-6)
        error = target - current

        # A stopping-speed envelope prevents large overshoot without snapping the
        # command or its velocity to the target when the target changes direction.
        stopping_speed = np.sqrt(
            np.maximum(0.0, 2.0 * self.max_deceleration * np.abs(error))
        )
        desired_speed = np.minimum(
            self.max_velocity,
            np.minimum(self.position_gain * np.abs(error), stopping_speed),
        )
        desired_velocity = np.sign(error) * desired_speed
        slowing = (
            self.velocity * desired_velocity <= 0.0
        ) | (np.abs(desired_velocity) < np.abs(self.velocity))
        acceleration_limit = np.where(
            slowing,
            self.max_deceleration,
            self.max_acceleration,
        )
        # Track the velocity envelope with a damped finite-bandwidth loop.
        # One-cycle velocity tracking ((v_des-v)/dt) becomes bang-bang under a
        # jerk limit and can sustain a position limit cycle near a static target.
        desired_acceleration = np.clip(
            2.0 * self.position_gain * (desired_velocity - self.velocity),
            -acceleration_limit,
            acceleration_limit,
        )
        jerk_step = self.max_jerk * dt
        next_acceleration = np.clip(
            desired_acceleration,
            self.acceleration - jerk_step,
            self.acceleration + jerk_step,
        )
        absolute_acceleration_limit = max(
            self.max_acceleration,
            self.max_deceleration,
        )
        next_acceleration = np.clip(
            next_acceleration,
            -absolute_acceleration_limit,
            absolute_acceleration_limit,
        )
        next_velocity = np.clip(
            self.velocity + next_acceleration * dt,
            -self.max_velocity,
            self.max_velocity,
        )
        next_position = current + next_velocity * dt

        settled = (
            np.abs(error) <= self.position_tolerance
        ) & (
            np.abs(next_velocity) <= self.position_tolerance
        ) & (
            np.abs(next_acceleration) <= self.position_tolerance
        )
        next_position[settled] = target[settled]
        next_velocity[settled] = 0.0
        next_acceleration[settled] = 0.0
        self.velocity = next_velocity
        self.acceleration = next_acceleration
        self.active = bool(np.any(~settled))
        return next_position


class TaskPreservingBimanualAntiswayQp:
    """Small modal QP with soft bimanual TCP task preservation."""

    def __init__(
        self,
        *,
        coupling: np.ndarray,
        modal_weights: np.ndarray,
        task_translation_weight: float,
        task_rotation_weight: float,
        correction_weight: float,
        max_joint_acceleration: float,
        nullspace_rcond: float,
        max_iterations: int,
        eps_abs: float,
        eps_rel: float,
        time_limit_s: float,
    ) -> None:
        self.coupling = np.asarray(coupling, dtype=float)
        self.modal_weights = np.asarray(modal_weights, dtype=float).reshape(2)
        self.task_translation_weight = float(task_translation_weight)
        self.task_rotation_weight = float(task_rotation_weight)
        self.correction_weight = float(correction_weight)
        self.max_joint_acceleration = float(max_joint_acceleration)
        self.nullspace_rcond = float(nullspace_rcond)
        self.max_iterations = int(max_iterations)
        self.eps_abs = float(eps_abs)
        self.eps_rel = float(eps_rel)
        self.time_limit_s = float(time_limit_s)
        if self.coupling.ndim != 2 or self.coupling.shape[0] != 2:
            raise ValueError("bimanual modal coupling must have shape (2, joint_count)")
        if np.any(self.modal_weights < 0.0):
            raise ValueError("modal weights must be non-negative")
        if self.task_translation_weight < 0.0 or self.task_rotation_weight < 0.0:
            raise ValueError("task weights must be non-negative")
        if self.correction_weight <= 0.0:
            raise ValueError("correction weight must be positive")
        if self.max_joint_acceleration <= 0.0:
            raise ValueError("maximum joint acceleration must be positive")

    def solve(
        self,
        *,
        task_jacobian: np.ndarray,
        nominal_acceleration: np.ndarray,
        target_modal_input: np.ndarray,
    ) -> TaskPreservingAntiswayResult:
        started = time.perf_counter()
        task_jacobian = np.asarray(task_jacobian, dtype=float)
        nominal_acceleration = np.asarray(nominal_acceleration, dtype=float)
        target_modal_input = np.asarray(target_modal_input, dtype=float).reshape(2)
        joint_count = self.coupling.shape[1]
        if (
            task_jacobian.ndim != 2
            or task_jacobian.shape[1] != joint_count
            or nominal_acceleration.shape != (joint_count,)
            or not np.all(np.isfinite(task_jacobian))
            or not np.all(np.isfinite(nominal_acceleration))
            or not np.all(np.isfinite(target_modal_input))
        ):
            return self._empty_result(
                nominal_acceleration,
                target_modal_input,
                "invalid_input",
                started,
            )

        _, singular_values, vh = np.linalg.svd(task_jacobian, full_matrices=True)
        threshold = (
            self.nullspace_rcond * singular_values[0]
            if singular_values.size
            else self.nullspace_rcond
        )
        rank = int(np.sum(singular_values > threshold))
        nullity = int(joint_count - rank)
        modal_residual = target_modal_input - self.coupling @ nominal_acceleration
        modal_weight = np.diag(self.modal_weights)
        task_weights = np.tile(
            np.array(
                [
                    self.task_translation_weight,
                    self.task_translation_weight,
                    self.task_translation_weight,
                    self.task_rotation_weight,
                    self.task_rotation_weight,
                    self.task_rotation_weight,
                ],
                dtype=float,
            ),
            task_jacobian.shape[0] // 6,
        )
        task_weight = np.diag(task_weights)
        hessian = (
            self.coupling.T @ modal_weight @ self.coupling
            + task_jacobian.T @ task_weight @ task_jacobian
            + self.correction_weight * np.eye(joint_count)
        )
        gradient = -(self.coupling.T @ modal_weight @ modal_residual)
        acceleration_correction = np.zeros(joint_count, dtype=float)
        iterations = 0
        status = "numpy"
        solved = False

        if osqp is not None and sparse is not None:
            try:
                solver = osqp.OSQP()
                settings = {
                    "verbose": False,
                    "polishing": False,
                    "warm_starting": False,
                    "max_iter": self.max_iterations,
                    "eps_abs": self.eps_abs,
                    "eps_rel": self.eps_rel,
                }
                if self.time_limit_s > 0.0:
                    settings["time_limit"] = self.time_limit_s
                solver.setup(
                    P=sparse.csc_matrix(2.0 * hessian),
                    q=2.0 * gradient,
                    A=sparse.eye(joint_count, format="csc"),
                    l=np.full(joint_count, -self.max_joint_acceleration),
                    u=np.full(joint_count, self.max_joint_acceleration),
                    **settings,
                )
                solution = solver.solve()
                status = str(solution.info.status).lower()
                iterations = int(solution.info.iter)
                if status in ("solved", "solved inaccurate") and solution.x is not None:
                    acceleration_correction = np.asarray(solution.x, dtype=float)
                    solved = True
                else:
                    status = f"osqp_{status.replace(' ', '_')}"
            except Exception as exc:  # noqa: BLE001
                status = f"osqp_exception_{type(exc).__name__}"

        if not solved:
            try:
                acceleration_correction = np.linalg.solve(hessian, -gradient)
            except np.linalg.LinAlgError:
                acceleration_correction = np.linalg.lstsq(
                    hessian,
                    -gradient,
                    rcond=None,
                )[0]
            peak = float(np.max(np.abs(acceleration_correction)))
            if peak > self.max_joint_acceleration:
                acceleration_correction *= self.max_joint_acceleration / peak
            status = f"{status}_fallback"

        predicted_modal_input = self.coupling @ (
            nominal_acceleration + acceleration_correction
        )
        task_residual = float(
            np.linalg.norm(task_jacobian @ acceleration_correction)
        )
        return TaskPreservingAntiswayResult(
            acceleration_correction=acceleration_correction,
            predicted_modal_input=predicted_modal_input,
            target_modal_input=target_modal_input,
            task_residual=task_residual,
            nullity=nullity,
            solve_ms=(time.perf_counter() - started) * 1e3,
            iterations=iterations,
            status=status,
        )

    def _empty_result(
        self,
        nominal_acceleration: np.ndarray,
        target_modal_input: np.ndarray,
        status: str,
        started: float,
        *,
        nullity: int = 0,
    ) -> TaskPreservingAntiswayResult:
        correction = np.zeros(self.coupling.shape[1], dtype=float)
        nominal = np.asarray(nominal_acceleration, dtype=float)
        predicted = (
            self.coupling @ nominal
            if nominal.shape == (self.coupling.shape[1],)
            else np.zeros(2, dtype=float)
        )
        return TaskPreservingAntiswayResult(
            acceleration_correction=correction,
            predicted_modal_input=predicted,
            target_modal_input=np.asarray(target_modal_input, dtype=float).reshape(2),
            task_residual=0.0,
            nullity=nullity,
            solve_ms=(time.perf_counter() - started) * 1e3,
            iterations=0,
            status=status,
        )


class DualModalInputShaper:
    def __init__(
        self,
        *,
        rate_hz: float,
        frequencies_hz: np.ndarray,
        coupling: np.ndarray,
        quality_factors: np.ndarray,
        strengths: np.ndarray,
        max_correction: float,
        max_correction_rate: float,
    ) -> None:
        if iirnotch is None or lfilter is None or lfilter_zi is None:
            raise RuntimeError("scipy.signal is required by the dual-modal input shaper")
        self.rate_hz = float(rate_hz)
        self.dt = 1.0 / self.rate_hz
        self.frequencies_hz = np.asarray(frequencies_hz, dtype=float).reshape(2)
        self.coupling = np.asarray(coupling, dtype=float).reshape(2, 3)
        self.quality_factors = np.asarray(quality_factors, dtype=float).reshape(2)
        self.strengths = np.clip(
            np.asarray(strengths, dtype=float).reshape(2),
            0.0,
            1.0,
        )
        self.max_correction = float(max_correction)
        self.max_correction_rate = float(max_correction_rate)
        if np.any(self.frequencies_hz <= 0.0) or np.any(
            self.frequencies_hz >= 0.5 * self.rate_hz
        ):
            raise ValueError("input-shaper frequencies must lie below Nyquist")
        if np.any(self.quality_factors <= 0.0):
            raise ValueError("input-shaper quality factors must be positive")
        gram = self.coupling @ self.coupling.T
        if np.linalg.cond(gram) > 1e8:
            raise ValueError("modal coupling rows are not independently controllable")
        self.modal_to_joint = self.coupling.T @ np.linalg.inv(gram)
        self.coefficients = [
            iirnotch(frequency, quality, fs=self.rate_hz)
            for frequency, quality in zip(
                self.frequencies_hz,
                self.quality_factors,
            )
        ]
        self.filter_state: list[np.ndarray] | None = None
        self.previous_correction = np.zeros(3, dtype=float)

    def reset(self, positions: np.ndarray) -> None:
        positions = np.asarray(positions, dtype=float).reshape(3)
        modal_input = self.coupling @ positions
        self.filter_state = [
            lfilter_zi(b, a) * modal_input[index]
            for index, (b, a) in enumerate(self.coefficients)
        ]
        self.previous_correction.fill(0.0)

    def process(
        self,
        positions: np.ndarray,
        strength_scales: np.ndarray | None = None,
    ) -> InputShaperResult:
        positions = np.asarray(positions, dtype=float).reshape(3)
        modal_input = self.coupling @ positions
        if strength_scales is None:
            effective_strengths = self.strengths
        else:
            effective_strengths = self.strengths * np.clip(
                np.asarray(strength_scales, dtype=float).reshape(2),
                0.0,
                1.0,
            )
        if self.filter_state is None:
            self.reset(positions)
        modal_filtered = np.zeros(2, dtype=float)
        for index, (b, a) in enumerate(self.coefficients):
            output, state = lfilter(
                b,
                a,
                [modal_input[index]],
                zi=self.filter_state[index],
            )
            modal_filtered[index] = output[0]
            self.filter_state[index] = state
        modal_output = modal_input + effective_strengths * (
            modal_filtered - modal_input
        )
        correction = self.modal_to_joint @ (modal_output - modal_input)
        if self.max_correction > 0.0:
            correction = np.clip(
                correction,
                -self.max_correction,
                self.max_correction,
            )
        if self.max_correction_rate > 0.0:
            max_step = self.max_correction_rate * self.dt
            correction = np.clip(
                correction,
                self.previous_correction - max_step,
                self.previous_correction + max_step,
            )
        self.previous_correction = correction.copy()
        return InputShaperResult(
            positions=positions + correction,
            modal_input=modal_input,
            modal_output=modal_output,
            correction=correction,
        )


class PredictiveAntiswayMpc:
    STATE_SIZE = 10
    INPUT_SIZE = 3

    def __init__(
        self,
        *,
        rate_hz: float,
        horizon_steps: int,
        modal_a: np.ndarray,
        modal_b: np.ndarray,
        frequencies_hz: np.ndarray,
        roll_weight: float,
        yaw_weight: float,
        tracking_weight: float,
        velocity_weight: float,
        acceleration_weight: float,
        acceleration_change_weight: float,
        terminal_weight_scale: float,
        max_acceleration: float,
        max_velocity: float,
        max_iterations: int,
        eps_abs: float,
        eps_rel: float,
        time_limit_s: float,
    ) -> None:
        if osqp is None or sparse is None:
            raise RuntimeError("OSQP and scipy.sparse are required by predictive anti-sway MPC")
        self.rate_hz = float(rate_hz)
        self.dt = 1.0 / self.rate_hz
        self.horizon_steps = int(horizon_steps)
        self.max_acceleration = float(max_acceleration)
        self.max_velocity = float(max_velocity)
        self.acceleration_change_weight = float(acceleration_change_weight)

        state_matrix = np.zeros((self.STATE_SIZE, self.STATE_SIZE), dtype=float)
        input_matrix = np.zeros((self.STATE_SIZE, self.INPUT_SIZE), dtype=float)
        state_matrix[:4, :4] = np.asarray(modal_a, dtype=float)[:4, :4]
        input_matrix[:4, :] = np.asarray(modal_b, dtype=float)[:4, :]
        state_matrix[4:7, 4:7] = np.eye(3)
        state_matrix[4:7, 7:10] = self.dt * np.eye(3)
        input_matrix[4:7, :] = 0.5 * self.dt * self.dt * np.eye(3)
        state_matrix[7:10, 7:10] = np.eye(3)
        input_matrix[7:10, :] = self.dt * np.eye(3)
        self.state_matrix = state_matrix
        self.input_matrix = input_matrix

        self.phi, self.gamma = self._prediction_matrices()
        omega_roll = 2.0 * np.pi * float(frequencies_hz[0])
        omega_yaw = 2.0 * np.pi * float(frequencies_hz[1])
        state_weight = np.diag(
            [
                roll_weight * omega_roll * omega_roll,
                roll_weight,
                yaw_weight * omega_yaw * omega_yaw,
                yaw_weight,
                tracking_weight,
                tracking_weight,
                tracking_weight,
                velocity_weight,
                velocity_weight,
                velocity_weight,
            ]
        )
        state_weight_blocks = [state_weight.copy() for _ in range(self.horizon_steps)]
        state_weight_blocks[-1] *= terminal_weight_scale
        self.state_weight = sparse.block_diag(state_weight_blocks, format="csc")
        self.difference_matrix = self._difference_matrix()

        input_count = self.horizon_steps * self.INPUT_SIZE
        hessian = (
            self.gamma.T @ self.state_weight @ self.gamma
            + acceleration_weight * np.eye(input_count)
            + acceleration_change_weight
            * (self.difference_matrix.T @ self.difference_matrix)
        )
        objective_matrix = sparse.csc_matrix(2.0 * hessian)
        objective_matrix = sparse.triu(objective_matrix, format="csc")

        velocity_selector = sparse.lil_matrix(
            (self.horizon_steps * 3, self.horizon_steps * self.STATE_SIZE),
            dtype=float,
        )
        for step in range(self.horizon_steps):
            row = 3 * step
            column = self.STATE_SIZE * step + 7
            velocity_selector[row : row + 3, column : column + 3] = np.eye(3)
        velocity_selector = velocity_selector.tocsc()
        self.velocity_phi = np.asarray(velocity_selector @ self.phi, dtype=float)
        self.velocity_gamma = sparse.csc_matrix(velocity_selector @ self.gamma)
        constraint_matrix = sparse.vstack(
            (sparse.eye(input_count, format="csc"), self.velocity_gamma),
            format="csc",
        )
        lower = np.concatenate(
            [
                np.full(input_count, -self.max_acceleration, dtype=float),
                np.full(self.horizon_steps * 3, -self.max_velocity, dtype=float),
            ]
        )
        upper = -lower
        self.solver = osqp.OSQP()
        settings = {
            "verbose": False,
            "polishing": False,
            "warm_starting": True,
            "max_iter": int(max_iterations),
            "eps_abs": float(eps_abs),
            "eps_rel": float(eps_rel),
        }
        if time_limit_s > 0.0:
            settings["time_limit"] = float(time_limit_s)
        self.solver.setup(
            P=objective_matrix,
            q=np.zeros(input_count, dtype=float),
            A=constraint_matrix,
            l=lower,
            u=upper,
            **settings,
        )
        self.previous_solution = np.zeros(input_count, dtype=float)

    def _prediction_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        state_count = self.horizon_steps * self.STATE_SIZE
        input_count = self.horizon_steps * self.INPUT_SIZE
        phi = np.zeros((state_count, self.STATE_SIZE), dtype=float)
        gamma = np.zeros((state_count, input_count), dtype=float)
        for step in range(self.horizon_steps):
            row = step * self.STATE_SIZE
            phi[row : row + self.STATE_SIZE, :] = np.linalg.matrix_power(
                self.state_matrix,
                step + 1,
            )
            for input_step in range(step + 1):
                column = input_step * self.INPUT_SIZE
                gamma[
                    row : row + self.STATE_SIZE,
                    column : column + self.INPUT_SIZE,
                ] = (
                    np.linalg.matrix_power(self.state_matrix, step - input_step)
                    @ self.input_matrix
                )
        return phi, gamma

    def _difference_matrix(self) -> np.ndarray:
        size = self.horizon_steps * self.INPUT_SIZE
        matrix = np.eye(size, dtype=float)
        for step in range(1, self.horizon_steps):
            row = step * self.INPUT_SIZE
            previous = (step - 1) * self.INPUT_SIZE
            matrix[row : row + self.INPUT_SIZE, previous : previous + self.INPUT_SIZE] = (
                -np.eye(self.INPUT_SIZE)
            )
        return matrix

    def solve(
        self,
        state: np.ndarray,
        previous_acceleration: np.ndarray,
        state_reference: np.ndarray | None = None,
    ) -> PredictiveAntiswayResult | None:
        state = np.asarray(state, dtype=float).reshape(self.STATE_SIZE)
        previous_acceleration = np.asarray(previous_acceleration, dtype=float).reshape(3)
        if state_reference is None:
            state_reference = np.zeros(
                self.horizon_steps * self.STATE_SIZE,
                dtype=float,
            )
        else:
            state_reference = np.asarray(state_reference, dtype=float).reshape(
                self.horizon_steps * self.STATE_SIZE
            )
        free_state_error = self.phi @ state - state_reference
        linear_term = 2.0 * self.gamma.T @ self.state_weight @ free_state_error
        difference_target = np.zeros(self.horizon_steps * self.INPUT_SIZE, dtype=float)
        difference_target[:3] = np.clip(
            previous_acceleration,
            -self.max_acceleration,
            self.max_acceleration,
        )
        linear_term -= (
            2.0
            * self.acceleration_change_weight
            * self.difference_matrix.T
            @ difference_target
        )
        free_velocity = self.velocity_phi @ state
        input_count = self.horizon_steps * self.INPUT_SIZE
        lower = np.concatenate(
            [
                np.full(input_count, -self.max_acceleration, dtype=float),
                -self.max_velocity - free_velocity,
            ]
        )
        upper = np.concatenate(
            [
                np.full(input_count, self.max_acceleration, dtype=float),
                self.max_velocity - free_velocity,
            ]
        )
        solve_start = time.perf_counter()
        try:
            self.solver.update(q=np.asarray(linear_term, dtype=float), l=lower, u=upper)
            self.solver.warm_start(x=self.previous_solution)
            result = self.solver.solve(raise_error=False)
        except Exception:  # noqa: BLE001
            return None
        solve_ms = (time.perf_counter() - solve_start) * 1e3
        status = str(getattr(result.info, "status", "unknown")).lower()
        if result.x is None or status not in ("solved", "solved inaccurate"):
            return PredictiveAntiswayResult(
                acceleration=np.zeros(3, dtype=float),
                sequence=np.zeros_like(self.previous_solution),
                solve_ms=solve_ms,
                iterations=int(getattr(result.info, "iter", 0)),
                status=status,
            )
        sequence = np.asarray(result.x, dtype=float)
        if sequence.shape != self.previous_solution.shape or not np.all(np.isfinite(sequence)):
            return None
        self.previous_solution = sequence.copy()
        return PredictiveAntiswayResult(
            acceleration=sequence[:3].copy(),
            sequence=sequence,
            solve_ms=solve_ms,
            iterations=int(getattr(result.info, "iter", 0)),
            status=status,
        )


class BimanualEsdfAvoidanceFilter(Node):
    def __init__(self) -> None:
        super().__init__("bimanual_esdf_avoidance_filter")

        self.declare_parameter("urdf_path", "")
        self.declare_parameter("robot_description_node", "/robot_state_publisher")
        self.declare_parameter("left_joint_names", LEFT_JOINT_NAMES)
        self.declare_parameter("right_joint_names", RIGHT_JOINT_NAMES)
        self.declare_parameter("left_input_command_topic", "/left_teleop_baseline/commands")
        self.declare_parameter("right_input_command_topic", "/right_teleop_baseline/commands")
        self.declare_parameter("left_output_command_topic", "/left_forward_position_controller/commands")
        self.declare_parameter("right_output_command_topic", "/right_forward_position_controller/commands")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("joint_state_timeout_s", 0.15)
        self.declare_parameter("use_measured_joint_state_start", True)
        self.declare_parameter("esdf_service", "")
        self.declare_parameter("global_frame", "world")
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("esdf_update_hz", 8.0)
        self.declare_parameter("aabb_padding", 0.25)
        self.declare_parameter("unobserved_value", -1000.0)
        self.declare_parameter("min_valid_esdf_distance", -0.25)
        self.declare_parameter("request_update_esdf", True)
        self.declare_parameter("nearest_observed_search_radius", 0.12)
        self.declare_parameter("enable_esdf_virtual_extension", False)
        self.declare_parameter("esdf_extension_obstacle_direction", [-1.0, 0.0, 0.0])
        self.declare_parameter("esdf_extension_length", 0.0)
        self.declare_parameter("esdf_extension_step", 0.03)
        self.declare_parameter("obstacle_source", "esdf")
        self.declare_parameter("cable_capsule_topic", "/perception/cable_capsules")
        self.declare_parameter("protected_cable_topic", "/protected_cables/estimate")
        self.declare_parameter("ground_truth_cable_topic", "/protected_cables/ground_truth")
        self.declare_parameter("legacy_cable_marker_fallback", True)
        self.declare_parameter("cable_capsule_timeout_s", 0.75)
        self.declare_parameter("cable_capsule_timeout_behavior", "use_last")
        self.declare_parameter("cable_capsule_radius_scale", 1.0)
        self.declare_parameter("cable_capsule_padding", 0.0)
        self.declare_parameter("cable_uncertainty_sigma_scale", 2.0)
        self.declare_parameter("cable_capsule_max_count", 32)
        self.declare_parameter("untangle_mode_topic", "/openarmx/untangle_mode")
        self.declare_parameter("untangle_tangent_weight", 1.2)
        self.declare_parameter("untangle_tangent_max_step", 0.012)
        self.declare_parameter("collision_model", "capsule")
        self.declare_parameter("capsule_sample_spacing", 0.06)
        self.declare_parameter("capsule_min_samples", 2)
        self.declare_parameter("capsule_samples_per_link", 3)
        self.declare_parameter("end_effector_collision_padding", 0.01)
        self.declare_parameter("esdf_skip_proximal_spheres", 3)
        self.declare_parameter("assisted_grasp_enabled", False)
        self.declare_parameter("left_grasp_target_topic", "/visual_cues/left_selected_target_pose")
        self.declare_parameter("right_grasp_target_topic", "/visual_cues/right_selected_target_pose")
        self.declare_parameter("target_locked_topic", "/visual_cues/target_locked")
        self.declare_parameter("assisted_grasp_activation_distance", 0.08)
        self.declare_parameter("assisted_grasp_ramp_duration", 0.40)
        self.declare_parameter("assisted_grasp_max_cartesian_step", 0.04)
        self.declare_parameter("assisted_grasp_require_open_gripper", True)
        self.declare_parameter("assisted_grasp_gripper_open_fraction", 0.10)
        self.declare_parameter("assisted_grasp_gripper_close_fraction", 0.03)
        self.declare_parameter("assisted_grasp_gripper_min", 0.0)
        self.declare_parameter("assisted_grasp_gripper_max", 0.044)

        self.declare_parameter("monitor_only", False)
        self.declare_parameter("avoidance_solver", "cbf_qp")
        self.declare_parameter("safety_margin", 0.02)
        self.declare_parameter("activation_margin", 0.08)
        self.declare_parameter("target_clearance_margin", 0.008)
        self.declare_parameter("clearance_filter_alpha", 0.35)
        self.declare_parameter("baseline_weight", 1.0)
        self.declare_parameter("max_baseline_joint_error", 0.20)
        self.declare_parameter("tcp_position_weight", 3.0)
        self.declare_parameter("tcp_orientation_weight", 2.5)
        self.declare_parameter("tcp_position_max_step", 0.025)
        self.declare_parameter("tcp_orientation_max_step", 0.12)
        self.declare_parameter("wrist_tcp_preservation_enabled", True)
        self.declare_parameter("wrist_baseline_weight_scale", 0.25)
        self.declare_parameter("wrist_tcp_position_weight_scale", 3.0)
        self.declare_parameter("wrist_tcp_orientation_weight_scale", 0.20)
        self.declare_parameter("latched_baseline_weight_scale", 0.25)
        self.declare_parameter("avoidance_weight", 0.7)
        self.declare_parameter("damping", 0.04)
        self.declare_parameter("iterations", 2)
        self.declare_parameter("max_adjust_per_joint", 0.02)
        self.declare_parameter("max_command_step", 0.035)
        self.declare_parameter("max_command_acceleration", 3.0)
        self.declare_parameter("max_command_jerk", 0.0)
        self.declare_parameter("command_governor_enabled", True)
        self.declare_parameter("command_governor_max_velocity", 1.00)
        self.declare_parameter("command_governor_max_acceleration", 2.0)
        self.declare_parameter("command_governor_max_deceleration", 3.0)
        self.declare_parameter("command_governor_max_jerk", 15.0)
        self.declare_parameter("command_governor_position_gain", 5.0)
        self.declare_parameter("command_governor_position_tolerance", 1e-4)
        self.declare_parameter("max_avoidance_delta", 0.05)
        self.declare_parameter("avoidance_delta_alpha", 1.0)
        self.declare_parameter("avoidance_release_ramp_duration", 0.5)
        self.declare_parameter("avoidance_release_min_scale", 0.15)
        self.declare_parameter("prefer_z_avoidance", True)
        self.declare_parameter("z_gradient_min_abs", 0.15)
        self.declare_parameter("xy_follow_weight", 1.4)
        self.declare_parameter("xy_follow_max_step", 0.025)
        self.declare_parameter("downward_bias_weight", 0.08)
        self.declare_parameter("downward_bias_step", 0.005)
        self.declare_parameter("downward_bias_z", -1.0)
        self.declare_parameter("downward_tangent_bias_enabled", True)
        self.declare_parameter("tangent_escape_enabled", True)
        self.declare_parameter("tangent_escape_hand_only", True)
        self.declare_parameter("tangent_escape_prefer_downward", True)
        self.declare_parameter("hand_bypass_allow_autonomous_completion", True)
        self.declare_parameter("tangent_escape_weight", 4.0)
        self.declare_parameter("tangent_escape_step", 0.010)
        self.declare_parameter("tangent_escape_activation_margin", 0.08)
        self.declare_parameter("tangent_escape_inward_threshold", 0.001)
        self.declare_parameter("predictive_rollout_enabled", False)
        self.declare_parameter("rollout_horizon_steps", 5)
        self.declare_parameter("rollout_down_offsets", [0.0, 0.02, 0.04, 0.06])
        self.declare_parameter("rollout_side_offsets", [0.0, -0.02, 0.02])
        self.declare_parameter("rollout_side_axis", [0.0, 1.0, 0.0])
        self.declare_parameter("rollout_trigger_margin", 0.04)
        self.declare_parameter("rollout_max_joint_delta", 0.10)
        self.declare_parameter("rollout_cartesian_damping", 0.04)
        self.declare_parameter("rollout_collision_weight", 1200.0)
        self.declare_parameter("rollout_activation_weight", 40.0)
        self.declare_parameter("rollout_xy_weight", 70.0)
        self.declare_parameter("rollout_z_weight", 8.0)
        self.declare_parameter("rollout_joint_weight", 4.0)
        self.declare_parameter("rollout_smoothness_weight", 12.0)
        self.declare_parameter("hold_on_invalid_clearance", True)
        self.declare_parameter("avoidance_latch_enabled", True)
        self.declare_parameter("avoidance_release_margin", 0.12)
        self.declare_parameter("avoidance_release_cycles", 6)
        self.declare_parameter("cbf_gain", 8.0)
        self.declare_parameter("cbf_slack_weight", 250.0)
        self.declare_parameter("cbf_max_esdf_constraints", 18)
        self.declare_parameter("cbf_max_inter_arm_constraints", 16)
        self.declare_parameter("cbf_max_iterations", 40)
        self.declare_parameter("cbf_ftol", 1e-4)
        self.declare_parameter("cbf_fallback_to_soft", True)
        self.declare_parameter("cbf_qp_backend", "osqp")
        self.declare_parameter("cbf_osqp_max_iterations", 200)
        self.declare_parameter("cbf_osqp_eps_abs", 1e-4)
        self.declare_parameter("cbf_osqp_eps_rel", 1e-4)
        self.declare_parameter("cbf_osqp_time_limit_s", 0.004)
        self.declare_parameter("cbf_master_motion_sync_enabled", True)
        self.declare_parameter("cbf_master_motion_enter_velocity", 0.02)
        self.declare_parameter("cbf_master_motion_release_velocity", 0.01)
        self.declare_parameter("cbf_master_motion_max_step_gain", 1.60)
        self.declare_parameter("cbf_master_motion_catchup_error_gain", 0.08)
        self.declare_parameter("cbf_master_motion_max_catchup_step", 0.012)
        self.declare_parameter("cbf_safe_baseline_catchup_enabled", True)
        self.declare_parameter("cbf_safe_baseline_catchup_margin", 0.005)

        self.declare_parameter("antisway_enabled", True)
        self.declare_parameter("antisway_monitor_only", True)
        self.declare_parameter("antisway_modal_state_topic", "/openarmx/antisway/modal_state")
        self.declare_parameter(
            "antisway_observer_diagnostics_topic",
            "/openarmx/antisway/observer_diagnostics",
        )
        self.declare_parameter("antisway_observer_valid_topic", "/openarmx/antisway/observer_valid")
        self.declare_parameter("antisway_diagnostics_topic", "/openarmx/antisway/controller_diagnostics")
        self.declare_parameter("antisway_observer_timeout_s", 0.15)
        self.declare_parameter("antisway_modal_weight", 0.02)
        self.declare_parameter("antisway_acceleration_change_weight", 0.0002)
        self.declare_parameter("antisway_acceleration_scale", 10.0)
        self.declare_parameter("antisway_nis_full_confidence", 6.0)
        self.declare_parameter("antisway_nis_zero_confidence", 20.0)
        self.declare_parameter("antisway_confidence_rise_time_s", 0.25)
        self.declare_parameter("antisway_confidence_fall_time_s", 0.20)
        self.declare_parameter("antisway_confidence_enter_threshold", 0.35)
        self.declare_parameter("antisway_confidence_exit_threshold", 0.10)
        self.declare_parameter("antisway_input_shaper_enabled", True)
        self.declare_parameter("antisway_input_shaper_quality_factors", [3.0, 3.0])
        self.declare_parameter("antisway_input_shaper_strengths", [0.45, 0.50])
        self.declare_parameter("antisway_input_shaper_max_correction", 0.02)
        self.declare_parameter("antisway_input_shaper_max_correction_rate", 0.12)
        self.declare_parameter("antisway_gru_control_enabled", False)
        self.declare_parameter(
            "antisway_gru_prediction_topic",
            "/openarmx/antisway/residual_predicted_state",
        )
        self.declare_parameter("antisway_gru_prediction_timeout_s", 0.20)
        self.declare_parameter("antisway_gru_equivalent_speed_low", 0.03)
        self.declare_parameter("antisway_gru_equivalent_speed_high", 0.18)
        self.declare_parameter("antisway_gru_min_strength_scale", 0.20)
        self.declare_parameter("antisway_gru_strength_rise_time_s", 0.12)
        self.declare_parameter("antisway_gru_strength_fall_time_s", 0.35)
        self.declare_parameter("antisway_predictive_enabled", False)
        self.declare_parameter("antisway_horizon_steps", 12)
        self.declare_parameter("antisway_roll_weight", 20.0)
        self.declare_parameter("antisway_yaw_weight", 40.0)
        self.declare_parameter("antisway_tracking_weight", 4.0)
        self.declare_parameter("antisway_velocity_weight", 0.05)
        self.declare_parameter("antisway_mpc_acceleration_weight", 0.02)
        self.declare_parameter("antisway_mpc_acceleration_change_weight", 0.20)
        self.declare_parameter("antisway_terminal_weight_scale", 2.0)
        self.declare_parameter("antisway_max_acceleration", 6.0)
        self.declare_parameter("antisway_max_velocity", 1.0)
        self.declare_parameter("antisway_max_reference_deviation", 0.15)
        self.declare_parameter("antisway_baseline_velocity_alpha", 0.20)
        self.declare_parameter("antisway_baseline_velocity_deadband", 0.01)
        self.declare_parameter("antisway_baseline_velocity_limit", 1.20)
        self.declare_parameter("antisway_motion_gate_enabled", True)
        self.declare_parameter("antisway_motion_gate_velocity", 0.03)
        self.declare_parameter("antisway_motion_gate_release_velocity", 0.015)
        self.declare_parameter("antisway_bimanual_task_qp_enabled", False)
        self.declare_parameter(
            "antisway_bimanual_modal_coupling_flat",
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
        self.declare_parameter(
            "antisway_bimanual_additional_damping_ratios",
            [0.08, 0.08],
        )
        self.declare_parameter(
            "antisway_bimanual_position_feedback_gains",
            [0.02, 0.02],
        )
        self.declare_parameter("antisway_bimanual_modal_weights", [1.0, 1.5])
        self.declare_parameter("antisway_bimanual_task_translation_weight", 0.20)
        self.declare_parameter("antisway_bimanual_task_rotation_weight", 0.10)
        self.declare_parameter("antisway_bimanual_correction_weight", 0.002)
        self.declare_parameter("antisway_bimanual_max_joint_acceleration", 1.5)
        self.declare_parameter("antisway_bimanual_position_horizon_s", 0.06)
        self.declare_parameter("antisway_bimanual_max_position_correction", 0.004)
        self.declare_parameter("antisway_bimanual_max_correction_rate", 0.04)
        self.declare_parameter("antisway_bimanual_acceleration_alpha", 0.20)
        self.declare_parameter("antisway_bimanual_min_confidence", 0.20)
        self.declare_parameter("antisway_bimanual_nullspace_rcond", 1e-4)
        self.declare_parameter("antisway_bimanual_qp_max_iterations", 50)
        self.declare_parameter("antisway_bimanual_qp_eps_abs", 1e-4)
        self.declare_parameter("antisway_bimanual_qp_eps_rel", 1e-4)
        self.declare_parameter("antisway_bimanual_qp_time_limit_s", 0.002)
        self.declare_parameter("antisway_mpc_max_iterations", 200)
        self.declare_parameter("antisway_mpc_eps_abs", 1e-4)
        self.declare_parameter("antisway_mpc_eps_rel", 1e-4)
        self.declare_parameter("antisway_mpc_time_limit_s", 0.004)
        self.declare_parameter("antisway_modal_frequencies_hz", [1.6846, 2.4155])
        self.declare_parameter("antisway_modal_damping_ratios", [0.07918, 0.05082])
        self.declare_parameter("antisway_disturbance_cutoff_hz", 1.0)
        self.declare_parameter(
            "antisway_modal_coupling_flat",
            [
                -0.23503855,
                -0.00061910,
                -0.09961852,
                0.01564604,
                0.17299390,
                0.00756648,
            ],
        )

        self.declare_parameter("enable_inter_arm_collision", True)
        self.declare_parameter("inter_arm_safety_margin", 0.03)
        self.declare_parameter("inter_arm_activation_margin", 0.09)
        self.declare_parameter("inter_arm_weight", 0.8)
        self.declare_parameter("inter_arm_skip_proximal_spheres", 3)

        self.declare_parameter("clear_robot_from_esdf", False)
        self.declare_parameter("clear_robot_padding", 0.015)
        self.declare_parameter("clear_robot_radius_scale", 1.0)
        self.declare_parameter("publish_markers", True)
        self.declare_parameter("marker_topic", "/openarmx/bimanual/esdf_spheres")
        self.declare_parameter("publish_avoidance_arrows", True)
        self.declare_parameter("avoidance_arrow_max_count", 6)
        self.declare_parameter("avoidance_arrow_min_length", 0.025)
        self.declare_parameter("avoidance_arrow_max_length", 0.12)
        self.declare_parameter("min_clearance_topic", "/openarmx/bimanual/min_esdf_clearance")
        self.declare_parameter(
            "left_min_clearance_topic",
            "/openarmx/bimanual/left_min_esdf_clearance",
        )
        self.declare_parameter(
            "right_min_clearance_topic",
            "/openarmx/bimanual/right_min_esdf_clearance",
        )
        self.declare_parameter("status_topic", "/openarmx/bimanual/esdf_avoidance_status")
        self.declare_parameter("structured_status_topic", "/openarmx/bimanual/safety_status")

        self.urdf_path = str(self.get_parameter("urdf_path").value).strip()
        self.robot_description_node = str(self.get_parameter("robot_description_node").value)
        self.left_joint_names = list(self.get_parameter("left_joint_names").value)
        self.right_joint_names = list(self.get_parameter("right_joint_names").value)
        self.joint_names = self.left_joint_names + self.right_joint_names
        self.left_input_command_topic = str(self.get_parameter("left_input_command_topic").value)
        self.right_input_command_topic = str(self.get_parameter("right_input_command_topic").value)
        self.left_output_command_topic = str(self.get_parameter("left_output_command_topic").value)
        self.right_output_command_topic = str(self.get_parameter("right_output_command_topic").value)
        self.joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self.joint_state_timeout_s = max(
            0.01,
            float(self.get_parameter("joint_state_timeout_s").value),
        )
        self.use_measured_joint_state_start = bool(
            self.get_parameter("use_measured_joint_state_start").value
        )
        self.esdf_service_name = str(self.get_parameter("esdf_service").value).strip()
        self.global_frame = str(self.get_parameter("global_frame").value)
        self.rate_hz = max(1.0, float(self.get_parameter("rate_hz").value))
        self.esdf_update_hz = max(0.5, float(self.get_parameter("esdf_update_hz").value))
        self.aabb_padding = max(0.05, float(self.get_parameter("aabb_padding").value))
        self.unobserved_value = float(self.get_parameter("unobserved_value").value)
        self.min_valid_esdf_distance = float(self.get_parameter("min_valid_esdf_distance").value)
        self.request_update_esdf = bool(self.get_parameter("request_update_esdf").value)
        self.nearest_observed_search_radius = max(
            0.0,
            float(self.get_parameter("nearest_observed_search_radius").value),
        )
        self.enable_esdf_virtual_extension = bool(
            self.get_parameter("enable_esdf_virtual_extension").value
        )
        self.esdf_extension_direction = self._parse_vector3_param(
            self.get_parameter("esdf_extension_obstacle_direction").value,
            "esdf_extension_obstacle_direction",
        )
        extension_direction_norm = float(np.linalg.norm(self.esdf_extension_direction))
        if not np.isfinite(extension_direction_norm) or extension_direction_norm < 1e-9:
            self.esdf_extension_direction = np.array([-1.0, 0.0, 0.0], dtype=float)
        else:
            self.esdf_extension_direction /= extension_direction_norm
        self.esdf_extension_length = max(0.0, float(self.get_parameter("esdf_extension_length").value))
        self.esdf_extension_step = max(1e-3, float(self.get_parameter("esdf_extension_step").value))
        self.esdf_extension_offsets = self._build_esdf_extension_offsets()
        self.obstacle_source = str(self.get_parameter("obstacle_source").value).strip().lower()
        if self.obstacle_source not in ("esdf", "cable_capsules", "hybrid"):
            self.get_logger().warn(
                f"Unknown obstacle_source={self.obstacle_source!r}; falling back to 'esdf'."
            )
            self.obstacle_source = "esdf"
        self.cable_capsule_topic = str(self.get_parameter("cable_capsule_topic").value)
        self.protected_cable_topic = str(self.get_parameter("protected_cable_topic").value)
        self.ground_truth_cable_topic = str(self.get_parameter("ground_truth_cable_topic").value)
        self.legacy_cable_marker_fallback = bool(
            self.get_parameter("legacy_cable_marker_fallback").value
        )
        self.cable_capsule_timeout_s = max(
            0.05,
            float(self.get_parameter("cable_capsule_timeout_s").value),
        )
        self.cable_capsule_timeout_behavior = str(
            self.get_parameter("cable_capsule_timeout_behavior").value
        ).strip().lower()
        if self.cable_capsule_timeout_behavior not in (
            "use_last",
            "hold",
            "fallback_esdf",
            "pass_through",
        ):
            self.get_logger().warn(
                "cable_capsule_timeout_behavior must be use_last, hold, fallback_esdf, "
                "or pass_through; using use_last."
            )
            self.cable_capsule_timeout_behavior = "use_last"
        self.cable_capsule_radius_scale = max(
            0.0,
            float(self.get_parameter("cable_capsule_radius_scale").value),
        )
        self.cable_capsule_padding = max(
            0.0,
            float(self.get_parameter("cable_capsule_padding").value),
        )
        self.cable_uncertainty_sigma_scale = max(
            0.0,
            float(self.get_parameter("cable_uncertainty_sigma_scale").value),
        )
        self.cable_capsule_max_count = max(
            1,
            int(self.get_parameter("cable_capsule_max_count").value),
        )
        self.untangle_tangent_weight = max(
            0.0,
            float(self.get_parameter("untangle_tangent_weight").value),
        )
        self.untangle_tangent_max_step = max(
            0.0,
            float(self.get_parameter("untangle_tangent_max_step").value),
        )
        self.collision_model = str(self.get_parameter("collision_model").value).strip().lower()
        if self.collision_model not in ("sphere", "capsule"):
            self.get_logger().warn(
                f"Unknown collision_model={self.collision_model!r}; falling back to 'capsule'."
            )
            self.collision_model = "capsule"
        self.capsule_sample_spacing = max(
            1e-3,
            float(self.get_parameter("capsule_sample_spacing").value),
        )
        self.capsule_min_samples = max(
            2,
            int(self.get_parameter("capsule_min_samples").value),
        )
        self.capsule_samples_per_link = max(
            2,
            int(self.get_parameter("capsule_samples_per_link").value),
        )
        self.end_effector_collision_padding = max(
            0.0,
            float(self.get_parameter("end_effector_collision_padding").value),
        )
        self.esdf_skip_proximal_spheres = max(
            0,
            int(self.get_parameter("esdf_skip_proximal_spheres").value),
        )

        self.monitor_only = bool(self.get_parameter("monitor_only").value)
        self.safety_margin = max(0.0, float(self.get_parameter("safety_margin").value))
        self.activation_margin = max(
            self.safety_margin,
            float(self.get_parameter("activation_margin").value),
        )
        self.target_clearance_margin = max(
            0.0,
            float(self.get_parameter("target_clearance_margin").value),
        )
        self.clearance_filter_alpha = float(
            np.clip(float(self.get_parameter("clearance_filter_alpha").value), 0.0, 1.0)
        )
        self.baseline_weight = max(1e-6, float(self.get_parameter("baseline_weight").value))
        self.max_baseline_joint_error = max(
            0.0,
            float(self.get_parameter("max_baseline_joint_error").value),
        )
        self.tcp_position_weight = max(
            0.0,
            float(self.get_parameter("tcp_position_weight").value),
        )
        self.tcp_orientation_weight = max(
            0.0,
            float(self.get_parameter("tcp_orientation_weight").value),
        )
        self.tcp_position_max_step = max(
            0.0,
            float(self.get_parameter("tcp_position_max_step").value),
        )
        self.tcp_orientation_max_step = max(
            0.0,
            float(self.get_parameter("tcp_orientation_max_step").value),
        )
        self.wrist_tcp_preservation_enabled = bool(
            self.get_parameter("wrist_tcp_preservation_enabled").value
        )
        self.wrist_baseline_weight_scale = float(
            np.clip(
                float(self.get_parameter("wrist_baseline_weight_scale").value),
                0.0,
                1.0,
            )
        )
        self.wrist_tcp_position_weight_scale = max(
            0.0,
            float(self.get_parameter("wrist_tcp_position_weight_scale").value),
        )
        self.wrist_tcp_orientation_weight_scale = max(
            0.0,
            float(self.get_parameter("wrist_tcp_orientation_weight_scale").value),
        )
        self.latched_baseline_weight_scale = float(
            np.clip(float(self.get_parameter("latched_baseline_weight_scale").value), 0.0, 1.0)
        )
        self.avoidance_weight = max(0.0, float(self.get_parameter("avoidance_weight").value))
        self.damping = max(1e-6, float(self.get_parameter("damping").value))
        self.iterations = max(1, int(self.get_parameter("iterations").value))
        self.max_adjust_per_joint = max(0.0, float(self.get_parameter("max_adjust_per_joint").value))
        self.max_command_step = max(0.0, float(self.get_parameter("max_command_step").value))
        self.max_command_acceleration = max(
            0.0,
            float(self.get_parameter("max_command_acceleration").value),
        )
        self.max_command_jerk = max(
            0.0,
            float(self.get_parameter("max_command_jerk").value),
        )
        self.command_governor_enabled = bool(
            self.get_parameter("command_governor_enabled").value
        )
        self.command_governor = JerkLimitedCommandGovernor(
            joint_count=len(BIMANUAL_JOINT_NAMES),
            max_velocity=max(
                1e-3,
                float(self.get_parameter("command_governor_max_velocity").value),
            ),
            max_acceleration=max(
                1e-3,
                float(self.get_parameter("command_governor_max_acceleration").value),
            ),
            max_deceleration=max(
                1e-3,
                float(self.get_parameter("command_governor_max_deceleration").value),
            ),
            max_jerk=max(
                1e-3,
                float(self.get_parameter("command_governor_max_jerk").value),
            ),
            position_gain=max(
                1e-3,
                float(self.get_parameter("command_governor_position_gain").value),
            ),
            position_tolerance=max(
                1e-8,
                float(self.get_parameter("command_governor_position_tolerance").value),
            ),
        )
        self.command_governor_tracking_error = np.zeros(
            len(BIMANUAL_JOINT_NAMES),
            dtype=float,
        )
        self.max_avoidance_delta = max(0.0, float(self.get_parameter("max_avoidance_delta").value))
        self.avoidance_delta_alpha = float(
            np.clip(float(self.get_parameter("avoidance_delta_alpha").value), 0.0, 1.0)
        )
        self.avoidance_release_ramp_duration = max(
            0.0,
            float(self.get_parameter("avoidance_release_ramp_duration").value),
        )
        self.avoidance_release_min_scale = float(
            np.clip(
                float(self.get_parameter("avoidance_release_min_scale").value),
                0.0,
                1.0,
            )
        )
        self.prefer_z_avoidance = bool(self.get_parameter("prefer_z_avoidance").value)
        self.z_gradient_min_abs = max(0.0, float(self.get_parameter("z_gradient_min_abs").value))
        self.xy_follow_weight = max(0.0, float(self.get_parameter("xy_follow_weight").value))
        self.xy_follow_max_step = max(0.0, float(self.get_parameter("xy_follow_max_step").value))
        self.downward_bias_weight = max(0.0, float(self.get_parameter("downward_bias_weight").value))
        self.downward_bias_step = max(0.0, float(self.get_parameter("downward_bias_step").value))
        self.downward_tangent_bias_enabled = bool(
            self.get_parameter("downward_tangent_bias_enabled").value
        )
        self.tangent_escape_enabled = bool(self.get_parameter("tangent_escape_enabled").value)
        self.tangent_escape_hand_only = bool(
            self.get_parameter("tangent_escape_hand_only").value
        )
        self.tangent_escape_prefer_downward = bool(
            self.get_parameter("tangent_escape_prefer_downward").value
        )
        self.hand_bypass_allow_autonomous_completion = bool(
            self.get_parameter("hand_bypass_allow_autonomous_completion").value
        )
        self.tangent_escape_weight = max(
            0.0,
            float(self.get_parameter("tangent_escape_weight").value),
        )
        self.tangent_escape_step = max(
            0.0,
            float(self.get_parameter("tangent_escape_step").value),
        )
        self.tangent_escape_activation_margin = max(
            self.safety_margin,
            float(self.get_parameter("tangent_escape_activation_margin").value),
        )
        self.tangent_escape_inward_threshold = max(
            0.0,
            float(self.get_parameter("tangent_escape_inward_threshold").value),
        )
        self.downward_direction = np.array(
            [0.0, 0.0, float(self.get_parameter("downward_bias_z").value)],
            dtype=float,
        )
        direction_norm = float(np.linalg.norm(self.downward_direction))
        if not np.isfinite(direction_norm) or direction_norm < 1e-9:
            self.downward_direction = np.array([0.0, 0.0, -1.0], dtype=float)
        else:
            self.downward_direction /= direction_norm
        self.predictive_rollout_enabled = bool(
            self.get_parameter("predictive_rollout_enabled").value
        )
        self.rollout_horizon_steps = max(
            1,
            int(self.get_parameter("rollout_horizon_steps").value),
        )
        self.rollout_down_offsets = self._parse_float_list_param(
            self.get_parameter("rollout_down_offsets").value,
            "rollout_down_offsets",
        )
        if not self.rollout_down_offsets:
            self.rollout_down_offsets = [0.0]
        self.rollout_side_offsets = self._parse_float_list_param(
            self.get_parameter("rollout_side_offsets").value,
            "rollout_side_offsets",
        )
        if not self.rollout_side_offsets:
            self.rollout_side_offsets = [0.0]
        self.rollout_side_axis = self._parse_vector3_param(
            self.get_parameter("rollout_side_axis").value,
            "rollout_side_axis",
        )
        side_axis_norm = float(np.linalg.norm(self.rollout_side_axis))
        if not np.isfinite(side_axis_norm) or side_axis_norm < 1e-9:
            self.rollout_side_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            self.rollout_side_axis /= side_axis_norm
        self.rollout_trigger_margin = max(
            0.0,
            float(self.get_parameter("rollout_trigger_margin").value),
        )
        self.rollout_max_joint_delta = max(
            0.0,
            float(self.get_parameter("rollout_max_joint_delta").value),
        )
        self.rollout_cartesian_damping = max(
            1e-6,
            float(self.get_parameter("rollout_cartesian_damping").value),
        )
        self.rollout_collision_weight = max(
            0.0,
            float(self.get_parameter("rollout_collision_weight").value),
        )
        self.rollout_activation_weight = max(
            0.0,
            float(self.get_parameter("rollout_activation_weight").value),
        )
        self.rollout_xy_weight = max(0.0, float(self.get_parameter("rollout_xy_weight").value))
        self.rollout_z_weight = max(0.0, float(self.get_parameter("rollout_z_weight").value))
        self.rollout_joint_weight = max(
            0.0,
            float(self.get_parameter("rollout_joint_weight").value),
        )
        self.rollout_smoothness_weight = max(
            0.0,
            float(self.get_parameter("rollout_smoothness_weight").value),
        )
        self.hold_on_invalid_clearance = bool(self.get_parameter("hold_on_invalid_clearance").value)
        self.avoidance_latch_enabled = bool(self.get_parameter("avoidance_latch_enabled").value)
        self.avoidance_release_margin = max(
            self.activation_margin,
            float(self.get_parameter("avoidance_release_margin").value),
        )
        self.avoidance_release_cycles = max(1, int(self.get_parameter("avoidance_release_cycles").value))
        self.enable_inter_arm_collision = bool(self.get_parameter("enable_inter_arm_collision").value)
        self.inter_arm_safety_margin = max(0.0, float(self.get_parameter("inter_arm_safety_margin").value))
        self.inter_arm_activation_margin = max(
            self.inter_arm_safety_margin,
            float(self.get_parameter("inter_arm_activation_margin").value),
        )
        self.inter_arm_weight = max(0.0, float(self.get_parameter("inter_arm_weight").value))
        self.inter_arm_skip_proximal_spheres = max(
            0,
            int(self.get_parameter("inter_arm_skip_proximal_spheres").value),
        )
        self.clear_robot_from_esdf = bool(self.get_parameter("clear_robot_from_esdf").value)
        self.clear_robot_padding = max(0.0, float(self.get_parameter("clear_robot_padding").value))
        self.clear_robot_radius_scale = max(0.0, float(self.get_parameter("clear_robot_radius_scale").value))
        self.publish_markers = bool(self.get_parameter("publish_markers").value)
        self.publish_avoidance_arrows = bool(
            self.get_parameter("publish_avoidance_arrows").value
        )
        self.avoidance_arrow_max_count = max(
            1,
            int(self.get_parameter("avoidance_arrow_max_count").value),
        )
        self.avoidance_arrow_min_length = max(
            0.005,
            float(self.get_parameter("avoidance_arrow_min_length").value),
        )
        self.avoidance_arrow_max_length = max(
            self.avoidance_arrow_min_length,
            float(self.get_parameter("avoidance_arrow_max_length").value),
        )
        self.avoidance_solver = str(self.get_parameter("avoidance_solver").value).strip().lower()
        if self.avoidance_solver not in ("soft", "cbf_qp"):
            self.get_logger().warn(
                f"Unknown avoidance_solver={self.avoidance_solver!r}; falling back to 'soft'."
            )
            self.avoidance_solver = "soft"
        self.cbf_gain = max(0.0, float(self.get_parameter("cbf_gain").value))
        self.cbf_slack_weight = max(1e-6, float(self.get_parameter("cbf_slack_weight").value))
        self.cbf_max_esdf_constraints = max(
            1,
            int(self.get_parameter("cbf_max_esdf_constraints").value),
        )
        self.cbf_max_inter_arm_constraints = max(
            0,
            int(self.get_parameter("cbf_max_inter_arm_constraints").value),
        )
        self.cbf_max_iterations = max(5, int(self.get_parameter("cbf_max_iterations").value))
        self.cbf_ftol = max(1e-8, float(self.get_parameter("cbf_ftol").value))
        self.cbf_fallback_to_soft = bool(self.get_parameter("cbf_fallback_to_soft").value)
        self.cbf_qp_backend = str(self.get_parameter("cbf_qp_backend").value).strip().lower()
        if self.cbf_qp_backend not in ("osqp", "slsqp"):
            raise RuntimeError("cbf_qp_backend must be 'osqp' or 'slsqp'")
        if self.cbf_qp_backend == "osqp" and (osqp is None or sparse is None):
            self.get_logger().warn("OSQP is unavailable; falling back to SLSQP.")
            self.cbf_qp_backend = "slsqp"
        self.cbf_osqp_max_iterations = max(
            20,
            int(self.get_parameter("cbf_osqp_max_iterations").value),
        )
        self.cbf_osqp_eps_abs = max(
            1e-7,
            float(self.get_parameter("cbf_osqp_eps_abs").value),
        )
        self.cbf_osqp_eps_rel = max(
            1e-7,
            float(self.get_parameter("cbf_osqp_eps_rel").value),
        )
        self.cbf_osqp_time_limit_s = max(
            0.0,
            float(self.get_parameter("cbf_osqp_time_limit_s").value),
        )
        self.cbf_master_motion_sync_enabled = bool(
            self.get_parameter("cbf_master_motion_sync_enabled").value
        )
        self.cbf_master_motion_enter_velocity = max(
            0.0,
            float(self.get_parameter("cbf_master_motion_enter_velocity").value),
        )
        self.cbf_master_motion_release_velocity = float(
            np.clip(
                float(self.get_parameter("cbf_master_motion_release_velocity").value),
                0.0,
                self.cbf_master_motion_enter_velocity,
            )
        )
        self.cbf_master_motion_max_step_gain = max(
            0.0,
            float(self.get_parameter("cbf_master_motion_max_step_gain").value),
        )
        self.cbf_master_motion_catchup_error_gain = max(
            0.0,
            float(
                self.get_parameter(
                    "cbf_master_motion_catchup_error_gain"
                ).value
            ),
        )
        self.cbf_master_motion_max_catchup_step = max(
            0.0,
            float(
                self.get_parameter(
                    "cbf_master_motion_max_catchup_step"
                ).value
            ),
        )
        self.cbf_safe_baseline_catchup_enabled = bool(
            self.get_parameter("cbf_safe_baseline_catchup_enabled").value
        )
        self.cbf_safe_baseline_catchup_margin = max(
            0.0,
            float(self.get_parameter("cbf_safe_baseline_catchup_margin").value),
        )

        self.antisway_enabled = bool(self.get_parameter("antisway_enabled").value)
        self.antisway_monitor_only = bool(self.get_parameter("antisway_monitor_only").value)
        self.antisway_observer_timeout_s = max(
            0.01,
            float(self.get_parameter("antisway_observer_timeout_s").value),
        )
        self.antisway_modal_weight = max(
            0.0,
            float(self.get_parameter("antisway_modal_weight").value),
        )
        self.antisway_acceleration_change_weight = max(
            0.0,
            float(self.get_parameter("antisway_acceleration_change_weight").value),
        )
        self.antisway_acceleration_scale = max(
            1e-3,
            float(self.get_parameter("antisway_acceleration_scale").value),
        )
        self.antisway_nis_full_confidence = max(
            0.0,
            float(self.get_parameter("antisway_nis_full_confidence").value),
        )
        self.antisway_nis_zero_confidence = max(
            self.antisway_nis_full_confidence + 1e-6,
            float(self.get_parameter("antisway_nis_zero_confidence").value),
        )
        self.antisway_predictive_enabled = bool(
            self.get_parameter("antisway_predictive_enabled").value
        )
        self.antisway_input_shaper_enabled = bool(
            self.get_parameter("antisway_input_shaper_enabled").value
        )
        self.antisway_gru_control_enabled = bool(
            self.get_parameter("antisway_gru_control_enabled").value
        )
        self.antisway_gru_prediction_timeout_s = max(
            0.02,
            float(self.get_parameter("antisway_gru_prediction_timeout_s").value),
        )
        self.antisway_gru_equivalent_speed_low = max(
            0.0,
            float(self.get_parameter("antisway_gru_equivalent_speed_low").value),
        )
        self.antisway_gru_equivalent_speed_high = max(
            self.antisway_gru_equivalent_speed_low + 1e-6,
            float(self.get_parameter("antisway_gru_equivalent_speed_high").value),
        )
        self.antisway_gru_min_strength_scale = float(
            np.clip(
                float(self.get_parameter("antisway_gru_min_strength_scale").value),
                0.0,
                1.0,
            )
        )
        self.antisway_gru_strength_rise_time_s = max(
            0.0,
            float(self.get_parameter("antisway_gru_strength_rise_time_s").value),
        )
        self.antisway_gru_strength_fall_time_s = max(
            0.0,
            float(self.get_parameter("antisway_gru_strength_fall_time_s").value),
        )
        self.antisway_horizon_steps = max(
            2,
            int(self.get_parameter("antisway_horizon_steps").value),
        )
        self.antisway_roll_weight = max(
            0.0,
            float(self.get_parameter("antisway_roll_weight").value),
        )
        self.antisway_yaw_weight = max(
            0.0,
            float(self.get_parameter("antisway_yaw_weight").value),
        )
        self.antisway_tracking_weight = max(
            1e-6,
            float(self.get_parameter("antisway_tracking_weight").value),
        )
        self.antisway_max_reference_deviation = max(
            0.0,
            float(self.get_parameter("antisway_max_reference_deviation").value),
        )
        self.antisway_baseline_velocity_alpha = float(
            np.clip(
                float(self.get_parameter("antisway_baseline_velocity_alpha").value),
                0.0,
                1.0,
            )
        )
        self.antisway_baseline_velocity_deadband = max(
            0.0,
            float(self.get_parameter("antisway_baseline_velocity_deadband").value),
        )
        self.antisway_baseline_velocity_limit = max(
            0.05,
            float(self.get_parameter("antisway_baseline_velocity_limit").value),
        )
        self.antisway_motion_gate_enabled = bool(
            self.get_parameter("antisway_motion_gate_enabled").value
        )
        self.antisway_motion_gate_velocity = max(
            0.0,
            float(self.get_parameter("antisway_motion_gate_velocity").value),
        )
        self.antisway_motion_gate_release_velocity = float(
            np.clip(
                float(
                    self.get_parameter(
                        "antisway_motion_gate_release_velocity"
                    ).value
                ),
                0.0,
                self.antisway_motion_gate_velocity,
            )
        )
        self.antisway_confidence_rise_time_s = max(
            0.0,
            float(self.get_parameter("antisway_confidence_rise_time_s").value),
        )
        self.antisway_confidence_fall_time_s = max(
            0.0,
            float(self.get_parameter("antisway_confidence_fall_time_s").value),
        )
        self.antisway_confidence_enter_threshold = float(
            np.clip(
                float(
                    self.get_parameter(
                        "antisway_confidence_enter_threshold"
                    ).value
                ),
                0.0,
                1.0,
            )
        )
        self.antisway_confidence_exit_threshold = float(
            np.clip(
                float(
                    self.get_parameter(
                        "antisway_confidence_exit_threshold"
                    ).value
                ),
                0.0,
                self.antisway_confidence_enter_threshold,
            )
        )
        self.antisway_bimanual_task_qp_enabled = bool(
            self.get_parameter("antisway_bimanual_task_qp_enabled").value
        )
        self.antisway_bimanual_additional_damping = np.asarray(
            self.get_parameter(
                "antisway_bimanual_additional_damping_ratios"
            ).value,
            dtype=float,
        )
        self.antisway_bimanual_position_feedback = np.asarray(
            self.get_parameter(
                "antisway_bimanual_position_feedback_gains"
            ).value,
            dtype=float,
        )
        self.antisway_bimanual_modal_weights = np.asarray(
            self.get_parameter("antisway_bimanual_modal_weights").value,
            dtype=float,
        )
        if (
            self.antisway_bimanual_additional_damping.shape != (2,)
            or self.antisway_bimanual_position_feedback.shape != (2,)
            or self.antisway_bimanual_modal_weights.shape != (2,)
        ):
            raise RuntimeError(
                "bimanual damping, position-feedback and modal-weight parameters "
                "must each contain 2 values"
            )
        if (
            np.any(self.antisway_bimanual_additional_damping < 0.0)
            or np.any(self.antisway_bimanual_position_feedback < 0.0)
            or np.any(self.antisway_bimanual_modal_weights < 0.0)
        ):
            raise RuntimeError("bimanual anti-sway gains and weights must be non-negative")
        self.antisway_bimanual_position_horizon_s = max(
            0.0,
            float(
                self.get_parameter(
                    "antisway_bimanual_position_horizon_s"
                ).value
            ),
        )
        self.antisway_bimanual_max_position_correction = max(
            0.0,
            float(
                self.get_parameter(
                    "antisway_bimanual_max_position_correction"
                ).value
            ),
        )
        self.antisway_bimanual_max_correction_rate = max(
            0.0,
            float(
                self.get_parameter(
                    "antisway_bimanual_max_correction_rate"
                ).value
            ),
        )
        self.antisway_bimanual_acceleration_alpha = float(
            np.clip(
                float(
                    self.get_parameter(
                        "antisway_bimanual_acceleration_alpha"
                    ).value
                ),
                0.0,
                1.0,
            )
        )
        self.antisway_bimanual_min_confidence = float(
            np.clip(
                float(
                    self.get_parameter(
                        "antisway_bimanual_min_confidence"
                    ).value
                ),
                0.0,
                1.0,
            )
        )
        self.antisway_confidence_enter_threshold = max(
            self.antisway_confidence_enter_threshold,
            self.antisway_bimanual_min_confidence,
        )
        antisway_frequencies = np.asarray(
            self.get_parameter("antisway_modal_frequencies_hz").value,
            dtype=float,
        )
        antisway_damping = np.asarray(
            self.get_parameter("antisway_modal_damping_ratios").value,
            dtype=float,
        )
        antisway_coupling_flat = np.asarray(
            self.get_parameter("antisway_modal_coupling_flat").value,
            dtype=float,
        )
        if antisway_frequencies.shape != (2,) or antisway_damping.shape != (2,):
            raise RuntimeError("antisway modal frequencies and damping must each contain 2 values")
        if antisway_coupling_flat.shape != (6,):
            raise RuntimeError("antisway_modal_coupling_flat must contain 6 values")
        self.antisway_frequencies = antisway_frequencies
        self.antisway_coupling = antisway_coupling_flat.reshape(2, 3)
        self.antisway_a, self.antisway_b = discretize_modal_model(
            self.rate_hz,
            antisway_frequencies,
            antisway_damping,
            self.antisway_coupling,
            max(
                1e-3,
                float(self.get_parameter("antisway_disturbance_cutoff_hz").value),
            ),
        )
        bimanual_coupling_values = np.asarray(
            self.get_parameter(
                "antisway_bimanual_modal_coupling_flat"
            ).value,
            dtype=float,
        )
        if bimanual_coupling_values.shape != (12,):
            raise RuntimeError(
                "antisway_bimanual_modal_coupling_flat must contain 12 values"
            )
        bimanual_modal_indices = np.asarray(
            [
                0,
                1,
                3,
                len(self.left_joint_names),
                len(self.left_joint_names) + 1,
                len(self.left_joint_names) + 3,
            ],
            dtype=int,
        )
        self.antisway_bimanual_modal_indices = bimanual_modal_indices
        self.antisway_bimanual_coupling = np.zeros(
            (2, len(self.joint_names)),
            dtype=float,
        )
        self.antisway_bimanual_coupling[:, bimanual_modal_indices] = (
            bimanual_coupling_values.reshape(2, 6)
        )
        self.antisway_bimanual_task_qp: (
            TaskPreservingBimanualAntiswayQp | None
        ) = None
        if self.antisway_enabled and self.antisway_bimanual_task_qp_enabled:
            self.antisway_bimanual_task_qp = TaskPreservingBimanualAntiswayQp(
                coupling=self.antisway_bimanual_coupling,
                modal_weights=self.antisway_bimanual_modal_weights,
                task_translation_weight=max(
                    0.0,
                    float(
                        self.get_parameter(
                            "antisway_bimanual_task_translation_weight"
                        ).value
                    ),
                ),
                task_rotation_weight=max(
                    0.0,
                    float(
                        self.get_parameter(
                            "antisway_bimanual_task_rotation_weight"
                        ).value
                    ),
                ),
                correction_weight=max(
                    1e-8,
                    float(
                        self.get_parameter(
                            "antisway_bimanual_correction_weight"
                        ).value
                    ),
                ),
                max_joint_acceleration=max(
                    0.01,
                    float(
                        self.get_parameter(
                            "antisway_bimanual_max_joint_acceleration"
                        ).value
                    ),
                ),
                nullspace_rcond=max(
                    1e-8,
                    float(
                        self.get_parameter(
                            "antisway_bimanual_nullspace_rcond"
                        ).value
                    ),
                ),
                max_iterations=max(
                    10,
                    int(
                        self.get_parameter(
                            "antisway_bimanual_qp_max_iterations"
                        ).value
                    ),
                ),
                eps_abs=max(
                    1e-7,
                    float(
                        self.get_parameter(
                            "antisway_bimanual_qp_eps_abs"
                        ).value
                    ),
                ),
                eps_rel=max(
                    1e-7,
                    float(
                        self.get_parameter(
                            "antisway_bimanual_qp_eps_rel"
                        ).value
                    ),
                ),
                time_limit_s=max(
                    0.0,
                    float(
                        self.get_parameter(
                            "antisway_bimanual_qp_time_limit_s"
                        ).value
                    ),
                ),
            )
        if self.antisway_bimanual_task_qp_enabled and (
            self.antisway_input_shaper_enabled
            or self.antisway_predictive_enabled
        ):
            self.get_logger().warn(
                "Bimanual task-preserving anti-sway QP selected; disabling the "
                "legacy right-arm input shaper and predictive MPC."
            )
            self.antisway_input_shaper_enabled = False
            self.antisway_predictive_enabled = False
        self.antisway_input_shaper: DualModalInputShaper | None = None
        if self.antisway_enabled and self.antisway_input_shaper_enabled:
            try:
                self.antisway_input_shaper = DualModalInputShaper(
                    rate_hz=self.rate_hz,
                    frequencies_hz=self.antisway_frequencies,
                    coupling=self.antisway_coupling,
                    quality_factors=np.asarray(
                        self.get_parameter(
                            "antisway_input_shaper_quality_factors"
                        ).value,
                        dtype=float,
                    ),
                    strengths=np.asarray(
                        self.get_parameter("antisway_input_shaper_strengths").value,
                        dtype=float,
                    ),
                    max_correction=max(
                        0.0,
                        float(
                            self.get_parameter(
                                "antisway_input_shaper_max_correction"
                            ).value
                        ),
                    ),
                    max_correction_rate=max(
                        0.0,
                        float(
                            self.get_parameter(
                                "antisway_input_shaper_max_correction_rate"
                            ).value
                        ),
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(
                    f"Failed to initialize dual-modal input shaper: {exc}"
                )
                self.antisway_input_shaper_enabled = False
        if self.antisway_input_shaper_enabled and self.antisway_predictive_enabled:
            self.get_logger().warn(
                "Both input shaper and predictive MPC were requested; disabling "
                "predictive MPC so only one anti-sway reference modifier is active."
            )
            self.antisway_predictive_enabled = False
        self.antisway_mpc: PredictiveAntiswayMpc | None = None
        if self.antisway_enabled and self.antisway_predictive_enabled:
            try:
                self.antisway_mpc = PredictiveAntiswayMpc(
                    rate_hz=self.rate_hz,
                    horizon_steps=self.antisway_horizon_steps,
                    modal_a=self.antisway_a,
                    modal_b=self.antisway_b,
                    frequencies_hz=self.antisway_frequencies,
                    roll_weight=self.antisway_roll_weight,
                    yaw_weight=self.antisway_yaw_weight,
                    tracking_weight=self.antisway_tracking_weight,
                    velocity_weight=max(
                        0.0,
                        float(self.get_parameter("antisway_velocity_weight").value),
                    ),
                    acceleration_weight=max(
                        1e-8,
                        float(
                            self.get_parameter("antisway_mpc_acceleration_weight").value
                        ),
                    ),
                    acceleration_change_weight=max(
                        0.0,
                        float(
                            self.get_parameter(
                                "antisway_mpc_acceleration_change_weight"
                            ).value
                        ),
                    ),
                    terminal_weight_scale=max(
                        1.0,
                        float(self.get_parameter("antisway_terminal_weight_scale").value),
                    ),
                    max_acceleration=max(
                        0.1,
                        float(self.get_parameter("antisway_max_acceleration").value),
                    ),
                    max_velocity=max(
                        0.05,
                        float(self.get_parameter("antisway_max_velocity").value),
                    ),
                    max_iterations=max(
                        20,
                        int(self.get_parameter("antisway_mpc_max_iterations").value),
                    ),
                    eps_abs=max(
                        1e-7,
                        float(self.get_parameter("antisway_mpc_eps_abs").value),
                    ),
                    eps_rel=max(
                        1e-7,
                        float(self.get_parameter("antisway_mpc_eps_rel").value),
                    ),
                    time_limit_s=max(
                        0.0,
                        float(self.get_parameter("antisway_mpc_time_limit_s").value),
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(
                    f"Failed to initialize predictive anti-sway MPC: {exc}"
                )
                self.antisway_predictive_enabled = False

        self._temp_urdf_path: str | None = None
        if not self.urdf_path:
            self.urdf_path = self._write_robot_description_to_temp_urdf()
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        self.baseline_data = self.model.createData()
        self.neutral_q = pin.neutral(self.model)
        self.lower = np.asarray(self.model.lowerPositionLimit, dtype=float).copy()
        self.upper = np.asarray(self.model.upperPositionLimit, dtype=float).copy()
        self.joint_q_indices = self._joint_q_indices(self.joint_names)
        self.joint_v_indices = self._joint_v_indices(self.joint_names)
        self.left_slice = slice(0, len(self.left_joint_names))
        self.right_slice = slice(len(self.left_joint_names), len(self.joint_names))
        self.spheres = self._default_bimanual_spheres()
        left_all_indices = [i for i, sphere in enumerate(self.spheres) if "_left_" in sphere.frame]
        right_all_indices = [i for i, sphere in enumerate(self.spheres) if "_right_" in sphere.frame]
        self.esdf_sphere_indices = (
            left_all_indices[self.esdf_skip_proximal_spheres :]
            + right_all_indices[self.esdf_skip_proximal_spheres :]
        )
        self.esdf_sphere_index_set = set(self.esdf_sphere_indices)
        self.left_sphere_indices = left_all_indices[self.inter_arm_skip_proximal_spheres :]
        self.right_sphere_indices = right_all_indices[self.inter_arm_skip_proximal_spheres :]
        self.sphere_frame_ids = [self.model.getFrameId(sphere.frame) for sphere in self.spheres]
        self.left_tcp_frame_id = self._optional_frame_id("openarmx_left_hand_tcp")
        self.right_tcp_frame_id = self._optional_frame_id("openarmx_right_hand_tcp")
        self.filtered_clearances: list[float | None] = [None] * len(self.spheres)
        self.assisted_grasp_enabled = bool(self.get_parameter("assisted_grasp_enabled").value)
        self.assisted_grasp_activation_distance = max(
            0.005,
            float(self.get_parameter("assisted_grasp_activation_distance").value),
        )
        self.assisted_grasp_ramp_duration = max(
            0.05,
            float(self.get_parameter("assisted_grasp_ramp_duration").value),
        )
        self.assisted_grasp_max_cartesian_step = max(
            0.0,
            float(self.get_parameter("assisted_grasp_max_cartesian_step").value),
        )
        self.assisted_grasp_require_open_gripper = bool(
            self.get_parameter("assisted_grasp_require_open_gripper").value
        )
        self.assisted_grasp_gripper_open_fraction = float(
            np.clip(
                float(self.get_parameter("assisted_grasp_gripper_open_fraction").value),
                0.0,
                1.0,
            )
        )
        self.assisted_grasp_gripper_close_fraction = float(
            np.clip(
                float(self.get_parameter("assisted_grasp_gripper_close_fraction").value),
                0.0,
                1.0,
            )
        )
        self.assisted_grasp_gripper_min = float(
            self.get_parameter("assisted_grasp_gripper_min").value
        )
        self.assisted_grasp_gripper_max = float(
            self.get_parameter("assisted_grasp_gripper_max").value
        )

        self.q_model_current = self.neutral_q.copy()
        self.have_joint_state = False
        self.last_joint_state_time = None
        self.latest_left_baseline: np.ndarray | None = None
        self.latest_right_baseline: np.ndarray | None = None
        self.previous_baseline = {"left": None, "right": None}
        self.previous_baseline_time = {"left": None, "right": None}
        self.baseline_velocity = {
            "left": np.zeros(len(self.left_joint_names), dtype=float),
            "right": np.zeros(len(self.right_joint_names), dtype=float),
        }
        self.latest_left_extra: list[float] = []
        self.latest_right_extra: list[float] = []
        self.last_command: np.ndarray | None = None
        self.last_command_step: np.ndarray | None = None
        self.last_command_step_change: np.ndarray | None = None
        self.previous_cbf_master_baseline: np.ndarray | None = None
        self.cbf_master_step = np.zeros(len(BIMANUAL_JOINT_NAMES), dtype=float)
        self.cbf_master_motion_active = {"left": False, "right": False}
        self.cbf_master_motion_scale = {"left": 1.0, "right": 1.0}
        self.cbf_master_motion_catchup_step = {"left": 0.0, "right": 0.0}
        self.cbf_safe_baseline_catchup_active = {"left": False, "right": False}
        self.antisway_modal_state: np.ndarray | None = None
        self.antisway_modal_state_time = None
        self.antisway_gru_prediction: np.ndarray | None = None
        self.antisway_gru_prediction_time = None
        self.antisway_gru_equivalent_speed = np.full(2, np.nan, dtype=float)
        self.antisway_gru_strength_scale = np.ones(2, dtype=float)
        self.antisway_gru_prediction_valid = False
        self.antisway_observer_valid = False
        self.antisway_observer_nis = float("nan")
        self.antisway_raw_confidence = 0.0
        self.antisway_confidence = 0.0
        self.antisway_observer_gate_active = False
        self.antisway_bimanual_motion_gate_active = False
        self.antisway_cost_applied = False
        self.antisway_last_acceleration = np.zeros(3, dtype=float)
        self.antisway_last_predicted_energy = np.full(2, np.nan, dtype=float)
        self.antisway_predictive_applied = False
        self.antisway_mpc_solve_ms = 0.0
        self.antisway_mpc_iterations = 0
        self.antisway_mpc_status = "not_run"
        self.antisway_mpc_acceleration = np.zeros(3, dtype=float)
        self.antisway_mpc_reference_delta = np.zeros(3, dtype=float)
        self.antisway_mpc_reference_velocity = np.zeros(3, dtype=float)
        self.antisway_mpc_acceleration_error = np.full(3, np.nan, dtype=float)
        self.antisway_shaper_applied = False
        self.antisway_shaper_modal_input = np.zeros(2, dtype=float)
        self.antisway_shaper_modal_output = np.zeros(2, dtype=float)
        self.antisway_shaper_correction = np.zeros(3, dtype=float)
        self.antisway_bimanual_applied = False
        self.antisway_bimanual_status = "not_run"
        self.antisway_bimanual_solve_ms = 0.0
        self.antisway_bimanual_iterations = 0
        self.antisway_bimanual_nullity = 0
        self.antisway_bimanual_task_residual = 0.0
        self.antisway_bimanual_nominal_acceleration = np.zeros(
            len(self.joint_names),
            dtype=float,
        )
        self.antisway_bimanual_previous_velocity: np.ndarray | None = None
        self.antisway_bimanual_acceleration_correction = np.zeros(
            len(self.joint_names),
            dtype=float,
        )
        self.antisway_bimanual_position_correction = np.zeros(
            len(self.joint_names),
            dtype=float,
        )
        self.antisway_bimanual_modal_target = np.zeros(2, dtype=float)
        self.antisway_bimanual_modal_prediction = np.zeros(2, dtype=float)
        self.avoidance_release_progress = 1.0
        self.tangent_escape_sign: dict[str, float] = {"left": 0.0, "right": 0.0}
        self.hand_bypass_active = {"left": False, "right": False}
        self.wrist_tcp_preservation_active = {"left": False, "right": False}
        self.grasp_targets: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self.grasp_target_times = {"left": None, "right": None}
        self.assisted_grasp_start_times = {"left": None, "right": None}
        self.assisted_grasp_active: dict[str, bool] = {"left": False, "right": False}
        self.assisted_grasp_alpha: dict[str, float] = {"left": 0.0, "right": 0.0}
        self.assisted_grasp_latched_reference: dict[str, np.ndarray | None] = {
            "left": None,
            "right": None,
        }
        self.assisted_grasp_gripper_fraction: dict[str, float | None] = {
            "left": None,
            "right": None,
        }
        self.esdf_grid: EsdfGrid | None = None
        self.esdf_pending = False
        self.last_esdf_request_time = self.get_clock().now()
        self.last_log_time = self.get_clock().now()
        self.min_clearance = float("nan")
        self.min_inter_arm_clearance = float("nan")
        self.baseline_error_norm = 0.0
        self.active_constraints = 0
        self.active_esdf_constraints = 0
        self.active_inter_arm_constraints = 0
        self.hold_due_to_invalid_clearance = False
        self.hold_due_to_avoidance_latch = False
        self.avoidance_latched = False
        self.avoidance_release_counter = 0
        self.baseline_min_clearance = float("nan")
        self.baseline_left_min_clearance = float("nan")
        self.baseline_right_min_clearance = float("nan")
        self.sampled_spheres = 0
        self.esdf_grid_shape = "none"
        self.esdf_observed_count = 0
        self.rejected_esdf_samples = 0
        self.cbf_qp_success = False
        self.cbf_qp_status = "not_run"
        self.cbf_qp_slack_max = 0.0
        self.cbf_qp_solve_ms = 0.0
        self.cbf_qp_iterations = 0
        self.cbf_qp_backend_used = "none"
        self.min_clearance_sphere = "none"
        self.min_clearance_position = np.array([np.nan, np.nan, np.nan], dtype=float)
        self.left_min_clearance = float("nan")
        self.right_min_clearance = float("nan")
        self.left_min_clearance_sphere = "none"
        self.right_min_clearance_sphere = "none"
        self.rollout_active = False
        self.rollout_candidate_count = 0
        self.rollout_selected = "baseline"
        self.rollout_best_cost = float("nan")
        self.rollout_baseline_cost = float("nan")
        self.cable_capsules: list[CableCapsule] = []
        self.ground_truth_cable_capsules: list[CableCapsule] = []
        self.last_cable_capsule_time = None
        self.last_protected_cable_time = None
        self.last_ground_truth_cable_time = None
        self.protected_cable_source_stamp = None
        self.ground_truth_cable_source_stamp = None
        self.ground_truth_min_clearance = float("nan")
        self.untangle_mode = False
        self.current_obstacle_source = "none"
        self.hold_due_to_missing_obstacle_source = False
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.esdf_client = None
        if self._may_use_esdf():
            self._connect_esdf_client()

        self.create_subscription(JointState, self.joint_states_topic, self._joint_state_cb, 20)
        self.create_subscription(Float64MultiArray, self.left_input_command_topic, self._left_command_cb, 10)
        self.create_subscription(Float64MultiArray, self.right_input_command_topic, self._right_command_cb, 10)
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("left_grasp_target_topic").value),
            lambda msg: self._grasp_target_cb("left", msg),
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("right_grasp_target_topic").value),
            lambda msg: self._grasp_target_cb("right", msg),
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("target_locked_topic").value),
            self._target_locked_cb,
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self.create_subscription(
            MarkerArray,
            self.cable_capsule_topic,
            self._cable_capsules_cb,
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self.create_subscription(
            ProtectedCableArray,
            self.protected_cable_topic,
            self._protected_cables_cb,
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        self.create_subscription(
            ProtectedCableArray,
            self.ground_truth_cable_topic,
            self._ground_truth_cables_cb,
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("untangle_mode_topic").value),
            self._untangle_mode_cb,
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("antisway_modal_state_topic").value),
            self._antisway_modal_state_cb,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("antisway_gru_prediction_topic").value),
            self._antisway_gru_prediction_cb,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("antisway_observer_diagnostics_topic").value),
            self._antisway_observer_diagnostics_cb,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("antisway_observer_valid_topic").value),
            self._antisway_observer_valid_cb,
            10,
        )
        self.left_command_pub = self.create_publisher(Float64MultiArray, self.left_output_command_topic, 10)
        self.right_command_pub = self.create_publisher(Float64MultiArray, self.right_output_command_topic, 10)
        self.marker_pub = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("marker_topic").value),
            10,
        )
        self.min_clearance_pub = self.create_publisher(
            Float32,
            str(self.get_parameter("min_clearance_topic").value),
            10,
        )
        self.left_min_clearance_pub = self.create_publisher(
            Float32,
            str(self.get_parameter("left_min_clearance_topic").value),
            10,
        )
        self.right_min_clearance_pub = self.create_publisher(
            Float32,
            str(self.get_parameter("right_min_clearance_topic").value),
            10,
        )
        self.status_pub = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            10,
        )
        self.structured_status_pub = self.create_publisher(
            SafetyStatus,
            str(self.get_parameter("structured_status_topic").value),
            10,
        )
        self.antisway_diagnostics_pub = self.create_publisher(
            Float64MultiArray,
            str(self.get_parameter("antisway_diagnostics_topic").value),
            10,
        )
        self.timer = self.create_timer(1.0 / self.rate_hz, self._timer_cb)

        self.get_logger().info(f"URDF: {self.urdf_path}")
        self.get_logger().info(f"left joints: {self.left_joint_names}")
        self.get_logger().info(f"right joints: {self.right_joint_names}")
        self.get_logger().info(
            "bimanual filter: "
            f"{self.left_input_command_topic}, {self.right_input_command_topic} -> "
            f"{self.left_output_command_topic}, {self.right_output_command_topic}"
        )
        self.get_logger().info(
            "avoidance: "
            f"solver={self.avoidance_solver}, obstacle_source={self.obstacle_source}, "
            f"monitor_only={self.monitor_only}, "
            f"safety_margin={self.safety_margin:.3f}, "
            f"activation_margin={self.activation_margin:.3f}, spheres={len(self.spheres)}, "
            f"target_clearance={self._target_clearance():.3f}, "
            f"clearance_filter_alpha={self.clearance_filter_alpha:.2f}, "
            f"baseline_weight={self.baseline_weight:.3f}, "
            f"latched_baseline_scale={self.latched_baseline_weight_scale:.3f}, "
            f"collision_model={self.collision_model}, "
            f"capsule_spacing={self.capsule_sample_spacing:.3f}, "
            f"capsule_samples_per_link={self.capsule_samples_per_link}, "
            f"ee_padding={self.end_effector_collision_padding:.3f}, "
            f"esdf_spheres={len(self.esdf_sphere_indices)}, "
            f"esdf_skip_proximal={self.esdf_skip_proximal_spheres}, "
            f"prefer_z={self.prefer_z_avoidance}, z_grad_min={self.z_gradient_min_abs:.3f}, "
            f"xy_follow_weight={self.xy_follow_weight:.3f}, "
            f"downward_bias_weight={self.downward_bias_weight:.3f}, "
            f"downward_bias_step={self.downward_bias_step:.3f}, "
            f"down_tangent={self.downward_tangent_bias_enabled}, "
            f"untangle_topic={self.get_parameter('untangle_mode_topic').value}, "
            f"untangle_tangent_weight={self.untangle_tangent_weight:.2f}, "
            f"rollout={self.predictive_rollout_enabled}, "
            f"rollout_horizon={self.rollout_horizon_steps}, "
            f"cbf_gain={self.cbf_gain:.3f}, cbf_slack_weight={self.cbf_slack_weight:.1f}, "
            f"cbf_backend={self.cbf_qp_backend}, "
            f"max_jerk={self.max_command_jerk:.2f}, "
            f"release_ramp={self.avoidance_release_ramp_duration:.2f}s, "
            f"avoidance_latch={self.avoidance_latch_enabled}, "
            f"release_margin={self.avoidance_release_margin:.3f}, "
            f"inter_arm={self.enable_inter_arm_collision}, "
            f"inter_margin={self.inter_arm_safety_margin:.3f}, "
            f"inter_activation={self.inter_arm_activation_margin:.3f}, "
            f"inter_skip_proximal={self.inter_arm_skip_proximal_spheres}, "
            f"min_valid_esdf_distance={self.min_valid_esdf_distance:.3f}"
        )
        self.get_logger().info(
            "anti-sway: "
            f"enabled={self.antisway_enabled}, monitor_only={self.antisway_monitor_only}, "
            f"modal_weight={self.antisway_modal_weight:.4f}, "
            f"acceleration_change_weight={self.antisway_acceleration_change_weight:.5f}, "
            f"observer_timeout={self.antisway_observer_timeout_s:.3f}s, "
            f"input_shaper={self.antisway_input_shaper_enabled}, "
            f"predictive={self.antisway_predictive_enabled}, "
            f"bimanual_task_qp={self.antisway_bimanual_task_qp_enabled}, "
            f"horizon={self.antisway_horizon_steps}, "
            f"roll/yaw_weight={self.antisway_roll_weight:.2f}/{self.antisway_yaw_weight:.2f}, "
            f"baseline_velocity_alpha={self.antisway_baseline_velocity_alpha:.2f}, "
            f"max_reference_deviation={self.antisway_max_reference_deviation:.3f}, "
            f"motion_gate={self.antisway_motion_gate_enabled}/"
            f"{self.antisway_motion_gate_velocity:.3f}/"
            f"{self.antisway_motion_gate_release_velocity:.3f}rad/s, "
            f"confidence_tau={self.antisway_confidence_rise_time_s:.2f}/"
            f"{self.antisway_confidence_fall_time_s:.2f}s, "
            f"confidence_gate={self.antisway_confidence_enter_threshold:.2f}/"
            f"{self.antisway_confidence_exit_threshold:.2f}, "
            f"bimanual_max_position_correction="
            f"{self.antisway_bimanual_max_position_correction:.4f}rad"
        )
        self.get_logger().info(
            "command governor: "
            f"enabled={self.command_governor_enabled}, "
            f"velocity={self.command_governor.max_velocity:.2f}rad/s, "
            f"acceleration={self.command_governor.max_acceleration:.2f}rad/s^2, "
            f"deceleration={self.command_governor.max_deceleration:.2f}rad/s^2, "
            f"jerk={self.command_governor.max_jerk:.2f}rad/s^3, "
            f"position_gain={self.command_governor.position_gain:.2f}/s"
        )
        if self.obstacle_source in ("cable_capsules", "hybrid"):
            self.get_logger().info(
                "Cable obstacle source: "
                f"estimate={self.protected_cable_topic}, truth={self.ground_truth_cable_topic}, "
                f"legacy={self.cable_capsule_topic}, timeout={self.cable_capsule_timeout_s:.2f}s, "
                f"timeout_behavior={self.cable_capsule_timeout_behavior}, "
                f"radius_scale={self.cable_capsule_radius_scale:.2f}, "
                f"padding={self.cable_capsule_padding:.3f}"
            )
        if self.esdf_extension_offsets:
            self.get_logger().info(
                "ESDF virtual extension: "
                f"obstacle_direction={np.array2string(self.esdf_extension_direction, precision=3)}, "
                f"length={self.esdf_extension_length:.3f}, step={self.esdf_extension_step:.3f}, "
                f"samples={len(self.esdf_extension_offsets) + 1}"
            )

    def _parse_float_list_param(self, value, name: str) -> list[float]:
        if isinstance(value, str):
            text = value.strip().strip("[]")
            values = [float(item.strip()) for item in text.replace(";", ",").split(",") if item.strip()]
        else:
            values = [float(item) for item in value]
        for item in values:
            if not np.isfinite(item):
                raise RuntimeError(f"{name} contains a non-finite value: {item}")
        return values

    def _parse_vector3_param(self, value, name: str) -> np.ndarray:
        if isinstance(value, str):
            text = value.strip().strip("[]")
            values = [float(item.strip()) for item in text.replace(";", ",").split(",") if item.strip()]
        else:
            values = [float(item) for item in value]
        if len(values) != 3:
            raise RuntimeError(f"{name} must contain 3 values, got {len(values)}")
        return np.asarray(values, dtype=float)

    def _build_esdf_extension_offsets(self) -> list[np.ndarray]:
        if (
            not self.enable_esdf_virtual_extension
            or self.esdf_extension_length <= 0.0
            or self.esdf_extension_step <= 0.0
        ):
            return []
        distances = np.arange(
            self.esdf_extension_step,
            self.esdf_extension_length + 0.5 * self.esdf_extension_step,
            self.esdf_extension_step,
            dtype=float,
        )
        return [(-distance * self.esdf_extension_direction).astype(float) for distance in distances]

    def _write_robot_description_to_temp_urdf(self) -> str:
        client = self.create_client(GetParameters, f"{self.robot_description_node}/get_parameters")
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(
                f"Service not available: {self.robot_description_node}/get_parameters; "
                "pass urdf_path explicitly or start robot_state_publisher first."
            )
        req = GetParameters.Request()
        req.names = ["robot_description"]
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        response = future.result()
        if response is None or not response.values or not response.values[0].string_value:
            raise RuntimeError("Failed to read robot_description from robot_state_publisher.")

        fd, path = tempfile.mkstemp(prefix="openarmx_bimanual_esdf_filter_", suffix=".urdf")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as file_obj:
            file_obj.write(response.values[0].string_value)
        self._temp_urdf_path = path
        return path

    def _joint_q_indices(self, joint_names: list[str]) -> list[int]:
        indices = []
        for name in joint_names:
            if not self.model.existJointName(name):
                raise RuntimeError(f"Joint {name!r} not found in URDF model")
            indices.append(int(self.model.joints[self.model.getJointId(name)].idx_q))
        return indices

    def _joint_v_indices(self, joint_names: list[str]) -> list[int]:
        return [int(self.model.joints[self.model.getJointId(name)].idx_v) for name in joint_names]

    def _optional_frame_id(self, frame_name: str) -> int | None:
        if not self.model.existFrame(frame_name):
            self.get_logger().warn(f"Frame {frame_name!r} not found; disabling rollout offsets for it.")
            return None
        return int(self.model.getFrameId(frame_name))

    def _default_bimanual_spheres(self) -> list[CollisionSphere]:
        if self.collision_model == "sphere":
            raw_one_arm = [
                ("link1", [0.0, 0.0, 0.03], 0.055),
                ("link2", [0.0, 0.0, 0.02], 0.050),
                ("link2", [0.0, 0.0, 0.075], 0.050),
                ("link3", [0.0, 0.0, 0.04], 0.055),
                ("link3", [0.0, 0.0, 0.10], 0.055),
                ("link4", [0.0, -0.015, 0.035], 0.035),
                ("link4", [0.0, -0.025, 0.095], 0.035),
                ("link5", [0.0, 0.0, 0.03], 0.035),
                ("link5", [0.0, 0.0, 0.080], 0.035),
                ("link6", [-0.020, 0.0, 0.0], 0.035),
                ("link7", [0.0, 0.0, 0.025], 0.035),
                ("hand", [0.0, 0.0, 0.025], 0.055),
                ("hand_tcp", [0.0, 0.0, 0.0], 0.020),
            ]
            return self._build_sphere_collision_points(raw_one_arm)

        raw_one_arm_capsules = [
            ("link1", [0.0, 0.0, 0.005], [0.0, 0.0, 0.065], 0.052),
            ("link2", [0.0, 0.0, 0.005], [0.0, 0.0, 0.105], 0.048),
            ("link3", [0.0, 0.0, 0.020], [0.0, 0.0, 0.125], 0.052),
            ("link4", [0.0, -0.015, 0.020], [0.0, -0.028, 0.115], 0.034),
            ("link5", [0.0, 0.0, 0.015], [0.0, 0.0, 0.095], 0.034),
            ("link6", [-0.035, 0.0, 0.0], [0.020, 0.0, 0.0], 0.034),
            ("link7", [0.0, 0.0, 0.005], [0.0, 0.0, 0.050], 0.032),
            (
                "hand",
                [0.0, 0.0, 0.000],
                [0.0, 0.0, 0.060],
                0.048 + self.end_effector_collision_padding,
            ),
        ]
        spheres = self._build_capsule_collision_points(raw_one_arm_capsules)
        if not spheres:
            raise RuntimeError("No valid capsule collision samples were configured.")
        return spheres

    def _build_sphere_collision_points(
        self,
        raw_one_arm: list[tuple[str, list[float], float]],
    ) -> list[CollisionSphere]:
        spheres: list[CollisionSphere] = []
        for side in ("left", "right"):
            for suffix, center, radius in raw_one_arm:
                frame = f"openarmx_{side}_{suffix}"
                if not self.model.existFrame(frame):
                    self.get_logger().warn(f"Skipping collision sphere on missing frame {frame!r}")
                    continue
                effective_radius = float(radius)
                if suffix.startswith("hand"):
                    effective_radius += self.end_effector_collision_padding
                spheres.append(
                    CollisionSphere(frame, np.asarray(center, dtype=float), effective_radius)
                )
        if not spheres:
            raise RuntimeError("No valid collision spheres were configured.")
        return spheres

    def _build_capsule_collision_points(
        self,
        raw_one_arm_capsules: list[tuple[str, list[float], list[float], float]],
    ) -> list[CollisionSphere]:
        spheres: list[CollisionSphere] = []
        for side in ("left", "right"):
            for suffix, start, end, radius in raw_one_arm_capsules:
                frame = f"openarmx_{side}_{suffix}"
                if not self.model.existFrame(frame):
                    self.get_logger().warn(f"Skipping collision capsule on missing frame {frame!r}")
                    continue
                start_np = np.asarray(start, dtype=float).reshape(3)
                end_np = np.asarray(end, dtype=float).reshape(3)
                sample_count = self.capsule_samples_per_link
                for sample_idx in range(sample_count):
                    alpha = 0.0 if sample_count == 1 else sample_idx / float(sample_count - 1)
                    center = (1.0 - alpha) * start_np + alpha * end_np
                    spheres.append(CollisionSphere(frame, center.astype(float), float(radius)))
        return spheres

    def _connect_esdf_client(self) -> None:
        service_name = self.esdf_service_name or self._discover_esdf_service()
        if not service_name:
            service_name = "/nvblox_node/get_esdf_and_gradients"
            self.get_logger().warn(
                f"No EsdfAndGradients service discovered yet; using default {service_name}"
            )
        self.esdf_service_name = service_name
        self.esdf_client = self.create_client(EsdfAndGradients, service_name)
        self.get_logger().info(f"ESDF service: {service_name}")

    def _may_use_esdf(self) -> bool:
        return (
            self.obstacle_source in ("esdf", "hybrid")
            or self.cable_capsule_timeout_behavior == "fallback_esdf"
        )

    def _discover_esdf_service(self) -> str:
        for name, types in self.get_service_names_and_types():
            if "nvblox_msgs/srv/EsdfAndGradients" in types:
                return name
        return ""

    def _joint_state_cb(self, msg: JointState) -> None:
        name_to_pos = {name: pos for name, pos in zip(msg.name, msg.position)}
        updated = False
        for name in msg.name:
            if name in name_to_pos and self.model.existJointName(name):
                joint = self.model.joints[self.model.getJointId(name)]
                if joint.nq == 1:
                    self.q_model_current[joint.idx_q] = float(name_to_pos[name])
                    updated = True
        if updated:
            self.have_joint_state = True
            self.last_joint_state_time = self.get_clock().now()

    def _left_command_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < len(self.left_joint_names):
            self.get_logger().warn(
                f"Ignoring left command with {len(msg.data)} positions; need {len(self.left_joint_names)}"
            )
            return
        self.latest_left_baseline = self._update_baseline_velocity(
            "left",
            np.asarray(msg.data[: len(self.left_joint_names)], dtype=float),
        )
        self.latest_left_extra = [float(v) for v in msg.data[len(self.left_joint_names) :]]
        self.assisted_grasp_gripper_fraction["left"] = self._gripper_fraction_from_extra(
            self.latest_left_extra
        )

    def _right_command_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < len(self.right_joint_names):
            self.get_logger().warn(
                f"Ignoring right command with {len(msg.data)} positions; need {len(self.right_joint_names)}"
            )
            return
        self.latest_right_baseline = self._update_baseline_velocity(
            "right",
            np.asarray(msg.data[: len(self.right_joint_names)], dtype=float),
        )
        self.latest_right_extra = [float(v) for v in msg.data[len(self.right_joint_names) :]]
        self.assisted_grasp_gripper_fraction["right"] = self._gripper_fraction_from_extra(
            self.latest_right_extra
        )

    def _update_baseline_velocity(
        self,
        side: str,
        positions: np.ndarray,
    ) -> np.ndarray:
        positions = np.asarray(positions, dtype=float).copy()
        now = self.get_clock().now()
        previous = self.previous_baseline[side]
        previous_time = self.previous_baseline_time[side]
        if previous is not None and previous_time is not None:
            dt = (now - previous_time).nanoseconds * 1e-9
            if 1e-3 <= dt <= 0.25:
                raw_velocity = np.clip(
                    (positions - previous) / dt,
                    -self.antisway_baseline_velocity_limit,
                    self.antisway_baseline_velocity_limit,
                )
                raw_velocity[
                    np.abs(raw_velocity) < self.antisway_baseline_velocity_deadband
                ] = 0.0
                alpha = self.antisway_baseline_velocity_alpha
                self.baseline_velocity[side] = (
                    alpha * raw_velocity
                    + (1.0 - alpha) * self.baseline_velocity[side]
                )
            else:
                self.baseline_velocity[side].fill(0.0)
        self.previous_baseline[side] = positions.copy()
        self.previous_baseline_time[side] = now
        return positions

    def _antisway_modal_state_cb(self, msg: Float64MultiArray) -> None:
        state = np.asarray(msg.data, dtype=float)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            return
        self.antisway_modal_state = state
        self.antisway_modal_state_time = self.get_clock().now()

    def _antisway_gru_prediction_cb(self, msg: Float64MultiArray) -> None:
        state = np.asarray(msg.data, dtype=float)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            return
        self.antisway_gru_prediction = state
        self.antisway_gru_prediction_time = self.get_clock().now()

    def _antisway_observer_diagnostics_cb(self, msg: Float64MultiArray) -> None:
        if not msg.data:
            return
        nis = float(msg.data[-1])
        self.antisway_observer_nis = nis if np.isfinite(nis) else float("nan")

    def _antisway_observer_valid_cb(self, msg: Bool) -> None:
        self.antisway_observer_valid = bool(msg.data)

    def _gripper_fraction_from_extra(self, extra: list[float]) -> float | None:
        if not extra:
            return None
        value = float(extra[0])
        if not np.isfinite(value):
            return None
        span = self.assisted_grasp_gripper_max - self.assisted_grasp_gripper_min
        if abs(span) < 1e-9:
            return None
        return float(np.clip((value - self.assisted_grasp_gripper_min) / span, 0.0, 1.0))

    def _grasp_target_cb(self, side: str, msg: PoseStamped) -> None:
        target = self._pose_position_to_global(msg)
        if target is None:
            return
        previous_target = self.grasp_targets[side]
        self.grasp_targets[side] = target
        self.grasp_target_times[side] = self.get_clock().now()
        if (
            previous_target is not None
            and float(np.linalg.norm(target - previous_target)) <= 1e-4
        ):
            return
        self.assisted_grasp_start_times[side] = None
        self.assisted_grasp_active[side] = False
        self.assisted_grasp_alpha[side] = 0.0
        self.assisted_grasp_latched_reference[side] = None

    def _target_locked_cb(self, msg: Bool) -> None:
        if bool(msg.data):
            return
        self.grasp_targets = {"left": None, "right": None}
        self.grasp_target_times = {"left": None, "right": None}
        self.assisted_grasp_start_times = {"left": None, "right": None}
        self.assisted_grasp_active = {"left": False, "right": False}
        self.assisted_grasp_alpha = {"left": 0.0, "right": 0.0}
        self.assisted_grasp_latched_reference = {"left": None, "right": None}

    def _cable_capsules_cb(self, msg: MarkerArray) -> None:
        if not self.legacy_cable_marker_fallback:
            return
        if (
            self.last_protected_cable_time is not None
            and self._age_from_receipt(self.last_protected_cable_time)
            <= self.cable_capsule_timeout_s
        ):
            return
        capsules: list[CableCapsule] = []
        transform_cache: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
        for marker in msg.markers:
            if (
                marker.action != Marker.ADD
                or marker.type != Marker.CYLINDER
                or marker.ns != "cable_capsules"
            ):
                continue
            length = float(marker.scale.z)
            radius = 0.25 * float(marker.scale.x + marker.scale.y)
            if (
                not np.isfinite(length)
                or not np.isfinite(radius)
                or length <= 1e-4
                or radius <= 0.0
            ):
                continue
            center = np.array(
                [
                    marker.pose.position.x,
                    marker.pose.position.y,
                    marker.pose.position.z,
                ],
                dtype=float,
            )
            marker_rotation = self._quaternion_to_rotation(
                marker.pose.orientation.x,
                marker.pose.orientation.y,
                marker.pose.orientation.z,
                marker.pose.orientation.w,
            )
            source_frame = marker.header.frame_id.strip() or self.global_frame
            if source_frame == self.global_frame:
                global_center = center
                global_rotation = marker_rotation
            else:
                if source_frame not in transform_cache:
                    transform_cache[source_frame] = self._lookup_frame_transform(source_frame)
                frame_transform = transform_cache[source_frame]
                if frame_transform is None:
                    continue
                translation, rotation = frame_transform
                global_center = rotation @ center + translation
                global_rotation = rotation @ marker_rotation
            direction = global_rotation[:, 2]
            direction_norm = float(np.linalg.norm(direction))
            if not np.isfinite(direction_norm) or direction_norm < 1e-9:
                continue
            direction /= direction_norm
            half = 0.5 * length * direction
            effective_radius = (
                radius * self.cable_capsule_radius_scale + self.cable_capsule_padding
            )
            capsules.append(
                CableCapsule(
                    start=(global_center - half).astype(float),
                    end=(global_center + half).astype(float),
                    radius=float(effective_radius),
                )
            )
            if len(capsules) >= self.cable_capsule_max_count:
                break
        if capsules or self.cable_capsule_timeout_behavior != "use_last":
            self.cable_capsules = capsules
            self.last_cable_capsule_time = self.get_clock().now()
            self.protected_cable_source_stamp = None

    def _protected_cables_cb(self, msg: ProtectedCableArray) -> None:
        capsules = self._protected_cables_to_global(msg, apply_control_padding=True)
        if not capsules and self.cable_capsule_timeout_behavior == "use_last":
            return
        self.cable_capsules = capsules
        now = self.get_clock().now()
        self.last_cable_capsule_time = now
        self.last_protected_cable_time = now
        self.protected_cable_source_stamp = self._source_stamp(msg)

    def _ground_truth_cables_cb(self, msg: ProtectedCableArray) -> None:
        self.ground_truth_cable_capsules = self._protected_cables_to_global(
            msg, apply_control_padding=False
        )
        self.last_ground_truth_cable_time = self.get_clock().now()
        self.ground_truth_cable_source_stamp = self._source_stamp(msg)

    def _protected_cables_to_global(
        self,
        msg: ProtectedCableArray,
        apply_control_padding: bool,
    ) -> list[CableCapsule]:
        source_frame = msg.header.frame_id.strip() or self.global_frame
        frame_transform = None
        if source_frame != self.global_frame:
            frame_transform = self._lookup_frame_transform(source_frame)
            if frame_transform is None:
                return []

        capsules: list[CableCapsule] = []
        for cable in msg.cables[: self.cable_capsule_max_count]:
            start = np.array([cable.start.x, cable.start.y, cable.start.z], dtype=float)
            end = np.array([cable.end.x, cable.end.y, cable.end.z], dtype=float)
            radius = float(cable.radius)
            if (
                not np.all(np.isfinite(start))
                or not np.all(np.isfinite(end))
                or not np.isfinite(radius)
                or float(np.linalg.norm(end - start)) <= 1e-4
                or radius <= 0.0
            ):
                continue
            if frame_transform is not None:
                translation, rotation = frame_transform
                start = rotation @ start + translation
                end = rotation @ end + translation
            if apply_control_padding:
                uncertainty = max(0.0, float(cable.position_std)) + max(
                    0.0, float(cable.radius_std)
                )
                radius = (
                    radius * self.cable_capsule_radius_scale
                    + self.cable_capsule_padding
                    + self.cable_uncertainty_sigma_scale * uncertainty
                )
            capsules.append(
                CableCapsule(
                    start=start.astype(float),
                    end=end.astype(float),
                    radius=float(radius),
                    position_std=max(0.0, float(cable.position_std)),
                    radius_std=max(0.0, float(cable.radius_std)),
                    confidence=float(np.clip(cable.confidence, 0.0, 1.0)),
                    cable_id=str(cable.cable_id),
                )
            )
        return capsules

    @staticmethod
    def _source_stamp(msg: ProtectedCableArray):
        if msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0:
            return None
        return Time.from_msg(msg.header.stamp)

    def _age_from_receipt(self, receipt_time) -> float:
        if receipt_time is None:
            return float("inf")
        return max(0.0, (self.get_clock().now() - receipt_time).nanoseconds * 1e-9)

    def _antisway_state_age(self) -> float:
        return self._age_from_receipt(self.antisway_modal_state_time)

    def _antisway_gru_prediction_age(self) -> float:
        return self._age_from_receipt(self.antisway_gru_prediction_time)

    def _antisway_raw_observer_confidence(self) -> tuple[bool, float]:
        sensor_ready = bool(
            self.antisway_enabled
            and self.antisway_observer_valid
            and self.antisway_modal_state is not None
            and self._antisway_state_age() <= self.antisway_observer_timeout_s
        )
        if not sensor_ready:
            return False, 0.0
        nis = self.antisway_observer_nis
        if not np.isfinite(nis):
            confidence = 0.0
        elif nis <= self.antisway_nis_full_confidence:
            confidence = 1.0
        elif nis >= self.antisway_nis_zero_confidence:
            confidence = 0.0
        else:
            confidence = (
                self.antisway_nis_zero_confidence - nis
            ) / (
                self.antisway_nis_zero_confidence
                - self.antisway_nis_full_confidence
            )
        return True, float(np.clip(confidence, 0.0, 1.0))

    def _update_antisway_observer_confidence(self) -> None:
        sensor_ready, raw_confidence = self._antisway_raw_observer_confidence()
        self.antisway_raw_confidence = raw_confidence
        time_constant = (
            self.antisway_confidence_rise_time_s
            if raw_confidence > self.antisway_confidence
            else self.antisway_confidence_fall_time_s
        )
        if time_constant <= 1e-9:
            alpha = 1.0
        else:
            alpha = 1.0 - np.exp(-self._control_dt() / time_constant)
        self.antisway_confidence += alpha * (
            raw_confidence - self.antisway_confidence
        )
        self.antisway_confidence = float(
            np.clip(self.antisway_confidence, 0.0, 1.0)
        )

        if not sensor_ready:
            self.antisway_observer_gate_active = False
        elif self.antisway_observer_gate_active:
            if (
                self.antisway_confidence
                <= self.antisway_confidence_exit_threshold
            ):
                self.antisway_observer_gate_active = False
        elif (
            self.antisway_confidence
            >= self.antisway_confidence_enter_threshold
        ):
            self.antisway_observer_gate_active = True

    def _antisway_observer_ready(self) -> bool:
        return bool(
            self.antisway_enabled
            and self.antisway_observer_gate_active
        )

    def _untangle_mode_cb(self, msg: Bool) -> None:
        enabled = bool(msg.data)
        if enabled == self.untangle_mode:
            return
        self.untangle_mode = enabled
        if enabled:
            self.avoidance_latched = False
            self.avoidance_release_counter = 0
        self.get_logger().info(f"Untangle mode {'enabled' if enabled else 'disabled'}.")

    def _lookup_frame_transform(
        self,
        source_frame: str,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.global_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.02),
            ).transform
        except Exception as exc:
            self.get_logger().warn(
                f"Cable capsule TF unavailable {self.global_frame}<-{source_frame}: {exc}",
                throttle_duration_sec=2.0,
            )
            return None
        translation = np.array(
            [
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
            ],
            dtype=float,
        )
        rotation = self._quaternion_to_rotation(
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        )
        return translation, rotation

    def _pose_position_to_global(self, msg: PoseStamped) -> np.ndarray | None:
        source_frame = msg.header.frame_id.strip() or self.global_frame
        point = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=float,
        )
        if not np.all(np.isfinite(point)):
            return None
        if source_frame == self.global_frame:
            return point
        transform = self._lookup_frame_transform(source_frame)
        if transform is None:
            return None
        translation, rotation = transform
        return rotation @ point + translation

    @staticmethod
    def _quaternion_to_rotation(
        x: float,
        y: float,
        z: float,
        w: float,
    ) -> np.ndarray:
        quaternion = np.array([x, y, z, w], dtype=float)
        norm = float(np.linalg.norm(quaternion))
        if not np.isfinite(norm) or norm < 1e-9:
            return np.eye(3, dtype=float)
        x, y, z, w = quaternion / norm
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=float,
        )

    def _cable_capsule_age(self) -> float:
        if self.last_cable_capsule_time is None:
            return float("inf")
        receipt_age = self._age_from_receipt(self.last_cable_capsule_time)
        if self.protected_cable_source_stamp is None:
            return receipt_age
        source_age = max(
            0.0,
            (self.get_clock().now() - self.protected_cable_source_stamp).nanoseconds * 1e-9,
        )
        return max(receipt_age, source_age)

    def _ground_truth_cable_age(self) -> float:
        receipt_age = self._age_from_receipt(self.last_ground_truth_cable_time)
        if self.ground_truth_cable_source_stamp is None:
            return receipt_age
        source_age = max(
            0.0,
            (self.get_clock().now() - self.ground_truth_cable_source_stamp).nanoseconds * 1e-9,
        )
        return max(receipt_age, source_age)

    def _select_active_obstacle_source(self) -> str:
        cable_ready = (
            bool(self.cable_capsules)
            and (
                self.cable_capsule_timeout_behavior == "use_last"
                or self._cable_capsule_age() <= self.cable_capsule_timeout_s
            )
        )
        esdf_ready = self.esdf_grid is not None
        if self.obstacle_source == "esdf":
            return "esdf" if esdf_ready else "none"
        if self.obstacle_source == "cable_capsules":
            if cable_ready:
                return "cable_capsules"
            if self.cable_capsule_timeout_behavior == "fallback_esdf" and esdf_ready:
                return "esdf"
            return "none"
        if cable_ready and esdf_ready:
            return "hybrid"
        if cable_ready:
            return "cable_capsules"
        if esdf_ready:
            return "esdf"
        return "none"

    def _timer_cb(self) -> None:
        if self.latest_left_baseline is None or self.latest_right_baseline is None:
            return
        self._update_antisway_observer_confidence()
        self.antisway_cost_applied = False
        self.antisway_predictive_applied = False
        self.antisway_mpc_status = "not_run"
        self.antisway_mpc_solve_ms = 0.0
        self.antisway_mpc_iterations = 0
        self.antisway_mpc_acceleration.fill(0.0)
        self.antisway_mpc_reference_delta.fill(0.0)
        self.antisway_mpc_reference_velocity.fill(0.0)
        self.antisway_mpc_acceleration_error.fill(np.nan)
        self.antisway_shaper_applied = False
        self.antisway_shaper_modal_input.fill(0.0)
        self.antisway_shaper_modal_output.fill(0.0)
        self.antisway_shaper_correction.fill(0.0)
        self.antisway_bimanual_applied = False
        self.antisway_bimanual_status = "not_run"
        self.antisway_bimanual_solve_ms = 0.0
        self.antisway_bimanual_iterations = 0
        self.antisway_bimanual_nullity = 0
        self.antisway_bimanual_task_residual = 0.0
        self.antisway_bimanual_acceleration_correction.fill(0.0)
        self.antisway_bimanual_modal_target.fill(0.0)
        self.antisway_bimanual_modal_prediction.fill(0.0)

        q_baseline = self._clamp_bimanual(
            np.concatenate([self.latest_left_baseline, self.latest_right_baseline])
        )
        self._update_cbf_master_step(q_baseline)
        q_start = self._current_safe_start(q_baseline)
        sphere_states = self._compute_sphere_states(self._model_q_from_bimanual(q_start))
        if self._may_use_esdf():
            self._maybe_request_esdf(sphere_states)
        previous_obstacle_source = self.current_obstacle_source
        self.current_obstacle_source = self._select_active_obstacle_source()
        if self.current_obstacle_source != previous_obstacle_source:
            self.filtered_clearances = [None] * len(self.spheres)
        self.hold_due_to_missing_obstacle_source = (
            self.obstacle_source in ("cable_capsules", "hybrid")
            and self.current_obstacle_source == "none"
            and self.cable_capsule_timeout_behavior in ("use_last", "hold")
        )
        q_shaped = self._apply_antisway_input_shaper(q_baseline)
        q_reference = self._select_predictive_rollout(q_shaped, q_start)
        q_reference = self._apply_assisted_grasp_reference(q_reference, q_start)
        q_reference = self._apply_bimanual_task_antisway_reference(
            q_reference,
            q_start,
        )
        q_reference = self._apply_predictive_antisway_reference(q_reference)

        if self.hold_due_to_missing_obstacle_source:
            self.min_clearance = float("nan")
            self.left_min_clearance = float("nan")
            self.right_min_clearance = float("nan")
            self.min_clearance_sphere = "none"
            self.min_clearance_position = np.array([np.nan, np.nan, np.nan], dtype=float)
            self.left_min_clearance_sphere = "none"
            self.right_min_clearance_sphere = "none"
            self.baseline_min_clearance = float("nan")
            self.baseline_left_min_clearance = float("nan")
            self.baseline_right_min_clearance = float("nan")
            self.sampled_spheres = 0
            self.active_constraints = 0
            self.active_esdf_constraints = 0
            self.active_inter_arm_constraints = 0
            self.cbf_qp_success = False
            self.cbf_qp_status = "missing_obstacle_source"
            self.cbf_qp_slack_max = 0.0
            q_safe = q_start.copy()
        elif self.current_obstacle_source != "none" or self.enable_inter_arm_collision:
            q_safe = self._avoid(q_reference, q_start)
        else:
            q_safe = q_reference.copy()

        if self.active_constraints > 0:
            self.avoidance_release_progress = 0.0
            q_safe = self._smooth_safe_command(q_safe, q_start)
            q_safe = self._apply_cbf_master_motion_sync(q_safe, q_start)
            q_safe = self._limit_command_step(
                q_safe,
                q_baseline,
                enforce_acceleration=True,
            )
        elif (
            self.hold_due_to_invalid_clearance
            or self.hold_due_to_avoidance_latch
            or self.hold_due_to_missing_obstacle_source
        ):
            self.avoidance_release_progress = 0.0
            q_safe = q_start.copy()
            self._reset_command_governor()
        else:
            q_release = self._apply_avoidance_release_ramp(q_reference.copy())
            q_safe = self._limit_command_step(q_release, q_baseline)
        q_safe = self._clamp_bimanual(q_safe)

        q_out = q_baseline if self.monitor_only else q_safe
        self.baseline_error_norm = float(np.linalg.norm(q_baseline - q_out))
        previous_command = self.last_command
        self._publish_antisway_diagnostics(q_out)
        if previous_command is None:
            command_step = np.zeros_like(q_out)
            command_step_change = np.zeros_like(q_out)
        else:
            command_step = np.asarray(q_out - previous_command, dtype=float)
            previous_step = (
                self.last_command_step
                if self.last_command_step is not None
                else np.zeros_like(command_step)
            )
            command_step_change = command_step - previous_step
        self.last_command_step = command_step
        self.last_command_step_change = command_step_change
        self.last_command = q_out.copy()
        self._update_ground_truth_metrics(q_out)
        self._publish_commands(q_out)
        self._publish_debug(q_out)

    def _apply_bimanual_task_antisway_reference(
        self,
        q_reference: np.ndarray,
        q_start: np.ndarray,
    ) -> np.ndarray:
        nominal_velocity = np.concatenate(
            [self.baseline_velocity["left"], self.baseline_velocity["right"]]
        )
        dt = self._control_dt()
        if self.antisway_bimanual_previous_velocity is None:
            raw_acceleration = np.zeros_like(nominal_velocity)
        else:
            raw_acceleration = (
                nominal_velocity - self.antisway_bimanual_previous_velocity
            ) / dt
        self.antisway_bimanual_previous_velocity = nominal_velocity.copy()
        alpha = self.antisway_bimanual_acceleration_alpha
        self.antisway_bimanual_nominal_acceleration = (
            alpha * raw_acceleration
            + (1.0 - alpha) * self.antisway_bimanual_nominal_acceleration
        )

        if (
            not self.antisway_enabled
            or not self.antisway_bimanual_task_qp_enabled
            or self.antisway_bimanual_task_qp is None
        ):
            return self._release_bimanual_position_correction(q_reference)

        task_jacobian = self._bimanual_tcp_task_jacobian(q_start)
        if task_jacobian is None:
            self.antisway_bimanual_status = "missing_tcp_jacobian"
            return self._release_bimanual_position_correction(q_reference)

        motion_speed = float(
            np.linalg.norm(
                nominal_velocity[self.antisway_bimanual_modal_indices]
            )
        )
        observer_ready = (
            self._antisway_observer_ready()
            and self.antisway_modal_state is not None
        )
        if not self.antisway_motion_gate_enabled:
            self.antisway_bimanual_motion_gate_active = True
        elif self.antisway_bimanual_motion_gate_active:
            if motion_speed <= self.antisway_motion_gate_release_velocity:
                self.antisway_bimanual_motion_gate_active = False
        elif motion_speed >= self.antisway_motion_gate_velocity:
            self.antisway_bimanual_motion_gate_active = True
        motion_allowed = self.antisway_bimanual_motion_gate_active
        assisted_active = any(self.assisted_grasp_active.values())
        if not observer_ready or not motion_allowed or assisted_active:
            if assisted_active:
                self.antisway_bimanual_status = "assisted_grasp_bypass"
            elif not observer_ready:
                self.antisway_bimanual_status = "observer_bypass"
            else:
                self.antisway_bimanual_status = "stationary_bypass"
            return self._release_bimanual_position_correction(q_reference)

        omega = 2.0 * np.pi * self.antisway_frequencies
        modal_angle = np.asarray(
            [self.antisway_modal_state[0], self.antisway_modal_state[2]],
            dtype=float,
        )
        modal_rate = np.asarray(
            [self.antisway_modal_state[1], self.antisway_modal_state[3]],
            dtype=float,
        )
        target_modal_input = (
            -self.antisway_bimanual_position_feedback
            * omega
            * omega
            * modal_angle
            -2.0
            * self.antisway_bimanual_additional_damping
            * omega
            * modal_rate
        )
        result = self.antisway_bimanual_task_qp.solve(
            task_jacobian=task_jacobian,
            nominal_acceleration=self.antisway_bimanual_nominal_acceleration,
            target_modal_input=target_modal_input,
        )
        confidence = self.antisway_confidence
        acceleration_correction = (
            confidence * result.acceleration_correction
        )
        self.antisway_bimanual_acceleration_correction = (
            acceleration_correction
        )
        self.antisway_bimanual_modal_target = result.target_modal_input
        self.antisway_bimanual_modal_prediction = result.predicted_modal_input
        self.antisway_bimanual_task_residual = result.task_residual
        self.antisway_bimanual_nullity = result.nullity
        self.antisway_bimanual_solve_ms = result.solve_ms
        self.antisway_bimanual_iterations = result.iterations
        self.antisway_bimanual_status = result.status

        target_position_correction = (
            0.5
            * self.antisway_bimanual_position_horizon_s**2
            * acceleration_correction
        )
        self._update_bimanual_position_correction(
            target_position_correction,
        )
        if self.antisway_monitor_only:
            return q_reference
        corrected = q_reference + self.antisway_bimanual_position_correction
        self.antisway_bimanual_applied = bool(
            np.linalg.norm(self.antisway_bimanual_position_correction) > 1e-9
        )
        return self._clamp_bimanual(corrected)

    def _bimanual_tcp_task_jacobian(
        self,
        q_bimanual: np.ndarray,
    ) -> np.ndarray | None:
        if self.left_tcp_frame_id is None or self.right_tcp_frame_id is None:
            return None
        q_model = self._model_q_from_bimanual(q_bimanual)
        pin.forwardKinematics(self.model, self.data, q_model)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q_model)
        rows = []
        for frame_id in (self.left_tcp_frame_id, self.right_tcp_frame_id):
            jacobian = pin.getFrameJacobian(
                self.model,
                self.data,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:, self.joint_v_indices]
            if not np.all(np.isfinite(jacobian)):
                return None
            rows.append(jacobian)
        return np.vstack(rows)

    def _update_bimanual_position_correction(
        self,
        target: np.ndarray,
    ) -> None:
        target = np.asarray(target, dtype=float).copy()
        maximum = self.antisway_bimanual_max_position_correction
        peak = float(np.max(np.abs(target)))
        if maximum > 0.0 and peak > maximum:
            target *= maximum / peak
        if self.antisway_bimanual_max_correction_rate > 0.0:
            max_step = (
                self.antisway_bimanual_max_correction_rate
                * self._control_dt()
            )
            target = np.clip(
                target,
                self.antisway_bimanual_position_correction - max_step,
                self.antisway_bimanual_position_correction + max_step,
            )

        self.antisway_bimanual_position_correction = target

    def _release_bimanual_position_correction(
        self,
        q_reference: np.ndarray,
    ) -> np.ndarray:
        self._update_bimanual_position_correction(
            np.zeros(len(self.joint_names), dtype=float)
        )
        if self.antisway_monitor_only:
            return q_reference
        corrected = q_reference + self.antisway_bimanual_position_correction
        self.antisway_bimanual_applied = bool(
            np.linalg.norm(self.antisway_bimanual_position_correction) > 1e-9
        )
        return self._clamp_bimanual(corrected)

    def _apply_antisway_input_shaper(
        self,
        q_reference: np.ndarray,
    ) -> np.ndarray:
        if (
            not self.antisway_enabled
            or not self.antisway_input_shaper_enabled
            or self.antisway_input_shaper is None
        ):
            return q_reference
        modal_indices = np.asarray(
            [
                len(self.left_joint_names),
                len(self.left_joint_names) + 1,
                len(self.left_joint_names) + 3,
            ],
            dtype=int,
        )
        strength_scales = self._update_antisway_gru_strength_scale()
        result = self.antisway_input_shaper.process(
            q_reference[modal_indices],
            strength_scales=strength_scales,
        )
        self.antisway_shaper_modal_input = result.modal_input
        self.antisway_shaper_modal_output = result.modal_output
        self.antisway_shaper_correction = result.correction
        if self.antisway_monitor_only or self.assisted_grasp_active["right"]:
            return q_reference
        shaped_reference = q_reference.copy()
        shaped_reference[modal_indices] = result.positions
        self.antisway_shaper_applied = bool(
            np.linalg.norm(result.correction) > 1e-9
        )
        return self._clamp_bimanual(shaped_reference)

    def _update_antisway_gru_strength_scale(self) -> np.ndarray:
        if not self.antisway_gru_control_enabled:
            self.antisway_gru_prediction_valid = False
            self.antisway_gru_equivalent_speed.fill(np.nan)
            self.antisway_gru_strength_scale.fill(1.0)
            return self.antisway_gru_strength_scale.copy()

        prediction_valid = bool(
            self.antisway_gru_prediction is not None
            and self._antisway_gru_prediction_age()
            <= self.antisway_gru_prediction_timeout_s
        )
        self.antisway_gru_prediction_valid = prediction_valid
        if prediction_valid:
            state = self.antisway_gru_prediction
            omega = 2.0 * np.pi * self.antisway_frequencies
            equivalent_speed = np.asarray(
                [
                    np.hypot(state[1], omega[0] * state[0]),
                    np.hypot(state[3], omega[1] * state[2]),
                ],
                dtype=float,
            )
            normalized = np.clip(
                (
                    equivalent_speed
                    - self.antisway_gru_equivalent_speed_low
                )
                / (
                    self.antisway_gru_equivalent_speed_high
                    - self.antisway_gru_equivalent_speed_low
                ),
                0.0,
                1.0,
            )
            activation = normalized * normalized * (3.0 - 2.0 * normalized)
            target_scale = self.antisway_gru_min_strength_scale + (
                1.0 - self.antisway_gru_min_strength_scale
            ) * activation
            self.antisway_gru_equivalent_speed = equivalent_speed
        else:
            # A stale predictor must not disable the fixed input shaper.
            target_scale = np.ones(2, dtype=float)
            self.antisway_gru_equivalent_speed.fill(np.nan)

        dt = 1.0 / max(self.rate_hz, 1e-6)
        rising = target_scale > self.antisway_gru_strength_scale
        time_constants = np.where(
            rising,
            self.antisway_gru_strength_rise_time_s,
            self.antisway_gru_strength_fall_time_s,
        )
        alpha = np.ones(2, dtype=float)
        positive_tau = time_constants > 1e-9
        alpha[positive_tau] = np.clip(
            dt / time_constants[positive_tau],
            0.0,
            1.0,
        )
        self.antisway_gru_strength_scale += alpha * (
            target_scale - self.antisway_gru_strength_scale
        )
        self.antisway_gru_strength_scale = np.clip(
            self.antisway_gru_strength_scale,
            0.0,
            1.0,
        )
        return self.antisway_gru_strength_scale.copy()

    def _apply_predictive_antisway_reference(
        self,
        q_reference: np.ndarray,
    ) -> np.ndarray:
        if (
            not self.antisway_enabled
            or not self.antisway_predictive_enabled
            or self.antisway_mpc is None
            or not self._antisway_observer_ready()
            or self.antisway_modal_state is None
            or self.last_command is None
            or self.last_command_step is None
        ):
            return q_reference

        modal_indices = np.asarray(
            [
                len(self.left_joint_names),
                len(self.left_joint_names) + 1,
                len(self.left_joint_names) + 3,
            ],
            dtype=int,
        )
        dt = self._control_dt()
        position = self.last_command[modal_indices]
        target = q_reference[modal_indices]
        velocity = np.clip(
            self.last_command_step[modal_indices] / dt,
            -self.antisway_mpc.max_velocity,
            self.antisway_mpc.max_velocity,
        )
        reference_velocity = self.baseline_velocity["right"][[0, 1, 3]].copy()
        if self.assisted_grasp_active["right"]:
            reference_velocity.fill(0.0)
        reference_velocity = np.clip(
            reference_velocity,
            -min(self.antisway_baseline_velocity_limit, self.antisway_mpc.max_velocity),
            min(self.antisway_baseline_velocity_limit, self.antisway_mpc.max_velocity),
        )
        self.antisway_mpc_reference_velocity = reference_velocity
        if (
            self.antisway_motion_gate_enabled
            and np.linalg.norm(reference_velocity) <= self.antisway_motion_gate_velocity
        ):
            self.antisway_mpc_status = "stationary_bypass"
            return q_reference
        if self.last_command_step_change is None:
            previous_acceleration = np.zeros(3, dtype=float)
        else:
            previous_acceleration = np.clip(
                self.last_command_step_change[modal_indices] / (dt * dt),
                -self.antisway_mpc.max_acceleration,
                self.antisway_mpc.max_acceleration,
            )
        state = np.concatenate(
            [
                self.antisway_modal_state[:4],
                position - target,
                velocity,
            ]
        )
        state_reference = np.zeros(
            (self.antisway_mpc.horizon_steps, self.antisway_mpc.STATE_SIZE),
            dtype=float,
        )
        preview_time = dt * np.arange(
            1,
            self.antisway_mpc.horizon_steps + 1,
            dtype=float,
        )
        state_reference[:, 4:7] = preview_time[:, None] * reference_velocity
        state_reference[:, 7:10] = reference_velocity
        result = self.antisway_mpc.solve(
            state,
            previous_acceleration,
            state_reference.reshape(-1),
        )
        if result is None:
            self.antisway_mpc_status = "exception"
            return q_reference
        self.antisway_mpc_solve_ms = result.solve_ms
        self.antisway_mpc_iterations = result.iterations
        self.antisway_mpc_status = result.status.replace(" ", "_")
        if result.status not in ("solved", "solved inaccurate"):
            return q_reference

        acceleration = np.clip(
            result.acceleration,
            -self.antisway_mpc.max_acceleration,
            self.antisway_mpc.max_acceleration,
        )
        next_position = position + dt * velocity + 0.5 * dt * dt * acceleration
        if self.antisway_max_reference_deviation > 0.0:
            next_position = np.clip(
                next_position,
                target - self.antisway_max_reference_deviation,
                target + self.antisway_max_reference_deviation,
            )
        confidence = self.antisway_confidence
        governed_position = target + confidence * (next_position - target)
        self.antisway_mpc_acceleration = acceleration
        self.antisway_mpc_reference_delta = governed_position - target
        if self.antisway_monitor_only:
            return q_reference
        governed_reference = q_reference.copy()
        governed_reference[modal_indices] = governed_position
        self.antisway_predictive_applied = True
        return self._clamp_bimanual(governed_reference)

    def _publish_antisway_diagnostics(self, q_out: np.ndarray) -> None:
        ready = self._antisway_observer_ready()
        modal_indices = np.asarray(
            [len(self.left_joint_names), len(self.left_joint_names) + 1, len(self.left_joint_names) + 3],
            dtype=int,
        )
        acceleration = np.zeros(3, dtype=float)
        predicted_energy = np.full(2, np.nan, dtype=float)
        current_energy = np.full(2, np.nan, dtype=float)
        acceleration_change_norm = float("nan")
        if self.last_command is not None and self.last_command_step is not None:
            dt = self._control_dt()
            command_step = q_out - self.last_command
            acceleration = (
                command_step[modal_indices] - self.last_command_step[modal_indices]
            ) / (dt * dt)
            if self.last_command_step_change is not None:
                previous_acceleration = (
                    self.last_command_step_change[modal_indices] / (dt * dt)
                )
                acceleration_change_norm = float(
                    np.linalg.norm(acceleration - previous_acceleration)
                )
            if ready and self.antisway_modal_state is not None:
                predicted_state = (
                    self.antisway_a @ self.antisway_modal_state
                    + self.antisway_b @ acceleration
                )
                omega_roll = 2.0 * np.pi * self.antisway_frequencies[0]
                omega_yaw = 2.0 * np.pi * self.antisway_frequencies[1]
                current_energy = np.asarray(
                    [
                        0.5
                        * (
                            self.antisway_modal_state[1] ** 2
                            + (omega_roll * self.antisway_modal_state[0]) ** 2
                        ),
                        0.5
                        * (
                            self.antisway_modal_state[3] ** 2
                            + (omega_yaw * self.antisway_modal_state[2]) ** 2
                        ),
                    ],
                    dtype=float,
                )
                predicted_energy = np.asarray(
                    [
                        0.5
                        * (
                            predicted_state[1] ** 2
                            + (omega_roll * predicted_state[0]) ** 2
                        ),
                        0.5
                        * (
                            predicted_state[3] ** 2
                            + (omega_yaw * predicted_state[2]) ** 2
                        ),
                    ],
                    dtype=float,
                )
        self.antisway_last_acceleration = acceleration
        if self.antisway_predictive_applied:
            self.antisway_mpc_acceleration_error = (
                acceleration - self.antisway_mpc_acceleration
            )
        else:
            self.antisway_mpc_acceleration_error.fill(np.nan)
        self.antisway_last_predicted_energy = predicted_energy
        self.antisway_diagnostics_pub.publish(
            Float64MultiArray(
                data=[
                    float(self.antisway_enabled),
                    float(self.antisway_monitor_only),
                    float(ready),
                    float(
                        self.antisway_cost_applied
                        or self.antisway_predictive_applied
                        or self.antisway_shaper_applied
                        or self.antisway_bimanual_applied
                    ),
                    float(self._antisway_state_age()),
                    float(self.antisway_observer_nis),
                    float(self.antisway_confidence),
                    *current_energy.tolist(),
                    *predicted_energy.tolist(),
                    *acceleration.tolist(),
                    acceleration_change_norm,
                    float(self.antisway_predictive_enabled),
                    float(self.antisway_predictive_applied),
                    float(self.antisway_mpc_solve_ms),
                    float(self.antisway_mpc_iterations),
                    *self.antisway_mpc_acceleration.tolist(),
                    *self.antisway_mpc_reference_delta.tolist(),
                    *self.antisway_mpc_reference_velocity.tolist(),
                    *self.antisway_mpc_acceleration_error.tolist(),
                    float(np.linalg.norm(self.antisway_mpc_acceleration_error)),
                    float(self.antisway_input_shaper_enabled),
                    float(self.antisway_shaper_applied),
                    *self.antisway_shaper_modal_input.tolist(),
                    *self.antisway_shaper_modal_output.tolist(),
                    *self.antisway_shaper_correction.tolist(),
                    float(self.antisway_bimanual_task_qp_enabled),
                    float(self.antisway_bimanual_applied),
                    float(self.antisway_bimanual_solve_ms),
                    float(self.antisway_bimanual_iterations),
                    float(self.antisway_bimanual_nullity),
                    float(self.antisway_bimanual_task_residual),
                    *self.antisway_bimanual_modal_target.tolist(),
                    *self.antisway_bimanual_modal_prediction.tolist(),
                    float(
                        np.linalg.norm(
                            self.antisway_bimanual_acceleration_correction
                        )
                    ),
                    float(
                        np.linalg.norm(
                            self.antisway_bimanual_position_correction
                        )
                    ),
                    float(self.antisway_raw_confidence),
                    float(self.antisway_observer_gate_active),
                    float(self.antisway_bimanual_motion_gate_active),
                    float(self.command_governor_enabled),
                    float(self.command_governor.active),
                    float(np.linalg.norm(self.command_governor_tracking_error)),
                    float(np.max(np.abs(self.command_governor_tracking_error))),
                    float(np.max(np.abs(self.command_governor.velocity))),
                    float(np.max(np.abs(self.command_governor.acceleration))),
                ]
            )
        )

    def _current_safe_start(self, q_baseline: np.ndarray) -> np.ndarray:
        if self.use_measured_joint_state_start and self._joint_state_is_fresh():
            return self._bimanual_from_model_q(self.q_model_current)
        if self.last_command is not None:
            return self._clamp_bimanual(self.last_command)
        if self.have_joint_state:
            return self._bimanual_from_model_q(self.q_model_current)
        return self._clamp_bimanual(q_baseline)

    def _joint_state_age(self) -> float:
        if self.last_joint_state_time is None:
            return float("inf")
        return max(
            0.0,
            (self.get_clock().now() - self.last_joint_state_time).nanoseconds * 1e-9,
        )

    def _joint_state_is_fresh(self) -> bool:
        return self.have_joint_state and self._joint_state_age() <= self.joint_state_timeout_s

    def _apply_assisted_grasp_reference(
        self,
        q_reference: np.ndarray,
        q_start: np.ndarray,
    ) -> np.ndarray:
        if not self.assisted_grasp_enabled:
            return q_reference
        q_work = q_reference.copy()
        tcp_positions = self._tcp_positions(q_start)
        for side in ("left", "right"):
            target = self.grasp_targets[side]
            current = tcp_positions.get(side)
            side_slice = self.left_slice if side == "left" else self.right_slice
            if target is None or current is None:
                self._clear_assisted_grasp_side(side)
                continue
            gripper_fraction = self.assisted_grasp_gripper_fraction[side]
            if self._gripper_is_closed(gripper_fraction):
                self._clear_assisted_grasp_side(side)
                self.grasp_targets[side] = None
                self.grasp_target_times[side] = None
                self.get_logger().info(
                    f"{side} filter-assisted grasp released and target cleared: gripper closed "
                    f"({self._format_gripper_fraction(gripper_fraction)})."
                )
                continue
            distance = float(np.linalg.norm(target - current))
            if (
                self.assisted_grasp_start_times[side] is None
                and distance <= self.assisted_grasp_activation_distance
            ):
                if self.assisted_grasp_require_open_gripper and not self._gripper_is_open(
                    gripper_fraction
                ):
                    self.assisted_grasp_active[side] = False
                    self.assisted_grasp_alpha[side] = 0.0
                    continue
                self.assisted_grasp_start_times[side] = self.get_clock().now()
                self.assisted_grasp_latched_reference[side] = q_start.copy()
                self.get_logger().info(
                    f"{side} filter-assisted grasp latched: "
                    f"distance={distance * 100.0:.1f} cm, "
                    f"ramp={self.assisted_grasp_ramp_duration:.2f}s, "
                    f"gripper={self._format_gripper_fraction(gripper_fraction)}"
                )
            start_time = self.assisted_grasp_start_times[side]
            if start_time is None:
                self.assisted_grasp_active[side] = False
                self.assisted_grasp_alpha[side] = 0.0
                continue
            q_work[side_slice] = q_start[side_slice]
            elapsed = (self.get_clock().now() - start_time).nanoseconds * 1e-9
            alpha = float(np.clip(elapsed / self.assisted_grasp_ramp_duration, 0.0, 1.0))
            offset = alpha * (target - current)
            norm = float(np.linalg.norm(offset))
            if self.assisted_grasp_max_cartesian_step > 0.0 and norm > self.assisted_grasp_max_cartesian_step:
                offset *= self.assisted_grasp_max_cartesian_step / max(norm, 1e-9)
            q_work = self._apply_tcp_offset(q_work, side, offset)
            self.assisted_grasp_active[side] = True
            self.assisted_grasp_alpha[side] = alpha
        return self._clamp_bimanual(q_work)

    def _clear_assisted_grasp_side(self, side: str) -> None:
        self.assisted_grasp_start_times[side] = None
        self.assisted_grasp_active[side] = False
        self.assisted_grasp_alpha[side] = 0.0
        self.assisted_grasp_latched_reference[side] = None

    def _gripper_is_open(self, fraction: float | None) -> bool:
        if fraction is None:
            return not self.assisted_grasp_require_open_gripper
        return fraction >= self.assisted_grasp_gripper_open_fraction

    def _gripper_is_closed(self, fraction: float | None) -> bool:
        if fraction is None:
            return False
        return fraction <= self.assisted_grasp_gripper_close_fraction

    @staticmethod
    def _format_gripper_fraction(fraction: float | None) -> str:
        if fraction is None:
            return "unknown"
        return f"{100.0 * fraction:.1f}%"

    def _model_q_from_bimanual(self, q_bimanual: np.ndarray) -> np.ndarray:
        q_model = self.q_model_current.copy() if self.have_joint_state else self.neutral_q.copy()
        for value, idx in zip(q_bimanual, self.joint_q_indices):
            q_model[idx] = float(value)
        return q_model

    def _bimanual_from_model_q(self, q_model: np.ndarray) -> np.ndarray:
        return self._clamp_bimanual(np.asarray([q_model[idx] for idx in self.joint_q_indices], dtype=float))

    def _clamp_bimanual(self, q_bimanual: np.ndarray) -> np.ndarray:
        q = np.asarray(q_bimanual, dtype=float).copy()
        for i, idx in enumerate(self.joint_q_indices):
            lower = self.lower[idx]
            upper = self.upper[idx]
            if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
                q[i] = float(np.clip(q[i], lower, upper))
        return q

    def _compute_sphere_states(self, q_model: np.ndarray) -> list[SphereState]:
        pin.forwardKinematics(self.model, self.data, q_model)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q_model)

        states: list[SphereState] = []
        for sphere, frame_id in zip(self.spheres, self.sphere_frame_ids):
            placement = self.data.oMf[frame_id]
            offset_world = placement.rotation @ sphere.center
            position = placement.translation + offset_world
            frame_jac = pin.getFrameJacobian(
                self.model,
                self.data,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            point_jac_full = frame_jac[:3, :] - _skew(offset_world) @ frame_jac[3:, :]
            point_jac = point_jac_full[:, self.joint_v_indices]
            states.append(SphereState(sphere=sphere, position=position, jacobian=point_jac))
        return states

    def _maybe_request_esdf(self, sphere_states: list[SphereState]) -> None:
        if self.esdf_pending or self.esdf_client is None:
            return
        now = self.get_clock().now()
        if (now - self.last_esdf_request_time).nanoseconds < int(1e9 / self.esdf_update_hz):
            return
        if not self.esdf_client.service_is_ready():
            service_name = self._discover_esdf_service()
            if service_name and service_name != self.esdf_service_name:
                self.esdf_service_name = service_name
                self.esdf_client = self.create_client(EsdfAndGradients, service_name)
                self.get_logger().info(f"Switched ESDF service to discovered service: {service_name}")
            else:
                self._throttled_warn(f"ESDF service not ready: {self.esdf_service_name}")
                return

        active_states = [sphere_states[idx] for idx in self.esdf_sphere_indices if idx < len(sphere_states)]
        if not active_states:
            return
        positions = np.vstack([state.position for state in active_states])
        if self.esdf_extension_offsets:
            extension_positions = [positions + offset.reshape(1, 3) for offset in self.esdf_extension_offsets]
            positions_for_aabb = np.vstack([positions] + extension_positions)
        else:
            positions_for_aabb = positions
        radii = np.asarray([state.sphere.radius for state in active_states], dtype=float)
        padding = self.aabb_padding + float(np.max(radii)) + self.activation_margin
        aabb_min = np.min(positions_for_aabb, axis=0) - padding
        aabb_max = np.max(positions_for_aabb, axis=0) + padding

        req = EsdfAndGradients.Request()
        req.update_esdf = self.request_update_esdf
        req.visualize_esdf = False
        req.use_aabb = True
        req.frame_id = self.global_frame
        req.aabb_min_m = _point_from_array(aabb_min)
        req.aabb_size_m = _vector_from_array(aabb_max - aabb_min)

        if self.clear_robot_from_esdf:
            for state in sphere_states:
                req.spheres_to_clear_center_m.append(_point_from_array(state.position))
                clear_radius = state.sphere.radius * self.clear_robot_radius_scale + self.clear_robot_padding
                req.spheres_to_clear_radius_m.append(float(clear_radius))

        future = self.esdf_client.call_async(req)
        future.add_done_callback(self._esdf_response_cb)
        self.esdf_pending = True
        self.last_esdf_request_time = now

    def _esdf_response_cb(self, future) -> None:
        self.esdf_pending = False
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self._throttled_warn(f"ESDF service call failed: {exc}")
            return
        if response is None or not response.success:
            self._throttled_warn("ESDF service returned failure")
            return
        grid = EsdfGrid(response, self.unobserved_value, self.nearest_observed_search_radius)
        if not grid.valid():
            self._throttled_warn("ESDF service returned an invalid or empty grid")
            return
        self.esdf_grid = grid
        self.esdf_grid_shape = "x".join(str(int(v)) for v in grid.size)
        self.esdf_observed_count = grid.observed_count

    def _avoid(self, q_baseline: np.ndarray, q_start: np.ndarray) -> np.ndarray:
        if self.avoidance_solver == "cbf_qp":
            if minimize is None:
                self.cbf_qp_success = False
                self.cbf_qp_status = "scipy_unavailable"
                self._throttled_warn("scipy.optimize is unavailable; using soft avoidance solver.")
                return self._avoid_soft(q_baseline, q_start)
            return self._avoid_cbf_qp(q_baseline, q_start)
        return self._avoid_soft(q_baseline, q_start)

    def _select_predictive_rollout(self, q_baseline: np.ndarray, q_start: np.ndarray) -> np.ndarray:
        self.rollout_active = False
        self.rollout_candidate_count = 0
        self.rollout_selected = "baseline"
        self.rollout_best_cost = float("nan")
        self.rollout_baseline_cost = float("nan")
        if (
            not self.predictive_rollout_enabled
            or self.current_obstacle_source == "none"
            or self.rollout_horizon_steps <= 1
        ):
            return q_baseline

        baseline_states = self._compute_sphere_states(self._model_q_from_bimanual(q_baseline))
        self._sample_spheres(baseline_states, update_filter=False)
        start_states = self._compute_sphere_states(self._model_q_from_bimanual(q_start))
        self._sample_spheres(start_states, update_filter=False)

        trigger = self.activation_margin + self.rollout_trigger_margin
        left_active = self._side_min_clearance(baseline_states, "_left_") < trigger
        right_active = self._side_min_clearance(baseline_states, "_right_") < trigger
        left_active = left_active or self._side_min_clearance(start_states, "_left_") < trigger
        right_active = right_active or self._side_min_clearance(start_states, "_right_") < trigger
        if not left_active and not right_active:
            return q_baseline

        candidates: list[tuple[str, np.ndarray]] = [("baseline", q_baseline.copy())]
        if left_active:
            candidates.extend(self._rollout_side_candidates(q_baseline, "left"))
        if right_active:
            candidates.extend(self._rollout_side_candidates(q_baseline, "right"))
        if left_active and right_active:
            for down in self.rollout_down_offsets:
                if abs(float(down)) < 1e-9:
                    continue
                q_candidate = q_baseline.copy()
                offset = self.downward_direction * float(down)
                q_candidate = self._apply_tcp_offset(q_candidate, "left", offset)
                q_candidate = self._apply_tcp_offset(q_candidate, "right", offset)
                candidates.append((f"both_down_{down:.3f}", q_candidate))

        unique_candidates: list[tuple[str, np.ndarray]] = []
        seen: set[tuple[float, ...]] = set()
        for label, q_candidate in candidates:
            q_candidate = self._clamp_rollout_candidate(q_candidate, q_baseline)
            key = tuple(np.round(q_candidate, 5))
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append((label, q_candidate))

        baseline_positions = self._tcp_positions(q_baseline)
        best_label = "baseline"
        best_q = q_baseline.copy()
        best_cost = float("inf")
        baseline_cost = float("nan")
        for label, q_candidate in unique_candidates:
            cost = self._score_rollout_candidate(
                q_candidate,
                q_baseline,
                q_start,
                baseline_positions,
            )
            if label == "baseline":
                baseline_cost = cost
            if cost < best_cost:
                best_cost = cost
                best_label = label
                best_q = q_candidate

        self.rollout_candidate_count = len(unique_candidates)
        self.rollout_selected = best_label
        self.rollout_best_cost = best_cost
        self.rollout_baseline_cost = baseline_cost
        self.rollout_active = best_label != "baseline"
        return best_q

    def _side_min_clearance(self, states: list[SphereState], side_token: str) -> float:
        clearances = [
            float(state.clearance)
            for state in states
            if side_token in state.sphere.frame
            and state.clearance is not None
            and np.isfinite(state.clearance)
        ]
        if not clearances:
            return float("inf")
        return float(np.min(clearances))

    def _rollout_side_candidates(self, q_baseline: np.ndarray, side: str) -> list[tuple[str, np.ndarray]]:
        candidates: list[tuple[str, np.ndarray]] = []
        for down in self.rollout_down_offsets:
            for side_offset in self.rollout_side_offsets:
                if abs(float(down)) < 1e-9 and abs(float(side_offset)) < 1e-9:
                    continue
                offset = (
                    self.downward_direction * float(down)
                    + self.rollout_side_axis * float(side_offset)
                )
                q_candidate = self._apply_tcp_offset(q_baseline, side, offset)
                candidates.append((f"{side}_down_{down:.3f}_side_{side_offset:.3f}", q_candidate))
        return candidates

    def _apply_tcp_offset(self, q_bimanual: np.ndarray, side: str, offset_world: np.ndarray) -> np.ndarray:
        frame_id = self.left_tcp_frame_id if side == "left" else self.right_tcp_frame_id
        side_slice = self.left_slice if side == "left" else self.right_slice
        if frame_id is None:
            return q_bimanual.copy()

        q_model = self._model_q_from_bimanual(q_bimanual)
        pin.forwardKinematics(self.model, self.data, q_model)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q_model)
        jac_full = pin.getFrameJacobian(
            self.model,
            self.data,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )[:3, :]
        side_indices = self.joint_v_indices[side_slice]
        jac = jac_full[:, side_indices]
        if not np.all(np.isfinite(jac)) or float(np.linalg.norm(jac)) < 1e-8:
            return q_bimanual.copy()

        offset = np.asarray(offset_world, dtype=float).reshape(3)
        damping2 = self.rollout_cartesian_damping * self.rollout_cartesian_damping
        lhs = jac @ jac.T + damping2 * np.eye(3)
        try:
            dq_side = jac.T @ np.linalg.solve(lhs, offset)
        except np.linalg.LinAlgError:
            dq_side = jac.T @ np.linalg.lstsq(lhs, offset, rcond=None)[0]
        if not np.all(np.isfinite(dq_side)):
            return q_bimanual.copy()

        q_candidate = q_bimanual.copy()
        q_candidate[side_slice] += dq_side
        return self._clamp_bimanual(q_candidate)

    def _clamp_rollout_candidate(
        self,
        q_candidate: np.ndarray,
        q_baseline: np.ndarray,
    ) -> np.ndarray:
        q_candidate = self._clamp_bimanual(q_candidate)
        if self.rollout_max_joint_delta > 0.0:
            delta = np.clip(
                q_candidate - q_baseline,
                -self.rollout_max_joint_delta,
                self.rollout_max_joint_delta,
            )
            q_candidate = self._clamp_bimanual(q_baseline + delta)
        return q_candidate

    def _tcp_positions(self, q_bimanual: np.ndarray) -> dict[str, np.ndarray]:
        q_model = self._model_q_from_bimanual(q_bimanual)
        pin.forwardKinematics(self.model, self.data, q_model)
        pin.updateFramePlacements(self.model, self.data)
        positions: dict[str, np.ndarray] = {}
        if self.left_tcp_frame_id is not None:
            positions["left"] = np.asarray(
                self.data.oMf[self.left_tcp_frame_id].translation,
                dtype=float,
            ).copy()
        if self.right_tcp_frame_id is not None:
            positions["right"] = np.asarray(
                self.data.oMf[self.right_tcp_frame_id].translation,
                dtype=float,
            ).copy()
        return positions

    @staticmethod
    def _limit_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
        result = np.asarray(vector, dtype=float).copy()
        norm = float(np.linalg.norm(result))
        if max_norm > 0.0 and norm > max_norm:
            result *= max_norm / max(norm, 1e-9)
        return result

    def _bounded_baseline_delta(
        self,
        q_baseline: np.ndarray,
        q_work: np.ndarray,
    ) -> np.ndarray:
        delta = np.asarray(q_baseline - q_work, dtype=float)
        if self.max_baseline_joint_error > 0.0:
            delta = np.clip(
                delta,
                -self.max_baseline_joint_error,
                self.max_baseline_joint_error,
            )
        return delta

    def _add_tcp_pose_tracking(
        self,
        lhs: np.ndarray,
        rhs: np.ndarray,
        q_work: np.ndarray,
        q_baseline: np.ndarray,
        wrist_avoidance_sides: set[str] | None = None,
    ) -> None:
        if self.tcp_position_weight <= 0.0 and self.tcp_orientation_weight <= 0.0:
            return
        wrist_avoidance_sides = wrist_avoidance_sides or set()

        q_baseline_model = self._model_q_from_bimanual(q_baseline)
        pin.forwardKinematics(self.model, self.baseline_data, q_baseline_model)
        pin.updateFramePlacements(self.model, self.baseline_data)

        q_work_model = self._model_q_from_bimanual(q_work)
        pin.forwardKinematics(self.model, self.data, q_work_model)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q_work_model)

        for side, frame_id in (
            ("left", self.left_tcp_frame_id),
            ("right", self.right_tcp_frame_id),
        ):
            if frame_id is None:
                continue
            preserve_tcp = side in wrist_avoidance_sides
            position_weight = self.tcp_position_weight * (
                self.wrist_tcp_position_weight_scale if preserve_tcp else 1.0
            )
            orientation_weight = self.tcp_orientation_weight * (
                self.wrist_tcp_orientation_weight_scale if preserve_tcp else 1.0
            )
            target = self.baseline_data.oMf[frame_id]
            current = self.data.oMf[frame_id]
            frame_jac_full = pin.getFrameJacobian(
                self.model,
                self.data,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            frame_jac = frame_jac_full[:, self.joint_v_indices]
            if not np.all(np.isfinite(frame_jac)):
                continue

            if position_weight > 0.0:
                position_error = self._limit_vector_norm(
                    np.asarray(target.translation - current.translation, dtype=float),
                    self.tcp_position_max_step,
                )
                linear_jac = frame_jac[:3, :]
                lhs += position_weight * (linear_jac.T @ linear_jac)
                rhs += position_weight * (linear_jac.T @ position_error)

            if orientation_weight > 0.0:
                rotation_error_local = np.asarray(
                    pin.log3(current.rotation.T @ target.rotation),
                    dtype=float,
                ).reshape(3)
                rotation_error_world = current.rotation @ rotation_error_local
                rotation_error_world = self._limit_vector_norm(
                    rotation_error_world,
                    self.tcp_orientation_max_step,
                )
                angular_jac = frame_jac[3:, :]
                lhs += orientation_weight * (angular_jac.T @ angular_jac)
                rhs += orientation_weight * (angular_jac.T @ rotation_error_world)

    def _score_rollout_candidate(
        self,
        q_candidate: np.ndarray,
        q_baseline: np.ndarray,
        q_start: np.ndarray,
        baseline_positions: dict[str, np.ndarray],
    ) -> float:
        total = 0.0
        previous_q = q_start
        for step in range(1, self.rollout_horizon_steps + 1):
            alpha = step / float(self.rollout_horizon_steps)
            q_step = self._clamp_bimanual(q_start + alpha * (q_candidate - q_start))
            states = self._compute_sphere_states(self._model_q_from_bimanual(q_step))
            self._sample_spheres(states, update_filter=False)
            total += self._rollout_esdf_cost(states)
            total += self._rollout_inter_arm_cost(states)
            total += self.rollout_smoothness_weight * float(np.sum((q_step - previous_q) ** 2))
            previous_q = q_step

            step_positions = self._tcp_positions(q_step)
            for side, baseline_position in baseline_positions.items():
                step_position = step_positions.get(side)
                if step_position is None:
                    continue
                delta = step_position - baseline_position
                total += self.rollout_xy_weight * float(delta[:2] @ delta[:2])
                total += self.rollout_z_weight * float(delta[2] * delta[2])

        joint_delta = q_candidate - q_baseline
        total += self.rollout_joint_weight * float(joint_delta @ joint_delta)
        return float(total)

    def _rollout_esdf_cost(self, states: list[SphereState]) -> float:
        cost = 0.0
        for state in states:
            if state.clearance is None or not np.isfinite(state.clearance):
                continue
            clearance = float(state.clearance)
            if clearance < self.safety_margin:
                violation = self.safety_margin - clearance
                cost += self.rollout_collision_weight * violation * violation
            elif clearance < self.activation_margin:
                proximity = self.activation_margin - clearance
                cost += self.rollout_activation_weight * proximity * proximity
        return float(cost)

    def _rollout_inter_arm_cost(self, states: list[SphereState]) -> float:
        if not self.enable_inter_arm_collision:
            return 0.0
        cost = 0.0
        for left_idx in self.left_sphere_indices:
            left = states[left_idx]
            for right_idx in self.right_sphere_indices:
                right = states[right_idx]
                distance = float(np.linalg.norm(left.position - right.position))
                if not np.isfinite(distance) or distance < 1e-9:
                    continue
                clearance = distance - left.sphere.radius - right.sphere.radius
                if clearance < self.inter_arm_safety_margin:
                    violation = self.inter_arm_safety_margin - clearance
                    cost += self.rollout_collision_weight * violation * violation
                elif clearance < self.inter_arm_activation_margin:
                    proximity = self.inter_arm_activation_margin - clearance
                    cost += self.rollout_activation_weight * proximity * proximity
        return float(cost)

    def _avoid_soft(self, q_baseline: np.ndarray, q_start: np.ndarray) -> np.ndarray:
        q_work = q_start.copy()
        self.min_clearance = float("nan")
        self.min_inter_arm_clearance = float("nan")
        self.active_constraints = 0
        self.active_esdf_constraints = 0
        self.active_inter_arm_constraints = 0
        self.baseline_error_norm = 0.0
        self.hold_due_to_invalid_clearance = False
        self.hold_due_to_avoidance_latch = False
        self.cbf_qp_success = False
        self.cbf_qp_status = "soft"
        self.cbf_qp_slack_max = 0.0
        self.rejected_esdf_samples = 0
        self.wrist_tcp_preservation_active = {"left": False, "right": False}
        baseline_states = self._compute_sphere_states(self._model_q_from_bimanual(q_baseline))
        self._sample_spheres(baseline_states, update_filter=False)
        self.baseline_min_clearance = self._min_clearance_from_states(baseline_states)
        self.baseline_left_min_clearance = self._min_clearance_for_side(
            baseline_states,
            "left",
        )
        self.baseline_right_min_clearance = self._min_clearance_for_side(
            baseline_states,
            "right",
        )

        for _ in range(self.iterations):
            states = self._compute_sphere_states(self._model_q_from_bimanual(q_work))
            self._sample_spheres(states)
            clearances = [state.clearance for state in states if state.clearance is not None]
            self.sampled_spheres = len(clearances)
            if clearances:
                self.min_clearance = float(np.min(clearances))

            wrist_sides = self._wrist_avoidance_sides(states)
            self.wrist_tcp_preservation_active = {
                "left": "left" in wrist_sides,
                "right": "right" in wrist_sides,
            }
            baseline_weight = self._effective_baseline_weight(self.baseline_min_clearance)
            baseline_weights = np.full(len(self.joint_names), baseline_weight, dtype=float)
            for side in wrist_sides:
                side_slice = self.left_slice if side == "left" else self.right_slice
                baseline_weights[side_slice] *= self.wrist_baseline_weight_scale
            lhs = np.diag(baseline_weights)
            rhs = baseline_weights * self._bounded_baseline_delta(q_baseline, q_work)
            self._add_tcp_pose_tracking(
                lhs,
                rhs,
                q_work,
                q_baseline,
                wrist_sides,
            )
            active_esdf = self._add_esdf_constraints(lhs, rhs, states, baseline_states)
            active_inter = self._add_inter_arm_constraints(lhs, rhs, states)
            active = active_esdf + active_inter

            self.active_esdf_constraints = active_esdf
            self.active_inter_arm_constraints = active_inter
            self.active_constraints = active
            if active == 0:
                if (
                    self.hold_on_invalid_clearance
                    and np.isfinite(self.min_clearance)
                    and self.min_clearance < self.safety_margin
                ):
                    self.hold_due_to_invalid_clearance = True
                    q_work = q_start.copy()
                elif self._should_hold_avoidance_latch(self.baseline_min_clearance):
                    self.hold_due_to_avoidance_latch = True
                    q_work = q_start.copy()
                else:
                    q_work = q_baseline.copy()
                break

            if self.avoidance_latch_enabled and not self.untangle_mode:
                if np.isfinite(self.baseline_min_clearance) and (
                    self.baseline_min_clearance >= self.avoidance_release_margin
                ):
                    self.avoidance_release_counter += 1
                    if self.avoidance_release_counter >= self.avoidance_release_cycles:
                        self.avoidance_latched = False
                        self.avoidance_release_counter = 0
                    else:
                        self.avoidance_latched = True
                else:
                    self.avoidance_latched = True
                    self.avoidance_release_counter = 0

            lhs += (self.damping * self.damping) * np.eye(len(self.joint_names))
            try:
                dq = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                dq = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

            if self.max_adjust_per_joint > 0.0:
                dq = np.clip(dq, -self.max_adjust_per_joint, self.max_adjust_per_joint)
            q_work = self._clamp_bimanual(q_work + dq)

        states = self._compute_sphere_states(self._model_q_from_bimanual(q_work))
        self._sample_spheres(states)
        clearances = [state.clearance for state in states if state.clearance is not None]
        self.sampled_spheres = len(clearances)
        if clearances:
            self.min_clearance = float(np.min(clearances))
        self._update_min_clearance_debug(states)
        self._update_min_inter_arm_clearance(states)
        if self.publish_markers:
            self._publish_markers(states)
        return q_work

    def _avoid_cbf_qp(self, q_baseline: np.ndarray, q_start: np.ndarray) -> np.ndarray:
        q_work = q_start.copy()
        self.min_clearance = float("nan")
        self.min_inter_arm_clearance = float("nan")
        self.active_constraints = 0
        self.active_esdf_constraints = 0
        self.active_inter_arm_constraints = 0
        self.baseline_error_norm = 0.0
        self.hold_due_to_invalid_clearance = False
        self.hold_due_to_avoidance_latch = False
        self.cbf_qp_success = False
        self.cbf_qp_status = "not_run"
        self.cbf_qp_slack_max = 0.0
        self.cbf_qp_solve_ms = 0.0
        self.cbf_qp_iterations = 0
        self.cbf_qp_backend_used = "none"
        self.rejected_esdf_samples = 0
        self.antisway_cost_applied = False
        self.wrist_tcp_preservation_active = {"left": False, "right": False}

        baseline_states = self._compute_sphere_states(self._model_q_from_bimanual(q_baseline))
        self._sample_spheres(baseline_states, update_filter=False)
        self.baseline_min_clearance = self._min_clearance_from_states(baseline_states)
        self.baseline_left_min_clearance = self._min_clearance_for_side(
            baseline_states,
            "left",
        )
        self.baseline_right_min_clearance = self._min_clearance_for_side(
            baseline_states,
            "right",
        )

        for _ in range(self.iterations):
            states = self._compute_sphere_states(self._model_q_from_bimanual(q_work))
            self._sample_spheres(states)
            clearances = [state.clearance for state in states if state.clearance is not None]
            self.sampled_spheres = len(clearances)
            if clearances:
                self.min_clearance = float(np.min(clearances))

            wrist_sides = self._wrist_avoidance_sides(states)
            self.wrist_tcp_preservation_active = {
                "left": "left" in wrist_sides,
                "right": "right" in wrist_sides,
            }
            baseline_weight = self._effective_baseline_weight(self.baseline_min_clearance)
            baseline_weights = np.full(len(self.joint_names), baseline_weight, dtype=float)
            for side in wrist_sides:
                side_slice = self.left_slice if side == "left" else self.right_slice
                baseline_weights[side_slice] *= self.wrist_baseline_weight_scale
            lhs = np.diag(baseline_weights)
            rhs = baseline_weights * self._bounded_baseline_delta(q_baseline, q_work)
            self._add_tcp_pose_tracking(
                lhs,
                rhs,
                q_work,
                q_baseline,
                wrist_sides,
            )
            self._add_cbf_soft_preferences(lhs, rhs, states, baseline_states)
            self._add_antisway_soft_cost(lhs, rhs, q_work)
            a_mat, b_vec, active_esdf, active_inter = self._build_cbf_constraints(states)
            active = active_esdf + active_inter

            self.active_esdf_constraints = active_esdf
            self.active_inter_arm_constraints = active_inter
            self.active_constraints = active
            if active == 0:
                if (
                    self.hold_on_invalid_clearance
                    and np.isfinite(self.min_clearance)
                    and self.min_clearance < self.safety_margin
                ):
                    self.hold_due_to_invalid_clearance = True
                    q_work = q_start.copy()
                    self.cbf_qp_status = "hold_invalid"
                elif self._should_hold_avoidance_latch(self.baseline_min_clearance):
                    self.hold_due_to_avoidance_latch = True
                    q_work = q_start.copy()
                    self.cbf_qp_status = "hold_latch"
                elif self.antisway_cost_applied:
                    lhs += (self.damping * self.damping) * np.eye(len(self.joint_names))
                    dq = self._solve_cbf_qp(lhs, rhs, a_mat, b_vec, q_work)
                    if dq is None:
                        q_work = q_baseline.copy()
                    else:
                        q_work = self._clamp_bimanual(q_work + dq)
                else:
                    q_work = q_baseline.copy()
                    self.cbf_qp_status = "no_active"
                break

            if self.avoidance_latch_enabled and not self.untangle_mode:
                if np.isfinite(self.baseline_min_clearance) and (
                    self.baseline_min_clearance >= self.avoidance_release_margin
                ):
                    self.avoidance_release_counter += 1
                    if self.avoidance_release_counter >= self.avoidance_release_cycles:
                        self.avoidance_latched = False
                        self.avoidance_release_counter = 0
                    else:
                        self.avoidance_latched = True
                else:
                    self.avoidance_latched = True
                    self.avoidance_release_counter = 0

            lhs += (self.damping * self.damping) * np.eye(len(self.joint_names))
            dq = self._solve_cbf_qp(lhs, rhs, a_mat, b_vec, q_work)
            if dq is None:
                if self.cbf_fallback_to_soft:
                    self._throttled_warn(
                        f"CBF-QP failed ({self.cbf_qp_status}); using soft avoidance solver."
                    )
                    return self._avoid_soft(q_baseline, q_start)
                q_work = q_start.copy()
                break
            q_work = self._clamp_bimanual(q_work + dq)

        states = self._compute_sphere_states(self._model_q_from_bimanual(q_work))
        self._sample_spheres(states)
        clearances = [state.clearance for state in states if state.clearance is not None]
        self.sampled_spheres = len(clearances)
        if clearances:
            self.min_clearance = float(np.min(clearances))
        self._update_min_clearance_debug(states)
        self._update_min_inter_arm_clearance(states)
        if self.publish_markers:
            self._publish_markers(states)
        return q_work

    def _add_antisway_soft_cost(
        self,
        lhs: np.ndarray,
        rhs: np.ndarray,
        q_work: np.ndarray,
    ) -> None:
        if (
            self.antisway_bimanual_task_qp_enabled
            or self.antisway_input_shaper_enabled
            or self.antisway_predictive_enabled
            or self.antisway_monitor_only
            or not self._antisway_observer_ready()
        ):
            return
        if self.last_command is None or self.last_command_step is None:
            return

        n = len(self.joint_names)
        modal_indices = np.asarray(
            [len(self.left_joint_names), len(self.left_joint_names) + 1, len(self.left_joint_names) + 3],
            dtype=int,
        )
        dt = self._control_dt()
        acceleration_map = np.zeros((3, n), dtype=float)
        acceleration_map[np.arange(3), modal_indices] = 1.0 / (dt * dt)
        acceleration_offset = (
            q_work[modal_indices]
            - self.last_command[modal_indices]
            - self.last_command_step[modal_indices]
        ) / (dt * dt)

        confidence = self.antisway_confidence
        smooth_weight = confidence * self.antisway_acceleration_change_weight
        if smooth_weight > 0.0:
            if self.last_command_step_change is None:
                previous_acceleration = np.zeros(3, dtype=float)
            else:
                previous_acceleration = (
                    self.last_command_step_change[modal_indices] / (dt * dt)
                )
            offset = acceleration_offset - previous_acceleration
            scale_squared = self.antisway_acceleration_scale**2
            lhs += (
                smooth_weight
                * (acceleration_map.T @ acceleration_map)
                / scale_squared
            )
            rhs -= (
                smooth_weight
                * (acceleration_map.T @ offset)
                / scale_squared
            )

        modal_weight = confidence * self.antisway_modal_weight
        if modal_weight > 0.0 and self.antisway_modal_state is not None:
            predicted_offset = (
                self.antisway_a @ self.antisway_modal_state
                + self.antisway_b @ acceleration_offset
            )
            predicted_map = self.antisway_b @ acceleration_map
            omega_roll = 2.0 * np.pi * self.antisway_frequencies[0]
            omega_yaw = 2.0 * np.pi * self.antisway_frequencies[1]
            energy_weight = np.diag(
                [omega_roll**2, 1.0, omega_yaw**2, 1.0]
            )
            modal_map = predicted_map[:4, :]
            modal_offset = predicted_offset[:4]
            lhs += modal_weight * modal_map.T @ energy_weight @ modal_map
            rhs -= modal_weight * modal_map.T @ energy_weight @ modal_offset

        self.antisway_cost_applied = bool(smooth_weight > 0.0 or modal_weight > 0.0)

    def _min_clearance_from_states(self, states: list[SphereState]) -> float:
        clearances = [state.clearance for state in states if state.clearance is not None]
        if not clearances:
            return float("nan")
        return float(np.min(clearances))

    def _min_clearance_for_side(
        self,
        states: list[SphereState],
        side: str,
    ) -> float:
        clearances = [
            float(state.clearance)
            for state in states
            if (
                state.clearance is not None
                and self._sphere_side(state.sphere.frame) == side
            )
        ]
        if not clearances:
            return float("nan")
        return float(np.min(clearances))

    def _update_min_clearance_debug(self, states: list[SphereState]) -> None:
        best_state: SphereState | None = None
        best_clearance = float("nan")
        left_best_state: SphereState | None = None
        left_best_clearance = float("nan")
        right_best_state: SphereState | None = None
        right_best_clearance = float("nan")
        for state in states:
            if state.clearance is None or not np.isfinite(state.clearance):
                continue
            if best_state is None or state.clearance < best_clearance:
                best_state = state
                best_clearance = float(state.clearance)
            if "_left_" in state.sphere.frame:
                if left_best_state is None or state.clearance < left_best_clearance:
                    left_best_state = state
                    left_best_clearance = float(state.clearance)
            elif "_right_" in state.sphere.frame:
                if right_best_state is None or state.clearance < right_best_clearance:
                    right_best_state = state
                    right_best_clearance = float(state.clearance)
        if best_state is None:
            self.min_clearance_sphere = "none"
            self.min_clearance_position = np.array([np.nan, np.nan, np.nan], dtype=float)
            self.left_min_clearance = float("nan")
            self.right_min_clearance = float("nan")
            self.left_min_clearance_sphere = "none"
            self.right_min_clearance_sphere = "none"
            return
        self.min_clearance_sphere = best_state.sphere.frame
        self.min_clearance_position = np.asarray(best_state.position, dtype=float).reshape(3)
        self.left_min_clearance = left_best_clearance
        self.right_min_clearance = right_best_clearance
        self.left_min_clearance_sphere = (
            left_best_state.sphere.frame if left_best_state is not None else "none"
        )
        self.right_min_clearance_sphere = (
            right_best_state.sphere.frame if right_best_state is not None else "none"
        )

    def _should_hold_avoidance_latch(self, baseline_min_clearance: float) -> bool:
        if self.untangle_mode:
            self.avoidance_latched = False
            self.avoidance_release_counter = 0
            return False
        if not self.avoidance_latch_enabled or not self.avoidance_latched:
            return False
        if not np.isfinite(baseline_min_clearance):
            self.avoidance_release_counter = 0
            return True
        if baseline_min_clearance < self.avoidance_release_margin:
            self.avoidance_release_counter = 0
            return True
        self.avoidance_release_counter += 1
        if self.avoidance_release_counter >= self.avoidance_release_cycles:
            self.avoidance_latched = False
            self.avoidance_release_counter = 0
            return False
        return True

    def _effective_baseline_weight(self, baseline_min_clearance: float) -> float:
        if self.untangle_mode:
            return self.baseline_weight
        if not self.avoidance_latch_enabled or not self.avoidance_latched:
            return self.baseline_weight
        if np.isfinite(baseline_min_clearance) and baseline_min_clearance >= self.avoidance_release_margin:
            return self.baseline_weight
        return max(1e-6, self.baseline_weight * self.latched_baseline_weight_scale)

    def _add_cbf_soft_preferences(
        self,
        lhs: np.ndarray,
        rhs: np.ndarray,
        states: list[SphereState],
        baseline_states: list[SphereState],
    ) -> None:
        escape_seen = {"left": False, "right": False}
        self.hand_bypass_active = {"left": False, "right": False}
        for idx, state in enumerate(states):
            if state.clearance is None or state.clearance >= self.activation_margin:
                continue
            proximity = self._activation_proximity(state.clearance)
            if idx < len(baseline_states):
                if self.untangle_mode:
                    self._add_untangle_tangent_preference(
                        lhs,
                        rhs,
                        state,
                        baseline_states[idx],
                        proximity,
                    )
                else:
                    if not self._is_wrist_collision_frame(state.sphere.frame):
                        self._add_xy_follow_constraint(
                            lhs,
                            rhs,
                            state,
                            baseline_states[idx],
                            proximity,
                        )
                    side = self._sphere_side(state.sphere.frame)
                    if side is not None and self._add_cable_tangent_escape(
                        lhs,
                        rhs,
                        state,
                        baseline_states[idx],
                        side,
                        proximity,
                    ):
                        escape_seen[side] = True
            if (
                state.clearance < self.safety_margin
                and not self.untangle_mode
                and self._is_hand_collision_frame(state.sphere.frame)
            ):
                self._add_downward_bias(lhs, rhs, state.jacobian, proximity, state.gradient)
        for side, seen in escape_seen.items():
            if not seen:
                self.tangent_escape_sign[side] = 0.0
                self.hand_bypass_active[side] = False

    def _build_cbf_constraints(
        self,
        states: list[SphereState],
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        esdf_candidates: list[tuple[float, np.ndarray, float]] = []
        target_clearance = self.activation_margin
        for state in states:
            if state.clearance is None or state.gradient is None:
                continue
            if state.clearance >= target_clearance:
                continue
            avoidance_gradient = self._select_avoidance_gradient(state.gradient)
            a = avoidance_gradient.reshape(1, 3) @ state.jacobian
            a = a.reshape(-1)
            if not np.all(np.isfinite(a)) or float(np.linalg.norm(a)) < 1e-8:
                continue
            h = float(state.clearance - self.safety_margin)
            b = -self._control_dt() * self.cbf_gain * h
            if not np.isfinite(b):
                continue
            esdf_candidates.append((float(state.clearance), a, float(b)))

        esdf_candidates.sort(key=lambda item: item[0])
        esdf_candidates = esdf_candidates[: self.cbf_max_esdf_constraints]

        inter_candidates: list[tuple[float, np.ndarray, float]] = []
        min_inter = float("nan")
        if self.enable_inter_arm_collision and self.cbf_max_inter_arm_constraints > 0:
            for left_idx in self.left_sphere_indices:
                left = states[left_idx]
                for right_idx in self.right_sphere_indices:
                    right = states[right_idx]
                    delta = left.position - right.position
                    distance = float(np.linalg.norm(delta))
                    if not np.isfinite(distance) or distance < 1e-6:
                        continue
                    clearance = distance - left.sphere.radius - right.sphere.radius
                    if not np.isfinite(min_inter) or clearance < min_inter:
                        min_inter = float(clearance)
                    if clearance >= self.inter_arm_activation_margin:
                        continue
                    direction = delta / distance
                    a = direction.reshape(1, 3) @ (left.jacobian - right.jacobian)
                    a = a.reshape(-1)
                    if not np.all(np.isfinite(a)) or float(np.linalg.norm(a)) < 1e-8:
                        continue
                    h = float(clearance - self.inter_arm_safety_margin)
                    b = -self._control_dt() * self.cbf_gain * h
                    if not np.isfinite(b):
                        continue
                    inter_candidates.append((float(clearance), a, float(b)))

        inter_candidates.sort(key=lambda item: item[0])
        inter_candidates = inter_candidates[: self.cbf_max_inter_arm_constraints]
        self.min_inter_arm_clearance = min_inter

        rows = [item[1] for item in esdf_candidates] + [item[1] for item in inter_candidates]
        bounds = [item[2] for item in esdf_candidates] + [item[2] for item in inter_candidates]
        if not rows:
            n = len(self.joint_names)
            return np.zeros((0, n), dtype=float), np.zeros(0, dtype=float), 0, 0
        return (
            np.vstack(rows).astype(float),
            np.asarray(bounds, dtype=float),
            len(esdf_candidates),
            len(inter_candidates),
        )

    def _solve_cbf_qp(
        self,
        lhs: np.ndarray,
        rhs: np.ndarray,
        a_mat: np.ndarray,
        b_vec: np.ndarray,
        q_work: np.ndarray,
    ) -> np.ndarray | None:
        solve_start = time.perf_counter()
        self.cbf_qp_solve_ms = 0.0
        self.cbf_qp_iterations = 0
        self.cbf_qp_backend_used = "none"
        n = len(self.joint_names)
        m = int(a_mat.shape[0])
        if m == 0:
            try:
                dq = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                dq = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            dq = self._clip_dq_to_bounds(dq, q_work)
            self.cbf_qp_success = True
            self.cbf_qp_status = "unconstrained"
            self.cbf_qp_slack_max = 0.0
            self.cbf_qp_backend_used = "numpy"
            self.cbf_qp_solve_ms = (time.perf_counter() - solve_start) * 1e3
            return dq

        lhs = np.asarray(lhs, dtype=float)
        rhs = np.asarray(rhs, dtype=float)
        a_mat = np.asarray(a_mat, dtype=float)
        b_vec = np.asarray(b_vec, dtype=float)
        if (
            lhs.shape != (n, n)
            or rhs.shape != (n,)
            or a_mat.shape != (m, n)
            or b_vec.shape != (m,)
            or not np.all(np.isfinite(lhs))
            or not np.all(np.isfinite(rhs))
            or not np.all(np.isfinite(a_mat))
            or not np.all(np.isfinite(b_vec))
        ):
            self.cbf_qp_success = False
            self.cbf_qp_status = "invalid_qp"
            return None

        bounds = self._dq_bounds(q_work)
        try:
            dq0 = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            dq0 = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        dq0 = self._clip_dq_to_bounds(dq0, q_work)
        slack0 = np.maximum(0.0, b_vec - a_mat @ dq0)
        x0 = np.concatenate([dq0, slack0])
        variable_bounds = bounds + [(0.0, None)] * m

        slack_weight = self.cbf_slack_weight
        if self.cbf_qp_backend == "osqp":
            return self._solve_cbf_qp_osqp(
                lhs,
                rhs,
                a_mat,
                b_vec,
                q_work,
                variable_bounds,
                x0,
                solve_start,
            )
        if minimize is None:
            self.cbf_qp_success = False
            self.cbf_qp_status = "scipy_unavailable"
            self.cbf_qp_solve_ms = (time.perf_counter() - solve_start) * 1e3
            return None

        def objective(x: np.ndarray) -> float:
            dq = x[:n]
            slack = x[n:]
            return float(
                0.5 * dq @ lhs @ dq
                - rhs @ dq
                + 0.5 * slack_weight * slack @ slack
            )

        def objective_jac(x: np.ndarray) -> np.ndarray:
            grad = np.zeros(n + m, dtype=float)
            dq = x[:n]
            slack = x[n:]
            grad[:n] = lhs @ dq - rhs
            grad[n:] = slack_weight * slack
            return grad

        def constraint_fun(x: np.ndarray) -> np.ndarray:
            return a_mat @ x[:n] + x[n:] - b_vec

        def constraint_jac(_: np.ndarray) -> np.ndarray:
            jac = np.zeros((m, n + m), dtype=float)
            jac[:, :n] = a_mat
            jac[:, n:] = np.eye(m)
            return jac

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            jac=objective_jac,
            bounds=variable_bounds,
            constraints=({"type": "ineq", "fun": constraint_fun, "jac": constraint_jac},),
            options={
                "maxiter": self.cbf_max_iterations,
                "ftol": self.cbf_ftol,
                "disp": False,
            },
        )
        if not result.success or result.x is None:
            self.cbf_qp_success = False
            self.cbf_qp_status = str(result.message)
            self.cbf_qp_slack_max = float(np.max(slack0)) if slack0.size else 0.0
            self.cbf_qp_backend_used = "slsqp"
            self.cbf_qp_iterations = int(getattr(result, "nit", 0))
            self.cbf_qp_solve_ms = (time.perf_counter() - solve_start) * 1e3
            return None
        x = np.asarray(result.x, dtype=float)
        dq = self._clip_dq_to_bounds(x[:n], q_work)
        slack = np.maximum(0.0, x[n:])
        self.cbf_qp_success = True
        self.cbf_qp_status = "ok"
        self.cbf_qp_slack_max = float(np.max(slack)) if slack.size else 0.0
        self.cbf_qp_backend_used = "slsqp"
        self.cbf_qp_iterations = int(getattr(result, "nit", 0))
        self.cbf_qp_solve_ms = (time.perf_counter() - solve_start) * 1e3
        return dq

    def _solve_cbf_qp_osqp(
        self,
        lhs: np.ndarray,
        rhs: np.ndarray,
        a_mat: np.ndarray,
        b_vec: np.ndarray,
        q_work: np.ndarray,
        variable_bounds: list[tuple[float | None, float | None]],
        x0: np.ndarray,
        solve_start: float,
    ) -> np.ndarray | None:
        n = len(self.joint_names)
        m = int(a_mat.shape[0])
        variable_count = n + m
        objective_matrix = sparse.block_diag(
            (
                sparse.csc_matrix(0.5 * (lhs + lhs.T)),
                self.cbf_slack_weight * sparse.eye(m, format="csc"),
            ),
            format="csc",
        )
        objective_matrix = sparse.triu(objective_matrix, format="csc")
        objective_vector = np.concatenate([-rhs, np.zeros(m, dtype=float)])

        cbf_matrix = sparse.hstack(
            (sparse.csc_matrix(a_mat), sparse.eye(m, format="csc")),
            format="csc",
        )
        constraint_matrix = sparse.vstack(
            (cbf_matrix, sparse.eye(variable_count, format="csc")),
            format="csc",
        )
        lower_bounds = np.asarray(
            [
                -np.inf if lower is None else float(lower)
                for lower, _ in variable_bounds
            ],
            dtype=float,
        )
        upper_bounds = np.asarray(
            [
                np.inf if upper is None else float(upper)
                for _, upper in variable_bounds
            ],
            dtype=float,
        )
        constraint_lower = np.concatenate([b_vec, lower_bounds])
        constraint_upper = np.concatenate(
            [np.full(m, np.inf, dtype=float), upper_bounds]
        )

        settings = {
            "verbose": False,
            "polishing": False,
            "warm_starting": True,
            "max_iter": self.cbf_osqp_max_iterations,
            "eps_abs": self.cbf_osqp_eps_abs,
            "eps_rel": self.cbf_osqp_eps_rel,
        }
        if self.cbf_osqp_time_limit_s > 0.0:
            settings["time_limit"] = self.cbf_osqp_time_limit_s
        try:
            solver = osqp.OSQP()
            solver.setup(
                P=objective_matrix,
                q=objective_vector,
                A=constraint_matrix,
                l=constraint_lower,
                u=constraint_upper,
                **settings,
            )
            solver.warm_start(x=x0)
            result = solver.solve(raise_error=False)
        except Exception as exc:  # noqa: BLE001
            self.cbf_qp_success = False
            self.cbf_qp_status = f"osqp_exception:{type(exc).__name__}"
            self.cbf_qp_backend_used = "osqp"
            self.cbf_qp_solve_ms = (time.perf_counter() - solve_start) * 1e3
            return None

        self.cbf_qp_backend_used = "osqp"
        self.cbf_qp_iterations = int(getattr(result.info, "iter", 0))
        self.cbf_qp_solve_ms = (time.perf_counter() - solve_start) * 1e3
        status = str(getattr(result.info, "status", "unknown")).lower()
        if result.x is None or status not in ("solved", "solved inaccurate"):
            self.cbf_qp_success = False
            self.cbf_qp_status = f"osqp:{status.replace(' ', '_')}"
            self.cbf_qp_slack_max = float(np.max(x0[n:])) if m else 0.0
            return None
        solution = np.asarray(result.x, dtype=float)
        if solution.shape != (variable_count,) or not np.all(np.isfinite(solution)):
            self.cbf_qp_success = False
            self.cbf_qp_status = "osqp:invalid_solution"
            return None
        dq = self._clip_dq_to_bounds(solution[:n], q_work)
        slack = np.maximum(0.0, solution[n:])
        self.cbf_qp_success = True
        self.cbf_qp_status = "ok"
        self.cbf_qp_slack_max = float(np.max(slack)) if slack.size else 0.0
        return dq

    def _dq_bounds(self, q_work: np.ndarray) -> list[tuple[float | None, float | None]]:
        bounds: list[tuple[float | None, float | None]] = []
        for i, idx in enumerate(self.joint_q_indices):
            lower = -np.inf
            upper = np.inf
            if self.max_adjust_per_joint > 0.0:
                lower = max(lower, -self.max_adjust_per_joint)
                upper = min(upper, self.max_adjust_per_joint)
            joint_lower = self.lower[idx]
            joint_upper = self.upper[idx]
            if np.isfinite(joint_lower):
                lower = max(lower, float(joint_lower - q_work[i]))
            if np.isfinite(joint_upper):
                upper = min(upper, float(joint_upper - q_work[i]))
            bounds.append(
                (
                    None if not np.isfinite(lower) else float(lower),
                    None if not np.isfinite(upper) else float(upper),
                )
            )
        return bounds

    def _clip_dq_to_bounds(self, dq: np.ndarray, q_work: np.ndarray) -> np.ndarray:
        dq = np.asarray(dq, dtype=float).copy()
        for i, (lower, upper) in enumerate(self._dq_bounds(q_work)):
            if lower is not None:
                dq[i] = max(float(lower), float(dq[i]))
            if upper is not None:
                dq[i] = min(float(upper), float(dq[i]))
        return dq

    def _control_dt(self) -> float:
        return 1.0 / max(self.rate_hz, 1e-6)

    def _add_esdf_constraints(
        self,
        lhs: np.ndarray,
        rhs: np.ndarray,
        states: list[SphereState],
        baseline_states: list[SphereState],
    ) -> int:
        active = 0
        escape_seen = {"left": False, "right": False}
        self.hand_bypass_active = {"left": False, "right": False}
        target_clearance = self._target_clearance()
        for idx, state in enumerate(states):
            if state.clearance is None or state.gradient is None:
                continue
            if state.clearance >= target_clearance:
                continue
            soft_delta = target_clearance - state.clearance
            avoidance_gradient = self._select_avoidance_gradient(state.gradient)
            a = avoidance_gradient.reshape(1, 3) @ state.jacobian
            a = a.reshape(-1)
            if not np.all(np.isfinite(a)) or float(np.linalg.norm(a)) < 1e-8:
                continue
            proximity = self._target_proximity(state.clearance)
            weight = self.avoidance_weight * proximity * proximity
            lhs += weight * np.outer(a, a)
            rhs += weight * a * soft_delta
            if idx < len(baseline_states):
                if self.untangle_mode:
                    self._add_untangle_tangent_preference(
                        lhs,
                        rhs,
                        state,
                        baseline_states[idx],
                        proximity,
                    )
                else:
                    if not self._is_wrist_collision_frame(state.sphere.frame):
                        self._add_xy_follow_constraint(
                            lhs,
                            rhs,
                            state,
                            baseline_states[idx],
                            proximity,
                        )
                    side = self._sphere_side(state.sphere.frame)
                    if side is not None and self._add_cable_tangent_escape(
                            lhs,
                            rhs,
                            state,
                            baseline_states[idx],
                            side,
                            proximity,
                    ):
                        escape_seen[side] = True
            if (
                state.clearance < self.safety_margin
                and not self.untangle_mode
                and self._is_hand_collision_frame(state.sphere.frame)
            ):
                self._add_downward_bias(lhs, rhs, state.jacobian, proximity, state.gradient)
            active += 1
        for side, seen in escape_seen.items():
            if not seen:
                self.tangent_escape_sign[side] = 0.0
                self.hand_bypass_active[side] = False
        return active

    def _target_clearance(self) -> float:
        return min(self.activation_margin, self.safety_margin + self.target_clearance_margin)

    def _target_proximity(self, clearance: float) -> float:
        return float(
            np.clip(
                (self._target_clearance() - float(clearance))
                / max(self._target_clearance() - self.safety_margin, 1e-6),
                0.0,
                1.0,
            )
        )

    def _activation_proximity(self, clearance: float) -> float:
        return float(
            np.clip(
                (self.activation_margin - float(clearance))
                / max(self.activation_margin - self.safety_margin, 1e-6),
                0.0,
                1.0,
            )
        )

    def _select_avoidance_gradient(self, gradient: np.ndarray) -> np.ndarray:
        gradient = np.asarray(gradient, dtype=float).reshape(3)
        # Collision safety must use the full ESDF gradient. Z/downward behavior
        # is handled by additional soft bias terms; truncating the gradient to Z
        # can let the arm pass through obstacles whose separating direction is
        # mostly lateral.
        return gradient

    def _add_xy_follow_constraint(
        self,
        lhs: np.ndarray,
        rhs: np.ndarray,
        state: SphereState,
        baseline_state: SphereState,
        proximity: float,
    ) -> None:
        if self.xy_follow_weight <= 0.0:
            return
        error_xy = np.asarray(baseline_state.position[:2] - state.position[:2], dtype=float)
        if not np.all(np.isfinite(error_xy)):
            return
        error_norm = float(np.linalg.norm(error_xy))
        if self.xy_follow_max_step > 0.0 and error_norm > self.xy_follow_max_step:
            error_xy *= self.xy_follow_max_step / max(error_norm, 1e-9)
        jac_xy = state.jacobian[:2, :]
        if not np.all(np.isfinite(jac_xy)) or float(np.linalg.norm(jac_xy)) < 1e-8:
            return
        weight = self.xy_follow_weight * float(np.clip(proximity, 0.0, 1.0))
        lhs += weight * (jac_xy.T @ jac_xy)
        rhs += weight * (jac_xy.T @ error_xy)

    def _add_untangle_tangent_preference(
        self,
        lhs: np.ndarray,
        rhs: np.ndarray,
        state: SphereState,
        baseline_state: SphereState,
        proximity: float,
    ) -> None:
        if (
            self.untangle_tangent_weight <= 0.0
            or not self.cable_capsules
            or state.gradient is None
        ):
            return
        cable_axis = self._nearest_cable_axis(state.position)
        if cable_axis is None:
            return
        normal = np.asarray(state.gradient, dtype=float).reshape(3)
        normal_norm = float(np.linalg.norm(normal))
        if not np.isfinite(normal_norm) or normal_norm < 1e-8:
            return
        normal /= normal_norm
        tangent = np.cross(cable_axis, normal)
        tangent_norm = float(np.linalg.norm(tangent))
        if not np.isfinite(tangent_norm) or tangent_norm < 1e-8:
            return
        tangent /= tangent_norm

        baseline_delta = np.asarray(
            baseline_state.position - state.position,
            dtype=float,
        ).reshape(3)
        tangent_step = float(tangent @ baseline_delta)
        tangent_step = float(
            np.clip(
                tangent_step,
                -self.untangle_tangent_max_step,
                self.untangle_tangent_max_step,
            )
        )
        if abs(tangent_step) < 1e-5:
            return
        a = (tangent.reshape(1, 3) @ state.jacobian).reshape(-1)
        if not np.all(np.isfinite(a)) or float(np.linalg.norm(a)) < 1e-8:
            return
        weight = self.untangle_tangent_weight * float(np.clip(proximity, 0.0, 1.0))
        lhs += weight * np.outer(a, a)
        rhs += weight * a * tangent_step

    @staticmethod
    def _sphere_side(frame_name: str) -> str | None:
        if "_left_" in frame_name:
            return "left"
        if "_right_" in frame_name:
            return "right"
        return None

    @staticmethod
    def _is_hand_collision_frame(frame_name: str) -> bool:
        return frame_name.endswith("_hand")

    @staticmethod
    def _is_wrist_collision_frame(frame_name: str) -> bool:
        return frame_name.endswith("_link6") or frame_name.endswith("_link7")

    def _wrist_avoidance_sides(self, states: list[SphereState]) -> set[str]:
        if not self.wrist_tcp_preservation_enabled:
            return set()
        sides: set[str] = set()
        for state in states:
            if (
                state.clearance is None
                or state.clearance >= self.activation_margin
                or not self._is_wrist_collision_frame(state.sphere.frame)
            ):
                continue
            side = self._sphere_side(state.sphere.frame)
            if side is not None:
                sides.add(side)
        return sides

    def _add_cable_tangent_escape(
        self,
        lhs: np.ndarray,
        rhs: np.ndarray,
        state: SphereState,
        baseline_state: SphereState,
        side: str,
        proximity: float,
    ) -> bool:
        if (
            not self.tangent_escape_enabled
            or self.tangent_escape_weight <= 0.0
            or self.tangent_escape_step <= 0.0
            or state.gradient is None
            or state.clearance is None
            or state.clearance >= self.tangent_escape_activation_margin
            or (
                self.tangent_escape_hand_only
                and not self._is_hand_collision_frame(state.sphere.frame)
            )
        ):
            return False
        cable_axis = self._nearest_cable_axis(state.position)
        if cable_axis is None:
            return False

        normal = np.asarray(state.gradient, dtype=float).reshape(3)
        normal_norm = float(np.linalg.norm(normal))
        if not np.isfinite(normal_norm) or normal_norm < 1e-8:
            return False
        normal /= normal_norm
        baseline_delta = np.asarray(
            baseline_state.position - state.position,
            dtype=float,
        ).reshape(3)
        inward_amount = -float(normal @ baseline_delta)
        if inward_amount <= self.tangent_escape_inward_threshold:
            return False

        tangent = np.cross(cable_axis, normal)
        tangent_norm = float(np.linalg.norm(tangent))
        if not np.isfinite(tangent_norm) or tangent_norm < 1e-8:
            return False
        tangent /= tangent_norm

        sign = self.tangent_escape_sign[side]
        if sign == 0.0:
            downward_alignment = float(tangent @ self.downward_direction)
            if (
                self.tangent_escape_prefer_downward
                and abs(downward_alignment) > 1e-4
            ):
                sign = 1.0 if downward_alignment >= 0.0 else -1.0
            else:
                baseline_tangent = float(tangent @ baseline_delta)
                sign = 1.0 if baseline_tangent >= 0.0 else -1.0
            self.tangent_escape_sign[side] = sign
        self.hand_bypass_active[side] = True

        direction = sign * tangent
        a = (direction.reshape(1, 3) @ state.jacobian).reshape(-1)
        if not np.all(np.isfinite(a)) or float(np.linalg.norm(a)) < 1e-8:
            return False
        weight = self.tangent_escape_weight * float(np.clip(proximity, 0.0, 1.0))
        lhs += weight * np.outer(a, a)
        rhs += weight * a * self.tangent_escape_step
        return True

    def _nearest_cable_axis(self, position: np.ndarray) -> np.ndarray | None:
        point = np.asarray(position, dtype=float).reshape(3)
        best_distance = float("inf")
        best_axis: np.ndarray | None = None
        for capsule in self.cable_capsules:
            segment = capsule.end - capsule.start
            length = float(np.linalg.norm(segment))
            if not np.isfinite(length) or length < 1e-9:
                continue
            axis = segment / length
            parameter = float(
                np.clip(((point - capsule.start) @ segment) / (length * length), 0.0, 1.0)
            )
            closest = capsule.start + parameter * segment
            distance = float(np.linalg.norm(point - closest) - capsule.radius)
            if distance < best_distance:
                best_distance = distance
                best_axis = axis
        return best_axis

    def _add_downward_bias(
        self,
        lhs: np.ndarray,
        rhs: np.ndarray,
        jacobian: np.ndarray,
        proximity: float,
        gradient: np.ndarray | None = None,
    ) -> None:
        if (
            not self.prefer_z_avoidance
            or self.downward_bias_weight <= 0.0
            or self.downward_bias_step <= 0.0
        ):
            return
        direction = self.downward_direction
        if self.downward_tangent_bias_enabled and gradient is not None:
            normal = np.asarray(gradient, dtype=float).reshape(3)
            normal_norm = float(np.linalg.norm(normal))
            if np.isfinite(normal_norm) and normal_norm > 1e-8:
                normal = normal / normal_norm
                tangent_down = self.downward_direction - normal * float(self.downward_direction @ normal)
                tangent_norm = float(np.linalg.norm(tangent_down))
                if tangent_norm < max(self.z_gradient_min_abs, 1e-6):
                    return
                direction = tangent_down / tangent_norm

        a = direction.reshape(1, 3) @ jacobian
        a = a.reshape(-1)
        if not np.all(np.isfinite(a)) or float(np.linalg.norm(a)) < 1e-8:
            return
        weight = self.downward_bias_weight * float(np.clip(proximity, 0.0, 1.0))
        lhs += weight * np.outer(a, a)
        rhs += weight * a * self.downward_bias_step

    def _add_inter_arm_constraints(
        self,
        lhs: np.ndarray,
        rhs: np.ndarray,
        states: list[SphereState],
    ) -> int:
        if not self.enable_inter_arm_collision or self.inter_arm_weight <= 0.0:
            return 0
        active = 0
        min_clearance = float("nan")
        for left_idx in self.left_sphere_indices:
            left = states[left_idx]
            for right_idx in self.right_sphere_indices:
                right = states[right_idx]
                delta = left.position - right.position
                distance = float(np.linalg.norm(delta))
                if not np.isfinite(distance) or distance < 1e-6:
                    continue
                clearance = distance - left.sphere.radius - right.sphere.radius
                if not np.isfinite(min_clearance) or clearance < min_clearance:
                    min_clearance = float(clearance)
                if clearance >= self.inter_arm_activation_margin:
                    continue
                direction = delta / distance
                a = direction.reshape(1, 3) @ (left.jacobian - right.jacobian)
                a = a.reshape(-1)
                if not np.all(np.isfinite(a)) or float(np.linalg.norm(a)) < 1e-8:
                    continue
                required_delta = self.inter_arm_safety_margin - clearance
                if required_delta <= 0.0:
                    soft_delta = 0.25 * (self.inter_arm_activation_margin - clearance)
                else:
                    soft_delta = required_delta
                proximity = (self.inter_arm_activation_margin - clearance) / max(
                    self.inter_arm_activation_margin - self.inter_arm_safety_margin,
                    1e-6,
                )
                proximity = float(np.clip(proximity, 0.0, 1.0))
                weight = self.inter_arm_weight * proximity * proximity
                lhs += weight * np.outer(a, a)
                rhs += weight * a * soft_delta
                active += 1
        self.min_inter_arm_clearance = min_clearance
        return active

    def _update_min_inter_arm_clearance(self, states: list[SphereState]) -> None:
        if not self.enable_inter_arm_collision:
            self.min_inter_arm_clearance = float("nan")
            return
        min_clearance = float("nan")
        for left_idx in self.left_sphere_indices:
            left = states[left_idx]
            for right_idx in self.right_sphere_indices:
                right = states[right_idx]
                distance = float(np.linalg.norm(left.position - right.position))
                if not np.isfinite(distance):
                    continue
                clearance = distance - left.sphere.radius - right.sphere.radius
                if not np.isfinite(min_clearance) or clearance < min_clearance:
                    min_clearance = float(clearance)
        self.min_inter_arm_clearance = min_clearance

    def _sample_spheres(self, states: list[SphereState], update_filter: bool = True) -> None:
        if len(self.filtered_clearances) != len(states):
            self.filtered_clearances = [None] * len(states)
        for idx, state in enumerate(states):
            if (
                idx not in self.esdf_sphere_index_set
                or self.current_obstacle_source == "none"
            ):
                state.distance = None
                state.gradient = None
                state.clearance = None
                continue
            distance, gradient = self._sample_environment(state.position)
            state.distance = distance
            state.gradient = gradient
            if distance is not None:
                raw_clearance = float(distance - state.sphere.radius)
                if (
                    update_filter
                    and 0.0 < self.clearance_filter_alpha < 1.0
                    and self.filtered_clearances[idx] is not None
                    and np.isfinite(self.filtered_clearances[idx])
                ):
                    filtered = (
                        self.clearance_filter_alpha * raw_clearance
                        + (1.0 - self.clearance_filter_alpha)
                        * float(self.filtered_clearances[idx])
                    )
                else:
                    filtered = raw_clearance
                if update_filter:
                    self.filtered_clearances[idx] = float(filtered)
                state.clearance = float(filtered)
            else:
                if update_filter:
                    self.filtered_clearances[idx] = None
                state.clearance = None

    def _sample_environment(
        self,
        position: np.ndarray,
    ) -> tuple[float | None, np.ndarray | None]:
        candidates: list[tuple[float, np.ndarray]] = []
        if self.current_obstacle_source in ("esdf", "hybrid"):
            distance, gradient = self._sample_esdf_with_virtual_extension(position)
            if distance is not None and gradient is not None:
                candidates.append((float(distance), np.asarray(gradient, dtype=float).reshape(3)))
        if self.current_obstacle_source in ("cable_capsules", "hybrid"):
            distance, gradient = self._sample_cable_capsules(position)
            if distance is not None and gradient is not None:
                candidates.append((float(distance), np.asarray(gradient, dtype=float).reshape(3)))
        if not candidates:
            return None, None
        distance, gradient = min(candidates, key=lambda item: item[0])
        return float(distance), gradient

    def _sample_cable_capsules(
        self,
        position: np.ndarray,
    ) -> tuple[float | None, np.ndarray | None]:
        return self._sample_capsule_collection(position, self.cable_capsules)

    @staticmethod
    def _sample_capsule_collection(
        position: np.ndarray,
        capsules: list[CableCapsule],
    ) -> tuple[float | None, np.ndarray | None]:
        point = np.asarray(position, dtype=float).reshape(3)
        best_distance = float("inf")
        best_gradient: np.ndarray | None = None
        for capsule in capsules:
            segment = capsule.end - capsule.start
            length_squared = float(segment @ segment)
            if length_squared <= 1e-12:
                closest = capsule.start
                axis = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                parameter = float(
                    np.clip(
                        ((point - capsule.start) @ segment) / length_squared,
                        0.0,
                        1.0,
                    )
                )
                closest = capsule.start + parameter * segment
                axis = segment / np.sqrt(length_squared)
            delta = point - closest
            centerline_distance = float(np.linalg.norm(delta))
            surface_distance = centerline_distance - capsule.radius
            if surface_distance >= best_distance:
                continue
            if centerline_distance > 1e-9:
                gradient = delta / centerline_distance
            else:
                reference = (
                    np.array([0.0, 0.0, 1.0], dtype=float)
                    if abs(float(axis[2])) < 0.9
                    else np.array([0.0, 1.0, 0.0], dtype=float)
                )
                gradient = np.cross(axis, reference)
                gradient_norm = float(np.linalg.norm(gradient))
                if gradient_norm < 1e-9:
                    gradient = np.array([0.0, 0.0, 1.0], dtype=float)
                else:
                    gradient /= gradient_norm
            best_distance = float(surface_distance)
            best_gradient = gradient.astype(float)
        if best_gradient is None or not np.isfinite(best_distance):
            return None, None
        return best_distance, best_gradient

    def _update_ground_truth_metrics(self, q_out: np.ndarray) -> None:
        if not self.ground_truth_cable_capsules:
            self.ground_truth_min_clearance = float("nan")
            return
        states = self._compute_sphere_states(self._model_q_from_bimanual(q_out))
        clearances: list[float] = []
        for idx in self.esdf_sphere_indices:
            state = states[idx]
            distance, _ = self._sample_capsule_collection(
                state.position,
                self.ground_truth_cable_capsules,
            )
            if distance is not None and np.isfinite(distance):
                clearances.append(float(distance - state.sphere.radius))
        self.ground_truth_min_clearance = (
            float(np.min(clearances)) if clearances else float("nan")
        )

    def _sample_esdf_with_virtual_extension(
        self,
        position: np.ndarray,
    ) -> tuple[float | None, np.ndarray | None]:
        if self.esdf_grid is None:
            return None, None

        best_distance, best_gradient = self.esdf_grid.sample(position)
        if self._is_valid_esdf_distance(best_distance):
            best_distance_value = float(best_distance)
        else:
            if best_distance is not None:
                self.rejected_esdf_samples += 1
            best_distance_value = None
            best_gradient = None
        for offset in self.esdf_extension_offsets:
            distance, gradient = self.esdf_grid.sample(position + offset)
            if not self._is_valid_esdf_distance(distance):
                if distance is not None:
                    self.rejected_esdf_samples += 1
                continue
            distance_value = float(distance)
            if best_distance_value is None or distance_value < best_distance_value:
                best_distance_value = distance_value
                best_gradient = gradient

        if best_distance_value is None:
            return None, None
        return best_distance_value, best_gradient

    def _is_valid_esdf_distance(self, distance: float | None) -> bool:
        if distance is None:
            return False
        value = float(distance)
        return np.isfinite(value) and value >= self.min_valid_esdf_distance

    def _smooth_safe_command(self, q_safe: np.ndarray, q_start: np.ndarray) -> np.ndarray:
        raw_delta = np.asarray(q_safe - q_start, dtype=float)
        if self.max_avoidance_delta > 0.0:
            raw_delta = np.clip(raw_delta, -self.max_avoidance_delta, self.max_avoidance_delta)
        return self._clamp_bimanual(q_start + self.avoidance_delta_alpha * raw_delta)

    def _update_cbf_master_step(self, q_baseline: np.ndarray) -> None:
        baseline = np.asarray(q_baseline, dtype=float)
        if self.previous_cbf_master_baseline is None:
            self.cbf_master_step.fill(0.0)
        else:
            self.cbf_master_step = baseline - self.previous_cbf_master_baseline
        self.previous_cbf_master_baseline = baseline.copy()

    def _apply_cbf_master_motion_sync(
        self,
        q_safe: np.ndarray,
        q_start: np.ndarray,
    ) -> np.ndarray:
        if (
            not self.cbf_master_motion_sync_enabled
            or any(self.assisted_grasp_active.values())
        ):
            self.cbf_master_motion_scale = {"left": 1.0, "right": 1.0}
            self.cbf_master_motion_catchup_step = {"left": 0.0, "right": 0.0}
            self.cbf_safe_baseline_catchup_active = {"left": False, "right": False}
            return q_safe

        synchronized = np.asarray(q_safe, dtype=float).copy()
        reference = self.last_command if self.last_command is not None else q_start
        dt = self._control_dt()
        for side, side_slice in (
            ("left", self.left_slice),
            ("right", self.right_slice),
        ):
            current_clearance = (
                self.left_min_clearance if side == "left" else self.right_min_clearance
            )
            baseline_clearance = (
                self.baseline_left_min_clearance
                if side == "left"
                else self.baseline_right_min_clearance
            )
            safe_baseline_catchup = bool(
                self.cbf_safe_baseline_catchup_enabled
                and np.isfinite(current_clearance)
                and np.isfinite(baseline_clearance)
                and baseline_clearance >= self.safety_margin
                and baseline_clearance
                >= current_clearance + self.cbf_safe_baseline_catchup_margin
            )
            self.cbf_safe_baseline_catchup_active[side] = safe_baseline_catchup
            speed = float(np.max(np.abs(self.baseline_velocity[side])))
            master_step_norm = float(np.linalg.norm(self.cbf_master_step[side_slice]))
            moving = self.cbf_master_motion_active[side]
            if moving:
                moving = bool(
                    speed > self.cbf_master_motion_release_velocity
                    or master_step_norm
                    > self.cbf_master_motion_release_velocity * dt
                )
            else:
                moving = bool(
                    speed >= self.cbf_master_motion_enter_velocity
                    or master_step_norm
                    >= self.cbf_master_motion_enter_velocity * dt
                )
            if (
                self.hand_bypass_allow_autonomous_completion
                and self.hand_bypass_active[side]
            ):
                moving = True
            if safe_baseline_catchup:
                moving = True
            self.cbf_master_motion_active[side] = moving

            if not moving:
                synchronized[side_slice] = reference[side_slice]
                self.cbf_master_motion_scale[side] = 0.0
                self.cbf_master_motion_catchup_step[side] = 0.0
                self.command_governor.reset_indices(side_slice)
                continue

            proposed_step = synchronized[side_slice] - reference[side_slice]
            proposed_norm = float(np.linalg.norm(proposed_step))
            if safe_baseline_catchup:
                self.cbf_master_motion_scale[side] = 1.0
                self.cbf_master_motion_catchup_step[side] = proposed_norm
                continue
            tracking_error = float(
                np.linalg.norm(
                    self.previous_cbf_master_baseline[side_slice]
                    - reference[side_slice]
                )
            )
            catchup_step = min(
                self.cbf_master_motion_max_catchup_step,
                self.cbf_master_motion_catchup_error_gain * tracking_error,
            )
            self.cbf_master_motion_catchup_step[side] = catchup_step
            master_motion_budget = self.cbf_master_motion_max_step_gain * max(
                master_step_norm,
                speed * dt,
            ) + catchup_step
            if proposed_norm > master_motion_budget > 0.0:
                scale = master_motion_budget / proposed_norm
                synchronized[side_slice] = (
                    reference[side_slice] + scale * proposed_step
                )
                self.cbf_master_motion_scale[side] = float(scale)
            elif master_motion_budget <= 0.0:
                synchronized[side_slice] = reference[side_slice]
                self.cbf_master_motion_scale[side] = 0.0
                self.command_governor.reset_indices(side_slice)
            else:
                self.cbf_master_motion_scale[side] = 1.0
        return synchronized

    def _apply_avoidance_release_ramp(self, q_target: np.ndarray) -> np.ndarray:
        if (
            self.last_command is None
            or self.avoidance_release_ramp_duration <= 0.0
            or self.avoidance_release_progress >= 1.0
        ):
            self.avoidance_release_progress = 1.0
            return q_target
        increment = self._control_dt() / self.avoidance_release_ramp_duration
        self.avoidance_release_progress = float(
            np.clip(self.avoidance_release_progress + increment, 0.0, 1.0)
        )
        progress = self.avoidance_release_progress
        smooth_progress = progress * progress * (3.0 - 2.0 * progress)
        scale = self.avoidance_release_min_scale + (
            1.0 - self.avoidance_release_min_scale
        ) * smooth_progress
        return self.last_command + scale * (q_target - self.last_command)

    def _limit_command_step(
        self,
        q_safe: np.ndarray,
        q_baseline: np.ndarray,
        enforce_acceleration: bool = True,
    ) -> np.ndarray:
        if self.command_governor_enabled:
            if self.last_command is None:
                self.command_governor.reset()
                self.command_governor_tracking_error.fill(0.0)
                return np.asarray(q_safe, dtype=float)
            governed = self.command_governor.step(
                self.last_command,
                q_safe,
                self._control_dt(),
            )
            self.command_governor_tracking_error = np.asarray(
                q_safe - governed,
                dtype=float,
            )
            return governed

        reference = self.last_command if self.last_command is not None else q_baseline
        delta = np.asarray(q_safe - reference, dtype=float)
        target_delta = delta.copy()
        if self.max_command_step > 0.0:
            delta = np.clip(delta, -self.max_command_step, self.max_command_step)
        if (
            enforce_acceleration
            and self.max_command_acceleration > 0.0
            and self.last_command_step is not None
        ):
            max_step_change = self.max_command_acceleration / max(self.rate_hz * self.rate_hz, 1e-6)
            delta = np.clip(
                delta,
                self.last_command_step - max_step_change,
                self.last_command_step + max_step_change,
            )
        if (
            enforce_acceleration
            and self.max_command_jerk > 0.0
            and self.last_command_step is not None
            and self.last_command_step_change is not None
        ):
            max_step_change_delta = self.max_command_jerk / max(
                self.rate_hz * self.rate_hz * self.rate_hz,
                1e-6,
            )
            step_change = delta - self.last_command_step
            step_change = np.clip(
                step_change,
                self.last_command_step_change - max_step_change_delta,
                self.last_command_step_change + max_step_change_delta,
            )
            delta = self.last_command_step + step_change
        # A rate limiter must never keep moving in the previous direction after
        # the position target reverses, or move farther than the current error.
        moving_away = delta * target_delta < 0.0
        delta[moving_away] = 0.0
        delta = np.sign(target_delta) * np.minimum(np.abs(delta), np.abs(target_delta))
        return reference + delta

    def _reset_command_governor(self) -> None:
        self.command_governor.reset()
        self.command_governor_tracking_error.fill(0.0)

    def _publish_commands(self, q_out: np.ndarray) -> None:
        left = [float(v) for v in q_out[self.left_slice]]
        right = [float(v) for v in q_out[self.right_slice]]
        self.left_command_pub.publish(Float64MultiArray(data=left + list(self.latest_left_extra)))
        self.right_command_pub.publish(Float64MultiArray(data=right + list(self.latest_right_extra)))

    def _publish_debug(self, q_out: np.ndarray) -> None:
        if np.isfinite(self.min_clearance):
            self.min_clearance_pub.publish(Float32(data=float(self.min_clearance)))
        if np.isfinite(self.left_min_clearance):
            self.left_min_clearance_pub.publish(Float32(data=float(self.left_min_clearance)))
        if np.isfinite(self.right_min_clearance):
            self.right_min_clearance_pub.publish(Float32(data=float(self.right_min_clearance)))
        mode = "monitor" if self.monitor_only else "active"
        status = (
            f"mode={mode} solver={self.avoidance_solver} "
            f"obstacle_source={self.obstacle_source}/{self.current_obstacle_source} "
            f"cable_capsules={len(self.cable_capsules)} "
            f"cable_age={self._cable_capsule_age():.3f} "
            f"joint_state_age={self._joint_state_age():.3f} "
            f"joint_state_fresh={self._joint_state_is_fresh()} "
            f"min_clearance={self.min_clearance:.3f} "
            f"target_clearance={self._target_clearance():.3f} "
            f"activation_margin={self.activation_margin:.3f} "
            f"clearance_filter_alpha={self.clearance_filter_alpha:.2f} "
            f"min_sphere={self.min_clearance_sphere} "
            f"min_pos={np.array2string(self.min_clearance_position, precision=3, suppress_small=True)} "
            f"left_min={self.left_min_clearance:.3f} left_min_sphere={self.left_min_clearance_sphere} "
            f"right_min={self.right_min_clearance:.3f} right_min_sphere={self.right_min_clearance_sphere} "
            f"baseline_min_clearance={self.baseline_min_clearance:.3f} "
            f"baseline_side_min={self.baseline_left_min_clearance:.3f}/"
            f"{self.baseline_right_min_clearance:.3f} "
            f"min_inter_arm_clearance={self.min_inter_arm_clearance:.3f} "
            f"baseline_error={self.baseline_error_norm:.3f} "
            f"hold_invalid={self.hold_due_to_invalid_clearance} "
            f"hold_latch={self.hold_due_to_avoidance_latch} "
            f"hold_missing_source={self.hold_due_to_missing_obstacle_source} "
            f"untangle={self.untangle_mode} "
            f"grasp_target_left={self.grasp_targets['left'] is not None} "
            f"grasp_target_right={self.grasp_targets['right'] is not None} "
            f"assist_left={self.assisted_grasp_active['left']}:{self.assisted_grasp_alpha['left']:.2f} "
            f"assist_right={self.assisted_grasp_active['right']}:{self.assisted_grasp_alpha['right']:.2f} "
            f"gripper_left={self._format_gripper_fraction(self.assisted_grasp_gripper_fraction['left'])} "
            f"gripper_right={self._format_gripper_fraction(self.assisted_grasp_gripper_fraction['right'])} "
            f"latched={self.avoidance_latched} "
            f"escape_sign_left={self.tangent_escape_sign['left']:.0f} "
            f"escape_sign_right={self.tangent_escape_sign['right']:.0f} "
            f"hand_bypass={self.hand_bypass_active['left']}/"
            f"{self.hand_bypass_active['right']} "
            f"wrist_tcp_preservation={self.wrist_tcp_preservation_active['left']}/"
            f"{self.wrist_tcp_preservation_active['right']} "
            f"release_count={self.avoidance_release_counter} "
            f"release_progress={self.avoidance_release_progress:.3f} "
            f"active_constraints={self.active_constraints} "
            f"active_esdf={self.active_esdf_constraints} active_inter_arm={self.active_inter_arm_constraints} "
            f"rollout_active={self.rollout_active} rollout_selected={self.rollout_selected} "
            f"rollout_candidates={self.rollout_candidate_count} "
            f"rollout_cost={self.rollout_best_cost:.3f}/{self.rollout_baseline_cost:.3f} "
            f"cbf_ok={self.cbf_qp_success} cbf_status={self.cbf_qp_status} "
            f"cbf_slack_max={self.cbf_qp_slack_max:.4f} "
            f"cbf_master_sync={self.cbf_master_motion_sync_enabled} "
            f"cbf_master_motion={self.cbf_master_motion_active['left']}/"
            f"{self.cbf_master_motion_active['right']} "
            f"cbf_master_scale={self.cbf_master_motion_scale['left']:.2f}/"
            f"{self.cbf_master_motion_scale['right']:.2f} "
            f"cbf_master_catchup={self.cbf_master_motion_catchup_step['left']:.4f}/"
            f"{self.cbf_master_motion_catchup_step['right']:.4f} "
            f"cbf_safe_catchup={self.cbf_safe_baseline_catchup_active['left']}/"
            f"{self.cbf_safe_baseline_catchup_active['right']} "
            f"qp_backend={self.cbf_qp_backend_used} "
            f"qp_ms={self.cbf_qp_solve_ms:.3f} qp_iter={self.cbf_qp_iterations} "
            f"antisway={self.antisway_enabled}/{self.antisway_monitor_only} "
            f"antisway_ready={self._antisway_observer_ready()} "
            f"antisway_applied="
            f"{self.antisway_cost_applied or self.antisway_predictive_applied or self.antisway_shaper_applied or self.antisway_bimanual_applied} "
            f"antisway_nis={self.antisway_observer_nis:.3f} "
            f"antisway_confidence={self.antisway_raw_confidence:.3f}/"
            f"{self.antisway_confidence:.3f} "
            f"antisway_gates={self.antisway_observer_gate_active}/"
            f"{self.antisway_bimanual_motion_gate_active} "
            f"antisway_predictive={self.antisway_predictive_enabled}/"
            f"{self.antisway_predictive_applied} "
            f"antisway_mpc_status={self.antisway_mpc_status} "
            f"antisway_mpc_ms={self.antisway_mpc_solve_ms:.3f} "
            f"antisway_mpc_iter={self.antisway_mpc_iterations} "
            f"antisway_mpc_accel_error="
            f"{np.linalg.norm(self.antisway_mpc_acceleration_error):.3f} "
            f"antisway_shaper={self.antisway_input_shaper_enabled}/"
            f"{self.antisway_shaper_applied} "
            f"antisway_shaper_delta="
            f"{np.linalg.norm(self.antisway_shaper_correction):.4f} "
            f"antisway_bimanual={self.antisway_bimanual_task_qp_enabled}/"
            f"{self.antisway_bimanual_applied} "
            f"antisway_bimanual_status={self.antisway_bimanual_status} "
            f"antisway_bimanual_ms={self.antisway_bimanual_solve_ms:.3f} "
            f"antisway_bimanual_nullity={self.antisway_bimanual_nullity} "
            f"antisway_bimanual_task_residual="
            f"{self.antisway_bimanual_task_residual:.3e} "
            f"antisway_bimanual_delta="
            f"{np.linalg.norm(self.antisway_bimanual_position_correction):.4f} "
            f"governor={self.command_governor_enabled}/{self.command_governor.active} "
            f"governor_error={np.max(np.abs(self.command_governor_tracking_error)):.4f} "
            f"governor_velocity={np.max(np.abs(self.command_governor.velocity)):.3f} "
            f"governor_acceleration={np.max(np.abs(self.command_governor.acceleration)):.3f} "
            f"sampled_spheres={self.sampled_spheres}/{len(self.spheres)} "
            f"rejected_esdf={self.rejected_esdf_samples} "
            f"esdf_grid={self.esdf_grid_shape} esdf_observed={self.esdf_observed_count} "
            f"clear_robot={self.clear_robot_from_esdf} "
            f"left={np.array2string(q_out[self.left_slice], precision=3, suppress_small=True)} "
            f"right={np.array2string(q_out[self.right_slice], precision=3, suppress_small=True)}"
        )
        self.status_pub.publish(String(data=status))
        self._publish_structured_status()
        self._throttled_info(status)

    def _publish_structured_status(self) -> None:
        msg = SafetyStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.global_frame
        msg.mode = "monitor" if self.monitor_only else "active"
        msg.solver = self.avoidance_solver
        msg.obstacle_source = f"{self.obstacle_source}/{self.current_obstacle_source}"
        if self.untangle_mode:
            msg.task_phase = "untangle"
        elif any(self.assisted_grasp_active.values()):
            msg.task_phase = "assisted_grasp"
        elif any(target is not None for target in self.grasp_targets.values()):
            msg.task_phase = "target_selected"
        else:
            msg.task_phase = "teleoperation"
        msg.qp_status = self.cbf_qp_status
        msg.cbf_success = bool(self.cbf_qp_success)
        msg.holding = bool(
            self.hold_due_to_invalid_clearance
            or self.hold_due_to_avoidance_latch
            or self.hold_due_to_missing_obstacle_source
        )
        truth_available = bool(
            self.ground_truth_cable_capsules
            and bool(np.isfinite(self.ground_truth_min_clearance))
        )
        msg.ground_truth_available = bool(truth_available)
        msg.ground_truth_collision = bool(
            truth_available and self.ground_truth_min_clearance < 0.0
        )
        msg.ground_truth_margin_violation = bool(
            truth_available and self.ground_truth_min_clearance < self.safety_margin
        )
        msg.estimated_min_clearance = float(self.min_clearance)
        msg.ground_truth_min_clearance = float(self.ground_truth_min_clearance)
        msg.left_min_clearance = float(self.left_min_clearance)
        msg.right_min_clearance = float(self.right_min_clearance)
        msg.inter_arm_min_clearance = float(self.min_inter_arm_clearance)
        msg.baseline_min_clearance = float(self.baseline_min_clearance)
        msg.command_deviation_norm = float(self.baseline_error_norm)
        msg.max_slack = float(self.cbf_qp_slack_max)
        msg.cable_estimate_age = float(self._cable_capsule_age())
        msg.cable_ground_truth_age = float(self._ground_truth_cable_age())
        msg.active_constraints = int(max(0, self.active_constraints))
        msg.active_environment_constraints = int(max(0, self.active_esdf_constraints))
        msg.active_inter_arm_constraints = int(max(0, self.active_inter_arm_constraints))
        msg.estimated_cable_count = len(self.cable_capsules)
        msg.ground_truth_cable_count = len(self.ground_truth_cable_capsules)
        self.structured_status_pub.publish(msg)

    def _publish_markers(self, states: list[SphereState]) -> None:
        array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        array.markers.append(delete_marker)
        if self.collision_model == "capsule":
            self._append_capsule_markers(array, states, stamp)
        else:
            self._append_sphere_markers(array, states, stamp)
        if self.publish_avoidance_arrows:
            self._append_avoidance_force_arrows(array, states, stamp)
        self.marker_pub.publish(array)

    def _append_avoidance_force_arrows(
        self,
        array: MarkerArray,
        states: list[SphereState],
        stamp,
    ) -> None:
        target_clearance = self._target_clearance()
        most_critical_by_link: dict[str, SphereState] = {}
        for state in states:
            if (
                state.clearance is None
                or state.gradient is None
                or not np.isfinite(state.clearance)
                or state.clearance >= target_clearance
            ):
                continue
            gradient = np.asarray(state.gradient, dtype=float).reshape(3)
            gradient_norm = float(np.linalg.norm(gradient))
            if not np.all(np.isfinite(gradient)) or gradient_norm < 1e-8:
                continue
            previous = most_critical_by_link.get(state.sphere.frame)
            if previous is None or state.clearance < previous.clearance:
                most_critical_by_link[state.sphere.frame] = state

        active_states = sorted(
            most_critical_by_link.values(),
            key=lambda state: float(state.clearance),
        )[: self.avoidance_arrow_max_count]

        for marker_id, state in enumerate(active_states):
            direction = np.asarray(state.gradient, dtype=float).reshape(3)
            direction /= float(np.linalg.norm(direction))
            proximity = self._target_proximity(float(state.clearance))
            length = self.avoidance_arrow_min_length + proximity * (
                self.avoidance_arrow_max_length - self.avoidance_arrow_min_length
            )

            marker = Marker()
            marker.header.frame_id = self.global_frame
            marker.header.stamp = stamp
            marker.ns = "avoidance_push_arrows"
            marker.id = marker_id
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.points = [
                _point_from_array(state.position),
                _point_from_array(state.position + length * direction),
            ]
            marker.scale.x = 0.012
            marker.scale.y = 0.026
            marker.scale.z = min(0.035, max(0.018, 0.35 * length))
            marker.color.a = 0.95
            marker.color.r = 1.0
            marker.color.g = 0.05 if state.clearance < self.safety_margin else 0.55
            marker.color.b = 0.02
            array.markers.append(marker)

    def _append_sphere_markers(self, array: MarkerArray, states: list[SphereState], stamp) -> None:
        for idx, state in enumerate(states):
            marker = Marker()
            marker.header.frame_id = self.global_frame
            marker.header.stamp = stamp
            marker.ns = "bimanual_esdf_spheres"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = _point_from_array(state.position)
            marker.pose.orientation.w = 1.0
            diameter = 2.0 * float(state.sphere.radius + self.safety_margin)
            marker.scale.x = diameter
            marker.scale.y = diameter
            marker.scale.z = diameter
            self._set_collision_marker_color(marker, state.clearance, state.sphere.frame, alpha=0.42)
            array.markers.append(marker)

    def _append_capsule_markers(self, array: MarkerArray, states: list[SphereState], stamp) -> None:
        groups: dict[tuple[str, float], list[SphereState]] = {}
        for state in states:
            key = (state.sphere.frame, round(float(state.sphere.radius), 5))
            groups.setdefault(key, []).append(state)

        marker_id = 0
        for (frame, radius), group in groups.items():
            positions = [np.asarray(state.position, dtype=float).reshape(3) for state in group]
            clearance = self._min_group_clearance(group)
            if len(positions) < 2:
                marker = self._make_sphere_marker(
                    stamp,
                    marker_id,
                    "bimanual_esdf_capsules",
                    positions[0],
                    radius,
                    clearance,
                    frame,
                )
                array.markers.append(marker)
                marker_id += 1
                continue

            start, end = self._farthest_points(positions)
            axis = end - start
            length = float(np.linalg.norm(axis))
            if not np.isfinite(length) or length < 1e-5:
                marker = self._make_sphere_marker(
                    stamp,
                    marker_id,
                    "bimanual_esdf_capsules",
                    start,
                    radius,
                    clearance,
                    frame,
                )
                array.markers.append(marker)
                marker_id += 1
                continue

            cylinder = Marker()
            cylinder.header.frame_id = self.global_frame
            cylinder.header.stamp = stamp
            cylinder.ns = "bimanual_esdf_capsules"
            cylinder.id = marker_id
            marker_id += 1
            cylinder.type = Marker.CYLINDER
            cylinder.action = Marker.ADD
            cylinder.pose.position = _point_from_array(0.5 * (start + end))
            qx, qy, qz, qw = self._quaternion_from_z_axis(axis)
            cylinder.pose.orientation.x = qx
            cylinder.pose.orientation.y = qy
            cylinder.pose.orientation.z = qz
            cylinder.pose.orientation.w = qw
            diameter = 2.0 * float(radius + self.safety_margin)
            cylinder.scale.x = diameter
            cylinder.scale.y = diameter
            cylinder.scale.z = length
            self._set_collision_marker_color(cylinder, clearance, frame, alpha=0.34)
            array.markers.append(cylinder)

            for endpoint in (start, end):
                marker = self._make_sphere_marker(
                    stamp,
                    marker_id,
                    "bimanual_esdf_capsules",
                    endpoint,
                    radius,
                    clearance,
                    frame,
                )
                array.markers.append(marker)
                marker_id += 1

    def _make_sphere_marker(
        self,
        stamp,
        marker_id: int,
        namespace: str,
        position: np.ndarray,
        radius: float,
        clearance: float | None,
        frame: str,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = _point_from_array(position)
        marker.pose.orientation.w = 1.0
        diameter = 2.0 * float(radius + self.safety_margin)
        marker.scale.x = diameter
        marker.scale.y = diameter
        marker.scale.z = diameter
        self._set_collision_marker_color(marker, clearance, frame, alpha=0.34)
        return marker

    def _min_group_clearance(self, group: list[SphereState]) -> float | None:
        clearances = [
            float(state.clearance)
            for state in group
            if state.clearance is not None and np.isfinite(state.clearance)
        ]
        if not clearances:
            return None
        return float(np.min(clearances))

    def _farthest_points(self, positions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        best_start = positions[0]
        best_end = positions[-1]
        best_distance = -1.0
        for i, start in enumerate(positions):
            for end in positions[i + 1 :]:
                distance = float(np.linalg.norm(end - start))
                if distance > best_distance:
                    best_distance = distance
                    best_start = start
                    best_end = end
        return best_start, best_end

    def _quaternion_from_z_axis(self, axis: np.ndarray) -> tuple[float, float, float, float]:
        target = np.asarray(axis, dtype=float).reshape(3)
        norm = float(np.linalg.norm(target))
        if not np.isfinite(norm) or norm < 1e-9:
            return 0.0, 0.0, 0.0, 1.0
        target /= norm
        source = np.array([0.0, 0.0, 1.0], dtype=float)
        dot = float(np.clip(source @ target, -1.0, 1.0))
        if dot > 1.0 - 1e-9:
            return 0.0, 0.0, 0.0, 1.0
        if dot < -1.0 + 1e-9:
            return 1.0, 0.0, 0.0, 0.0
        cross = np.cross(source, target)
        quat = np.array([cross[0], cross[1], cross[2], 1.0 + dot], dtype=float)
        quat_norm = float(np.linalg.norm(quat))
        if not np.isfinite(quat_norm) or quat_norm < 1e-9:
            return 0.0, 0.0, 0.0, 1.0
        quat /= quat_norm
        return float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])

    def _set_collision_marker_color(
        self,
        marker: Marker,
        clearance: float | None,
        frame: str,
        alpha: float,
    ) -> None:
        marker.color.a = float(alpha)
        if clearance is None:
            marker.color.r = 0.4
            marker.color.g = 0.4
            marker.color.b = 0.4
        elif clearance < self.safety_margin:
            marker.color.r = 1.0
            marker.color.g = 0.05
            marker.color.b = 0.02
        elif clearance < self.activation_margin:
            marker.color.r = 1.0
            marker.color.g = 0.75
            marker.color.b = 0.0
        elif "_left_" in frame:
            marker.color.r = 0.0
            marker.color.g = 0.55
            marker.color.b = 1.0
        else:
            marker.color.r = 0.0
            marker.color.g = 0.85
            marker.color.b = 0.25

    def _throttled_info(self, message: str) -> None:
        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds > 2_000_000_000:
            self.get_logger().info(message)
            self.last_log_time = now

    def _throttled_warn(self, message: str) -> None:
        self.get_logger().warn(message, throttle_duration_sec=2.0)

    def destroy_node(self) -> bool:
        temp_path = self._temp_urdf_path
        result = super().destroy_node()
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        return result


def main() -> None:
    rclpy.init()
    node = BimanualEsdfAvoidanceFilter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
