from __future__ import annotations

import os
import tempfile

import numpy as np
import pinocchio as pin
import rclpy
from geometry_msgs.msg import Point, Vector3
from nvblox_msgs.srv import EsdfAndGradients
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

from openarmx_obstacle_avoidance.right_arm_esdf_avoidance_filter import (
    RIGHT_JOINT_NAMES,
    CollisionSphere,
    EsdfGrid,
    SphereState,
    _point_from_array,
    _skew,
    _vector_from_array,
)


class RightArmEsdfPredictivePlanner(Node):
    def __init__(self) -> None:
        super().__init__("right_arm_esdf_predictive_planner")

        self.declare_parameter("urdf_path", "")
        self.declare_parameter("robot_description_node", "/robot_state_publisher")
        self.declare_parameter("joint_names", RIGHT_JOINT_NAMES)
        self.declare_parameter("command_message_type", "float64_multi_array")
        self.declare_parameter("input_command_topic", "/right_teleop_baseline/commands")
        self.declare_parameter("output_command_topic", "/right_teleop_planned/commands")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("esdf_service", "")
        self.declare_parameter("global_frame", "world")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("horizon_steps", 6)
        self.declare_parameter("horizon_dt", 0.1)
        self.declare_parameter("esdf_update_hz", 6.0)
        self.declare_parameter("aabb_padding", 0.25)
        self.declare_parameter("unobserved_value", -1000.0)
        self.declare_parameter("request_update_esdf", True)
        self.declare_parameter("nearest_observed_search_radius", 0.12)

        self.declare_parameter("monitor_only", False)
        self.declare_parameter("safety_margin", 0.02)
        self.declare_parameter("activation_margin", 0.12)
        self.declare_parameter("track_weight", 1.0)
        self.declare_parameter("obstacle_weight", 1.0)
        self.declare_parameter("damping", 0.05)
        self.declare_parameter("correction_alpha", 0.6)
        self.declare_parameter("plan_delta_filter_alpha", 0.35)
        self.declare_parameter("inactive_delta_decay", 0.45)
        self.declare_parameter("active_hold_cycles", 3)
        self.declare_parameter("max_plan_delta_per_joint", 0.06)
        self.declare_parameter("max_command_step", 0.06)
        self.declare_parameter("qdot_filter_alpha", 0.35)
        self.declare_parameter("qdot_limit", 1.2)
        self.declare_parameter("publish_markers", True)
        self.declare_parameter("marker_topic", "/openarm/right_arm/predicted_spheres")
        self.declare_parameter("min_clearance_topic", "/openarm/right_arm/predictive_min_clearance")
        self.declare_parameter("status_topic", "/openarm/right_arm/predictive_planner_status")

        self.urdf_path = str(self.get_parameter("urdf_path").value).strip()
        self.robot_description_node = str(self.get_parameter("robot_description_node").value)
        self.joint_names = list(self.get_parameter("joint_names").value)
        self.command_message_type = str(self.get_parameter("command_message_type").value).strip().lower()
        self.input_command_topic = str(self.get_parameter("input_command_topic").value)
        self.output_command_topic = str(self.get_parameter("output_command_topic").value)
        self.joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self.esdf_service_name = str(self.get_parameter("esdf_service").value).strip()
        self.global_frame = str(self.get_parameter("global_frame").value)
        self.rate_hz = max(1.0, float(self.get_parameter("rate_hz").value))
        self.horizon_steps = max(1, int(self.get_parameter("horizon_steps").value))
        self.horizon_dt = max(0.02, float(self.get_parameter("horizon_dt").value))
        self.esdf_update_hz = max(0.5, float(self.get_parameter("esdf_update_hz").value))
        self.aabb_padding = max(0.05, float(self.get_parameter("aabb_padding").value))
        self.unobserved_value = float(self.get_parameter("unobserved_value").value)
        self.request_update_esdf = bool(self.get_parameter("request_update_esdf").value)
        self.nearest_observed_search_radius = max(
            0.0,
            float(self.get_parameter("nearest_observed_search_radius").value),
        )

        self.monitor_only = bool(self.get_parameter("monitor_only").value)
        self.safety_margin = max(0.0, float(self.get_parameter("safety_margin").value))
        self.activation_margin = max(
            self.safety_margin,
            float(self.get_parameter("activation_margin").value),
        )
        self.track_weight = max(1e-6, float(self.get_parameter("track_weight").value))
        self.obstacle_weight = max(0.0, float(self.get_parameter("obstacle_weight").value))
        self.damping = max(1e-6, float(self.get_parameter("damping").value))
        self.correction_alpha = float(
            np.clip(float(self.get_parameter("correction_alpha").value), 0.0, 1.0)
        )
        self.plan_delta_filter_alpha = float(
            np.clip(float(self.get_parameter("plan_delta_filter_alpha").value), 0.0, 1.0)
        )
        self.inactive_delta_decay = float(
            np.clip(float(self.get_parameter("inactive_delta_decay").value), 0.0, 1.0)
        )
        self.active_hold_cycles = max(0, int(self.get_parameter("active_hold_cycles").value))
        self.max_plan_delta_per_joint = max(
            0.0,
            float(self.get_parameter("max_plan_delta_per_joint").value),
        )
        self.max_command_step = max(0.0, float(self.get_parameter("max_command_step").value))
        self.qdot_filter_alpha = float(
            np.clip(float(self.get_parameter("qdot_filter_alpha").value), 0.0, 1.0)
        )
        self.qdot_limit = max(0.0, float(self.get_parameter("qdot_limit").value))
        self.publish_markers = bool(self.get_parameter("publish_markers").value)

        self._temp_urdf_path: str | None = None
        if not self.urdf_path:
            self.urdf_path = self._write_robot_description_to_temp_urdf()
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        self.neutral_q = pin.neutral(self.model)
        self.lower = np.asarray(self.model.lowerPositionLimit, dtype=float).copy()
        self.upper = np.asarray(self.model.upperPositionLimit, dtype=float).copy()
        self.joint_q_indices = self._joint_q_indices(self.joint_names)
        self.joint_v_indices = self._joint_v_indices(self.joint_names)
        self.spheres = self._default_right_arm_spheres()
        self.sphere_frame_ids = [self.model.getFrameId(sphere.frame) for sphere in self.spheres]

        self.q_model_current = self.neutral_q.copy()
        self.have_joint_state = False
        self.latest_baseline: np.ndarray | None = None
        self.previous_baseline: np.ndarray | None = None
        self.latest_baseline_time = self.get_clock().now()
        self.previous_baseline_time = self.latest_baseline_time
        self.qdot_ref = np.zeros(len(self.joint_names), dtype=float)
        self.latest_extra: list[float] = []
        self.latest_trajectory: JointTrajectory | None = None
        self.last_command: np.ndarray | None = None
        self.last_plan_delta = np.zeros(len(self.joint_names), dtype=float)
        self.inactive_cycles = 0
        self.esdf_grid: EsdfGrid | None = None
        self.esdf_pending = False
        self.last_esdf_request_time = self.get_clock().now()
        self.last_log_time = self.get_clock().now()
        self.min_clearance = float("nan")
        self.predicted_min_clearance = float("nan")
        self.active_constraints = 0
        self.sampled_spheres = 0
        self.esdf_grid_shape = "none"
        self.esdf_observed_count = 0

        self.esdf_client = None
        self._connect_esdf_client()

        self.create_subscription(JointState, self.joint_states_topic, self._joint_state_cb, 20)
        if self.command_message_type in ("float64_multi_array", "std_msgs/msg/float64multiarray"):
            self.command_message_type = "float64_multi_array"
            self.create_subscription(Float64MultiArray, self.input_command_topic, self._float_command_cb, 10)
            self.command_pub = self.create_publisher(Float64MultiArray, self.output_command_topic, 10)
        elif self.command_message_type in ("joint_trajectory", "trajectory_msgs/msg/jointtrajectory"):
            self.command_message_type = "joint_trajectory"
            self.create_subscription(JointTrajectory, self.input_command_topic, self._trajectory_command_cb, 10)
            self.command_pub = self.create_publisher(JointTrajectory, self.output_command_topic, 10)
        else:
            raise RuntimeError(
                "Unsupported command_message_type "
                f"{self.command_message_type!r}; use float64_multi_array or joint_trajectory"
            )

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
        self.status_pub = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            10,
        )
        self.timer = self.create_timer(1.0 / self.rate_hz, self._timer_cb)

        self.get_logger().info(f"URDF: {self.urdf_path}")
        self.get_logger().info(f"right joints: {self.joint_names}")
        self.get_logger().info(f"predictive planner: {self.input_command_topic} -> {self.output_command_topic}")
        self.get_logger().info(
            "planner: "
            f"rate_hz={self.rate_hz:.1f}, horizon_steps={self.horizon_steps}, "
            f"horizon_dt={self.horizon_dt:.3f}, monitor_only={self.monitor_only}, "
            f"safety_margin={self.safety_margin:.3f}, activation_margin={self.activation_margin:.3f}, "
            f"obstacle_weight={self.obstacle_weight:.3f}, correction_alpha={self.correction_alpha:.3f}, "
            f"plan_delta_filter_alpha={self.plan_delta_filter_alpha:.3f}, "
            f"inactive_delta_decay={self.inactive_delta_decay:.3f}, "
            f"active_hold_cycles={self.active_hold_cycles}"
        )

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

        fd, path = tempfile.mkstemp(prefix="openarmx_esdf_planner_", suffix=".urdf")
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
        indices = []
        for name in joint_names:
            indices.append(int(self.model.joints[self.model.getJointId(name)].idx_v))
        return indices

    def _default_right_arm_spheres(self) -> list[CollisionSphere]:
        raw = [
            ("openarm_right_link1", [0.0, 0.0, 0.03], 0.055),
            ("openarm_right_link2", [0.0, 0.0, 0.02], 0.050),
            ("openarm_right_link2", [0.0, 0.0, 0.075], 0.050),
            ("openarm_right_link3", [0.0, 0.0, 0.04], 0.055),
            ("openarm_right_link3", [0.0, 0.0, 0.10], 0.055),
            ("openarm_right_link4", [0.0, -0.015, 0.035], 0.035),
            ("openarm_right_link4", [0.0, -0.025, 0.095], 0.035),
            ("openarm_right_link5", [0.0, 0.0, 0.03], 0.035),
            ("openarm_right_link5", [0.0, 0.0, 0.080], 0.035),
            ("openarm_right_link6", [-0.020, 0.0, 0.0], 0.035),
            ("openarm_right_link7", [0.0, 0.0, 0.025], 0.035),
            ("openarm_right_hand", [0.0, 0.0, 0.025], 0.055),
            ("openarm_right_hand_tcp", [0.0, 0.0, 0.0], 0.020),
        ]
        spheres: list[CollisionSphere] = []
        for frame, center, radius in raw:
            if not self.model.existFrame(frame):
                self.get_logger().warn(f"Skipping collision sphere on missing frame {frame!r}")
                continue
            spheres.append(CollisionSphere(frame, np.asarray(center, dtype=float), float(radius)))
        if not spheres:
            raise RuntimeError("No valid collision spheres were configured.")
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

    def _float_command_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < len(self.joint_names):
            self.get_logger().warn(
                f"Ignoring command with {len(msg.data)} positions; need at least {len(self.joint_names)}"
            )
            return
        self._update_baseline(np.asarray(msg.data[: len(self.joint_names)], dtype=float))
        self.latest_extra = [float(v) for v in msg.data[len(self.joint_names) :]]

    def _trajectory_command_cb(self, msg: JointTrajectory) -> None:
        if not msg.points:
            return
        point = msg.points[0]
        name_to_pos = {name: pos for name, pos in zip(msg.joint_names, point.positions)}
        if not all(name in name_to_pos for name in self.joint_names):
            self.get_logger().warn("Ignoring trajectory command missing one or more right-arm joints")
            return
        self._update_baseline(np.asarray([name_to_pos[name] for name in self.joint_names], dtype=float))
        self.latest_trajectory = msg

    def _update_baseline(self, q_baseline: np.ndarray) -> None:
        now = self.get_clock().now()
        q_baseline = self._clamp_right(q_baseline)
        if self.latest_baseline is not None:
            dt = (now - self.latest_baseline_time).nanoseconds * 1e-9
            if dt > 1e-4:
                raw_qdot = (q_baseline - self.latest_baseline) / dt
                if self.qdot_limit > 0.0:
                    raw_qdot = np.clip(raw_qdot, -self.qdot_limit, self.qdot_limit)
                alpha = self.qdot_filter_alpha
                self.qdot_ref = (1.0 - alpha) * self.qdot_ref + alpha * raw_qdot
            self.previous_baseline = self.latest_baseline.copy()
            self.previous_baseline_time = self.latest_baseline_time
        self.latest_baseline = q_baseline
        self.latest_baseline_time = now

    def _timer_cb(self) -> None:
        if self.latest_baseline is None:
            return

        q_baseline = self._clamp_right(self.latest_baseline)
        q_sequence = self._predict_baseline_sequence(q_baseline)
        predicted_states = self._compute_sequence_sphere_states(q_sequence)
        self._maybe_request_esdf(predicted_states)

        if self.esdf_grid is None:
            q_plan = q_baseline.copy()
            self.active_constraints = 0
            self.min_clearance = float("nan")
            self.predicted_min_clearance = float("nan")
            self.sampled_spheres = 0
        else:
            q_plan = self._plan(q_baseline, q_sequence, predicted_states)

        if self.active_constraints > 0 or np.linalg.norm(self.last_plan_delta) > 1e-6:
            q_plan = self._limit_command_step(q_plan, q_baseline)
        else:
            q_plan = q_baseline.copy()
        q_plan = self._clamp_right(q_plan)

        if self.monitor_only:
            q_out = q_baseline
        else:
            q_out = q_plan

        self.last_command = q_out.copy()
        self._publish_command(q_out)
        self._publish_debug(q_out)

    def _predict_baseline_sequence(self, q_baseline: np.ndarray) -> list[np.ndarray]:
        sequence = []
        for step in range(self.horizon_steps + 1):
            t = step * self.horizon_dt
            sequence.append(self._clamp_right(q_baseline + t * self.qdot_ref))
        return sequence

    def _compute_sequence_sphere_states(self, q_sequence: list[np.ndarray]) -> list[list[SphereState]]:
        return [
            self._compute_sphere_states(self._model_q_from_right(q_right))
            for q_right in q_sequence
        ]

    def _model_q_from_right(self, q_right: np.ndarray) -> np.ndarray:
        q_model = self.q_model_current.copy() if self.have_joint_state else self.neutral_q.copy()
        for value, idx in zip(q_right, self.joint_q_indices):
            q_model[idx] = float(value)
        return q_model

    def _clamp_right(self, q_right: np.ndarray) -> np.ndarray:
        q = np.asarray(q_right, dtype=float).copy()
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

    def _maybe_request_esdf(self, sequence_states: list[list[SphereState]]) -> None:
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

        flat_states = [state for states in sequence_states for state in states]
        if not flat_states:
            return
        positions = np.vstack([state.position for state in flat_states])
        radii = np.asarray([state.sphere.radius for state in flat_states], dtype=float)
        padding = self.aabb_padding + float(np.max(radii)) + self.activation_margin
        aabb_min = np.min(positions, axis=0) - padding
        aabb_max = np.max(positions, axis=0) + padding

        req = EsdfAndGradients.Request()
        req.update_esdf = self.request_update_esdf
        req.visualize_esdf = False
        req.use_aabb = True
        req.frame_id = self.global_frame
        req.aabb_min_m = _point_from_array(aabb_min)
        req.aabb_size_m = _vector_from_array(aabb_max - aabb_min)

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

    def _plan(
        self,
        q_baseline: np.ndarray,
        q_sequence: list[np.ndarray],
        sequence_states: list[list[SphereState]],
    ) -> np.ndarray:
        lhs = self.track_weight * np.eye(len(self.joint_names))
        rhs = np.zeros(len(self.joint_names), dtype=float)
        active = 0
        clearances: list[float] = []
        sampled = 0

        for step, states in enumerate(sequence_states):
            self._sample_spheres(states)
            horizon_scale = 1.0 - 0.55 * (step / max(1, self.horizon_steps))
            for state in states:
                if state.clearance is None or state.gradient is None:
                    continue
                sampled += 1
                clearances.append(state.clearance)
                if state.clearance >= self.activation_margin:
                    continue

                required_delta = self.safety_margin - state.clearance
                if required_delta <= 0.0:
                    soft_delta = 0.25 * (self.activation_margin - state.clearance)
                else:
                    soft_delta = required_delta

                a = state.gradient.reshape(1, 3) @ state.jacobian
                a = a.reshape(-1)
                if not np.all(np.isfinite(a)) or float(np.linalg.norm(a)) < 1e-8:
                    continue

                proximity = (self.activation_margin - state.clearance) / max(
                    self.activation_margin - self.safety_margin,
                    1e-6,
                )
                proximity = float(np.clip(proximity, 0.0, 1.0))
                weight = self.obstacle_weight * horizon_scale * proximity * proximity
                lhs += weight * np.outer(a, a)
                rhs += weight * a * soft_delta
                active += 1

        self.sampled_spheres = sampled
        self.active_constraints = active
        if clearances:
            self.predicted_min_clearance = float(np.min(clearances))
            now_states = sequence_states[0]
            now_clearances = [state.clearance for state in now_states if state.clearance is not None]
            self.min_clearance = float(np.min(now_clearances)) if now_clearances else float("nan")
        else:
            self.predicted_min_clearance = float("nan")
            self.min_clearance = float("nan")

        if self.publish_markers:
            self._publish_markers(sequence_states)

        if active == 0:
            self.inactive_cycles += 1
            if self.inactive_cycles <= self.active_hold_cycles:
                self.last_plan_delta *= self.inactive_delta_decay
                return self._clamp_right(q_baseline + self.last_plan_delta)
            self.last_plan_delta[:] = 0.0
            return q_baseline.copy()

        lhs += (self.damping * self.damping) * np.eye(len(self.joint_names))
        try:
            delta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

        if self.max_plan_delta_per_joint > 0.0:
            delta = np.clip(delta, -self.max_plan_delta_per_joint, self.max_plan_delta_per_joint)
        target_delta = self.correction_alpha * delta
        alpha = self.plan_delta_filter_alpha
        self.last_plan_delta = (1.0 - alpha) * self.last_plan_delta + alpha * target_delta
        if self.max_plan_delta_per_joint > 0.0:
            self.last_plan_delta = np.clip(
                self.last_plan_delta,
                -self.max_plan_delta_per_joint,
                self.max_plan_delta_per_joint,
            )
        self.inactive_cycles = 0
        return self._clamp_right(q_baseline + self.last_plan_delta)

    def _sample_spheres(self, states: list[SphereState]) -> None:
        if self.esdf_grid is None:
            return
        for state in states:
            distance, gradient = self.esdf_grid.sample(state.position)
            state.distance = distance
            state.gradient = gradient
            if distance is not None:
                state.clearance = float(distance - state.sphere.radius)

    def _limit_command_step(self, q_plan: np.ndarray, q_baseline: np.ndarray) -> np.ndarray:
        reference = self.last_command if self.last_command is not None else q_baseline
        if self.max_command_step <= 0.0:
            return q_plan
        delta = np.clip(q_plan - reference, -self.max_command_step, self.max_command_step)
        return reference + delta

    def _publish_command(self, q_out: np.ndarray) -> None:
        positions = [float(v) for v in q_out]
        if self.command_message_type == "float64_multi_array":
            self.command_pub.publish(Float64MultiArray(data=positions + list(self.latest_extra)))
            return

        if self.latest_trajectory is None:
            return
        msg = JointTrajectory()
        msg.header = self.latest_trajectory.header
        msg.joint_names = list(self.latest_trajectory.joint_names)
        point = JointTrajectoryPoint()
        point.time_from_start = self.latest_trajectory.points[0].time_from_start
        name_to_pos = {
            name: pos for name, pos in zip(self.latest_trajectory.joint_names, self.latest_trajectory.points[0].positions)
        }
        for name, value in zip(self.joint_names, positions):
            name_to_pos[name] = value
        point.positions = [float(name_to_pos[name]) for name in msg.joint_names]
        msg.points.append(point)
        self.command_pub.publish(msg)

    def _publish_debug(self, q_out: np.ndarray) -> None:
        if np.isfinite(self.predicted_min_clearance):
            self.min_clearance_pub.publish(Float32(data=float(self.predicted_min_clearance)))
        mode = "monitor" if self.monitor_only else "active"
        status = (
            f"mode={mode} predicted_min_clearance={self.predicted_min_clearance:.3f} "
            f"current_min_clearance={self.min_clearance:.3f} "
            f"active_constraints={self.active_constraints} "
            f"sampled_spheres={self.sampled_spheres}/{len(self.spheres) * (self.horizon_steps + 1)} "
            f"esdf_grid={self.esdf_grid_shape} esdf_observed={self.esdf_observed_count} "
            f"qdot={np.array2string(self.qdot_ref, precision=3, suppress_small=True)} "
            f"plan_delta={np.array2string(self.last_plan_delta, precision=3, suppress_small=True)} "
            f"q={np.array2string(q_out, precision=3, suppress_small=True)}"
        )
        self.status_pub.publish(String(data=status))
        self._throttled_info(status)

    def _publish_markers(self, sequence_states: list[list[SphereState]]) -> None:
        array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        marker_id = 0
        for step, states in enumerate(sequence_states):
            for state in states:
                marker = Marker()
                marker.header.frame_id = self.global_frame
                marker.header.stamp = stamp
                marker.ns = "right_arm_esdf_predicted_spheres"
                marker.id = marker_id
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position = _point_from_array(state.position)
                marker.pose.orientation.w = 1.0
                diameter = 2.0 * float(state.sphere.radius + self.safety_margin)
                marker.scale.x = diameter
                marker.scale.y = diameter
                marker.scale.z = diameter
                marker.color.a = max(0.08, 0.45 - 0.05 * step)
                clearance = state.clearance
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
                else:
                    marker.color.r = 0.0
                    marker.color.g = 0.45
                    marker.color.b = 1.0
                array.markers.append(marker)
                marker_id += 1
        self.marker_pub.publish(array)

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
    node = RightArmEsdfPredictivePlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
