from __future__ import annotations

import os
import tempfile

import numpy as np
import pinocchio as pin
import rclpy
from nvblox_msgs.srv import EsdfAndGradients
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Float64MultiArray, String
from visualization_msgs.msg import Marker, MarkerArray

from openarmx_obstacle_avoidance.bimanual_esdf_avoidance_filter import (
    LEFT_JOINT_NAMES,
    RIGHT_JOINT_NAMES,
)
from openarmx_obstacle_avoidance.right_arm_esdf_avoidance_filter import (
    CollisionSphere,
    EsdfGrid,
    SphereState,
    _point_from_array,
    _skew,
    _vector_from_array,
)


class BimanualEsdfPredictivePlanner(Node):
    def __init__(self) -> None:
        super().__init__("bimanual_esdf_predictive_planner")

        self.declare_parameter("urdf_path", "")
        self.declare_parameter("robot_description_node", "/robot_state_publisher")
        self.declare_parameter("left_joint_names", LEFT_JOINT_NAMES)
        self.declare_parameter("right_joint_names", RIGHT_JOINT_NAMES)
        self.declare_parameter("left_input_command_topic", "/left_teleop_baseline/commands")
        self.declare_parameter("right_input_command_topic", "/right_teleop_baseline/commands")
        self.declare_parameter("left_output_command_topic", "/left_teleop_planned/commands")
        self.declare_parameter("right_output_command_topic", "/right_teleop_planned/commands")
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
        self.declare_parameter("obstacle_weight", 0.8)
        self.declare_parameter("damping", 0.05)
        self.declare_parameter("correction_alpha", 0.5)
        self.declare_parameter("plan_delta_filter_alpha", 0.25)
        self.declare_parameter("inactive_delta_decay", 0.6)
        self.declare_parameter("active_hold_cycles", 5)
        self.declare_parameter("max_plan_delta_per_joint", 0.05)
        self.declare_parameter("max_command_step", 0.05)
        self.declare_parameter("qdot_filter_alpha", 0.25)
        self.declare_parameter("qdot_limit", 1.0)
        self.declare_parameter("downward_bias_weight", 0.20)
        self.declare_parameter("downward_bias_step", 0.015)
        self.declare_parameter("downward_bias_z", -1.0)

        self.declare_parameter("enable_inter_arm_collision", True)
        self.declare_parameter("inter_arm_safety_margin", 0.03)
        self.declare_parameter("inter_arm_activation_margin", 0.10)
        self.declare_parameter("inter_arm_weight", 0.8)
        self.declare_parameter("inter_arm_skip_proximal_spheres", 3)

        self.declare_parameter("clear_robot_from_esdf", False)
        self.declare_parameter("clear_robot_padding", 0.015)
        self.declare_parameter("clear_robot_radius_scale", 1.0)
        self.declare_parameter("publish_markers", True)
        self.declare_parameter("marker_topic", "/openarm/bimanual/predicted_spheres")
        self.declare_parameter("min_clearance_topic", "/openarm/bimanual/predictive_min_clearance")
        self.declare_parameter("status_topic", "/openarm/bimanual/predictive_planner_status")

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
        self.max_plan_delta_per_joint = max(0.0, float(self.get_parameter("max_plan_delta_per_joint").value))
        self.max_command_step = max(0.0, float(self.get_parameter("max_command_step").value))
        self.qdot_filter_alpha = float(
            np.clip(float(self.get_parameter("qdot_filter_alpha").value), 0.0, 1.0)
        )
        self.qdot_limit = max(0.0, float(self.get_parameter("qdot_limit").value))
        self.downward_bias_weight = max(0.0, float(self.get_parameter("downward_bias_weight").value))
        self.downward_bias_step = max(0.0, float(self.get_parameter("downward_bias_step").value))
        self.downward_direction = np.array(
            [0.0, 0.0, float(self.get_parameter("downward_bias_z").value)],
            dtype=float,
        )
        direction_norm = float(np.linalg.norm(self.downward_direction))
        if not np.isfinite(direction_norm) or direction_norm < 1e-9:
            self.downward_direction = np.array([0.0, 0.0, -1.0], dtype=float)
        else:
            self.downward_direction /= direction_norm
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
        self.left_slice = slice(0, len(self.left_joint_names))
        self.right_slice = slice(len(self.left_joint_names), len(self.joint_names))
        self.spheres = self._default_bimanual_spheres()
        left_all_indices = [i for i, sphere in enumerate(self.spheres) if "_left_" in sphere.frame]
        right_all_indices = [i for i, sphere in enumerate(self.spheres) if "_right_" in sphere.frame]
        self.left_sphere_indices = left_all_indices[self.inter_arm_skip_proximal_spheres :]
        self.right_sphere_indices = right_all_indices[self.inter_arm_skip_proximal_spheres :]
        self.sphere_frame_ids = [self.model.getFrameId(sphere.frame) for sphere in self.spheres]

        self.q_model_current = self.neutral_q.copy()
        self.have_joint_state = False
        self.latest_left_baseline: np.ndarray | None = None
        self.latest_right_baseline: np.ndarray | None = None
        self.latest_left_extra: list[float] = []
        self.latest_right_extra: list[float] = []
        self.last_combined_baseline: np.ndarray | None = None
        self.last_combined_baseline_time = self.get_clock().now()
        self.qdot_ref = np.zeros(len(self.joint_names), dtype=float)
        self.last_command: np.ndarray | None = None
        self.last_plan_delta = np.zeros(len(self.joint_names), dtype=float)
        self.inactive_cycles = 0
        self.esdf_grid: EsdfGrid | None = None
        self.esdf_pending = False
        self.last_esdf_request_time = self.get_clock().now()
        self.last_log_time = self.get_clock().now()
        self.min_clearance = float("nan")
        self.predicted_min_clearance = float("nan")
        self.min_inter_arm_clearance = float("nan")
        self.predicted_min_inter_arm_clearance = float("nan")
        self.active_constraints = 0
        self.active_esdf_constraints = 0
        self.active_inter_arm_constraints = 0
        self.sampled_spheres = 0
        self.esdf_grid_shape = "none"
        self.esdf_observed_count = 0

        self.esdf_client = None
        self._connect_esdf_client()

        self.create_subscription(JointState, self.joint_states_topic, self._joint_state_cb, 20)
        self.create_subscription(Float64MultiArray, self.left_input_command_topic, self._left_command_cb, 10)
        self.create_subscription(Float64MultiArray, self.right_input_command_topic, self._right_command_cb, 10)
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
        self.status_pub = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            10,
        )
        self.timer = self.create_timer(1.0 / self.rate_hz, self._timer_cb)

        self.get_logger().info(f"URDF: {self.urdf_path}")
        self.get_logger().info(
            "bimanual predictive planner: "
            f"{self.left_input_command_topic}, {self.right_input_command_topic} -> "
            f"{self.left_output_command_topic}, {self.right_output_command_topic}"
        )
        self.get_logger().info(
            "planner: "
            f"rate_hz={self.rate_hz:.1f}, horizon_steps={self.horizon_steps}, "
            f"horizon_dt={self.horizon_dt:.3f}, monitor_only={self.monitor_only}, "
            f"safety_margin={self.safety_margin:.3f}, activation_margin={self.activation_margin:.3f}, "
            f"downward_bias_weight={self.downward_bias_weight:.3f}, "
            f"downward_bias_step={self.downward_bias_step:.3f}, "
            f"inter_arm={self.enable_inter_arm_collision}, "
            f"inter_skip_proximal={self.inter_arm_skip_proximal_spheres}"
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

        fd, path = tempfile.mkstemp(prefix="openarmx_bimanual_esdf_planner_", suffix=".urdf")
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

    def _default_bimanual_spheres(self) -> list[CollisionSphere]:
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
        spheres: list[CollisionSphere] = []
        for side in ("left", "right"):
            for suffix, center, radius in raw_one_arm:
                frame = f"openarm_{side}_{suffix}"
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

    def _left_command_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < len(self.left_joint_names):
            self.get_logger().warn(
                f"Ignoring left command with {len(msg.data)} positions; need {len(self.left_joint_names)}"
            )
            return
        self.latest_left_baseline = np.asarray(msg.data[: len(self.left_joint_names)], dtype=float)
        self.latest_left_extra = [float(v) for v in msg.data[len(self.left_joint_names) :]]

    def _right_command_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < len(self.right_joint_names):
            self.get_logger().warn(
                f"Ignoring right command with {len(msg.data)} positions; need {len(self.right_joint_names)}"
            )
            return
        self.latest_right_baseline = np.asarray(msg.data[: len(self.right_joint_names)], dtype=float)
        self.latest_right_extra = [float(v) for v in msg.data[len(self.right_joint_names) :]]

    def _timer_cb(self) -> None:
        if self.latest_left_baseline is None or self.latest_right_baseline is None:
            return

        q_baseline = self._clamp_bimanual(
            np.concatenate([self.latest_left_baseline, self.latest_right_baseline])
        )
        self._update_qdot_ref(q_baseline)
        q_sequence = self._predict_baseline_sequence(q_baseline)
        sequence_states = self._compute_sequence_sphere_states(q_sequence)
        self._maybe_request_esdf(sequence_states)

        if self.esdf_grid is None and not self.enable_inter_arm_collision:
            q_plan = q_baseline.copy()
            self.active_constraints = 0
            self.active_esdf_constraints = 0
            self.active_inter_arm_constraints = 0
            self.min_clearance = float("nan")
            self.predicted_min_clearance = float("nan")
            self.min_inter_arm_clearance = float("nan")
            self.predicted_min_inter_arm_clearance = float("nan")
            self.sampled_spheres = 0
        else:
            q_plan = self._plan(q_baseline, sequence_states)

        if self.active_constraints > 0 or np.linalg.norm(self.last_plan_delta) > 1e-6:
            q_plan = self._limit_command_step(q_plan, q_baseline)
        else:
            q_plan = q_baseline.copy()
        q_plan = self._clamp_bimanual(q_plan)

        q_out = q_baseline if self.monitor_only else q_plan
        self.last_command = q_out.copy()
        self._publish_commands(q_out)
        self._publish_debug(q_out)

    def _update_qdot_ref(self, q_baseline: np.ndarray) -> None:
        now = self.get_clock().now()
        if self.last_combined_baseline is not None:
            dt = (now - self.last_combined_baseline_time).nanoseconds * 1e-9
            if dt > 1e-4:
                raw_qdot = (q_baseline - self.last_combined_baseline) / dt
                if self.qdot_limit > 0.0:
                    raw_qdot = np.clip(raw_qdot, -self.qdot_limit, self.qdot_limit)
                alpha = self.qdot_filter_alpha
                self.qdot_ref = (1.0 - alpha) * self.qdot_ref + alpha * raw_qdot
        self.last_combined_baseline = q_baseline.copy()
        self.last_combined_baseline_time = now

    def _predict_baseline_sequence(self, q_baseline: np.ndarray) -> list[np.ndarray]:
        sequence = []
        for step in range(self.horizon_steps + 1):
            t = step * self.horizon_dt
            sequence.append(self._clamp_bimanual(q_baseline + t * self.qdot_ref))
        return sequence

    def _compute_sequence_sphere_states(self, q_sequence: list[np.ndarray]) -> list[list[SphereState]]:
        return [
            self._compute_sphere_states(self._model_q_from_bimanual(q_bimanual))
            for q_bimanual in q_sequence
        ]

    def _model_q_from_bimanual(self, q_bimanual: np.ndarray) -> np.ndarray:
        q_model = self.q_model_current.copy() if self.have_joint_state else self.neutral_q.copy()
        for value, idx in zip(q_bimanual, self.joint_q_indices):
            q_model[idx] = float(value)
        return q_model

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
        padding = self.aabb_padding + float(np.max(radii)) + max(
            self.activation_margin,
            self.inter_arm_activation_margin,
        )
        aabb_min = np.min(positions, axis=0) - padding
        aabb_max = np.max(positions, axis=0) + padding

        req = EsdfAndGradients.Request()
        req.update_esdf = self.request_update_esdf
        req.visualize_esdf = False
        req.use_aabb = True
        req.frame_id = self.global_frame
        req.aabb_min_m = _point_from_array(aabb_min)
        req.aabb_size_m = _vector_from_array(aabb_max - aabb_min)

        if self.clear_robot_from_esdf:
            for state in flat_states:
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

    def _plan(self, q_baseline: np.ndarray, sequence_states: list[list[SphereState]]) -> np.ndarray:
        lhs = self.track_weight * np.eye(len(self.joint_names))
        rhs = np.zeros(len(self.joint_names), dtype=float)
        active_esdf = 0
        active_inter = 0
        clearances: list[float] = []
        current_clearances: list[float] = []
        inter_clearances: list[float] = []
        current_inter_clearances: list[float] = []
        sampled = 0

        for step, states in enumerate(sequence_states):
            self._sample_spheres(states)
            horizon_scale = 1.0 - 0.55 * (step / max(1, self.horizon_steps))
            esdf_active, esdf_sampled, step_clearances = self._add_esdf_constraints(
                lhs,
                rhs,
                states,
                horizon_scale,
            )
            active_esdf += esdf_active
            sampled += esdf_sampled
            clearances.extend(step_clearances)
            if step == 0:
                current_clearances.extend(step_clearances)

            inter_active, step_inter_clearances = self._add_inter_arm_constraints(
                lhs,
                rhs,
                states,
                horizon_scale,
            )
            active_inter += inter_active
            inter_clearances.extend(step_inter_clearances)
            if step == 0:
                current_inter_clearances.extend(step_inter_clearances)

        self.sampled_spheres = sampled
        self.active_esdf_constraints = active_esdf
        self.active_inter_arm_constraints = active_inter
        self.active_constraints = active_esdf + active_inter
        self.predicted_min_clearance = float(np.min(clearances)) if clearances else float("nan")
        self.min_clearance = float(np.min(current_clearances)) if current_clearances else float("nan")
        self.predicted_min_inter_arm_clearance = (
            float(np.min(inter_clearances)) if inter_clearances else float("nan")
        )
        self.min_inter_arm_clearance = (
            float(np.min(current_inter_clearances)) if current_inter_clearances else float("nan")
        )

        if self.publish_markers:
            self._publish_markers(sequence_states)

        if self.active_constraints == 0:
            self.inactive_cycles += 1
            if self.inactive_cycles <= self.active_hold_cycles:
                self.last_plan_delta *= self.inactive_delta_decay
                return self._clamp_bimanual(q_baseline + self.last_plan_delta)
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
        return self._clamp_bimanual(q_baseline + self.last_plan_delta)

    def _add_esdf_constraints(
        self,
        lhs: np.ndarray,
        rhs: np.ndarray,
        states: list[SphereState],
        horizon_scale: float,
    ) -> tuple[int, int, list[float]]:
        active = 0
        sampled = 0
        clearances: list[float] = []
        if self.esdf_grid is None:
            return active, sampled, clearances

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
            self._add_downward_bias(lhs, rhs, state.jacobian, horizon_scale, proximity)
            active += 1
        return active, sampled, clearances

    def _add_downward_bias(
        self,
        lhs: np.ndarray,
        rhs: np.ndarray,
        jacobian: np.ndarray,
        horizon_scale: float,
        proximity: float,
    ) -> None:
        if self.downward_bias_weight <= 0.0 or self.downward_bias_step <= 0.0:
            return
        a = self.downward_direction.reshape(1, 3) @ jacobian
        a = a.reshape(-1)
        if not np.all(np.isfinite(a)) or float(np.linalg.norm(a)) < 1e-8:
            return
        weight = self.downward_bias_weight * horizon_scale * float(np.clip(proximity, 0.0, 1.0))
        lhs += weight * np.outer(a, a)
        rhs += weight * a * self.downward_bias_step

    def _add_inter_arm_constraints(
        self,
        lhs: np.ndarray,
        rhs: np.ndarray,
        states: list[SphereState],
        horizon_scale: float,
    ) -> tuple[int, list[float]]:
        if not self.enable_inter_arm_collision or self.inter_arm_weight <= 0.0:
            return 0, []

        active = 0
        clearances: list[float] = []
        for left_idx in self.left_sphere_indices:
            left = states[left_idx]
            for right_idx in self.right_sphere_indices:
                right = states[right_idx]
                delta = left.position - right.position
                distance = float(np.linalg.norm(delta))
                if not np.isfinite(distance) or distance < 1e-6:
                    continue
                clearance = distance - left.sphere.radius - right.sphere.radius
                clearances.append(float(clearance))
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
                weight = self.inter_arm_weight * horizon_scale * proximity * proximity
                lhs += weight * np.outer(a, a)
                rhs += weight * a * soft_delta
                active += 1
        return active, clearances

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

    def _publish_commands(self, q_out: np.ndarray) -> None:
        left = [float(v) for v in q_out[self.left_slice]]
        right = [float(v) for v in q_out[self.right_slice]]
        self.left_command_pub.publish(Float64MultiArray(data=left + list(self.latest_left_extra)))
        self.right_command_pub.publish(Float64MultiArray(data=right + list(self.latest_right_extra)))

    def _publish_debug(self, q_out: np.ndarray) -> None:
        if np.isfinite(self.predicted_min_clearance):
            self.min_clearance_pub.publish(Float32(data=float(self.predicted_min_clearance)))
        mode = "monitor" if self.monitor_only else "active"
        status = (
            f"mode={mode} predicted_min_clearance={self.predicted_min_clearance:.3f} "
            f"current_min_clearance={self.min_clearance:.3f} "
            f"predicted_min_inter_arm_clearance={self.predicted_min_inter_arm_clearance:.3f} "
            f"current_min_inter_arm_clearance={self.min_inter_arm_clearance:.3f} "
            f"active_constraints={self.active_constraints} "
            f"active_esdf={self.active_esdf_constraints} active_inter_arm={self.active_inter_arm_constraints} "
            f"sampled_spheres={self.sampled_spheres}/{len(self.spheres) * (self.horizon_steps + 1)} "
            f"esdf_grid={self.esdf_grid_shape} esdf_observed={self.esdf_observed_count} "
            f"qdot={np.array2string(self.qdot_ref, precision=3, suppress_small=True)} "
            f"plan_delta={np.array2string(self.last_plan_delta, precision=3, suppress_small=True)} "
            f"left={np.array2string(q_out[self.left_slice], precision=3, suppress_small=True)} "
            f"right={np.array2string(q_out[self.right_slice], precision=3, suppress_small=True)}"
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
                marker.ns = "bimanual_esdf_predicted_spheres"
                marker.id = marker_id
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position = _point_from_array(state.position)
                marker.pose.orientation.w = 1.0
                diameter = 2.0 * float(state.sphere.radius + self.safety_margin)
                marker.scale.x = diameter
                marker.scale.y = diameter
                marker.scale.z = diameter
                marker.color.a = max(0.08, 0.42 - 0.045 * step)
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
                elif "_left_" in state.sphere.frame:
                    marker.color.r = 0.0
                    marker.color.g = 0.45
                    marker.color.b = 1.0
                else:
                    marker.color.r = 0.0
                    marker.color.g = 0.85
                    marker.color.b = 0.25
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
    node = BimanualEsdfPredictivePlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
