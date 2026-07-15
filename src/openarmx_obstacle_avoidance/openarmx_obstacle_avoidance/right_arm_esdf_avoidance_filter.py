from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

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


RIGHT_JOINT_NAMES = [f"openarm_right_joint{i}" for i in range(1, 8)]


@dataclass(frozen=True)
class CollisionSphere:
    frame: str
    center: np.ndarray
    radius: float


@dataclass
class SphereState:
    sphere: CollisionSphere
    position: np.ndarray
    jacobian: np.ndarray
    distance: float | None = None
    gradient: np.ndarray | None = None
    clearance: float | None = None


class EsdfGrid:
    def __init__(
        self,
        response: EsdfAndGradients.Response,
        unobserved_value: float,
        nearest_observed_search_radius: float,
    ) -> None:
        self.frame_id = response.header.frame_id
        self.origin = np.array(
            [response.origin_m.x, response.origin_m.y, response.origin_m.z],
            dtype=float,
        )
        self.voxel_size = float(response.voxel_size_m)
        self.unobserved_value = float(unobserved_value)
        self.nearest_observed_search_radius = max(0.0, float(nearest_observed_search_radius))
        layout = response.esdf_and_gradients.layout
        dims = {dim.label: int(dim.size) for dim in layout.dim}
        self.size = np.array(
            [dims.get("x", 0), dims.get("y", 0), dims.get("z", 0)],
            dtype=int,
        )
        self.data = np.asarray(response.esdf_and_gradients.data, dtype=np.float32)
        self.observed_mask = np.isfinite(self.data) & (self.data > self.unobserved_value * 0.5)
        self.observed_count = int(np.count_nonzero(self.observed_mask))
        self.observed_centers: np.ndarray | None = None
        self.observed_values: np.ndarray | None = None
        self._build_observed_cache()

    def valid(self) -> bool:
        return (
            self.voxel_size > 0.0
            and self.size.shape == (3,)
            and np.all(self.size > 2)
            and self.data.size >= int(np.prod(self.size))
        )

    def sample(self, point: np.ndarray) -> tuple[float | None, np.ndarray | None]:
        if not self.valid():
            return None, None

        idx = np.floor((np.asarray(point, dtype=float) - self.origin) / self.voxel_size).astype(int)
        if np.any(idx < 1) or np.any(idx >= self.size - 1):
            return self._sample_nearest_observed(point)

        distance = self._value(idx)
        if not self._is_observed(distance):
            return self._sample_nearest_observed(point)

        grad = np.array(
            [
                self._central_difference(idx, 0),
                self._central_difference(idx, 1),
                self._central_difference(idx, 2),
            ],
            dtype=float,
        )
        norm = float(np.linalg.norm(grad))
        if not np.isfinite(norm) or norm < 1e-6:
            nearest_distance, nearest_gradient = self._sample_nearest_observed(point)
            if nearest_gradient is not None:
                return nearest_distance, nearest_gradient
            return float(distance), None
        return float(distance), grad / norm

    def _build_observed_cache(self) -> None:
        if self.size.shape != (3,) or np.any(self.size <= 0):
            return
        sx, sy, sz = [int(v) for v in self.size]
        expected = sx * sy * sz
        if expected <= 0 or self.data.size < expected:
            return
        observed_linear = np.flatnonzero(self.observed_mask[:expected])
        if observed_linear.size == 0:
            return
        x = observed_linear // (sy * sz)
        rem = observed_linear - x * sy * sz
        y = rem // sz
        z = rem - y * sz
        indices = np.stack([x, y, z], axis=1).astype(float)
        self.observed_centers = self.origin + (indices + 0.5) * self.voxel_size
        self.observed_values = self.data[observed_linear].astype(float)

    def _sample_nearest_observed(self, point: np.ndarray) -> tuple[float | None, np.ndarray | None]:
        if (
            self.nearest_observed_search_radius <= 0.0
            or self.observed_centers is None
            or self.observed_values is None
            or self.observed_centers.size == 0
        ):
            return None, None
        point_np = np.asarray(point, dtype=float).reshape(3)
        deltas = point_np.reshape(1, 3) - self.observed_centers
        spatial_distances = np.linalg.norm(deltas, axis=1)
        within = spatial_distances <= self.nearest_observed_search_radius
        if not np.any(within):
            return None, None
        candidate_costs = self.observed_values[within] + spatial_distances[within]
        local_index = int(np.argmin(candidate_costs))
        all_indices = np.flatnonzero(within)
        best_index = int(all_indices[local_index])
        direction = deltas[best_index]
        direction_norm = float(np.linalg.norm(direction))
        if not np.isfinite(direction_norm) or direction_norm < 1e-6:
            gradient = None
        else:
            gradient = direction / direction_norm
        return float(candidate_costs[local_index]), gradient

    def _linear_index(self, idx: np.ndarray) -> int:
        x, y, z = [int(v) for v in idx]
        sx, sy, sz = [int(v) for v in self.size]
        return z + y * sz + x * sy * sz

    def _value(self, idx: np.ndarray) -> float:
        linear = self._linear_index(idx)
        if linear < 0 or linear >= self.data.size:
            return self.unobserved_value
        return float(self.data[linear])

    def _is_observed(self, value: float) -> bool:
        return np.isfinite(value) and value > self.unobserved_value * 0.5

    def _central_difference(self, idx: np.ndarray, axis: int) -> float:
        minus = idx.copy()
        plus = idx.copy()
        minus[axis] -= 1
        plus[axis] += 1
        v_minus = self._value(minus)
        v_plus = self._value(plus)
        if not self._is_observed(v_minus) or not self._is_observed(v_plus):
            return 0.0
        return (v_plus - v_minus) / (2.0 * self.voxel_size)


class RightArmEsdfAvoidanceFilter(Node):
    def __init__(self) -> None:
        super().__init__("right_arm_esdf_avoidance_filter")

        self.declare_parameter("urdf_path", "")
        self.declare_parameter("robot_description_node", "/robot_state_publisher")
        self.declare_parameter("joint_names", RIGHT_JOINT_NAMES)
        self.declare_parameter("command_message_type", "float64_multi_array")
        self.declare_parameter("input_command_topic", "/right_teleop_baseline/commands")
        self.declare_parameter("output_command_topic", "/right_forward_position_controller/commands")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("esdf_service", "")
        self.declare_parameter("global_frame", "world")
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("esdf_update_hz", 8.0)
        self.declare_parameter("aabb_padding", 0.25)
        self.declare_parameter("unobserved_value", -1000.0)
        self.declare_parameter("request_update_esdf", True)
        self.declare_parameter("nearest_observed_search_radius", 0.18)

        self.declare_parameter("monitor_only", True)
        self.declare_parameter("safety_margin", 0.06)
        self.declare_parameter("activation_margin", 0.16)
        self.declare_parameter("baseline_weight", 1.0)
        self.declare_parameter("avoidance_weight", 12.0)
        self.declare_parameter("damping", 0.03)
        self.declare_parameter("iterations", 2)
        self.declare_parameter("max_adjust_per_joint", 0.02)
        self.declare_parameter("max_command_step", 0.025)
        self.declare_parameter("max_avoidance_delta", 0.08)
        self.declare_parameter("avoidance_delta_alpha", 0.25)
        self.declare_parameter("clear_robot_from_esdf", True)
        self.declare_parameter("clear_robot_padding", 0.015)
        self.declare_parameter("clear_robot_radius_scale", 1.0)
        self.declare_parameter("publish_markers", True)
        self.declare_parameter("marker_topic", "/openarm/right_arm/esdf_spheres")
        self.declare_parameter("min_clearance_topic", "/openarm/right_arm/min_esdf_clearance")
        self.declare_parameter("status_topic", "/openarm/right_arm/esdf_avoidance_status")

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
        self.baseline_weight = max(1e-6, float(self.get_parameter("baseline_weight").value))
        self.avoidance_weight = max(0.0, float(self.get_parameter("avoidance_weight").value))
        self.damping = max(1e-6, float(self.get_parameter("damping").value))
        self.iterations = max(1, int(self.get_parameter("iterations").value))
        self.max_adjust_per_joint = max(0.0, float(self.get_parameter("max_adjust_per_joint").value))
        self.max_command_step = max(0.0, float(self.get_parameter("max_command_step").value))
        self.max_avoidance_delta = max(0.0, float(self.get_parameter("max_avoidance_delta").value))
        self.avoidance_delta_alpha = float(
            np.clip(float(self.get_parameter("avoidance_delta_alpha").value), 0.0, 1.0)
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
        self.spheres = self._default_right_arm_spheres()
        self.sphere_frame_ids = [self.model.getFrameId(sphere.frame) for sphere in self.spheres]

        self.q_model_current = self.neutral_q.copy()
        self.have_joint_state = False
        self.latest_baseline: np.ndarray | None = None
        self.latest_extra: list[float] = []
        self.latest_trajectory: JointTrajectory | None = None
        self.last_command: np.ndarray | None = None
        self.esdf_grid: EsdfGrid | None = None
        self.esdf_pending = False
        self.last_esdf_request_time = self.get_clock().now()
        self.last_log_time = self.get_clock().now()
        self.min_clearance = float("nan")
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
        self.get_logger().info(f"command filter: {self.input_command_topic} -> {self.output_command_topic}")
        self.get_logger().info(
            "avoidance: "
            f"monitor_only={self.monitor_only}, safety_margin={self.safety_margin:.3f}, "
            f"activation_margin={self.activation_margin:.3f}, spheres={len(self.spheres)}, "
            f"max_adjust_per_joint={self.max_adjust_per_joint:.3f}, "
            f"max_command_step={self.max_command_step:.3f}, "
            f"avoidance_delta_alpha={self.avoidance_delta_alpha:.3f}, "
            f"clear_robot_from_esdf={self.clear_robot_from_esdf}, "
            f"clear_robot_radius_scale={self.clear_robot_radius_scale:.3f}, "
            f"clear_robot_padding={self.clear_robot_padding:.3f}"
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

        fd, path = tempfile.mkstemp(prefix="openarmx_esdf_filter_", suffix=".urdf")
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
        self.latest_baseline = np.asarray(msg.data[: len(self.joint_names)], dtype=float)
        self.latest_extra = [float(v) for v in msg.data[len(self.joint_names) :]]

    def _trajectory_command_cb(self, msg: JointTrajectory) -> None:
        if not msg.points:
            return
        point = msg.points[0]
        name_to_pos = {name: pos for name, pos in zip(msg.joint_names, point.positions)}
        if not all(name in name_to_pos for name in self.joint_names):
            self.get_logger().warn("Ignoring trajectory command missing one or more right-arm joints")
            return
        self.latest_baseline = np.asarray([name_to_pos[name] for name in self.joint_names], dtype=float)
        self.latest_trajectory = msg

    def _timer_cb(self) -> None:
        if self.latest_baseline is None:
            return

        q_baseline = self._clamp_right(self.latest_baseline)
        q_start = self._current_safe_start(q_baseline)
        q_model = self._model_q_from_right(q_start)
        sphere_states = self._compute_sphere_states(q_model)
        self._maybe_request_esdf(sphere_states)

        q_safe = q_start.copy()
        if self.esdf_grid is not None:
            q_safe = self._avoid(q_baseline, q_start)
        else:
            q_safe = q_baseline.copy()
        q_safe = self._smooth_safe_command(q_safe, q_start)
        q_safe = self._limit_command_step(q_safe, q_baseline)
        q_safe = self._clamp_right(q_safe)

        if self.monitor_only:
            q_out = q_baseline
        else:
            q_out = q_safe

        self.last_command = q_out.copy()
        self._publish_command(q_out)
        self._publish_debug(q_out)

    def _current_safe_start(self, q_baseline: np.ndarray) -> np.ndarray:
        if self.last_command is not None:
            return self._clamp_right(self.last_command)
        if self.have_joint_state:
            return self._clamp_right(self._right_from_model_q(self.q_model_current))
        return self._clamp_right(q_baseline)

    def _model_q_from_right(self, q_right: np.ndarray) -> np.ndarray:
        q_model = self.q_model_current.copy() if self.have_joint_state else self.neutral_q.copy()
        for value, idx in zip(q_right, self.joint_q_indices):
            q_model[idx] = float(value)
        return q_model

    def _right_from_model_q(self, q_model: np.ndarray) -> np.ndarray:
        return np.asarray([q_model[idx] for idx in self.joint_q_indices], dtype=float)

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

        positions = np.vstack([state.position for state in sphere_states])
        radii = np.asarray([state.sphere.radius for state in sphere_states], dtype=float)
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
        q_work = q_start.copy()
        self.min_clearance = float("nan")
        self.active_constraints = 0

        for _ in range(self.iterations):
            states = self._compute_sphere_states(self._model_q_from_right(q_work))
            self._sample_spheres(states)
            valid_clearances = [state.clearance for state in states if state.clearance is not None]
            self.sampled_spheres = len(valid_clearances)
            if valid_clearances:
                self.min_clearance = float(np.min(valid_clearances))

            lhs = self.baseline_weight * np.eye(len(self.joint_names))
            rhs = self.baseline_weight * (q_baseline - q_work)
            active = 0

            for state in states:
                if state.clearance is None or state.gradient is None:
                    continue
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
                weight = self.avoidance_weight * proximity * proximity
                lhs += weight * np.outer(a, a)
                rhs += weight * a * soft_delta
                active += 1

            self.active_constraints = active
            if active == 0:
                q_work = q_baseline.copy()
                break

            lhs += (self.damping * self.damping) * np.eye(len(self.joint_names))
            try:
                dq = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                dq = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

            if self.max_adjust_per_joint > 0.0:
                dq = np.clip(dq, -self.max_adjust_per_joint, self.max_adjust_per_joint)
            q_work = self._clamp_right(q_work + dq)

        states = self._compute_sphere_states(self._model_q_from_right(q_work))
        self._sample_spheres(states)
        clearances = [state.clearance for state in states if state.clearance is not None]
        self.sampled_spheres = len(clearances)
        if clearances:
            self.min_clearance = float(np.min(clearances))
        if self.publish_markers:
            self._publish_markers(states)
        return q_work

    def _smooth_safe_command(self, q_safe: np.ndarray, q_start: np.ndarray) -> np.ndarray:
        raw_delta = np.asarray(q_safe - q_start, dtype=float)
        if self.max_avoidance_delta > 0.0:
            raw_delta = np.clip(raw_delta, -self.max_avoidance_delta, self.max_avoidance_delta)
        alpha = self.avoidance_delta_alpha
        filtered_delta = alpha * raw_delta
        return self._clamp_right(q_start + filtered_delta)

    def _sample_spheres(self, states: list[SphereState]) -> None:
        if self.esdf_grid is None:
            return
        for state in states:
            distance, gradient = self.esdf_grid.sample(state.position)
            state.distance = distance
            state.gradient = gradient
            if distance is not None:
                state.clearance = float(distance - state.sphere.radius)

    def _limit_command_step(self, q_safe: np.ndarray, q_baseline: np.ndarray) -> np.ndarray:
        reference = self.last_command if self.last_command is not None else q_baseline
        if self.max_command_step <= 0.0:
            return q_safe
        delta = np.clip(q_safe - reference, -self.max_command_step, self.max_command_step)
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
        if np.isfinite(self.min_clearance):
            self.min_clearance_pub.publish(Float32(data=float(self.min_clearance)))
        mode = "monitor" if self.monitor_only else "active"
        status = (
            f"mode={mode} min_clearance={self.min_clearance:.3f} "
            f"active_constraints={self.active_constraints} sampled_spheres={self.sampled_spheres}/{len(self.spheres)} "
            f"esdf_grid={self.esdf_grid_shape} esdf_observed={self.esdf_observed_count} "
            f"clear_robot={self.clear_robot_from_esdf} "
            f"clear_scale={self.clear_robot_radius_scale:.2f} clear_padding={self.clear_robot_padding:.3f} "
            f"q={np.array2string(q_out, precision=3, suppress_small=True)}"
        )
        self.status_pub.publish(String(data=status))
        self._throttled_info(status)

    def _publish_markers(self, states: list[SphereState]) -> None:
        array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for idx, state in enumerate(states):
            marker = Marker()
            marker.header.frame_id = self.global_frame
            marker.header.stamp = stamp
            marker.ns = "right_arm_esdf_spheres"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = _point_from_array(state.position)
            marker.pose.orientation.w = 1.0
            diameter = 2.0 * float(state.sphere.radius + self.safety_margin)
            marker.scale.x = diameter
            marker.scale.y = diameter
            marker.scale.z = diameter
            marker.color.a = 0.45
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
                marker.color.g = 0.8
                marker.color.b = 0.15
            array.markers.append(marker)
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


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = [float(v) for v in vector]
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def _point_from_array(values: np.ndarray) -> Point:
    point = Point()
    point.x = float(values[0])
    point.y = float(values[1])
    point.z = float(values[2])
    return point


def _vector_from_array(values: np.ndarray) -> Vector3:
    vector = Vector3()
    vector.x = float(values[0])
    vector.y = float(values[1])
    vector.z = float(values[2])
    return vector


def main() -> None:
    rclpy.init()
    node = RightArmEsdfAvoidanceFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
