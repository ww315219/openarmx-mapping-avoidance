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
from visualization_msgs.msg import Marker, MarkerArray

from openarmx_obstacle_avoidance.right_arm_esdf_avoidance_filter import (
    CollisionSphere,
    SphereState,
    _point_from_array,
    _skew,
    _vector_from_array,
)


LEFT_JOINT_NAMES = [f"openarmx_left_joint{i}" for i in range(1, 8)]
RIGHT_JOINT_NAMES = [f"openarmx_right_joint{i}" for i in range(1, 8)]


class BimanualRobotEsdfClearer(Node):
    """Periodically asks nvblox to clear the robot body from the reconstructed map."""

    def __init__(self) -> None:
        super().__init__("bimanual_robot_esdf_clearer")

        self.declare_parameter("urdf_path", "")
        self.declare_parameter("robot_description_node", "/robot_state_publisher")
        self.declare_parameter("left_joint_names", LEFT_JOINT_NAMES)
        self.declare_parameter("right_joint_names", RIGHT_JOINT_NAMES)
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("esdf_service", "")
        self.declare_parameter("global_frame", "world")
        self.declare_parameter("rate_hz", 2.0)
        self.declare_parameter("request_update_esdf", False)
        self.declare_parameter("aabb_padding", 0.08)
        self.declare_parameter("clear_robot_padding", 0.015)
        self.declare_parameter("clear_robot_radius_scale", 1.0)
        self.declare_parameter("min_joint_delta_to_clear", 0.08)
        self.declare_parameter("force_clear_period_s", 4.0)
        self.declare_parameter("collision_model", "capsule")
        self.declare_parameter("capsule_samples_per_link", 3)
        self.declare_parameter("publish_debug_markers", True)
        self.declare_parameter("debug_marker_topic", "/openarmx/robot_clear_shapes")

        self.urdf_path = str(self.get_parameter("urdf_path").value).strip()
        self.robot_description_node = str(self.get_parameter("robot_description_node").value)
        self.left_joint_names = list(self.get_parameter("left_joint_names").value)
        self.right_joint_names = list(self.get_parameter("right_joint_names").value)
        self.joint_names = self.left_joint_names + self.right_joint_names
        self.joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self.esdf_service_name = str(self.get_parameter("esdf_service").value).strip()
        self.global_frame = str(self.get_parameter("global_frame").value)
        self.rate_hz = max(0.5, float(self.get_parameter("rate_hz").value))
        self.request_update_esdf = bool(self.get_parameter("request_update_esdf").value)
        self.aabb_padding = max(0.01, float(self.get_parameter("aabb_padding").value))
        self.clear_robot_padding = max(0.0, float(self.get_parameter("clear_robot_padding").value))
        self.clear_robot_radius_scale = max(
            0.0,
            float(self.get_parameter("clear_robot_radius_scale").value),
        )
        self.min_joint_delta_to_clear = max(
            0.0,
            float(self.get_parameter("min_joint_delta_to_clear").value),
        )
        self.force_clear_period_s = max(
            0.0,
            float(self.get_parameter("force_clear_period_s").value),
        )
        self.collision_model = str(self.get_parameter("collision_model").value).strip().lower()
        if self.collision_model not in ("sphere", "capsule"):
            self.get_logger().warn(
                f"Unknown collision_model={self.collision_model!r}; falling back to 'capsule'."
            )
            self.collision_model = "capsule"
        self.capsule_samples_per_link = max(
            2,
            int(self.get_parameter("capsule_samples_per_link").value),
        )
        self.publish_debug_markers = bool(self.get_parameter("publish_debug_markers").value)

        self._temp_urdf_path: str | None = None
        if not self.urdf_path:
            self.urdf_path = self._write_robot_description_to_temp_urdf()
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        self.q_model_current = pin.neutral(self.model)
        self.joint_q_indices = self._joint_q_indices(self.joint_names)
        self.joint_v_indices = self._joint_v_indices(self.joint_names)
        self.spheres = self._default_bimanual_spheres()
        self.sphere_frame_ids = [self.model.getFrameId(sphere.frame) for sphere in self.spheres]

        self.have_joint_state = False
        self.esdf_pending = False
        self.last_clear_q: np.ndarray | None = None
        self.last_clear_time = self.get_clock().now()
        self.last_missing_joint_warn_time = self.get_clock().now()

        self.esdf_client = None
        self._connect_esdf_client()

        self.create_subscription(JointState, self.joint_states_topic, self._joint_state_cb, 20)
        self.marker_pub = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("debug_marker_topic").value),
            10,
        )
        self.timer = self.create_timer(1.0 / self.rate_hz, self._timer_cb)

        self.get_logger().info(
            "robot ESDF clearer: "
            f"joints={len(self.joint_names)}, clear_spheres={len(self.spheres)}, "
            f"rate={self.rate_hz:.1f}Hz, service={self.esdf_service_name or 'auto'}, "
            f"min_joint_delta={self.min_joint_delta_to_clear:.3f}, "
            f"force_period={self.force_clear_period_s:.1f}s, "
            f"radius_scale={self.clear_robot_radius_scale:.2f}, "
            f"padding={self.clear_robot_padding:.3f}"
        )

    def destroy_node(self) -> bool:
        if self._temp_urdf_path and os.path.exists(self._temp_urdf_path):
            try:
                os.remove(self._temp_urdf_path)
            except OSError:
                pass
        return super().destroy_node()

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

        fd, path = tempfile.mkstemp(prefix="openarmx_robot_esdf_clearer_", suffix=".urdf")
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
            ("hand", [0.0, 0.0, 0.000], [0.0, 0.0, 0.060], 0.048),
        ]
        return self._build_capsule_collision_points(raw_one_arm_capsules)

    def _build_sphere_collision_points(
        self,
        raw_one_arm: list[tuple[str, list[float], float]],
    ) -> list[CollisionSphere]:
        spheres: list[CollisionSphere] = []
        for side in ("left", "right"):
            for suffix, center, radius in raw_one_arm:
                frame = f"openarmx_{side}_{suffix}"
                if not self.model.existFrame(frame):
                    self.get_logger().warn(f"Skipping clear sphere on missing frame {frame!r}")
                    continue
                spheres.append(CollisionSphere(frame, np.asarray(center, dtype=float), float(radius)))
        if not spheres:
            raise RuntimeError("No valid robot clear spheres were configured.")
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
                    self.get_logger().warn(f"Skipping clear capsule on missing frame {frame!r}")
                    continue
                start_np = np.asarray(start, dtype=float).reshape(3)
                end_np = np.asarray(end, dtype=float).reshape(3)
                for sample_idx in range(self.capsule_samples_per_link):
                    alpha = sample_idx / float(self.capsule_samples_per_link - 1)
                    center = (1.0 - alpha) * start_np + alpha * end_np
                    spheres.append(CollisionSphere(frame, center.astype(float), float(radius)))
        if not spheres:
            raise RuntimeError("No valid robot clear capsule samples were configured.")
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

        missing = [name for name in self.joint_names if name not in name_to_pos]
        now = self.get_clock().now()
        if missing and (now - self.last_missing_joint_warn_time).nanoseconds > 5_000_000_000:
            self.last_missing_joint_warn_time = now
            self.get_logger().warn(
                f"JointState missing {len(missing)} configured joints; using previous/default values."
            )

    def _compute_sphere_states(self) -> list[SphereState]:
        pin.forwardKinematics(self.model, self.data, self.q_model_current)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, self.q_model_current)

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

    def _timer_cb(self) -> None:
        if not self.have_joint_state or self.esdf_pending:
            return
        now = self.get_clock().now()
        q_configured = np.asarray([self.q_model_current[idx] for idx in self.joint_q_indices], dtype=float)
        if self.last_clear_q is not None:
            joint_delta = float(np.max(np.abs(q_configured - self.last_clear_q)))
            elapsed = (now - self.last_clear_time).nanoseconds * 1e-9
            if (
                joint_delta < self.min_joint_delta_to_clear
                and (self.force_clear_period_s <= 0.0 or elapsed < self.force_clear_period_s)
            ):
                return
        if self.esdf_client is None:
            self._connect_esdf_client()
        if self.esdf_client is None:
            return
        if not self.esdf_client.service_is_ready():
            service_name = self._discover_esdf_service()
            if service_name and service_name != self.esdf_service_name:
                self.esdf_service_name = service_name
                self.esdf_client = self.create_client(EsdfAndGradients, service_name)
                self.get_logger().info(f"Switched ESDF service to discovered service: {service_name}")
            else:
                self.get_logger().warn(f"ESDF service not ready: {self.esdf_service_name}")
                return

        states = self._compute_sphere_states()
        if not states:
            return

        positions = np.vstack([state.position for state in states])
        clear_radii = np.asarray(
            [
                state.sphere.radius * self.clear_robot_radius_scale + self.clear_robot_padding
                for state in states
            ],
            dtype=float,
        )
        padding = self.aabb_padding + float(np.max(clear_radii))
        aabb_min = np.min(positions, axis=0) - padding
        aabb_max = np.max(positions, axis=0) + padding

        req = EsdfAndGradients.Request()
        req.update_esdf = self.request_update_esdf
        req.visualize_esdf = False
        req.use_aabb = True
        req.frame_id = self.global_frame
        req.aabb_min_m = _point_from_array(aabb_min)
        req.aabb_size_m = _vector_from_array(aabb_max - aabb_min)
        for state, clear_radius in zip(states, clear_radii):
            req.spheres_to_clear_center_m.append(_point_from_array(state.position))
            req.spheres_to_clear_radius_m.append(float(clear_radius))

        future = self.esdf_client.call_async(req)
        future.add_done_callback(self._esdf_response_cb)
        self.esdf_pending = True
        self.last_clear_q = q_configured.copy()
        self.last_clear_time = now

        if self.publish_debug_markers:
            self._publish_debug_markers(states, clear_radii)

    def _esdf_response_cb(self, future) -> None:
        self.esdf_pending = False
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Robot ESDF clear service call failed: {exc}")
            return
        if response is None or not response.success:
            self.get_logger().warn("Robot ESDF clear service returned failure")

    def _publish_debug_markers(self, states: list[SphereState], clear_radii: np.ndarray) -> None:
        array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        array.markers.append(delete_marker)

        stamp = self.get_clock().now().to_msg()
        for idx, (state, radius) in enumerate(zip(states, clear_radii)):
            marker = Marker()
            marker.header.frame_id = self.global_frame
            marker.header.stamp = stamp
            marker.ns = "robot_esdf_clear_spheres"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = _point_from_array(state.position)
            marker.pose.orientation.w = 1.0
            diameter = 2.0 * float(radius)
            marker.scale.x = diameter
            marker.scale.y = diameter
            marker.scale.z = diameter
            marker.color.r = 0.2
            marker.color.g = 0.8
            marker.color.b = 1.0
            marker.color.a = 0.20
            array.markers.append(marker)
        self.marker_pub.publish(array)


def main() -> None:
    rclpy.init()
    node = BimanualRobotEsdfClearer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
