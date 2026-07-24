#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from openarmx_safety_msgs.msg import ProtectedCable, ProtectedCableArray
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray


def _latched_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _input_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def _rotation_from_quaternion(x: float, y: float, z: float, w: float) -> np.ndarray:
    quat = np.asarray([x, y, z, w], dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return np.eye(3)
    x, y, z, w = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _rpy_rotation_degrees(values: list[float]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.asarray(values, dtype=float))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


class ProtectedCablePerturbationNode(Node):
    """Convert cable markers to structured truth/reference and perturbed estimates."""

    def __init__(self) -> None:
        super().__init__("protected_cable_perturbation")
        self.declare_parameter("input_marker_topic", "/perception/cable_capsules")
        self.declare_parameter("ground_truth_marker_topic", "")
        self.declare_parameter("ground_truth_pose_array_topic", "")
        self.declare_parameter("use_ground_truth_as_estimate", False)
        self.declare_parameter("simulation_cable_radius_m", 0.0075)
        self.declare_parameter("ground_truth_topic", "/protected_cables/ground_truth")
        self.declare_parameter("estimate_topic", "/protected_cables/estimate")
        self.declare_parameter("estimate_marker_topic", "/protected_cables/estimate_markers")
        self.declare_parameter("source", int(ProtectedCable.SOURCE_PERCEPTION_REFERENCE))
        self.declare_parameter("confidence", 1.0)
        self.declare_parameter("translation_bias_m", [0.0, 0.0, 0.0])
        self.declare_parameter("rotation_bias_deg", [0.0, 0.0, 0.0])
        self.declare_parameter("rotation_pivot", "centroid")
        self.declare_parameter("gaussian_position_noise_std_m", 0.0)
        self.declare_parameter("gaussian_radius_noise_std_m", 0.0)
        self.declare_parameter("reported_position_std_m", -1.0)
        self.declare_parameter("reported_radius_std_m", -1.0)
        self.declare_parameter("latency_ms", 0.0)
        self.declare_parameter("dropout_probability", 0.0)
        self.declare_parameter("random_seed", 7)
        self.declare_parameter("publish_estimate_markers", True)

        self.translation_bias = self._vector_parameter("translation_bias_m")
        self.rotation_bias = _rpy_rotation_degrees(self._vector_parameter("rotation_bias_deg"))
        self.rotation_pivot = str(self.get_parameter("rotation_pivot").value).strip().lower()
        if self.rotation_pivot not in ("origin", "centroid"):
            raise ValueError("rotation_pivot must be 'origin' or 'centroid'")
        self.position_noise_std = max(
            0.0, float(self.get_parameter("gaussian_position_noise_std_m").value)
        )
        self.radius_noise_std = max(
            0.0, float(self.get_parameter("gaussian_radius_noise_std_m").value)
        )
        self.reported_position_std = float(self.get_parameter("reported_position_std_m").value)
        self.reported_radius_std = float(self.get_parameter("reported_radius_std_m").value)
        self.latency_s = max(0.0, float(self.get_parameter("latency_ms").value) * 1e-3)
        self.dropout_probability = float(
            np.clip(float(self.get_parameter("dropout_probability").value), 0.0, 1.0)
        )
        self.rng = np.random.default_rng(int(self.get_parameter("random_seed").value))
        self.pending: deque[tuple[float, ProtectedCableArray]] = deque()

        qos = _latched_qos()
        input_qos = _input_qos()
        self.truth_pub = self.create_publisher(
            ProtectedCableArray, str(self.get_parameter("ground_truth_topic").value), qos
        )
        self.estimate_pub = self.create_publisher(
            ProtectedCableArray, str(self.get_parameter("estimate_topic").value), qos
        )
        self.estimate_marker_pub = self.create_publisher(
            MarkerArray, str(self.get_parameter("estimate_marker_topic").value), qos
        )
        input_topic = str(self.get_parameter("input_marker_topic").value)
        truth_marker_topic = str(self.get_parameter("ground_truth_marker_topic").value)
        self.has_separate_truth = bool(truth_marker_topic and truth_marker_topic != input_topic)
        truth_pose_array_topic = str(
            self.get_parameter("ground_truth_pose_array_topic").value
        )
        self.use_ground_truth_as_estimate = bool(
            self.get_parameter("use_ground_truth_as_estimate").value
        )
        self.simulation_cable_radius = max(
            1e-4, float(self.get_parameter("simulation_cable_radius_m").value)
        )
        self.create_subscription(
            MarkerArray,
            input_topic,
            self._markers_cb,
            input_qos,
        )
        if self.has_separate_truth:
            self.create_subscription(
                MarkerArray,
                truth_marker_topic,
                self._truth_markers_cb,
                input_qos,
            )
        if truth_pose_array_topic:
            self.create_subscription(
                PoseArray,
                truth_pose_array_topic,
                self._truth_pose_array_cb,
                input_qos,
            )
        self.timer = self.create_timer(0.005, self._publish_due)
        self.get_logger().info(
            "Protected cable interface: "
            f"estimate_input={input_topic}, "
            f"truth_input={truth_pose_array_topic or truth_marker_topic or input_topic}, "
            f"truth={self.get_parameter('ground_truth_topic').value}, "
            f"estimate={self.get_parameter('estimate_topic').value}, "
            f"bias={self.translation_bias.tolist()} m, latency={self.latency_s * 1000.0:.1f} ms"
        )

    def _vector_parameter(self, name: str) -> np.ndarray:
        value = self.get_parameter(name).value
        if isinstance(value, str):
            values = [float(item) for item in value.strip().strip("[]").split(",")]
        else:
            values = [float(item) for item in value]
        if len(values) != 3 or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain three finite values")
        return np.asarray(values, dtype=float)

    def _markers_cb(self, msg: MarkerArray) -> None:
        truth = self._from_markers(msg)
        if not self.has_separate_truth:
            self.truth_pub.publish(truth)
        self._queue_estimate(truth)

    def _queue_estimate(self, truth: ProtectedCableArray) -> None:
        if self.rng.random() < self.dropout_probability:
            return
        estimate = self._perturb(truth)
        self.pending.append((time.monotonic() + self.latency_s, estimate))

    def _truth_markers_cb(self, msg: MarkerArray) -> None:
        self.truth_pub.publish(self._from_markers(msg, source=ProtectedCable.SOURCE_SIMULATION))

    def _truth_pose_array_cb(self, msg: PoseArray) -> None:
        truth = self._from_simulation_centerlines(msg)
        self.truth_pub.publish(truth)
        if self.use_ground_truth_as_estimate:
            self._queue_estimate(truth)

    def _from_simulation_centerlines(self, msg: PoseArray) -> ProtectedCableArray:
        result = ProtectedCableArray()
        result.header = msg.header
        grouped: dict[int, list[np.ndarray]] = {}
        for pose in msg.poses:
            cable_index = max(0, int(round(float(pose.orientation.w))) - 1)
            point = np.array(
                [pose.position.x, pose.position.y, pose.position.z], dtype=float
            )
            if np.all(np.isfinite(point)):
                grouped.setdefault(cable_index, []).append(point)
        for cable_index in sorted(grouped):
            points = grouped[cable_index]
            for segment_index, (start, end) in enumerate(zip(points[:-1], points[1:])):
                if float(np.linalg.norm(end - start)) <= 1e-6:
                    continue
                cable = ProtectedCable()
                cable.cable_id = f"sim_cable_{cable_index}_segment_{segment_index}"
                cable.start.x, cable.start.y, cable.start.z = start.tolist()
                cable.end.x, cable.end.y, cable.end.z = end.tolist()
                cable.radius = self.simulation_cable_radius
                cable.position_std = 0.0
                cable.radius_std = 0.0
                cable.confidence = 1.0
                cable.source = ProtectedCable.SOURCE_SIMULATION
                result.cables.append(cable)
        return result

    def _from_markers(
        self, msg: MarkerArray, source: int | None = None
    ) -> ProtectedCableArray:
        result = ProtectedCableArray()
        if source is None:
            source = int(self.get_parameter("source").value)
        confidence = float(np.clip(float(self.get_parameter("confidence").value), 0.0, 1.0))
        for marker in msg.markers:
            if marker.action != Marker.ADD or marker.type != Marker.CYLINDER:
                continue
            if marker.ns != "cable_capsules":
                continue
            if not result.header.frame_id:
                result.header = marker.header
            length = float(marker.scale.z)
            radius = 0.25 * float(marker.scale.x + marker.scale.y)
            if length <= 1e-6 or radius <= 0.0:
                continue
            center = np.array(
                [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z], dtype=float
            )
            rotation = _rotation_from_quaternion(
                marker.pose.orientation.x,
                marker.pose.orientation.y,
                marker.pose.orientation.z,
                marker.pose.orientation.w,
            )
            half = 0.5 * length * rotation[:, 2]
            cable = ProtectedCable()
            cable.cable_id = f"cable_{len(result.cables)}"
            cable.start.x, cable.start.y, cable.start.z = (center - half).tolist()
            cable.end.x, cable.end.y, cable.end.z = (center + half).tolist()
            cable.radius = radius
            cable.position_std = 0.0
            cable.radius_std = 0.0
            cable.confidence = confidence
            cable.source = source
            result.cables.append(cable)
        if not result.header.frame_id and msg.markers:
            result.header = msg.markers[0].header
        return result

    def _perturb(self, truth: ProtectedCableArray) -> ProtectedCableArray:
        result = ProtectedCableArray()
        result.header = truth.header
        endpoints = []
        for cable in truth.cables:
            endpoints.extend(
                ([cable.start.x, cable.start.y, cable.start.z], [cable.end.x, cable.end.y, cable.end.z])
            )
        pivot = np.mean(np.asarray(endpoints, dtype=float), axis=0) if endpoints else np.zeros(3)
        if self.rotation_pivot == "origin":
            pivot = np.zeros(3)

        position_std = (
            self.position_noise_std if self.reported_position_std < 0.0 else self.reported_position_std
        )
        radius_std = self.radius_noise_std if self.reported_radius_std < 0.0 else self.reported_radius_std
        for source in truth.cables:
            start = np.array([source.start.x, source.start.y, source.start.z], dtype=float)
            end = np.array([source.end.x, source.end.y, source.end.z], dtype=float)
            start = self.rotation_bias @ (start - pivot) + pivot + self.translation_bias
            end = self.rotation_bias @ (end - pivot) + pivot + self.translation_bias
            if self.position_noise_std > 0.0:
                common_noise = self.rng.normal(0.0, self.position_noise_std, size=3)
                start += common_noise
                end += common_noise
            radius = float(source.radius)
            if self.radius_noise_std > 0.0:
                radius += float(self.rng.normal(0.0, self.radius_noise_std))

            cable = ProtectedCable()
            cable.cable_id = source.cable_id
            cable.start.x, cable.start.y, cable.start.z = start.tolist()
            cable.end.x, cable.end.y, cable.end.z = end.tolist()
            cable.radius = max(1e-4, radius)
            cable.position_std = max(0.0, position_std)
            cable.radius_std = max(0.0, radius_std)
            cable.confidence = source.confidence
            cable.source = source.source
            result.cables.append(cable)
        return result

    def _publish_due(self) -> None:
        now = time.monotonic()
        while self.pending and self.pending[0][0] <= now:
            _, msg = self.pending.popleft()
            self.estimate_pub.publish(msg)
            if bool(self.get_parameter("publish_estimate_markers").value):
                self.estimate_marker_pub.publish(self._to_markers(msg))

    @staticmethod
    def _to_markers(msg: ProtectedCableArray) -> MarkerArray:
        result = MarkerArray()
        delete = Marker()
        delete.header = msg.header
        delete.action = Marker.DELETEALL
        result.markers.append(delete)
        for idx, cable in enumerate(msg.cables):
            start = np.array([cable.start.x, cable.start.y, cable.start.z], dtype=float)
            end = np.array([cable.end.x, cable.end.y, cable.end.z], dtype=float)
            direction = end - start
            length = float(np.linalg.norm(direction))
            if length < 1e-9:
                continue
            z = direction / length
            reference = np.array([0.0, 0.0, 1.0])
            dot = float(np.clip(reference @ z, -1.0, 1.0))
            if dot > 0.999999:
                quat = (0.0, 0.0, 0.0, 1.0)
            elif dot < -0.999999:
                quat = (1.0, 0.0, 0.0, 0.0)
            else:
                axis = np.cross(reference, z)
                axis /= np.linalg.norm(axis)
                half = 0.5 * math.acos(dot)
                quat = (*((axis * math.sin(half)).tolist()), math.cos(half))
            marker = Marker()
            marker.header = msg.header
            marker.ns = "protected_cable_estimate"
            marker.id = idx
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            center = 0.5 * (start + end)
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = center.tolist()
            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]
            marker.scale.x = marker.scale.y = 2.0 * cable.radius
            marker.scale.z = length
            marker.color.r = 1.0
            marker.color.g = 0.2
            marker.color.b = 0.1
            marker.color.a = 0.55
            result.markers.append(marker)
        return result


def main() -> None:
    rclpy.init()
    node = ProtectedCablePerturbationNode()
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
