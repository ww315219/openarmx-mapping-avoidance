#!/usr/bin/env python3
from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Vector3
from nvblox_msgs.srv import EsdfAndGradients
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


def _marker_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _point(values: np.ndarray) -> Point:
    return Point(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def _vector(values: np.ndarray) -> Vector3:
    return Vector3(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def _rotate_local_z(marker: Marker) -> np.ndarray:
    q = marker.pose.orientation
    return np.array(
        [
            2.0 * (q.x * q.z + q.w * q.y),
            2.0 * (q.y * q.z - q.w * q.x),
            1.0 - 2.0 * (q.x * q.x + q.y * q.y),
        ],
        dtype=np.float64,
    )


class FixedCableVoxelClearer(Node):
    def __init__(self) -> None:
        super().__init__("fixed_cable_voxel_clearer")
        self.declare_parameter("capsule_topic", "/perception/cable_capsules")
        self.declare_parameter("esdf_service", "/nvblox_node/get_esdf_and_gradients")
        self.declare_parameter("global_frame", "world")
        self.declare_parameter("clear_delay_s", 10.0)
        self.declare_parameter("clear_radius_m", 0.035)
        self.declare_parameter("sample_spacing_m", 0.03)
        self.declare_parameter("aabb_padding_m", 0.03)
        self.declare_parameter("retry_period_s", 1.0)
        self.declare_parameter("max_attempts", 8)
        self.declare_parameter("status_topic", "/perception/cable_voxel_clear_status")

        self.global_frame = str(self.get_parameter("global_frame").value)
        self.clear_radius = max(0.005, float(self.get_parameter("clear_radius_m").value))
        self.sample_spacing = max(0.005, float(self.get_parameter("sample_spacing_m").value))
        self.aabb_padding = max(0.0, float(self.get_parameter("aabb_padding_m").value))
        self.max_attempts = max(1, int(self.get_parameter("max_attempts").value))
        self.started_ns = self.get_clock().now().nanoseconds
        self.latest_capsules: list[tuple[np.ndarray, np.ndarray]] = []
        self.pending = False
        self.finished = False
        self.attempts = 0

        self.esdf_service_name = str(self.get_parameter("esdf_service").value)
        self.client = self.create_client(EsdfAndGradients, self.esdf_service_name)
        self.status_pub = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            _marker_qos(),
        )
        self.create_subscription(
            MarkerArray,
            str(self.get_parameter("capsule_topic").value),
            self._capsule_cb,
            _marker_qos(),
        )
        retry_period = max(0.2, float(self.get_parameter("retry_period_s").value))
        self.timer = self.create_timer(retry_period, self._timer_cb)

    def _capsule_cb(self, msg: MarkerArray) -> None:
        capsules: list[tuple[np.ndarray, np.ndarray]] = []
        for marker in msg.markers:
            if (
                marker.action != Marker.ADD
                or marker.type != Marker.CYLINDER
                or marker.ns != "cable_capsules"
            ):
                continue
            if marker.header.frame_id != self.global_frame:
                self.get_logger().error(
                    f"Cable capsule frame {marker.header.frame_id!r} does not match "
                    f"global_frame {self.global_frame!r}; refusing to clear."
                )
                return
            center = np.array(
                [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z],
                dtype=np.float64,
            )
            direction = _rotate_local_z(marker)
            half_length = 0.5 * float(marker.scale.z)
            capsules.append(
                (center - half_length * direction, center + half_length * direction)
            )
        if capsules:
            self.latest_capsules = capsules

    def _timer_cb(self) -> None:
        if self.finished or self.pending:
            return
        delay_ns = int(max(0.0, float(self.get_parameter("clear_delay_s").value)) * 1e9)
        if self.get_clock().now().nanoseconds - self.started_ns < delay_ns:
            return
        if not self.latest_capsules:
            self._publish_status("waiting_for_fixed_capsules")
            return
        if not self.client.service_is_ready():
            discovered = self._discover_esdf_service()
            if discovered:
                self.esdf_service_name = discovered
                self.client = self.create_client(EsdfAndGradients, discovered)
            self._publish_status("waiting_for_nvblox_service")
            return
        if self.attempts >= self.max_attempts:
            self.finished = True
            self._publish_status("failed_max_attempts")
            self.get_logger().error("Cable voxel clearing failed after maximum attempts.")
            return

        centers = self._sample_capsules()
        if not centers:
            self.finished = True
            self._publish_status("failed_no_samples")
            return
        positions = np.vstack(centers)
        padding = self.clear_radius + self.aabb_padding
        aabb_min = np.min(positions, axis=0) - padding
        aabb_max = np.max(positions, axis=0) + padding

        request = EsdfAndGradients.Request()
        request.update_esdf = True
        request.visualize_esdf = False
        request.use_aabb = True
        request.frame_id = self.global_frame
        request.aabb_min_m = _point(aabb_min)
        request.aabb_size_m = _vector(aabb_max - aabb_min)
        for center in centers:
            request.spheres_to_clear_center_m.append(_point(center))
            request.spheres_to_clear_radius_m.append(float(self.clear_radius))

        self.attempts += 1
        self.pending = True
        self._publish_status(
            f"clearing attempt={self.attempts} capsules={len(self.latest_capsules)} "
            f"samples={len(centers)} radius={self.clear_radius:.3f}m"
        )
        future = self.client.call_async(request)
        future.add_done_callback(self._response_cb)

    def _discover_esdf_service(self) -> str:
        for name, service_types in self.get_service_names_and_types():
            if "nvblox_msgs/srv/EsdfAndGradients" in service_types:
                return name
        return ""

    def _sample_capsules(self) -> list[np.ndarray]:
        centers: list[np.ndarray] = []
        for start, end in self.latest_capsules:
            length = float(np.linalg.norm(end - start))
            count = max(2, int(math.ceil(length / self.sample_spacing)) + 1)
            for alpha in np.linspace(0.0, 1.0, count):
                centers.append((1.0 - alpha) * start + alpha * end)
        return centers

    def _response_cb(self, future) -> None:
        self.pending = False
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self._publish_status(f"service_error attempt={self.attempts} error={exc}")
            return
        if response is None or not response.success:
            self._publish_status(f"service_failure attempt={self.attempts}")
            return
        self.finished = True
        message = (
            f"cleared capsules={len(self.latest_capsules)} radius={self.clear_radius:.3f}m "
            f"attempts={self.attempts}"
        )
        self._publish_status(message)
        self.get_logger().info(message)

    def _publish_status(self, text: str) -> None:
        self.status_pub.publish(String(data=text))


def main() -> None:
    rclpy.init()
    node = FixedCableVoxelClearer()
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
