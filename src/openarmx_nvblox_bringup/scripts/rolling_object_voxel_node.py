#!/usr/bin/env python3

"""Publish a short-lived voxel view of currently observed non-cable objects."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class Capsule:
    start: np.ndarray
    end: np.ndarray
    radius: float


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1.0e-9:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class RollingObjectVoxelNode(Node):
    def __init__(self) -> None:
        super().__init__("rolling_object_voxels")
        self.declare_parameter("global_frame", "world")
        self.declare_parameter("input_pointcloud_topic", "/foundation_stereo/points")
        self.declare_parameter("capsule_topic", "/perception/cable_capsules")
        self.declare_parameter("output_marker_topic", "/perception/rolling_object_voxels")
        self.declare_parameter("output_pointcloud_topic", "/perception/rolling_object_points")
        self.declare_parameter("status_topic", "/perception/rolling_object_voxel_status")
        self.declare_parameter("voxel_size_m", 0.015)
        self.declare_parameter("ttl_s", 1.2)
        self.declare_parameter("min_depth_m", 0.10)
        self.declare_parameter("max_depth_m", 1.40)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("cable_exclusion_padding_m", 0.025)
        self.declare_parameter("max_voxels", 100000)

        self.global_frame = str(self.get_parameter("global_frame").value)
        self.voxel_size = max(0.003, float(self.get_parameter("voxel_size_m").value))
        self.ttl_ns = int(max(0.05, float(self.get_parameter("ttl_s").value)) * 1.0e9)
        self.min_depth = max(0.0, float(self.get_parameter("min_depth_m").value))
        self.max_depth = max(
            self.min_depth, float(self.get_parameter("max_depth_m").value)
        )
        self.cable_padding = max(
            0.0, float(self.get_parameter("cable_exclusion_padding_m").value)
        )
        self.max_voxels = max(100, int(self.get_parameter("max_voxels").value))

        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.capsules: List[Capsule] = []
        self.voxels: Dict[Tuple[int, int, int], int] = {}
        self.received_clouds = 0
        self.last_status_ns = 0
        self.marker_visible = False

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        capsule_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("input_pointcloud_topic").value),
            self._pointcloud_callback,
            sensor_qos,
        )
        self.create_subscription(
            MarkerArray,
            str(self.get_parameter("capsule_topic").value),
            self._capsule_callback,
            capsule_qos,
        )
        self.marker_pub = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("output_marker_topic").value),
            output_qos,
        )
        self.cloud_pub = self.create_publisher(
            PointCloud2,
            str(self.get_parameter("output_pointcloud_topic").value),
            output_qos,
        )
        self.status_pub = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), output_qos
        )
        rate = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.create_timer(1.0 / rate, self._publish)
        self.get_logger().info(
            "Rolling object voxels: input=%s depth=[%.2f, %.2f]m voxel=%.3fm "
            "ttl=%.2fs cable_padding=%.3fm"
            % (
                self.get_parameter("input_pointcloud_topic").value,
                self.min_depth,
                self.max_depth,
                self.voxel_size,
                self.ttl_ns / 1.0e9,
                self.cable_padding,
            )
        )

    def _lookup_transform(self, target: str, source: str):
        if not source or source == target:
            return None
        return self.tf_buffer.lookup_transform(
            target, source, Time(), timeout=Duration(seconds=0.05)
        )

    @staticmethod
    def _apply_transform(points: np.ndarray, transform) -> np.ndarray:
        if transform is None:
            return points
        t = transform.transform.translation
        q = transform.transform.rotation
        rotation = _quat_to_matrix(q.x, q.y, q.z, q.w)
        translation = np.array([t.x, t.y, t.z], dtype=np.float64)
        return points @ rotation.T + translation

    def _capsule_callback(self, msg: MarkerArray) -> None:
        capsules: List[Capsule] = []
        clear_requested = False
        for marker in msg.markers:
            if marker.action == Marker.DELETEALL:
                capsules.clear()
                clear_requested = True
                continue
            if marker.action == Marker.DELETE or marker.type != Marker.CYLINDER:
                continue
            frame = marker.header.frame_id or self.global_frame
            center = np.array(
                [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z],
                dtype=np.float64,
            )
            q = marker.pose.orientation
            rotation = _quat_to_matrix(q.x, q.y, q.z, q.w)
            half_axis = rotation[:, 2] * (0.5 * float(marker.scale.z))
            endpoints = np.vstack((center - half_axis, center + half_axis))
            try:
                transform = self._lookup_transform(self.global_frame, frame)
                endpoints = self._apply_transform(endpoints, transform)
            except TransformException as exc:
                self.get_logger().warning(
                    f"Cannot transform cable capsule {frame}->{self.global_frame}: {exc}",
                    throttle_duration_sec=2.0,
                )
                continue
            capsules.append(
                Capsule(
                    start=endpoints[0],
                    end=endpoints[1],
                    radius=0.5 * max(float(marker.scale.x), float(marker.scale.y)),
                )
            )
        if capsules or clear_requested:
            self.capsules = capsules

    @staticmethod
    def _cloud_xyz(msg: PointCloud2) -> np.ndarray:
        values = point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
        if isinstance(values, np.ndarray):
            if values.dtype.names:
                points = np.column_stack((values["x"], values["y"], values["z"]))
            else:
                points = np.asarray(values)
        else:
            points = np.asarray(list(values))
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(points[:, :3], dtype=np.float64)

    def _exclude_cables(self, points: np.ndarray) -> np.ndarray:
        if not self.capsules or points.size == 0:
            return points
        keep = np.ones(points.shape[0], dtype=bool)
        for capsule in self.capsules:
            segment = capsule.end - capsule.start
            length_sq = float(np.dot(segment, segment))
            if length_sq < 1.0e-10:
                distance_sq = np.sum((points - capsule.start) ** 2, axis=1)
            else:
                ratio = np.clip(((points - capsule.start) @ segment) / length_sq, 0.0, 1.0)
                closest = capsule.start + ratio[:, None] * segment
                distance_sq = np.sum((points - closest) ** 2, axis=1)
            limit = capsule.radius + self.cable_padding
            keep &= distance_sq > limit * limit
        return points[keep]

    def _pointcloud_callback(self, msg: PointCloud2) -> None:
        try:
            points = self._cloud_xyz(msg)
            finite = np.all(np.isfinite(points), axis=1)
            depth = points[:, 2]
            points = points[
                finite & (depth >= self.min_depth) & (depth <= self.max_depth)
            ]
            transform = self._lookup_transform(self.global_frame, msg.header.frame_id)
            points = self._apply_transform(points, transform)
        except (TransformException, ValueError) as exc:
            self.get_logger().warning(
                f"Skipping object cloud: {exc}", throttle_duration_sec=2.0
            )
            return
        points = self._exclude_cables(points)
        if points.size:
            indices = np.floor(points / self.voxel_size).astype(np.int64)
            for index in np.unique(indices, axis=0):
                key = (int(index[0]), int(index[1]), int(index[2]))
                self.voxels[key] = self.get_clock().now().nanoseconds
        self.received_clouds += 1
        self._expire(self.get_clock().now().nanoseconds)

    def _expire(self, now_ns: int) -> None:
        cutoff = now_ns - self.ttl_ns
        expired = [key for key, last_seen in self.voxels.items() if last_seen < cutoff]
        for key in expired:
            del self.voxels[key]
        if len(self.voxels) > self.max_voxels:
            remove_count = len(self.voxels) - self.max_voxels
            oldest = sorted(self.voxels, key=self.voxels.get)[:remove_count]
            for key in oldest:
                del self.voxels[key]

    def _centers(self) -> np.ndarray:
        if not self.voxels:
            return np.empty((0, 3), dtype=np.float32)
        indices = np.asarray(list(self.voxels.keys()), dtype=np.float32)
        return (indices + 0.5) * self.voxel_size

    def _publish(self) -> None:
        now = self.get_clock().now()
        self._expire(now.nanoseconds)
        centers = self._centers()
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = now.to_msg()
        marker.ns = "rolling_object_voxels"
        marker.id = 0
        marker.action = Marker.ADD if centers.size else Marker.DELETE
        marker.type = Marker.CUBE_LIST
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.voxel_size
        marker.scale.y = self.voxel_size
        marker.scale.z = self.voxel_size
        marker.color.r = 0.55
        marker.color.g = 0.58
        marker.color.b = 0.62
        marker.color.a = 0.9
        marker.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in centers]
        self.marker_pub.publish(MarkerArray(markers=[marker]))

        header = Header(stamp=now.to_msg(), frame_id=self.global_frame)
        self.cloud_pub.publish(point_cloud2.create_cloud_xyz32(header, centers.tolist()))

        if now.nanoseconds - self.last_status_ns > int(2.0e9):
            status = String()
            status.data = (
                f"clouds={self.received_clouds} voxels={len(self.voxels)} "
                f"capsules={len(self.capsules)} ttl_s={self.ttl_ns / 1.0e9:.2f}"
            )
            self.status_pub.publish(status)
            self.last_status_ns = now.nanoseconds


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RollingObjectVoxelNode()
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
