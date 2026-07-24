#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


def _sensor_qos(depth: int = 5) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def _input_qos(depth: int = 10) -> QoSProfile:
    """Match the FoundationStereo and RealSense reliable publishers exactly."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def _marker_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class FixedCableDepthMaskNode(Node):
    """Remove only depth samples consistent with persistent cable capsules."""

    def __init__(self) -> None:
        super().__init__("fixed_cable_depth_mask")
        self.declare_parameter("enabled", True)
        self.declare_parameter("input_depth_topic", "/foundation_stereo/depth")
        self.declare_parameter("input_camera_info_topic", "/camera/infra1/camera_info")
        self.declare_parameter("capsule_topic", "/perception/cable_capsules")
        self.declare_parameter("output_depth_topic", "/perception/cable_removed_depth")
        self.declare_parameter(
            "output_camera_info_topic", "/perception/cable_removed_depth/camera_info"
        )
        self.declare_parameter("output_mask_topic", "/perception/fixed_cable_depth_mask")
        self.declare_parameter("status_topic", "/perception/fixed_cable_depth_mask_status")
        self.declare_parameter("mask_radius_m", 0.03)
        self.declare_parameter("depth_tolerance_m", 0.05)
        self.declare_parameter("sample_spacing_m", 0.012)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_pixel_radius", 40)
        self.declare_parameter("publish_debug_mask", True)
        self.declare_parameter("hold_until_ready", True)
        self.declare_parameter("log_period_s", 2.0)
        self.declare_parameter("input_watchdog_timeout_s", 3.0)
        self.declare_parameter("depth_hole_fill_enabled", True)
        self.declare_parameter("depth_hole_fill_min_neighbors", 3)
        self.declare_parameter("depth_hole_fill_max_spread_m", 0.03)

        self.enabled = bool(self.get_parameter("enabled").value)
        self.input_depth_topic = str(self.get_parameter("input_depth_topic").value)
        self.input_info_topic = str(self.get_parameter("input_camera_info_topic").value)
        self.output_depth_topic = str(self.get_parameter("output_depth_topic").value)
        self.output_info_topic = str(self.get_parameter("output_camera_info_topic").value)
        self.mask_radius_m = max(0.001, float(self.get_parameter("mask_radius_m").value))
        self.depth_tolerance_m = max(0.001, float(self.get_parameter("depth_tolerance_m").value))
        self.sample_spacing_m = max(0.003, float(self.get_parameter("sample_spacing_m").value))
        self.min_depth_m = max(0.001, float(self.get_parameter("min_depth_m").value))
        self.max_pixel_radius = max(1, int(self.get_parameter("max_pixel_radius").value))
        self.publish_debug_mask = bool(self.get_parameter("publish_debug_mask").value)
        self.hold_until_ready = bool(self.get_parameter("hold_until_ready").value)
        self.log_period_s = max(0.2, float(self.get_parameter("log_period_s").value))
        self.input_watchdog_timeout_s = max(
            1.0, float(self.get_parameter("input_watchdog_timeout_s").value)
        )
        self.depth_hole_fill_enabled = bool(
            self.get_parameter("depth_hole_fill_enabled").value
        )
        self.depth_hole_fill_min_neighbors = min(
            8, max(1, int(self.get_parameter("depth_hole_fill_min_neighbors").value))
        )
        self.depth_hole_fill_max_spread_m = max(
            0.001, float(self.get_parameter("depth_hole_fill_max_spread_m").value)
        )

        self.bridge = CvBridge()
        self.camera_info: Optional[CameraInfo] = None
        self.capsules: list[tuple[str, np.ndarray, np.ndarray, float]] = []
        self.last_log_ns = 0
        self.frame_count = 0
        self.last_removed_pixels = 0
        self.last_filled_pixels = 0
        self.last_depth_receive_ns = self.get_clock().now().nanoseconds

        self.tf_buffer = tf2_ros.Buffer()
        # This node is already spun by main(). Spinning the same node again from
        # TransformListener can starve its depth and camera-info subscriptions.
        # TF lookups below are non-blocking, so one executor is sufficient.
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=False)

        self.depth_pub = self.create_publisher(Image, self.output_depth_topic, _sensor_qos())
        self.info_pub = self.create_publisher(CameraInfo, self.output_info_topic, _sensor_qos())
        self.mask_pub = self.create_publisher(
            Image, str(self.get_parameter("output_mask_topic").value), _sensor_qos()
        )
        self.status_pub = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10
        )
        self.depth_sub = self.create_subscription(
            Image, self.input_depth_topic, self._depth_cb, _input_qos()
        )
        self.info_sub = self.create_subscription(
            CameraInfo, self.input_info_topic, self._info_cb, _input_qos()
        )
        self.create_subscription(
            MarkerArray,
            str(self.get_parameter("capsule_topic").value),
            self._capsules_cb,
            _marker_qos(),
        )
        self.input_watchdog = self.create_timer(1.0, self._input_watchdog_cb)
        self.get_logger().info(
            "Fixed cable depth mask: "
            f"{self.input_depth_topic} -> {self.output_depth_topic}, enabled={self.enabled}, "
            f"radius={self.mask_radius_m:.3f}m tolerance={self.depth_tolerance_m:.3f}m, "
            f"hole_fill={self.depth_hole_fill_enabled}"
        )

    def _info_cb(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def _capsules_cb(self, msg: MarkerArray) -> None:
        capsules = []
        for marker in msg.markers:
            if marker.action != Marker.ADD or marker.type != Marker.CYLINDER:
                continue
            if marker.ns != "cable_capsules" or marker.scale.z <= 0.0:
                continue
            rotation = _quat_to_matrix(
                marker.pose.orientation.x,
                marker.pose.orientation.y,
                marker.pose.orientation.z,
                marker.pose.orientation.w,
            )
            center = np.array(
                [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z],
                dtype=np.float64,
            )
            half_axis = rotation[:, 2] * (0.5 * float(marker.scale.z))
            radius = 0.5 * max(float(marker.scale.x), float(marker.scale.y))
            capsules.append((marker.header.frame_id, center - half_axis, center + half_axis, radius))
        if capsules:
            self.capsules = capsules

    def _depth_cb(self, msg: Image) -> None:
        self.last_depth_receive_ns = self.get_clock().now().nanoseconds
        if not self.enabled:
            self.depth_pub.publish(msg)
            self._publish_info(msg)
            return
        if self.camera_info is None or not self.capsules:
            if not self.hold_until_ready:
                self.depth_pub.publish(msg)
                self._publish_info(msg)
            self._maybe_log("waiting for camera info/capsules")
            return

        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().warn(f"Depth conversion failed: {exc}")
            return
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]
        depth_m = self._to_meters(depth_raw, msg.encoding)
        expected_depth = np.full(depth_m.shape, np.nan, dtype=np.float32)

        camera_frame = self.camera_info.header.frame_id or msg.header.frame_id
        if not camera_frame:
            self._maybe_log("camera frame is empty")
            return
        try:
            self._rasterize_capsules(expected_depth, camera_frame)
        except Exception as exc:
            if not self.hold_until_ready:
                self.depth_pub.publish(msg)
                self._publish_info(msg)
            self._maybe_log(f"TF unavailable: {exc}")
            return

        projected = np.isfinite(expected_depth)
        valid = np.isfinite(depth_m) & (depth_m > self.min_depth_m)
        remove = projected & valid & (
            np.abs(depth_m - expected_depth) <= self.depth_tolerance_m
        )
        output = np.array(depth_raw, copy=True)
        output[remove] = 0
        output_m, filled_pixels = self._fill_small_depth_holes(
            self._to_meters(output, msg.encoding), projected
        )
        if msg.encoding.upper() == "16UC1" or output.dtype == np.uint16:
            output = np.clip(
                np.rint(output_m * 1000.0), 0, np.iinfo(np.uint16).max
            ).astype(np.uint16)
        else:
            output = output_m.astype(output.dtype, copy=False)
        output_msg = self.bridge.cv2_to_imgmsg(output, encoding=msg.encoding)
        output_msg.header = msg.header
        self.depth_pub.publish(output_msg)
        self._publish_info(msg)

        if self.publish_debug_mask:
            mask_msg = self.bridge.cv2_to_imgmsg((remove.astype(np.uint8) * 255), encoding="mono8")
            mask_msg.header = msg.header
            self.mask_pub.publish(mask_msg)
        self.frame_count += 1
        self.last_removed_pixels = int(np.count_nonzero(remove))
        self.last_filled_pixels = filled_pixels
        self._maybe_log("active")

    def _input_watchdog_cb(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        age_s = (now_ns - self.last_depth_receive_ns) * 1e-9
        if age_s < self.input_watchdog_timeout_s:
            return
        self.get_logger().warn(
            f"No depth callback for {age_s:.1f}s; recreating input subscriptions."
        )
        self.destroy_subscription(self.depth_sub)
        self.destroy_subscription(self.info_sub)
        self.depth_sub = self.create_subscription(
            Image, self.input_depth_topic, self._depth_cb, _input_qos()
        )
        self.info_sub = self.create_subscription(
            CameraInfo, self.input_info_topic, self._info_cb, _input_qos()
        )
        self.last_depth_receive_ns = now_ns

    def _rasterize_capsules(self, expected: np.ndarray, camera_frame: str) -> None:
        info = self.camera_info
        assert info is not None
        fx, fy = float(info.k[0]), float(info.k[4])
        cx, cy = float(info.k[2]), float(info.k[5])
        height, width = expected.shape
        for frame_id, start, end, marker_radius in self.capsules:
            transform = self.tf_buffer.lookup_transform(
                camera_frame,
                frame_id,
                Time(),
                timeout=Duration(seconds=0.0),
            ).transform
            rot = _quat_to_matrix(
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w,
            )
            translation = np.array(
                [transform.translation.x, transform.translation.y, transform.translation.z],
                dtype=np.float64,
            )
            length = float(np.linalg.norm(end - start))
            count = max(2, int(math.ceil(length / self.sample_spacing_m)) + 1)
            radius_m = max(self.mask_radius_m, marker_radius)
            for alpha in np.linspace(0.0, 1.0, count):
                point = rot @ (start + alpha * (end - start)) + translation
                z = float(point[2])
                if z <= self.min_depth_m:
                    continue
                u = int(round(fx * float(point[0]) / z + cx))
                v = int(round(fy * float(point[1]) / z + cy))
                radius_px = min(
                    self.max_pixel_radius,
                    max(1, int(math.ceil(max(fx, fy) * radius_m / z))),
                )
                if u + radius_px < 0 or u - radius_px >= width:
                    continue
                if v + radius_px < 0 or v - radius_px >= height:
                    continue
                cv2.circle(expected, (u, v), radius_px, z, thickness=-1)

    def _publish_info(self, depth_msg: Image) -> None:
        if self.camera_info is None:
            return
        info = CameraInfo()
        info.header = depth_msg.header
        info.height = self.camera_info.height
        info.width = self.camera_info.width
        info.distortion_model = self.camera_info.distortion_model
        info.d = list(self.camera_info.d)
        info.k = list(self.camera_info.k)
        info.r = list(self.camera_info.r)
        info.p = list(self.camera_info.p)
        info.binning_x = self.camera_info.binning_x
        info.binning_y = self.camera_info.binning_y
        info.roi = self.camera_info.roi
        self.info_pub.publish(info)

    @staticmethod
    def _to_meters(depth: np.ndarray, encoding: str) -> np.ndarray:
        if encoding.upper() == "16UC1" or depth.dtype == np.uint16:
            return depth.astype(np.float32) * 0.001
        return depth.astype(np.float32)

    def _fill_small_depth_holes(
        self, depth_m: np.ndarray, protected_mask: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """Fill isolated 1-pixel holes without extending across depth edges."""
        if not self.depth_hole_fill_enabled:
            return depth_m, 0

        valid = np.isfinite(depth_m) & (depth_m > self.min_depth_m)
        kernel = np.ones((3, 3), dtype=np.float32)
        neighbor_count = cv2.filter2D(
            valid.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT
        )
        depth_sum = cv2.filter2D(
            np.where(valid, depth_m, 0.0).astype(np.float32),
            -1,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
        )

        local_min = cv2.erode(
            np.where(valid, depth_m, np.inf).astype(np.float32), kernel
        )
        local_max = cv2.dilate(
            np.where(valid, depth_m, -np.inf).astype(np.float32), kernel
        )
        finite_range = np.isfinite(local_min) & np.isfinite(local_max)
        local_spread = np.full(depth_m.shape, np.inf, dtype=np.float32)
        np.subtract(local_max, local_min, out=local_spread, where=finite_range)
        fill = (
            ~valid
            & ~protected_mask
            & (neighbor_count >= float(self.depth_hole_fill_min_neighbors))
            & finite_range
            & (local_spread <= self.depth_hole_fill_max_spread_m)
        )
        filled_pixels = int(np.count_nonzero(fill))
        if filled_pixels == 0:
            return depth_m, 0

        result = np.array(depth_m, copy=True)
        result[fill] = depth_sum[fill] / neighbor_count[fill]
        return result, filled_pixels

    def _maybe_log(self, state: str) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_log_ns < int(self.log_period_s * 1e9):
            return
        self.last_log_ns = now_ns
        status = (
            f"state={state} capsules={len(self.capsules)} frames={self.frame_count} "
            f"removed_pixels={self.last_removed_pixels} "
            f"filled_pixels={self.last_filled_pixels}"
        )
        self.status_pub.publish(String(data=status))
        self.get_logger().info(status)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FixedCableDepthMaskNode()
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
