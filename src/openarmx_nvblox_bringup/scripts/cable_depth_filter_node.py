#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image


def _sensor_qos(depth: int = 5) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


class CableDepthFilterNode(Node):
    """Extract a cable-only depth image from a clean-background depth stream.

    This first version is deliberately geometry-based:
    - valid depth pixels form a foreground mask;
    - long Hough line segments are treated as cable centerlines;
    - cable depth is kept only around those centerlines;
    - a connected-component slenderness fallback handles weak Hough detections.
    """

    def __init__(self) -> None:
        super().__init__("cable_depth_filter")

        self.declare_parameter("depth_topic", "/foundation_stereo/depth")
        self.declare_parameter("camera_info_topic", "/camera/infra1/camera_info")
        self.declare_parameter("output_depth_topic", "/perception/cable_depth")
        self.declare_parameter("output_camera_info_topic", "/perception/cable_depth/camera_info")
        self.declare_parameter("output_mask_topic", "/perception/cable_mask")
        self.declare_parameter("output_encoding", "32FC1")

        self.declare_parameter("z_min", 0.15)
        self.declare_parameter("z_max", 2.0)
        self.declare_parameter("median_blur_ksize", 3)
        self.declare_parameter("morph_close_px", 0)

        self.declare_parameter("remove_thick_regions", True)
        self.declare_parameter("max_cable_half_width_px", 6.0)
        self.declare_parameter("thick_region_dilate_px", 8)

        self.declare_parameter("min_line_length_px", 80)
        self.declare_parameter("max_line_gap_px", 35)
        self.declare_parameter("hough_threshold", 20)
        self.declare_parameter("line_thickness_px", 5)
        self.declare_parameter("line_mask_dilate_px", 2)
        self.declare_parameter("fill_depth_gaps", True)
        self.declare_parameter("depth_gap_close_px", 4)
        self.declare_parameter("max_depth_fill_distance_px", 12)

        self.declare_parameter("use_component_fallback", True)
        self.declare_parameter("component_min_area_px", 20)
        self.declare_parameter("component_max_area_px", 5000)
        self.declare_parameter("component_min_aspect", 3.0)
        self.declare_parameter("component_max_minor_px", 22)

        self.declare_parameter("remove_large_blobs", True)
        self.declare_parameter("large_blob_min_area_px", 3000)
        self.declare_parameter("large_blob_max_aspect", 1.8)
        self.declare_parameter("large_blob_min_minor_px", 28)

        self.declare_parameter("debug_log_period", 2.0)

        self.bridge = CvBridge()
        self.last_camera_info: Optional[CameraInfo] = None
        self.last_log_time = self.get_clock().now()

        self.depth_pub = self.create_publisher(
            Image,
            str(self.get_parameter("output_depth_topic").value),
            _sensor_qos(),
        )
        self.info_pub = self.create_publisher(
            CameraInfo,
            str(self.get_parameter("output_camera_info_topic").value),
            _sensor_qos(),
        )
        self.mask_pub = self.create_publisher(
            Image,
            str(self.get_parameter("output_mask_topic").value),
            _sensor_qos(),
        )

        self.create_subscription(
            Image,
            str(self.get_parameter("depth_topic").value),
            self._depth_cb,
            _sensor_qos(),
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._camera_info_cb,
            _sensor_qos(),
        )

        self.get_logger().info(
            "Cable depth filter: "
            f"{self.get_parameter('depth_topic').value} -> "
            f"{self.get_parameter('output_depth_topic').value}, "
            f"mask={self.get_parameter('output_mask_topic').value}"
        )

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self.last_camera_info = msg

    def _depth_cb(self, msg: Image) -> None:
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().warn(f"Failed to convert depth image: {exc}")
            return

        depth_m = self._to_depth_meters(depth_raw, msg.encoding)
        valid_mask = self._valid_depth_mask(depth_m)
        cable_mask, fill_candidate_mask = self._extract_cable_masks(valid_mask)

        cable_depth, output_mask = self._build_cable_depth(depth_m, cable_mask, fill_candidate_mask)

        output_encoding = str(self.get_parameter("output_encoding").value).strip()
        if output_encoding == "16UC1":
            depth_out = np.clip(cable_depth * 1000.0, 0.0, 65535.0).astype(np.uint16)
            depth_msg = self.bridge.cv2_to_imgmsg(depth_out, encoding="16UC1")
        else:
            depth_msg = self.bridge.cv2_to_imgmsg(cable_depth.astype(np.float32), encoding="32FC1")
        depth_msg.header = msg.header
        self.depth_pub.publish(depth_msg)

        mask_msg = self.bridge.cv2_to_imgmsg(output_mask.astype(np.uint8), encoding="mono8")
        mask_msg.header = msg.header
        self.mask_pub.publish(mask_msg)

        if self.last_camera_info is not None:
            info = CameraInfo()
            info.header = msg.header
            info.height = self.last_camera_info.height
            info.width = self.last_camera_info.width
            info.distortion_model = self.last_camera_info.distortion_model
            info.d = list(self.last_camera_info.d)
            info.k = list(self.last_camera_info.k)
            info.r = list(self.last_camera_info.r)
            info.p = list(self.last_camera_info.p)
            info.binning_x = self.last_camera_info.binning_x
            info.binning_y = self.last_camera_info.binning_y
            info.roi = self.last_camera_info.roi
            self.info_pub.publish(info)

        self._maybe_log(valid_mask, output_mask)

    def _to_depth_meters(self, depth_raw: np.ndarray, encoding: str) -> np.ndarray:
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]
        if encoding.upper() == "16UC1" or depth_raw.dtype == np.uint16:
            return depth_raw.astype(np.float32) * 0.001
        return depth_raw.astype(np.float32)

    def _valid_depth_mask(self, depth_m: np.ndarray) -> np.ndarray:
        z_min = float(self.get_parameter("z_min").value)
        z_max = float(self.get_parameter("z_max").value)
        mask = np.isfinite(depth_m) & (depth_m >= z_min) & (depth_m <= z_max)
        mask_u8 = (mask.astype(np.uint8) * 255)

        median_ksize = int(self.get_parameter("median_blur_ksize").value)
        if median_ksize >= 3:
            if median_ksize % 2 == 0:
                median_ksize += 1
            mask_u8 = cv2.medianBlur(mask_u8, median_ksize)

        close_px = int(self.get_parameter("morph_close_px").value)
        if close_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        return mask_u8

    def _extract_cable_masks(self, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        work_mask = valid_mask.copy()
        if bool(self.get_parameter("remove_thick_regions").value):
            work_mask = self._remove_thick_regions(work_mask)
        if bool(self.get_parameter("remove_large_blobs").value):
            work_mask = self._remove_large_blobs(work_mask)

        candidate_mask = np.zeros_like(valid_mask, dtype=np.uint8)
        lines = cv2.HoughLinesP(
            work_mask,
            rho=1,
            theta=np.pi / 180.0,
            threshold=int(self.get_parameter("hough_threshold").value),
            minLineLength=int(self.get_parameter("min_line_length_px").value),
            maxLineGap=int(self.get_parameter("max_line_gap_px").value),
        )
        if lines is not None:
            thickness = max(1, int(self.get_parameter("line_thickness_px").value))
            for line in lines.reshape(-1, 4):
                x1, y1, x2, y2 = [int(v) for v in line]
                if not self._is_reasonable_line(x1, y1, x2, y2):
                    continue
                cv2.line(candidate_mask, (x1, y1), (x2, y2), 255, thickness=thickness)

        if bool(self.get_parameter("use_component_fallback").value):
            candidate_mask = cv2.bitwise_or(candidate_mask, self._slender_component_mask(work_mask))

        dilate_px = int(self.get_parameter("line_mask_dilate_px").value)
        if dilate_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
            candidate_mask = cv2.dilate(candidate_mask, kernel)

        cable_mask = cv2.bitwise_and(candidate_mask, valid_mask)
        return cable_mask, candidate_mask

    def _build_cable_depth(
        self,
        depth_m: np.ndarray,
        cable_mask: np.ndarray,
        fill_candidate_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cable_depth = np.zeros(depth_m.shape, dtype=np.float32)
        cable_depth[cable_mask > 0] = depth_m[cable_mask > 0]

        if not bool(self.get_parameter("fill_depth_gaps").value):
            return cable_depth, cable_mask

        fill_mask = fill_candidate_mask.copy()
        close_px = int(self.get_parameter("depth_gap_close_px").value)
        if close_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
            fill_mask = cv2.morphologyEx(fill_mask, cv2.MORPH_CLOSE, kernel)

        max_fill_px = max(0, int(self.get_parameter("max_depth_fill_distance_px").value))
        if max_fill_px <= 0 or int(np.count_nonzero(cable_mask)) == 0:
            return cable_depth, cable_mask

        filled_depth = cable_depth.copy()
        known = cable_mask > 0
        target = fill_mask > 0
        kernel = np.ones((3, 3), dtype=np.float32)
        for _ in range(max_fill_px):
            known_f = known.astype(np.float32)
            depth_sum = cv2.filter2D(filled_depth, -1, kernel, borderType=cv2.BORDER_CONSTANT)
            depth_count = cv2.filter2D(known_f, -1, kernel, borderType=cv2.BORDER_CONSTANT)
            new_pixels = target & (~known) & (depth_count > 0.0)
            if not np.any(new_pixels):
                break
            filled_depth[new_pixels] = depth_sum[new_pixels] / depth_count[new_pixels]
            known[new_pixels] = True

        output_mask = (known.astype(np.uint8) * 255)
        return filled_depth, output_mask

    def _remove_thick_regions(self, mask: np.ndarray) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)
        if int(np.count_nonzero(binary)) == 0:
            return mask
        max_half_width = max(0.0, float(self.get_parameter("max_cable_half_width_px").value))
        if max_half_width <= 0.0:
            return mask
        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        thick_core = ((distance > max_half_width).astype(np.uint8) * 255)
        dilate_px = int(self.get_parameter("thick_region_dilate_px").value)
        if dilate_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * dilate_px + 1, 2 * dilate_px + 1),
            )
            thick_core = cv2.dilate(thick_core, kernel)
        return cv2.bitwise_and(mask, cv2.bitwise_not(thick_core))

    def _remove_large_blobs(self, mask: np.ndarray) -> np.ndarray:
        output = np.zeros_like(mask, dtype=np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        min_area = int(self.get_parameter("large_blob_min_area_px").value)
        max_aspect = float(self.get_parameter("large_blob_max_aspect").value)
        min_minor = int(self.get_parameter("large_blob_min_minor_px").value)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            minor = min(w, h)
            aspect = max(w, h) / max(1.0, float(minor))
            is_large_blob = area >= min_area and minor >= min_minor and aspect <= max_aspect
            if not is_large_blob:
                output[labels == label] = 255
        return output

    def _slender_component_mask(self, mask: np.ndarray) -> np.ndarray:
        output = np.zeros_like(mask, dtype=np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        min_area = int(self.get_parameter("component_min_area_px").value)
        max_area = int(self.get_parameter("component_max_area_px").value)
        min_aspect = float(self.get_parameter("component_min_aspect").value)
        max_minor = int(self.get_parameter("component_max_minor_px").value)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            minor = min(w, h)
            aspect = max(w, h) / max(1.0, float(minor))
            if aspect >= min_aspect or minor <= max_minor:
                output[labels == label] = 255
        return output

    def _is_reasonable_line(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        length = math.hypot(float(x2 - x1), float(y2 - y1))
        return length >= float(self.get_parameter("min_line_length_px").value)

    def _maybe_log(self, valid_mask: np.ndarray, cable_mask: np.ndarray) -> None:
        period = float(self.get_parameter("debug_log_period").value)
        if period <= 0.0:
            return
        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds * 1e-9 < period:
            return
        self.last_log_time = now
        valid_px = int(np.count_nonzero(valid_mask))
        cable_px = int(np.count_nonzero(cable_mask))
        ratio = 0.0 if valid_px == 0 else cable_px / valid_px
        self.get_logger().info(
            f"cable depth filter: valid_px={valid_px}, cable_px={cable_px}, keep_ratio={ratio:.3f}"
        )


def main() -> None:
    rclpy.init()
    node = CableDepthFilterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
