#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from nvblox_msgs.msg import Mesh, VoxelBlockLayer
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray


def _sensor_qos(depth: int = 5) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def _marker_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return vec
    return vec / norm


def _quat_from_z_axis(direction: np.ndarray) -> tuple[float, float, float, float]:
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    direction = _normalize(direction.astype(float))
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    if dot > 0.999999:
        return 0.0, 0.0, 0.0, 1.0
    if dot < -0.999999:
        return 1.0, 0.0, 0.0, 0.0
    axis = _normalize(np.cross(z_axis, direction))
    angle = math.acos(dot)
    half = 0.5 * angle
    sin_half = math.sin(half)
    return (
        float(axis[0] * sin_half),
        float(axis[1] * sin_half),
        float(axis[2] * sin_half),
        float(math.cos(half)),
    )


class CableCapsuleMarkerNode(Node):
    def __init__(self) -> None:
        super().__init__("cable_capsule_marker")

        self.declare_parameter("source_mode", "voxel_layer")
        self.declare_parameter("depth_topic", "/perception/cable_depth")
        self.declare_parameter("camera_info_topic", "/perception/cable_depth/camera_info")
        self.declare_parameter("pointcloud_topic", "/nvblox_node/static_esdf_pointcloud")
        self.declare_parameter("mesh_topic", "/nvblox_node/mesh")
        self.declare_parameter("voxel_layer_topic", "/nvblox_node/color_layer")
        self.declare_parameter("voxel_centers_are_global", True)
        self.declare_parameter("marker_topic", "/perception/cable_capsules")
        self.declare_parameter("radius_m", 0.02)
        self.declare_parameter("alpha", 0.45)
        self.declare_parameter("color_r", 0.2)
        self.declare_parameter("color_g", 0.55)
        self.declare_parameter("color_b", 1.0)
        self.declare_parameter("min_line_length_px", 70)
        self.declare_parameter("max_line_gap_px", 45)
        self.declare_parameter("hough_threshold", 18)
        self.declare_parameter("sample_band_px", 5)
        self.declare_parameter("min_points_per_capsule", 20)
        self.declare_parameter("min_capsule_length_m", 0.08)
        self.declare_parameter("max_capsules", 8)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("depth_max_m", 3.0)
        self.declare_parameter("duplicate_center_distance_m", 0.08)
        self.declare_parameter("duplicate_direction_dot", 0.92)
        self.declare_parameter("point_voxel_size_m", 0.015)
        self.declare_parameter("mesh_vertices_are_global", True)
        self.declare_parameter("max_fit_points", 6000)
        self.declare_parameter("line_endpoint_percentile", 0.5)
        self.declare_parameter("mesh_direct_fit_enabled", True)
        self.declare_parameter("mesh_cluster_voxel_size_m", 0.03)
        self.declare_parameter("mesh_cluster_neighbor_voxels", 2)
        self.declare_parameter("mesh_cluster_min_points", 10)
        self.declare_parameter("mesh_cluster_min_length_m", 0.12)
        self.declare_parameter("mesh_cluster_max_radius_m", 0.10)
        self.declare_parameter("mesh_cluster_endpoint_percentile", 0.0)
        self.declare_parameter("component_fit_enabled", True)
        self.declare_parameter("component_min_points", 6)
        self.declare_parameter("component_min_length_m", 0.03)
        self.declare_parameter("component_max_radius_m", 0.055)
        self.declare_parameter("component_neighbor_voxels", 1)
        self.declare_parameter("component_merge_enabled", True)
        self.declare_parameter("component_merge_gap_m", 0.18)
        self.declare_parameter("component_merge_perp_m", 0.045)
        self.declare_parameter("component_merge_direction_dot", 0.90)
        self.declare_parameter("capsule_ema_alpha", 0.18)
        self.declare_parameter("capsule_match_distance_m", 0.12)
        self.declare_parameter("capsule_hold_cycles", 8)
        self.declare_parameter("ransac_fallback_enabled", False)
        self.declare_parameter("ransac_iterations", 180)
        self.declare_parameter("ransac_inlier_distance_m", 0.035)
        self.declare_parameter("ransac_min_inliers", 25)
        self.declare_parameter("ransac_remove_distance_m", 0.055)
        self.declare_parameter("voxel_fit_mode", "seeded_axis")
        self.declare_parameter("seed_reference_x", 0.0)
        self.declare_parameter("seed_reference_y", 0.0)
        self.declare_parameter("seed_reference_z", 0.0)
        self.declare_parameter("left_seed_reference_frame", "openarmx_left_link1")
        self.declare_parameter("right_seed_reference_frame", "openarmx_right_link1")
        self.declare_parameter("left_seed_reference_x", 0.0)
        self.declare_parameter("left_seed_reference_y", 0.089)
        self.declare_parameter("left_seed_reference_z", 0.698)
        self.declare_parameter("right_seed_reference_x", 0.0)
        self.declare_parameter("right_seed_reference_y", -0.089)
        self.declare_parameter("right_seed_reference_z", 0.698)
        self.declare_parameter("seed_axis_x", 1.0)
        self.declare_parameter("seed_axis_y", 0.0)
        self.declare_parameter("seed_axis_z", 0.0)
        self.declare_parameter("seed_axis_neighbor_radius_m", 0.075)
        self.declare_parameter("seed_axis_bin_size_m", 0.03)
        self.declare_parameter("seed_axis_max_gap_m", 0.15)
        self.declare_parameter("seed_axis_max_half_length_m", 0.2)
        self.declare_parameter("seed_axis_output_half_length_m", 0.6)
        self.declare_parameter("seed_axis_min_points", 6)
        self.declare_parameter("seed_axis_endpoint_padding_m", 0.03)
        self.declare_parameter("seed_min_neighbor_voxels", 3)
        self.declare_parameter("seed_neighbor_radius_m", 0.08)
        self.declare_parameter("seed_search_max_candidates", 128)

        self.bridge = CvBridge()
        self.source_mode = str(self.get_parameter("source_mode").value).strip().lower()
        self.camera_info: Optional[CameraInfo] = None
        self.latest_depth_msg: Optional[Image] = None
        self.latest_pointcloud_msg: Optional[PointCloud2] = None
        self.latest_voxel_layer_msg: Optional[VoxelBlockLayer] = None
        self.mesh_blocks: dict[tuple[int, int, int], np.ndarray] = {}
        self.mesh_header = None
        self.voxel_blocks: dict[tuple[int, int, int], np.ndarray] = {}
        self.voxel_header = None
        self.rng = np.random.default_rng(7)
        self.tracked_capsules: list[dict[str, object]] = []
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.marker_pub = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("marker_topic").value),
            _marker_qos(),
        )
        if self.source_mode == "depth":
            self.create_subscription(
                CameraInfo,
                str(self.get_parameter("camera_info_topic").value),
                self._camera_info_cb,
                _sensor_qos(),
            )
            self.create_subscription(
                Image,
                str(self.get_parameter("depth_topic").value),
                self._depth_cb,
                _sensor_qos(),
            )
        elif self.source_mode == "pointcloud":
            self.create_subscription(
                PointCloud2,
                str(self.get_parameter("pointcloud_topic").value),
                self._pointcloud_cb,
                _sensor_qos(),
            )
        elif self.source_mode == "mesh":
            self.create_subscription(
                Mesh,
                str(self.get_parameter("mesh_topic").value),
                self._mesh_cb,
                _sensor_qos(),
            )
        elif self.source_mode == "voxel_layer":
            self.create_subscription(
                VoxelBlockLayer,
                str(self.get_parameter("voxel_layer_topic").value),
                self._voxel_layer_cb,
                _sensor_qos(),
            )
        else:
            raise ValueError("source_mode must be one of: depth, pointcloud, mesh, voxel_layer")

        rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.timer = self.create_timer(1.0 / rate_hz, self._timer_cb)
        self.get_logger().info(
            f"Cable capsule marker: source_mode={self.source_mode}, "
            f"markers={self.get_parameter('marker_topic').value}"
        )

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def _depth_cb(self, msg: Image) -> None:
        self.latest_depth_msg = msg

    def _pointcloud_cb(self, msg: PointCloud2) -> None:
        self.latest_pointcloud_msg = msg

    def _mesh_cb(self, msg: Mesh) -> None:
        if msg.clear:
            self.mesh_blocks.clear()
        block_size = float(msg.block_size_m)
        vertices_are_global = bool(self.get_parameter("mesh_vertices_are_global").value)
        for index, block in zip(msg.block_indices, msg.blocks):
            key = (int(index.x), int(index.y), int(index.z))
            origin = np.array([index.x * block_size, index.y * block_size, index.z * block_size], dtype=np.float64)
            if not block.vertices:
                self.mesh_blocks.pop(key, None)
                continue
            local = np.asarray([[v.x, v.y, v.z] for v in block.vertices], dtype=np.float64)
            self.mesh_blocks[key] = local if vertices_are_global else local + origin
        self.mesh_header = msg.header

    def _voxel_layer_cb(self, msg: VoxelBlockLayer) -> None:
        self.latest_voxel_layer_msg = msg
        if msg.clear:
            self.voxel_blocks.clear()
        block_size = float(msg.block_size_m)
        centers_are_global = bool(self.get_parameter("voxel_centers_are_global").value)
        for index, block in zip(msg.block_indices, msg.blocks):
            key = (int(index.x), int(index.y), int(index.z))
            if not block.centers:
                self.voxel_blocks.pop(key, None)
                continue
            origin = np.array([index.x * block_size, index.y * block_size, index.z * block_size], dtype=np.float64)
            local = np.asarray([[p.x, p.y, p.z] for p in block.centers], dtype=np.float64)
            self.voxel_blocks[key] = local if centers_are_global else local + origin
        self.voxel_header = msg.header

    def _timer_cb(self) -> None:
        if self.source_mode == "depth":
            if self.camera_info is None or self.latest_depth_msg is None:
                return

            try:
                depth_raw = self.bridge.imgmsg_to_cv2(self.latest_depth_msg, desired_encoding="passthrough")
            except Exception as exc:
                self.get_logger().warn(f"Failed to convert cable depth image: {exc}")
                return

            depth_m = self._to_depth_meters(depth_raw, self.latest_depth_msg.encoding)
            capsules = self._fit_capsules(depth_m)
            self.marker_pub.publish(self._markers_from_capsules(capsules, self.latest_depth_msg.header))
            return

        points_and_header = self._points_from_map_source()
        if points_and_header is None:
            return
        points, header = points_and_header
        capsules = self._fit_capsules_from_points(points, str(header.frame_id))
        capsules = self._smooth_capsules(capsules)
        self.marker_pub.publish(self._markers_from_capsules(capsules, header))

    def _points_from_map_source(self) -> Optional[tuple[np.ndarray, object]]:
        if self.source_mode == "pointcloud":
            if self.latest_pointcloud_msg is None:
                return None
            return self._points_from_pointcloud2(self.latest_pointcloud_msg), self.latest_pointcloud_msg.header
        if self.source_mode == "mesh":
            if not self.mesh_blocks or self.mesh_header is None:
                return None
            return np.concatenate(list(self.mesh_blocks.values()), axis=0), self.mesh_header
        if self.source_mode == "voxel_layer":
            if not self.voxel_blocks or self.voxel_header is None:
                return None
            return np.concatenate(list(self.voxel_blocks.values()), axis=0), self.voxel_header
        return None

    def _points_from_pointcloud2(self, msg: PointCloud2) -> np.ndarray:
        points = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        if isinstance(points, np.ndarray):
            if points.dtype.names:
                return np.stack((points["x"], points["y"], points["z"]), axis=1).astype(np.float64)
            return np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return np.asarray([[p[0], p[1], p[2]] for p in points], dtype=np.float64)

    def _points_from_voxel_layer(self, msg: VoxelBlockLayer) -> np.ndarray:
        blocks: list[np.ndarray] = []
        block_size = float(msg.block_size_m)
        for index, block in zip(msg.block_indices, msg.blocks):
            if not block.centers:
                continue
            origin = np.array([index.x * block_size, index.y * block_size, index.z * block_size], dtype=np.float64)
            local = np.asarray([[p.x, p.y, p.z] for p in block.centers], dtype=np.float64)
            blocks.append(local + origin)
        if not blocks:
            return np.empty((0, 3), dtype=np.float64)
        return np.concatenate(blocks, axis=0)

    def _to_depth_meters(self, depth_raw: np.ndarray, encoding: str) -> np.ndarray:
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]
        if encoding.upper() == "16UC1" or depth_raw.dtype == np.uint16:
            return depth_raw.astype(np.float32) * 0.001
        return depth_raw.astype(np.float32)

    def _fit_capsules(self, depth_m: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        depth_max = float(self.get_parameter("depth_max_m").value)
        valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= depth_max)
        mask = (valid.astype(np.uint8) * 255)
        if int(np.count_nonzero(mask)) == 0:
            return []

        lines = cv2.HoughLinesP(
            mask,
            rho=1,
            theta=np.pi / 180.0,
            threshold=int(self.get_parameter("hough_threshold").value),
            minLineLength=int(self.get_parameter("min_line_length_px").value),
            maxLineGap=int(self.get_parameter("max_line_gap_px").value),
        )
        if lines is None:
            return []

        candidates: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        band_px = max(1, int(self.get_parameter("sample_band_px").value))
        min_points = int(self.get_parameter("min_points_per_capsule").value)
        min_length = float(self.get_parameter("min_capsule_length_m").value)
        for line in lines.reshape(-1, 4):
            line_mask = np.zeros(mask.shape, dtype=np.uint8)
            x1, y1, x2, y2 = [int(v) for v in line]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=2 * band_px + 1)
            ys, xs = np.where((line_mask > 0) & valid)
            if len(xs) < min_points:
                continue
            points = self._unproject_points(xs, ys, depth_m[ys, xs])
            if points.shape[0] < min_points:
                continue
            start, end, direction = self._line_segment_from_points(points)
            length = float(np.linalg.norm(end - start))
            if length < min_length:
                continue
            candidates.append((length, start, end, direction))

        candidates.sort(key=lambda item: item[0], reverse=True)
        capsules: list[tuple[np.ndarray, np.ndarray]] = []
        accepted: list[tuple[np.ndarray, np.ndarray]] = []
        max_capsules = int(self.get_parameter("max_capsules").value)
        duplicate_dist = float(self.get_parameter("duplicate_center_distance_m").value)
        duplicate_dot = float(self.get_parameter("duplicate_direction_dot").value)
        for _, start, end, direction in candidates:
            center = 0.5 * (start + end)
            duplicate = False
            for old_center, old_direction in accepted:
                if np.linalg.norm(center - old_center) < duplicate_dist:
                    if abs(float(np.dot(direction, old_direction))) > duplicate_dot:
                        duplicate = True
                        break
            if duplicate:
                continue
            capsules.append((start, end))
            accepted.append((center, direction))
            if len(capsules) >= max_capsules:
                break
        return capsules

    def _unproject_points(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
        assert self.camera_info is not None
        k = self.camera_info.k
        fx = float(k[0])
        fy = float(k[4])
        cx = float(k[2])
        cy = float(k[5])
        zs = zs.astype(np.float64)
        x = (xs.astype(np.float64) - cx) * zs / fx
        y = (ys.astype(np.float64) - cy) * zs / fy
        return np.stack((x, y, zs), axis=1)

    def _line_segment_from_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        endpoint_percentile = float(self.get_parameter("line_endpoint_percentile").value)
        return self._line_segment_from_points_with_percentile(points, endpoint_percentile)

    def _line_segment_from_points_with_percentile(
        self,
        points: np.ndarray,
        endpoint_percentile: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        center = np.mean(points, axis=0)
        centered = points - center
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = _normalize(vh[0])
        projection = centered @ direction
        endpoint_percentile = max(0.0, min(10.0, endpoint_percentile))
        if endpoint_percentile <= 0.0:
            low = float(np.min(projection))
            high = float(np.max(projection))
        else:
            low = float(np.percentile(projection, endpoint_percentile))
            high = float(np.percentile(projection, 100.0 - endpoint_percentile))
        start = center + low * direction
        end = center + high * direction
        return start, end, direction

    def _fit_capsules_from_points(
        self,
        points: np.ndarray,
        points_frame: str = "",
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        points = self._clean_and_downsample_points(points)
        min_points = min(
            int(self.get_parameter("ransac_min_inliers").value),
            int(self.get_parameter("component_min_points").value),
        )
        if points.shape[0] < min_points:
            return []

        if self.source_mode == "voxel_layer":
            voxel_fit_mode = str(self.get_parameter("voxel_fit_mode").value).strip().lower()
            if voxel_fit_mode == "seeded_axis":
                references = self._seed_reference_points(points_frame)
                return self._fit_capsules_from_seeded_axis(points, references)
            if voxel_fit_mode not in ("components", "cluster"):
                self.get_logger().warn(
                    f"Unknown voxel_fit_mode={voxel_fit_mode!r}; using component clusters.",
                    throttle_duration_sec=2.0,
                )

        if self.source_mode in ("mesh", "voxel_layer") and bool(self.get_parameter("mesh_direct_fit_enabled").value):
            return self._fit_capsules_from_mesh_clusters(points)

        component_capsules: list[tuple[np.ndarray, np.ndarray]] = []
        if bool(self.get_parameter("component_fit_enabled").value):
            component_capsules = self._fit_capsules_from_components(points)
            if not bool(self.get_parameter("ransac_fallback_enabled").value):
                return component_capsules
            if len(component_capsules) >= int(self.get_parameter("max_capsules").value):
                return component_capsules

        capsules: list[tuple[np.ndarray, np.ndarray]] = []
        max_capsules = int(self.get_parameter("max_capsules").value)
        min_length = float(self.get_parameter("min_capsule_length_m").value)
        inlier_dist = float(self.get_parameter("ransac_inlier_distance_m").value)
        remove_dist = float(self.get_parameter("ransac_remove_distance_m").value)
        iterations = int(self.get_parameter("ransac_iterations").value)
        remaining = points.copy()

        for _ in range(max_capsules):
            if remaining.shape[0] < min_points:
                break
            best_inliers: np.ndarray | None = None
            best_score = -1.0
            for _ in range(iterations):
                sample_indices = self.rng.choice(remaining.shape[0], size=2, replace=False)
                p0 = remaining[sample_indices[0]]
                p1 = remaining[sample_indices[1]]
                direction = p1 - p0
                length = float(np.linalg.norm(direction))
                if length < min_length:
                    continue
                direction = direction / length
                distances = np.linalg.norm(np.cross(remaining - p0, direction), axis=1)
                inliers = distances <= inlier_dist
                inlier_count = int(np.count_nonzero(inliers))
                if inlier_count < min_points:
                    continue
                candidate_points = remaining[inliers]
                start, end, _ = self._line_segment_from_points(candidate_points)
                candidate_length = float(np.linalg.norm(end - start))
                if candidate_length < min_length:
                    continue
                score = float(inlier_count) * candidate_length
                if score > best_score:
                    best_score = score
                    best_inliers = inliers

            if best_inliers is None:
                break
            inlier_points = remaining[best_inliers]
            start, end, direction = self._line_segment_from_points(inlier_points)
            if self._is_duplicate_capsule(capsules, start, end, direction):
                remaining = remaining[~best_inliers]
                continue
            capsules.append((start, end))

            distances = np.linalg.norm(np.cross(remaining - start, direction), axis=1)
            projection = (remaining - start) @ direction
            length = float(np.linalg.norm(end - start))
            on_segment = (projection >= -remove_dist) & (projection <= length + remove_dist)
            remaining = remaining[~((distances <= remove_dist) & on_segment)]

        combined = list(component_capsules)
        for start, end in capsules:
            direction = _normalize(end - start)
            if not self._is_duplicate_capsule(combined, start, end, direction):
                combined.append((start, end))
            if len(combined) >= max_capsules:
                break
        return combined

    def _fit_capsules_from_seeded_axis(
        self,
        points: np.ndarray,
        references: list[np.ndarray] | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if not references:
            references = [
                np.array(
                    [
                        float(self.get_parameter("seed_reference_x").value),
                        float(self.get_parameter("seed_reference_y").value),
                        float(self.get_parameter("seed_reference_z").value),
                    ],
                    dtype=np.float64,
                )
            ]
        axis = np.array(
            [
                float(self.get_parameter("seed_axis_x").value),
                float(self.get_parameter("seed_axis_y").value),
                float(self.get_parameter("seed_axis_z").value),
            ],
            dtype=np.float64,
        )
        axis_norm = float(np.linalg.norm(axis))
        if not np.isfinite(axis_norm) or axis_norm < 1e-9:
            self.get_logger().warn(
                "seed_axis has zero length; using world +X.",
                throttle_duration_sec=2.0,
            )
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis /= axis_norm

        neighbor_radius = max(
            1e-3,
            float(self.get_parameter("seed_axis_neighbor_radius_m").value),
        )
        bin_size = max(1e-3, float(self.get_parameter("seed_axis_bin_size_m").value))
        max_gap = max(bin_size, float(self.get_parameter("seed_axis_max_gap_m").value))
        max_half_length = max(
            bin_size,
            float(self.get_parameter("seed_axis_max_half_length_m").value),
        )
        min_length = float(self.get_parameter("min_capsule_length_m").value)
        output_half_length = max(
            0.5 * min_length,
            float(self.get_parameter("seed_axis_output_half_length_m").value),
        )
        min_points = max(2, int(self.get_parameter("seed_axis_min_points").value))
        endpoint_padding = max(
            0.0,
            float(self.get_parameter("seed_axis_endpoint_padding_m").value),
        )
        max_capsules = max(1, int(self.get_parameter("max_capsules").value))

        remaining = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        capsules: list[tuple[np.ndarray, np.ndarray]] = []
        max_rejected_seeds = min(200, max(20, remaining.shape[0]))

        for reference in references:
            if remaining.shape[0] < min_points or len(capsules) >= max_capsules:
                break
            rejected_seeds = 0
            while remaining.shape[0] >= min_points and rejected_seeds < max_rejected_seeds:
                seed_index = self._nearest_supported_seed_index(remaining, reference)
                if seed_index < 0:
                    self.get_logger().warn(
                        "No cable seed has enough neighboring voxels; skipping this arm reference.",
                        throttle_duration_sec=2.0,
                    )
                    break
                seed = remaining[seed_index]
                offsets = remaining - seed
                axial = offsets @ axis
                radial_vectors = offsets - axial[:, None] * axis[None, :]
                radial = np.linalg.norm(radial_vectors, axis=1)
                tube_mask = (radial <= neighbor_radius) & (np.abs(axial) <= max_half_length)

                tube_indices = np.flatnonzero(tube_mask)
                if tube_indices.size < min_points:
                    remaining = np.delete(remaining, seed_index, axis=0)
                    rejected_seeds += 1
                    continue

                tube_axial = axial[tube_indices]
                bins = np.floor(tube_axial / bin_size).astype(np.int64)
                occupied_bins = np.unique(bins)
                seed_bin = 0
                low_bin = self._grow_seeded_axis_bins(
                    occupied_bins,
                    seed_bin,
                    -1,
                    bin_size,
                    max_gap,
                )
                high_bin = self._grow_seeded_axis_bins(
                    occupied_bins,
                    seed_bin,
                    1,
                    bin_size,
                    max_gap,
                )
                contiguous_mask = (bins >= low_bin) & (bins <= high_bin)
                selected_indices = tube_indices[contiguous_mask]
                if selected_indices.size < min_points:
                    remaining = np.delete(remaining, seed_index, axis=0)
                    rejected_seeds += 1
                    continue

                selected = remaining[selected_indices]
                selected_axial = (selected - seed) @ axis
                detected_low = max(
                    -max_half_length,
                    float(np.min(selected_axial)) - endpoint_padding,
                )
                detected_high = min(
                    max_half_length,
                    float(np.max(selected_axial)) + endpoint_padding,
                )
                if detected_high - detected_low < min_length:
                    remaining = np.delete(remaining, seed_index, axis=0)
                    rejected_seeds += 1
                    continue

                selected_axis_coordinates = selected @ axis
                selected_perpendicular = selected - selected_axis_coordinates[:, None] * axis[None, :]
                center_perpendicular = np.mean(selected_perpendicular, axis=0)
                seed_axis_coordinate = float(np.dot(seed, axis))
                center_axis_point = center_perpendicular + seed_axis_coordinate * axis
                detected_center = 0.5 * (detected_low + detected_high)
                start = center_axis_point + (detected_center - output_half_length) * axis
                end = center_axis_point + (detected_center + output_half_length) * axis

                if not self._is_duplicate_capsule(capsules, start, end, axis):
                    capsules.append((start, end))

                remove_low = detected_low - max_gap
                remove_high = detected_high + max_gap
                remove_mask = tube_mask & (axial >= remove_low) & (axial <= remove_high)
                if not np.any(remove_mask):
                    remove_mask[seed_index] = True
                remaining = remaining[~remove_mask]
                break

        return capsules

    def _nearest_supported_seed_index(
        self,
        points: np.ndarray,
        reference: np.ndarray,
    ) -> int:
        if points.shape[0] == 0:
            return -1
        min_neighbors = max(0, int(self.get_parameter("seed_min_neighbor_voxels").value))
        if min_neighbors == 0:
            return int(np.argmin(np.linalg.norm(points - reference, axis=1)))
        neighbor_radius = max(
            1e-4,
            float(self.get_parameter("seed_neighbor_radius_m").value),
        )
        max_candidates = max(
            1,
            int(self.get_parameter("seed_search_max_candidates").value),
        )
        reference_distances = np.linalg.norm(points - reference, axis=1)
        candidate_indices = np.argsort(reference_distances)[:max_candidates]
        radius_squared = neighbor_radius * neighbor_radius
        for candidate_index in candidate_indices:
            delta = points - points[int(candidate_index)]
            squared_distances = np.einsum("ij,ij->i", delta, delta)
            neighbor_count = int(
                np.count_nonzero(
                    (squared_distances <= radius_squared)
                    & (squared_distances > 1e-12)
                )
            )
            if neighbor_count >= min_neighbors:
                return int(candidate_index)
        return -1

    def _seed_reference_points(self, points_frame: str) -> list[np.ndarray]:
        target_frame = points_frame.strip() or "world"
        references: list[np.ndarray] = []
        fallback_parameters = {
            "left": (
                "left_seed_reference_x",
                "left_seed_reference_y",
                "left_seed_reference_z",
            ),
            "right": (
                "right_seed_reference_x",
                "right_seed_reference_y",
                "right_seed_reference_z",
            ),
        }
        for side in ("left", "right"):
            source_frame = str(self.get_parameter(f"{side}_seed_reference_frame").value).strip()
            point = None
            if source_frame:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        target_frame,
                        source_frame,
                        Time(),
                        timeout=Duration(seconds=0.02),
                    )
                    point = np.array(
                        [
                            transform.transform.translation.x,
                            transform.transform.translation.y,
                            transform.transform.translation.z,
                        ],
                        dtype=np.float64,
                    )
                except Exception as exc:
                    self.get_logger().warn(
                        f"Seed TF unavailable {target_frame}<-{source_frame}: {exc}; "
                        "using URDF fallback.",
                        throttle_duration_sec=2.0,
                    )
            if point is None:
                names = fallback_parameters[side]
                point = np.array(
                    [float(self.get_parameter(name).value) for name in names],
                    dtype=np.float64,
                )
            references.append(point)
        return references

    @staticmethod
    def _grow_seeded_axis_bins(
        occupied_bins: np.ndarray,
        seed_bin: int,
        direction: int,
        bin_size: float,
        max_gap: float,
    ) -> int:
        if direction > 0:
            candidates = np.sort(occupied_bins[occupied_bins >= seed_bin])
        else:
            candidates = np.sort(occupied_bins[occupied_bins <= seed_bin])[::-1]
        last = seed_bin
        for candidate in candidates:
            candidate_int = int(candidate)
            if abs(candidate_int - last) * bin_size > max_gap:
                break
            last = candidate_int
        return last

    def _fit_capsules_from_components(self, points: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        voxel = float(self.get_parameter("point_voxel_size_m").value)
        if voxel <= 0.0 or points.shape[0] == 0:
            return []

        keys = np.floor(points / voxel).astype(np.int64)
        voxel_to_indices: dict[tuple[int, int, int], list[int]] = {}
        for idx, key in enumerate(keys):
            voxel_to_indices.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(idx)

        min_points = int(self.get_parameter("component_min_points").value)
        min_length = float(self.get_parameter("component_min_length_m").value)
        max_radius = float(self.get_parameter("component_max_radius_m").value)
        neighbor = max(1, int(self.get_parameter("component_neighbor_voxels").value))
        visited: set[tuple[int, int, int]] = set()
        components: list[np.ndarray] = []

        offsets = [
            (dx, dy, dz)
            for dx in range(-neighbor, neighbor + 1)
            for dy in range(-neighbor, neighbor + 1)
            for dz in range(-neighbor, neighbor + 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]
        for seed in voxel_to_indices.keys():
            if seed in visited:
                continue
            stack = [seed]
            visited.add(seed)
            indices: list[int] = []
            while stack:
                key = stack.pop()
                indices.extend(voxel_to_indices.get(key, []))
                for offset in offsets:
                    nxt = (key[0] + offset[0], key[1] + offset[1], key[2] + offset[2])
                    if nxt in visited or nxt not in voxel_to_indices:
                        continue
                    visited.add(nxt)
                    stack.append(nxt)
            if len(indices) >= min_points:
                components.append(points[np.asarray(indices, dtype=np.int64)])

        candidates: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for component in components:
            start, end, direction = self._line_segment_from_points(component)
            length = float(np.linalg.norm(end - start))
            if length < min_length:
                continue
            distances = np.linalg.norm(np.cross(component - start, direction), axis=1)
            radius95 = float(np.percentile(distances, 95.0)) if distances.size else 0.0
            if radius95 > max_radius:
                continue
            score = length * float(component.shape[0])
            candidates.append((score, start, end, direction, component))

        if bool(self.get_parameter("component_merge_enabled").value):
            candidates = self._merge_collinear_component_candidates(candidates)

        candidates.sort(key=lambda item: item[0], reverse=True)
        max_capsules = int(self.get_parameter("max_capsules").value)
        capsules: list[tuple[np.ndarray, np.ndarray]] = []
        for _, start, end, direction, _ in candidates:
            if self._is_duplicate_capsule(capsules, start, end, direction):
                continue
            capsules.append((start, end))
            if len(capsules) >= max_capsules:
                break
        return capsules

    def _fit_capsules_from_mesh_clusters(self, points: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        voxel = float(self.get_parameter("mesh_cluster_voxel_size_m").value)
        if voxel <= 0.0 or points.shape[0] == 0:
            return []

        keys = np.floor(points / voxel).astype(np.int64)
        voxel_to_indices: dict[tuple[int, int, int], list[int]] = {}
        for idx, key in enumerate(keys):
            voxel_to_indices.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(idx)

        neighbor = max(1, int(self.get_parameter("mesh_cluster_neighbor_voxels").value))
        min_points = int(self.get_parameter("mesh_cluster_min_points").value)
        min_length = float(self.get_parameter("mesh_cluster_min_length_m").value)
        max_radius = float(self.get_parameter("mesh_cluster_max_radius_m").value)
        endpoint_percentile = float(self.get_parameter("mesh_cluster_endpoint_percentile").value)
        endpoint_percentile = max(0.0, min(10.0, endpoint_percentile))
        offsets = [
            (dx, dy, dz)
            for dx in range(-neighbor, neighbor + 1)
            for dy in range(-neighbor, neighbor + 1)
            for dz in range(-neighbor, neighbor + 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

        visited: set[tuple[int, int, int]] = set()
        clusters: list[np.ndarray] = []
        for seed in voxel_to_indices.keys():
            if seed in visited:
                continue
            stack = [seed]
            visited.add(seed)
            indices: list[int] = []
            while stack:
                key = stack.pop()
                indices.extend(voxel_to_indices.get(key, []))
                for offset in offsets:
                    nxt = (key[0] + offset[0], key[1] + offset[1], key[2] + offset[2])
                    if nxt in visited or nxt not in voxel_to_indices:
                        continue
                    visited.add(nxt)
                    stack.append(nxt)
            if len(indices) >= min_points:
                clusters.append(points[np.asarray(indices, dtype=np.int64)])

        candidates: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        for cluster in clusters:
            start, end, direction = self._line_segment_from_points_with_percentile(
                cluster, endpoint_percentile
            )
            length = float(np.linalg.norm(end - start))
            if length < min_length:
                continue
            distances = np.linalg.norm(np.cross(cluster - start, direction), axis=1)
            radius95 = float(np.percentile(distances, 95.0)) if distances.size else 0.0
            if radius95 > max_radius:
                continue
            candidates.append((length * float(cluster.shape[0]), start, end, direction))

        candidates.sort(key=lambda item: item[0], reverse=True)
        max_capsules = int(self.get_parameter("max_capsules").value)
        capsules: list[tuple[np.ndarray, np.ndarray]] = []
        for _, start, end, direction in candidates:
            if self._is_duplicate_capsule(capsules, start, end, direction):
                continue
            capsules.append((start, end))
            if len(capsules) >= max_capsules:
                break
        return capsules

    def _merge_collinear_component_candidates(
        self,
        candidates: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        if len(candidates) <= 1:
            return candidates

        gap_limit = float(self.get_parameter("component_merge_gap_m").value)
        perp_limit = float(self.get_parameter("component_merge_perp_m").value)
        direction_dot = float(self.get_parameter("component_merge_direction_dot").value)
        n = len(candidates)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            _, s1, e1, d1, _ = candidates[i]
            for j in range(i + 1, n):
                _, s2, e2, d2, _ = candidates[j]
                if abs(float(np.dot(d1, d2))) < direction_dot:
                    continue
                endpoint_gap = min(
                    float(np.linalg.norm(a - b))
                    for a in (s1, e1)
                    for b in (s2, e2)
                )
                if endpoint_gap > gap_limit:
                    continue
                perp_12 = max(
                    float(np.linalg.norm(np.cross(s2 - s1, d1))),
                    float(np.linalg.norm(np.cross(e2 - s1, d1))),
                )
                perp_21 = max(
                    float(np.linalg.norm(np.cross(s1 - s2, d2))),
                    float(np.linalg.norm(np.cross(e1 - s2, d2))),
                )
                if min(perp_12, perp_21) <= perp_limit:
                    union(i, j)

        groups: dict[int, list[np.ndarray]] = {}
        for idx, candidate in enumerate(candidates):
            groups.setdefault(find(idx), []).append(candidate[4])

        merged: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        max_radius = float(self.get_parameter("component_max_radius_m").value)
        min_length = float(self.get_parameter("component_min_length_m").value)
        for parts in groups.values():
            points = np.concatenate(parts, axis=0)
            start, end, direction = self._line_segment_from_points(points)
            length = float(np.linalg.norm(end - start))
            if length < min_length:
                continue
            distances = np.linalg.norm(np.cross(points - start, direction), axis=1)
            radius95 = float(np.percentile(distances, 95.0)) if distances.size else 0.0
            if radius95 > max_radius:
                continue
            score = length * float(points.shape[0])
            merged.append((score, start, end, direction, points))
        return merged

    def _clean_and_downsample_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        finite = np.all(np.isfinite(points), axis=1)
        points = points[finite]
        depth_max = float(self.get_parameter("depth_max_m").value)
        if depth_max > 0.0:
            points = points[np.linalg.norm(points, axis=1) <= depth_max]
        if points.shape[0] == 0:
            return points

        voxel = float(self.get_parameter("point_voxel_size_m").value)
        if voxel > 0.0:
            keys = np.floor(points / voxel).astype(np.int64)
            _, unique_indices = np.unique(keys, axis=0, return_index=True)
            points = points[np.sort(unique_indices)]

        max_points = int(self.get_parameter("max_fit_points").value)
        if max_points > 0 and points.shape[0] > max_points:
            indices = self.rng.choice(points.shape[0], size=max_points, replace=False)
            points = points[indices]
        return points

    def _is_duplicate_capsule(
        self,
        capsules: list[tuple[np.ndarray, np.ndarray]],
        start: np.ndarray,
        end: np.ndarray,
        direction: np.ndarray,
    ) -> bool:
        center = 0.5 * (start + end)
        duplicate_dist = float(self.get_parameter("duplicate_center_distance_m").value)
        duplicate_dot = float(self.get_parameter("duplicate_direction_dot").value)
        for old_start, old_end in capsules:
            old_center = 0.5 * (old_start + old_end)
            old_direction = _normalize(old_end - old_start)
            if np.linalg.norm(center - old_center) < duplicate_dist:
                if abs(float(np.dot(direction, old_direction))) > duplicate_dot:
                    return True
        return False

    def _smooth_capsules(self, detections: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
        alpha = float(self.get_parameter("capsule_ema_alpha").value)
        alpha = max(0.0, min(1.0, alpha))
        match_distance = float(self.get_parameter("capsule_match_distance_m").value)
        hold_cycles = max(0, int(self.get_parameter("capsule_hold_cycles").value))

        updated: list[dict[str, object]] = []
        used_detection: set[int] = set()
        for track in self.tracked_capsules:
            old_start = np.asarray(track["start"], dtype=np.float64)
            old_end = np.asarray(track["end"], dtype=np.float64)
            old_center = 0.5 * (old_start + old_end)
            old_direction = _normalize(old_end - old_start)
            best_idx = -1
            best_cost = float("inf")
            for idx, (start, end) in enumerate(detections):
                if idx in used_detection:
                    continue
                center = 0.5 * (start + end)
                direction = _normalize(end - start)
                direction_cost = 1.0 - abs(float(np.dot(old_direction, direction)))
                center_cost = float(np.linalg.norm(center - old_center))
                cost = center_cost + 0.08 * direction_cost
                if center_cost <= match_distance and cost < best_cost:
                    best_idx = idx
                    best_cost = cost
            if best_idx >= 0:
                new_start, new_end = detections[best_idx]
                if float(np.linalg.norm(new_start - old_end) + np.linalg.norm(new_end - old_start)) < float(
                    np.linalg.norm(new_start - old_start) + np.linalg.norm(new_end - old_end)
                ):
                    new_start, new_end = new_end, new_start
                start = (1.0 - alpha) * old_start + alpha * new_start
                end = (1.0 - alpha) * old_end + alpha * new_end
                updated.append({"start": start, "end": end, "misses": 0})
                used_detection.add(best_idx)
            else:
                misses = int(track.get("misses", 0)) + 1
                if misses <= hold_cycles:
                    updated.append({"start": old_start, "end": old_end, "misses": misses})

        for idx, (start, end) in enumerate(detections):
            if idx not in used_detection:
                updated.append({"start": start, "end": end, "misses": 0})

        max_capsules = int(self.get_parameter("max_capsules").value)
        self.tracked_capsules = updated[:max_capsules]
        return [
            (np.asarray(track["start"], dtype=np.float64), np.asarray(track["end"], dtype=np.float64))
            for track in self.tracked_capsules
        ]

    def _markers_from_capsules(self, capsules: list[tuple[np.ndarray, np.ndarray]], header) -> MarkerArray:
        marker_array = MarkerArray()
        delete = Marker()
        delete.header = header
        delete.ns = "cable_capsules_clear"
        delete.id = 0
        delete.action = Marker.DELETEALL
        marker_array.markers.append(delete)

        radius = float(self.get_parameter("radius_m").value)
        rgba = (
            float(self.get_parameter("color_r").value),
            float(self.get_parameter("color_g").value),
            float(self.get_parameter("color_b").value),
            float(self.get_parameter("alpha").value),
        )
        marker_id = 0
        for start, end in capsules:
            marker_array.markers.append(
                self._cylinder_marker(header, marker_id, start, end, radius, rgba)
            )
            marker_id += 1
            marker_array.markers.append(self._sphere_marker(header, marker_id, start, radius, rgba))
            marker_id += 1
            marker_array.markers.append(self._sphere_marker(header, marker_id, end, radius, rgba))
            marker_id += 1
        return marker_array

    def _base_marker(self, header, marker_id: int, rgba: tuple[float, float, float, float]) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = "cable_capsules"
        marker.id = marker_id
        marker.action = Marker.ADD
        marker.color.r = rgba[0]
        marker.color.g = rgba[1]
        marker.color.b = rgba[2]
        marker.color.a = rgba[3]
        marker.lifetime.sec = 0
        marker.frame_locked = False
        return marker

    def _cylinder_marker(
        self,
        header,
        marker_id: int,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        rgba: tuple[float, float, float, float],
    ) -> Marker:
        direction = end - start
        length = float(np.linalg.norm(direction))
        center = 0.5 * (start + end)
        qx, qy, qz, qw = _quat_from_z_axis(direction)
        marker = self._base_marker(header, marker_id, rgba)
        marker.type = Marker.CYLINDER
        marker.pose.position.x = float(center[0])
        marker.pose.position.y = float(center[1])
        marker.pose.position.z = float(center[2])
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw
        marker.scale.x = 2.0 * radius
        marker.scale.y = 2.0 * radius
        marker.scale.z = max(length, 1e-4)
        return marker

    def _sphere_marker(
        self,
        header,
        marker_id: int,
        point: np.ndarray,
        radius: float,
        rgba: tuple[float, float, float, float],
    ) -> Marker:
        marker = self._base_marker(header, marker_id, rgba)
        marker.type = Marker.SPHERE
        marker.pose.position = Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))
        marker.pose.orientation.w = 1.0
        marker.scale.x = 2.0 * radius
        marker.scale.y = 2.0 * radius
        marker.scale.z = 2.0 * radius
        return marker


def main() -> None:
    rclpy.init()
    node = CableCapsuleMarkerNode()
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
