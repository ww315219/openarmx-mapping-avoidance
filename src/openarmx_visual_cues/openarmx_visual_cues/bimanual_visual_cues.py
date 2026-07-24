from __future__ import annotations

import os
import re
import tempfile
import math
import traceback
from dataclasses import dataclass

import cv2
import numpy as np
import pinocchio as pin
import rclpy
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from nav_msgs.msg import Path
from rcl_interfaces.srv import GetParameters
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration as RclpyDuration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float32, Float64, Header, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


LEFT_JOINT_NAMES = [f"openarmx_left_joint{i}" for i in range(1, 8)]
RIGHT_JOINT_NAMES = [f"openarmx_right_joint{i}" for i in range(1, 8)]

Color = tuple[float, float, float, float]

GREEN: Color = (0.05, 0.90, 0.25, 0.90)
YELLOW: Color = (1.00, 0.72, 0.05, 0.90)
RED: Color = (1.00, 0.08, 0.04, 0.95)
GREY: Color = (0.50, 0.53, 0.58, 0.55)
CYAN: Color = (0.05, 0.85, 1.00, 0.85)
MAGENTA: Color = (0.90, 0.20, 0.85, 0.85)
WHITE: Color = (0.95, 0.95, 0.95, 0.95)


@dataclass
class PoseData:
    position: np.ndarray
    rotation: np.ndarray


@dataclass
class CableCapsule:
    center: np.ndarray
    axis: np.ndarray
    half_length: float
    radius: float


def _rotation_from_xyzw(quaternion) -> np.ndarray:
    values = np.array(
        [quaternion.x, quaternion.y, quaternion.z, quaternion.w],
        dtype=float,
    )
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm < 1e-9:
        return np.eye(3)
    x, y, z, w = values / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _xyzw_from_rotation(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    values = np.array([x, y, z, w], dtype=float)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm < 1e-9:
        return 0.0, 0.0, 0.0, 1.0
    values /= norm
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])


class BimanualVisualCues(Node):
    def __init__(self) -> None:
        super().__init__("bimanual_visual_cues")

        self.declare_parameter("urdf_path", "")
        self.declare_parameter("robot_description_node", "/robot_state_publisher")
        self.declare_parameter("global_frame", "world")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("left_desired_pose_topic", "/openarm_mini/left_target_pose")
        self.declare_parameter("right_desired_pose_topic", "/openarm_mini/right_target_pose")
        self.declare_parameter(
            "left_assisted_pose_topic",
            "/visual_cues/left_assisted_target_pose",
        )
        self.declare_parameter(
            "right_assisted_pose_topic",
            "/visual_cues/right_assisted_target_pose",
        )
        self.declare_parameter("clicked_point_topic", "/clicked_point")
        self.declare_parameter("active_arm_topic", "/visual_cues/active_arm")
        self.declare_parameter("active_arm", "right")
        self.declare_parameter("marker_topic", "/openarmx/visual_cues")
        self.declare_parameter(
            "left_clearance_topic",
            "/openarmx/bimanual/left_min_esdf_clearance",
        )
        self.declare_parameter(
            "right_clearance_topic",
            "/openarmx/bimanual/right_min_esdf_clearance",
        )
        self.declare_parameter(
            "avoidance_status_topic",
            "/openarmx/bimanual/esdf_avoidance_status",
        )
        self.declare_parameter("left_joint_names", LEFT_JOINT_NAMES)
        self.declare_parameter("right_joint_names", RIGHT_JOINT_NAMES)
        self.declare_parameter("left_ee_frame", "openarmx_left_hand_tcp")
        self.declare_parameter("right_ee_frame", "openarmx_right_hand_tcp")
        self.declare_parameter("rate_hz", 15.0)
        self.declare_parameter("safety_margin", 0.04)
        self.declare_parameter("activation_margin", 0.10)
        self.declare_parameter("ready_distance", 0.01)
        self.declare_parameter("assisted_grasp_enabled", False)
        self.declare_parameter("assist_activation_distance", 0.08)
        self.declare_parameter("assist_min_alpha", 0.0)
        self.declare_parameter("assist_max_alpha", 1.00)
        self.declare_parameter("assist_ramp_duration", 0.40)
        self.declare_parameter("axis_length", 0.10)
        self.declare_parameter("clearance_timeout_s", 1.0)
        self.declare_parameter("show_actual_axes", False)
        self.declare_parameter("show_desired_axes", False)
        self.declare_parameter("show_tracking_line", False)
        self.declare_parameter("show_tracking_text", False)
        self.declare_parameter("show_clearance_text", False)
        self.declare_parameter("show_target_text", True)
        self.declare_parameter("tracking_line_min_error", 0.015)
        self.declare_parameter("ray_selection_enabled", True)
        self.declare_parameter("target_selection_mode", "ray")
        self.declare_parameter("bimanual_target_selection", True)
        self.declare_parameter("show_rviz_selection_ray", False)
        self.declare_parameter("pointcloud_topic", "/foundation_stereo/points")
        self.declare_parameter("ray_axis", [0.0, 0.0, 1.0])
        self.declare_parameter("ray_min_distance", 0.10)
        self.declare_parameter("ray_length", 1.50)
        self.declare_parameter("ray_hit_radius", 0.045)
        self.declare_parameter("ray_bin_size", 0.04)
        self.declare_parameter("ray_min_bin_points", 3)
        self.declare_parameter("ray_update_hz", 5.0)
        self.declare_parameter("pointcloud_max_points", 40000)
        self.declare_parameter("pointcloud_timeout_s", 1.00)
        self.declare_parameter("candidate_hold_s", 0.75)
        self.declare_parameter("candidate_ema_alpha", 0.35)
        self.declare_parameter("nearest_target_radius", 0.08)
        self.declare_parameter("nearest_target_min_distance", 0.008)
        self.declare_parameter("nearest_target_forward_only", True)
        self.declare_parameter("nearest_target_min_forward_distance", 0.035)
        self.declare_parameter("nearest_target_lateral_radius", 0.040)
        self.declare_parameter("nearest_target_support_radius", 0.018)
        self.declare_parameter("nearest_target_min_points", 3)
        self.declare_parameter("nearest_target_lock_delay_s", 0.20)
        self.declare_parameter("nearest_target_max_jitter", 0.015)
        self.declare_parameter("nearest_target_self_filter_hand_radius", 0.045)
        self.declare_parameter("nearest_target_self_filter_padding", 0.010)
        self.declare_parameter("show_aim_reticle", True)
        self.declare_parameter("aim_reticle_distance", 0.70)
        self.declare_parameter("use_aim_reticle_as_fallback_target", True)
        self.declare_parameter("prefer_aim_reticle_target", True)
        self.declare_parameter("aim_reticle_pick_radius_px", 24)
        self.declare_parameter("aim_reticle_pick_min_points", 1)
        self.declare_parameter("color_image_topic", "/camera/color/image_raw")
        self.declare_parameter(
            "compressed_color_image_topic",
            "/camera/color/image_raw/compressed",
        )
        self.declare_parameter("color_camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("annotated_image_topic", "/visual_cues/annotated_image")
        self.declare_parameter(
            "compressed_annotated_image_topic",
            "/visual_cues/annotated_image/compressed",
        )
        self.declare_parameter("use_compressed_image", True)
        self.declare_parameter("jpeg_quality", 85)
        self.declare_parameter("left_gripper_topic", "/openarm_mini/left_gripper")
        self.declare_parameter("right_gripper_topic", "/openarm_mini/right_gripper")
        # Mini values 32/12 on its 0-100 scale, mapped by the UDP bridge to 0-0.044 m.
        self.declare_parameter("gripper_open_threshold", 0.01408)
        self.declare_parameter("gripper_closed_threshold", 0.00528)
        self.declare_parameter("lock_close_count", 1)
        self.declare_parameter("gesture_timeout_s", 3.0)
        self.declare_parameter("target_tracking_search_radius", 0.25)
        self.declare_parameter("target_tracking_min_points", 2)
        self.declare_parameter("target_roi_pointcloud_topic", "/visual_cues/target_roi_points")
        self.declare_parameter("target_roi_active_topic", "/visual_cues/target_roi_active")
        self.declare_parameter("target_bbox_marker_topic", "/visual_cues/target_bbox")
        self.declare_parameter("target_roi_max_points", 3000)
        self.declare_parameter("target_bbox_padding", 0.015)
        self.declare_parameter("target_bbox_min_size", 0.03)
        self.declare_parameter("target_lock_offset_x", 0.03)
        self.declare_parameter("target_lock_offset_y", 0.0)
        self.declare_parameter("target_lock_offset_z", 0.0)
        self.declare_parameter("show_target_roi", False)
        self.declare_parameter("cable_capsules_topic", "/perception/cable_capsules")
        self.declare_parameter("exclude_cable_capsule_points", True)
        self.declare_parameter("cable_capsule_exclusion_padding", 0.025)
        self.declare_parameter("exclude_robot_points", True)
        self.declare_parameter("robot_self_filter_link_radius", 0.075)
        self.declare_parameter("robot_self_filter_hand_radius", 0.055)
        self.declare_parameter("robot_self_filter_padding", 0.010)
        self.declare_parameter("show_robot_self_filter", False)
        self.declare_parameter("untangle_mode_topic", "/openarmx/untangle_mode")
        self.declare_parameter("selection_state_topic", "/visual_cues/selection_state")
        self.declare_parameter("target_locked_topic", "/visual_cues/target_locked")
        self.declare_parameter("target_distance_topic", "/visual_cues/target_distance")
        self.declare_parameter("grasp_ready_topic", "/visual_cues/grasp_ready")
        self.declare_parameter("show_untangle_preview", True)
        self.declare_parameter("show_rviz_untangle_preview", False)
        self.declare_parameter("left_untangle_path_topic", "/untangle/left_ee_path")
        self.declare_parameter("right_untangle_path_topic", "/untangle/right_ee_path")
        self.declare_parameter("untangle_preview_status_topic", "/untangle/preview_status")
        self.declare_parameter("untangle_preview_path_width", 0.012)

        self.global_frame = str(self.get_parameter("global_frame").value)
        self.robot_description_node = str(self.get_parameter("robot_description_node").value)
        self.left_joint_names = list(self.get_parameter("left_joint_names").value)
        self.right_joint_names = list(self.get_parameter("right_joint_names").value)
        self.left_ee_frame = str(self.get_parameter("left_ee_frame").value)
        self.right_ee_frame = str(self.get_parameter("right_ee_frame").value)
        self.rate_hz = max(1.0, float(self.get_parameter("rate_hz").value))
        self.safety_margin = max(0.0, float(self.get_parameter("safety_margin").value))
        self.activation_margin = max(
            self.safety_margin,
            float(self.get_parameter("activation_margin").value),
        )
        self.ready_distance = max(0.005, float(self.get_parameter("ready_distance").value))
        self.assisted_grasp_enabled = bool(
            self.get_parameter("assisted_grasp_enabled").value
        )
        self.assist_activation_distance = max(
            self.ready_distance + 1e-3,
            float(self.get_parameter("assist_activation_distance").value),
        )
        self.assist_min_alpha = float(
            np.clip(float(self.get_parameter("assist_min_alpha").value), 0.0, 1.0)
        )
        self.assist_max_alpha = float(
            np.clip(float(self.get_parameter("assist_max_alpha").value), 0.0, 1.0)
        )
        if self.assist_max_alpha < self.assist_min_alpha:
            self.assist_max_alpha = self.assist_min_alpha
        self.assist_ramp_duration = max(
            0.05,
            float(self.get_parameter("assist_ramp_duration").value),
        )
        self.axis_length = max(0.02, float(self.get_parameter("axis_length").value))
        self.clearance_timeout_s = max(
            0.1,
            float(self.get_parameter("clearance_timeout_s").value),
        )
        self.show_actual_axes = bool(self.get_parameter("show_actual_axes").value)
        self.show_desired_axes = bool(self.get_parameter("show_desired_axes").value)
        self.show_tracking_line = bool(self.get_parameter("show_tracking_line").value)
        self.show_tracking_text = bool(self.get_parameter("show_tracking_text").value)
        self.show_clearance_text = bool(self.get_parameter("show_clearance_text").value)
        self.show_target_text = bool(self.get_parameter("show_target_text").value)
        self.tracking_line_min_error = max(
            0.0,
            float(self.get_parameter("tracking_line_min_error").value),
        )
        self.ray_selection_enabled = bool(self.get_parameter("ray_selection_enabled").value)
        self.target_selection_mode = str(
            self.get_parameter("target_selection_mode").value
        ).strip().lower()
        if self.target_selection_mode not in ("nearest_tcp", "ray"):
            raise RuntimeError(
                "target_selection_mode must be either 'nearest_tcp' or 'ray'"
            )
        self.bimanual_target_selection = bool(
            self.get_parameter("bimanual_target_selection").value
        )
        self.show_rviz_selection_ray = bool(
            self.get_parameter("show_rviz_selection_ray").value
        )
        self.ray_axis = np.asarray(self.get_parameter("ray_axis").value, dtype=float).reshape(3)
        ray_axis_norm = float(np.linalg.norm(self.ray_axis))
        if not np.isfinite(ray_axis_norm) or ray_axis_norm < 1e-9:
            self.ray_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            self.ray_axis /= ray_axis_norm
        self.ray_min_distance = max(
            0.0,
            float(self.get_parameter("ray_min_distance").value),
        )
        self.ray_length = max(
            self.ray_min_distance + 0.05,
            float(self.get_parameter("ray_length").value),
        )
        self.ray_hit_radius = max(0.005, float(self.get_parameter("ray_hit_radius").value))
        self.ray_bin_size = max(0.005, float(self.get_parameter("ray_bin_size").value))
        self.ray_min_bin_points = max(
            1,
            int(self.get_parameter("ray_min_bin_points").value),
        )
        self.ray_update_hz = max(1.0, float(self.get_parameter("ray_update_hz").value))
        self.pointcloud_max_points = max(
            1000,
            int(self.get_parameter("pointcloud_max_points").value),
        )
        self.pointcloud_timeout_s = max(
            0.1,
            float(self.get_parameter("pointcloud_timeout_s").value),
        )
        self.candidate_hold_s = max(
            0.0,
            float(self.get_parameter("candidate_hold_s").value),
        )
        self.candidate_ema_alpha = float(
            np.clip(float(self.get_parameter("candidate_ema_alpha").value), 0.0, 1.0)
        )
        self.nearest_target_radius = max(
            0.01,
            float(self.get_parameter("nearest_target_radius").value),
        )
        self.nearest_target_min_distance = float(
            np.clip(
                float(self.get_parameter("nearest_target_min_distance").value),
                0.0,
                self.nearest_target_radius - 1e-3,
            )
        )
        self.nearest_target_forward_only = bool(
            self.get_parameter("nearest_target_forward_only").value
        )
        self.nearest_target_min_forward_distance = float(
            np.clip(
                float(
                    self.get_parameter(
                        "nearest_target_min_forward_distance"
                    ).value
                ),
                0.0,
                self.nearest_target_radius,
            )
        )
        self.nearest_target_lateral_radius = max(
            0.005,
            float(self.get_parameter("nearest_target_lateral_radius").value),
        )
        self.nearest_target_support_radius = max(
            0.003,
            float(self.get_parameter("nearest_target_support_radius").value),
        )
        self.nearest_target_min_points = max(
            1,
            int(self.get_parameter("nearest_target_min_points").value),
        )
        self.nearest_target_lock_delay_s = max(
            0.0,
            float(self.get_parameter("nearest_target_lock_delay_s").value),
        )
        self.nearest_target_max_jitter = max(
            0.001,
            float(self.get_parameter("nearest_target_max_jitter").value),
        )
        self.nearest_target_self_filter_hand_radius = max(
            0.005,
            float(
                self.get_parameter(
                    "nearest_target_self_filter_hand_radius"
                ).value
            ),
        )
        self.nearest_target_self_filter_padding = max(
            0.0,
            float(
                self.get_parameter(
                    "nearest_target_self_filter_padding"
                ).value
            ),
        )
        self.show_aim_reticle = bool(self.get_parameter("show_aim_reticle").value)
        self.aim_reticle_distance = max(
            0.05,
            float(self.get_parameter("aim_reticle_distance").value),
        )
        self.use_aim_reticle_as_fallback_target = bool(
            self.get_parameter("use_aim_reticle_as_fallback_target").value
        )
        self.prefer_aim_reticle_target = bool(
            self.get_parameter("prefer_aim_reticle_target").value
        )
        self.aim_reticle_pick_radius_px = max(
            2,
            int(self.get_parameter("aim_reticle_pick_radius_px").value),
        )
        self.aim_reticle_pick_min_points = max(
            1,
            int(self.get_parameter("aim_reticle_pick_min_points").value),
        )
        self.use_compressed_image = bool(self.get_parameter("use_compressed_image").value)
        self.jpeg_quality = int(
            np.clip(int(self.get_parameter("jpeg_quality").value), 40, 100)
        )
        self.gripper_open_threshold = float(
            self.get_parameter("gripper_open_threshold").value
        )
        self.gripper_closed_threshold = float(
            self.get_parameter("gripper_closed_threshold").value
        )
        if self.gripper_closed_threshold >= self.gripper_open_threshold:
            raise RuntimeError("gripper_closed_threshold must be below gripper_open_threshold")
        self.lock_close_count = max(1, int(self.get_parameter("lock_close_count").value))
        self.gesture_timeout_s = max(
            0.2,
            float(self.get_parameter("gesture_timeout_s").value),
        )
        self.target_tracking_search_radius = max(
            0.02,
            float(self.get_parameter("target_tracking_search_radius").value),
        )
        self.target_tracking_min_points = max(
            1,
            int(self.get_parameter("target_tracking_min_points").value),
        )
        self.target_roi_max_points = max(
            100,
            int(self.get_parameter("target_roi_max_points").value),
        )
        self.target_bbox_padding = max(
            0.0,
            float(self.get_parameter("target_bbox_padding").value),
        )
        self.target_bbox_min_size = max(
            0.005,
            float(self.get_parameter("target_bbox_min_size").value),
        )
        self.target_lock_offset = np.array(
            [
                float(self.get_parameter("target_lock_offset_x").value),
                float(self.get_parameter("target_lock_offset_y").value),
                float(self.get_parameter("target_lock_offset_z").value),
            ],
            dtype=float,
        )
        self.show_target_roi = bool(self.get_parameter("show_target_roi").value)
        self.exclude_cable_capsule_points = bool(
            self.get_parameter("exclude_cable_capsule_points").value
        )
        self.cable_capsule_exclusion_padding = max(
            0.0,
            float(self.get_parameter("cable_capsule_exclusion_padding").value),
        )
        self.exclude_robot_points = bool(self.get_parameter("exclude_robot_points").value)
        self.robot_self_filter_link_radius = max(
            0.005,
            float(self.get_parameter("robot_self_filter_link_radius").value),
        )
        self.robot_self_filter_hand_radius = max(
            0.005,
            float(self.get_parameter("robot_self_filter_hand_radius").value),
        )
        self.robot_self_filter_padding = max(
            0.0,
            float(self.get_parameter("robot_self_filter_padding").value),
        )
        self.show_robot_self_filter = bool(
            self.get_parameter("show_robot_self_filter").value
        )
        self.show_untangle_preview = bool(
            self.get_parameter("show_untangle_preview").value
        )
        self.show_rviz_untangle_preview = bool(
            self.get_parameter("show_rviz_untangle_preview").value
        )
        self.untangle_preview_path_width = max(
            0.002,
            float(self.get_parameter("untangle_preview_path_width").value),
        )
        self.active_arm = self._validated_arm(str(self.get_parameter("active_arm").value))
        self.bridge = CvBridge()

        urdf_path = str(self.get_parameter("urdf_path").value).strip()
        self._temp_urdf_path: str | None = None
        if not urdf_path:
            urdf_path = self._write_robot_description_to_temp_urdf()
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)
        self.left_q_indices = self._joint_q_indices(self.left_joint_names)
        self.right_q_indices = self._joint_q_indices(self.right_joint_names)
        self.left_joint_ids = [int(self.model.getJointId(name)) for name in self.left_joint_names]
        self.right_joint_ids = [int(self.model.getJointId(name)) for name in self.right_joint_names]
        self.left_frame_id = self._required_frame_id(self.left_ee_frame)
        self.right_frame_id = self._required_frame_id(self.right_ee_frame)

        self.joint_values: dict[str, float] = {}
        self.actual: dict[str, PoseData | None] = {"left": None, "right": None}
        self.desired: dict[str, PoseData | None] = {"left": None, "right": None}
        self.desired_msgs: dict[str, PoseStamped | None] = {"left": None, "right": None}
        self.targets: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self.target_lock_origins: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self.target_bboxes: dict[str, tuple[np.ndarray, np.ndarray] | None] = {
            "left": None,
            "right": None,
        }
        self.candidates: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self.clearances: dict[str, float | None] = {"left": None, "right": None}
        self.clearance_times = {"left": None, "right": None}
        self.gripper_states: dict[str, str] = {"left": "unknown", "right": "unknown"}
        self.gripper_values: dict[str, float | None] = {"left": None, "right": None}
        self.gesture_counts: dict[str, int] = {"left": 0, "right": 0}
        self.last_gesture_times = {"left": None, "right": None}
        self.last_grasp_trigger_distances: dict[str, float | None] = {
            "left": None,
            "right": None,
        }
        self.last_grasp_trigger_ready: dict[str, bool | None] = {"left": None, "right": None}
        self.assist_active: dict[str, bool] = {"left": False, "right": False}
        self.assist_alpha: dict[str, float] = {"left": 0.0, "right": 0.0}
        self.assist_start_times = {"left": None, "right": None}
        self.target_drift_start_times = {"left": None, "right": None}
        self.last_candidate_times = {"left": None, "right": None}
        self.auto_candidate_start_times = {"left": None, "right": None}
        self.latest_pointcloud: PointCloud2 | None = None
        self.latest_pointcloud_time = None
        self.cable_capsules: list[CableCapsule] = []
        self.untangle_preview_paths: dict[str, list[np.ndarray]] = {
            "left": [],
            "right": [],
        }
        self.untangle_preview_status = ""
        self.color_camera_info: CameraInfo | None = None
        self.last_ray_update_time = None
        self.untangle_mode = False

        self.tf_buffer = Buffer(cache_time=RclpyDuration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.image_callback_group = MutuallyExclusiveCallbackGroup()
        self.ray_callback_group = MutuallyExclusiveCallbackGroup()

        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_states_topic").value),
            self._joint_state_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("left_desired_pose_topic").value),
            lambda msg: self._desired_pose_cb("left", msg),
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("right_desired_pose_topic").value),
            lambda msg: self._desired_pose_cb("right", msg),
            10,
        )
        self.create_subscription(
            PointStamped,
            str(self.get_parameter("clicked_point_topic").value),
            self._clicked_point_cb,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("active_arm_topic").value),
            self._active_arm_cb,
            10,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("left_clearance_topic").value),
            lambda msg: self._clearance_cb("left", msg),
            10,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("right_clearance_topic").value),
            lambda msg: self._clearance_cb("right", msg),
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("avoidance_status_topic").value),
            self._avoidance_status_cb,
            10,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("pointcloud_topic").value),
            self._pointcloud_cb,
            qos_profile_sensor_data,
            callback_group=self.ray_callback_group,
        )
        self.cable_capsules_sub = self.create_subscription(
            MarkerArray,
            str(self.get_parameter("cable_capsules_topic").value),
            self._cable_capsules_cb,
            10,
        )
        preview_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            Path,
            str(self.get_parameter("left_untangle_path_topic").value),
            lambda msg: self._untangle_preview_path_cb("left", msg),
            preview_qos,
        )
        self.create_subscription(
            Path,
            str(self.get_parameter("right_untangle_path_topic").value),
            lambda msg: self._untangle_preview_path_cb("right", msg),
            preview_qos,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("untangle_preview_status_topic").value),
            self._untangle_preview_status_cb,
            preview_qos,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("color_camera_info_topic").value),
            self._color_camera_info_cb,
            qos_profile_sensor_data,
            callback_group=self.image_callback_group,
        )
        if self.use_compressed_image:
            self.create_subscription(
                CompressedImage,
                str(self.get_parameter("compressed_color_image_topic").value),
                self._compressed_color_image_cb,
                qos_profile_sensor_data,
                callback_group=self.image_callback_group,
            )
        else:
            self.create_subscription(
                Image,
                str(self.get_parameter("color_image_topic").value),
                self._color_image_cb,
                qos_profile_sensor_data,
                callback_group=self.image_callback_group,
            )
        self.create_subscription(
            Float64,
            str(self.get_parameter("left_gripper_topic").value),
            lambda msg: self._gripper_cb("left", msg),
            10,
        )
        self.create_subscription(
            Float64,
            str(self.get_parameter("right_gripper_topic").value),
            lambda msg: self._gripper_cb("right", msg),
            10,
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("marker_topic").value),
            10,
        )
        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.left_target_pub = self.create_publisher(
            PoseStamped,
            "/visual_cues/left_selected_target_pose",
            latched_qos,
        )
        self.right_target_pub = self.create_publisher(
            PoseStamped,
            "/visual_cues/right_selected_target_pose",
            latched_qos,
        )
        self.left_assisted_pose_pub = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("left_assisted_pose_topic").value),
            10,
        )
        self.right_assisted_pose_pub = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("right_assisted_pose_topic").value),
            10,
        )
        self.target_roi_points_pub = self.create_publisher(
            PointCloud2,
            str(self.get_parameter("target_roi_pointcloud_topic").value),
            qos_profile_sensor_data,
        )
        self.target_roi_active_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("target_roi_active_topic").value),
            10,
        )
        self.target_bbox_pub = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("target_bbox_marker_topic").value),
            10,
        )
        self.untangle_mode_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("untangle_mode_topic").value),
            latched_qos,
        )
        self.target_locked_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("target_locked_topic").value),
            latched_qos,
        )
        self.target_distance_pub = self.create_publisher(
            Float32,
            str(self.get_parameter("target_distance_topic").value),
            10,
        )
        self.grasp_ready_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("grasp_ready_topic").value),
            10,
        )
        self.selection_state_pub = self.create_publisher(
            String,
            str(self.get_parameter("selection_state_topic").value),
            10,
        )
        self.annotated_image_pub = None
        self.compressed_annotated_image_pub = None
        if self.use_compressed_image:
            self.compressed_annotated_image_pub = self.create_publisher(
                CompressedImage,
                str(self.get_parameter("compressed_annotated_image_topic").value),
                qos_profile_sensor_data,
            )
        else:
            self.annotated_image_pub = self.create_publisher(
                Image,
                str(self.get_parameter("annotated_image_topic").value),
                qos_profile_sensor_data,
            )
        self.create_service(Trigger, "/visual_cues/clear_targets", self._clear_targets_cb)
        self.timer = self.create_timer(
            1.0 / self.rate_hz,
            self._timer_cb,
            callback_group=self.ray_callback_group,
        )
        self._publish_mode_state()

        self.get_logger().info(
            "Bimanual visual cues: "
            f"frame={self.global_frame}, active_arm={self.active_arm}, "
            f"selection={self.target_selection_mode}, "
            f"simultaneous_selection={self.bimanual_target_selection}, "
            f"marker={self.get_parameter('marker_topic').value}, "
            f"pointcloud={self.get_parameter('pointcloud_topic').value}. "
            + (
                f"Open gripper to auto-lock the nearest supported 3D point within "
                f"{self.nearest_target_radius * 100.0:.1f} cm."
                if self.target_selection_mode == "nearest_tcp"
                else f"Aim with TCP +Z and close the Mini gripper "
                f"{self.lock_close_count} times to lock."
            )
        )

    def _write_robot_description_to_temp_urdf(self) -> str:
        service = f"{self.robot_description_node}/get_parameters"
        client = self.create_client(GetParameters, service)
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(
                f"Timed out waiting for {service}. Start OpenArmX robot_state_publisher first "
                "or pass urdf_path."
            )
        request = GetParameters.Request()
        request.names = ["robot_description"]
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        response = future.result()
        if response is None or not response.values or not response.values[0].string_value:
            raise RuntimeError(f"{service} did not return robot_description")
        fd, path = tempfile.mkstemp(prefix="openarmx_visual_cues_", suffix=".urdf")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as file_obj:
            file_obj.write(response.values[0].string_value)
        self._temp_urdf_path = path
        return path

    def _joint_q_indices(self, joint_names: list[str]) -> list[int]:
        indices: list[int] = []
        for name in joint_names:
            if not self.model.existJointName(name):
                raise RuntimeError(f"Joint {name!r} not found in URDF")
            joint = self.model.joints[self.model.getJointId(name)]
            if joint.nq != 1:
                raise RuntimeError(f"Joint {name!r} has nq={joint.nq}; expected 1")
            indices.append(int(joint.idx_q))
        return indices

    def _required_frame_id(self, frame_name: str) -> int:
        if not self.model.existFrame(frame_name):
            raise RuntimeError(f"Frame {frame_name!r} not found in URDF")
        return int(self.model.getFrameId(frame_name))

    def _validated_arm(self, arm: str) -> str:
        normalized = arm.strip().lower()
        if normalized not in ("left", "right"):
            self.get_logger().warn(f"Unknown arm {arm!r}; using right.")
            return "right"
        return normalized

    def _joint_state_cb(self, msg: JointState) -> None:
        for name, position in zip(msg.name, msg.position):
            if np.isfinite(position):
                self.joint_values[name] = float(position)

    def _desired_pose_cb(self, arm: str, msg: PoseStamped) -> None:
        self.desired_msgs[arm] = msg
        pose = self._pose_to_global(msg)
        if pose is not None:
            self.desired[arm] = pose
        self._publish_assisted_pose_target(arm)

    def _clicked_point_cb(self, msg: PointStamped) -> None:
        position = self._point_to_global(msg)
        if position is None:
            return
        self._lock_target(self.active_arm, position)

    def _active_arm_cb(self, msg: String) -> None:
        requested = msg.data.strip().lower()
        if requested not in ("left", "right"):
            self.get_logger().warn("active_arm must be 'left' or 'right'.")
            return
        self.active_arm = requested
        self.gesture_counts[requested] = 0
        self.last_gesture_times[requested] = None
        self.get_logger().info(f"Visual-cue active arm changed to {self.active_arm}.")

    def _pointcloud_cb(self, msg: PointCloud2) -> None:
        self.latest_pointcloud = msg
        self.latest_pointcloud_time = self.get_clock().now()

    def _cable_capsules_cb(self, msg: MarkerArray) -> None:
        capsules: list[CableCapsule] = []
        for marker in msg.markers:
            if marker.action in (Marker.DELETE, Marker.DELETEALL):
                continue
            if marker.type != Marker.CYLINDER:
                continue
            if marker.scale.z <= 1e-6 or marker.scale.x <= 1e-6:
                continue
            source_frame = marker.header.frame_id.strip() or self.global_frame
            center = np.array(
                [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z],
                dtype=float,
            )
            rotation = _rotation_from_xyzw(marker.pose.orientation)
            if source_frame != self.global_frame:
                transform = self._lookup_transform(source_frame)
                if transform is None:
                    continue
                translation, frame_rotation = transform
                center = frame_rotation @ center + translation
                rotation = frame_rotation @ rotation
            axis = np.asarray(rotation[:, 2], dtype=float)
            axis_norm = float(np.linalg.norm(axis))
            if not np.isfinite(axis_norm) or axis_norm < 1e-9:
                continue
            axis /= axis_norm
            radius = 0.5 * float(max(marker.scale.x, marker.scale.y))
            half_length = 0.5 * float(marker.scale.z)
            capsules.append(CableCapsule(center, axis, half_length, radius))
        self.cable_capsules = capsules

    def _untangle_preview_path_cb(self, arm: str, msg: Path) -> None:
        if not msg.poses:
            self.untangle_preview_paths[arm] = []
            return
        source_frame = msg.header.frame_id.strip() or self.global_frame
        transform = None
        if source_frame != self.global_frame:
            transform = self._lookup_transform(source_frame)
            if transform is None:
                return
        points: list[np.ndarray] = []
        for pose in msg.poses:
            point = np.array(
                [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z],
                dtype=float,
            )
            if transform is not None:
                translation, rotation = transform
                point = rotation @ point + translation
            if np.all(np.isfinite(point)):
                points.append(point)
        self.untangle_preview_paths[arm] = points

    def _untangle_preview_status_cb(self, msg: String) -> None:
        self.untangle_preview_status = msg.data

    def _color_camera_info_cb(self, msg: CameraInfo) -> None:
        self.color_camera_info = msg

    def _color_image_cb(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f"Cannot convert visual-cue color image: {exc}",
                throttle_duration_sec=2.0,
            )
            return
        annotated = self._annotate_color_image(np.asarray(image), msg.header)
        output = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        output.header = msg.header
        if self.annotated_image_pub is not None:
            self.annotated_image_pub.publish(output)

    def _compressed_color_image_cb(self, msg: CompressedImage) -> None:
        encoded = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().warn(
                "Cannot decode visual-cue compressed color image.",
                throttle_duration_sec=2.0,
            )
            return
        annotated = self._annotate_color_image(image, msg.header)
        success, encoded_output = cv2.imencode(
            ".jpg",
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not success:
            self.get_logger().warn(
                "Cannot encode visual-cue compressed image.",
                throttle_duration_sec=2.0,
            )
            return
        output = CompressedImage()
        output.header = msg.header
        output.format = "jpeg"
        output.data = encoded_output.tobytes()
        if self.compressed_annotated_image_pub is not None:
            self.compressed_annotated_image_pub.publish(output)

    def _annotate_color_image(self, image: np.ndarray, header) -> np.ndarray:
        annotated = np.asarray(image).copy()
        arm = self.active_arm
        target = self.targets[arm]
        candidate = self.candidates[arm]
        reticle_point = self._aim_reticle_point(arm)
        point = target if target is not None else candidate
        locked = target is not None
        ready = False
        distance: float | None = None
        if locked and self.actual[arm] is not None:
            distance = float(np.linalg.norm(target - self.actual[arm].position))
            ready = distance < self.ready_distance

        actual_pixel = None
        aim_pixel = None
        if self.actual[arm] is not None:
            actual_pixel = self._global_point_to_pixel(
                self.actual[arm].position,
                header.frame_id,
                annotated.shape[1],
                annotated.shape[0],
            )

        if self.show_aim_reticle and self.target_selection_mode == "ray":
            reticle_arms = (
                ("left", "right")
                if self.bimanual_target_selection
                else (arm,)
            )
            for reticle_arm in reticle_arms:
                if self.targets[reticle_arm] is not None:
                    continue
                reticle_pose = self._aim_reticle_point(reticle_arm)
                if reticle_pose is None:
                    continue
                reticle_pixel = self._global_point_to_pixel(
                    reticle_pose,
                    header.frame_id,
                    annotated.shape[1],
                    annotated.shape[0],
                )
                if reticle_pixel is None:
                    continue
                if reticle_arm == arm:
                    aim_pixel = reticle_pixel
                reticle_color = (
                    (255, 220, 30) if reticle_arm == "left" else (220, 40, 220)
                )
                reticle_actual_pixel = None
                if self.actual[reticle_arm] is not None:
                    reticle_actual_pixel = self._global_point_to_pixel(
                        self.actual[reticle_arm].position,
                        header.frame_id,
                        annotated.shape[1],
                        annotated.shape[0],
                    )
                if reticle_actual_pixel is not None:
                    cv2.line(
                        annotated,
                        reticle_actual_pixel,
                        reticle_pixel,
                        reticle_color,
                        1,
                        cv2.LINE_AA,
                    )
                cv2.drawMarker(
                    annotated,
                    reticle_pixel,
                    reticle_color,
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=24,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )
                cv2.circle(
                    annotated,
                    reticle_pixel,
                    12,
                    reticle_color,
                    2,
                    cv2.LINE_AA,
                )
                self._draw_image_label(
                    annotated,
                    reticle_arm[0].upper(),
                    (reticle_pixel[0] + 14, max(24, reticle_pixel[1] - 12)),
                    reticle_color,
                )

        target_pixel = None
        if point is not None:
            target_pixel = self._global_point_to_pixel(
                point,
                header.frame_id,
                annotated.shape[1],
                annotated.shape[0],
            )
            if target_pixel is not None:
                color = (40, 220, 40) if ready else ((0, 190, 255) if not locked else (0, 150, 255))
                cv2.drawMarker(
                    annotated,
                    target_pixel,
                    color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=28,
                    thickness=3,
                    line_type=cv2.LINE_AA,
                )
                cv2.circle(annotated, target_pixel, 17, color, 2, cv2.LINE_AA)
                if distance is not None:
                    label = "CLOSE GRIPPER" if ready else f"{distance * 100.0:.1f} cm"
                    self._draw_image_label(
                        annotated,
                        label,
                        (target_pixel[0] + 20, max(30, target_pixel[1] - 18)),
                        color,
                    )

        if actual_pixel is not None:
            cv2.circle(annotated, actual_pixel, 9, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.circle(annotated, actual_pixel, 6, (0, 0, 255), -1, cv2.LINE_AA)

        # The active arm keeps the detailed aiming overlay above. Also show the
        # other arm's independently computed candidate/lock so a bimanual
        # operator can confirm both grasp points without switching modes.
        if self.bimanual_target_selection:
            other_arm = "left" if arm == "right" else "right"
            other_target = self.targets[other_arm]
            other_candidate = self.candidates[other_arm]
            other_point = other_target if other_target is not None else other_candidate
            other_color = (255, 220, 30) if other_arm == "left" else (220, 40, 220)
            if other_point is not None:
                other_pixel = self._global_point_to_pixel(
                    other_point,
                    header.frame_id,
                    annotated.shape[1],
                    annotated.shape[0],
                )
                other_actual_pixel = None
                if self.actual[other_arm] is not None:
                    other_actual_pixel = self._global_point_to_pixel(
                        self.actual[other_arm].position,
                        header.frame_id,
                        annotated.shape[1],
                        annotated.shape[0],
                    )
                if other_pixel is not None:
                    if other_target is not None and other_actual_pixel is not None:
                        cv2.arrowedLine(
                            annotated,
                            other_actual_pixel,
                            other_pixel,
                            other_color,
                            2,
                            cv2.LINE_AA,
                            tipLength=0.08,
                        )
                    cv2.drawMarker(
                        annotated,
                        other_pixel,
                        other_color,
                        markerType=cv2.MARKER_CROSS,
                        markerSize=24,
                        thickness=2,
                        line_type=cv2.LINE_AA,
                    )
                    other_state = "LOCK" if other_target is not None else "AIM"
                    self._draw_image_label(
                        annotated,
                        f"{other_arm[0].upper()} {other_state}",
                        (other_pixel[0] + 16, max(28, other_pixel[1] - 14)),
                        other_color,
                    )

        preview_visible = False
        if self.show_untangle_preview:
            for preview_arm, preview_color in (
                ("left", (255, 220, 30)),
                ("right", (220, 40, 220)),
            ):
                pixels = [
                    self._global_point_to_pixel(
                        preview_point,
                        header.frame_id,
                        annotated.shape[1],
                        annotated.shape[0],
                    )
                    for preview_point in self.untangle_preview_paths[preview_arm]
                ]
                valid_pixels = [pixel for pixel in pixels if pixel is not None]
                if len(valid_pixels) < 2:
                    continue
                preview_visible = True
                for first, second in zip(pixels, pixels[1:]):
                    if first is not None and second is not None:
                        cv2.line(annotated, first, second, preview_color, 3, cv2.LINE_AA)
                cv2.circle(annotated, valid_pixels[-1], 8, preview_color, 2, cv2.LINE_AA)

        if ready:
            status = "CLOSE GRIPPER"
            status_color = (40, 220, 40)
        elif locked and distance is not None:
            status = f"AUTO APPROACH - {distance * 100.0:.1f} cm"
            status_color = (0, 150, 255)
        elif candidate is not None:
            if self.target_selection_mode == "nearest_tcp":
                status = "TARGET FOUND - HOLD OPEN"
            else:
                status = f"AIM - close gripper {self.gesture_counts[arm]}/{self.lock_close_count}"
            status_color = (0, 190, 255)
        elif (
            self.target_selection_mode == "ray"
            and self.use_aim_reticle_as_fallback_target
            and reticle_point is not None
        ):
            status = f"AIM* - close gripper {self.gesture_counts[arm]}/{self.lock_close_count}"
            status_color = (255, 255, 0)
        elif self.target_selection_mode == "nearest_tcp":
            if self.gripper_states[arm] == "open":
                status = f"SEARCHING WITHIN {self.nearest_target_radius * 100.0:.0f} CM"
            else:
                status = "OPEN GRIPPER TO SEARCH"
            status_color = (170, 170, 170)
        else:
            status = "SEARCHING"
            status_color = (170, 170, 170)
        self._draw_image_label(annotated, status, (18, 34), status_color)
        if preview_visible:
            self._draw_image_label(
                annotated,
                "SIM PREVIEW - NOT EXECUTED",
                (18, 68),
                (0, 220, 255),
            )
        return annotated

    def _aim_reticle_point(self, arm: str) -> np.ndarray | None:
        actual = self.actual.get(arm)
        if actual is None:
            return None
        direction = actual.rotation @ self.ray_axis
        direction_norm = float(np.linalg.norm(direction))
        if not np.isfinite(direction_norm) or direction_norm < 1e-9:
            return None
        return actual.position + self.aim_reticle_distance * direction / direction_norm

    def _global_point_to_pixel(
        self,
        point: np.ndarray,
        image_frame: str,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int] | None:
        info = self.color_camera_info
        if info is None:
            return None
        camera_frame = info.header.frame_id.strip() or image_frame.strip()
        if not camera_frame:
            return None
        transform = self._lookup_transform(camera_frame)
        if transform is None:
            return None
        translation, rotation = transform
        point_camera = rotation.T @ (np.asarray(point, dtype=float).reshape(3) - translation)
        depth = float(point_camera[2])
        if not np.isfinite(depth) or depth <= 1e-5:
            return None
        scale_x = image_width / max(int(info.width), 1)
        scale_y = image_height / max(int(info.height), 1)
        fx = float(info.k[0]) * scale_x
        fy = float(info.k[4]) * scale_y
        cx = float(info.k[2]) * scale_x
        cy = float(info.k[5]) * scale_y
        u = int(round(fx * float(point_camera[0]) / depth + cx))
        v = int(round(fy * float(point_camera[1]) / depth + cy))
        if u < 0 or u >= image_width or v < 0 or v >= image_height:
            return None
        return u, v

    def _global_points_to_pixels(
        self,
        points: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        info = self.color_camera_info
        if info is None:
            return np.empty((0, 2), dtype=np.int32), np.zeros(0, dtype=bool)
        camera_frame = info.header.frame_id.strip()
        if not camera_frame:
            return np.empty((0, 2), dtype=np.int32), np.zeros(points.shape[0], dtype=bool)
        transform = self._lookup_transform(camera_frame)
        if transform is None:
            return np.empty((0, 2), dtype=np.int32), np.zeros(points.shape[0], dtype=bool)
        translation, rotation = transform
        points_global = np.asarray(points, dtype=float).reshape(-1, 3)
        points_camera = (points_global - translation.reshape(1, 3)) @ rotation
        depth = points_camera[:, 2]
        valid = np.isfinite(depth) & (depth > 1e-5)
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.int32), valid
        scale_x = image_width / max(int(info.width), 1)
        scale_y = image_height / max(int(info.height), 1)
        fx = float(info.k[0]) * scale_x
        fy = float(info.k[4]) * scale_y
        cx = float(info.k[2]) * scale_x
        cy = float(info.k[5]) * scale_y
        u = np.rint(fx * points_camera[:, 0] / depth + cx).astype(np.int32)
        v = np.rint(fy * points_camera[:, 1] / depth + cy).astype(np.int32)
        valid &= (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.int32), valid
        return np.column_stack((u[valid], v[valid])).astype(np.int32), valid

    @staticmethod
    def _draw_image_label(
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.65
        thickness = 2
        (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
        x = max(4, min(int(origin[0]), max(4, image.shape[1] - width - 8)))
        y = max(height + 6, min(int(origin[1]), image.shape[0] - baseline - 6))
        cv2.rectangle(
            image,
            (x - 5, y - height - 5),
            (x + width + 5, y + baseline + 5),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            image,
            text,
            (x, y),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _gripper_cb(self, arm: str, msg: Float64) -> None:
        value = float(msg.data)
        if not np.isfinite(value):
            return
        self.gripper_values[arm] = value
        previous = self.gripper_states[arm]
        if value <= self.gripper_closed_threshold:
            current = "closed"
        elif value >= self.gripper_open_threshold:
            current = "open"
        else:
            current = previous
        if current == previous:
            return
        self.gripper_states[arm] = current
        if current == "open":
            self.auto_candidate_start_times[arm] = None
        if current != "closed" or previous != "open":
            return
        if self.targets[arm] is not None:
            self._handle_locked_target_grasp_event(arm)
            return
        if self.target_selection_mode == "nearest_tcp":
            self.candidates[arm] = None
            self.last_candidate_times[arm] = None
            self.auto_candidate_start_times[arm] = None
            return
        candidate = None
        if self.prefer_aim_reticle_target and self.use_aim_reticle_as_fallback_target:
            candidate = self._scene_point_under_aim_reticle(arm)
        if candidate is None:
            candidate = self.candidates[arm]
        if (
            candidate is None
            and not self.prefer_aim_reticle_target
            and self.use_aim_reticle_as_fallback_target
        ):
            candidate = self._scene_point_under_aim_reticle(arm)
        if candidate is None:
            self.gesture_counts[arm] = 0
            self.last_gesture_times[arm] = None
            return
        now = self.get_clock().now()
        previous_time = self.last_gesture_times[arm]
        if previous_time is not None:
            age = (now - previous_time).nanoseconds * 1e-9
            if age > self.gesture_timeout_s:
                self.gesture_counts[arm] = 0
        self.gesture_counts[arm] += 1
        self.last_gesture_times[arm] = now
        count = self.gesture_counts[arm]
        self.get_logger().info(
            f"{arm} target confirmation {count}/{self.lock_close_count}"
        )
        if count >= self.lock_close_count:
            self._lock_target(arm, candidate)

    def _scene_point_under_aim_reticle(self, arm: str) -> np.ndarray | None:
        if self.latest_pointcloud is None or self.color_camera_info is None:
            return None
        if self.latest_pointcloud_time is not None:
            age = (self.get_clock().now() - self.latest_pointcloud_time).nanoseconds * 1e-9
            if age > self.pointcloud_timeout_s:
                return None
        reticle_point = self._aim_reticle_point(arm)
        if reticle_point is None:
            return None
        width = max(int(self.color_camera_info.width), 1)
        height = max(int(self.color_camera_info.height), 1)
        reticle_pixel = self._global_point_to_pixel(
            reticle_point,
            self.color_camera_info.header.frame_id,
            width,
            height,
        )
        if reticle_pixel is None:
            return None
        points = self._pointcloud_points_in_global(self.latest_pointcloud)
        if points.shape[0] == 0:
            return None
        return self._scene_point_under_aim_reticle_from_points(arm, points)

    def _scene_point_under_aim_reticle_from_points(
        self,
        arm: str,
        points: np.ndarray,
    ) -> np.ndarray | None:
        if self.color_camera_info is None or points.shape[0] == 0:
            return None
        reticle_point = self._aim_reticle_point(arm)
        if reticle_point is None:
            return None
        width = max(int(self.color_camera_info.width), 1)
        height = max(int(self.color_camera_info.height), 1)
        reticle_pixel = self._global_point_to_pixel(
            reticle_point,
            self.color_camera_info.header.frame_id,
            width,
            height,
        )
        if reticle_pixel is None:
            return None
        pixels, valid = self._global_points_to_pixels(points, width, height)
        if pixels.shape[0] == 0:
            return None
        target_pixel = np.asarray(reticle_pixel, dtype=float).reshape(1, 2)
        pixel_delta = pixels.astype(float) - target_pixel
        pixel_distance = np.linalg.norm(pixel_delta, axis=1)
        near = np.flatnonzero(pixel_distance <= float(self.aim_reticle_pick_radius_px))
        if near.size < self.aim_reticle_pick_min_points:
            return None
        valid_points = points[valid]
        nearest = near[int(np.argmin(pixel_distance[near]))]
        return valid_points[nearest].copy()

    def _handle_locked_target_grasp_event(self, arm: str) -> None:
        target = self.targets[arm]
        actual = self.actual[arm]
        if target is None or actual is None:
            self.last_grasp_trigger_distances[arm] = None
            self.last_grasp_trigger_ready[arm] = False
            return
        distance = float(np.linalg.norm(target - actual.position))
        ready = distance < self.ready_distance
        self.last_grasp_trigger_distances[arm] = distance
        self.last_grasp_trigger_ready[arm] = ready
        level = self.get_logger().info if ready else self.get_logger().warn
        level(
            f"{arm} grasp trigger: TCP-target distance={distance * 100.0:.1f} cm "
            f"({'ready' if ready else 'too far'}; threshold={self.ready_distance * 100.0:.1f} cm)"
        )
        reason = "grasp completed" if ready else "gripper closed; assisted target cancelled"
        self._unlock_target(arm, reason)

    def _clearance_cb(self, arm: str, msg: Float32) -> None:
        value = float(msg.data)
        if np.isfinite(value):
            self.clearances[arm] = value
            self.clearance_times[arm] = self.get_clock().now()

    def _avoidance_status_cb(self, msg: String) -> None:
        for arm, field in (("left", "left_min"), ("right", "right_min")):
            current_time = self.clearance_times[arm]
            if current_time is not None:
                age = (self.get_clock().now() - current_time).nanoseconds * 1e-9
                if age <= self.clearance_timeout_s:
                    continue
            match = re.search(rf"(?:^|\s){field}=([-+a-zA-Z0-9.eE]+)", msg.data)
            if match is None:
                continue
            try:
                value = float(match.group(1))
            except ValueError:
                continue
            if np.isfinite(value):
                self.clearances[arm] = value
                self.clearance_times[arm] = self.get_clock().now()

    def _clear_targets_cb(self, _request, response):
        self.targets = {"left": None, "right": None}
        self.target_lock_origins = {"left": None, "right": None}
        self.target_bboxes = {"left": None, "right": None}
        self.candidates = {"left": None, "right": None}
        self.last_candidate_times = {"left": None, "right": None}
        self.auto_candidate_start_times = {"left": None, "right": None}
        self.gesture_counts = {"left": 0, "right": 0}
        self.last_gesture_times = {"left": None, "right": None}
        self.last_grasp_trigger_distances = {"left": None, "right": None}
        self.last_grasp_trigger_ready = {"left": None, "right": None}
        self.assist_active = {"left": False, "right": False}
        self.assist_alpha = {"left": 0.0, "right": 0.0}
        self.assist_start_times = {"left": None, "right": None}
        self.target_drift_start_times = {"left": None, "right": None}
        self.untangle_mode = False
        self._publish_empty_target_roi()
        self._publish_mode_state()
        response.success = True
        response.message = "Visual cue targets cleared."
        return response

    def _timer_cb(self) -> None:
        self._sync_assisted_grasp_enabled()
        self._update_actual_poses()
        self._update_ray_candidate()
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        delete = Marker()
        delete.action = Marker.DELETEALL
        markers.markers.append(delete)

        for arm in ("left", "right"):
            actual = self.actual[arm]
            desired = self.desired[arm]
            base_id = 0 if arm == "left" else 100
            if actual is not None:
                if self.show_actual_axes:
                    self._append_axes(markers, actual, stamp, base_id, f"{arm}_actual")
                self._append_clearance_cue(markers, arm, actual.position, stamp, base_id + 10)
            if desired is not None and self.show_desired_axes:
                self._append_axes(markers, desired, stamp, base_id + 20, f"{arm}_desired", 0.75)
            if actual is not None and desired is not None:
                self._append_tracking_cue(
                    markers,
                    arm,
                    actual.position,
                    desired.position,
                    stamp,
                    base_id + 30,
                )
            if actual is not None and self.targets[arm] is not None:
                self._append_target_cue(
                    markers,
                    arm,
                    actual.position,
                    self.targets[arm],
                    stamp,
                    base_id + 40,
                )
                if self.show_target_roi:
                    self._append_target_roi_cue(
                        markers,
                        arm,
                        stamp,
                        base_id + 50,
                    )
            if (
                arm == self.active_arm
                and actual is not None
                and self.ray_selection_enabled
                and self.show_rviz_selection_ray
            ):
                self._append_ray_selection_cue(
                    markers,
                    arm,
                    actual,
                    self.candidates[arm],
                    stamp,
                    base_id + 60,
                )
            if (
                self.show_rviz_untangle_preview
                and self.untangle_preview_paths[arm]
            ):
                self._append_untangle_preview_cue(
                    markers,
                    arm,
                    stamp,
                    base_id + 80,
                )

        if self.show_robot_self_filter:
            self._append_robot_self_filter_cues(markers, stamp)

        self.marker_pub.publish(markers)
        self._publish_target_feedback()

    def _sync_assisted_grasp_enabled(self) -> None:
        enabled = bool(self.get_parameter("assisted_grasp_enabled").value)
        if enabled == self.assisted_grasp_enabled:
            return
        self.assisted_grasp_enabled = enabled
        if enabled:
            self.get_logger().info("Assisted grasp enabled.")
            return

        for arm in ("left", "right"):
            self.targets[arm] = None
            self.target_lock_origins[arm] = None
            self.target_bboxes[arm] = None
            self.candidates[arm] = None
            self.last_candidate_times[arm] = None
            self.auto_candidate_start_times[arm] = None
            self.assist_active[arm] = False
            self.assist_alpha[arm] = 0.0
            self.assist_start_times[arm] = None
        self.untangle_mode = False
        self._publish_empty_target_roi()
        self._publish_mode_state()
        self.get_logger().info("Assisted grasp disabled; all targets cleared.")

    def _append_untangle_preview_cue(
        self,
        array: MarkerArray,
        arm: str,
        stamp,
        marker_id: int,
    ) -> None:
        points = self.untangle_preview_paths[arm]
        if len(points) < 2:
            return
        color = CYAN if arm == "left" else MAGENTA
        path = self._base_marker(f"{arm}_untangle_preview", marker_id, Marker.LINE_STRIP, stamp)
        path.points = [self._point(point) for point in points]
        path.scale.x = self.untangle_preview_path_width
        self._set_color(path, color)
        array.markers.append(path)
        array.markers.append(
            self._arrow_marker(
                f"{arm}_untangle_preview",
                marker_id + 1,
                points[-2],
                points[-1],
                color,
                stamp,
                self.untangle_preview_path_width * 1.5,
                self.untangle_preview_path_width * 3.0,
            )
        )
        label_position = points[0] + np.array([0.0, 0.0, 0.06])
        array.markers.append(
            self._text_marker(
                f"{arm}_untangle_preview",
                marker_id + 2,
                label_position,
                f"{arm.upper()} PREVIEW ONLY",
                color,
                stamp,
                0.045,
            )
        )

    def _lock_target(
        self,
        arm: str,
        position: np.ndarray,
        *,
        apply_lock_offset: bool = True,
    ) -> None:
        raw_target = np.asarray(position, dtype=float).reshape(3)
        applied_offset = self.target_lock_offset if apply_lock_offset else np.zeros(3)
        target = raw_target + applied_offset
        self.targets[arm] = target
        self.target_lock_origins[arm] = target.copy()
        self.target_bboxes[arm] = None
        self.candidates[arm] = target.copy()
        self.gesture_counts[arm] = 0
        self.last_gesture_times[arm] = None
        self.last_grasp_trigger_distances[arm] = None
        self.last_grasp_trigger_ready[arm] = None
        self.auto_candidate_start_times[arm] = None
        self.untangle_mode = True
        self.target_drift_start_times[arm] = None
        self.assist_active[arm] = False
        self.assist_alpha[arm] = 0.0
        self.assist_start_times[arm] = None
        self._publish_selected_target(arm, target)
        self._publish_mode_state()
        self.get_logger().info(
            f"Locked {arm} target at "
            f"[{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}] "
            f"from raw [{raw_target[0]:.3f}, {raw_target[1]:.3f}, {raw_target[2]:.3f}] "
            f"with world offset {np.array2string(applied_offset, precision=3)}; "
            "untangle mode enabled."
        )

    def _unlock_target(self, arm: str, reason: str, candidate: np.ndarray | None = None) -> None:
        self.targets[arm] = None
        self.target_lock_origins[arm] = None
        self.target_bboxes[arm] = None
        self.gesture_counts[arm] = 0
        self.last_gesture_times[arm] = None
        self.last_grasp_trigger_distances[arm] = None
        self.last_grasp_trigger_ready[arm] = None
        self.target_drift_start_times[arm] = None
        self.assist_active[arm] = False
        self.assist_alpha[arm] = 0.0
        self.assist_start_times[arm] = None
        self.auto_candidate_start_times[arm] = None
        if candidate is not None:
            self.candidates[arm] = np.asarray(candidate, dtype=float).reshape(3).copy()
            self.last_candidate_times[arm] = self.get_clock().now()
        if not any(target is not None for target in self.targets.values()):
            self.untangle_mode = False
            self._publish_empty_target_roi()
        self._publish_mode_state()
        self.get_logger().info(f"Unlocked {arm} target: {reason}.")

    def _publish_mode_state(self) -> None:
        locked = any(target is not None for target in self.targets.values())
        self.target_locked_pub.publish(Bool(data=locked))
        self.untangle_mode_pub.publish(Bool(data=bool(self.untangle_mode and locked)))

    def _publish_target_feedback(self) -> None:
        arm = self.active_arm
        target = self.targets[arm]
        actual = self.actual[arm]
        ready = False
        if target is not None and actual is not None:
            self._publish_selected_target(arm, target)
            distance = float(np.linalg.norm(target - actual.position))
            self.target_distance_pub.publish(Float32(data=distance))
            ready = distance < self.ready_distance
            state = "close_gripper" if ready else "auto_approach"
            detail = f"{arm}:{state}:distance={distance:.4f}"
        elif self.candidates[arm] is not None:
            detail = (
                f"{arm}:aiming:confirm={self.gesture_counts[arm]}/{self.lock_close_count}"
            )
        else:
            detail = f"{arm}:searching"
        gripper_value = self.gripper_values[arm]
        gripper_text = "nan" if gripper_value is None else f"{gripper_value:.4f}"
        detail += f":gripper={gripper_text}/{self.gripper_states[arm]}"
        if self.assist_active[arm]:
            detail += f":assist=active:{self.assist_alpha[arm]:.2f}"
        else:
            detail += ":assist=idle"
        grasp_distance = self.last_grasp_trigger_distances[arm]
        grasp_ready = self.last_grasp_trigger_ready[arm]
        if grasp_distance is not None and grasp_ready is not None:
            grasp_state = "ready" if grasp_ready else "too_far"
            detail += f":grasp={grasp_state}:grasp_distance={grasp_distance:.4f}"
        self.grasp_ready_pub.publish(Bool(data=ready))
        self.selection_state_pub.publish(String(data=detail))

    def _publish_assisted_pose_targets(self) -> None:
        for arm in ("left", "right"):
            self._publish_assisted_pose_target(arm)

    def _publish_assisted_pose_target(self, arm: str) -> None:
        desired_msg = self.desired_msgs[arm]
        if desired_msg is None:
            self.assist_active[arm] = False
            self.assist_alpha[arm] = 0.0
            self.assist_start_times[arm] = None
            return

        target = self.targets[arm]
        actual = self.actual[arm]
        output_msg = self._copy_pose_msg(desired_msg)
        active = False
        alpha = 0.0

        if target is None:
            self.assist_start_times[arm] = None
        elif self.assisted_grasp_enabled and actual is not None:
            distance = float(np.linalg.norm(target - actual.position))
            if self.assist_start_times[arm] is None and distance <= self.assist_activation_distance:
                self.assist_start_times[arm] = self.get_clock().now()
                self.get_logger().info(
                    f"{arm} assisted grasp latched: distance={distance * 100.0:.1f} cm, "
                    f"ramp={self.assist_ramp_duration:.2f}s"
                )
            if self.assist_start_times[arm] is not None:
                elapsed = (
                    self.get_clock().now() - self.assist_start_times[arm]
                ).nanoseconds * 1e-9
                ramp = float(np.clip(elapsed / self.assist_ramp_duration, 0.0, 1.0))
                alpha = self.assist_min_alpha + ramp * (
                    self.assist_max_alpha - self.assist_min_alpha
                )
                alpha = float(np.clip(alpha, 0.0, 1.0))
                # In idle this function publishes an exact pass-through of the Mini
                # pose. Once latched, we add a bounded correction to that raw input,
                # preserving the relative-input semantics expected by Jacobian servo.
                delta_global = alpha * (target - actual.position)
                delta_input = self._global_delta_to_pose_frame(
                    delta_global,
                    desired_msg.header.frame_id,
                )
                if delta_input is not None:
                    output_msg.pose.position.x += float(delta_input[0])
                    output_msg.pose.position.y += float(delta_input[1])
                    output_msg.pose.position.z += float(delta_input[2])
                    active = True

        self.assist_active[arm] = active
        self.assist_alpha[arm] = alpha
        self._publish_assisted_pose(arm, output_msg)

    def _copy_pose_msg(self, msg: PoseStamped) -> PoseStamped:
        message = PoseStamped()
        message.header.stamp = msg.header.stamp
        message.header.frame_id = msg.header.frame_id
        message.pose.position.x = msg.pose.position.x
        message.pose.position.y = msg.pose.position.y
        message.pose.position.z = msg.pose.position.z
        message.pose.orientation.x = msg.pose.orientation.x
        message.pose.orientation.y = msg.pose.orientation.y
        message.pose.orientation.z = msg.pose.orientation.z
        message.pose.orientation.w = msg.pose.orientation.w
        return message

    def _global_delta_to_pose_frame(
        self,
        delta_global: np.ndarray,
        source_frame: str,
    ) -> np.ndarray | None:
        source_frame = source_frame.strip() or self.global_frame
        if source_frame == self.global_frame:
            return np.asarray(delta_global, dtype=float)
        transform = self._lookup_transform(source_frame)
        if transform is None:
            return None
        _, rotation = transform
        return rotation.T @ np.asarray(delta_global, dtype=float)

    def _publish_assisted_pose(self, arm: str, message: PoseStamped) -> None:
        if arm == "left":
            self.left_assisted_pose_pub.publish(message)
        else:
            self.right_assisted_pose_pub.publish(message)

    def _update_ray_candidate(self) -> None:
        if not self.assisted_grasp_enabled or not self.ray_selection_enabled:
            return
        arms = ("left", "right") if self.bimanual_target_selection else (self.active_arm,)
        if self.latest_pointcloud is None or self.latest_pointcloud_time is None:
            for arm in arms:
                if self.targets[arm] is None:
                    self._expire_ray_candidate(arm)
            return
        now = self.get_clock().now()
        cloud_age = (now - self.latest_pointcloud_time).nanoseconds * 1e-9
        if cloud_age > self.pointcloud_timeout_s:
            for arm in arms:
                if self.targets[arm] is None:
                    self._expire_ray_candidate(arm)
            return
        if self.last_ray_update_time is not None:
            update_age = (now - self.last_ray_update_time).nanoseconds * 1e-9
            if update_age < 1.0 / self.ray_update_hz:
                return
        self.last_ray_update_time = now

        points = self._pointcloud_points_in_global(self.latest_pointcloud)
        if points.shape[0] == 0:
            for arm in arms:
                if self.targets[arm] is None:
                    self._expire_ray_candidate(arm)
            return

        for arm in arms:
            self._update_arm_ray_candidate(arm, points, now)

    def _update_arm_ray_candidate(
        self,
        arm: str,
        points: np.ndarray,
        now,
    ) -> None:
        actual = self.actual[arm]
        if actual is None:
            if self.targets[arm] is None:
                self._expire_ray_candidate(arm)
            return
        if self.targets[arm] is not None:
            self._track_locked_target_from_points(arm, points, now)
            return
        if self.target_selection_mode == "nearest_tcp":
            self._update_nearest_tcp_candidate(arm, points, now)
            return
        if self.prefer_aim_reticle_target and self.use_aim_reticle_as_fallback_target:
            candidate = self._scene_point_under_aim_reticle_from_points(arm, points)
        else:
            direction = actual.rotation @ self.ray_axis
            direction_norm = float(np.linalg.norm(direction))
            if not np.isfinite(direction_norm) or direction_norm < 1e-9:
                if self.targets[arm] is None:
                    self._expire_ray_candidate(arm)
                return
            direction /= direction_norm
            candidate = self._ray_hit(points, actual.position, direction)
        if candidate is None:
            self._expire_ray_candidate(arm)
            return
        previous = self.candidates[arm]
        if previous is None or float(np.linalg.norm(candidate - previous)) > 0.12:
            self.candidates[arm] = candidate
        else:
            alpha = self.candidate_ema_alpha
            self.candidates[arm] = (1.0 - alpha) * previous + alpha * candidate
        self.last_candidate_times[arm] = now

    def _update_nearest_tcp_candidate(
        self,
        arm: str,
        points: np.ndarray,
        now,
    ) -> None:
        if self.gripper_states[arm] != "open":
            self.candidates[arm] = None
            self.last_candidate_times[arm] = None
            self.auto_candidate_start_times[arm] = None
            return

        candidate = self._nearest_tcp_scene_point(arm, points)
        if candidate is None:
            self._expire_ray_candidate(arm)
            self.auto_candidate_start_times[arm] = None
            return

        previous = self.candidates[arm]
        if (
            previous is None
            or float(np.linalg.norm(candidate - previous)) > self.nearest_target_max_jitter
        ):
            self.candidates[arm] = candidate
            self.auto_candidate_start_times[arm] = now
        else:
            alpha = self.candidate_ema_alpha
            self.candidates[arm] = (1.0 - alpha) * previous + alpha * candidate
            if self.auto_candidate_start_times[arm] is None:
                self.auto_candidate_start_times[arm] = now
        self.last_candidate_times[arm] = now

        stable_since = self.auto_candidate_start_times[arm]
        if stable_since is None:
            return
        stable_age = (now - stable_since).nanoseconds * 1e-9
        if stable_age >= self.nearest_target_lock_delay_s:
            locked_target = self.candidates[arm]
            if locked_target is not None:
                self._lock_target(
                    arm,
                    locked_target,
                    apply_lock_offset=False,
                )

    def _nearest_tcp_scene_point(
        self,
        arm: str,
        points: np.ndarray,
    ) -> np.ndarray | None:
        actual = self.actual[arm]
        if actual is None:
            return None
        scene_points = np.asarray(points, dtype=float).reshape(-1, 3)
        if scene_points.shape[0] == 0:
            return None

        relative = scene_points - actual.position.reshape(1, 3)
        distances = np.linalg.norm(relative, axis=1)
        inside = distances >= self.nearest_target_min_distance
        if self.nearest_target_forward_only:
            direction = actual.rotation @ self.ray_axis
            direction_norm = float(np.linalg.norm(direction))
            if not np.isfinite(direction_norm) or direction_norm < 1e-9:
                return None
            direction /= direction_norm
            axial = relative @ direction
            radial = relative - axial.reshape(-1, 1) * direction.reshape(1, 3)
            radial_distance = np.linalg.norm(radial, axis=1)
            inside &= (
                (axial >= self.nearest_target_min_forward_distance)
                & (axial <= self.nearest_target_radius)
                & (radial_distance <= self.nearest_target_lateral_radius)
            )
        else:
            inside &= distances <= self.nearest_target_radius
        scene_points = scene_points[inside]
        if scene_points.shape[0] < self.nearest_target_min_points:
            return None

        scene_points = self._exclude_cable_capsule_points(scene_points)
        scene_points = self._exclude_robot_self_points(
            scene_points,
            hand_radius_override=self.nearest_target_self_filter_hand_radius,
            padding_override=self.nearest_target_self_filter_padding,
            skip_hand_arm=arm,
        )
        if scene_points.shape[0] < self.nearest_target_min_points:
            return None

        distances = np.linalg.norm(
            scene_points - actual.position.reshape(1, 3),
            axis=1,
        )
        nearest_order = np.argsort(distances)
        for index in nearest_order[: min(64, nearest_order.size)]:
            candidate = scene_points[int(index)]
            support_count = int(
                np.count_nonzero(
                    np.linalg.norm(scene_points - candidate.reshape(1, 3), axis=1)
                    <= self.nearest_target_support_radius
                )
            )
            if support_count >= self.nearest_target_min_points:
                return candidate.copy()
        return None

    def _track_locked_target_from_points(
        self,
        arm: str,
        points: np.ndarray,
        now,
    ) -> None:
        target = self.targets[arm]
        lock_origin = self.target_lock_origins[arm]
        if target is None or lock_origin is None:
            self.target_drift_start_times[arm] = None
            return
        relative = points - target.reshape(1, 3)
        distances = np.linalg.norm(relative, axis=1)
        nearby_indices = np.flatnonzero(distances <= self.target_tracking_search_radius)
        if nearby_indices.size < self.target_tracking_min_points:
            self.target_drift_start_times[arm] = None
            self.target_bboxes[arm] = None
            self._publish_target_roi(
                True,
                target,
                np.empty((0, 3), dtype=np.float32),
                now,
                None,
                None,
            )
            return

        roi_points = points[nearby_indices]
        bbox_points = self._exclude_cable_capsule_points(roi_points)
        bbox_points = self._exclude_robot_self_points(bbox_points)
        if bbox_points.shape[0] < self.target_tracking_min_points:
            self.target_drift_start_times[arm] = None
            self.target_bboxes[arm] = None
            self._publish_target_roi(
                True,
                target,
                np.empty((0, 3), dtype=np.float32),
                now,
                None,
                None,
            )
            return

        bbox_min, bbox_max, _bbox_center = self._target_bbox_from_points(bbox_points)
        tracked = target.copy()
        self.target_bboxes[arm] = (bbox_min, bbox_max)
        self.candidates[arm] = tracked.copy()
        self.last_candidate_times[arm] = now
        self._publish_target_roi(True, tracked, bbox_points, now, bbox_min, bbox_max)
        self.target_drift_start_times[arm] = None

    def _exclude_cable_capsule_points(self, points: np.ndarray) -> np.ndarray:
        roi_points = np.asarray(points, dtype=float).reshape(-1, 3)
        if (
            not self.exclude_cable_capsule_points
            or roi_points.shape[0] == 0
            or not self.cable_capsules
        ):
            return roi_points
        keep = np.ones(roi_points.shape[0], dtype=bool)
        padding = self.cable_capsule_exclusion_padding
        for capsule in self.cable_capsules:
            rel = roi_points - capsule.center.reshape(1, 3)
            axial = rel @ capsule.axis
            radial = rel - axial.reshape(-1, 1) * capsule.axis.reshape(1, 3)
            radial_distance = np.linalg.norm(radial, axis=1)
            inside = (
                (np.abs(axial) <= capsule.half_length + padding)
                & (radial_distance <= capsule.radius + padding)
            )
            keep &= ~inside
        return roi_points[keep]

    def _exclude_robot_self_points(
        self,
        points: np.ndarray,
        *,
        hand_radius_override: float | None = None,
        padding_override: float | None = None,
        skip_hand_arm: str | None = None,
    ) -> np.ndarray:
        roi_points = np.asarray(points, dtype=float).reshape(-1, 3)
        if not self.exclude_robot_points or roi_points.shape[0] == 0:
            return roi_points
        segments = self._robot_self_filter_segments()
        if not segments:
            return roi_points
        keep = np.ones(roi_points.shape[0], dtype=bool)
        padding = (
            self.robot_self_filter_padding
            if padding_override is None
            else max(0.0, float(padding_override))
        )
        for segment_arm, start, end, radius, is_hand in segments:
            if is_hand and segment_arm == skip_hand_arm:
                continue
            if is_hand and hand_radius_override is not None:
                radius = max(0.005, float(hand_radius_override))
            segment = end - start
            length_squared = float(segment @ segment)
            if length_squared <= 1e-12:
                closest = np.broadcast_to(start.reshape(1, 3), roi_points.shape)
            else:
                parameter = np.clip(
                    ((roi_points - start.reshape(1, 3)) @ segment) / length_squared,
                    0.0,
                    1.0,
                )
                closest = start.reshape(1, 3) + parameter.reshape(-1, 1) * segment.reshape(1, 3)
            distance = np.linalg.norm(roi_points - closest, axis=1)
            keep &= distance > radius + padding
        return roi_points[keep]

    def _robot_self_filter_segments(
        self,
    ) -> list[tuple[str, np.ndarray, np.ndarray, float, bool]]:
        segments: list[tuple[str, np.ndarray, np.ndarray, float, bool]] = []
        for arm, joint_ids, tcp_frame_id in (
            ("left", self.left_joint_ids, self.left_frame_id),
            ("right", self.right_joint_ids, self.right_frame_id),
        ):
            if self.actual[arm] is None:
                continue
            centers = [
                np.asarray(self.data.oMi[joint_id].translation, dtype=float).copy()
                for joint_id in joint_ids
            ]
            centers.append(
                np.asarray(self.data.oMf[tcp_frame_id].translation, dtype=float).copy()
            )
            for index, (start, end) in enumerate(zip(centers[:-1], centers[1:])):
                is_hand = index >= len(centers) - 3
                radius = (
                    self.robot_self_filter_hand_radius
                    if is_hand
                    else self.robot_self_filter_link_radius
                )
                segments.append((arm, start, end, radius, is_hand))
        return segments

    def _append_robot_self_filter_cues(
        self,
        array: MarkerArray,
        stamp,
    ) -> None:
        padding = self.robot_self_filter_padding
        arm_segment_counts = {"left": 0, "right": 0}
        for arm, start, end, radius, is_hand in self._robot_self_filter_segments():
            segment_index = arm_segment_counts[arm]
            arm_segment_counts[arm] += 1
            effective_radius = radius + padding
            segment = end - start
            length = float(np.linalg.norm(segment))
            if length <= 1e-9:
                continue

            namespace = f"{arm}_robot_self_filter"
            base_id = segment_index * 3
            color = (
                (1.0, 0.68, 0.05, 0.30)
                if is_hand
                else ((0.05, 0.85, 1.0, 0.18) if arm == "left" else (0.90, 0.20, 0.85, 0.18))
            )

            cylinder = self._base_marker(namespace, base_id, Marker.CYLINDER, stamp)
            cylinder.pose.position = self._point(0.5 * (start + end))
            qx, qy, qz, qw = self._cylinder_orientation(segment)
            cylinder.pose.orientation.x = qx
            cylinder.pose.orientation.y = qy
            cylinder.pose.orientation.z = qz
            cylinder.pose.orientation.w = qw
            diameter = 2.0 * effective_radius
            cylinder.scale.x = diameter
            cylinder.scale.y = diameter
            cylinder.scale.z = length
            self._set_color(cylinder, color)
            array.markers.append(cylinder)

            for endpoint_id, endpoint in enumerate((start, end), start=1):
                sphere = self._base_marker(
                    namespace,
                    base_id + endpoint_id,
                    Marker.SPHERE,
                    stamp,
                )
                sphere.pose.position = self._point(endpoint)
                sphere.pose.orientation.w = 1.0
                sphere.scale.x = diameter
                sphere.scale.y = diameter
                sphere.scale.z = diameter
                self._set_color(sphere, color)
                array.markers.append(sphere)

    @staticmethod
    def _cylinder_orientation(
        direction: np.ndarray,
    ) -> tuple[float, float, float, float]:
        z_axis = np.asarray(direction, dtype=float).reshape(3)
        z_axis /= np.linalg.norm(z_axis)
        reference = (
            np.array([1.0, 0.0, 0.0])
            if abs(float(z_axis[0])) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        x_axis = np.cross(reference, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        rotation = np.column_stack((x_axis, y_axis, z_axis))
        return _xyzw_from_rotation(rotation)

    def _publish_empty_target_roi(self) -> None:
        self._publish_target_roi(
            False,
            None,
            np.empty((0, 3), dtype=np.float32),
            self.get_clock().now(),
            None,
            None,
        )

    def _publish_target_roi(
        self,
        active: bool,
        center: np.ndarray | None,
        points: np.ndarray,
        now,
        bbox_min: np.ndarray | None,
        bbox_max: np.ndarray | None,
    ) -> None:
        self.target_roi_active_pub.publish(Bool(data=bool(active)))
        header = Header()
        header.stamp = now.to_msg()
        header.frame_id = self.global_frame
        roi_points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if roi_points.shape[0] > self.target_roi_max_points:
            stride = int(np.ceil(roi_points.shape[0] / self.target_roi_max_points))
            roi_points = roi_points[::stride]
        self.target_roi_points_pub.publish(
            point_cloud2.create_cloud_xyz32(header, roi_points.tolist())
        )
        self._publish_target_bbox_marker(active, center, bbox_min, bbox_max, now)

    def _target_bbox_from_points(
        self,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        roi_points = np.asarray(points, dtype=float).reshape(-1, 3)
        bbox_min = np.min(roi_points, axis=0) - self.target_bbox_padding
        bbox_max = np.max(roi_points, axis=0) + self.target_bbox_padding
        center = 0.5 * (bbox_min + bbox_max)
        size = bbox_max - bbox_min
        min_size = np.full(3, self.target_bbox_min_size, dtype=float)
        too_small = size < min_size
        if np.any(too_small):
            half = 0.5 * np.maximum(size, min_size)
            bbox_min = center - half
            bbox_max = center + half
        return bbox_min, bbox_max, center

    def _publish_target_bbox_marker(
        self,
        active: bool,
        center: np.ndarray | None,
        bbox_min: np.ndarray | None,
        bbox_max: np.ndarray | None,
        now,
    ) -> None:
        markers = MarkerArray()
        delete = Marker()
        delete.action = Marker.DELETEALL
        markers.markers.append(delete)
        if active and center is not None and bbox_min is not None and bbox_max is not None:
            marker = self._base_marker("target_bbox", 0, Marker.CUBE, now.to_msg())
            bbox_center = 0.5 * (np.asarray(bbox_min, dtype=float) + np.asarray(bbox_max, dtype=float))
            marker.pose.position = self._point(bbox_center)
            marker.pose.orientation.w = 1.0
            size = np.maximum(np.asarray(bbox_max) - np.asarray(bbox_min), self.target_bbox_min_size)
            marker.scale.x = float(size[0])
            marker.scale.y = float(size[1])
            marker.scale.z = float(size[2])
            self._set_color(marker, (0.05, 0.65, 1.0, 0.22))
            markers.markers.append(marker)
        self.target_bbox_pub.publish(markers)

    def _expire_ray_candidate(self, arm: str) -> None:
        timestamp = self.last_candidate_times[arm]
        if timestamp is None:
            self.candidates[arm] = None
            return
        age = (self.get_clock().now() - timestamp).nanoseconds * 1e-9
        if age > self.candidate_hold_s:
            self.candidates[arm] = None
            self.last_candidate_times[arm] = None
            self.auto_candidate_start_times[arm] = None

    def _pointcloud_points_in_global(self, msg: PointCloud2) -> np.ndarray:
        total_points = int(msg.width) * int(msg.height)
        sample_indices = None
        if total_points > self.pointcloud_max_points:
            stride = int(np.ceil(total_points / self.pointcloud_max_points))
            sample_indices = np.arange(0, total_points, stride, dtype=np.int64)
        raw = point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=False,
            uvs=sample_indices,
        )
        if isinstance(raw, np.ndarray):
            if raw.dtype.names:
                points = np.column_stack([raw[name] for name in ("x", "y", "z")])
            else:
                points = np.asarray(raw, dtype=float).reshape(-1, 3)
        else:
            points = np.asarray(list(raw), dtype=float).reshape(-1, 3)
        if points.shape[0] == 0:
            return np.empty((0, 3), dtype=float)
        points = np.asarray(points, dtype=float)
        points = points[np.all(np.isfinite(points), axis=1)]
        source_frame = msg.header.frame_id.strip() or self.global_frame
        if source_frame == self.global_frame:
            return points
        transform = self._lookup_transform(source_frame)
        if transform is None:
            return np.empty((0, 3), dtype=float)
        translation, rotation = transform
        return points @ rotation.T + translation

    def _ray_hit(
        self,
        points: np.ndarray,
        origin: np.ndarray,
        direction: np.ndarray,
    ) -> np.ndarray | None:
        relative = points - origin.reshape(1, 3)
        axial = relative @ direction
        radial_squared = np.einsum("ij,ij->i", relative, relative) - axial * axial
        mask = (
            (axial >= self.ray_min_distance)
            & (axial <= self.ray_length)
            & (radial_squared <= self.ray_hit_radius * self.ray_hit_radius)
        )
        indices = np.flatnonzero(mask)
        if indices.size < self.ray_min_bin_points:
            return None
        candidate_axial = axial[indices]
        bins = np.floor(candidate_axial / self.ray_bin_size).astype(np.int64)
        occupied, counts = np.unique(bins, return_counts=True)
        valid_bins = occupied[counts >= self.ray_min_bin_points]
        if valid_bins.size == 0:
            return None
        first_bin = int(np.min(valid_bins))
        bin_indices = indices[bins == first_bin]
        best_local = int(np.argmin(axial[bin_indices]))
        return points[int(bin_indices[best_local])].copy()

    def _update_actual_poses(self) -> None:
        left_ready = all(name in self.joint_values for name in self.left_joint_names)
        right_ready = all(name in self.joint_values for name in self.right_joint_names)
        if not left_ready and not right_ready:
            return
        q = self.q.copy()
        if left_ready:
            for name, index in zip(self.left_joint_names, self.left_q_indices):
                q[index] = self.joint_values[name]
        if right_ready:
            for name, index in zip(self.right_joint_names, self.right_q_indices):
                q[index] = self.joint_values[name]
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        if left_ready:
            self.actual["left"] = self._pose_from_frame(self.left_frame_id)
        if right_ready:
            self.actual["right"] = self._pose_from_frame(self.right_frame_id)

    def _pose_from_frame(self, frame_id: int) -> PoseData:
        placement = self.data.oMf[frame_id]
        return PoseData(
            np.asarray(placement.translation, dtype=float).copy(),
            np.asarray(placement.rotation, dtype=float).copy(),
        )

    def _pose_to_global(self, msg: PoseStamped) -> PoseData | None:
        source_frame = msg.header.frame_id.strip() or self.global_frame
        source_pose = PoseData(
            np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                dtype=float,
            ),
            _rotation_from_xyzw(msg.pose.orientation),
        )
        if source_frame == self.global_frame:
            return source_pose
        transform = self._lookup_transform(source_frame)
        if transform is None:
            return None
        translation, rotation = transform
        return PoseData(
            rotation @ source_pose.position + translation,
            rotation @ source_pose.rotation,
        )

    def _point_to_global(self, msg: PointStamped) -> np.ndarray | None:
        source_frame = msg.header.frame_id.strip() or self.global_frame
        point = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=float)
        if source_frame == self.global_frame:
            return point
        transform = self._lookup_transform(source_frame)
        if transform is None:
            return None
        translation, rotation = transform
        return rotation @ point + translation

    def _lookup_transform(self, source_frame: str) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.global_frame,
                source_frame,
                Time(),
                timeout=RclpyDuration(seconds=0.05),
            ).transform
        except TransformException as exc:
            self.get_logger().warn(
                f"Cannot transform {source_frame} -> {self.global_frame}: {exc}",
                throttle_duration_sec=2.0,
            )
            return None
        translation = np.array(
            [transform.translation.x, transform.translation.y, transform.translation.z],
            dtype=float,
        )
        return translation, _rotation_from_xyzw(transform.rotation)

    def _append_axes(
        self,
        array: MarkerArray,
        pose: PoseData,
        stamp,
        base_id: int,
        namespace: str,
        alpha: float = 1.0,
    ) -> None:
        colors = ((1.0, 0.05, 0.05, alpha), (0.05, 0.90, 0.10, alpha), (0.05, 0.25, 1.0, alpha))
        for axis, color in enumerate(colors):
            end = pose.position + self.axis_length * pose.rotation[:, axis]
            marker = self._arrow_marker(
                namespace,
                base_id + axis,
                pose.position,
                end,
                color,
                stamp,
                shaft=0.009,
                head=0.018,
            )
            array.markers.append(marker)

    def _append_tracking_cue(
        self,
        array: MarkerArray,
        arm: str,
        start: np.ndarray,
        end: np.ndarray,
        stamp,
        base_id: int,
    ) -> None:
        error = float(np.linalg.norm(end - start))
        color = CYAN if arm == "left" else MAGENTA
        if self.show_tracking_line and error >= self.tracking_line_min_error:
            array.markers.append(
                self._line_marker(f"{arm}_tracking", base_id, start, end, color, stamp, 0.007)
            )
        if self.show_tracking_text:
            midpoint = 0.5 * (start + end) + np.array([0.0, 0.0, 0.035])
            array.markers.append(
                self._text_marker(
                    f"{arm}_tracking",
                    base_id + 1,
                    midpoint,
                    f"{error * 100.0:.1f} cm",
                    color,
                    stamp,
                    0.035,
                )
            )

    def _append_clearance_cue(
        self,
        array: MarkerArray,
        arm: str,
        position: np.ndarray,
        stamp,
        base_id: int,
    ) -> None:
        clearance = self._fresh_clearance(arm)
        color = self._clearance_color(clearance)
        halo = self._base_marker(f"{arm}_clearance", base_id, Marker.SPHERE, stamp)
        halo.pose.position = self._point(position)
        halo.pose.orientation.w = 1.0
        halo.scale.x = 0.075
        halo.scale.y = 0.075
        halo.scale.z = 0.075
        self._set_color(halo, (color[0], color[1], color[2], 0.28))
        array.markers.append(halo)

        if self.show_clearance_text:
            label_position = position + np.array([0.0, 0.0, 0.08])
            label = "ESDF --" if clearance is None else f"ESDF {clearance * 100.0:.1f} cm"
            array.markers.append(
                self._text_marker(
                    f"{arm}_clearance",
                    base_id + 1,
                    label_position,
                    label,
                    color,
                    stamp,
                    0.032,
                )
            )

    def _append_target_cue(
        self,
        array: MarkerArray,
        arm: str,
        actual: np.ndarray,
        target: np.ndarray,
        stamp,
        base_id: int,
    ) -> None:
        distance = float(np.linalg.norm(target - actual))
        ready = distance < self.ready_distance
        color = GREEN if ready else YELLOW

        boundary = self._base_marker(f"{arm}_target", base_id, Marker.SPHERE, stamp)
        boundary.pose.position = self._point(target)
        boundary.pose.orientation.w = 1.0
        diameter = 2.0 * self.ready_distance
        boundary.scale.x = diameter
        boundary.scale.y = diameter
        boundary.scale.z = diameter
        self._set_color(boundary, (color[0], color[1], color[2], 0.16))
        array.markers.append(boundary)

        target_marker = self._base_marker(f"{arm}_target", base_id + 1, Marker.SPHERE, stamp)
        target_marker.pose.position = self._point(target)
        target_marker.pose.orientation.w = 1.0
        target_marker.scale.x = 0.025
        target_marker.scale.y = 0.025
        target_marker.scale.z = 0.025
        self._set_color(target_marker, color)
        array.markers.append(target_marker)

        if self.show_target_text:
            text_position = target + np.array([0.0, 0.0, 0.055])
            state = "CLOSE GRIPPER" if ready else f"{distance * 100.0:.1f} cm"
            array.markers.append(
                self._text_marker(
                    f"{arm}_target",
                    base_id + 3,
                    text_position,
                    state,
                    color,
                    stamp,
                    0.040,
                )
            )

    def _append_target_roi_cue(
        self,
        array: MarkerArray,
        arm: str,
        stamp,
        base_id: int,
    ) -> None:
        bbox = self.target_bboxes[arm]
        if bbox is None:
            return
        bbox_min, bbox_max = bbox
        center = 0.5 * (bbox_min + bbox_max)
        size = np.maximum(np.asarray(bbox_max) - np.asarray(bbox_min), self.target_bbox_min_size)
        roi = self._base_marker(f"{arm}_target_roi", base_id, Marker.CUBE, stamp)
        roi.pose.position = self._point(center)
        roi.pose.orientation.w = 1.0
        roi.scale.x = float(size[0])
        roi.scale.y = float(size[1])
        roi.scale.z = float(size[2])
        self._set_color(roi, (0.05, 0.65, 1.0, 0.22))
        array.markers.append(roi)

    def _append_ray_selection_cue(
        self,
        array: MarkerArray,
        arm: str,
        actual: PoseData,
        candidate: np.ndarray | None,
        stamp,
        base_id: int,
    ) -> None:
        if self.targets[arm] is not None:
            return
        direction = actual.rotation @ self.ray_axis
        direction_norm = float(np.linalg.norm(direction))
        if not np.isfinite(direction_norm) or direction_norm < 1e-9:
            return
        direction /= direction_norm
        ray_end = (
            candidate
            if candidate is not None
            else actual.position + self.ray_length * direction
        )
        color = YELLOW if candidate is not None else GREY
        array.markers.append(
            self._line_marker(
                f"{arm}_selection_ray",
                base_id,
                actual.position,
                ray_end,
                color,
                stamp,
                0.006,
            )
        )
        if candidate is None:
            return
        reticle = self._base_marker(
            f"{arm}_selection_ray",
            base_id + 1,
            Marker.SPHERE,
            stamp,
        )
        reticle.pose.position = self._point(candidate)
        reticle.pose.orientation.w = 1.0
        reticle.scale.x = 0.035
        reticle.scale.y = 0.035
        reticle.scale.z = 0.035
        self._set_color(reticle, YELLOW)
        array.markers.append(reticle)

    def _fresh_clearance(self, arm: str) -> float | None:
        timestamp = self.clearance_times[arm]
        if timestamp is None:
            return None
        age = (self.get_clock().now() - timestamp).nanoseconds * 1e-9
        if age > self.clearance_timeout_s:
            return None
        return self.clearances[arm]

    def _clearance_color(self, clearance: float | None) -> Color:
        if clearance is None:
            return GREY
        if clearance <= self.safety_margin:
            return RED
        if clearance <= self.activation_margin:
            return YELLOW
        return GREEN

    def _publish_selected_target(self, arm: str, position: np.ndarray) -> None:
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.global_frame
        message.pose.position = self._point(position)
        message.pose.orientation.w = 1.0
        if arm == "left":
            self.left_target_pub.publish(message)
        else:
            self.right_target_pub.publish(message)

    def _base_marker(self, namespace: str, marker_id: int, marker_type: int, stamp) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.lifetime = Duration(sec=0, nanosec=0)
        return marker

    def _arrow_marker(
        self,
        namespace: str,
        marker_id: int,
        start: np.ndarray,
        end: np.ndarray,
        color: Color,
        stamp,
        shaft: float,
        head: float,
    ) -> Marker:
        marker = self._base_marker(namespace, marker_id, Marker.ARROW, stamp)
        marker.points = [self._point(start), self._point(end)]
        marker.scale.x = shaft
        marker.scale.y = head
        marker.scale.z = max(0.02, 1.5 * head)
        self._set_color(marker, color)
        return marker

    def _line_marker(
        self,
        namespace: str,
        marker_id: int,
        start: np.ndarray,
        end: np.ndarray,
        color: Color,
        stamp,
        width: float,
    ) -> Marker:
        marker = self._base_marker(namespace, marker_id, Marker.LINE_STRIP, stamp)
        marker.points = [self._point(start), self._point(end)]
        marker.scale.x = width
        self._set_color(marker, color)
        return marker

    def _text_marker(
        self,
        namespace: str,
        marker_id: int,
        position: np.ndarray,
        text: str,
        color: Color,
        stamp,
        height: float,
    ) -> Marker:
        marker = self._base_marker(namespace, marker_id, Marker.TEXT_VIEW_FACING, stamp)
        marker.pose.position = self._point(position)
        marker.pose.orientation.w = 1.0
        marker.scale.z = height
        marker.text = text
        self._set_color(marker, color)
        return marker

    @staticmethod
    def _point(values: np.ndarray) -> Point:
        return Point(x=float(values[0]), y=float(values[1]), z=float(values[2]))

    @staticmethod
    def _set_color(marker: Marker, color: Color) -> None:
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])

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
    node = BimanualVisualCues()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception:  # noqa: BLE001
        node.get_logger().error(
            "Unhandled visual-cues exception:\n" + traceback.format_exc()
        )
        raise
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
