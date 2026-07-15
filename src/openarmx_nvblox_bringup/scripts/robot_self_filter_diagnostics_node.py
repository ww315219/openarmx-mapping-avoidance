#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pinocchio as pin
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from rcl_interfaces.srv import GetParameters
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


LEFT_JOINT_NAMES = [f"openarmx_left_joint{i}" for i in range(1, 8)]
RIGHT_JOINT_NAMES = [f"openarmx_right_joint{i}" for i in range(1, 8)]


@dataclass(frozen=True)
class CollisionCapsule:
    frame: str
    start: np.ndarray
    end: np.ndarray
    radius: float


@dataclass
class ProjectedCapsule:
    frame: str
    start_world: np.ndarray
    end_world: np.ndarray
    start_cam: np.ndarray
    end_cam: np.ndarray
    radius: float
    visible: bool


def _sensor_qos(depth: int = 5) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def _latched_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


class RobotSelfFilterDiagnosticsNode(Node):
    """Visualize and score the robot self-filter projection.

    This node does not modify the map. It reuses the same approximate robot capsule
    model as semantic_obstacle_depth_filter_node.py, projects it into the depth image,
    and publishes overlays that expose whether the mask is shifted, too small, or not
    being generated because of missing TF/joint/camera information.
    """

    def __init__(self) -> None:
        super().__init__("robot_self_filter_diagnostics")
        self.depth_callback_group = MutuallyExclusiveCallbackGroup()
        self.image_callback_group = MutuallyExclusiveCallbackGroup()
        self.joint_state_callback_group = MutuallyExclusiveCallbackGroup()
        self.metadata_callback_group = MutuallyExclusiveCallbackGroup()
        self.robot_state_lock = threading.Lock()

        self.declare_parameter("depth_topic", "/foundation_stereo/depth")
        self.declare_parameter("camera_info_topic", "/camera/infra1/camera_info")
        self.declare_parameter("color_image_topic", "/camera/color/image_raw")
        self.declare_parameter("compressed_color_image_topic", "/camera/color/image_raw/compressed")
        self.declare_parameter("use_compressed_image", True)
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("robot_description_node", "/robot_state_publisher")
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("global_frame", "world")
        self.declare_parameter("left_joint_names", LEFT_JOINT_NAMES)
        self.declare_parameter("right_joint_names", RIGHT_JOINT_NAMES)

        self.declare_parameter("robot_mask_padding_px", 8)
        self.declare_parameter("robot_radius_scale", 1.15)
        self.declare_parameter("min_project_depth_m", 0.05)
        self.declare_parameter("max_project_depth_m", 5.0)
        self.declare_parameter("max_mask_thickness_px", 140)
        self.declare_parameter("near_miss_radius_px", 28)
        self.declare_parameter("valid_depth_min_m", 0.05)
        self.declare_parameter("valid_depth_max_m", 3.0)
        self.declare_parameter("overlay_alpha", 0.45)
        self.declare_parameter("jpeg_quality", 85)
        self.declare_parameter("debug_log_period_s", 1.0)

        self.declare_parameter("mask_topic", "/debug/robot_self_filter/mask")
        self.declare_parameter("near_miss_mask_topic", "/debug/robot_self_filter/near_miss_mask")
        self.declare_parameter("overlay_topic", "/debug/robot_self_filter/overlay")
        self.declare_parameter(
            "compressed_overlay_topic",
            "/debug/robot_self_filter/overlay/compressed",
        )
        self.declare_parameter("status_topic", "/debug/robot_self_filter/status")
        self.declare_parameter("marker_topic", "/debug/robot_self_filter/capsules")

        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.color_image_topic = str(self.get_parameter("color_image_topic").value)
        self.compressed_color_image_topic = str(
            self.get_parameter("compressed_color_image_topic").value
        )
        self.use_compressed_image = bool(self.get_parameter("use_compressed_image").value)
        self.joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self.robot_description_node = str(self.get_parameter("robot_description_node").value)
        self.urdf_path = str(self.get_parameter("urdf_path").value).strip()
        self.global_frame = str(self.get_parameter("global_frame").value)
        self.left_joint_names = list(self.get_parameter("left_joint_names").value)
        self.right_joint_names = list(self.get_parameter("right_joint_names").value)
        self.joint_names = self.left_joint_names + self.right_joint_names

        self.robot_mask_padding_px = max(0, int(self.get_parameter("robot_mask_padding_px").value))
        self.robot_radius_scale = max(0.0, float(self.get_parameter("robot_radius_scale").value))
        self.min_project_depth_m = max(1e-3, float(self.get_parameter("min_project_depth_m").value))
        self.max_project_depth_m = max(
            self.min_project_depth_m,
            float(self.get_parameter("max_project_depth_m").value),
        )
        self.max_mask_thickness_px = max(1, int(self.get_parameter("max_mask_thickness_px").value))
        self.near_miss_radius_px = max(1, int(self.get_parameter("near_miss_radius_px").value))
        self.valid_depth_min_m = max(0.0, float(self.get_parameter("valid_depth_min_m").value))
        self.valid_depth_max_m = max(
            self.valid_depth_min_m + 1e-6,
            float(self.get_parameter("valid_depth_max_m").value),
        )
        self.overlay_alpha = float(np.clip(float(self.get_parameter("overlay_alpha").value), 0.0, 1.0))
        self.jpeg_quality = int(np.clip(int(self.get_parameter("jpeg_quality").value), 1, 100))
        self.debug_log_period_s = max(0.2, float(self.get_parameter("debug_log_period_s").value))

        self.bridge = CvBridge()
        self.last_camera_info: Optional[CameraInfo] = None
        self.last_color: Optional[np.ndarray] = None
        self.have_joint_state = False
        self.last_log_time = self.get_clock().now()
        self.frame_count = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=False)

        self._temp_urdf_path: str | None = None
        self.model = None
        self.data = None
        self.q_model_current = None
        self.capsules: list[CollisionCapsule] = []
        self.capsule_frame_ids: list[int] = []
        self._init_robot_model()

        self.mask_pub = self.create_publisher(
            Image,
            str(self.get_parameter("mask_topic").value),
            _sensor_qos(),
        )
        self.near_miss_mask_pub = self.create_publisher(
            Image,
            str(self.get_parameter("near_miss_mask_topic").value),
            _sensor_qos(),
        )
        self.overlay_pub = self.create_publisher(
            Image,
            str(self.get_parameter("overlay_topic").value),
            _sensor_qos(),
        )
        self.compressed_overlay_pub = self.create_publisher(
            CompressedImage,
            str(self.get_parameter("compressed_overlay_topic").value),
            _sensor_qos(),
        )
        self.status_pub = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            _latched_qos(),
        )
        self.marker_pub = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("marker_topic").value),
            _sensor_qos(),
        )

        self.create_subscription(
            Image,
            self.depth_topic,
            self._depth_cb,
            _sensor_qos(),
            callback_group=self.depth_callback_group,
        )
        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._camera_info_cb,
            _sensor_qos(),
            callback_group=self.metadata_callback_group,
        )
        if self.use_compressed_image:
            self.create_subscription(
                CompressedImage,
                self.compressed_color_image_topic,
                self._compressed_color_cb,
                _sensor_qos(),
                callback_group=self.image_callback_group,
            )
        else:
            self.create_subscription(
                Image,
                self.color_image_topic,
                self._color_cb,
                _sensor_qos(),
                callback_group=self.image_callback_group,
            )
        self.create_subscription(
            JointState,
            self.joint_states_topic,
            self._joint_state_cb,
            20,
            callback_group=self.joint_state_callback_group,
        )

        self.get_logger().info(
            "Robot self-filter diagnostics: "
            f"depth={self.depth_topic}, camera_info={self.camera_info_topic}, "
            f"color={'compressed ' + self.compressed_color_image_topic if self.use_compressed_image else self.color_image_topic}, "
            f"capsules={len(self.capsules)}, radius_scale={self.robot_radius_scale:.2f}, "
            f"padding_px={self.robot_mask_padding_px}"
        )

    def destroy_node(self) -> bool:
        if self._temp_urdf_path and os.path.exists(self._temp_urdf_path):
            try:
                os.remove(self._temp_urdf_path)
            except OSError:
                pass
        return super().destroy_node()

    def _init_robot_model(self) -> None:
        if not self.urdf_path:
            self.urdf_path = self._write_robot_description_to_temp_urdf()
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        self.q_model_current = pin.neutral(self.model)
        self.capsules = self._default_bimanual_capsules()
        self.capsule_frame_ids = [self.model.getFrameId(capsule.frame) for capsule in self.capsules]

    def _write_robot_description_to_temp_urdf(self) -> str:
        client = self.create_client(GetParameters, f"{self.robot_description_node}/get_parameters")
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(
                f"Service not available: {self.robot_description_node}/get_parameters; "
                "start robot_state_publisher or pass urdf_path."
            )
        req = GetParameters.Request()
        req.names = ["robot_description"]
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        response = future.result()
        if response is None or not response.values or not response.values[0].string_value:
            raise RuntimeError("Failed to read robot_description from robot_state_publisher.")

        fd, path = tempfile.mkstemp(prefix="openarmx_robot_self_filter_diag_", suffix=".urdf")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as file_obj:
            file_obj.write(response.values[0].string_value)
        self._temp_urdf_path = path
        return path

    def _default_bimanual_capsules(self) -> list[CollisionCapsule]:
        assert self.model is not None
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
        capsules: list[CollisionCapsule] = []
        for side in ("left", "right"):
            for suffix, start, end, radius in raw_one_arm_capsules:
                frame = f"openarmx_{side}_{suffix}"
                if not self.model.existFrame(frame):
                    self.get_logger().warn(f"Skipping diagnostic capsule on missing frame {frame!r}")
                    continue
                capsules.append(
                    CollisionCapsule(
                        frame=frame,
                        start=np.asarray(start, dtype=float).reshape(3),
                        end=np.asarray(end, dtype=float).reshape(3),
                        radius=float(radius),
                    )
                )
        if not capsules:
            raise RuntimeError("No valid diagnostic capsules were configured.")
        return capsules

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self.last_camera_info = msg

    def _color_cb(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"Failed to convert color image: {exc}", throttle_duration_sec=2.0)
            return
        self.last_color = np.asarray(image, dtype=np.uint8)

    def _compressed_color_cb(self, msg: CompressedImage) -> None:
        array = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().warn("Failed to decode compressed color image", throttle_duration_sec=2.0)
            return
        self.last_color = image

    def _joint_state_cb(self, msg: JointState) -> None:
        if self.model is None or self.q_model_current is None:
            return
        name_to_pos = {name: pos for name, pos in zip(msg.name, msg.position)}
        updated = False
        with self.robot_state_lock:
            for name, position in name_to_pos.items():
                if self.model.existJointName(name):
                    joint = self.model.joints[self.model.getJointId(name)]
                    if joint.nq == 1:
                        self.q_model_current[joint.idx_q] = float(position)
                        updated = True
        if updated:
            self.have_joint_state = True

    def _depth_cb(self, msg: Image) -> None:
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().warn(f"Failed to convert depth image: {exc}")
            return
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]
        depth_m = self._depth_to_meters(np.asarray(depth_raw), msg.encoding)
        height, width = depth_m.shape[:2]

        robot_mask, projected_capsules, reason = self._build_robot_mask(
            msg.header.frame_id,
            height,
            width,
        )
        valid_depth = (
            np.isfinite(depth_m)
            & (depth_m >= self.valid_depth_min_m)
            & (depth_m <= self.valid_depth_max_m)
        )
        if robot_mask.shape != valid_depth.shape:
            self.get_logger().warn(
                f"Mask/depth shape mismatch mask={robot_mask.shape} depth={valid_depth.shape}",
                throttle_duration_sec=2.0,
            )
            return

        inside_mask = valid_depth & (robot_mask > 0)
        expanded = cv2.dilate(
            robot_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * self.near_miss_radius_px + 1, 2 * self.near_miss_radius_px + 1),
            ),
        )
        near_miss = valid_depth & (robot_mask == 0) & (expanded > 0)

        mask_msg = self.bridge.cv2_to_imgmsg(robot_mask, encoding="mono8")
        mask_msg.header = msg.header
        self.mask_pub.publish(mask_msg)

        near_miss_mask = np.zeros_like(robot_mask)
        near_miss_mask[near_miss] = 255
        near_msg = self.bridge.cv2_to_imgmsg(near_miss_mask, encoding="mono8")
        near_msg.header = msg.header
        self.near_miss_mask_pub.publish(near_msg)

        overlay = self._build_overlay(width, height, valid_depth, robot_mask, near_miss)
        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        overlay_msg.header = msg.header
        self.overlay_pub.publish(overlay_msg)
        self._publish_compressed_overlay(overlay, msg.header)
        self._publish_capsule_markers(projected_capsules, msg.header.stamp)

        status = self._status_string(
            reason,
            valid_depth,
            robot_mask,
            inside_mask,
            near_miss,
            projected_capsules,
        )
        self.status_pub.publish(String(data=status))

        self.frame_count += 1
        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds * 1e-9 >= self.debug_log_period_s:
            self.last_log_time = now
            self.get_logger().info(status)

    def _depth_to_meters(self, depth: np.ndarray, encoding: str) -> np.ndarray:
        if np.issubdtype(depth.dtype, np.integer):
            if encoding in ("16UC1", "mono16"):
                return depth.astype(np.float32) * 0.001
            return depth.astype(np.float32)
        return depth.astype(np.float32)

    def _build_robot_mask(
        self,
        camera_frame: str,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, list[ProjectedCapsule], str]:
        mask = np.zeros((height, width), dtype=np.uint8)
        if not camera_frame:
            return mask, [], "missing_depth_frame"
        if self.last_camera_info is None:
            return mask, [], "missing_camera_info"
        if not self.have_joint_state:
            return mask, [], "missing_joint_state"
        if self.model is None or self.data is None or self.q_model_current is None:
            return mask, [], "missing_robot_model"

        query_time = Time()
        if not self.tf_buffer.can_transform(camera_frame, self.global_frame, query_time):
            return mask, [], f"missing_tf:{self.global_frame}->{camera_frame}"
        tf = self.tf_buffer.lookup_transform(camera_frame, self.global_frame, query_time)
        transform = tf.transform
        rotation = _quat_to_matrix(
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        )
        translation = np.array(
            [transform.translation.x, transform.translation.y, transform.translation.z],
            dtype=float,
        )

        with self.robot_state_lock:
            q_snapshot = np.array(self.q_model_current, copy=True)
        pin.forwardKinematics(self.model, self.data, q_snapshot)
        pin.updateFramePlacements(self.model, self.data)

        k = self.last_camera_info.k
        fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
        if fx <= 0.0 or fy <= 0.0:
            return mask, [], "invalid_camera_intrinsics"

        projected: list[ProjectedCapsule] = []
        for capsule, frame_id in zip(self.capsules, self.capsule_frame_ids):
            placement = self.data.oMf[frame_id]
            start_world = placement.translation + placement.rotation @ capsule.start
            end_world = placement.translation + placement.rotation @ capsule.end
            start_cam = rotation @ start_world + translation
            end_cam = rotation @ end_world + translation
            visible = self._draw_projected_capsule(
                mask,
                start_cam,
                end_cam,
                capsule.radius * self.robot_radius_scale,
                fx,
                fy,
                cx,
                cy,
                width,
                height,
            )
            projected.append(
                ProjectedCapsule(
                    frame=capsule.frame,
                    start_world=start_world,
                    end_world=end_world,
                    start_cam=start_cam,
                    end_cam=end_cam,
                    radius=capsule.radius * self.robot_radius_scale,
                    visible=visible,
                )
            )
        return mask, projected, "ok"

    def _draw_projected_capsule(
        self,
        mask: np.ndarray,
        start_cam: np.ndarray,
        end_cam: np.ndarray,
        radius_m: float,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
    ) -> bool:
        z0 = float(start_cam[2])
        z1 = float(end_cam[2])
        if z0 <= self.min_project_depth_m and z1 <= self.min_project_depth_m:
            return False
        if z0 > self.max_project_depth_m and z1 > self.max_project_depth_m:
            return False

        z0 = max(z0, self.min_project_depth_m)
        z1 = max(z1, self.min_project_depth_m)
        u0 = fx * float(start_cam[0]) / z0 + cx
        v0 = fy * float(start_cam[1]) / z0 + cy
        u1 = fx * float(end_cam[0]) / z1 + cx
        v1 = fy * float(end_cam[1]) / z1 + cy
        if not np.all(np.isfinite([u0, v0, u1, v1])):
            return False

        limit = max(width, height) * 4
        if (
            max(u0, u1) < -limit
            or min(u0, u1) > width + limit
            or max(v0, v1) < -limit
            or min(v0, v1) > height + limit
        ):
            return False

        z_mid = max(0.5 * (z0 + z1), self.min_project_depth_m)
        pixel_radius = int(round(max(fx, fy) * float(radius_m) / z_mid)) + self.robot_mask_padding_px
        pixel_radius = int(np.clip(pixel_radius, 1, self.max_mask_thickness_px))
        p0 = (int(round(np.clip(u0, -limit, width + limit))), int(round(np.clip(v0, -limit, height + limit))))
        p1 = (int(round(np.clip(u1, -limit, width + limit))), int(round(np.clip(v1, -limit, height + limit))))
        thickness = max(1, 2 * pixel_radius)
        cv2.line(mask, p0, p1, 255, thickness=thickness, lineType=cv2.LINE_AA)
        cv2.circle(mask, p0, pixel_radius, 255, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(mask, p1, pixel_radius, 255, thickness=-1, lineType=cv2.LINE_AA)
        return True

    def _build_overlay(
        self,
        width: int,
        height: int,
        valid_depth: np.ndarray,
        robot_mask: np.ndarray,
        near_miss: np.ndarray,
    ) -> np.ndarray:
        if self.last_color is not None:
            base = cv2.resize(self.last_color, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            base = np.zeros((height, width, 3), dtype=np.uint8)
            base[valid_depth] = (90, 90, 90)

        overlay = base.copy()
        # Green: projected robot self-filter mask.
        overlay[robot_mask > 0] = (40, 220, 40)
        # Red: valid depth points close to the mask but not covered by it.
        overlay[near_miss] = (30, 30, 255)
        blended = cv2.addWeighted(overlay, self.overlay_alpha, base, 1.0 - self.overlay_alpha, 0.0)

        contours, _ = cv2.findContours(robot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (0, 255, 0), 1)
        near_contours, _ = cv2.findContours(
            near_miss.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(blended, near_contours, -1, (0, 0, 255), 1)
        cv2.putText(
            blended,
            "green=model mask red=valid depth near but outside mask",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return blended

    def _publish_compressed_overlay(self, image: np.ndarray, header) -> None:
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return
        msg = CompressedImage()
        msg.header = header
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        self.compressed_overlay_pub.publish(msg)

    def _publish_capsule_markers(
        self,
        projected_capsules: list[ProjectedCapsule],
        stamp,
    ) -> None:
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        for index, capsule in enumerate(projected_capsules):
            color = (0.0, 0.95, 0.2, 0.28) if capsule.visible else (0.9, 0.2, 0.2, 0.16)
            self._append_capsule_marker(markers, index * 3, capsule.start_world, capsule.end_world, capsule.radius, color, stamp)
        self.marker_pub.publish(markers)

    def _append_capsule_marker(
        self,
        markers: MarkerArray,
        base_id: int,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        color: tuple[float, float, float, float],
        stamp,
    ) -> None:
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length <= 1e-9:
            return
        diameter = 2.0 * radius
        cylinder = Marker()
        cylinder.header.frame_id = self.global_frame
        cylinder.header.stamp = stamp
        cylinder.ns = "robot_self_filter_diag"
        cylinder.id = base_id
        cylinder.type = Marker.CYLINDER
        cylinder.action = Marker.ADD
        center = 0.5 * (start + end)
        cylinder.pose.position.x = float(center[0])
        cylinder.pose.position.y = float(center[1])
        cylinder.pose.position.z = float(center[2])
        qx, qy, qz, qw = self._cylinder_orientation(segment)
        cylinder.pose.orientation.x = qx
        cylinder.pose.orientation.y = qy
        cylinder.pose.orientation.z = qz
        cylinder.pose.orientation.w = qw
        cylinder.scale.x = diameter
        cylinder.scale.y = diameter
        cylinder.scale.z = length
        cylinder.color.r, cylinder.color.g, cylinder.color.b, cylinder.color.a = color
        markers.markers.append(cylinder)

        for endpoint_id, endpoint in enumerate((start, end), start=1):
            sphere = Marker()
            sphere.header.frame_id = self.global_frame
            sphere.header.stamp = stamp
            sphere.ns = "robot_self_filter_diag"
            sphere.id = base_id + endpoint_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(endpoint[0])
            sphere.pose.position.y = float(endpoint[1])
            sphere.pose.position.z = float(endpoint[2])
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = diameter
            sphere.scale.y = diameter
            sphere.scale.z = diameter
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = color
            markers.markers.append(sphere)

    def _cylinder_orientation(self, direction: np.ndarray) -> tuple[float, float, float, float]:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        vector = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-9:
            return 0.0, 0.0, 0.0, 1.0
        vector /= norm
        axis = np.cross(z_axis, vector)
        axis_norm = float(np.linalg.norm(axis))
        dot = float(np.clip(z_axis @ vector, -1.0, 1.0))
        if axis_norm <= 1e-9:
            if dot > 0.0:
                return 0.0, 0.0, 0.0, 1.0
            return 1.0, 0.0, 0.0, 0.0
        axis /= axis_norm
        angle = math.acos(dot)
        half = 0.5 * angle
        sin_half = math.sin(half)
        return (
            float(axis[0] * sin_half),
            float(axis[1] * sin_half),
            float(axis[2] * sin_half),
            float(math.cos(half)),
        )

    def _status_string(
        self,
        reason: str,
        valid_depth: np.ndarray,
        robot_mask: np.ndarray,
        inside_mask: np.ndarray,
        near_miss: np.ndarray,
        projected_capsules: list[ProjectedCapsule],
    ) -> str:
        valid_px = int(np.count_nonzero(valid_depth))
        mask_px = int(np.count_nonzero(robot_mask))
        inside_px = int(np.count_nonzero(inside_mask))
        near_px = int(np.count_nonzero(near_miss))
        visible_capsules = sum(1 for capsule in projected_capsules if capsule.visible)
        inside_ratio = inside_px / max(mask_px, 1)
        near_ratio = near_px / max(valid_px, 1)

        diagnosis: list[str] = []
        if reason != "ok":
            diagnosis.append(reason)
        if mask_px == 0:
            diagnosis.append("robot_mask_empty: check TF, joint_states, camera frame, or robot out of view")
        if visible_capsules == 0 and reason == "ok":
            diagnosis.append("no_capsule_projects_into_camera")
        if mask_px > 0 and inside_px == 0:
            diagnosis.append("mask_has_no_depth_overlap: likely TF/extrinsic or joint-state mismatch")
        if near_px > max(250, 0.02 * valid_px):
            diagnosis.append("many_near_miss_depth_pixels: mask likely too small/shifted or hand capsule incomplete")
        if not diagnosis:
            diagnosis.append("projection_available")

        return (
            f"frames={self.frame_count} reason={reason} "
            f"camera_info={self.last_camera_info is not None} joint_state={self.have_joint_state} "
            f"capsules_visible={visible_capsules}/{len(projected_capsules)} "
            f"mask_px={mask_px} valid_depth_px={valid_px} "
            f"valid_inside_mask_px={inside_px} inside_per_mask={inside_ratio:.3f} "
            f"near_miss_px={near_px} near_miss_per_valid={near_ratio:.3f} "
            f"diagnosis={'|'.join(diagnosis)}"
        )


def main() -> None:
    rclpy.init()
    node = RobotSelfFilterDiagnosticsNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
