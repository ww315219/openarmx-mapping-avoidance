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
from sensor_msgs.msg import CameraInfo, Image, JointState


LEFT_JOINT_NAMES = [f"openarmx_left_joint{i}" for i in range(1, 8)]
RIGHT_JOINT_NAMES = [f"openarmx_right_joint{i}" for i in range(1, 8)]


def _sensor_qos(depth: int = 5) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


@dataclass(frozen=True)
class CollisionCapsule:
    frame: str
    start: np.ndarray
    end: np.ndarray
    radius: float


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


class SemanticObstacleDepthFilterNode(Node):
    """Remove robot body and optional target mask from the depth stream before nvblox."""

    def __init__(self) -> None:
        super().__init__("semantic_obstacle_depth_filter")
        self.depth_callback_group = MutuallyExclusiveCallbackGroup()
        self.joint_state_callback_group = MutuallyExclusiveCallbackGroup()
        self.metadata_callback_group = MutuallyExclusiveCallbackGroup()
        self.robot_state_lock = threading.Lock()

        self.declare_parameter("input_depth_topic", "/foundation_stereo/depth")
        self.declare_parameter("input_camera_info_topic", "/camera/infra1/camera_info")
        self.declare_parameter("output_depth_topic", "/perception/obstacle_depth")
        self.declare_parameter("output_camera_info_topic", "/perception/obstacle_depth/camera_info")
        self.declare_parameter("output_robot_mask_topic", "/perception/robot_body_mask")
        self.declare_parameter("output_combined_mask_topic", "/perception/semantic_obstacle_removed_mask")
        self.declare_parameter("target_mask_topic", "")

        self.declare_parameter("urdf_path", "")
        self.declare_parameter("robot_description_node", "/robot_state_publisher")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("global_frame", "world")
        self.declare_parameter("left_joint_names", LEFT_JOINT_NAMES)
        self.declare_parameter("right_joint_names", RIGHT_JOINT_NAMES)

        self.declare_parameter("enable_robot_mask", True)
        self.declare_parameter("enable_target_mask", False)
        self.declare_parameter("robot_mask_padding_px", 8)
        self.declare_parameter("robot_radius_scale", 1.15)
        self.declare_parameter("min_project_depth_m", 0.05)
        self.declare_parameter("max_project_depth_m", 5.0)
        self.declare_parameter("max_mask_thickness_px", 140)
        self.declare_parameter("pass_through_on_missing_robot_state", True)
        self.declare_parameter("debug_log_period_s", 2.0)

        self.input_depth_topic = str(self.get_parameter("input_depth_topic").value)
        self.input_camera_info_topic = str(self.get_parameter("input_camera_info_topic").value)
        self.output_depth_topic = str(self.get_parameter("output_depth_topic").value)
        self.output_camera_info_topic = str(self.get_parameter("output_camera_info_topic").value)
        self.target_mask_topic = str(self.get_parameter("target_mask_topic").value).strip()
        self.urdf_path = str(self.get_parameter("urdf_path").value).strip()
        self.robot_description_node = str(self.get_parameter("robot_description_node").value)
        self.joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self.global_frame = str(self.get_parameter("global_frame").value)
        self.left_joint_names = list(self.get_parameter("left_joint_names").value)
        self.right_joint_names = list(self.get_parameter("right_joint_names").value)
        self.joint_names = self.left_joint_names + self.right_joint_names
        self.enable_robot_mask = bool(self.get_parameter("enable_robot_mask").value)
        self.enable_target_mask = bool(self.get_parameter("enable_target_mask").value)
        self.robot_mask_padding_px = max(0, int(self.get_parameter("robot_mask_padding_px").value))
        self.robot_radius_scale = max(0.0, float(self.get_parameter("robot_radius_scale").value))
        self.min_project_depth_m = max(1e-3, float(self.get_parameter("min_project_depth_m").value))
        self.max_project_depth_m = max(
            self.min_project_depth_m,
            float(self.get_parameter("max_project_depth_m").value),
        )
        self.max_mask_thickness_px = max(1, int(self.get_parameter("max_mask_thickness_px").value))
        self.pass_through_on_missing_robot_state = bool(
            self.get_parameter("pass_through_on_missing_robot_state").value
        )
        self.debug_log_period_s = max(0.2, float(self.get_parameter("debug_log_period_s").value))

        self.bridge = CvBridge()
        self.last_camera_info: Optional[CameraInfo] = None
        self.last_target_mask: Optional[np.ndarray] = None
        self.have_joint_state = False
        self.last_log_time = self.get_clock().now()
        self.frame_count = 0
        self.last_robot_mask_pixels = 0
        self.last_target_mask_pixels = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self,
            spin_thread=False,
        )

        self._temp_urdf_path: str | None = None
        self.model = None
        self.data = None
        self.q_model_current = None
        self.joint_q_indices: list[int] = []
        self.capsules: list[CollisionCapsule] = []
        self.capsule_frame_ids: list[int] = []
        if self.enable_robot_mask:
            self._init_robot_model()

        self.depth_pub = self.create_publisher(Image, self.output_depth_topic, _sensor_qos())
        self.info_pub = self.create_publisher(
            CameraInfo,
            self.output_camera_info_topic,
            _sensor_qos(),
        )
        self.robot_mask_pub = self.create_publisher(
            Image,
            str(self.get_parameter("output_robot_mask_topic").value),
            _sensor_qos(),
        )
        self.combined_mask_pub = self.create_publisher(
            Image,
            str(self.get_parameter("output_combined_mask_topic").value),
            _sensor_qos(),
        )

        self.create_subscription(
            Image,
            self.input_depth_topic,
            self._depth_cb,
            _sensor_qos(),
            callback_group=self.depth_callback_group,
        )
        self.create_subscription(
            CameraInfo,
            self.input_camera_info_topic,
            self._camera_info_cb,
            _sensor_qos(),
            callback_group=self.metadata_callback_group,
        )
        if self.enable_robot_mask:
            self.create_subscription(
                JointState,
                self.joint_states_topic,
                self._joint_state_cb,
                20,
                callback_group=self.joint_state_callback_group,
            )
        if self.enable_target_mask and self.target_mask_topic:
            self.create_subscription(
                Image,
                self.target_mask_topic,
                self._target_mask_cb,
                _sensor_qos(),
                callback_group=self.metadata_callback_group,
            )

        self.get_logger().info(
            "Semantic obstacle depth filter: "
            f"{self.input_depth_topic} -> {self.output_depth_topic}, "
            f"robot_mask={self.enable_robot_mask}, target_mask={self.enable_target_mask}, "
            f"target_topic={self.target_mask_topic or 'none'}"
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
        self.joint_q_indices = self._joint_q_indices(self.joint_names)
        self.capsules = self._default_bimanual_capsules()
        self.capsule_frame_ids = [self.model.getFrameId(capsule.frame) for capsule in self.capsules]
        self.get_logger().info(
            f"Robot projection mask loaded URDF: capsules={len(self.capsules)}, joints={len(self.joint_names)}"
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

        fd, path = tempfile.mkstemp(prefix="openarmx_semantic_depth_filter_", suffix=".urdf")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as file_obj:
            file_obj.write(response.values[0].string_value)
        self._temp_urdf_path = path
        return path

    def _joint_q_indices(self, joint_names: list[str]) -> list[int]:
        assert self.model is not None
        indices = []
        for name in joint_names:
            if not self.model.existJointName(name):
                raise RuntimeError(f"Joint {name!r} not found in URDF model")
            joint = self.model.joints[self.model.getJointId(name)]
            if joint.nq != 1:
                raise RuntimeError(f"Joint {name!r} has nq={joint.nq}, expected 1")
            indices.append(int(joint.idx_q))
        return indices

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
                    self.get_logger().warn(f"Skipping robot mask capsule on missing frame {frame!r}")
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
            raise RuntimeError("No valid robot mask capsules were configured.")
        return capsules

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self.last_camera_info = msg

    def _joint_state_cb(self, msg: JointState) -> None:
        if self.model is None or self.q_model_current is None:
            return
        name_to_pos = {name: pos for name, pos in zip(msg.name, msg.position)}
        updated = False
        with self.robot_state_lock:
            for name in msg.name:
                if name in name_to_pos and self.model.existJointName(name):
                    joint = self.model.joints[self.model.getJointId(name)]
                    if joint.nq == 1:
                        self.q_model_current[joint.idx_q] = float(name_to_pos[name])
                        updated = True
        if updated:
            self.have_joint_state = True

    def _target_mask_cb(self, msg: Image) -> None:
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception:
            try:
                raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                mask = np.asarray(raw)
            except Exception as exc:
                self.get_logger().warn(f"Failed to convert target mask: {exc}")
                return
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        self.last_target_mask = (mask > 0).astype(np.uint8) * 255

    def _depth_cb(self, msg: Image) -> None:
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().warn(f"Failed to convert depth image: {exc}")
            return
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]
        depth_out = np.array(depth_raw, copy=True)
        height, width = depth_out.shape[:2]

        robot_mask = np.zeros((height, width), dtype=np.uint8)
        if self.enable_robot_mask:
            try:
                robot_mask = self._build_robot_mask(msg.header.frame_id, height, width)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(
                    f"Robot mask generation failed; publishing unmasked depth: {exc}",
                    throttle_duration_sec=2.0,
                )

        combined_mask = robot_mask.copy()
        target_mask_pixels = 0
        if self.enable_target_mask and self.last_target_mask is not None:
            target_mask = self._resize_mask_if_needed(self.last_target_mask, width, height)
            target_mask_pixels = int(np.count_nonzero(target_mask))
            combined_mask = cv2.bitwise_or(combined_mask, target_mask)

        if np.any(combined_mask):
            depth_out[combined_mask > 0] = 0

        depth_msg = self.bridge.cv2_to_imgmsg(depth_out, encoding=msg.encoding)
        depth_msg.header = msg.header
        self.depth_pub.publish(depth_msg)

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

        robot_mask_msg = self.bridge.cv2_to_imgmsg(robot_mask, encoding="mono8")
        robot_mask_msg.header = msg.header
        self.robot_mask_pub.publish(robot_mask_msg)
        combined_mask_msg = self.bridge.cv2_to_imgmsg(combined_mask, encoding="mono8")
        combined_mask_msg.header = msg.header
        self.combined_mask_pub.publish(combined_mask_msg)

        self.frame_count += 1
        self.last_robot_mask_pixels = int(np.count_nonzero(robot_mask))
        self.last_target_mask_pixels = target_mask_pixels
        self._maybe_log()

    def _resize_mask_if_needed(self, mask: np.ndarray, width: int, height: int) -> np.ndarray:
        if mask.shape[:2] == (height, width):
            return mask
        return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    def _build_robot_mask(self, camera_frame: str, height: int, width: int) -> np.ndarray:
        if (
            not camera_frame
            or self.last_camera_info is None
            or not self.have_joint_state
            or self.model is None
            or self.data is None
            or self.q_model_current is None
        ):
            return np.zeros((height, width), dtype=np.uint8)

        query_time = Time()
        if not self.tf_buffer.can_transform(camera_frame, self.global_frame, query_time):
            self.get_logger().warn(
                f"Robot mask TF unavailable {self.global_frame}->{camera_frame}",
                throttle_duration_sec=2.0,
            )
            return np.zeros((height, width), dtype=np.uint8)
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
            return np.zeros((height, width), dtype=np.uint8)

        mask = np.zeros((height, width), dtype=np.uint8)
        for capsule, frame_id in zip(self.capsules, self.capsule_frame_ids):
            placement = self.data.oMf[frame_id]
            start_world = placement.translation + placement.rotation @ capsule.start
            end_world = placement.translation + placement.rotation @ capsule.end
            start_cam = rotation @ start_world + translation
            end_cam = rotation @ end_world + translation
            self._draw_projected_capsule(
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
        return mask

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
    ) -> None:
        z0 = float(start_cam[2])
        z1 = float(end_cam[2])
        if z0 <= self.min_project_depth_m and z1 <= self.min_project_depth_m:
            return
        if z0 > self.max_project_depth_m and z1 > self.max_project_depth_m:
            return

        z0 = max(z0, self.min_project_depth_m)
        z1 = max(z1, self.min_project_depth_m)
        u0 = fx * float(start_cam[0]) / z0 + cx
        v0 = fy * float(start_cam[1]) / z0 + cy
        u1 = fx * float(end_cam[0]) / z1 + cx
        v1 = fy * float(end_cam[1]) / z1 + cy
        if not np.all(np.isfinite([u0, v0, u1, v1])):
            return

        limit = max(width, height) * 4
        if (
            max(u0, u1) < -limit
            or min(u0, u1) > width + limit
            or max(v0, v1) < -limit
            or min(v0, v1) > height + limit
        ):
            return

        z_mid = max(0.5 * (z0 + z1), self.min_project_depth_m)
        pixel_radius = int(
            round(max(fx, fy) * float(radius_m) / z_mid)
        ) + self.robot_mask_padding_px
        pixel_radius = int(np.clip(pixel_radius, 1, self.max_mask_thickness_px))
        p0 = (int(round(np.clip(u0, -limit, width + limit))), int(round(np.clip(v0, -limit, height + limit))))
        p1 = (int(round(np.clip(u1, -limit, width + limit))), int(round(np.clip(v1, -limit, height + limit))))
        thickness = max(1, 2 * pixel_radius)
        cv2.line(mask, p0, p1, 255, thickness=thickness, lineType=cv2.LINE_AA)
        cv2.circle(mask, p0, pixel_radius, 255, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(mask, p1, pixel_radius, 255, thickness=-1, lineType=cv2.LINE_AA)

    def _maybe_log(self) -> None:
        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds * 1e-9 < self.debug_log_period_s:
            return
        self.last_log_time = now
        self.get_logger().info(
            f"Semantic obstacle filter frames={self.frame_count}, "
            f"robot_mask_px={self.last_robot_mask_pixels}, "
            f"target_mask_px={self.last_target_mask_pixels}, "
            f"camera_info={self.last_camera_info is not None}, "
            f"joint_state={self.have_joint_state}"
        )


def main() -> None:
    rclpy.init()
    node = SemanticObstacleDepthFilterNode()
    executor = MultiThreadedExecutor(num_threads=3)
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
