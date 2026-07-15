#!/usr/bin/env python3
from __future__ import annotations

import rclpy
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


def _reliable_qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


class DepthFreezeGateNode(Node):
    """Forward depth for a short warmup window, then stop nvblox integration input."""

    def __init__(self) -> None:
        super().__init__("depth_freeze_gate")

        self.declare_parameter("input_depth_topic", "/foundation_stereo/depth")
        self.declare_parameter("input_camera_info_topic", "/camera/infra1/camera_info")
        self.declare_parameter("output_depth_topic", "/nvblox/frozen_depth")
        self.declare_parameter("output_camera_info_topic", "/nvblox/frozen_depth/camera_info")
        self.declare_parameter("freeze_after_s", 5.0)
        self.declare_parameter("start_on_first_depth", True)
        self.declare_parameter("wait_for_output_subscriber", True)
        self.declare_parameter("startup_stabilization_s", 2.0)
        self.declare_parameter("keep_publishing_last_frame", False)
        self.declare_parameter("last_frame_publish_hz", 1.0)
        self.declare_parameter("debug_log_period_s", 1.0)

        self.freeze_after_s = max(0.0, float(self.get_parameter("freeze_after_s").value))
        self.start_on_first_depth = bool(self.get_parameter("start_on_first_depth").value)
        self.wait_for_output_subscriber = bool(
            self.get_parameter("wait_for_output_subscriber").value
        )
        self.startup_stabilization_s = max(
            0.0,
            float(self.get_parameter("startup_stabilization_s").value),
        )
        self.keep_publishing_last_frame = bool(
            self.get_parameter("keep_publishing_last_frame").value
        )
        self.last_frame_publish_hz = max(
            0.1,
            float(self.get_parameter("last_frame_publish_hz").value),
        )
        self.debug_log_period_s = max(0.2, float(self.get_parameter("debug_log_period_s").value))

        self.start_time = None if self.start_on_first_depth else self.get_clock().now()
        self.ready_time = None
        self.frozen = False
        self.last_depth_msg: Image | None = None
        self.last_info_msg: CameraInfo | None = None
        self.last_log_time = self.get_clock().now()
        self.forwarded_depth_count = 0

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

        self.depth_sub = self.create_subscription(
            Image,
            str(self.get_parameter("input_depth_topic").value),
            self._depth_cb,
            _reliable_qos(),
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            str(self.get_parameter("input_camera_info_topic").value),
            self._camera_info_cb,
            _reliable_qos(),
        )

        self.last_frame_timer = self.create_timer(
            1.0 / self.last_frame_publish_hz,
            self._last_frame_timer_cb,
        )
        self.debug_timer = self.create_timer(
            self.debug_log_period_s,
            self._debug_timer_cb,
        )

        self.get_logger().info(
            "Depth freeze gate: "
            f"{self.get_parameter('input_depth_topic').value} -> "
            f"{self.get_parameter('output_depth_topic').value}, "
            f"freeze_after={self.freeze_after_s:.2f}s, "
            f"keep_last={self.keep_publishing_last_frame}"
        )

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if not self.frozen:
            self.last_info_msg = msg

    def _depth_cb(self, msg: Image) -> None:
        now = self.get_clock().now()
        output_ready = (
            not self.wait_for_output_subscriber
            or self.depth_pub.get_subscription_count() > 0
        )
        if self.last_info_msg is None or not output_ready:
            return

        if self.ready_time is None:
            self.ready_time = now
            self.get_logger().info(
                "Depth input, camera info, and nvblox subscriber are ready; "
                f"stabilizing for {self.startup_stabilization_s:.2f}s."
            )
            return

        ready_elapsed = (now - self.ready_time).nanoseconds * 1e-9
        if ready_elapsed < self.startup_stabilization_s:
            return

        if self.start_time is None:
            self.start_time = now
            self.get_logger().info(
                "Nvblox input window started after all inputs became ready."
            )

        elapsed = (now - self.start_time).nanoseconds * 1e-9
        if elapsed >= self.freeze_after_s:
            if not self.frozen:
                self.frozen = True
                self.get_logger().info(
                    "Depth stream frozen for nvblox: "
                    f"forwarded={self.forwarded_depth_count} frames over {elapsed:.2f}s."
                )
            return

        self.last_depth_msg = msg
        self.depth_pub.publish(msg)
        if self.last_info_msg is not None:
            info = CameraInfo()
            info.header = msg.header
            info.height = self.last_info_msg.height
            info.width = self.last_info_msg.width
            info.distortion_model = self.last_info_msg.distortion_model
            info.d = list(self.last_info_msg.d)
            info.k = list(self.last_info_msg.k)
            info.r = list(self.last_info_msg.r)
            info.p = list(self.last_info_msg.p)
            info.binning_x = self.last_info_msg.binning_x
            info.binning_y = self.last_info_msg.binning_y
            info.roi = self.last_info_msg.roi
            self.info_pub.publish(info)
        self.forwarded_depth_count += 1
        self._maybe_log(now, elapsed)

    def _last_frame_timer_cb(self) -> None:
        if not self.frozen or not self.keep_publishing_last_frame:
            return
        if self.last_depth_msg is not None:
            self.depth_pub.publish(self.last_depth_msg)
        if self.last_info_msg is not None:
            self.info_pub.publish(self.last_info_msg)

    def _debug_timer_cb(self) -> None:
        if self.start_time is not None or self.frozen:
            return
        self.get_logger().warn(
            "Waiting to start nvblox input: "
            f"camera_info={'yes' if self.last_info_msg is not None else 'no'}, "
            f"depth_subscribers={self.depth_pub.get_subscription_count()}"
        )

    def _maybe_log(self, now, elapsed: float) -> None:
        if (now - self.last_log_time).nanoseconds * 1e-9 < self.debug_log_period_s:
            return
        self.last_log_time = now
        remaining = max(0.0, self.freeze_after_s - elapsed)
        self.get_logger().info(
            f"Depth gate active: forwarded={self.forwarded_depth_count}, "
            f"freezing in {remaining:.1f}s"
        )


def main() -> None:
    rclpy.init()
    node = DepthFreezeGateNode()
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
