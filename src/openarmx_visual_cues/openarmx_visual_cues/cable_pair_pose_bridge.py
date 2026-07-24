#!/usr/bin/env python3
"""Convert the two longest cable Marker cylinders to an Isaac-safe PoseArray."""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray


def marker_qos():
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class CablePairPoseBridge(Node):
    def __init__(self):
        super().__init__("cable_pair_pose_bridge")
        self.declare_parameter("marker_topic", "/perception/cable_capsules")
        self.declare_parameter("pose_topic", "/perception/cable_pair_pose_array")
        self.publisher = self.create_publisher(
            PoseArray, str(self.get_parameter("pose_topic").value), marker_qos()
        )
        self.create_subscription(
            MarkerArray,
            str(self.get_parameter("marker_topic").value),
            self._markers_cb,
            marker_qos(),
        )
        self.get_logger().info(
            f"Bridging {self.get_parameter('marker_topic').value} -> "
            f"{self.get_parameter('pose_topic').value}"
        )

    def _markers_cb(self, msg):
        cylinders = [
            marker
            for marker in msg.markers
            if marker.action == Marker.ADD
            and marker.type == Marker.CYLINDER
            and marker.ns == "cable_capsules"
            and marker.scale.z >= 0.15
        ]
        if len(cylinders) < 2:
            return
        cylinders.sort(key=lambda marker: float(marker.scale.z), reverse=True)
        first = cylinders[0]
        q0 = np.array(
            [
                first.pose.orientation.x,
                first.pose.orientation.y,
                first.pose.orientation.z,
                first.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        # Orientation similarity is |q dot q| for the same local cylinder axis.
        second = max(
            cylinders[1:],
            key=lambda marker: float(marker.scale.z)
            * abs(
                float(
                    np.dot(
                        q0,
                        np.array(
                            [
                                marker.pose.orientation.x,
                                marker.pose.orientation.y,
                                marker.pose.orientation.z,
                                marker.pose.orientation.w,
                            ]
                        ),
                    )
                )
            ),
        )
        output = PoseArray()
        output.header = first.header
        output.poses = [first.pose, second.pose]
        self.publisher.publish(output)


def main():
    rclpy.init()
    node = CablePairPoseBridge()
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
