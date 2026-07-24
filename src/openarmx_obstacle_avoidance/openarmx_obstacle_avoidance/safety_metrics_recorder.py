from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import rclpy
from openarmx_safety_msgs.msg import SafetyStatus
from rclpy.node import Node


FIELDS = [
    "stamp_sec",
    "stamp_nanosec",
    "mode",
    "solver",
    "obstacle_source",
    "task_phase",
    "qp_status",
    "cbf_success",
    "holding",
    "ground_truth_available",
    "ground_truth_collision",
    "ground_truth_margin_violation",
    "estimated_min_clearance",
    "ground_truth_min_clearance",
    "left_min_clearance",
    "right_min_clearance",
    "inter_arm_min_clearance",
    "baseline_min_clearance",
    "command_deviation_norm",
    "max_slack",
    "cable_estimate_age",
    "cable_ground_truth_age",
    "active_constraints",
    "active_environment_constraints",
    "active_inter_arm_constraints",
    "estimated_cable_count",
    "ground_truth_cable_count",
]


class SafetyMetricsRecorder(Node):
    def __init__(self) -> None:
        super().__init__("safety_metrics_recorder")
        self.declare_parameter("status_topic", "/openarmx/bimanual/safety_status")
        self.declare_parameter("output_path", "")
        self.declare_parameter("flush_every", 10)

        output = str(self.get_parameter("output_path").value).strip()
        if not output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = str(Path.home() / ".ros" / "openarmx_safety" / f"safety_{timestamp}.csv")
        self.output_path = Path(output).expanduser()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.output_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=FIELDS)
        self.writer.writeheader()
        self.flush_every = max(1, int(self.get_parameter("flush_every").value))
        self.rows = 0
        self.create_subscription(
            SafetyStatus,
            str(self.get_parameter("status_topic").value),
            self._status_cb,
            50,
        )
        self.get_logger().info(f"Recording structured safety metrics to {self.output_path}")

    def _status_cb(self, msg: SafetyStatus) -> None:
        row = {field: getattr(msg, field) for field in FIELDS if hasattr(msg, field)}
        row["stamp_sec"] = msg.header.stamp.sec
        row["stamp_nanosec"] = msg.header.stamp.nanosec
        self.writer.writerow(row)
        self.rows += 1
        if self.rows % self.flush_every == 0:
            self.stream.flush()

    def destroy_node(self):
        self.stream.flush()
        self.stream.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = SafetyMetricsRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
