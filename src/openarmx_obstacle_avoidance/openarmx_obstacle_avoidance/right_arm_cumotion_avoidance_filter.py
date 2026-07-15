from __future__ import annotations

import math
from typing import Any

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

RIGHT_JOINT_NAMES = [f"openarmx_right_joint{i}" for i in range(1, 8)]


class RightArmCumotionAvoidanceFilter(Node):
    """Experimental cuMotion bridge for teleop baseline commands.

    The node asks cuMotion for a collision-aware C-space trajectory from the
    current commanded/measured state to the latest teleop baseline. Runtime
    output is still rate-limited position commands, so replanning does not
    directly switch the controller to a discontinuous solution.
    """

    def __init__(self) -> None:
        super().__init__("right_arm_cumotion_avoidance_filter")

        self.declare_parameter("joint_names", RIGHT_JOINT_NAMES)
        self.declare_parameter("input_command_topic", "/right_teleop_baseline/commands")
        self.declare_parameter("output_command_topic", "/right_forward_position_controller/commands")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("cumotion_action_name", "cumotion/motion_plan")
        self.declare_parameter("global_frame", "world")
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("replan_hz", 5.0)
        self.declare_parameter("baseline_replan_threshold", 0.025)
        self.declare_parameter("max_command_step", 0.025)
        self.declare_parameter("trajectory_lookahead_index", 1)
        self.declare_parameter("trajectory_point_reached_tolerance", 0.015)
        self.declare_parameter("plan_time_dilation_factor", 0.5)
        self.declare_parameter("update_esdf", True)
        self.declare_parameter("clear_esdf", True)
        self.declare_parameter("enable_aabb_clearing", True)
        self.declare_parameter("visualize_trajectory", True)
        self.declare_parameter("passthrough_without_cumotion", True)
        self.declare_parameter("fallback_to_baseline_on_failure", True)
        self.declare_parameter("status_topic", "/openarm/right_arm/cumotion_avoidance_status")

        self.joint_names = list(self.get_parameter("joint_names").value)
        self.input_command_topic = str(self.get_parameter("input_command_topic").value)
        self.output_command_topic = str(self.get_parameter("output_command_topic").value)
        self.joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self.cumotion_action_name = str(self.get_parameter("cumotion_action_name").value)
        self.global_frame = str(self.get_parameter("global_frame").value)
        self.rate_hz = max(1.0, float(self.get_parameter("rate_hz").value))
        self.replan_hz = max(0.1, float(self.get_parameter("replan_hz").value))
        self.baseline_replan_threshold = max(
            0.0,
            float(self.get_parameter("baseline_replan_threshold").value),
        )
        self.max_command_step = max(0.0, float(self.get_parameter("max_command_step").value))
        self.trajectory_lookahead_index = max(0, int(self.get_parameter("trajectory_lookahead_index").value))
        self.trajectory_point_reached_tolerance = max(
            0.0,
            float(self.get_parameter("trajectory_point_reached_tolerance").value),
        )
        self.plan_time_dilation_factor = float(self.get_parameter("plan_time_dilation_factor").value)
        self.update_esdf = bool(self.get_parameter("update_esdf").value)
        self.clear_esdf = bool(self.get_parameter("clear_esdf").value)
        self.enable_aabb_clearing = bool(self.get_parameter("enable_aabb_clearing").value)
        self.visualize_trajectory = bool(self.get_parameter("visualize_trajectory").value)
        self.passthrough_without_cumotion = bool(self.get_parameter("passthrough_without_cumotion").value)
        self.fallback_to_baseline_on_failure = bool(self.get_parameter("fallback_to_baseline_on_failure").value)

        self.latest_baseline: np.ndarray | None = None
        self.latest_extra: list[float] = []
        self.last_planned_baseline: np.ndarray | None = None
        self.q_measured: np.ndarray | None = None
        self.q_command: np.ndarray | None = None
        self.pending_goal = False
        self.have_cumotion_interfaces = False
        self.MotionPlan: Any | None = None
        self.current_plan: list[np.ndarray] = []
        self.current_plan_index = 0
        self.last_plan_ok = False
        self.last_plan_message = "no_plan"
        self.last_replan_time = self.get_clock().now()
        self.last_status_time = self.get_clock().now()

        self.create_subscription(JointState, self.joint_states_topic, self._joint_state_cb, 20)
        self.create_subscription(Float64MultiArray, self.input_command_topic, self._baseline_cb, 10)
        self.command_pub = self.create_publisher(Float64MultiArray, self.output_command_topic, 10)
        self.status_pub = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            10,
        )
        self.timer = self.create_timer(1.0 / self.rate_hz, self._timer_cb)

        self.action_client: ActionClient | None = self._make_action_client()

        self.get_logger().info(
            f"cuMotion avoidance filter: {self.input_command_topic} -> {self.output_command_topic}"
        )
        self.get_logger().info(
            f"action={self.cumotion_action_name}, replan_hz={self.replan_hz:.1f}, "
            f"max_command_step={self.max_command_step:.3f}, update_esdf={self.update_esdf}, "
            f"clear_esdf={self.clear_esdf}, passthrough_without_cumotion={self.passthrough_without_cumotion}"
        )

    def _make_action_client(self) -> ActionClient | None:
        try:
            from isaac_ros_cumotion_interfaces.action import MotionPlan
        except Exception as exc:
            self.get_logger().error(
                "isaac_ros_cumotion_interfaces is not available. "
                "Install/source isaac_ros_cumotion before expecting obstacle avoidance. "
                f"Import error: {exc}"
            )
            return None
        self.MotionPlan = MotionPlan
        self.have_cumotion_interfaces = True
        return ActionClient(self, MotionPlan, self.cumotion_action_name)

    def _joint_state_cb(self, msg: JointState) -> None:
        name_to_position = {name: position for name, position in zip(msg.name, msg.position)}
        if not all(name in name_to_position for name in self.joint_names):
            return
        self.q_measured = np.asarray([name_to_position[name] for name in self.joint_names], dtype=float)
        if self.q_command is None:
            self.q_command = self.q_measured.copy()

    def _baseline_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < len(self.joint_names):
            self.get_logger().warn(
                f"Ignoring baseline with {len(msg.data)} positions; need {len(self.joint_names)}",
                throttle_duration_sec=1.0,
            )
            return
        self.latest_baseline = np.asarray(msg.data[: len(self.joint_names)], dtype=float)
        self.latest_extra = [float(v) for v in msg.data[len(self.joint_names) :]]
        if self.q_command is None:
            self.q_command = self.latest_baseline.copy()

    def _timer_cb(self) -> None:
        if self.latest_baseline is None:
            self._publish_status("waiting_for_baseline")
            return
        if self.q_command is None:
            self.q_command = self.latest_baseline.copy()

        self._maybe_request_plan()
        target = self._next_target()
        q_out = self._step_toward(self.q_command, target, self.max_command_step)
        self.q_command = q_out
        self.command_pub.publish(Float64MultiArray(data=[float(v) for v in q_out] + self.latest_extra))
        self._publish_status(self.last_plan_message)

    def _maybe_request_plan(self) -> None:
        if self.action_client is None or self.MotionPlan is None:
            return
        if self.pending_goal:
            return

        now = self.get_clock().now()
        elapsed = (now - self.last_replan_time).nanoseconds * 1e-9
        if elapsed < 1.0 / self.replan_hz:
            return

        if self.last_planned_baseline is not None:
            baseline_delta = float(np.max(np.abs(self.latest_baseline - self.last_planned_baseline)))
            if baseline_delta < self.baseline_replan_threshold and self.current_plan:
                return

        if not self.action_client.wait_for_server(timeout_sec=0.001):
            self.last_plan_ok = False
            self.last_plan_message = "waiting_for_cumotion_action"
            return

        goal_msg = self._build_motion_plan_goal()
        self.pending_goal = True
        self.last_replan_time = now
        self.last_planned_baseline = self.latest_baseline.copy()

        send_future = self.action_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self._goal_response_cb)

    def _build_motion_plan_goal(self) -> Any:
        goal_msg = self.MotionPlan.Goal()
        goal_msg.start_state = self._joint_state_from_array(self.q_command)
        goal_msg.goal_state = self._joint_state_from_array(self.latest_baseline)
        goal_msg.time_dilation_factor = float(self.plan_time_dilation_factor)
        goal_msg.update_esdf = bool(self.update_esdf)
        goal_msg.clear_esdf = bool(self.clear_esdf)
        goal_msg.plan_cspace = True
        goal_msg.plan_pose = False
        goal_msg.visualize_trajectory = bool(self.visualize_trajectory)
        goal_msg.execute_trajectory = False
        goal_msg.visualize_world = False
        goal_msg.use_current_state = False
        goal_msg.use_planning_scene = False
        goal_msg.hold_partial_pose = False
        goal_msg.plan_grasp = False
        goal_msg.plan_approach_to_grasp = False
        goal_msg.plan_grasp_to_retract = False
        goal_msg.object_frame = self.global_frame
        goal_msg.world_frame = self.global_frame
        goal_msg.enable_aabb_clearing = bool(self.enable_aabb_clearing)
        return goal_msg

    def _joint_state_from_array(self, q: np.ndarray) -> JointState:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joint_names)
        msg.position = [float(v) for v in q]
        return msg

    def _goal_response_cb(self, future: Any) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.pending_goal = False
            self.last_plan_ok = False
            self.last_plan_message = f"send_goal_failed: {exc}"
            return

        if not goal_handle.accepted:
            self.pending_goal = False
            self.last_plan_ok = False
            self.last_plan_message = "goal_rejected"
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future: Any) -> None:
        self.pending_goal = False
        try:
            result = future.result().result
        except Exception as exc:
            self.last_plan_ok = False
            self.last_plan_message = f"result_failed: {exc}"
            return

        success = bool(getattr(result, "success", False))
        message = str(getattr(result, "message", ""))
        if not success:
            self.last_plan_ok = False
            self.last_plan_message = f"plan_failed: {message}"
            if not self.fallback_to_baseline_on_failure:
                self.current_plan = []
            return

        plan = self._extract_plan(result)
        if not plan:
            self.last_plan_ok = False
            self.last_plan_message = "plan_empty"
            return

        self.current_plan = plan
        self.current_plan_index = 0
        self.last_plan_ok = True
        self.last_plan_message = f"plan_ok: points={len(plan)}"

    def _extract_plan(self, result: Any) -> list[np.ndarray]:
        trajectories = list(getattr(result, "planned_trajectory", []))
        if not trajectories:
            return []
        joint_trajectory = trajectories[0].joint_trajectory
        if not joint_trajectory.points:
            return []

        name_to_index = {name: index for index, name in enumerate(joint_trajectory.joint_names)}
        if not all(name in name_to_index for name in self.joint_names):
            self.get_logger().warn(
                "Ignoring cuMotion trajectory because it does not contain all configured joints.",
                throttle_duration_sec=1.0,
            )
            return []

        plan: list[np.ndarray] = []
        for point in joint_trajectory.points:
            if len(point.positions) < len(joint_trajectory.joint_names):
                continue
            q = np.asarray([point.positions[name_to_index[name]] for name in self.joint_names], dtype=float)
            if np.all(np.isfinite(q)):
                plan.append(q)
        return plan

    def _next_target(self) -> np.ndarray:
        if self.action_client is None and not self.passthrough_without_cumotion:
            return self.q_command.copy()

        if not self.current_plan:
            return self.latest_baseline.copy()

        while self.current_plan_index < len(self.current_plan):
            target = self.current_plan[self.current_plan_index]
            error = float(np.max(np.abs(target - self.q_command)))
            if error > self.trajectory_point_reached_tolerance:
                break
            self.current_plan_index += 1

        index = min(
            self.current_plan_index + self.trajectory_lookahead_index,
            len(self.current_plan) - 1,
        )
        return self.current_plan[index].copy()

    def _step_toward(self, q_current: np.ndarray, q_target: np.ndarray, max_step: float) -> np.ndarray:
        if max_step <= 0.0 or not math.isfinite(max_step):
            return q_target.copy()
        delta = np.clip(q_target - q_current, -max_step, max_step)
        return q_current + delta

    def _publish_status(self, state: str) -> None:
        now = self.get_clock().now()
        if (now - self.last_status_time).nanoseconds * 1e-9 < 0.5:
            return
        self.last_status_time = now
        status = {
            "state": state,
            "have_interfaces": self.have_cumotion_interfaces,
            "pending_goal": self.pending_goal,
            "last_plan_ok": self.last_plan_ok,
            "plan_points": len(self.current_plan),
            "plan_index": self.current_plan_index,
        }
        self.status_pub.publish(String(data=str(status)))


def main() -> None:
    rclpy.init()
    node = RightArmCumotionAvoidanceFilter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
