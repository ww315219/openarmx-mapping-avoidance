from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("input_command_topic", default_value="/right_teleop_baseline/commands"),
            DeclareLaunchArgument("output_command_topic", default_value="/right_forward_position_controller/commands"),
            DeclareLaunchArgument("joint_states_topic", default_value="/joint_states"),
            DeclareLaunchArgument("cumotion_action_name", default_value="cumotion/motion_plan"),
            DeclareLaunchArgument("global_frame", default_value="world"),
            DeclareLaunchArgument("rate_hz", default_value="50.0"),
            DeclareLaunchArgument("replan_hz", default_value="5.0"),
            DeclareLaunchArgument("baseline_replan_threshold", default_value="0.025"),
            DeclareLaunchArgument("max_command_step", default_value="0.025"),
            DeclareLaunchArgument("trajectory_lookahead_index", default_value="1"),
            DeclareLaunchArgument("trajectory_point_reached_tolerance", default_value="0.015"),
            DeclareLaunchArgument("plan_time_dilation_factor", default_value="0.5"),
            DeclareLaunchArgument("update_esdf", default_value="true"),
            DeclareLaunchArgument("clear_esdf", default_value="true"),
            DeclareLaunchArgument("enable_aabb_clearing", default_value="true"),
            DeclareLaunchArgument("visualize_trajectory", default_value="true"),
            DeclareLaunchArgument("passthrough_without_cumotion", default_value="true"),
            DeclareLaunchArgument("fallback_to_baseline_on_failure", default_value="true"),
            DeclareLaunchArgument(
                "status_topic",
                default_value="/openarm/right_arm/cumotion_avoidance_status",
            ),
            Node(
                package="openarmx_obstacle_avoidance",
                executable="right_arm_cumotion_avoidance_filter",
                name="right_arm_cumotion_avoidance_filter",
                output="screen",
                additional_env={"PYTHONNOUSERSITE": "1"},
                parameters=[
                    {
                        "input_command_topic": LaunchConfiguration("input_command_topic"),
                        "output_command_topic": LaunchConfiguration("output_command_topic"),
                        "joint_states_topic": LaunchConfiguration("joint_states_topic"),
                        "cumotion_action_name": LaunchConfiguration("cumotion_action_name"),
                        "global_frame": LaunchConfiguration("global_frame"),
                        "rate_hz": LaunchConfiguration("rate_hz"),
                        "replan_hz": LaunchConfiguration("replan_hz"),
                        "baseline_replan_threshold": LaunchConfiguration("baseline_replan_threshold"),
                        "max_command_step": LaunchConfiguration("max_command_step"),
                        "trajectory_lookahead_index": LaunchConfiguration("trajectory_lookahead_index"),
                        "trajectory_point_reached_tolerance": LaunchConfiguration(
                            "trajectory_point_reached_tolerance"
                        ),
                        "plan_time_dilation_factor": LaunchConfiguration("plan_time_dilation_factor"),
                        "update_esdf": LaunchConfiguration("update_esdf"),
                        "clear_esdf": LaunchConfiguration("clear_esdf"),
                        "enable_aabb_clearing": LaunchConfiguration("enable_aabb_clearing"),
                        "visualize_trajectory": LaunchConfiguration("visualize_trajectory"),
                        "passthrough_without_cumotion": LaunchConfiguration("passthrough_without_cumotion"),
                        "fallback_to_baseline_on_failure": LaunchConfiguration(
                            "fallback_to_baseline_on_failure"
                        ),
                        "status_topic": LaunchConfiguration("status_topic"),
                    }
                ],
            ),
        ]
    )
