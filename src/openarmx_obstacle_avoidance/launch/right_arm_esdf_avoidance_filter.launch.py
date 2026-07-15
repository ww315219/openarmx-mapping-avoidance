from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("urdf_path", default_value=""),
            DeclareLaunchArgument("robot_description_node", default_value="/robot_state_publisher"),
            DeclareLaunchArgument("command_message_type", default_value="float64_multi_array"),
            DeclareLaunchArgument("input_command_topic", default_value="/right_teleop_baseline/commands"),
            DeclareLaunchArgument("output_command_topic", default_value="/right_forward_position_controller/commands"),
            DeclareLaunchArgument("joint_states_topic", default_value="/joint_states"),
            DeclareLaunchArgument("esdf_service", default_value=""),
            DeclareLaunchArgument("global_frame", default_value="world"),
            DeclareLaunchArgument("rate_hz", default_value="30.0"),
            DeclareLaunchArgument("esdf_update_hz", default_value="8.0"),
            DeclareLaunchArgument("request_update_esdf", default_value="true"),
            DeclareLaunchArgument("nearest_observed_search_radius", default_value="0.16"),
            DeclareLaunchArgument("monitor_only", default_value="false"),
            DeclareLaunchArgument("safety_margin", default_value="0.02"),
            DeclareLaunchArgument("activation_margin", default_value="0.10"),
            DeclareLaunchArgument("avoidance_weight", default_value="0.3"),
            DeclareLaunchArgument("max_adjust_per_joint", default_value="0.02"),
            DeclareLaunchArgument("max_command_step", default_value="0.04"),
            DeclareLaunchArgument("max_avoidance_delta", default_value="0.08"),
            DeclareLaunchArgument("avoidance_delta_alpha", default_value="0.5"),
            DeclareLaunchArgument("aabb_padding", default_value="0.25"),
            DeclareLaunchArgument("clear_robot_from_esdf", default_value="true"),
            DeclareLaunchArgument("clear_robot_padding", default_value="0.015"),
            DeclareLaunchArgument("clear_robot_radius_scale", default_value="1.0"),
            Node(
                package="openarmx_obstacle_avoidance",
                executable="right_arm_esdf_avoidance_filter",
                name="right_arm_esdf_avoidance_filter",
                output="screen",
                additional_env={"PYTHONNOUSERSITE": "1"},
                parameters=[
                    {
                        "urdf_path": LaunchConfiguration("urdf_path"),
                        "robot_description_node": LaunchConfiguration("robot_description_node"),
                        "command_message_type": LaunchConfiguration("command_message_type"),
                        "input_command_topic": LaunchConfiguration("input_command_topic"),
                        "output_command_topic": LaunchConfiguration("output_command_topic"),
                        "joint_states_topic": LaunchConfiguration("joint_states_topic"),
                        "esdf_service": LaunchConfiguration("esdf_service"),
                        "global_frame": LaunchConfiguration("global_frame"),
                        "rate_hz": LaunchConfiguration("rate_hz"),
                        "esdf_update_hz": LaunchConfiguration("esdf_update_hz"),
                        "request_update_esdf": LaunchConfiguration("request_update_esdf"),
                        "nearest_observed_search_radius": LaunchConfiguration("nearest_observed_search_radius"),
                        "monitor_only": LaunchConfiguration("monitor_only"),
                        "safety_margin": LaunchConfiguration("safety_margin"),
                        "activation_margin": LaunchConfiguration("activation_margin"),
                        "avoidance_weight": LaunchConfiguration("avoidance_weight"),
                        "max_adjust_per_joint": LaunchConfiguration("max_adjust_per_joint"),
                        "max_command_step": LaunchConfiguration("max_command_step"),
                        "max_avoidance_delta": LaunchConfiguration("max_avoidance_delta"),
                        "avoidance_delta_alpha": LaunchConfiguration("avoidance_delta_alpha"),
                        "aabb_padding": LaunchConfiguration("aabb_padding"),
                        "clear_robot_from_esdf": LaunchConfiguration("clear_robot_from_esdf"),
                        "clear_robot_padding": LaunchConfiguration("clear_robot_padding"),
                        "clear_robot_radius_scale": LaunchConfiguration("clear_robot_radius_scale"),
                    }
                ],
            ),
        ]
    )
