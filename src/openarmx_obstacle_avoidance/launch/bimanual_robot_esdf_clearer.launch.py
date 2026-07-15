from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("urdf_path", default_value=""),
            DeclareLaunchArgument("robot_description_node", default_value="/robot_state_publisher"),
            DeclareLaunchArgument("joint_states_topic", default_value="/joint_states"),
            DeclareLaunchArgument("esdf_service", default_value=""),
            DeclareLaunchArgument("global_frame", default_value="world"),
            DeclareLaunchArgument("rate_hz", default_value="2.0"),
            DeclareLaunchArgument("request_update_esdf", default_value="false"),
            DeclareLaunchArgument("aabb_padding", default_value="0.08"),
            DeclareLaunchArgument("clear_robot_padding", default_value="0.015"),
            DeclareLaunchArgument("clear_robot_radius_scale", default_value="1.0"),
            DeclareLaunchArgument("min_joint_delta_to_clear", default_value="0.08"),
            DeclareLaunchArgument("force_clear_period_s", default_value="4.0"),
            DeclareLaunchArgument("collision_model", default_value="capsule"),
            DeclareLaunchArgument("capsule_samples_per_link", default_value="3"),
            DeclareLaunchArgument("publish_debug_markers", default_value="true"),
            DeclareLaunchArgument("debug_marker_topic", default_value="/openarmx/robot_clear_shapes"),
            Node(
                package="openarmx_obstacle_avoidance",
                executable="bimanual_robot_esdf_clearer",
                name="bimanual_robot_esdf_clearer",
                output="screen",
                additional_env={"PYTHONNOUSERSITE": "1"},
                parameters=[
                    {
                        "urdf_path": LaunchConfiguration("urdf_path"),
                        "robot_description_node": LaunchConfiguration("robot_description_node"),
                        "joint_states_topic": LaunchConfiguration("joint_states_topic"),
                        "esdf_service": LaunchConfiguration("esdf_service"),
                        "global_frame": LaunchConfiguration("global_frame"),
                        "rate_hz": LaunchConfiguration("rate_hz"),
                        "request_update_esdf": LaunchConfiguration("request_update_esdf"),
                        "aabb_padding": LaunchConfiguration("aabb_padding"),
                        "clear_robot_padding": LaunchConfiguration("clear_robot_padding"),
                        "clear_robot_radius_scale": LaunchConfiguration("clear_robot_radius_scale"),
                        "min_joint_delta_to_clear": LaunchConfiguration("min_joint_delta_to_clear"),
                        "force_clear_period_s": LaunchConfiguration("force_clear_period_s"),
                        "collision_model": LaunchConfiguration("collision_model"),
                        "capsule_samples_per_link": LaunchConfiguration("capsule_samples_per_link"),
                        "publish_debug_markers": LaunchConfiguration("publish_debug_markers"),
                        "debug_marker_topic": LaunchConfiguration("debug_marker_topic"),
                    }
                ],
            ),
        ]
    )
