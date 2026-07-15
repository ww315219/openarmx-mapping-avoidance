from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("depth_topic", default_value="/foundation_stereo/depth"),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera/infra1/camera_info"),
            DeclareLaunchArgument("color_image_topic", default_value="/camera/color/image_raw"),
            DeclareLaunchArgument(
                "compressed_color_image_topic",
                default_value="/camera/color/image_raw/compressed",
            ),
            DeclareLaunchArgument("use_compressed_image", default_value="true"),
            DeclareLaunchArgument("joint_states_topic", default_value="/joint_states"),
            DeclareLaunchArgument("robot_description_node", default_value="/robot_state_publisher"),
            DeclareLaunchArgument("global_frame", default_value="world"),
            DeclareLaunchArgument("robot_mask_padding_px", default_value="8"),
            DeclareLaunchArgument("robot_radius_scale", default_value="1.15"),
            DeclareLaunchArgument("near_miss_radius_px", default_value="28"),
            DeclareLaunchArgument("valid_depth_min_m", default_value="0.05"),
            DeclareLaunchArgument("valid_depth_max_m", default_value="3.0"),
            DeclareLaunchArgument("debug_log_period_s", default_value="1.0"),
            Node(
                package="openarmx_nvblox_bringup",
                executable="robot_self_filter_diagnostics_node.py",
                name="robot_self_filter_diagnostics",
                output="screen",
                parameters=[
                    {
                        "depth_topic": LaunchConfiguration("depth_topic"),
                        "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                        "color_image_topic": LaunchConfiguration("color_image_topic"),
                        "compressed_color_image_topic": LaunchConfiguration(
                            "compressed_color_image_topic"
                        ),
                        "use_compressed_image": LaunchConfiguration("use_compressed_image"),
                        "joint_states_topic": LaunchConfiguration("joint_states_topic"),
                        "robot_description_node": LaunchConfiguration("robot_description_node"),
                        "global_frame": LaunchConfiguration("global_frame"),
                        "robot_mask_padding_px": LaunchConfiguration("robot_mask_padding_px"),
                        "robot_radius_scale": LaunchConfiguration("robot_radius_scale"),
                        "near_miss_radius_px": LaunchConfiguration("near_miss_radius_px"),
                        "valid_depth_min_m": LaunchConfiguration("valid_depth_min_m"),
                        "valid_depth_max_m": LaunchConfiguration("valid_depth_max_m"),
                        "debug_log_period_s": LaunchConfiguration("debug_log_period_s"),
                    }
                ],
            ),
        ]
    )
