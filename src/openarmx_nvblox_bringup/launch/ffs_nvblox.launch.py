from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    depth_topic = LaunchConfiguration("depth_topic")
    depth_camera_info_topic = LaunchConfiguration("depth_camera_info_topic")
    color_topic = LaunchConfiguration("color_topic")
    color_camera_info_topic = LaunchConfiguration("color_camera_info_topic")
    run_rviz = LaunchConfiguration("run_rviz")
    log_level = LaunchConfiguration("log_level")

    config_path = os.path.join(
        get_package_share_directory("openarmx_nvblox_bringup"),
        "config",
        "ffs_nvblox.yaml",
    )
    rviz_config_path = os.path.join(
        get_package_share_directory("openarmx_nvblox_bringup"),
        "config",
        "ffs_nvblox.rviz",
    )

    nvblox_node = Node(
        package="nvblox_ros",
        executable="nvblox_node",
        name="nvblox_node",
        output="screen",
        parameters=[config_path],
        arguments=["--ros-args", "--log-level", log_level],
        remappings=[
            ("camera_0/depth/image", depth_topic),
            ("camera_0/depth/camera_info", depth_camera_info_topic),
            ("camera_0/color/image", color_topic),
            ("camera_0/color/camera_info", color_camera_info_topic),
        ],
    )

    rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare("nvblox_examples_bringup"),
            "/launch/visualization/rviz.launch.py",
        ]),
        launch_arguments={
            "mode": "static",
            "camera": "realsense",
            "rviz_config": rviz_config_path,
        }.items(),
        condition=IfCondition(run_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "depth_topic",
            default_value="/foundation_stereo/depth_color_aligned",
            description="Depth image in meters, aligned to the color camera frame.",
        ),
        DeclareLaunchArgument(
            "depth_camera_info_topic",
            default_value="/foundation_stereo/depth_color_aligned/camera_info",
            description="CameraInfo matching the depth image.",
        ),
        DeclareLaunchArgument(
            "color_topic",
            default_value="/camera/color/image_raw",
            description="Color image topic.",
        ),
        DeclareLaunchArgument(
            "color_camera_info_topic",
            default_value="/camera/color/camera_info",
            description="CameraInfo matching the color image.",
        ),
        DeclareLaunchArgument("run_rviz", default_value="true"),
        DeclareLaunchArgument("log_level", default_value="info"),
        nvblox_node,
        rviz_launch,
    ])
