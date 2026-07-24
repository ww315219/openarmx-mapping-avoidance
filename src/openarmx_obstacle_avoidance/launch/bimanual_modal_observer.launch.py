from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("imu_topic", default_value="/camera/imu"),
            DeclareLaunchArgument("joint_states_topic", default_value="/joint_states"),
            DeclareLaunchArgument("rate_hz", default_value="100.0"),
            DeclareLaunchArgument("calibration_duration", default_value="2.0"),
            DeclareLaunchArgument("sensor_timeout", default_value="0.15"),
            DeclareLaunchArgument(
                "disturbance_cutoff_hz", default_value="1.0"
            ),
            DeclareLaunchArgument(
                "disturbance_process_noise", default_value="1e-4"
            ),
            Node(
                package="openarmx_obstacle_avoidance",
                executable="bimanual_modal_observer",
                name="bimanual_modal_observer",
                output="screen",
                parameters=[
                    {
                        "imu_topic": LaunchConfiguration("imu_topic"),
                        "joint_states_topic": LaunchConfiguration("joint_states_topic"),
                        "rate_hz": LaunchConfiguration("rate_hz"),
                        "calibration_duration": LaunchConfiguration(
                            "calibration_duration"
                        ),
                        "sensor_timeout": LaunchConfiguration("sensor_timeout"),
                        "disturbance_cutoff_hz": LaunchConfiguration(
                            "disturbance_cutoff_hz"
                        ),
                        "disturbance_process_noise": LaunchConfiguration(
                            "disturbance_process_noise"
                        ),
                    }
                ],
            ),
        ]
    )
