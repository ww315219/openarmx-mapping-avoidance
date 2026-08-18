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
            DeclareLaunchArgument("residual_model_enabled", default_value="false"),
            DeclareLaunchArgument(
                "python_executable",
                default_value="/home/wanghua/miniconda3/envs/ffs/bin/python",
            ),
            DeclareLaunchArgument(
                "residual_model_monitor_only", default_value="true"
            ),
            DeclareLaunchArgument("residual_model_backend", default_value="torchscript"),
            DeclareLaunchArgument("residual_model_device", default_value="cpu"),
            DeclareLaunchArgument("residual_model_path", default_value=""),
            DeclareLaunchArgument(
                "residual_model_correction_gain", default_value="1.0"
            ),
            Node(
                package="openarmx_obstacle_avoidance",
                executable="bimanual_modal_observer",
                name="bimanual_modal_observer",
                output="screen",
                prefix=[LaunchConfiguration("python_executable")],
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
                        "residual_model_enabled": LaunchConfiguration(
                            "residual_model_enabled"
                        ),
                        "residual_model_monitor_only": LaunchConfiguration(
                            "residual_model_monitor_only"
                        ),
                        "residual_model_backend": LaunchConfiguration(
                            "residual_model_backend"
                        ),
                        "residual_model_device": LaunchConfiguration(
                            "residual_model_device"
                        ),
                        "residual_model_path": LaunchConfiguration(
                            "residual_model_path"
                        ),
                        "residual_model_correction_gain": LaunchConfiguration(
                            "residual_model_correction_gain"
                        ),
                    }
                ],
            ),
        ]
    )
