from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("input_marker_topic", default_value="/perception/cable_capsules"),
            DeclareLaunchArgument("ground_truth_marker_topic", default_value=""),
            DeclareLaunchArgument("ground_truth_pose_array_topic", default_value=""),
            DeclareLaunchArgument("use_ground_truth_as_estimate", default_value="false"),
            DeclareLaunchArgument("simulation_cable_radius_m", default_value="0.0075"),
            DeclareLaunchArgument("ground_truth_topic", default_value="/protected_cables/ground_truth"),
            DeclareLaunchArgument("estimate_topic", default_value="/protected_cables/estimate"),
            DeclareLaunchArgument(
                "estimate_marker_topic", default_value="/protected_cables/estimate_markers"
            ),
            DeclareLaunchArgument("source", default_value="3"),
            DeclareLaunchArgument("translation_bias_m", default_value="[0.0, 0.0, 0.0]"),
            DeclareLaunchArgument("rotation_bias_deg", default_value="[0.0, 0.0, 0.0]"),
            DeclareLaunchArgument("gaussian_position_noise_std_m", default_value="0.0"),
            DeclareLaunchArgument("gaussian_radius_noise_std_m", default_value="0.0"),
            DeclareLaunchArgument("reported_position_std_m", default_value="-1.0"),
            DeclareLaunchArgument("reported_radius_std_m", default_value="-1.0"),
            DeclareLaunchArgument("latency_ms", default_value="0.0"),
            DeclareLaunchArgument("dropout_probability", default_value="0.0"),
            DeclareLaunchArgument("random_seed", default_value="7"),
            Node(
                package="openarmx_nvblox_bringup",
                executable="protected_cable_perturbation_node.py",
                name="protected_cable_perturbation",
                output="screen",
                parameters=[
                    {
                        "input_marker_topic": LaunchConfiguration("input_marker_topic"),
                        "ground_truth_marker_topic": LaunchConfiguration(
                            "ground_truth_marker_topic"
                        ),
                        "ground_truth_pose_array_topic": LaunchConfiguration(
                            "ground_truth_pose_array_topic"
                        ),
                        "use_ground_truth_as_estimate": LaunchConfiguration(
                            "use_ground_truth_as_estimate"
                        ),
                        "simulation_cable_radius_m": LaunchConfiguration(
                            "simulation_cable_radius_m"
                        ),
                        "ground_truth_topic": LaunchConfiguration("ground_truth_topic"),
                        "estimate_topic": LaunchConfiguration("estimate_topic"),
                        "estimate_marker_topic": LaunchConfiguration("estimate_marker_topic"),
                        "source": LaunchConfiguration("source"),
                        "translation_bias_m": LaunchConfiguration("translation_bias_m"),
                        "rotation_bias_deg": LaunchConfiguration("rotation_bias_deg"),
                        "gaussian_position_noise_std_m": LaunchConfiguration(
                            "gaussian_position_noise_std_m"
                        ),
                        "gaussian_radius_noise_std_m": LaunchConfiguration(
                            "gaussian_radius_noise_std_m"
                        ),
                        "reported_position_std_m": LaunchConfiguration(
                            "reported_position_std_m"
                        ),
                        "reported_radius_std_m": LaunchConfiguration("reported_radius_std_m"),
                        "latency_ms": LaunchConfiguration("latency_ms"),
                        "dropout_probability": LaunchConfiguration("dropout_probability"),
                        "random_seed": LaunchConfiguration("random_seed"),
                    }
                ],
            ),
        ]
    )
