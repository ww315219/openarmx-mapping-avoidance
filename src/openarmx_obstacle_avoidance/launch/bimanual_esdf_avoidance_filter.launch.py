import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _python_vendor_path() -> str:
    launch_path = Path(__file__).resolve()
    for parent in launch_path.parents:
        candidate = parent / "third_party" / "python3.12" / "site-packages"
        if candidate.is_dir():
            existing = os.environ.get("PYTHONPATH", "")
            return str(candidate) if not existing else f"{candidate}:{existing}"
    return os.environ.get("PYTHONPATH", "")


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("urdf_path", default_value=""),
            DeclareLaunchArgument("robot_description_node", default_value="/robot_state_publisher"),
            DeclareLaunchArgument("left_input_command_topic", default_value="/left_teleop_baseline/commands"),
            DeclareLaunchArgument("right_input_command_topic", default_value="/right_teleop_baseline/commands"),
            DeclareLaunchArgument("left_output_command_topic", default_value="/left_forward_position_controller/commands"),
            DeclareLaunchArgument("right_output_command_topic", default_value="/right_forward_position_controller/commands"),
            DeclareLaunchArgument("joint_states_topic", default_value="/joint_states"),
            DeclareLaunchArgument("joint_state_timeout_s", default_value="0.15"),
            DeclareLaunchArgument("use_measured_joint_state_start", default_value="true"),
            DeclareLaunchArgument("esdf_service", default_value=""),
            DeclareLaunchArgument("global_frame", default_value="world"),
            DeclareLaunchArgument("rate_hz", default_value="30.0"),
            DeclareLaunchArgument("esdf_update_hz", default_value="8.0"),
            DeclareLaunchArgument("request_update_esdf", default_value="true"),
            DeclareLaunchArgument("min_valid_esdf_distance", default_value="-0.25"),
            DeclareLaunchArgument("nearest_observed_search_radius", default_value="0.20"),
            DeclareLaunchArgument("enable_esdf_virtual_extension", default_value="true"),
            DeclareLaunchArgument("esdf_extension_obstacle_direction", default_value="[0.0, 0.0, 0.0]"),
            DeclareLaunchArgument("esdf_extension_length", default_value="0.08"),
            DeclareLaunchArgument("esdf_extension_step", default_value="0.03"),
            DeclareLaunchArgument(
                "obstacle_source",
                default_value="cable_capsules",
                description="Environment obstacle source: esdf, cable_capsules, or hybrid.",
            ),
            DeclareLaunchArgument(
                "cable_capsule_topic",
                default_value="/perception/cable_capsules",
            ),
            DeclareLaunchArgument(
                "protected_cable_topic",
                default_value="/protected_cables/estimate",
            ),
            DeclareLaunchArgument(
                "ground_truth_cable_topic",
                default_value="/protected_cables/ground_truth",
            ),
            DeclareLaunchArgument("legacy_cable_marker_fallback", default_value="true"),
            DeclareLaunchArgument("cable_capsule_timeout_s", default_value="0.75"),
            DeclareLaunchArgument(
                "cable_capsule_timeout_behavior",
                default_value="use_last",
                description=(
                    "Behavior when cable capsules are stale: use_last, hold, fallback_esdf, "
                    "or pass_through."
                ),
            ),
            DeclareLaunchArgument("cable_capsule_radius_scale", default_value="1.0"),
            DeclareLaunchArgument("cable_capsule_padding", default_value="0.0"),
            DeclareLaunchArgument("cable_uncertainty_sigma_scale", default_value="2.0"),
            DeclareLaunchArgument("cable_capsule_max_count", default_value="32"),
            DeclareLaunchArgument("untangle_mode_topic", default_value="/openarmx/untangle_mode"),
            DeclareLaunchArgument("untangle_tangent_weight", default_value="1.2"),
            DeclareLaunchArgument("untangle_tangent_max_step", default_value="0.012"),
            DeclareLaunchArgument("collision_model", default_value="capsule"),
            DeclareLaunchArgument("capsule_sample_spacing", default_value="0.06"),
            DeclareLaunchArgument("capsule_min_samples", default_value="2"),
            DeclareLaunchArgument("capsule_samples_per_link", default_value="3"),
            DeclareLaunchArgument("end_effector_collision_padding", default_value="0.01"),
            DeclareLaunchArgument("esdf_skip_proximal_spheres", default_value="3"),
            DeclareLaunchArgument("assisted_grasp_enabled", default_value="false"),
            DeclareLaunchArgument(
                "left_grasp_target_topic",
                default_value="/visual_cues/left_selected_target_pose",
            ),
            DeclareLaunchArgument(
                "right_grasp_target_topic",
                default_value="/visual_cues/right_selected_target_pose",
            ),
            DeclareLaunchArgument("target_locked_topic", default_value="/visual_cues/target_locked"),
            DeclareLaunchArgument("assisted_grasp_activation_distance", default_value="0.08"),
            DeclareLaunchArgument("assisted_grasp_ramp_duration", default_value="0.40"),
            DeclareLaunchArgument("assisted_grasp_max_cartesian_step", default_value="0.04"),
            DeclareLaunchArgument("assisted_grasp_require_open_gripper", default_value="true"),
            DeclareLaunchArgument("assisted_grasp_gripper_open_fraction", default_value="0.10"),
            DeclareLaunchArgument("assisted_grasp_gripper_close_fraction", default_value="0.03"),
            DeclareLaunchArgument("assisted_grasp_gripper_min", default_value="0.0"),
            DeclareLaunchArgument("assisted_grasp_gripper_max", default_value="0.044"),
            DeclareLaunchArgument("monitor_only", default_value="false"),
            DeclareLaunchArgument("avoidance_solver", default_value="cbf_qp"),
            DeclareLaunchArgument("safety_margin", default_value="0.04"),
            DeclareLaunchArgument("activation_margin", default_value="0.08"),
            DeclareLaunchArgument("target_clearance_margin", default_value="0.008"),
            DeclareLaunchArgument("clearance_filter_alpha", default_value="0.22"),
            DeclareLaunchArgument("baseline_weight", default_value="0.8"),
            DeclareLaunchArgument("max_baseline_joint_error", default_value="0.20"),
            DeclareLaunchArgument("tcp_position_weight", default_value="3.0"),
            DeclareLaunchArgument("tcp_orientation_weight", default_value="2.5"),
            DeclareLaunchArgument("tcp_position_max_step", default_value="0.025"),
            DeclareLaunchArgument("tcp_orientation_max_step", default_value="0.12"),
            DeclareLaunchArgument("wrist_tcp_preservation_enabled", default_value="true"),
            DeclareLaunchArgument("wrist_baseline_weight_scale", default_value="0.25"),
            DeclareLaunchArgument("wrist_tcp_position_weight_scale", default_value="3.0"),
            DeclareLaunchArgument(
                "wrist_tcp_orientation_weight_scale",
                default_value="0.20",
            ),
            DeclareLaunchArgument("latched_baseline_weight_scale", default_value="0.20"),
            DeclareLaunchArgument("avoidance_weight", default_value="3.0"),
            DeclareLaunchArgument("max_adjust_per_joint", default_value="0.035"),
            DeclareLaunchArgument("max_command_step", default_value="0.025"),
            DeclareLaunchArgument("max_command_acceleration", default_value="3.0"),
            DeclareLaunchArgument("max_command_jerk", default_value="0.0"),
            DeclareLaunchArgument("command_governor_enabled", default_value="true"),
            DeclareLaunchArgument("command_governor_max_velocity", default_value="1.00"),
            DeclareLaunchArgument("command_governor_max_acceleration", default_value="2.0"),
            DeclareLaunchArgument("command_governor_max_deceleration", default_value="3.0"),
            DeclareLaunchArgument("command_governor_max_jerk", default_value="15.0"),
            DeclareLaunchArgument("command_governor_position_gain", default_value="5.0"),
            DeclareLaunchArgument(
                "command_governor_position_tolerance",
                default_value="0.0001",
            ),
            DeclareLaunchArgument("max_avoidance_delta", default_value="0.025"),
            DeclareLaunchArgument("avoidance_delta_alpha", default_value="1.0"),
            DeclareLaunchArgument("avoidance_release_ramp_duration", default_value="0.5"),
            DeclareLaunchArgument("avoidance_release_min_scale", default_value="0.15"),
            DeclareLaunchArgument("prefer_z_avoidance", default_value="true"),
            DeclareLaunchArgument("z_gradient_min_abs", default_value="0.15"),
            DeclareLaunchArgument("xy_follow_weight", default_value="1.4"),
            DeclareLaunchArgument("xy_follow_max_step", default_value="0.025"),
            DeclareLaunchArgument("downward_bias_weight", default_value="0.08"),
            DeclareLaunchArgument("downward_bias_step", default_value="0.005"),
            DeclareLaunchArgument("downward_bias_z", default_value="-1.0"),
            DeclareLaunchArgument("downward_tangent_bias_enabled", default_value="true"),
            DeclareLaunchArgument("tangent_escape_enabled", default_value="true"),
            DeclareLaunchArgument("tangent_escape_hand_only", default_value="true"),
            DeclareLaunchArgument("tangent_escape_prefer_downward", default_value="true"),
            DeclareLaunchArgument(
                "hand_bypass_allow_autonomous_completion",
                default_value="true",
            ),
            DeclareLaunchArgument("tangent_escape_weight", default_value="4.0"),
            DeclareLaunchArgument("tangent_escape_step", default_value="0.010"),
            DeclareLaunchArgument("tangent_escape_activation_margin", default_value="0.08"),
            DeclareLaunchArgument("tangent_escape_inward_threshold", default_value="0.001"),
            DeclareLaunchArgument("predictive_rollout_enabled", default_value="false"),
            DeclareLaunchArgument("rollout_horizon_steps", default_value="5"),
            DeclareLaunchArgument("rollout_down_offsets", default_value="[0.0, 0.02, 0.04, 0.06]"),
            DeclareLaunchArgument("rollout_side_offsets", default_value="[0.0, -0.02, 0.02]"),
            DeclareLaunchArgument("rollout_side_axis", default_value="[0.0, 1.0, 0.0]"),
            DeclareLaunchArgument("rollout_trigger_margin", default_value="0.04"),
            DeclareLaunchArgument("rollout_max_joint_delta", default_value="0.10"),
            DeclareLaunchArgument("rollout_cartesian_damping", default_value="0.04"),
            DeclareLaunchArgument("rollout_collision_weight", default_value="1200.0"),
            DeclareLaunchArgument("rollout_activation_weight", default_value="40.0"),
            DeclareLaunchArgument("rollout_xy_weight", default_value="70.0"),
            DeclareLaunchArgument("rollout_z_weight", default_value="8.0"),
            DeclareLaunchArgument("rollout_joint_weight", default_value="4.0"),
            DeclareLaunchArgument("rollout_smoothness_weight", default_value="12.0"),
            DeclareLaunchArgument("hold_on_invalid_clearance", default_value="true"),
            DeclareLaunchArgument("avoidance_latch_enabled", default_value="false"),
            DeclareLaunchArgument("avoidance_release_margin", default_value="0.10"),
            DeclareLaunchArgument("avoidance_release_cycles", default_value="3"),
            DeclareLaunchArgument("cbf_gain", default_value="4.0"),
            DeclareLaunchArgument("cbf_slack_weight", default_value="120.0"),
            DeclareLaunchArgument("cbf_max_esdf_constraints", default_value="18"),
            DeclareLaunchArgument("cbf_max_inter_arm_constraints", default_value="16"),
            DeclareLaunchArgument("cbf_max_iterations", default_value="40"),
            DeclareLaunchArgument("cbf_ftol", default_value="1e-4"),
            DeclareLaunchArgument("cbf_fallback_to_soft", default_value="true"),
            DeclareLaunchArgument("cbf_qp_backend", default_value="osqp"),
            DeclareLaunchArgument("cbf_osqp_max_iterations", default_value="200"),
            DeclareLaunchArgument("cbf_osqp_eps_abs", default_value="1e-4"),
            DeclareLaunchArgument("cbf_osqp_eps_rel", default_value="1e-4"),
            DeclareLaunchArgument("cbf_osqp_time_limit_s", default_value="0.004"),
            DeclareLaunchArgument("cbf_master_motion_sync_enabled", default_value="true"),
            DeclareLaunchArgument("cbf_master_motion_enter_velocity", default_value="0.02"),
            DeclareLaunchArgument(
                "cbf_master_motion_release_velocity",
                default_value="0.01",
            ),
            DeclareLaunchArgument("cbf_master_motion_max_step_gain", default_value="1.60"),
            DeclareLaunchArgument(
                "cbf_master_motion_catchup_error_gain",
                default_value="0.08",
            ),
            DeclareLaunchArgument(
                "cbf_master_motion_max_catchup_step",
                default_value="0.012",
            ),
            DeclareLaunchArgument(
                "cbf_safe_baseline_catchup_enabled",
                default_value="true",
            ),
            DeclareLaunchArgument(
                "cbf_safe_baseline_catchup_margin",
                default_value="0.005",
            ),
            DeclareLaunchArgument("antisway_enabled", default_value="true"),
            DeclareLaunchArgument("antisway_monitor_only", default_value="false"),
            DeclareLaunchArgument("start_antisway_observer", default_value="true"),
            DeclareLaunchArgument("antisway_imu_topic", default_value="/camera/imu"),
            DeclareLaunchArgument("antisway_observer_rate_hz", default_value="100.0"),
            DeclareLaunchArgument(
                "antisway_observer_calibration_duration",
                default_value="2.0",
            ),
            DeclareLaunchArgument(
                "antisway_observer_sensor_timeout",
                default_value="0.15",
            ),
            DeclareLaunchArgument(
                "antisway_observer_disturbance_cutoff_hz",
                default_value="1.0",
            ),
            DeclareLaunchArgument(
                "antisway_observer_disturbance_process_noise",
                default_value="1e-4",
            ),
            DeclareLaunchArgument(
                "antisway_modal_state_topic",
                default_value="/openarmx/antisway/modal_state",
            ),
            DeclareLaunchArgument(
                "antisway_observer_diagnostics_topic",
                default_value="/openarmx/antisway/observer_diagnostics",
            ),
            DeclareLaunchArgument(
                "antisway_observer_valid_topic",
                default_value="/openarmx/antisway/observer_valid",
            ),
            DeclareLaunchArgument(
                "antisway_diagnostics_topic",
                default_value="/openarmx/antisway/controller_diagnostics",
            ),
            DeclareLaunchArgument("antisway_observer_timeout_s", default_value="0.15"),
            DeclareLaunchArgument("antisway_modal_weight", default_value="0.02"),
            DeclareLaunchArgument(
                "antisway_acceleration_change_weight",
                default_value="0.0002",
            ),
            DeclareLaunchArgument("antisway_acceleration_scale", default_value="10.0"),
            DeclareLaunchArgument("antisway_nis_full_confidence", default_value="6.0"),
            DeclareLaunchArgument("antisway_nis_zero_confidence", default_value="20.0"),
            DeclareLaunchArgument(
                "antisway_confidence_rise_time_s",
                default_value="0.25",
            ),
            DeclareLaunchArgument(
                "antisway_confidence_fall_time_s",
                default_value="0.20",
            ),
            DeclareLaunchArgument(
                "antisway_confidence_enter_threshold",
                default_value="0.35",
            ),
            DeclareLaunchArgument(
                "antisway_confidence_exit_threshold",
                default_value="0.10",
            ),
            DeclareLaunchArgument("antisway_input_shaper_enabled", default_value="true"),
            DeclareLaunchArgument(
                "antisway_input_shaper_quality_factors",
                default_value="[3.0, 3.0]",
            ),
            DeclareLaunchArgument(
                "antisway_input_shaper_strengths",
                default_value="[0.45, 0.50]",
            ),
            DeclareLaunchArgument(
                "antisway_input_shaper_max_correction",
                default_value="0.02",
            ),
            DeclareLaunchArgument(
                "antisway_input_shaper_max_correction_rate",
                default_value="0.12",
            ),
            DeclareLaunchArgument("antisway_predictive_enabled", default_value="false"),
            DeclareLaunchArgument("antisway_horizon_steps", default_value="12"),
            DeclareLaunchArgument("antisway_roll_weight", default_value="20.0"),
            DeclareLaunchArgument("antisway_yaw_weight", default_value="40.0"),
            DeclareLaunchArgument("antisway_tracking_weight", default_value="4.0"),
            DeclareLaunchArgument("antisway_velocity_weight", default_value="0.05"),
            DeclareLaunchArgument(
                "antisway_mpc_acceleration_weight",
                default_value="0.02",
            ),
            DeclareLaunchArgument(
                "antisway_mpc_acceleration_change_weight",
                default_value="0.20",
            ),
            DeclareLaunchArgument("antisway_terminal_weight_scale", default_value="2.0"),
            DeclareLaunchArgument("antisway_max_acceleration", default_value="6.0"),
            DeclareLaunchArgument("antisway_max_velocity", default_value="1.0"),
            DeclareLaunchArgument(
                "antisway_max_reference_deviation",
                default_value="0.15",
            ),
            DeclareLaunchArgument(
                "antisway_baseline_velocity_alpha",
                default_value="0.20",
            ),
            DeclareLaunchArgument(
                "antisway_baseline_velocity_deadband",
                default_value="0.01",
            ),
            DeclareLaunchArgument(
                "antisway_baseline_velocity_limit",
                default_value="1.20",
            ),
            DeclareLaunchArgument("antisway_motion_gate_enabled", default_value="true"),
            DeclareLaunchArgument("antisway_motion_gate_velocity", default_value="0.03"),
            DeclareLaunchArgument(
                "antisway_motion_gate_release_velocity",
                default_value="0.015",
            ),
            DeclareLaunchArgument(
                "antisway_bimanual_task_qp_enabled",
                default_value="false",
            ),
            DeclareLaunchArgument(
                "antisway_bimanual_task_translation_weight",
                default_value="0.20",
            ),
            DeclareLaunchArgument(
                "antisway_bimanual_task_rotation_weight",
                default_value="0.10",
            ),
            DeclareLaunchArgument(
                "antisway_bimanual_correction_weight",
                default_value="0.002",
            ),
            DeclareLaunchArgument(
                "antisway_bimanual_max_joint_acceleration",
                default_value="1.5",
            ),
            DeclareLaunchArgument(
                "antisway_bimanual_position_horizon_s",
                default_value="0.06",
            ),
            DeclareLaunchArgument(
                "antisway_bimanual_max_position_correction",
                default_value="0.004",
            ),
            DeclareLaunchArgument(
                "antisway_bimanual_max_correction_rate",
                default_value="0.04",
            ),
            DeclareLaunchArgument(
                "antisway_bimanual_acceleration_alpha",
                default_value="0.20",
            ),
            DeclareLaunchArgument(
                "antisway_bimanual_min_confidence",
                default_value="0.20",
            ),
            DeclareLaunchArgument(
                "antisway_bimanual_qp_time_limit_s",
                default_value="0.002",
            ),
            DeclareLaunchArgument("antisway_mpc_max_iterations", default_value="200"),
            DeclareLaunchArgument("antisway_mpc_eps_abs", default_value="1e-4"),
            DeclareLaunchArgument("antisway_mpc_eps_rel", default_value="1e-4"),
            DeclareLaunchArgument("antisway_mpc_time_limit_s", default_value="0.004"),
            DeclareLaunchArgument("aabb_padding", default_value="0.25"),
            DeclareLaunchArgument("enable_inter_arm_collision", default_value="true"),
            DeclareLaunchArgument("inter_arm_safety_margin", default_value="0.03"),
            DeclareLaunchArgument("inter_arm_activation_margin", default_value="0.09"),
            DeclareLaunchArgument("inter_arm_weight", default_value="0.8"),
            DeclareLaunchArgument("inter_arm_skip_proximal_spheres", default_value="3"),
            DeclareLaunchArgument("clear_robot_from_esdf", default_value="false"),
            DeclareLaunchArgument("clear_robot_padding", default_value="0.015"),
            DeclareLaunchArgument("clear_robot_radius_scale", default_value="1.0"),
            DeclareLaunchArgument("publish_avoidance_arrows", default_value="true"),
            DeclareLaunchArgument("avoidance_arrow_max_count", default_value="6"),
            DeclareLaunchArgument("avoidance_arrow_min_length", default_value="0.025"),
            DeclareLaunchArgument("avoidance_arrow_max_length", default_value="0.12"),
            DeclareLaunchArgument(
                "structured_status_topic",
                default_value="/openarmx/bimanual/safety_status",
            ),
            DeclareLaunchArgument("record_safety_metrics", default_value="false"),
            DeclareLaunchArgument("safety_metrics_output", default_value=""),
            Node(
                package="openarmx_obstacle_avoidance",
                executable="bimanual_modal_observer",
                name="bimanual_modal_observer",
                output="screen",
                condition=IfCondition(
                    LaunchConfiguration("start_antisway_observer")
                ),
                parameters=[
                    {
                        "imu_topic": LaunchConfiguration("antisway_imu_topic"),
                        "joint_states_topic": LaunchConfiguration(
                            "joint_states_topic"
                        ),
                        "rate_hz": LaunchConfiguration(
                            "antisway_observer_rate_hz"
                        ),
                        "calibration_duration": LaunchConfiguration(
                            "antisway_observer_calibration_duration"
                        ),
                        "sensor_timeout": LaunchConfiguration(
                            "antisway_observer_sensor_timeout"
                        ),
                        "disturbance_cutoff_hz": LaunchConfiguration(
                            "antisway_observer_disturbance_cutoff_hz"
                        ),
                        "disturbance_process_noise": LaunchConfiguration(
                            "antisway_observer_disturbance_process_noise"
                        ),
                    }
                ],
            ),
            Node(
                package="openarmx_obstacle_avoidance",
                executable="bimanual_esdf_avoidance_filter",
                name="bimanual_esdf_avoidance_filter",
                output="screen",
                additional_env={
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONPATH": _python_vendor_path(),
                },
                parameters=[
                    {
                        "urdf_path": LaunchConfiguration("urdf_path"),
                        "robot_description_node": LaunchConfiguration("robot_description_node"),
                        "left_input_command_topic": LaunchConfiguration("left_input_command_topic"),
                        "right_input_command_topic": LaunchConfiguration("right_input_command_topic"),
                        "left_output_command_topic": LaunchConfiguration("left_output_command_topic"),
                        "right_output_command_topic": LaunchConfiguration("right_output_command_topic"),
                        "joint_states_topic": LaunchConfiguration("joint_states_topic"),
                        "joint_state_timeout_s": LaunchConfiguration("joint_state_timeout_s"),
                        "use_measured_joint_state_start": LaunchConfiguration(
                            "use_measured_joint_state_start"
                        ),
                        "esdf_service": LaunchConfiguration("esdf_service"),
                        "global_frame": LaunchConfiguration("global_frame"),
                        "rate_hz": LaunchConfiguration("rate_hz"),
                        "esdf_update_hz": LaunchConfiguration("esdf_update_hz"),
                        "request_update_esdf": LaunchConfiguration("request_update_esdf"),
                        "min_valid_esdf_distance": LaunchConfiguration("min_valid_esdf_distance"),
                        "nearest_observed_search_radius": LaunchConfiguration("nearest_observed_search_radius"),
                        "enable_esdf_virtual_extension": LaunchConfiguration("enable_esdf_virtual_extension"),
                        "esdf_extension_obstacle_direction": LaunchConfiguration(
                            "esdf_extension_obstacle_direction"
                        ),
                        "esdf_extension_length": LaunchConfiguration("esdf_extension_length"),
                        "esdf_extension_step": LaunchConfiguration("esdf_extension_step"),
                        "obstacle_source": LaunchConfiguration("obstacle_source"),
                        "cable_capsule_topic": LaunchConfiguration("cable_capsule_topic"),
                        "protected_cable_topic": LaunchConfiguration("protected_cable_topic"),
                        "ground_truth_cable_topic": LaunchConfiguration(
                            "ground_truth_cable_topic"
                        ),
                        "legacy_cable_marker_fallback": LaunchConfiguration(
                            "legacy_cable_marker_fallback"
                        ),
                        "cable_capsule_timeout_s": LaunchConfiguration("cable_capsule_timeout_s"),
                        "cable_capsule_timeout_behavior": LaunchConfiguration(
                            "cable_capsule_timeout_behavior"
                        ),
                        "cable_capsule_radius_scale": LaunchConfiguration(
                            "cable_capsule_radius_scale"
                        ),
                        "cable_capsule_padding": LaunchConfiguration("cable_capsule_padding"),
                        "cable_uncertainty_sigma_scale": LaunchConfiguration(
                            "cable_uncertainty_sigma_scale"
                        ),
                        "cable_capsule_max_count": LaunchConfiguration(
                            "cable_capsule_max_count"
                        ),
                        "untangle_mode_topic": LaunchConfiguration("untangle_mode_topic"),
                        "untangle_tangent_weight": LaunchConfiguration(
                            "untangle_tangent_weight"
                        ),
                        "untangle_tangent_max_step": LaunchConfiguration(
                            "untangle_tangent_max_step"
                        ),
                        "collision_model": LaunchConfiguration("collision_model"),
                        "capsule_sample_spacing": LaunchConfiguration("capsule_sample_spacing"),
                        "capsule_min_samples": LaunchConfiguration("capsule_min_samples"),
                        "capsule_samples_per_link": LaunchConfiguration("capsule_samples_per_link"),
                        "end_effector_collision_padding": LaunchConfiguration(
                            "end_effector_collision_padding"
                        ),
                        "esdf_skip_proximal_spheres": LaunchConfiguration("esdf_skip_proximal_spheres"),
                        "assisted_grasp_enabled": LaunchConfiguration("assisted_grasp_enabled"),
                        "left_grasp_target_topic": LaunchConfiguration("left_grasp_target_topic"),
                        "right_grasp_target_topic": LaunchConfiguration("right_grasp_target_topic"),
                        "target_locked_topic": LaunchConfiguration("target_locked_topic"),
                        "assisted_grasp_activation_distance": LaunchConfiguration(
                            "assisted_grasp_activation_distance"
                        ),
                        "assisted_grasp_ramp_duration": LaunchConfiguration(
                            "assisted_grasp_ramp_duration"
                        ),
                        "assisted_grasp_max_cartesian_step": LaunchConfiguration(
                            "assisted_grasp_max_cartesian_step"
                        ),
                        "assisted_grasp_require_open_gripper": LaunchConfiguration(
                            "assisted_grasp_require_open_gripper"
                        ),
                        "assisted_grasp_gripper_open_fraction": LaunchConfiguration(
                            "assisted_grasp_gripper_open_fraction"
                        ),
                        "assisted_grasp_gripper_close_fraction": LaunchConfiguration(
                            "assisted_grasp_gripper_close_fraction"
                        ),
                        "assisted_grasp_gripper_min": LaunchConfiguration(
                            "assisted_grasp_gripper_min"
                        ),
                        "assisted_grasp_gripper_max": LaunchConfiguration(
                            "assisted_grasp_gripper_max"
                        ),
                        "monitor_only": LaunchConfiguration("monitor_only"),
                        "avoidance_solver": LaunchConfiguration("avoidance_solver"),
                        "safety_margin": LaunchConfiguration("safety_margin"),
                        "activation_margin": LaunchConfiguration("activation_margin"),
                        "target_clearance_margin": LaunchConfiguration("target_clearance_margin"),
                        "clearance_filter_alpha": LaunchConfiguration("clearance_filter_alpha"),
                        "baseline_weight": LaunchConfiguration("baseline_weight"),
                        "max_baseline_joint_error": LaunchConfiguration(
                            "max_baseline_joint_error"
                        ),
                        "tcp_position_weight": LaunchConfiguration("tcp_position_weight"),
                        "tcp_orientation_weight": LaunchConfiguration("tcp_orientation_weight"),
                        "tcp_position_max_step": LaunchConfiguration("tcp_position_max_step"),
                        "tcp_orientation_max_step": LaunchConfiguration(
                            "tcp_orientation_max_step"
                        ),
                        "wrist_tcp_preservation_enabled": LaunchConfiguration(
                            "wrist_tcp_preservation_enabled"
                        ),
                        "wrist_baseline_weight_scale": LaunchConfiguration(
                            "wrist_baseline_weight_scale"
                        ),
                        "wrist_tcp_position_weight_scale": LaunchConfiguration(
                            "wrist_tcp_position_weight_scale"
                        ),
                        "wrist_tcp_orientation_weight_scale": LaunchConfiguration(
                            "wrist_tcp_orientation_weight_scale"
                        ),
                        "latched_baseline_weight_scale": LaunchConfiguration(
                            "latched_baseline_weight_scale"
                        ),
                        "avoidance_weight": LaunchConfiguration("avoidance_weight"),
                        "max_adjust_per_joint": LaunchConfiguration("max_adjust_per_joint"),
                        "max_command_step": LaunchConfiguration("max_command_step"),
                        "max_command_acceleration": LaunchConfiguration(
                            "max_command_acceleration"
                        ),
                        "max_command_jerk": LaunchConfiguration("max_command_jerk"),
                        "command_governor_enabled": LaunchConfiguration(
                            "command_governor_enabled"
                        ),
                        "command_governor_max_velocity": LaunchConfiguration(
                            "command_governor_max_velocity"
                        ),
                        "command_governor_max_acceleration": LaunchConfiguration(
                            "command_governor_max_acceleration"
                        ),
                        "command_governor_max_deceleration": LaunchConfiguration(
                            "command_governor_max_deceleration"
                        ),
                        "command_governor_max_jerk": LaunchConfiguration(
                            "command_governor_max_jerk"
                        ),
                        "command_governor_position_gain": LaunchConfiguration(
                            "command_governor_position_gain"
                        ),
                        "command_governor_position_tolerance": LaunchConfiguration(
                            "command_governor_position_tolerance"
                        ),
                        "max_avoidance_delta": LaunchConfiguration("max_avoidance_delta"),
                        "avoidance_delta_alpha": LaunchConfiguration("avoidance_delta_alpha"),
                        "avoidance_release_ramp_duration": LaunchConfiguration(
                            "avoidance_release_ramp_duration"
                        ),
                        "avoidance_release_min_scale": LaunchConfiguration(
                            "avoidance_release_min_scale"
                        ),
                        "prefer_z_avoidance": LaunchConfiguration("prefer_z_avoidance"),
                        "z_gradient_min_abs": LaunchConfiguration("z_gradient_min_abs"),
                        "xy_follow_weight": LaunchConfiguration("xy_follow_weight"),
                        "xy_follow_max_step": LaunchConfiguration("xy_follow_max_step"),
                        "downward_bias_weight": LaunchConfiguration("downward_bias_weight"),
                        "downward_bias_step": LaunchConfiguration("downward_bias_step"),
                        "downward_bias_z": LaunchConfiguration("downward_bias_z"),
                        "downward_tangent_bias_enabled": LaunchConfiguration(
                            "downward_tangent_bias_enabled"
                        ),
                        "tangent_escape_enabled": LaunchConfiguration("tangent_escape_enabled"),
                        "tangent_escape_hand_only": LaunchConfiguration(
                            "tangent_escape_hand_only"
                        ),
                        "tangent_escape_prefer_downward": LaunchConfiguration(
                            "tangent_escape_prefer_downward"
                        ),
                        "hand_bypass_allow_autonomous_completion": LaunchConfiguration(
                            "hand_bypass_allow_autonomous_completion"
                        ),
                        "tangent_escape_weight": LaunchConfiguration("tangent_escape_weight"),
                        "tangent_escape_step": LaunchConfiguration("tangent_escape_step"),
                        "tangent_escape_activation_margin": LaunchConfiguration(
                            "tangent_escape_activation_margin"
                        ),
                        "tangent_escape_inward_threshold": LaunchConfiguration(
                            "tangent_escape_inward_threshold"
                        ),
                        "predictive_rollout_enabled": LaunchConfiguration("predictive_rollout_enabled"),
                        "rollout_horizon_steps": LaunchConfiguration("rollout_horizon_steps"),
                        "rollout_down_offsets": LaunchConfiguration("rollout_down_offsets"),
                        "rollout_side_offsets": LaunchConfiguration("rollout_side_offsets"),
                        "rollout_side_axis": LaunchConfiguration("rollout_side_axis"),
                        "rollout_trigger_margin": LaunchConfiguration("rollout_trigger_margin"),
                        "rollout_max_joint_delta": LaunchConfiguration("rollout_max_joint_delta"),
                        "rollout_cartesian_damping": LaunchConfiguration("rollout_cartesian_damping"),
                        "rollout_collision_weight": LaunchConfiguration("rollout_collision_weight"),
                        "rollout_activation_weight": LaunchConfiguration("rollout_activation_weight"),
                        "rollout_xy_weight": LaunchConfiguration("rollout_xy_weight"),
                        "rollout_z_weight": LaunchConfiguration("rollout_z_weight"),
                        "rollout_joint_weight": LaunchConfiguration("rollout_joint_weight"),
                        "rollout_smoothness_weight": LaunchConfiguration("rollout_smoothness_weight"),
                        "hold_on_invalid_clearance": LaunchConfiguration("hold_on_invalid_clearance"),
                        "avoidance_latch_enabled": LaunchConfiguration("avoidance_latch_enabled"),
                        "avoidance_release_margin": LaunchConfiguration("avoidance_release_margin"),
                        "avoidance_release_cycles": LaunchConfiguration("avoidance_release_cycles"),
                        "cbf_gain": LaunchConfiguration("cbf_gain"),
                        "cbf_slack_weight": LaunchConfiguration("cbf_slack_weight"),
                        "cbf_max_esdf_constraints": LaunchConfiguration("cbf_max_esdf_constraints"),
                        "cbf_max_inter_arm_constraints": LaunchConfiguration("cbf_max_inter_arm_constraints"),
                        "cbf_max_iterations": LaunchConfiguration("cbf_max_iterations"),
                        "cbf_ftol": LaunchConfiguration("cbf_ftol"),
                        "cbf_fallback_to_soft": LaunchConfiguration("cbf_fallback_to_soft"),
                        "cbf_qp_backend": LaunchConfiguration("cbf_qp_backend"),
                        "cbf_osqp_max_iterations": LaunchConfiguration(
                            "cbf_osqp_max_iterations"
                        ),
                        "cbf_osqp_eps_abs": LaunchConfiguration("cbf_osqp_eps_abs"),
                        "cbf_osqp_eps_rel": LaunchConfiguration("cbf_osqp_eps_rel"),
                        "cbf_osqp_time_limit_s": LaunchConfiguration(
                            "cbf_osqp_time_limit_s"
                        ),
                        "cbf_master_motion_sync_enabled": LaunchConfiguration(
                            "cbf_master_motion_sync_enabled"
                        ),
                        "cbf_master_motion_enter_velocity": LaunchConfiguration(
                            "cbf_master_motion_enter_velocity"
                        ),
                        "cbf_master_motion_release_velocity": LaunchConfiguration(
                            "cbf_master_motion_release_velocity"
                        ),
                        "cbf_master_motion_max_step_gain": LaunchConfiguration(
                            "cbf_master_motion_max_step_gain"
                        ),
                        "cbf_master_motion_catchup_error_gain": LaunchConfiguration(
                            "cbf_master_motion_catchup_error_gain"
                        ),
                        "cbf_master_motion_max_catchup_step": LaunchConfiguration(
                            "cbf_master_motion_max_catchup_step"
                        ),
                        "cbf_safe_baseline_catchup_enabled": LaunchConfiguration(
                            "cbf_safe_baseline_catchup_enabled"
                        ),
                        "cbf_safe_baseline_catchup_margin": LaunchConfiguration(
                            "cbf_safe_baseline_catchup_margin"
                        ),
                        "antisway_enabled": LaunchConfiguration("antisway_enabled"),
                        "antisway_monitor_only": LaunchConfiguration("antisway_monitor_only"),
                        "antisway_modal_state_topic": LaunchConfiguration(
                            "antisway_modal_state_topic"
                        ),
                        "antisway_observer_diagnostics_topic": LaunchConfiguration(
                            "antisway_observer_diagnostics_topic"
                        ),
                        "antisway_observer_valid_topic": LaunchConfiguration(
                            "antisway_observer_valid_topic"
                        ),
                        "antisway_diagnostics_topic": LaunchConfiguration(
                            "antisway_diagnostics_topic"
                        ),
                        "antisway_observer_timeout_s": LaunchConfiguration(
                            "antisway_observer_timeout_s"
                        ),
                        "antisway_modal_weight": LaunchConfiguration("antisway_modal_weight"),
                        "antisway_acceleration_change_weight": LaunchConfiguration(
                            "antisway_acceleration_change_weight"
                        ),
                        "antisway_acceleration_scale": LaunchConfiguration(
                            "antisway_acceleration_scale"
                        ),
                        "antisway_nis_full_confidence": LaunchConfiguration(
                            "antisway_nis_full_confidence"
                        ),
                        "antisway_nis_zero_confidence": LaunchConfiguration(
                            "antisway_nis_zero_confidence"
                        ),
                        "antisway_confidence_rise_time_s": LaunchConfiguration(
                            "antisway_confidence_rise_time_s"
                        ),
                        "antisway_confidence_fall_time_s": LaunchConfiguration(
                            "antisway_confidence_fall_time_s"
                        ),
                        "antisway_confidence_enter_threshold": LaunchConfiguration(
                            "antisway_confidence_enter_threshold"
                        ),
                        "antisway_confidence_exit_threshold": LaunchConfiguration(
                            "antisway_confidence_exit_threshold"
                        ),
                        "antisway_input_shaper_enabled": LaunchConfiguration(
                            "antisway_input_shaper_enabled"
                        ),
                        "antisway_input_shaper_quality_factors": LaunchConfiguration(
                            "antisway_input_shaper_quality_factors"
                        ),
                        "antisway_input_shaper_strengths": LaunchConfiguration(
                            "antisway_input_shaper_strengths"
                        ),
                        "antisway_input_shaper_max_correction": LaunchConfiguration(
                            "antisway_input_shaper_max_correction"
                        ),
                        "antisway_input_shaper_max_correction_rate": LaunchConfiguration(
                            "antisway_input_shaper_max_correction_rate"
                        ),
                        "antisway_predictive_enabled": LaunchConfiguration(
                            "antisway_predictive_enabled"
                        ),
                        "antisway_horizon_steps": LaunchConfiguration(
                            "antisway_horizon_steps"
                        ),
                        "antisway_roll_weight": LaunchConfiguration(
                            "antisway_roll_weight"
                        ),
                        "antisway_yaw_weight": LaunchConfiguration(
                            "antisway_yaw_weight"
                        ),
                        "antisway_tracking_weight": LaunchConfiguration(
                            "antisway_tracking_weight"
                        ),
                        "antisway_velocity_weight": LaunchConfiguration(
                            "antisway_velocity_weight"
                        ),
                        "antisway_mpc_acceleration_weight": LaunchConfiguration(
                            "antisway_mpc_acceleration_weight"
                        ),
                        "antisway_mpc_acceleration_change_weight": LaunchConfiguration(
                            "antisway_mpc_acceleration_change_weight"
                        ),
                        "antisway_terminal_weight_scale": LaunchConfiguration(
                            "antisway_terminal_weight_scale"
                        ),
                        "antisway_max_acceleration": LaunchConfiguration(
                            "antisway_max_acceleration"
                        ),
                        "antisway_max_velocity": LaunchConfiguration(
                            "antisway_max_velocity"
                        ),
                        "antisway_max_reference_deviation": LaunchConfiguration(
                            "antisway_max_reference_deviation"
                        ),
                        "antisway_baseline_velocity_alpha": LaunchConfiguration(
                            "antisway_baseline_velocity_alpha"
                        ),
                        "antisway_baseline_velocity_deadband": LaunchConfiguration(
                            "antisway_baseline_velocity_deadband"
                        ),
                        "antisway_baseline_velocity_limit": LaunchConfiguration(
                            "antisway_baseline_velocity_limit"
                        ),
                        "antisway_motion_gate_enabled": LaunchConfiguration(
                            "antisway_motion_gate_enabled"
                        ),
                        "antisway_motion_gate_velocity": LaunchConfiguration(
                            "antisway_motion_gate_velocity"
                        ),
                        "antisway_motion_gate_release_velocity": LaunchConfiguration(
                            "antisway_motion_gate_release_velocity"
                        ),
                        "antisway_bimanual_task_qp_enabled": LaunchConfiguration(
                            "antisway_bimanual_task_qp_enabled"
                        ),
                        "antisway_bimanual_task_translation_weight": LaunchConfiguration(
                            "antisway_bimanual_task_translation_weight"
                        ),
                        "antisway_bimanual_task_rotation_weight": LaunchConfiguration(
                            "antisway_bimanual_task_rotation_weight"
                        ),
                        "antisway_bimanual_correction_weight": LaunchConfiguration(
                            "antisway_bimanual_correction_weight"
                        ),
                        "antisway_bimanual_max_joint_acceleration": LaunchConfiguration(
                            "antisway_bimanual_max_joint_acceleration"
                        ),
                        "antisway_bimanual_position_horizon_s": LaunchConfiguration(
                            "antisway_bimanual_position_horizon_s"
                        ),
                        "antisway_bimanual_max_position_correction": LaunchConfiguration(
                            "antisway_bimanual_max_position_correction"
                        ),
                        "antisway_bimanual_max_correction_rate": LaunchConfiguration(
                            "antisway_bimanual_max_correction_rate"
                        ),
                        "antisway_bimanual_acceleration_alpha": LaunchConfiguration(
                            "antisway_bimanual_acceleration_alpha"
                        ),
                        "antisway_bimanual_min_confidence": LaunchConfiguration(
                            "antisway_bimanual_min_confidence"
                        ),
                        "antisway_bimanual_qp_time_limit_s": LaunchConfiguration(
                            "antisway_bimanual_qp_time_limit_s"
                        ),
                        "antisway_mpc_max_iterations": LaunchConfiguration(
                            "antisway_mpc_max_iterations"
                        ),
                        "antisway_mpc_eps_abs": LaunchConfiguration(
                            "antisway_mpc_eps_abs"
                        ),
                        "antisway_mpc_eps_rel": LaunchConfiguration(
                            "antisway_mpc_eps_rel"
                        ),
                        "antisway_mpc_time_limit_s": LaunchConfiguration(
                            "antisway_mpc_time_limit_s"
                        ),
                        "aabb_padding": LaunchConfiguration("aabb_padding"),
                        "enable_inter_arm_collision": LaunchConfiguration("enable_inter_arm_collision"),
                        "inter_arm_safety_margin": LaunchConfiguration("inter_arm_safety_margin"),
                        "inter_arm_activation_margin": LaunchConfiguration("inter_arm_activation_margin"),
                        "inter_arm_weight": LaunchConfiguration("inter_arm_weight"),
                        "inter_arm_skip_proximal_spheres": LaunchConfiguration("inter_arm_skip_proximal_spheres"),
                        "clear_robot_from_esdf": LaunchConfiguration("clear_robot_from_esdf"),
                        "clear_robot_padding": LaunchConfiguration("clear_robot_padding"),
                        "clear_robot_radius_scale": LaunchConfiguration("clear_robot_radius_scale"),
                        "publish_avoidance_arrows": LaunchConfiguration(
                            "publish_avoidance_arrows"
                        ),
                        "avoidance_arrow_max_count": LaunchConfiguration(
                            "avoidance_arrow_max_count"
                        ),
                        "avoidance_arrow_min_length": LaunchConfiguration(
                            "avoidance_arrow_min_length"
                        ),
                        "avoidance_arrow_max_length": LaunchConfiguration(
                            "avoidance_arrow_max_length"
                        ),
                        "structured_status_topic": LaunchConfiguration(
                            "structured_status_topic"
                        ),
                    }
                ],
            ),
            Node(
                package="openarmx_obstacle_avoidance",
                executable="safety_metrics_recorder",
                name="safety_metrics_recorder",
                output="screen",
                condition=IfCondition(LaunchConfiguration("record_safety_metrics")),
                parameters=[
                    {
                        "status_topic": LaunchConfiguration("structured_status_topic"),
                        "output_path": LaunchConfiguration("safety_metrics_output"),
                    }
                ],
            ),
        ]
    )
