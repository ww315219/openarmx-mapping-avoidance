from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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
            DeclareLaunchArgument("cable_capsule_timeout_s", default_value="0.75"),
            DeclareLaunchArgument(
                "cable_capsule_timeout_behavior",
                default_value="hold",
                description="Behavior when cable capsules are missing: hold, fallback_esdf, or pass_through.",
            ),
            DeclareLaunchArgument("cable_capsule_radius_scale", default_value="1.0"),
            DeclareLaunchArgument("cable_capsule_padding", default_value="0.0"),
            DeclareLaunchArgument("cable_capsule_max_count", default_value="8"),
            DeclareLaunchArgument("untangle_mode_topic", default_value="/openarmx/untangle_mode"),
            DeclareLaunchArgument("untangle_tangent_weight", default_value="1.2"),
            DeclareLaunchArgument("untangle_tangent_max_step", default_value="0.012"),
            DeclareLaunchArgument("collision_model", default_value="capsule"),
            DeclareLaunchArgument("capsule_sample_spacing", default_value="0.06"),
            DeclareLaunchArgument("capsule_min_samples", default_value="2"),
            DeclareLaunchArgument("capsule_samples_per_link", default_value="3"),
            DeclareLaunchArgument("esdf_skip_proximal_spheres", default_value="3"),
            DeclareLaunchArgument("assisted_grasp_enabled", default_value="true"),
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
            DeclareLaunchArgument("latched_baseline_weight_scale", default_value="0.20"),
            DeclareLaunchArgument("avoidance_weight", default_value="3.0"),
            DeclareLaunchArgument("max_adjust_per_joint", default_value="0.035"),
            DeclareLaunchArgument("max_command_step", default_value="0.025"),
            DeclareLaunchArgument("max_avoidance_delta", default_value="0.025"),
            DeclareLaunchArgument("avoidance_delta_alpha", default_value="0.22"),
            DeclareLaunchArgument("prefer_z_avoidance", default_value="true"),
            DeclareLaunchArgument("z_gradient_min_abs", default_value="0.15"),
            DeclareLaunchArgument("xy_follow_weight", default_value="1.4"),
            DeclareLaunchArgument("xy_follow_max_step", default_value="0.025"),
            DeclareLaunchArgument("downward_bias_weight", default_value="0.08"),
            DeclareLaunchArgument("downward_bias_step", default_value="0.005"),
            DeclareLaunchArgument("downward_bias_z", default_value="-1.0"),
            DeclareLaunchArgument("downward_tangent_bias_enabled", default_value="true"),
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
            Node(
                package="openarmx_obstacle_avoidance",
                executable="bimanual_esdf_avoidance_filter",
                name="bimanual_esdf_avoidance_filter",
                output="screen",
                additional_env={"PYTHONNOUSERSITE": "1"},
                parameters=[
                    {
                        "urdf_path": LaunchConfiguration("urdf_path"),
                        "robot_description_node": LaunchConfiguration("robot_description_node"),
                        "left_input_command_topic": LaunchConfiguration("left_input_command_topic"),
                        "right_input_command_topic": LaunchConfiguration("right_input_command_topic"),
                        "left_output_command_topic": LaunchConfiguration("left_output_command_topic"),
                        "right_output_command_topic": LaunchConfiguration("right_output_command_topic"),
                        "joint_states_topic": LaunchConfiguration("joint_states_topic"),
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
                        "cable_capsule_timeout_s": LaunchConfiguration("cable_capsule_timeout_s"),
                        "cable_capsule_timeout_behavior": LaunchConfiguration(
                            "cable_capsule_timeout_behavior"
                        ),
                        "cable_capsule_radius_scale": LaunchConfiguration(
                            "cable_capsule_radius_scale"
                        ),
                        "cable_capsule_padding": LaunchConfiguration("cable_capsule_padding"),
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
                        "latched_baseline_weight_scale": LaunchConfiguration(
                            "latched_baseline_weight_scale"
                        ),
                        "avoidance_weight": LaunchConfiguration("avoidance_weight"),
                        "max_adjust_per_joint": LaunchConfiguration("max_adjust_per_joint"),
                        "max_command_step": LaunchConfiguration("max_command_step"),
                        "max_avoidance_delta": LaunchConfiguration("max_avoidance_delta"),
                        "avoidance_delta_alpha": LaunchConfiguration("avoidance_delta_alpha"),
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
                    }
                ],
            ),
        ]
    )
