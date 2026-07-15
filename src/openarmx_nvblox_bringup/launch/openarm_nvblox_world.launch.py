from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    depth_topic = LaunchConfiguration("depth_topic")
    depth_camera_info_topic = LaunchConfiguration("depth_camera_info_topic")
    run_depth_freeze_gate = LaunchConfiguration("run_depth_freeze_gate")
    frozen_depth_topic = LaunchConfiguration("frozen_depth_topic")
    frozen_depth_camera_info_topic = LaunchConfiguration("frozen_depth_camera_info_topic")
    depth_freeze_after_s = LaunchConfiguration("depth_freeze_after_s")
    depth_freeze_keep_last_frame = LaunchConfiguration("depth_freeze_keep_last_frame")
    run_semantic_obstacle_filter = LaunchConfiguration("run_semantic_obstacle_filter")
    semantic_obstacle_depth_topic = LaunchConfiguration("semantic_obstacle_depth_topic")
    semantic_obstacle_camera_info_topic = LaunchConfiguration("semantic_obstacle_camera_info_topic")
    semantic_robot_mask_topic = LaunchConfiguration("semantic_robot_mask_topic")
    semantic_combined_mask_topic = LaunchConfiguration("semantic_combined_mask_topic")
    semantic_target_mask_topic = LaunchConfiguration("semantic_target_mask_topic")
    semantic_enable_target_mask = LaunchConfiguration("semantic_enable_target_mask")
    semantic_robot_mask_padding_px = LaunchConfiguration("semantic_robot_mask_padding_px")
    semantic_robot_radius_scale = LaunchConfiguration("semantic_robot_radius_scale")
    run_rviz = LaunchConfiguration("run_rviz")
    run_cable_capsules = LaunchConfiguration("run_cable_capsules")
    run_robot_esdf_clearer = LaunchConfiguration("run_robot_esdf_clearer")
    run_visual_cues = LaunchConfiguration("run_visual_cues")
    visual_cues_active_arm = LaunchConfiguration("visual_cues_active_arm")
    visual_cues_left_desired_pose_topic = LaunchConfiguration(
        "visual_cues_left_desired_pose_topic"
    )
    visual_cues_right_desired_pose_topic = LaunchConfiguration(
        "visual_cues_right_desired_pose_topic"
    )
    visual_cues_left_assisted_pose_topic = LaunchConfiguration(
        "visual_cues_left_assisted_pose_topic"
    )
    visual_cues_right_assisted_pose_topic = LaunchConfiguration(
        "visual_cues_right_assisted_pose_topic"
    )
    visual_cues_marker_topic = LaunchConfiguration("visual_cues_marker_topic")
    visual_cues_rate_hz = LaunchConfiguration("visual_cues_rate_hz")
    visual_cues_safety_margin = LaunchConfiguration("visual_cues_safety_margin")
    visual_cues_activation_margin = LaunchConfiguration("visual_cues_activation_margin")
    visual_cues_ready_distance = LaunchConfiguration("visual_cues_ready_distance")
    visual_cues_assisted_grasp_enabled = LaunchConfiguration(
        "visual_cues_assisted_grasp_enabled"
    )
    visual_cues_assist_activation_distance = LaunchConfiguration(
        "visual_cues_assist_activation_distance"
    )
    visual_cues_assist_min_alpha = LaunchConfiguration("visual_cues_assist_min_alpha")
    visual_cues_assist_max_alpha = LaunchConfiguration("visual_cues_assist_max_alpha")
    visual_cues_assist_ramp_duration = LaunchConfiguration("visual_cues_assist_ramp_duration")
    visual_cues_pointcloud_topic = LaunchConfiguration("visual_cues_pointcloud_topic")
    visual_cues_left_gripper_topic = LaunchConfiguration("visual_cues_left_gripper_topic")
    visual_cues_right_gripper_topic = LaunchConfiguration("visual_cues_right_gripper_topic")
    visual_cues_ray_length = LaunchConfiguration("visual_cues_ray_length")
    visual_cues_ray_hit_radius = LaunchConfiguration("visual_cues_ray_hit_radius")
    visual_cues_show_aim_reticle = LaunchConfiguration("visual_cues_show_aim_reticle")
    visual_cues_show_rviz_selection_ray = LaunchConfiguration(
        "visual_cues_show_rviz_selection_ray"
    )
    visual_cues_aim_reticle_distance = LaunchConfiguration("visual_cues_aim_reticle_distance")
    visual_cues_use_aim_reticle_as_fallback_target = LaunchConfiguration(
        "visual_cues_use_aim_reticle_as_fallback_target"
    )
    visual_cues_prefer_aim_reticle_target = LaunchConfiguration(
        "visual_cues_prefer_aim_reticle_target"
    )
    visual_cues_aim_reticle_pick_radius_px = LaunchConfiguration(
        "visual_cues_aim_reticle_pick_radius_px"
    )
    visual_cues_aim_reticle_pick_min_points = LaunchConfiguration(
        "visual_cues_aim_reticle_pick_min_points"
    )
    visual_cues_color_image_topic = LaunchConfiguration("visual_cues_color_image_topic")
    visual_cues_use_compressed_image = LaunchConfiguration("visual_cues_use_compressed_image")
    visual_cues_compressed_color_image_topic = LaunchConfiguration(
        "visual_cues_compressed_color_image_topic"
    )
    visual_cues_color_camera_info_topic = LaunchConfiguration(
        "visual_cues_color_camera_info_topic"
    )
    visual_cues_annotated_image_topic = LaunchConfiguration(
        "visual_cues_annotated_image_topic"
    )
    visual_cues_gripper_open_threshold = LaunchConfiguration(
        "visual_cues_gripper_open_threshold"
    )
    visual_cues_gripper_closed_threshold = LaunchConfiguration(
        "visual_cues_gripper_closed_threshold"
    )
    visual_cues_lock_close_count = LaunchConfiguration("visual_cues_lock_close_count")
    visual_cues_target_tracking_search_radius = LaunchConfiguration(
        "visual_cues_target_tracking_search_radius"
    )
    visual_cues_target_tracking_min_points = LaunchConfiguration(
        "visual_cues_target_tracking_min_points"
    )
    visual_cues_target_roi_pointcloud_topic = LaunchConfiguration(
        "visual_cues_target_roi_pointcloud_topic"
    )
    visual_cues_target_roi_active_topic = LaunchConfiguration(
        "visual_cues_target_roi_active_topic"
    )
    visual_cues_target_bbox_marker_topic = LaunchConfiguration(
        "visual_cues_target_bbox_marker_topic"
    )
    visual_cues_target_roi_max_points = LaunchConfiguration(
        "visual_cues_target_roi_max_points"
    )
    visual_cues_target_bbox_padding = LaunchConfiguration("visual_cues_target_bbox_padding")
    visual_cues_target_bbox_min_size = LaunchConfiguration("visual_cues_target_bbox_min_size")
    visual_cues_target_lock_offset_x = LaunchConfiguration("visual_cues_target_lock_offset_x")
    visual_cues_target_lock_offset_y = LaunchConfiguration("visual_cues_target_lock_offset_y")
    visual_cues_target_lock_offset_z = LaunchConfiguration("visual_cues_target_lock_offset_z")
    visual_cues_show_target_roi = LaunchConfiguration("visual_cues_show_target_roi")
    visual_cues_cable_capsules_topic = LaunchConfiguration("visual_cues_cable_capsules_topic")
    visual_cues_exclude_cable_capsule_points = LaunchConfiguration(
        "visual_cues_exclude_cable_capsule_points"
    )
    visual_cues_cable_capsule_exclusion_padding = LaunchConfiguration(
        "visual_cues_cable_capsule_exclusion_padding"
    )
    visual_cues_show_robot_self_filter = LaunchConfiguration(
        "visual_cues_show_robot_self_filter"
    )
    publish_camera_tf = LaunchConfiguration("publish_camera_tf")
    publish_square_test_tf = LaunchConfiguration("publish_square_test_tf")
    log_level = LaunchConfiguration("log_level")
    robot_description_node = LaunchConfiguration("robot_description_node")
    joint_states_topic = LaunchConfiguration("joint_states_topic")
    robot_clearer_rate_hz = LaunchConfiguration("robot_clearer_rate_hz")
    robot_clearer_aabb_padding = LaunchConfiguration("robot_clearer_aabb_padding")
    robot_clearer_padding = LaunchConfiguration("robot_clearer_padding")
    robot_clearer_radius_scale = LaunchConfiguration("robot_clearer_radius_scale")
    robot_clearer_min_joint_delta = LaunchConfiguration("robot_clearer_min_joint_delta")
    robot_clearer_force_period_s = LaunchConfiguration("robot_clearer_force_period_s")
    robot_clearer_capsule_samples_per_link = LaunchConfiguration(
        "robot_clearer_capsule_samples_per_link"
    )
    robot_clearer_debug_markers = LaunchConfiguration("robot_clearer_debug_markers")
    cable_capsule_topic = LaunchConfiguration("cable_capsule_topic")
    cable_capsule_source_mode = LaunchConfiguration("cable_capsule_source_mode")
    cable_capsule_pointcloud_topic = LaunchConfiguration("cable_capsule_pointcloud_topic")
    cable_capsule_mesh_topic = LaunchConfiguration("cable_capsule_mesh_topic")
    cable_capsule_voxel_layer_topic = LaunchConfiguration("cable_capsule_voxel_layer_topic")
    cable_capsule_voxel_centers_are_global = LaunchConfiguration("cable_capsule_voxel_centers_are_global")
    cable_capsule_radius_m = LaunchConfiguration("cable_capsule_radius_m")
    cable_capsule_alpha = LaunchConfiguration("cable_capsule_alpha")
    cable_capsule_max_capsules = LaunchConfiguration("cable_capsule_max_capsules")
    cable_capsule_min_line_length_px = LaunchConfiguration("cable_capsule_min_line_length_px")
    cable_capsule_max_line_gap_px = LaunchConfiguration("cable_capsule_max_line_gap_px")
    cable_capsule_ransac_inlier_distance_m = LaunchConfiguration("cable_capsule_ransac_inlier_distance_m")
    cable_capsule_ransac_min_inliers = LaunchConfiguration("cable_capsule_ransac_min_inliers")
    cable_capsule_point_voxel_size_m = LaunchConfiguration("cable_capsule_point_voxel_size_m")
    cable_capsule_voxel_fit_mode = LaunchConfiguration("cable_capsule_voxel_fit_mode")
    cable_capsule_left_seed_reference_frame = LaunchConfiguration(
        "cable_capsule_left_seed_reference_frame"
    )
    cable_capsule_right_seed_reference_frame = LaunchConfiguration(
        "cable_capsule_right_seed_reference_frame"
    )
    cable_capsule_seed_axis_neighbor_radius_m = LaunchConfiguration(
        "cable_capsule_seed_axis_neighbor_radius_m"
    )
    cable_capsule_seed_axis_max_gap_m = LaunchConfiguration(
        "cable_capsule_seed_axis_max_gap_m"
    )
    cable_capsule_seed_axis_max_half_length_m = LaunchConfiguration(
        "cable_capsule_seed_axis_max_half_length_m"
    )
    cable_capsule_seed_axis_output_half_length_m = LaunchConfiguration(
        "cable_capsule_seed_axis_output_half_length_m"
    )
    cable_capsule_seed_min_neighbor_voxels = LaunchConfiguration(
        "cable_capsule_seed_min_neighbor_voxels"
    )
    cable_capsule_seed_neighbor_radius_m = LaunchConfiguration(
        "cable_capsule_seed_neighbor_radius_m"
    )
    cable_capsule_mesh_vertices_are_global = LaunchConfiguration("cable_capsule_mesh_vertices_are_global")
    cable_capsule_mesh_direct_fit_enabled = LaunchConfiguration("cable_capsule_mesh_direct_fit_enabled")
    cable_capsule_mesh_cluster_voxel_size_m = LaunchConfiguration("cable_capsule_mesh_cluster_voxel_size_m")
    cable_capsule_mesh_cluster_neighbor_voxels = LaunchConfiguration("cable_capsule_mesh_cluster_neighbor_voxels")
    cable_capsule_mesh_cluster_min_points = LaunchConfiguration("cable_capsule_mesh_cluster_min_points")
    cable_capsule_mesh_cluster_min_length_m = LaunchConfiguration("cable_capsule_mesh_cluster_min_length_m")
    cable_capsule_mesh_cluster_max_radius_m = LaunchConfiguration("cable_capsule_mesh_cluster_max_radius_m")
    cable_capsule_mesh_cluster_endpoint_percentile = LaunchConfiguration("cable_capsule_mesh_cluster_endpoint_percentile")
    cable_capsule_component_min_points = LaunchConfiguration("cable_capsule_component_min_points")
    cable_capsule_component_min_length_m = LaunchConfiguration("cable_capsule_component_min_length_m")
    cable_capsule_component_max_radius_m = LaunchConfiguration("cable_capsule_component_max_radius_m")
    cable_capsule_component_neighbor_voxels = LaunchConfiguration("cable_capsule_component_neighbor_voxels")
    cable_capsule_component_merge_enabled = LaunchConfiguration("cable_capsule_component_merge_enabled")
    cable_capsule_component_merge_gap_m = LaunchConfiguration("cable_capsule_component_merge_gap_m")
    cable_capsule_component_merge_perp_m = LaunchConfiguration("cable_capsule_component_merge_perp_m")
    cable_capsule_component_merge_direction_dot = LaunchConfiguration("cable_capsule_component_merge_direction_dot")
    cable_capsule_ema_alpha = LaunchConfiguration("cable_capsule_ema_alpha")
    cable_capsule_hold_cycles = LaunchConfiguration("cable_capsule_hold_cycles")
    cable_capsule_ransac_fallback_enabled = LaunchConfiguration("cable_capsule_ransac_fallback_enabled")

    camera_x = LaunchConfiguration("camera_x")
    camera_y = LaunchConfiguration("camera_y")
    camera_z = LaunchConfiguration("camera_z")
    camera_qx = LaunchConfiguration("camera_qx")
    camera_qy = LaunchConfiguration("camera_qy")
    camera_qz = LaunchConfiguration("camera_qz")
    camera_qw = LaunchConfiguration("camera_qw")
    world_frame = LaunchConfiguration("world_frame")
    camera_frame = LaunchConfiguration("camera_frame")
    square_test_frame = LaunchConfiguration("square_test_frame")

    config_path = os.path.join(
        get_package_share_directory("openarmx_nvblox_bringup"),
        "config",
        "ffs_nvblox_world.yaml",
    )
    rviz_config_path = os.path.join(
        get_package_share_directory("openarmx_nvblox_bringup"),
        "config",
        "openarm_nvblox_world.rviz",
    )

    camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_ffs_camera_tf",
        arguments=[
            "--x",
            camera_x,
            "--y",
            camera_y,
            "--z",
            camera_z,
            "--qx",
            camera_qx,
            "--qy",
            camera_qy,
            "--qz",
            camera_qz,
            "--qw",
            camera_qw,
            "--frame-id",
            world_frame,
            "--child-frame-id",
            camera_frame,
        ],
        condition=IfCondition(publish_camera_tf),
    )

    square_test_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_square_test_tf",
        arguments=[
            "--x",
            "0.0",
            "--y",
            "0.0",
            "--z",
            "0.0",
            "--qx",
            "0.0",
            "--qy",
            "0.0",
            "--qz",
            "0.0",
            "--qw",
            "1.0",
            "--frame-id",
            world_frame,
            "--child-frame-id",
            square_test_frame,
        ],
        condition=IfCondition(publish_square_test_tf),
    )

    raw_direct_condition = IfCondition(
        PythonExpression([
            "'",
            run_depth_freeze_gate,
            "' == 'false' and '",
            run_semantic_obstacle_filter,
            "' == 'false'",
        ])
    )
    raw_frozen_condition = IfCondition(
        PythonExpression([
            "'",
            run_depth_freeze_gate,
            "' == 'true' and '",
            run_semantic_obstacle_filter,
            "' == 'false'",
        ])
    )
    semantic_direct_condition = IfCondition(
        PythonExpression([
            "'",
            run_depth_freeze_gate,
            "' == 'false' and '",
            run_semantic_obstacle_filter,
            "' == 'true'",
        ])
    )
    semantic_frozen_condition = IfCondition(
        PythonExpression([
            "'",
            run_depth_freeze_gate,
            "' == 'true' and '",
            run_semantic_obstacle_filter,
            "' == 'true'",
        ])
    )

    semantic_obstacle_filter_node = Node(
        package="openarmx_nvblox_bringup",
        executable="semantic_obstacle_depth_filter_node.py",
        name="semantic_obstacle_depth_filter",
        output="screen",
        parameters=[
            {
                "input_depth_topic": depth_topic,
                "input_camera_info_topic": depth_camera_info_topic,
                "output_depth_topic": semantic_obstacle_depth_topic,
                "output_camera_info_topic": semantic_obstacle_camera_info_topic,
                "output_robot_mask_topic": semantic_robot_mask_topic,
                "output_combined_mask_topic": semantic_combined_mask_topic,
                "target_mask_topic": semantic_target_mask_topic,
                "robot_description_node": robot_description_node,
                "joint_states_topic": joint_states_topic,
                "global_frame": world_frame,
                "enable_robot_mask": True,
                "enable_target_mask": semantic_enable_target_mask,
                "robot_mask_padding_px": semantic_robot_mask_padding_px,
                "robot_radius_scale": semantic_robot_radius_scale,
            }
        ],
        condition=IfCondition(run_semantic_obstacle_filter),
    )

    depth_freeze_gate_node_raw = Node(
        package="openarmx_nvblox_bringup",
        executable="depth_freeze_gate_node.py",
        name="depth_freeze_gate",
        output="screen",
        parameters=[
            {
                "input_depth_topic": depth_topic,
                "input_camera_info_topic": depth_camera_info_topic,
                "output_depth_topic": frozen_depth_topic,
                "output_camera_info_topic": frozen_depth_camera_info_topic,
                "freeze_after_s": depth_freeze_after_s,
                "keep_publishing_last_frame": depth_freeze_keep_last_frame,
            }
        ],
        condition=raw_frozen_condition,
    )

    depth_freeze_gate_node_semantic = Node(
        package="openarmx_nvblox_bringup",
        executable="depth_freeze_gate_node.py",
        name="depth_freeze_gate",
        output="screen",
        parameters=[
            {
                "input_depth_topic": semantic_obstacle_depth_topic,
                "input_camera_info_topic": semantic_obstacle_camera_info_topic,
                "output_depth_topic": frozen_depth_topic,
                "output_camera_info_topic": frozen_depth_camera_info_topic,
                "freeze_after_s": depth_freeze_after_s,
                "keep_publishing_last_frame": depth_freeze_keep_last_frame,
            }
        ],
        condition=semantic_frozen_condition,
    )

    nvblox_node_direct = Node(
        package="nvblox_ros",
        executable="nvblox_node",
        name="nvblox_node",
        output="screen",
        parameters=[config_path],
        arguments=["--ros-args", "--log-level", log_level],
        remappings=[
            ("camera_0/depth/image", depth_topic),
            ("camera_0/depth/camera_info", depth_camera_info_topic),
        ],
        condition=raw_direct_condition,
    )

    nvblox_node_frozen = Node(
        package="nvblox_ros",
        executable="nvblox_node",
        name="nvblox_node",
        output="screen",
        parameters=[config_path],
        arguments=["--ros-args", "--log-level", log_level],
        remappings=[
            ("camera_0/depth/image", frozen_depth_topic),
            ("camera_0/depth/camera_info", frozen_depth_camera_info_topic),
        ],
        condition=raw_frozen_condition,
    )

    nvblox_node_semantic = Node(
        package="nvblox_ros",
        executable="nvblox_node",
        name="nvblox_node",
        output="screen",
        parameters=[config_path],
        arguments=["--ros-args", "--log-level", log_level],
        remappings=[
            ("camera_0/depth/image", semantic_obstacle_depth_topic),
            ("camera_0/depth/camera_info", semantic_obstacle_camera_info_topic),
        ],
        condition=semantic_direct_condition,
    )

    nvblox_node_semantic_frozen = Node(
        package="nvblox_ros",
        executable="nvblox_node",
        name="nvblox_node",
        output="screen",
        parameters=[config_path],
        arguments=["--ros-args", "--log-level", log_level],
        remappings=[
            ("camera_0/depth/image", frozen_depth_topic),
            ("camera_0/depth/camera_info", frozen_depth_camera_info_topic),
        ],
        condition=semantic_frozen_condition,
    )

    cable_capsule_marker_node = Node(
        package="openarmx_nvblox_bringup",
        executable="cable_capsule_marker_node.py",
        name="cable_capsule_marker",
        output="screen",
        parameters=[
            {
                "source_mode": cable_capsule_source_mode,
                "depth_topic": depth_topic,
                "camera_info_topic": depth_camera_info_topic,
                "pointcloud_topic": cable_capsule_pointcloud_topic,
                "mesh_topic": cable_capsule_mesh_topic,
                "voxel_layer_topic": cable_capsule_voxel_layer_topic,
                "voxel_centers_are_global": cable_capsule_voxel_centers_are_global,
                "marker_topic": cable_capsule_topic,
                "radius_m": cable_capsule_radius_m,
                "alpha": cable_capsule_alpha,
                "max_capsules": cable_capsule_max_capsules,
                "min_line_length_px": cable_capsule_min_line_length_px,
                "max_line_gap_px": cable_capsule_max_line_gap_px,
                "ransac_inlier_distance_m": cable_capsule_ransac_inlier_distance_m,
                "ransac_min_inliers": cable_capsule_ransac_min_inliers,
                "point_voxel_size_m": cable_capsule_point_voxel_size_m,
                "voxel_fit_mode": cable_capsule_voxel_fit_mode,
                "left_seed_reference_frame": cable_capsule_left_seed_reference_frame,
                "right_seed_reference_frame": cable_capsule_right_seed_reference_frame,
                "seed_reference_x": 0.0,
                "seed_reference_y": 0.0,
                "seed_reference_z": 0.0,
                "seed_axis_x": 1.0,
                "seed_axis_y": 0.0,
                "seed_axis_z": 0.0,
                "seed_axis_neighbor_radius_m": cable_capsule_seed_axis_neighbor_radius_m,
                "seed_axis_max_gap_m": cable_capsule_seed_axis_max_gap_m,
                "seed_axis_max_half_length_m": cable_capsule_seed_axis_max_half_length_m,
                "seed_axis_output_half_length_m": cable_capsule_seed_axis_output_half_length_m,
                "seed_min_neighbor_voxels": cable_capsule_seed_min_neighbor_voxels,
                "seed_neighbor_radius_m": cable_capsule_seed_neighbor_radius_m,
                "mesh_vertices_are_global": cable_capsule_mesh_vertices_are_global,
                "mesh_direct_fit_enabled": cable_capsule_mesh_direct_fit_enabled,
                "mesh_cluster_voxel_size_m": cable_capsule_mesh_cluster_voxel_size_m,
                "mesh_cluster_neighbor_voxels": cable_capsule_mesh_cluster_neighbor_voxels,
                "mesh_cluster_min_points": cable_capsule_mesh_cluster_min_points,
                "mesh_cluster_min_length_m": cable_capsule_mesh_cluster_min_length_m,
                "mesh_cluster_max_radius_m": cable_capsule_mesh_cluster_max_radius_m,
                "mesh_cluster_endpoint_percentile": cable_capsule_mesh_cluster_endpoint_percentile,
                "component_fit_enabled": True,
                "component_min_points": cable_capsule_component_min_points,
                "component_min_length_m": cable_capsule_component_min_length_m,
                "component_max_radius_m": cable_capsule_component_max_radius_m,
                "component_neighbor_voxels": cable_capsule_component_neighbor_voxels,
                "component_merge_enabled": cable_capsule_component_merge_enabled,
                "component_merge_gap_m": cable_capsule_component_merge_gap_m,
                "component_merge_perp_m": cable_capsule_component_merge_perp_m,
                "component_merge_direction_dot": cable_capsule_component_merge_direction_dot,
                "capsule_ema_alpha": cable_capsule_ema_alpha,
                "capsule_hold_cycles": cable_capsule_hold_cycles,
                "ransac_fallback_enabled": cable_capsule_ransac_fallback_enabled,
            }
        ],
        condition=IfCondition(run_cable_capsules),
    )

    robot_esdf_clearer_node = Node(
        package="openarmx_obstacle_avoidance",
        executable="bimanual_robot_esdf_clearer",
        name="bimanual_robot_esdf_clearer",
        output="screen",
        additional_env={"PYTHONNOUSERSITE": "1"},
        parameters=[
            {
                "robot_description_node": robot_description_node,
                "joint_states_topic": joint_states_topic,
                "global_frame": world_frame,
                "rate_hz": robot_clearer_rate_hz,
                "request_update_esdf": False,
                "aabb_padding": robot_clearer_aabb_padding,
                "clear_robot_padding": robot_clearer_padding,
                "clear_robot_radius_scale": robot_clearer_radius_scale,
                "min_joint_delta_to_clear": robot_clearer_min_joint_delta,
                "force_clear_period_s": robot_clearer_force_period_s,
                "collision_model": "capsule",
                "capsule_samples_per_link": robot_clearer_capsule_samples_per_link,
                "publish_debug_markers": robot_clearer_debug_markers,
            }
        ],
        condition=IfCondition(run_robot_esdf_clearer),
    )

    visual_cues_node = Node(
        package="openarmx_visual_cues",
        executable="bimanual_visual_cues",
        name="bimanual_visual_cues",
        output="screen",
        respawn=True,
        respawn_delay=1.0,
        additional_env={"PYTHONNOUSERSITE": "1"},
        parameters=[
            {
                "robot_description_node": robot_description_node,
                "global_frame": world_frame,
                "joint_states_topic": joint_states_topic,
                "left_desired_pose_topic": visual_cues_left_desired_pose_topic,
                "right_desired_pose_topic": visual_cues_right_desired_pose_topic,
                "left_assisted_pose_topic": visual_cues_left_assisted_pose_topic,
                "right_assisted_pose_topic": visual_cues_right_assisted_pose_topic,
                "active_arm": visual_cues_active_arm,
                "marker_topic": visual_cues_marker_topic,
                "rate_hz": visual_cues_rate_hz,
                "safety_margin": visual_cues_safety_margin,
                "activation_margin": visual_cues_activation_margin,
                "ready_distance": visual_cues_ready_distance,
                "assisted_grasp_enabled": visual_cues_assisted_grasp_enabled,
                "assist_activation_distance": visual_cues_assist_activation_distance,
                "assist_min_alpha": visual_cues_assist_min_alpha,
                "assist_max_alpha": visual_cues_assist_max_alpha,
                "assist_ramp_duration": visual_cues_assist_ramp_duration,
                "ray_selection_enabled": True,
                "show_rviz_selection_ray": visual_cues_show_rviz_selection_ray,
                "pointcloud_topic": visual_cues_pointcloud_topic,
                "left_gripper_topic": visual_cues_left_gripper_topic,
                "right_gripper_topic": visual_cues_right_gripper_topic,
                "ray_length": visual_cues_ray_length,
                "ray_hit_radius": visual_cues_ray_hit_radius,
                "show_aim_reticle": visual_cues_show_aim_reticle,
                "aim_reticle_distance": visual_cues_aim_reticle_distance,
                "use_aim_reticle_as_fallback_target": visual_cues_use_aim_reticle_as_fallback_target,
                "prefer_aim_reticle_target": visual_cues_prefer_aim_reticle_target,
                "aim_reticle_pick_radius_px": visual_cues_aim_reticle_pick_radius_px,
                "aim_reticle_pick_min_points": visual_cues_aim_reticle_pick_min_points,
                "color_image_topic": visual_cues_color_image_topic,
                "color_camera_info_topic": visual_cues_color_camera_info_topic,
                "annotated_image_topic": visual_cues_annotated_image_topic,
                "use_compressed_image": visual_cues_use_compressed_image,
                "compressed_color_image_topic": visual_cues_compressed_color_image_topic,
                "compressed_annotated_image_topic": "/visual_cues/annotated_image/compressed",
                "gripper_open_threshold": visual_cues_gripper_open_threshold,
                "gripper_closed_threshold": visual_cues_gripper_closed_threshold,
                "lock_close_count": visual_cues_lock_close_count,
                "target_tracking_search_radius": visual_cues_target_tracking_search_radius,
                "target_tracking_min_points": visual_cues_target_tracking_min_points,
                "target_roi_pointcloud_topic": visual_cues_target_roi_pointcloud_topic,
                "target_roi_active_topic": visual_cues_target_roi_active_topic,
                "target_bbox_marker_topic": visual_cues_target_bbox_marker_topic,
                "target_roi_max_points": visual_cues_target_roi_max_points,
                "target_bbox_padding": visual_cues_target_bbox_padding,
                "target_bbox_min_size": visual_cues_target_bbox_min_size,
                "target_lock_offset_x": visual_cues_target_lock_offset_x,
                "target_lock_offset_y": visual_cues_target_lock_offset_y,
                "target_lock_offset_z": visual_cues_target_lock_offset_z,
                "show_target_roi": visual_cues_show_target_roi,
                "cable_capsules_topic": visual_cues_cable_capsules_topic,
                "exclude_cable_capsule_points": visual_cues_exclude_cable_capsule_points,
                "cable_capsule_exclusion_padding": visual_cues_cable_capsule_exclusion_padding,
                "show_robot_self_filter": visual_cues_show_robot_self_filter,
            }
        ],
        condition=IfCondition(run_visual_cues),
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="openarm_nvblox_rviz",
        output="screen",
        arguments=["-d", rviz_config_path],
        condition=IfCondition(run_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "depth_topic",
            default_value="/foundation_stereo/depth",
            description="FFS depth image in the infra1 optical frame.",
        ),
        DeclareLaunchArgument(
            "depth_camera_info_topic",
            default_value="/camera/infra1/camera_info",
            description="CameraInfo matching the FFS depth image.",
        ),
        DeclareLaunchArgument(
            "run_depth_freeze_gate",
            default_value="true",
            description="Forward depth to nvblox only for the first depth_freeze_after_s seconds.",
        ),
        DeclareLaunchArgument("frozen_depth_topic", default_value="/nvblox/frozen_depth"),
        DeclareLaunchArgument(
            "frozen_depth_camera_info_topic",
            default_value="/nvblox/frozen_depth/camera_info",
        ),
        DeclareLaunchArgument("depth_freeze_after_s", default_value="5.0"),
        DeclareLaunchArgument(
            "depth_freeze_keep_last_frame",
            default_value="true",
            description="False stops depth publication after freeze. True republishes the last frame.",
        ),
        DeclareLaunchArgument(
            "run_semantic_obstacle_filter",
            default_value="false",
            description="Mask robot body and optional target mask out of the depth stream before nvblox.",
        ),
        DeclareLaunchArgument(
            "semantic_obstacle_depth_topic",
            default_value="/perception/obstacle_depth",
        ),
        DeclareLaunchArgument(
            "semantic_obstacle_camera_info_topic",
            default_value="/perception/obstacle_depth/camera_info",
        ),
        DeclareLaunchArgument(
            "semantic_robot_mask_topic",
            default_value="/perception/robot_body_mask",
        ),
        DeclareLaunchArgument(
            "semantic_combined_mask_topic",
            default_value="/perception/semantic_obstacle_removed_mask",
        ),
        DeclareLaunchArgument(
            "semantic_target_mask_topic",
            default_value="",
            description="Optional mono8 target mask topic to remove from the obstacle map.",
        ),
        DeclareLaunchArgument("semantic_enable_target_mask", default_value="false"),
        DeclareLaunchArgument("semantic_robot_mask_padding_px", default_value="8"),
        DeclareLaunchArgument("semantic_robot_radius_scale", default_value="1.15"),
        DeclareLaunchArgument("world_frame", default_value="world"),
        DeclareLaunchArgument("robot_description_node", default_value="/robot_state_publisher"),
        DeclareLaunchArgument("joint_states_topic", default_value="/joint_states"),
        DeclareLaunchArgument(
            "camera_frame",
            default_value="camera_link",
            description="Static transform child frame. Use camera_link so RealSense owns the internal optical-frame TFs.",
        ),
        DeclareLaunchArgument(
            "publish_camera_tf",
            default_value="true",
            description="Publish a static world->camera_link transform. Disable if calibrated TF is already published.",
        ),
        DeclareLaunchArgument(
            "publish_square_test_tf",
            default_value="true",
            description="Publish a static world->square_test_world transform for visualizing square_pose_input PoseStamped topics.",
        ),
        DeclareLaunchArgument(
            "square_test_frame",
            default_value="square_test_world",
            description="Frame used by square_pose_input_node for /pico_*_controller/pose in square-test mode.",
        ),
        DeclareLaunchArgument("camera_x", default_value="0.05"),
        DeclareLaunchArgument("camera_y", default_value="0.0"),
        DeclareLaunchArgument(
            "camera_z",
            default_value="0.81",
            description="Measured camera_link height above the OpenArmX world/body origin.",
        ),
        DeclareLaunchArgument("camera_qx", default_value="0.0"),
        DeclareLaunchArgument("camera_qy", default_value="0.0"),
        DeclareLaunchArgument("camera_qz", default_value="0.0"),
        DeclareLaunchArgument("camera_qw", default_value="1.0"),
        DeclareLaunchArgument(
            "run_rviz",
            default_value="true",
            description=(
                "openarmx.bimanual.launch.py already starts RViz. Set true only "
                "when running nvblox visualization standalone."
            ),
        ),
        DeclareLaunchArgument(
            "run_visual_cues",
            default_value="false",
            description="Show low-cost teleoperation target, tracking, and ESDF cues in RViz.",
        ),
        DeclareLaunchArgument("visual_cues_active_arm", default_value="right"),
        DeclareLaunchArgument(
            "visual_cues_left_desired_pose_topic",
            default_value="/openarm_mini/left_target_pose",
        ),
        DeclareLaunchArgument(
            "visual_cues_right_desired_pose_topic",
            default_value="/openarm_mini/right_target_pose",
        ),
        DeclareLaunchArgument(
            "visual_cues_left_assisted_pose_topic",
            default_value="/visual_cues/left_assisted_target_pose",
        ),
        DeclareLaunchArgument(
            "visual_cues_right_assisted_pose_topic",
            default_value="/visual_cues/right_assisted_target_pose",
        ),
        DeclareLaunchArgument(
            "visual_cues_marker_topic",
            default_value="/openarmx/visual_cues",
        ),
        DeclareLaunchArgument("visual_cues_rate_hz", default_value="15.0"),
        DeclareLaunchArgument("visual_cues_safety_margin", default_value="0.04"),
        DeclareLaunchArgument("visual_cues_activation_margin", default_value="0.10"),
        DeclareLaunchArgument("visual_cues_ready_distance", default_value="0.01"),
        DeclareLaunchArgument("visual_cues_assisted_grasp_enabled", default_value="true"),
        DeclareLaunchArgument("visual_cues_assist_activation_distance", default_value="0.08"),
        DeclareLaunchArgument("visual_cues_assist_min_alpha", default_value="0.0"),
        DeclareLaunchArgument("visual_cues_assist_max_alpha", default_value="1.00"),
        DeclareLaunchArgument("visual_cues_assist_ramp_duration", default_value="0.40"),
        DeclareLaunchArgument(
            "visual_cues_pointcloud_topic",
            default_value="/foundation_stereo/points",
        ),
        DeclareLaunchArgument(
            "visual_cues_left_gripper_topic",
            default_value="/openarm_mini/left_gripper",
        ),
        DeclareLaunchArgument(
            "visual_cues_right_gripper_topic",
            default_value="/openarm_mini/right_gripper",
        ),
        DeclareLaunchArgument("visual_cues_ray_length", default_value="1.5"),
        DeclareLaunchArgument("visual_cues_ray_hit_radius", default_value="0.045"),
        DeclareLaunchArgument("visual_cues_show_aim_reticle", default_value="true"),
        DeclareLaunchArgument(
            "visual_cues_show_rviz_selection_ray",
            default_value="false",
        ),
        DeclareLaunchArgument("visual_cues_aim_reticle_distance", default_value="0.70"),
        DeclareLaunchArgument(
            "visual_cues_use_aim_reticle_as_fallback_target",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "visual_cues_prefer_aim_reticle_target",
            default_value="true",
        ),
        DeclareLaunchArgument("visual_cues_aim_reticle_pick_radius_px", default_value="24"),
        DeclareLaunchArgument("visual_cues_aim_reticle_pick_min_points", default_value="1"),
        DeclareLaunchArgument(
            "visual_cues_color_image_topic",
            default_value="/camera/color/image_raw",
        ),
        DeclareLaunchArgument("visual_cues_use_compressed_image", default_value="true"),
        DeclareLaunchArgument(
            "visual_cues_compressed_color_image_topic",
            default_value="/camera/color/image_raw/compressed",
        ),
        DeclareLaunchArgument(
            "visual_cues_color_camera_info_topic",
            default_value="/camera/color/camera_info",
        ),
        DeclareLaunchArgument(
            "visual_cues_annotated_image_topic",
            default_value="/visual_cues/annotated_image",
        ),
        DeclareLaunchArgument("visual_cues_gripper_open_threshold", default_value="0.01408"),
        DeclareLaunchArgument("visual_cues_gripper_closed_threshold", default_value="0.00528"),
        DeclareLaunchArgument("visual_cues_lock_close_count", default_value="1"),
        DeclareLaunchArgument("visual_cues_target_tracking_search_radius", default_value="0.25"),
        DeclareLaunchArgument("visual_cues_target_tracking_min_points", default_value="5"),
        DeclareLaunchArgument(
            "visual_cues_target_roi_pointcloud_topic",
            default_value="/visual_cues/target_roi_points",
        ),
        DeclareLaunchArgument(
            "visual_cues_target_roi_active_topic",
            default_value="/visual_cues/target_roi_active",
        ),
        DeclareLaunchArgument(
            "visual_cues_target_bbox_marker_topic",
            default_value="/visual_cues/target_bbox",
        ),
        DeclareLaunchArgument("visual_cues_target_roi_max_points", default_value="3000"),
        DeclareLaunchArgument("visual_cues_target_bbox_padding", default_value="0.015"),
        DeclareLaunchArgument("visual_cues_target_bbox_min_size", default_value="0.03"),
        DeclareLaunchArgument("visual_cues_target_lock_offset_x", default_value="0.03"),
        DeclareLaunchArgument("visual_cues_target_lock_offset_y", default_value="0.0"),
        DeclareLaunchArgument("visual_cues_target_lock_offset_z", default_value="0.0"),
        DeclareLaunchArgument("visual_cues_show_target_roi", default_value="true"),
        DeclareLaunchArgument(
            "visual_cues_cable_capsules_topic",
            default_value="/perception/cable_capsules",
        ),
        DeclareLaunchArgument("visual_cues_exclude_cable_capsule_points", default_value="true"),
        DeclareLaunchArgument(
            "visual_cues_cable_capsule_exclusion_padding",
            default_value="0.025",
        ),
        DeclareLaunchArgument(
            "visual_cues_show_robot_self_filter",
            default_value="false",
        ),
        DeclareLaunchArgument("log_level", default_value="info"),
        DeclareLaunchArgument(
            "run_cable_capsules",
            default_value="true",
            description="Publish RViz MarkerArray capsules fitted to the depth_topic cable points.",
        ),
        DeclareLaunchArgument(
            "run_robot_esdf_clearer",
            default_value="false",
            description="Clear OpenArmX body spheres from nvblox through the ESDF service. Disabled by default because it can erase thin cable maps.",
        ),
        DeclareLaunchArgument("robot_clearer_rate_hz", default_value="2.0"),
        DeclareLaunchArgument("robot_clearer_aabb_padding", default_value="0.08"),
        DeclareLaunchArgument("robot_clearer_padding", default_value="0.015"),
        DeclareLaunchArgument("robot_clearer_radius_scale", default_value="1.0"),
        DeclareLaunchArgument("robot_clearer_min_joint_delta", default_value="0.08"),
        DeclareLaunchArgument("robot_clearer_force_period_s", default_value="4.0"),
        DeclareLaunchArgument("robot_clearer_capsule_samples_per_link", default_value="3"),
        DeclareLaunchArgument("robot_clearer_debug_markers", default_value="true"),
        DeclareLaunchArgument(
            "cable_capsule_topic",
            default_value="/perception/cable_capsules",
            description="MarkerArray topic for cable capsule visualization.",
        ),
        DeclareLaunchArgument(
            "cable_capsule_source_mode",
            default_value="voxel_layer",
            description="Capsule fitting source: voxel_layer, mesh, pointcloud, or depth.",
        ),
        DeclareLaunchArgument("cable_capsule_pointcloud_topic", default_value="/nvblox_node/static_esdf_pointcloud"),
        DeclareLaunchArgument("cable_capsule_mesh_topic", default_value="/nvblox_node/mesh"),
        DeclareLaunchArgument("cable_capsule_voxel_layer_topic", default_value="/nvblox_node/color_layer"),
        DeclareLaunchArgument("cable_capsule_voxel_centers_are_global", default_value="true"),
        DeclareLaunchArgument(
            "cable_capsule_radius_m",
            default_value="0.02",
            description="Visual capsule radius around the detected cable.",
        ),
        DeclareLaunchArgument("cable_capsule_alpha", default_value="0.45"),
        DeclareLaunchArgument("cable_capsule_max_capsules", default_value="8"),
        DeclareLaunchArgument("cable_capsule_min_line_length_px", default_value="70"),
        DeclareLaunchArgument("cable_capsule_max_line_gap_px", default_value="45"),
        DeclareLaunchArgument("cable_capsule_ransac_inlier_distance_m", default_value="0.035"),
        DeclareLaunchArgument("cable_capsule_ransac_min_inliers", default_value="25"),
        DeclareLaunchArgument("cable_capsule_point_voxel_size_m", default_value="0.015"),
        DeclareLaunchArgument(
            "cable_capsule_voxel_fit_mode",
            default_value="seeded_axis",
            description="Voxel fitting mode: seeded_axis or components.",
        ),
        DeclareLaunchArgument(
            "cable_capsule_left_seed_reference_frame",
            default_value="openarmx_left_link1",
        ),
        DeclareLaunchArgument(
            "cable_capsule_right_seed_reference_frame",
            default_value="openarmx_right_link1",
        ),
        DeclareLaunchArgument(
            "cable_capsule_seed_axis_neighbor_radius_m",
            default_value="0.045",
            description="Maximum perpendicular voxel distance from the world-X seeded axis.",
        ),
        DeclareLaunchArgument(
            "cable_capsule_seed_axis_max_gap_m",
            default_value="0.15",
            description="Largest gap allowed while growing along the seeded axis.",
        ),
        DeclareLaunchArgument(
            "cable_capsule_seed_axis_max_half_length_m",
            default_value="0.1",
            description="Local voxel collection distance in each direction from the seed.",
        ),
        DeclareLaunchArgument(
            "cable_capsule_seed_axis_output_half_length_m",
            default_value="0.6",
            description="Final capsule half-length after extending the locally fitted segment.",
        ),
        DeclareLaunchArgument(
            "cable_capsule_seed_min_neighbor_voxels",
            default_value="3",
            description="Minimum neighboring voxels required around a capsule seed, excluding itself.",
        ),
        DeclareLaunchArgument(
            "cable_capsule_seed_neighbor_radius_m",
            default_value="0.08",
            description="Radius used to reject isolated capsule seed voxels.",
        ),
        DeclareLaunchArgument("cable_capsule_mesh_vertices_are_global", default_value="true"),
        DeclareLaunchArgument("cable_capsule_mesh_direct_fit_enabled", default_value="true"),
        DeclareLaunchArgument("cable_capsule_mesh_cluster_voxel_size_m", default_value="0.03"),
        DeclareLaunchArgument("cable_capsule_mesh_cluster_neighbor_voxels", default_value="2"),
        DeclareLaunchArgument("cable_capsule_mesh_cluster_min_points", default_value="10"),
        DeclareLaunchArgument("cable_capsule_mesh_cluster_min_length_m", default_value="0.12"),
        DeclareLaunchArgument("cable_capsule_mesh_cluster_max_radius_m", default_value="0.10"),
        DeclareLaunchArgument("cable_capsule_mesh_cluster_endpoint_percentile", default_value="0.0"),
        DeclareLaunchArgument("cable_capsule_component_min_points", default_value="6"),
        DeclareLaunchArgument("cable_capsule_component_min_length_m", default_value="0.03"),
        DeclareLaunchArgument("cable_capsule_component_max_radius_m", default_value="0.055"),
        DeclareLaunchArgument("cable_capsule_component_neighbor_voxels", default_value="1"),
        DeclareLaunchArgument("cable_capsule_component_merge_enabled", default_value="true"),
        DeclareLaunchArgument("cable_capsule_component_merge_gap_m", default_value="0.18"),
        DeclareLaunchArgument("cable_capsule_component_merge_perp_m", default_value="0.045"),
        DeclareLaunchArgument("cable_capsule_component_merge_direction_dot", default_value="0.90"),
        DeclareLaunchArgument("cable_capsule_ema_alpha", default_value="0.18"),
        DeclareLaunchArgument("cable_capsule_hold_cycles", default_value="8"),
        DeclareLaunchArgument(
            "cable_capsule_ransac_fallback_enabled",
            default_value="false",
            description="Enable global RANSAC fitting after connected components. False avoids bridging separated cable segments.",
        ),
        camera_tf,
        square_test_tf,
        semantic_obstacle_filter_node,
        depth_freeze_gate_node_raw,
        depth_freeze_gate_node_semantic,
        nvblox_node_direct,
        nvblox_node_frozen,
        nvblox_node_semantic,
        nvblox_node_semantic_frozen,
        cable_capsule_marker_node,
        robot_esdf_clearer_node,
        visual_cues_node,
        rviz_node,
    ])
