from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("urdf_path", default_value=""),
        DeclareLaunchArgument("robot_description_node", default_value="/robot_state_publisher"),
        DeclareLaunchArgument("global_frame", default_value="world"),
        DeclareLaunchArgument("joint_states_topic", default_value="/joint_states"),
        DeclareLaunchArgument(
            "left_desired_pose_topic",
            default_value="/openarm_mini/left_target_pose",
        ),
        DeclareLaunchArgument(
            "right_desired_pose_topic",
            default_value="/openarm_mini/right_target_pose",
        ),
        DeclareLaunchArgument(
            "left_assisted_pose_topic",
            default_value="/visual_cues/left_assisted_target_pose",
        ),
        DeclareLaunchArgument(
            "right_assisted_pose_topic",
            default_value="/visual_cues/right_assisted_target_pose",
        ),
        DeclareLaunchArgument("clicked_point_topic", default_value="/clicked_point"),
        DeclareLaunchArgument("active_arm_topic", default_value="/visual_cues/active_arm"),
        DeclareLaunchArgument("active_arm", default_value="right"),
        DeclareLaunchArgument("marker_topic", default_value="/openarmx/visual_cues"),
        DeclareLaunchArgument(
            "left_clearance_topic",
            default_value="/openarmx/bimanual/left_min_esdf_clearance",
        ),
        DeclareLaunchArgument(
            "right_clearance_topic",
            default_value="/openarmx/bimanual/right_min_esdf_clearance",
        ),
        DeclareLaunchArgument(
            "avoidance_status_topic",
            default_value="/openarmx/bimanual/esdf_avoidance_status",
        ),
        DeclareLaunchArgument("rate_hz", default_value="15.0"),
        DeclareLaunchArgument("safety_margin", default_value="0.04"),
        DeclareLaunchArgument("activation_margin", default_value="0.10"),
        DeclareLaunchArgument("ready_distance", default_value="0.01"),
        DeclareLaunchArgument("assisted_grasp_enabled", default_value="true"),
        DeclareLaunchArgument("assist_activation_distance", default_value="0.08"),
        DeclareLaunchArgument("assist_min_alpha", default_value="0.0"),
        DeclareLaunchArgument("assist_max_alpha", default_value="1.00"),
        DeclareLaunchArgument("assist_ramp_duration", default_value="0.40"),
        DeclareLaunchArgument("show_actual_axes", default_value="false"),
        DeclareLaunchArgument("show_desired_axes", default_value="false"),
        DeclareLaunchArgument("show_tracking_line", default_value="false"),
        DeclareLaunchArgument("show_tracking_text", default_value="false"),
        DeclareLaunchArgument("show_clearance_text", default_value="false"),
        DeclareLaunchArgument("show_target_text", default_value="true"),
        DeclareLaunchArgument("tracking_line_min_error", default_value="0.015"),
        DeclareLaunchArgument("ray_selection_enabled", default_value="true"),
        DeclareLaunchArgument("show_rviz_selection_ray", default_value="false"),
        DeclareLaunchArgument("pointcloud_topic", default_value="/foundation_stereo/points"),
        DeclareLaunchArgument("color_image_topic", default_value="/camera/color/image_raw"),
        DeclareLaunchArgument(
            "compressed_color_image_topic",
            default_value="/camera/color/image_raw/compressed",
        ),
        DeclareLaunchArgument(
            "color_camera_info_topic",
            default_value="/camera/color/camera_info",
        ),
        DeclareLaunchArgument(
            "annotated_image_topic",
            default_value="/visual_cues/annotated_image",
        ),
        DeclareLaunchArgument(
            "compressed_annotated_image_topic",
            default_value="/visual_cues/annotated_image/compressed",
        ),
        DeclareLaunchArgument("use_compressed_image", default_value="true"),
        DeclareLaunchArgument("jpeg_quality", default_value="85"),
        DeclareLaunchArgument("ray_length", default_value="1.5"),
        DeclareLaunchArgument("ray_hit_radius", default_value="0.045"),
        DeclareLaunchArgument("pointcloud_timeout_s", default_value="1.0"),
        DeclareLaunchArgument("candidate_hold_s", default_value="0.75"),
        DeclareLaunchArgument("show_aim_reticle", default_value="true"),
        DeclareLaunchArgument("aim_reticle_distance", default_value="0.70"),
        DeclareLaunchArgument("use_aim_reticle_as_fallback_target", default_value="true"),
        DeclareLaunchArgument("prefer_aim_reticle_target", default_value="true"),
        DeclareLaunchArgument("aim_reticle_pick_radius_px", default_value="24"),
        DeclareLaunchArgument("aim_reticle_pick_min_points", default_value="1"),
        DeclareLaunchArgument("left_gripper_topic", default_value="/openarm_mini/left_gripper"),
        DeclareLaunchArgument("right_gripper_topic", default_value="/openarm_mini/right_gripper"),
        DeclareLaunchArgument("gripper_open_threshold", default_value="0.01408"),
        DeclareLaunchArgument("gripper_closed_threshold", default_value="0.00528"),
        DeclareLaunchArgument("lock_close_count", default_value="1"),
        DeclareLaunchArgument("gesture_timeout_s", default_value="3.0"),
        DeclareLaunchArgument("target_tracking_search_radius", default_value="0.25"),
        DeclareLaunchArgument("target_tracking_min_points", default_value="5"),
        DeclareLaunchArgument("target_roi_pointcloud_topic", default_value="/visual_cues/target_roi_points"),
        DeclareLaunchArgument("target_roi_active_topic", default_value="/visual_cues/target_roi_active"),
        DeclareLaunchArgument("target_bbox_marker_topic", default_value="/visual_cues/target_bbox"),
        DeclareLaunchArgument("target_roi_max_points", default_value="3000"),
        DeclareLaunchArgument("target_bbox_padding", default_value="0.015"),
        DeclareLaunchArgument("target_bbox_min_size", default_value="0.03"),
        DeclareLaunchArgument("target_lock_offset_x", default_value="0.03"),
        DeclareLaunchArgument("target_lock_offset_y", default_value="0.0"),
        DeclareLaunchArgument("target_lock_offset_z", default_value="0.0"),
        DeclareLaunchArgument("show_target_roi", default_value="true"),
        DeclareLaunchArgument("cable_capsules_topic", default_value="/perception/cable_capsules"),
        DeclareLaunchArgument("exclude_cable_capsule_points", default_value="true"),
        DeclareLaunchArgument("cable_capsule_exclusion_padding", default_value="0.025"),
        DeclareLaunchArgument("exclude_robot_points", default_value="true"),
        DeclareLaunchArgument("robot_self_filter_link_radius", default_value="0.075"),
        DeclareLaunchArgument("robot_self_filter_hand_radius", default_value="0.055"),
        DeclareLaunchArgument("robot_self_filter_padding", default_value="0.010"),
        DeclareLaunchArgument("show_robot_self_filter", default_value="false"),
        DeclareLaunchArgument("untangle_mode_topic", default_value="/openarmx/untangle_mode"),
    ]

    node = Node(
        package="openarmx_visual_cues",
        executable="bimanual_visual_cues",
        name="bimanual_visual_cues",
        output="screen",
        respawn=True,
        respawn_delay=1.0,
        additional_env={"PYTHONNOUSERSITE": "1"},
        parameters=[
            {
                "urdf_path": LaunchConfiguration("urdf_path"),
                "robot_description_node": LaunchConfiguration("robot_description_node"),
                "global_frame": LaunchConfiguration("global_frame"),
                "joint_states_topic": LaunchConfiguration("joint_states_topic"),
                "left_desired_pose_topic": LaunchConfiguration("left_desired_pose_topic"),
                "right_desired_pose_topic": LaunchConfiguration("right_desired_pose_topic"),
                "left_assisted_pose_topic": LaunchConfiguration("left_assisted_pose_topic"),
                "right_assisted_pose_topic": LaunchConfiguration("right_assisted_pose_topic"),
                "clicked_point_topic": LaunchConfiguration("clicked_point_topic"),
                "active_arm_topic": LaunchConfiguration("active_arm_topic"),
                "active_arm": LaunchConfiguration("active_arm"),
                "marker_topic": LaunchConfiguration("marker_topic"),
                "left_clearance_topic": LaunchConfiguration("left_clearance_topic"),
                "right_clearance_topic": LaunchConfiguration("right_clearance_topic"),
                "avoidance_status_topic": LaunchConfiguration("avoidance_status_topic"),
                "rate_hz": LaunchConfiguration("rate_hz"),
                "safety_margin": LaunchConfiguration("safety_margin"),
                "activation_margin": LaunchConfiguration("activation_margin"),
                "ready_distance": LaunchConfiguration("ready_distance"),
                "assisted_grasp_enabled": LaunchConfiguration("assisted_grasp_enabled"),
                "assist_activation_distance": LaunchConfiguration("assist_activation_distance"),
                "assist_min_alpha": LaunchConfiguration("assist_min_alpha"),
                "assist_max_alpha": LaunchConfiguration("assist_max_alpha"),
                "assist_ramp_duration": LaunchConfiguration("assist_ramp_duration"),
                "show_actual_axes": LaunchConfiguration("show_actual_axes"),
                "show_desired_axes": LaunchConfiguration("show_desired_axes"),
                "show_tracking_line": LaunchConfiguration("show_tracking_line"),
                "show_tracking_text": LaunchConfiguration("show_tracking_text"),
                "show_clearance_text": LaunchConfiguration("show_clearance_text"),
                "show_target_text": LaunchConfiguration("show_target_text"),
                "tracking_line_min_error": LaunchConfiguration("tracking_line_min_error"),
                "ray_selection_enabled": LaunchConfiguration("ray_selection_enabled"),
                "show_rviz_selection_ray": LaunchConfiguration(
                    "show_rviz_selection_ray"
                ),
                "pointcloud_topic": LaunchConfiguration("pointcloud_topic"),
                "color_image_topic": LaunchConfiguration("color_image_topic"),
                "compressed_color_image_topic": LaunchConfiguration(
                    "compressed_color_image_topic"
                ),
                "color_camera_info_topic": LaunchConfiguration("color_camera_info_topic"),
                "annotated_image_topic": LaunchConfiguration("annotated_image_topic"),
                "compressed_annotated_image_topic": LaunchConfiguration(
                    "compressed_annotated_image_topic"
                ),
                "use_compressed_image": LaunchConfiguration("use_compressed_image"),
                "jpeg_quality": LaunchConfiguration("jpeg_quality"),
                "ray_length": LaunchConfiguration("ray_length"),
                "ray_hit_radius": LaunchConfiguration("ray_hit_radius"),
                "pointcloud_timeout_s": LaunchConfiguration("pointcloud_timeout_s"),
                "candidate_hold_s": LaunchConfiguration("candidate_hold_s"),
                "show_aim_reticle": LaunchConfiguration("show_aim_reticle"),
                "aim_reticle_distance": LaunchConfiguration("aim_reticle_distance"),
                "use_aim_reticle_as_fallback_target": LaunchConfiguration(
                    "use_aim_reticle_as_fallback_target"
                ),
                "prefer_aim_reticle_target": LaunchConfiguration(
                    "prefer_aim_reticle_target"
                ),
                "aim_reticle_pick_radius_px": LaunchConfiguration("aim_reticle_pick_radius_px"),
                "aim_reticle_pick_min_points": LaunchConfiguration("aim_reticle_pick_min_points"),
                "left_gripper_topic": LaunchConfiguration("left_gripper_topic"),
                "right_gripper_topic": LaunchConfiguration("right_gripper_topic"),
                "gripper_open_threshold": LaunchConfiguration("gripper_open_threshold"),
                "gripper_closed_threshold": LaunchConfiguration("gripper_closed_threshold"),
                "lock_close_count": LaunchConfiguration("lock_close_count"),
                "gesture_timeout_s": LaunchConfiguration("gesture_timeout_s"),
                "target_tracking_search_radius": LaunchConfiguration(
                    "target_tracking_search_radius"
                ),
                "target_tracking_min_points": LaunchConfiguration("target_tracking_min_points"),
                "target_roi_pointcloud_topic": LaunchConfiguration("target_roi_pointcloud_topic"),
                "target_roi_active_topic": LaunchConfiguration("target_roi_active_topic"),
                "target_bbox_marker_topic": LaunchConfiguration("target_bbox_marker_topic"),
                "target_roi_max_points": LaunchConfiguration("target_roi_max_points"),
                "target_bbox_padding": LaunchConfiguration("target_bbox_padding"),
                "target_bbox_min_size": LaunchConfiguration("target_bbox_min_size"),
                "target_lock_offset_x": LaunchConfiguration("target_lock_offset_x"),
                "target_lock_offset_y": LaunchConfiguration("target_lock_offset_y"),
                "target_lock_offset_z": LaunchConfiguration("target_lock_offset_z"),
                "show_target_roi": LaunchConfiguration("show_target_roi"),
                "cable_capsules_topic": LaunchConfiguration("cable_capsules_topic"),
                "exclude_cable_capsule_points": LaunchConfiguration(
                    "exclude_cable_capsule_points"
                ),
                "cable_capsule_exclusion_padding": LaunchConfiguration(
                    "cable_capsule_exclusion_padding"
                ),
                "exclude_robot_points": LaunchConfiguration("exclude_robot_points"),
                "robot_self_filter_link_radius": LaunchConfiguration(
                    "robot_self_filter_link_radius"
                ),
                "robot_self_filter_hand_radius": LaunchConfiguration(
                    "robot_self_filter_hand_radius"
                ),
                "robot_self_filter_padding": LaunchConfiguration(
                    "robot_self_filter_padding"
                ),
                "show_robot_self_filter": LaunchConfiguration(
                    "show_robot_self_filter"
                ),
                "untangle_mode_topic": LaunchConfiguration("untangle_mode_topic"),
            }
        ],
    )

    return LaunchDescription(arguments + [node])
