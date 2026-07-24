cmake_minimum_required(VERSION 3.8)
project(openarmx_nvblox_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/cable_capsule_marker_node.py
  scripts/cable_depth_filter_node.py
  scripts/depth_freeze_gate_node.py
  scripts/fixed_cable_depth_mask_node.py
  scripts/fixed_cable_voxel_clearer_node.py
  scripts/protected_cable_perturbation_node.py
  scripts/robot_self_filter_diagnostics_node.py
  scripts/rolling_object_voxel_node.py
  scripts/semantic_obstacle_depth_filter_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
