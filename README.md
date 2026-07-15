# OpenArmX Mapping and Obstacle Avoidance

ROS 2 Jazzy packages used by the OpenArmX teleoperation system for:

- Fast-FoundationStereo depth integration with NVIDIA nvblox
- Frozen-map and semantic/self-filter experiments
- Cable capsule fitting and RViz visualization
- Bimanual ESDF obstacle-avoidance filtering
- Target-selection and assisted-grasp visual cues

## Packages

- `openarmx_nvblox_bringup`: nvblox launch files, mapping configuration,
  depth gating/filtering, cable capsule fitting, and RViz configuration.
- `openarmx_obstacle_avoidance`: right-arm and bimanual ESDF avoidance
  filters, predictive experiments, and robot ESDF clearing utilities.
- `openarmx_visual_cues`: image/RViz target-selection cues and assisted-grasp
  target publications.

## Build

Copy or clone this repository into a ROS 2 workspace, then run:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install \
  --packages-select \
  openarmx_nvblox_bringup \
  openarmx_obstacle_avoidance \
  openarmx_visual_cues
source install/setup.bash
```

The system also requires the OpenArmX robot description/TF publishers,
NVIDIA Isaac ROS nvblox, and the message packages declared in each
`package.xml`.

## Mapping

The convenience script starts Fast-FoundationStereo and the nvblox world
launch file:

```bash
WORKSPACE=/path/to/ros2_ws \
FFS_DIR=/path/to/Fast-FoundationStereo \
MODEL_DIR=/path/to/model.pth \
./scripts/start_ffs_nvblox_world.sh
```

Start the RealSense infrared/color streams and the OpenArmX robot state/TF
publishers before running the script. Its paths and inference parameters can
be overridden through environment variables; run it with `--help` for the
full list.

## Obstacle Avoidance

After mapping and robot TF are available:

```bash
ros2 launch openarmx_obstacle_avoidance \
  bimanual_esdf_avoidance_filter.launch.py
```

Review the launch-file topic defaults before connecting output topics to
physical robot controllers.
