#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE="${WORKSPACE:-/home/wanghua/openarmx_robotstride_ws‑cc}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
WS_SETUP="${WS_SETUP:-${WORKSPACE}/install/setup.bash}"
CONDA_SH="${CONDA_SH:-/home/wanghua/miniconda3/etc/profile.d/conda.sh}"
FFS_ENV="${FFS_ENV:-ffs}"
FFS_DIR="${FFS_DIR:-/home/wanghua/Fast-FoundationStereo}"
FFS_SCRIPT="${FFS_SCRIPT:-${FFS_DIR}/scripts/run_ros2_realsense.py}"
MODEL_DIR="${MODEL_DIR:-${FFS_DIR}/weights/20-26-39/model_best_bp2_serialize.pth}"

LEFT_TOPIC="${LEFT_TOPIC:-/camera/infra1/image_rect_raw}"
RIGHT_TOPIC="${RIGHT_TOPIC:-/camera/infra2/image_rect_raw}"
LEFT_INFO_TOPIC="${LEFT_INFO_TOPIC:-/camera/infra1/camera_info}"
RIGHT_INFO_TOPIC="${RIGHT_INFO_TOPIC:-/camera/infra2/camera_info}"
SCALE="${SCALE:-1.0}"
VALID_ITERS="${VALID_ITERS:-4}"
MAX_DISP="${MAX_DISP:-192}"
ZFAR="${ZFAR:-1.4}"
SHOW="${SHOW:-false}"
FFS_DEPTH_TOPIC="${FFS_DEPTH_TOPIC:-/foundation_stereo/depth}"
FFS_START_TIMEOUT="${FFS_START_TIMEOUT:-90}"
MAP_BUILD_SECONDS="${MAP_BUILD_SECONDS:-5.0}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [nvblox launch arguments...]

Starts:
  1. Fast-FoundationStereo ROS 2 node in Conda environment '${FFS_ENV}'
  2. openarm_nvblox_world.launch.py with run_visual_cues:=true

Examples:
  $(basename "$0")
  $(basename "$0") run_rviz:=false
  ZFAR=1.2 VALID_ITERS=6 $(basename "$0")

Environment overrides:
  WORKSPACE ROS_SETUP WS_SETUP CONDA_SH
  FFS_ENV FFS_DIR FFS_SCRIPT MODEL_DIR
  LEFT_TOPIC RIGHT_TOPIC LEFT_INFO_TOPIC RIGHT_INFO_TOPIC
  SCALE VALID_ITERS MAX_DISP ZFAR SHOW
  FFS_DEPTH_TOPIC FFS_START_TIMEOUT MAP_BUILD_SECONDS

Prerequisite:
  Start the D435/D435i infrared and color streams before running this script.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

for required_file in \
  "${ROS_SETUP}" \
  "${WS_SETUP}" \
  "${CONDA_SH}" \
  "${FFS_SCRIPT}" \
  "${MODEL_DIR}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "[ffs-nvblox] required file not found: ${required_file}" >&2
    exit 1
  fi
done

pids=()
cleanup() {
  trap - INT TERM EXIT
  for pid in "${pids[@]:-}"; do
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  done
  sleep 1.0
  for pid in "${pids[@]:-}"; do
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
  done
  wait "${pids[@]:-}" 2>/dev/null || true
}
trap 'cleanup; exit 130' INT TERM
trap cleanup EXIT

echo "[ffs-nvblox] workspace=${WORKSPACE}"
echo "[ffs-nvblox] ffs_dir=${FFS_DIR} conda_env=${FFS_ENV}"
echo "[ffs-nvblox] model=${MODEL_DIR}"
echo "[ffs-nvblox] stereo=${LEFT_TOPIC},${RIGHT_TOPIC}"
echo "[ffs-nvblox] valid_iters=${VALID_ITERS} max_disp=${MAX_DISP} zfar=${ZFAR}"
echo "[ffs-nvblox] mapping_mode=freeze_after_${MAP_BUILD_SECONDS}s semantic_filter=false"

(
  set +u
  source "${CONDA_SH}"
  conda activate "${FFS_ENV}"
  source "${ROS_SETUP}"
  source "${WS_SETUP}"
  set -u
  cd "${FFS_DIR}"
  exec setsid env PYTHONUNBUFFERED=1 python "${FFS_SCRIPT}" \
    --ros-args \
    -p model_dir:="${MODEL_DIR}" \
    -p left_topic:="${LEFT_TOPIC}" \
    -p right_topic:="${RIGHT_TOPIC}" \
    -p left_info_topic:="${LEFT_INFO_TOPIC}" \
    -p right_info_topic:="${RIGHT_INFO_TOPIC}" \
    -p scale:="${SCALE}" \
    -p valid_iters:="${VALID_ITERS}" \
    -p max_disp:="${MAX_DISP}" \
    -p zfar:="${ZFAR}" \
    -p show:="${SHOW}"
) &
pids+=("$!")

echo "[ffs-nvblox] waiting up to ${FFS_START_TIMEOUT}s for ${FFS_DEPTH_TOPIC} ..."
depth_ready=false
for ((second = 0; second < FFS_START_TIMEOUT; second++)); do
  if ! kill -0 "${pids[0]}" 2>/dev/null; then
    echo "[ffs-nvblox] FFS exited before publishing depth." >&2
    wait "${pids[0]}" || true
    exit 1
  fi
  if (
    set +u
    source "${ROS_SETUP}"
    source "${WS_SETUP}"
    set -u
    timeout 1.5 ros2 topic echo --once "${FFS_DEPTH_TOPIC}" --field header \
      >/dev/null 2>&1
  ); then
    depth_ready=true
    break
  fi
  sleep 0.5
done
if [[ "${depth_ready}" != "true" ]]; then
  echo "[ffs-nvblox] timed out waiting for ${FFS_DEPTH_TOPIC}." >&2
  exit 1
fi
echo "[ffs-nvblox] FFS depth is ready; starting nvblox."

(
  set +u
  source "${ROS_SETUP}"
  source "${WS_SETUP}"
  set -u
  cd "${WORKSPACE}"
  exec setsid ros2 launch openarmx_nvblox_bringup openarm_nvblox_world.launch.py \
    run_visual_cues:=true \
    run_depth_freeze_gate:=true \
    depth_freeze_after_s:="${MAP_BUILD_SECONDS}" \
    depth_freeze_keep_last_frame:=false \
    run_semantic_obstacle_filter:=false \
    "$@"
) &
pids+=("$!")

echo "[ffs-nvblox] started. Press Ctrl-C to stop both processes."
echo "[ffs-nvblox] checks:"
echo "  ros2 topic hz /foundation_stereo/depth"
echo "  ros2 topic hz /nvblox_node/color_layer"
echo "  ros2 topic hz /visual_cues/annotated_image/compressed"

set +e
wait -n "${pids[@]}"
exit_code=$?
set -e
echo "[ffs-nvblox] one child exited with code ${exit_code}; stopping the other."
exit "${exit_code}"
