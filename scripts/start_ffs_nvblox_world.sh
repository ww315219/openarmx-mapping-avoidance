#!/usr/bin/env bash
set -Eeuo pipefail

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-33}"

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
ZFAR="${ZFAR:-1.25}"
SHOW="${SHOW:-false}"
DEPTH_MEDIAN_KERNEL="${DEPTH_MEDIAN_KERNEL:-3}"
DEPTH_MIN_VALID_NEIGHBORS="${DEPTH_MIN_VALID_NEIGHBORS:-2}"
DEPTH_SPECKLE_MAX_SIZE="${DEPTH_SPECKLE_MAX_SIZE:-60}"
DEPTH_EDGE_MAX_DELTA="${DEPTH_EDGE_MAX_DELTA:-0.0}"
FFS_DEPTH_TOPIC="${FFS_DEPTH_TOPIC:-/foundation_stereo/depth}"
FFS_START_TIMEOUT="${FFS_START_TIMEOUT:-15}"
SUPERVISOR_LOG="${SUPERVISOR_LOG:-${HOME}/.ros/log/start_ffs_nvblox_world_supervisor.log}"
THREAD_SNAPSHOT="${THREAD_SNAPSHOT:-${HOME}/.ros/log/start_ffs_nvblox_world_threads.log}"

mkdir -p "$(dirname "${SUPERVISOR_LOG}")"
supervisor_log() {
  local line
  line="[$(date '+%F %T.%3N')] $*"
  printf '%s\n' "${line}" >>"${SUPERVISOR_LOG}" || true
  printf '%s\n' "${line}" || true
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [nvblox launch arguments...]

Starts:
  1. Fast-FoundationStereo ROS 2 node in Conda environment '${FFS_ENV}'
  2. openarm_nvblox_world.launch.py with visual cues and preview-only untangling

Examples:
  $(basename "$0")
  $(basename "$0") run_rviz:=false
  ZFAR=1.2 VALID_ITERS=6 $(basename "$0")

Environment overrides:
  WORKSPACE ROS_SETUP WS_SETUP CONDA_SH
  FFS_ENV FFS_DIR FFS_SCRIPT MODEL_DIR
  LEFT_TOPIC RIGHT_TOPIC LEFT_INFO_TOPIC RIGHT_INFO_TOPIC
  SCALE VALID_ITERS MAX_DISP ZFAR SHOW
  DEPTH_MEDIAN_KERNEL DEPTH_MIN_VALID_NEIGHBORS
  DEPTH_SPECKLE_MAX_SIZE DEPTH_EDGE_MAX_DELTA
  FFS_DEPTH_TOPIC FFS_START_TIMEOUT

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
shutdown_reason="normal exit"
cleanup() {
  trap - INT TERM EXIT
  supervisor_log "cleanup: ${shutdown_reason}; children=${pids[*]:-none}"
  for pid in "${pids[@]:-}"; do
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  done
  sleep 1.0
  for pid in "${pids[@]:-}"; do
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
  done
  wait "${pids[@]:-}" 2>/dev/null || true
}
handle_signal() {
  shutdown_reason="received $1"
  cleanup
  exit 130
}
trap 'handle_signal SIGINT' INT
trap 'handle_signal SIGTERM' TERM
# Keep mapping alive if a VS Code/NoMachine terminal session is temporarily detached.
trap '' HUP
trap cleanup EXIT

echo "[ffs-nvblox] workspace=${WORKSPACE}"
echo "[ffs-nvblox] ffs_dir=${FFS_DIR} conda_env=${FFS_ENV}"
echo "[ffs-nvblox] model=${MODEL_DIR}"
echo "[ffs-nvblox] stereo=${LEFT_TOPIC},${RIGHT_TOPIC}"
echo "[ffs-nvblox] valid_iters=${VALID_ITERS} max_disp=${MAX_DISP} zfar=${ZFAR}"
echo "[ffs-nvblox] depth_filter=median:${DEPTH_MEDIAN_KERNEL} neighbors:${DEPTH_MIN_VALID_NEIGHBORS} speckle:${DEPTH_SPECKLE_MAX_SIZE} edge:${DEPTH_EDGE_MAX_DELTA}"
echo "[ffs-nvblox] mapping_mode=continuous fixed_cable_depth_mask=true semantic_filter=false"

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
    -p depth_median_kernel:="${DEPTH_MEDIAN_KERNEL}" \
    -p depth_min_valid_neighbors:="${DEPTH_MIN_VALID_NEIGHBORS}" \
    -p depth_speckle_max_size:="${DEPTH_SPECKLE_MAX_SIZE}" \
    -p depth_edge_max_delta:="${DEPTH_EDGE_MAX_DELTA}" \
    -p show:="${SHOW}"
) &
pids+=("$!")

(
  set +u
  source "${ROS_SETUP}"
  source "${WS_SETUP}"
  set -u
  cd "${WORKSPACE}"
  exec setsid ros2 launch openarmx_nvblox_bringup openarm_nvblox_world.launch.py \
    run_visual_cues:=true \
    visual_cues_assisted_grasp_enabled:=false \
    visual_cues_target_selection_mode:=ray \
    run_untangle_preview:=true \
    run_depth_freeze_gate:=false \
    run_fixed_cable_depth_mask:=true \
    run_fixed_cable_voxel_clearer:=false \
    run_semantic_obstacle_filter:=false \
    rolling_object_max_depth_m:="${ZFAR}" \
    "$@"
) &
pids+=("$!")
supervisor_log "started wrapper=$$ FFS=${pids[0]} nvblox_launch=${pids[1]} zfar=${ZFAR}"

# Save thread ownership while processes are alive. Kernel segfault reports use
# thread IDs (LWP), which otherwise cannot be mapped after process teardown.
(
  while kill -0 "${pids[0]}" 2>/dev/null || kill -0 "${pids[1]}" 2>/dev/null; do
    {
      printf '\n[%s] wrapper=%s FFS=%s nvblox_launch=%s\n' \
        "$(date '+%F %T.%3N')" "$$" "${pids[0]}" "${pids[1]}"
      ps -eLo pid,ppid,lwp,comm,args --sort=pid
    } >"${THREAD_SNAPSHOT}.tmp"
    mv "${THREAD_SNAPSHOT}.tmp" "${THREAD_SNAPSHOT}"
    sleep 2
  done
) &
thread_monitor_pid=$!

echo "[ffs-nvblox] nvblox/RViz started; they will wait for FFS depth."
echo "[ffs-nvblox] waiting up to ${FFS_START_TIMEOUT}s for ${FFS_DEPTH_TOPIC} ..."
depth_ready=false
if (
  set +u
  source "${ROS_SETUP}"
  source "${WS_SETUP}"
  set -u
  timeout "${FFS_START_TIMEOUT}" ros2 topic echo --once "${FFS_DEPTH_TOPIC}" \
    --field header >/dev/null 2>&1
); then
  depth_ready=true
fi
if [[ "${depth_ready}" == "true" ]]; then
  echo "[ffs-nvblox] FFS depth is ready."
else
  supervisor_log \
    "warning: depth readiness probe timed out after ${FFS_START_TIMEOUT}s; keeping mapping alive"
  echo "[ffs-nvblox] Check manually: ros2 topic hz ${FFS_DEPTH_TOPIC}" >&2
fi

echo "[ffs-nvblox] started. Press Ctrl-C to stop both processes."
echo "[ffs-nvblox] checks:"
echo "  ros2 topic hz /foundation_stereo/depth"
echo "  ros2 topic hz /nvblox_node/color_layer"
  echo "  ros2 topic hz /visual_cues/annotated_image/compressed"
  echo "  ros2 topic echo --once /untangle/preview_status"

set +e
finished_pid=""
wait -n -p finished_pid "${pids[@]}"
exit_code=$?
set -e
finished_pid="${finished_pid:-unknown}"
if [[ "${finished_pid}" == "${pids[0]}" ]]; then
  finished_name="FFS"
elif [[ "${finished_pid}" == "${pids[1]}" ]]; then
  finished_name="nvblox_launch"
else
  finished_name="unknown"
fi
shutdown_reason="${finished_name} pid=${finished_pid:-unknown} exited with code ${exit_code}"
supervisor_log "${shutdown_reason}; stopping the other child"
exit "${exit_code}"
