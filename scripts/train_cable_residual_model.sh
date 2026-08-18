#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASET="${DATASET:-${WORKSPACE}/experiment_data/cable_dynamics_residual/learning/cable_residual_dataset_attention.npz}"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/experiment_data/cable_dynamics_residual/learning/models/cable_residual_attention_gru_${TIMESTAMP}}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"

set +u
source /opt/ros/jazzy/setup.bash
if [[ -f "${WORKSPACE}/install/setup.bash" ]]; then
  source "${WORKSPACE}/install/setup.bash"
fi
set -u

echo "[residual-training] Preparing E4-E6 bags: ${DATASET}"
/usr/bin/python3 "${WORKSPACE}/scripts/prepare_cable_residual_dataset.py" \
  --data-root "${WORKSPACE}/experiment_data/cable_dynamics_residual" \
  --output "${DATASET}" \
  --include-e7-baseline

echo "[residual-training] Training attention-GRU ensemble: ${OUTPUT_DIR}"
"${PYTHON_EXECUTABLE}" \
  "${WORKSPACE}/scripts/train_cable_residual_gru.py" \
  --dataset "${DATASET}" \
  --output-dir "${OUTPUT_DIR}" \
  --architecture attention_gru \
  "$@"

LATEST_LINK="$(dirname "${OUTPUT_DIR}")/latest_attention_gru"
ln -sfn "$(basename "${OUTPUT_DIR}")" "${LATEST_LINK}"
echo "[residual-training] Complete: ${OUTPUT_DIR}"
echo "[residual-training] Latest model: ${LATEST_LINK}"
