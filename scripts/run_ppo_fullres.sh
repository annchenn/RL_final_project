#!/usr/bin/env bash
# Train PPO on full-resolution images (no dataset resize) for both training and
# in-loop eval. Saves under runs/ppo_fullres_<timestamp>/ so it does not
# overwrite runs/ppo_toy/.
set -euo pipefail

cd "$(dirname "$0")/.."

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="ppo_fullres_${TIMESTAMP}"
LOG_DIR="runs/${RUN_NAME}"

python -m src.train.train_ppo \
  --env-config configs/toy_v0.yaml \
  --ppo-config configs/ppo_default.yaml \
  --run-name "${RUN_NAME}" \
  --log-dir "${LOG_DIR}" \
  --full-res \
  "$@"

echo "[run_ppo_fullres] training finished. Checkpoint: ${LOG_DIR}/ppo_best.zip"
echo "[run_ppo_fullres] eval with: CKPT=${LOG_DIR}/ppo_best.zip bash scripts/eval.sh"
