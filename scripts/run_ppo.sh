#!/usr/bin/env bash
# Train PPO end-to-end. Eval lives in scripts/eval.sh.
#
# Each run gets its own timestamped log dir so re-running never overwrites prior
# checkpoints. `runs/latest` is updated to point at the most recent run, which
# is what scripts/eval.sh reads by default.
set -euo pipefail

cd "$(dirname "$0")/.."

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="ppo_${TIMESTAMP}"
LOG_DIR="runs/${RUN_NAME}"

python -m src.train.train_ppo \
  --env-config configs/toy_v0.yaml \
  --ppo-config configs/ppo_default.yaml \
  --run-name "${RUN_NAME}" \
  --log-dir "${LOG_DIR}" \
  "$@"

echo "[run_ppo] training finished."
echo "[run_ppo]   best:        ${LOG_DIR}/best_model.zip"
echo "[run_ppo]   final:       ${LOG_DIR}/final_model.zip"
echo "[run_ppo]   checkpoints: ${LOG_DIR}/checkpoints/"
echo "[run_ppo]   latest:      runs/latest -> ${RUN_NAME}"
echo "[run_ppo] next: bash scripts/eval.sh"
