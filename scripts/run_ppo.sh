#!/usr/bin/env bash
# Train PPO end-to-end. Eval lives in scripts/eval.sh.
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_NAME="ppo_toy_$(date +%Y%m%d_%H%M%S)"

python -m src.train.train_ppo \
  --env-config configs/toy_v0.yaml \
  --ppo-config configs/ppo_default.yaml \
  --run-name "${RUN_NAME}" \
  "$@"

echo "[run_ppo] training finished. Checkpoint: runs/ppo_toy/ppo_best.zip"
echo "[run_ppo] next: bash scripts/eval.sh"
