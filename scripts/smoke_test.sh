#!/usr/bin/env bash
set -euo pipefail

# Smoke test: end-to-end pipeline with a tiny step budget.
# Assumes data/toy/ contains at least 2 images.

cd "$(dirname "$0")/.."

python -m src.train.train_ppo \
  --env-config configs/toy_v0.yaml \
  --ppo-config configs/ppo_default.yaml \
  --total-steps 256 \
  --no-wandb \
  --run-name ppo_smoke

python -m src.eval.run_eval \
  --policy identity \
  --env-config configs/toy_v0.yaml \
  --out /tmp/identity_smoke.csv \
  --n-episodes 5

python -m src.eval.run_eval \
  --policy ppo \
  --ckpt runs/ppo_toy/ppo_best.zip \
  --env-config configs/toy_v0.yaml \
  --out /tmp/ppo_smoke.csv \
  --n-episodes 5

echo "smoke test passed"
