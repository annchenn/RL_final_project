#!/usr/bin/env bash
# Eval PPO checkpoint deterministically over every image in the eval dataset
# (one episode per image) and print Input/Output/Improvement TOPIQ for the
# "Our RL Model" row.
# Override the checkpoint with: CKPT=path/to/ckpt.zip bash scripts/eval.sh
set -euo pipefail

cd "$(dirname "$0")/.."
TIMESTAMP="tmp"
CKPT="runs/ppo_toy/ppo_final.zip"
ENV_CFG="configs/toy_v0.yaml"
RESULT_DIR="results/${TIMESTAMP}"
OUT="${RESULT_DIR}/ppo_eval.csv"

if [ ! -f "${CKPT}" ]; then
  echo "[eval] ERROR: PPO checkpoint not found at ${CKPT}"
  echo "[eval] run scripts/run_ppo.sh first, or pass CKPT=... env var"
  exit 1
fi

mkdir -p "${RESULT_DIR}"

echo "[eval] PPO checkpoint: ${CKPT}"
python -m src.eval.run_eval \
  --policy ppo \
  --ckpt "${CKPT}" \
  --env-config "${ENV_CFG}" \
  --out "${OUT}" \
  --qual-dir "${RESULT_DIR}" \
  --all-images

python - "${OUT}" <<'EOF'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
inp = df["score_before"].mean()
out = df["score_after"].mean()
print()
print(f"=== Our RL Model (n={len(df)} images) ===")
print(f"Input TOPIQ:   {inp:.4f}")
print(f"Output TOPIQ:  {out:.4f}")
print(f"Improvement:   {out - inp:+.4f}")
EOF
