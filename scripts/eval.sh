#!/usr/bin/env bash
# Eval PPO checkpoint deterministically over every image in the eval dataset
# (one episode per image) and print Input/Output/Improvement TOPIQ for the
# "Our RL Model" row.
#
# Defaults to the best-by-eval checkpoint of the most recent training run:
#   runs/latest/best_model.zip
# Override with CKPT=path/to/ckpt.zip bash scripts/eval.sh
set -euo pipefail

cd "$(dirname "$0")/.."

CKPT="${CKPT:-runs/latest/best_model.zip}"
ENV_CFG="configs/toy_v0.yaml"
# Tag results dir by the run name (or "manual" if user supplied CKPT directly).
RUN_NAME="$(basename "$(dirname "$(readlink -f "${CKPT}" 2>/dev/null || echo "${CKPT}")")" 2>/dev/null || echo "manual")"
CKPT_TAG="$(basename "${CKPT}" .zip)"
RESULT_DIR="results/${RUN_NAME}/${CKPT_TAG}"
OUT="${RESULT_DIR}/ppo_eval.csv"

if [ ! -f "${CKPT}" ]; then
  echo "[eval] ERROR: PPO checkpoint not found at ${CKPT}"
  echo "[eval] run scripts/run_ppo.sh first, or pass CKPT=... env var"
  exit 1
fi

mkdir -p "${RESULT_DIR}"

echo "[eval] PPO checkpoint: ${CKPT}"
echo "[eval] results dir:    ${RESULT_DIR}"
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
print(f"=== Our RL Model (n={len(df)} images, scaled units) ===")
print(f"Input TOPIQ*scale:   {inp:.4f}")
print(f"Output TOPIQ*scale:  {out:.4f}")
print(f"Improvement:         {out - inp:+.4f}")
EOF
