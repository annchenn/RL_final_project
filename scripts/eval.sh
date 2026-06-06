#!/usr/bin/env bash
# End-to-end test-set eval:
#   1. Apply the latest PPO policy to every image in TEST_DIR. Each enhanced
#      output and both CSVs go to results/<run>/<ckpt>/.
#   2. Run evaluation.py over that folder to print TOPIQ scores.
#
# Override the checkpoint with: CKPT=path/to/ckpt.zip ./scripts/eval.sh
# Override the test dir with:   TEST_DIR=path/to/folder ./scripts/eval.sh
set -euo pipefail

cd "$(dirname "$0")/.."

CKPT="${CKPT:-runs/latest/best_model.zip}"
TEST_DIR="${TEST_DIR:-degraded-data/landscape-50-test}"
ENV_CFG="configs/toy_v0.yaml"

# Tag output dir by the run name (or "manual" if user supplied CKPT directly).
RUN_NAME="$(basename "$(dirname "$(readlink -f "${CKPT}" 2>/dev/null || echo "${CKPT}")")" 2>/dev/null || echo "manual")"
CKPT_TAG="$(basename "${CKPT}" .zip)"
RESULT_DIR="results/${RUN_NAME}/${CKPT_TAG}"
OUT="${RESULT_DIR}/ppo_eval.csv"
TOPIQ_OUT="${RESULT_DIR}/topiq_scores.csv"

if [ ! -f "${CKPT}" ]; then
  echo "[eval] ERROR: PPO checkpoint not found at ${CKPT}"
  echo "[eval] run scripts/run_ppo.sh first, or pass CKPT=... env var"
  exit 1
fi

if [ ! -d "${TEST_DIR}" ]; then
  echo "[eval] ERROR: test image dir not found at ${TEST_DIR}"
  exit 1
fi

mkdir -p "${RESULT_DIR}"

echo "[eval] checkpoint:  ${CKPT}"
echo "[eval] test dir:    ${TEST_DIR}"
echo "[eval] output dir:  ${RESULT_DIR}  (enhanced images + CSVs all here)"
echo

# Step 1: roll out the policy on every test image. Enhanced outputs are saved
# with the original filename into RESULT_DIR; per-episode rollout stats go to
# ppo_eval.csv in the same dir.
python -m src.eval.run_eval \
  --policy ppo \
  --ckpt "${CKPT}" \
  --env-config "${ENV_CFG}" \
  --image-dir "${TEST_DIR}" \
  --save-enhanced-dir "${RESULT_DIR}" \
  --out "${OUT}" \
  --all-images

echo
echo "[eval] === before/after summary (from rollout CSV, scaled TOPIQ units) ==="
python - "${OUT}" <<'EOF'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
inp = df["score_before"].mean()
out = df["score_after"].mean()
print(f"  n images           : {len(df)}")
print(f"  Input TOPIQ*scale  : {inp:.4f}")
print(f"  Output TOPIQ*scale : {out:.4f}")
print(f"  Improvement        : {out - inp:+.4f}")
EOF

# Step 2: independent scoring of the enhanced images via evaluation.py
# (scale=100 to match the units used everywhere else in the pipeline).
echo
echo "[eval] === evaluation.py on enhanced images ==="
python evaluation.py "${RESULT_DIR}" --scale 100 --out "${TOPIQ_OUT}"
