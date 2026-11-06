#!/bin/bash
# Launch Goedel-Prover-V2 on the GH200 via vLLM OpenAI-compatible API.
#
# Usage:
#   bash launch_goedel_vllm.sh [MODEL_SIZE]
#       MODEL_SIZE: 8B (default) or 32B
#
# Server listens on http://localhost:8002
# Stop with Ctrl-C or `pkill -f vllm.entrypoints`
#
# Memory budget on GH200 480GB:
#   8B  bf16 ≈ 16 GB weights → KV-cache ~440 GB → tons of room
#   32B bf16 ≈ 64 GB weights → KV-cache ~400 GB → fine for 40K context

set -euo pipefail

# ── repo bootstrap: locate the checkout (also from a SLURM spool copy) + site config ──
if [ -z "${REPO_ROOT:-}" ]; then
  REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null \
            || git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
fi
export REPO_ROOT
[ -f "$REPO_ROOT/configs/site.env" ] && : "${SITE_ENV:=$REPO_ROOT/configs/site.env}"
source "${SITE_ENV:-$REPO_ROOT/configs/site.env.example}"
# ─────────────────────────────────────────────────────────────────────────────


SIZE="${1:-32B}"
MODEL="Goedel-LM/Goedel-Prover-V2-${SIZE}"
PORT="${GOEDEL_PORT:-8002}"
MAX_LEN="${GOEDEL_MAX_LEN:-40960}"

# Load HPC modules + env
# (run via `bash -l -c` so module commands are recognised)

if [ -z "${MODULESHOME:-}" ]; then
  echo "ERROR: this script must be run via 'bash -l -c' so module commands work." >&2
  exit 1
fi

module load Stages/2026
module load Python
module load CUDA-Python/13
module load cuDNN/9.19.0.56-CUDA-13
module load NCCL/default-CUDA-13

source ${VLLM_ENV}/bin/activate
unset PYTHONPATH  # Critical: Python-bundle-PyPI shadows venv otherwise

# HF token from ~/.bashrc (so we can pull the gated model if needed; Goedel is Apache-2.0 but no harm)
source ~/.bashrc 2>/dev/null || true

echo "[$(date +%H:%M:%S)] Launching $MODEL on port $PORT (max_model_len=$MAX_LEN)"
echo "[$(date +%H:%M:%S)] First run will download ~64GB (32B) or ~16GB (8B) — be patient"

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --served-model-name "$MODEL" \
  --enable-prefix-caching
