#!/bin/bash
# Launch Gemma4-31B FP8 with bigger prefill batch + higher seq cap
# Used by enrich_async smoke test and the enrich sbatch script.
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

PORT="${PORT:-8002}"
GPU="${GPU:-0}"
LOG="${LOG:-${LOG_DIR}/vllm_gemma4_p${PORT}.log}"

module load Stages/2026 Python CUDA/13 CUDA-Python/13 cuDNN/9.19.0.56-CUDA-13 NCCL/default-CUDA-13 nvidia-compilers/25.9-CUDA-13
source ${VLLM_ENV}/bin/activate
unset PYTHONPATH

echo "[$(date +%H:%M:%S)] Gemma4-31B FP8 on GPU=${GPU} port=${PORT}, log=${LOG}"
exec env CUDA_VISIBLE_DEVICES="${GPU}" vllm serve RedHatAI/gemma-4-31B-it-FP8-Dynamic \
  --max-model-len 32768 \
  --max-num-batched-tokens 65536 \
  --max-num-seqs 512 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.92 \
  --kv-cache-dtype fp8 \
  --reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice \
  --limit-mm-per-prompt '{"image":0,"audio":0}' \
  --port "${PORT}" \
  >> "${LOG}" 2>&1
