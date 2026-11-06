#!/bin/bash
#SBATCH --job-name=bulk-revalidate-goedel
# runs on a GPU-partition node (CPU-only work, but those nodes have lots of cores)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=600G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:gh200:0
#SBATCH --output=logs/bulk_revalidate_goedel_%j.out
#SBATCH --error=logs/bulk_revalidate_goedel_%j.err

# Bulk revalidate the 102K goedel_retry records' final_code through Lean.
# CPU-only — no GPU needed. Uses ProcessPool with N workers, each spawning
# `lake env lean` subprocesses (Mathlib import is ~5-10 GB resident — keep
# workers below RAM/10 to avoid swap thrash).

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


module load Stages/2026
module load Python

source ${VLLM_ENV}/bin/activate
unset PYTHONPATH

export AR_LEAN_PROJECT=${LEAN_TOOLCHAIN}/ar_lean_check
export AR_LEAN_LAKE_BIN=${LEAN_TOOLCHAIN}/.elan/bin/lake

# 400 GB / 8 GB-per-Mathlib-worker ≈ 50, but lake also spawns subprocesses
# of its own; cap at 32 to be safe and let CPU contention sort it out.
WORKERS="${REVALIDATE_WORKERS:-32}"
TIMEOUT_SEC="${REVALIDATE_TIMEOUT:-120}"

cd "${SRC_DIR}"
echo "[$(date +%H:%M:%S)] starting bulk revalidate ($WORKERS workers, ${TIMEOUT_SEC}s timeout per snippet)"
echo "  inputs: ${ANNEAL_ROOT}/proofwiki/active_reading.goedel_retry.job*of4.shard*.jsonl"

python3 -m math_lean.active_reading.bulk_revalidate_goedel \
  --inputs "${ANNEAL_ROOT}/proofwiki/active_reading.goedel_retry.job*of4.shard*.jsonl" \
  --workers "$WORKERS" \
  --timeout-sec "$TIMEOUT_SEC" \
  --output-suffix .revalidated.jsonl

echo "[$(date +%H:%M:%S)] done"
