#!/bin/bash
# Submit 4 parallel sbatch jobs, each on its own node, processing 1/4 of snippets.
# Total: 4 nodes × 4 GH200 = 16 GPUs running.
#
# Usage:
#   bash submit_4parallel.sh [GOEDEL_SKIP_LEAN=1] [GOEDEL_MAX_SNIPPETS=N]
# Env vars (export before running) are forwarded to each sbatch.

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


SCRIPT=${REPO_ROOT}/scripts/active_reading_proofwiki/goedel_retry_dp4.sbatch
NUM_JOBS=4

echo "Submitting $NUM_JOBS parallel sbatch jobs (each = 1 node × 4 GH200)..."
JOB_IDS=()
for J in $(seq 0 $((NUM_JOBS-1))); do
  JID=$(GOEDEL_JOB_SHARD_ID=$J GOEDEL_JOB_NUM_SHARDS=$NUM_JOBS sbatch --parsable "$SCRIPT")
  echo "  job $J/$NUM_JOBS → JobID $JID"
  JOB_IDS+=("$JID")
done

echo ""
echo "Submitted: ${JOB_IDS[*]}"
echo ""
echo "Watch with:  squeue -u \$USER -j $(IFS=,; echo "${JOB_IDS[*]}")"
echo ""
echo "When ALL 4 finish, merge across jobs:"
echo "  bash ${REPO_ROOT}/scripts/active_reading_proofwiki/merge_goedel_outputs.sh $NUM_JOBS"
