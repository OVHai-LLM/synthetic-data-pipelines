#!/bin/bash
# Watcher: polls the queue and submits 4 goedel jobs as soon as the running
# CPU count drops below the threshold (i.e., enough enrich jobs have finished).
#
# Run in background:
#   nohup bash auto_submit_goedel_when_free.sh > goedel_watcher.log 2>&1 &
set -uo pipefail

# ── repo bootstrap: locate the checkout (also from a SLURM spool copy) + site config ──
if [ -z "${REPO_ROOT:-}" ]; then
  REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null \
            || git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
fi
export REPO_ROOT
[ -f "$REPO_ROOT/configs/site.env" ] && : "${SITE_ENV:=$REPO_ROOT/configs/site.env}"
source "${SITE_ENV:-$REPO_ROOT/configs/site.env.example}"
# ─────────────────────────────────────────────────────────────────────────────


THRESHOLD_CPUS="${THRESHOLD_CPUS:-7500}"   # need ~1152 CPU headroom for 4 goedel jobs (4*288); current 8640 cap
SBATCH=${REPO_ROOT}/scripts/active_reading_proofwiki/goedel_retry_dp4.sbatch
NUM_JOBS=4

echo "[$(date)] watcher started — will submit when running CPUs < $THRESHOLD_CPUS"

while true; do
  RUNNING_CPUS=$(squeue --me -h -t R -o '%C' | awk '{s+=$1} END {print s+0}')
  echo "[$(date +%H:%M:%S)] running CPUs: $RUNNING_CPUS"
  if [ "$RUNNING_CPUS" -lt "$THRESHOLD_CPUS" ]; then
    echo "[$(date +%H:%M:%S)] threshold reached, submitting 4 goedel jobs..."
    OK=0
    for J in 0 1 2 3; do
      JID=$(GOEDEL_JOB_SHARD_ID=$J GOEDEL_JOB_NUM_SHARDS=$NUM_JOBS \
            sbatch --parsable "$SBATCH" 2>&1)
      if [[ "$JID" =~ ^[0-9]+$ ]]; then
        echo "  shard $J/$NUM_JOBS → $JID"
        OK=$((OK+1))
      else
        echo "  shard $J/$NUM_JOBS → still rejected: $JID"
      fi
    done
    if [ "$OK" -eq "$NUM_JOBS" ]; then
      echo "[$(date)] all 4 submitted — exiting watcher"
      exit 0
    else
      echo "[$(date)] only $OK / $NUM_JOBS submitted — backing off and retrying"
      sleep 60
    fi
  fi
  sleep 120
done
