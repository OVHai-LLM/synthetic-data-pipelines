#!/usr/bin/env bash
# Orchestrator — runs every step in order. Each step is idempotent;
# safe to re-invoke after a crash.
#
# Knobs live in config.sh. Override before running, e.g.
#   N_BATCHES=4 BATCH_SIZE=250 WORKERS_AR=32 ./run_all.sh
#
# Recommended invocation for the real run:
#   nohup ./run_all.sh > /tmp/ar_full.log 2>&1 &
#   disown
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

START=$(date +%s)
log "=== Active Reading full pipeline ==="
log "OUT_DIR=$OUT_DIR"
log "AR endpoint:    $AR_ENDPOINT  ($AR_MODEL)"
log "retry endpoint: $RETRY_ENDPOINT  ($RETRY_MODEL)"
log "batches: $N_BATCHES × $BATCH_SIZE = $((N_BATCHES * BATCH_SIZE)) theorems"
log "policy: $PRETRAIN_POLICY"
log

bash "$HERE/01_generate_batched.sh"
bash "$HERE/02_concat.sh"
bash "$HERE/03_lean_check.sh"
#bash "$HERE/04_lean_retry.sh" 1
#bash "$HERE/04_lean_retry.sh" 2
#bash "$HERE/05_to_pretraining.sh"

ELAPSED=$(( $(date +%s) - START ))
ok "pipeline complete in $((ELAPSED / 3600))h $(( (ELAPSED % 3600) / 60))m"
ok "final corpus: $PRETRAIN_OUT"
