#!/usr/bin/env bash
# Step 1 — Active Reading generation, batched and resumable.
# Skips any batch whose output file already has BATCH_SIZE lines.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

cd "$SRC_DIR"

# Sanity: vLLM reachable
if ! curl -sf --max-time 5 "$AR_ENDPOINT/v1/models" >/dev/null; then
  err "vLLM not reachable at $AR_ENDPOINT — aborting"
  exit 1
fi

log "AR generation: $N_BATCHES batches × $BATCH_SIZE theorems → $OUT_DIR"
log "  endpoint=$AR_ENDPOINT model=$AR_MODEL workers=$WORKERS_AR"

for i in $(seq 0 $((N_BATCHES - 1))); do
  OFFSET=$((i * BATCH_SIZE))
  OUT="$OUT_DIR/batch_${i}.jsonl"
  LOG="$OUT_DIR/batch_${i}.log"

  if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge "$BATCH_SIZE" ]; then
    ok "batch $i complete ($(wc -l < "$OUT") records), skipping"
    continue
  fi

  log "batch $i (offset=$OFFSET) → $OUT"
  "$PYTHON" -u -m math_lean.active_reading \
      --proofwiki "$PROOFWIKI" \
      --offset "$OFFSET" --n "$BATCH_SIZE" --seed "$SEED" \
      --workers "$WORKERS_AR" \
      --flavours "$FLAVOURS" \
      --endpoint "$AR_ENDPOINT" --model "$AR_MODEL" \
      --apply-max-tokens "$APPLY_MAX_TOKENS" \
      --min-proof-chars "$MIN_PROOF_CHARS" \
      --max-chunk-chars "$MAX_CHUNK_CHARS" \
      --out "$OUT" \
      > "$LOG" 2>&1

  ok "batch $i finished — $(wc -l < "$OUT") records (log: $LOG)"
done

ok "all batches done"
