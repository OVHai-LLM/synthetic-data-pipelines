#!/usr/bin/env bash
# Step 4 — Lean retry round. Pass the round number as $1 (1 or 2).
#
# Round 1 reads $ALL_CHECK; round 2 first composes a check-style file
# from round 1's output, then runs against that.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

ROUND="${1:-}"
if [ "$ROUND" != "1" ] && [ "$ROUND" != "2" ]; then
  err "usage: $0 <1|2>"
  exit 2
fi

cd "$SRC_DIR"

if ! curl -sf --max-time 5 "$RETRY_ENDPOINT/v1/models" >/dev/null; then
  err "retry endpoint $RETRY_ENDPOINT not reachable — aborting"
  exit 1
fi

if [ "$ROUND" = "1" ]; then
  CHECK_IN="$ALL_CHECK"
  RETRY_OUT="$ALL_RETRY1"
  LOG="$OUT_DIR/lean_retry1.log"
else
  if [ ! -f "$ALL_RETRY1" ]; then
    err "round 1 output $ALL_RETRY1 missing — run round 1 first"
    exit 1
  fi
  log "composing post-round-1 check file from $ALL_RETRY1"
  "$PYTHON" "$HERE/helpers/compose_after_retry.py" \
      "$ALL_RETRY1" "$CHECK_AFTER_R1"
  CHECK_IN="$CHECK_AFTER_R1"
  RETRY_OUT="$ALL_RETRY2"
  LOG="$OUT_DIR/lean_retry2.log"
fi

if [ -f "$RETRY_OUT" ] && [ "$(wc -l < "$RETRY_OUT")" -gt 0 ]; then
  log "$RETRY_OUT already exists ($(wc -l < "$RETRY_OUT") records); delete to re-run"
  exit 0
fi

log "lean_retry round $ROUND  ($CHECK_IN → $RETRY_OUT)"
log "  endpoint=$RETRY_ENDPOINT model=$RETRY_MODEL workers=$WORKERS_RETRY"

"$PYTHON" -u -m math_lean.active_reading.lean_retry \
    --runs-input "$ALL_AR" \
    --check-input "$CHECK_IN" \
    --out "$RETRY_OUT" \
    --endpoint "$RETRY_ENDPOINT" \
    --model "$RETRY_MODEL" \
    --workers "$WORKERS_RETRY" \
    --retry-temperature "$RETRY_TEMPERATURE" \
    --retry-max-tokens "$RETRY_MAX_TOKENS" \
    > "$LOG" 2>&1

ok "round $ROUND finished — $(wc -l < "$RETRY_OUT") records (log: $LOG)"
