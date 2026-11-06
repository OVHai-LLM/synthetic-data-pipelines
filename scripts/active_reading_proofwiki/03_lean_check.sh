#!/usr/bin/env bash
# Step 3 — Compile-check every Lean snippet in the merged AR output.
# Idempotent: skips if the output file exists and matches the input length.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

cd "$SRC_DIR"

if [ ! -f "$ALL_AR" ]; then
  err "$ALL_AR missing — run 02_concat.sh first"
  exit 1
fi

if [ -f "$ALL_CHECK" ] && [ "$(wc -l < "$ALL_CHECK")" -gt 0 ]; then
  log "$ALL_CHECK already exists ($(wc -l < "$ALL_CHECK") records); delete to re-run"
  exit 0
fi

log "lean_check on $ALL_AR → $ALL_CHECK"
"$PYTHON" -u -m math_lean.active_reading.lean_check \
    --input "$ALL_AR" \
    --out "$ALL_CHECK" \
    --workers "$WORKERS_CHECK" \
    --timeout-sec "$CHECK_TIMEOUT_SEC" \
    > "$OUT_DIR/lean_check.log" 2>&1

ok "lean_check finished — $(wc -l < "$ALL_CHECK") snippet verdicts (log: $OUT_DIR/lean_check.log)"
