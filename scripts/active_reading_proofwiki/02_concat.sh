#!/usr/bin/env bash
# Step 2 — Concatenate all batch_N.jsonl files into a single all.jsonl.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

shopt -s nullglob
BATCHES=("$OUT_DIR"/batch_*.jsonl)
if [ "${#BATCHES[@]}" -eq 0 ]; then
  err "no batch_*.jsonl files in $OUT_DIR — run 01_generate_batched.sh first"
  exit 1
fi

log "concatenating ${#BATCHES[@]} batches → $ALL_AR"
cat "${BATCHES[@]}" > "$ALL_AR"
ok "wrote $ALL_AR — $(wc -l < "$ALL_AR") records, $(du -h "$ALL_AR" | cut -f1)"
