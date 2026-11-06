#!/usr/bin/env bash
# Step 1 (per-language) — Active Reading on one Wikipedia language.
# Batched and resumable: skips any batch whose output file already has
# BATCH_SIZE lines.
#
# Usage:  ./01_generate_lang.sh <lang>
#         ./01_generate_lang.sh lt
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config_wiki.sh"

LANG_CODE="${1:-}"
if [ -z "$LANG_CODE" ]; then
  err "usage: $0 <lang>"; exit 2
fi

DUMP="$(lang_dump_path "$LANG_CODE")" || {
  err "no dump found under $WIKI_DUMP_ROOT/$LANG_CODE/wikipedia-*/$LANG_CODE.jsonl.gz"
  exit 1
}
LANG_OUT="$(lang_out_dir "$LANG_CODE")"
mkdir -p "$LANG_OUT"

# Sanity: vLLM reachable
if ! curl -sf --max-time 5 "$AR_ENDPOINT/v1/models" >/dev/null; then
  err "vLLM not reachable at $AR_ENDPOINT — aborting"
  exit 1
fi

# How many batches do we need? min(MAX_ARTICLES_PER_LANG, eligible) / BATCH_SIZE
# Cap by MAX_ARTICLES_PER_LANG to keep multi-language runs tractable.
if [ "$MAX_ARTICLES_PER_LANG" -gt 0 ]; then
  TOTAL_TARGET="$MAX_ARTICLES_PER_LANG"
else
  # full language — query the loader for the eligible count
  TOTAL_TARGET=$("$PYTHON" -c "
from math_lean.active_reading.wiki_loader import count_eligible
print(count_eligible('$DUMP', min_tokens=$MIN_TOKENS, max_tokens=$MAX_TOKENS))
")
fi
N_BATCHES=$(( (TOTAL_TARGET + BATCH_SIZE - 1) / BATCH_SIZE ))

log "[$LANG_CODE] dump=$DUMP"
log "[$LANG_CODE] total_target=$TOTAL_TARGET batches=$N_BATCHES × $BATCH_SIZE"
log "[$LANG_CODE] endpoint=$AR_ENDPOINT model=$AR_MODEL workers=$WORKERS_AR flavours=$FLAVOURS"

cd "$SRC_DIR"
for i in $(seq 0 $((N_BATCHES - 1))); do
  OFFSET=$((i * BATCH_SIZE))
  OUT="$LANG_OUT/batch_${i}.jsonl"
  LOG="$LANG_OUT/batch_${i}.log"

  if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge "$BATCH_SIZE" ]; then
    ok "[$LANG_CODE] batch $i complete ($(wc -l < "$OUT") records), skipping"
    continue
  fi

  log "[$LANG_CODE] batch $i (offset=$OFFSET) → $OUT"
  "$PYTHON" -u -m math_lean.active_reading.wiki_main \
      --input "$DUMP" \
      --language "$LANG_CODE" \
      --offset "$OFFSET" --n "$BATCH_SIZE" --seed "$SEED" \
      --workers "$WORKERS_AR" \
      --flavours "$FLAVOURS" \
      --endpoint "$AR_ENDPOINT" --model "$AR_MODEL" \
      --apply-max-tokens "$APPLY_MAX_TOKENS" \
      --min-tokens "$MIN_TOKENS" \
      --max-tokens "$MAX_TOKENS" \
      --out "$OUT" \
      > "$LOG" 2>&1

  ok "[$LANG_CODE] batch $i finished — $(wc -l < "$OUT") records"
done

# Concat
ALL="$LANG_OUT/all.jsonl"
shopt -s nullglob
BATCHES=("$LANG_OUT"/batch_*.jsonl)
cat "${BATCHES[@]}" > "$ALL"
ok "[$LANG_CODE] all batches done → $ALL ($(wc -l < "$ALL") records)"
