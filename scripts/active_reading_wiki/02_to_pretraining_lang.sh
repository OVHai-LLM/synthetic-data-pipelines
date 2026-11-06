#!/usr/bin/env bash
# Step 2 (per-language) — Compose the pretraining JSONL for one language.
# No lean_check involved (wiki has no Lean blocks); we just run
# to_pretraining with no --lean-check so all artifacts pass.
#
# Usage:  ./02_to_pretraining_lang.sh <lang>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config_wiki.sh"

LANG_CODE="${1:-}"
[ -z "$LANG_CODE" ] && { err "usage: $0 <lang>"; exit 2; }
LANG_OUT="$(lang_out_dir "$LANG_CODE")"
ALL="$LANG_OUT/all.jsonl"

if [ ! -f "$ALL" ]; then
  err "[$LANG_CODE] missing $ALL — run 01_generate_lang.sh first"
  exit 1
fi

OUT="$LANG_OUT/pretraining_${PRETRAIN_POLICY}.jsonl"

cd "$SRC_DIR"
log "[$LANG_CODE] building pretraining slice → $OUT (policy=$PRETRAIN_POLICY)"

# wiki has no Lean blocks → no --lean-check needed; the medium policy
# falls through to "keep all" since every artifact has 0 lean blocks
"$PYTHON" -m math_lean.active_reading.to_pretraining \
    --runs-input "$ALL" \
    --out "$OUT" \
    --policy "$PRETRAIN_POLICY" \
    > "$LANG_OUT/to_pretraining.log" 2>&1

N=$(wc -l < "$OUT")
SIZE=$(du -h "$OUT" | cut -f1)
ok "[$LANG_CODE] wrote $OUT — $N records, $SIZE"
