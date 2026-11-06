#!/usr/bin/env bash
# Step 5 — Compose final per-snippet verdicts and emit the pretraining JSONL.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

cd "$SRC_DIR"

if [ ! -f "$ALL_AR" ] || [ ! -f "$ALL_CHECK" ]; then
  err "missing $ALL_AR or $ALL_CHECK — run earlier steps first"
  exit 1
fi

# 5a. Compose final verdicts (overlay retry1 then retry2 on the original check)
log "composing final verdicts → $CHECK_FINAL"
RETRY1_ARG="$ALL_RETRY1"; [ -f "$RETRY1_ARG" ] || RETRY1_ARG=""
RETRY2_ARG="$ALL_RETRY2"; [ -f "$RETRY2_ARG" ] || RETRY2_ARG=""
"$PYTHON" "$HERE/helpers/compose_final_verdicts.py" \
    "$ALL_CHECK" "$RETRY1_ARG" "$RETRY2_ARG" "$CHECK_FINAL"

# 5b. Build agnostic side
log "building agnostic pretraining slice → $PRETRAIN_AGNOSTIC"
"$PYTHON" -m math_lean.active_reading.to_pretraining \
    --runs-input "$ALL_AR" \
    --out "$PRETRAIN_AGNOSTIC" \
    --policy "$PRETRAIN_POLICY" \
    --only-flavour agnostic \
    > "$OUT_DIR/to_pretraining_agnostic.log" 2>&1

# 5c. Build lean side using composed verdicts
log "building lean pretraining slice (policy=$PRETRAIN_POLICY) → $PRETRAIN_LEAN"
"$PYTHON" -m math_lean.active_reading.to_pretraining \
    --runs-input "$ALL_AR" \
    --lean-check "$CHECK_FINAL" \
    --out "$PRETRAIN_LEAN" \
    --policy "$PRETRAIN_POLICY" \
    --only-flavour lean \
    > "$OUT_DIR/to_pretraining_lean.log" 2>&1

# 5d. Concat
cat "$PRETRAIN_AGNOSTIC" "$PRETRAIN_LEAN" > "$PRETRAIN_OUT"

N_AG=$(wc -l < "$PRETRAIN_AGNOSTIC")
N_LN=$(wc -l < "$PRETRAIN_LEAN")
N_TOTAL=$(wc -l < "$PRETRAIN_OUT")
SIZE=$(du -h "$PRETRAIN_OUT" | cut -f1)
ok "wrote $PRETRAIN_OUT — $N_TOTAL records ($N_AG agnostic + $N_LN lean), $SIZE"
