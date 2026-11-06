#!/usr/bin/env bash
# Outer loop — runs every language sequentially. Each language is fully
# resumable on its own; if a language run dies you can re-invoke this
# script and finished languages get skipped automatically.
#
# Recommended:
#   nohup ./run_all_languages.sh > /tmp/wiki_full.log 2>&1 &
#   disown
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config_wiki.sh"

START=$(date +%s)
log "=== Wikipedia Active Reading — $(echo "$LANGUAGES" | wc -w) languages ==="
log "OUT_DIR=$OUT_DIR  MAX_ARTICLES_PER_LANG=$MAX_ARTICLES_PER_LANG  WORKERS_AR=$WORKERS_AR"

DONE=0
SKIPPED=0
for LANG_CODE in $LANGUAGES; do
  LANG_OUT="$(lang_out_dir "$LANG_CODE")"
  PRETRAIN="$LANG_OUT/pretraining_${PRETRAIN_POLICY}.jsonl"

  if [ -f "$PRETRAIN" ] && [ "$(wc -l < "$PRETRAIN")" -gt 0 ]; then
    ok "[$LANG_CODE] pretraining slice already exists, skipping"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  if ! lang_dump_path "$LANG_CODE" >/dev/null 2>&1; then
    err "[$LANG_CODE] no dump found; skipping"
    continue
  fi

  bash "$HERE/run_lang.sh" "$LANG_CODE"
  DONE=$((DONE + 1))
done

ELAPSED=$(( $(date +%s) - START ))
ok "all languages: done=$DONE skipped=$SKIPPED in $((ELAPSED / 3600))h $(( (ELAPSED % 3600) / 60))m"
