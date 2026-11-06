#!/usr/bin/env bash
# Per-language orchestrator: generate then build pretraining JSONL.
#
# Usage:  ./run_lang.sh <lang>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config_wiki.sh"

LANG_CODE="${1:-}"
[ -z "$LANG_CODE" ] && { err "usage: $0 <lang>"; exit 2; }

START=$(date +%s)
log "=== [$LANG_CODE] full pipeline ==="

bash "$HERE/01_generate_lang.sh" "$LANG_CODE"
bash "$HERE/02_to_pretraining_lang.sh" "$LANG_CODE"

ELAPSED=$(( $(date +%s) - START ))
ok "[$LANG_CODE] done in $((ELAPSED / 3600))h $(( (ELAPSED % 3600) / 60))m"
