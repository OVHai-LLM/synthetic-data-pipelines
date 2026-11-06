#!/usr/bin/env bash
# Shared configuration for the per-language Wikipedia AR pipeline.
# Sourced by every wiki step script. Override any var by exporting it.

# ─── source dump root ───────────────────────────────────────────────────

# ── repo bootstrap: locate the checkout (also from a SLURM spool copy) + site config ──
if [ -z "${REPO_ROOT:-}" ]; then
  REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null \
            || git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
fi
export REPO_ROOT
[ -f "$REPO_ROOT/configs/site.env" ] && : "${SITE_ENV:=$REPO_ROOT/configs/site.env}"
source "${SITE_ENV:-$REPO_ROOT/configs/site.env.example}"
# ─────────────────────────────────────────────────────────────────────────────

: "${WIKI_DUMP_ROOT:=$ANNEAL_ROOT/data}"
: "${OUT_DIR:=$AR_DATA_DIR/wiki}"
: "${PYTHON:=python3}"

# ─── languages to process ───────────────────────────────────────────────
# 28 ISO-639 codes available under $WIKI_DUMP_ROOT — set explicitly so the
# loop is deterministic. Trim/reorder as needed.
: "${LANGUAGES:=ar bs ca cs da et fi ga he hr hu is it ka ko lb lt lv no ro sk sl sq sr sv tr uk zh}"

# ─── per-language run knobs ─────────────────────────────────────────────
: "${BATCH_SIZE:=1000}"           # articles per batch — keeps re-runs cheap
: "${MAX_ARTICLES_PER_LANG:=20000}"  # 0 = no cap (full language). Default 20k
                                  # for tractability across 27+ languages.
: "${SEED:=0}"
: "${WORKERS_AR:=32}"             # raise to 48-64 if vLLM has headroom
: "${FLAVOURS:=agnostic,factual}"
: "${APPLY_MAX_TOKENS:=2048}"
: "${MIN_TOKENS:=200}"
: "${MAX_TOKENS:=6000}"

# ─── vLLM endpoint ──────────────────────────────────────────────────────
: "${AR_ENDPOINT:=http://localhost:8002}"
: "${AR_MODEL:=RedHatAI/gemma-4-31B-it-FP8-Dynamic}"

# ─── filter policy for the pretraining corpus ───────────────────────────
# For wiki we don't have compile gating, so 'medium' and 'strict' both
# fall through to 'keep everything kept' — wiki uses 'medium' as default
# which means: keep all artifacts (no Lean to break).
: "${PRETRAIN_POLICY:=medium}"

# ─── helpers ────────────────────────────────────────────────────────────
mkdir -p "$OUT_DIR"

log() { printf '\033[1;34m[wiki]\033[0m %s\n' "$*" >&2; }
ok()  { printf '\033[1;32m[OK]  \033[0m %s\n' "$*" >&2; }
err() { printf '\033[1;31m[!!]  \033[0m %s\n' "$*" >&2; }

# Locate the .jsonl.gz dump for a given language code. The annealing
# tree puts it under <root>/<lang>/wikipedia-<N>/<lang>.jsonl.gz where
# N varies. We just glob.
lang_dump_path() {
  local lang="$1"
  local hits=("$WIKI_DUMP_ROOT/$lang"/wikipedia-*/"$lang.jsonl.gz")
  for h in "${hits[@]}"; do
    [ -f "$h" ] && { echo "$h"; return 0; }
  done
  return 1
}

# Per-language output directory
lang_out_dir() { echo "$OUT_DIR/$1"; }
