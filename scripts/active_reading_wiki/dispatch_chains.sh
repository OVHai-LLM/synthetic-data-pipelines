#!/usr/bin/env bash
# Dispatch N parallel chains of wiki_dp4.sbatch jobs, each chain handling
# a balanced subset of languages. Each chain is a sequence of 12 h jobs
# linked by --dependency=afterany so they run on the same node-class but
# do not need to be physically the same node.
#
# Within a chain, wiki_dp4.sbatch is fully resumable: it iterates the
# chain's LANGUAGES list and skips any batch_*.jsonl that is already at
# BATCH_SIZE lines.
#
# The chains were bin-packed greedy-largest-first across the per-lang
# cost (in GPU·hours) at full coverage — see the table below for the
# default 4-chain layout.
#
# Usage:
#   ./dispatch_chains.sh                 # default 4 chains, 27 jobs each
#   N_CHAINS=8 ./dispatch_chains.sh      # 8 chains, 14 jobs each
#   JOBS_PER_CHAIN=14 ./dispatch_chains.sh
#   MAX_ARTICLES_PER_LANG=20000 N_CHAINS=4 JOBS_PER_CHAIN=4 ./dispatch_chains.sh
#                                        # 20k cap on each lang × 4 chains × 4 jobs
set -euo pipefail

# ── repo bootstrap: locate the checkout (also from a SLURM spool copy) + site config ──
if [ -z "${REPO_ROOT:-}" ]; then
  REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null \
            || git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
fi
export REPO_ROOT
[ -f "$REPO_ROOT/configs/site.env" ] && : "${SITE_ENV:=$REPO_ROOT/configs/site.env}"
source "${SITE_ENV:-$REPO_ROOT/configs/site.env.example}"
# ─────────────────────────────────────────────────────────────────────────────


HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="$HERE/wiki_dp4.sbatch"
[ -f "$SBATCH_FILE" ] || { echo "ERROR: $SBATCH_FILE missing"; exit 1; }

# ─── default chain layout (4-way bin-pack at full coverage) ──────────
# To change the layout, override CHAINS_FILE pointing to a file with one
# whitespace-separated language list per line.
declare -a DEFAULT_CHAINS=(
  "it cs fi hr sk sq"
  "zh ca he tr lt ka lb"
  "uk ar no ro sl bs ga"
  "sv sr hu ko da et lv is"
)

# ─── overrides ───────────────────────────────────────────────────────
N_CHAINS="${N_CHAINS:-${#DEFAULT_CHAINS[@]}}"
JOBS_PER_CHAIN="${JOBS_PER_CHAIN:-27}"   # default for full-coverage 4-chain run
MAX_ARTICLES_PER_LANG="${MAX_ARTICLES_PER_LANG:-20000}"  # match wiki_dp4.sbatch default
CHAINS_FILE="${CHAINS_FILE:-}"

if [ -n "$CHAINS_FILE" ] && [ -f "$CHAINS_FILE" ]; then
  mapfile -t CHAINS < "$CHAINS_FILE"
else
  CHAINS=("${DEFAULT_CHAINS[@]}")
fi

if [ "$N_CHAINS" -ne "${#CHAINS[@]}" ]; then
  echo "WARN: N_CHAINS=$N_CHAINS but ${#CHAINS[@]} chains in layout — using ${#CHAINS[@]}"
  N_CHAINS="${#CHAINS[@]}"
fi

echo "# Dispatching $N_CHAINS chains × $JOBS_PER_CHAIN jobs each"
echo "# MAX_ARTICLES_PER_LANG=$MAX_ARTICLES_PER_LANG"
echo

# ─── submit ──────────────────────────────────────────────────────────
for ((c=0; c<N_CHAINS; c++)); do
  LANGS="${CHAINS[$c]}"
  echo "=== chain $c: $LANGS ==="
  prev=""
  for ((j=0; j<JOBS_PER_CHAIN; j++)); do
    if [ -z "$prev" ]; then
      JOBID=$(sbatch --parsable \
        --export=ALL,LANGUAGES="$LANGS",MAX_ARTICLES_PER_LANG="$MAX_ARTICLES_PER_LANG" \
        --job-name="wiki-c${c}-j${j}" \
        "$SBATCH_FILE")
    else
      JOBID=$(sbatch --parsable \
        --dependency=afterany:"$prev" \
        --export=ALL,LANGUAGES="$LANGS",MAX_ARTICLES_PER_LANG="$MAX_ARTICLES_PER_LANG" \
        --job-name="wiki-c${c}-j${j}" \
        "$SBATCH_FILE")
    fi
    echo "  c=$c j=$j → $JOBID"
    prev="$JOBID"
  done
done
echo
echo "# done. squeue -u $(whoami) | grep wiki-"
