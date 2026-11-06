#!/bin/bash
# Submit one enrich job per language (30 jobs).
# Each job is 12h on 1 node (4 GH200) and reads its own independent seed file
# from $SEED_DIR (built by generate_30_seeds.sbatch).
#
# Usage:
#   bash submit_per_language_enrich.sh
#
# Override the language list:
#   ENRICH_LANGUAGES_OVERRIDE="fr de es" bash submit_per_language_enrich.sh
#
# Quick subset (e.g. test 3 langs with N=10000):
#   ENRICH_N=10000 ENRICH_LANGUAGES_OVERRIDE="fr de zh" bash submit_per_language_enrich.sh
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH="${SCRIPT_DIR}/enrich_per_lang.sbatch"
SEED_DIR="${SEED_DIR:-${ANNEAL_ROOT}/data/maths/math_enrich_seeds}"

# ── Pull authoritative language list straight from enrich.py ───────
DEFAULT_LANGS=$(python3 -c "
import re
with open('${SRC_DIR}/math_generator/enrich.py') as f:
    src = f.read()
m = re.search(r'LANGUAGES\s*=\s*\{([^}]+)\}', src, re.DOTALL)
keys = re.findall(r'\"([a-z][a-z]?)\"\s*:', m.group(1))
print(' '.join(keys))
")

LANGS_STR="${ENRICH_LANGUAGES_OVERRIDE:-$DEFAULT_LANGS}"
read -ra LANGS <<< "$LANGS_STR"
TOTAL=${#LANGS[@]}

# ── Pre-flight: every language must have a seed file ───────────────
echo "Checking seed files in $SEED_DIR ..."
MISSING=()
for L in "${LANGS[@]}"; do
  F="${SEED_DIR}/seeds_${L}.jsonl"
  [ -s "$F" ] || MISSING+=("$L")
done
if [ "${#MISSING[@]}" -gt 0 ]; then
  echo ""
  echo "ERROR: missing seed files for: ${MISSING[*]}"
  echo "  Run first:  sbatch $(dirname "$SBATCH")/generate_30_seeds.sbatch"
  exit 1
fi

echo "All seed files present. Submitting $TOTAL per-language enrich jobs:"
printf '  langs: %s\n' "${LANGS[*]}"

JOB_IDS=()
for L in "${LANGS[@]}"; do
  JID=$(sbatch \
    --job-name="enrich-${L}" \
    --export=ALL,ENRICH_LANG=${L} \
    --parsable \
    "$SBATCH")
  printf '  %-3s → %s\n' "$L" "$JID"
  JOB_IDS+=("$JID")
done

echo ""
echo "All $TOTAL jobs submitted. Monitor:"
echo "  squeue --me -o '%.10i %.20j %.2t %.10M %.10l %R'"
echo ""
echo "Per-language outputs land in:"
echo "  ${ANNEAL_ROOT}/data/maths/math_enrich/enrich_<lang>.all.jsonl"
echo ""
echo "After all jobs finish, merge across languages:"
echo "  cat ${ANNEAL_ROOT}/data/maths/math_enrich/enrich_*.all.jsonl \\"
echo "      > ${ANNEAL_ROOT}/data/maths/math_enrich/enrich_multilang.final.jsonl"
