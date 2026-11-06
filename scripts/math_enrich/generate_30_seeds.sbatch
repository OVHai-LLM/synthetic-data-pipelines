#!/bin/bash
#SBATCH --job-name=enrich-seeds-gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --output=logs/enrich_seeds_gen_%j.out
#SBATCH --error=logs/enrich_seeds_gen_%j.err

# Generate 30 INDEPENDENT raw seed corpora — one per language — using
# math_generator's current registry. Each language gets its own random seed
# (base + lang_index) so the problem distributions are fully independent
# across languages (no two languages enrich the same source problem).
#
# Output: seeds_<lang>.jsonl  (raw shape: {id, problem, answer, meta, text})
#         in $SEED_DIR (default: data/maths/math_enrich_seeds/)
#
# Submit:  sbatch generate_30_seeds.sbatch
# Override scale: RECORDS_PER_LANG=2000000 sbatch ...

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


# ─────────────────────────────────────────────────────────────────
# Inputs / outputs
# ─────────────────────────────────────────────────────────────────
RECORDS_PER_LANG="${RECORDS_PER_LANG:-2000000}"
DIFFICULTY="${SEED_DIFFICULTY:-mix}"
BASE_SEED="${SEED_BASE_SEED:-20260506}"
SEED_DIR="${SEED_DIR:-${ANNEAL_ROOT}/data/maths/math_enrich_seeds}"
mkdir -p "$SEED_DIR"

# Authoritative language list straight from enrich.py
LANGS=$(python3 -c "
import re
with open('${SRC_DIR}/math_generator/enrich.py') as f:
    src = f.read()
m = re.search(r'LANGUAGES\s*=\s*\{([^}]+)\}', src, re.DOTALL)
keys = re.findall(r'\"([a-z][a-z]?)\"\s*:', m.group(1))
print(' '.join(keys))
")
read -ra LANG_ARR <<< "$LANGS"
TOTAL=${#LANG_ARR[@]}

echo "[$(date +%H:%M:%S)] Generating seeds: $TOTAL langs × $RECORDS_PER_LANG records"
echo "  Output dir: $SEED_DIR"
echo "  Difficulty mix: $DIFFICULTY,  base_seed=$BASE_SEED"

# ─────────────────────────────────────────────────────────────────
# Modules + venv
# ─────────────────────────────────────────────────────────────────
module load Stages/2026
module load Python

# vllm_env_gh200 has sympy (prepare_data_env doesn't); CPU-only generation here
source ${VLLM_ENV}/bin/activate
unset PYTHONPATH

cd "${SRC_DIR}"
# ─────────────────────────────────────────────────────────────────
# Spawn ONE math_generator process per language, all in parallel.
# Each gets a unique --seed → fully independent problem distributions.
# ─────────────────────────────────────────────────────────────────
PIDS=()
for i in "${!LANG_ARR[@]}"; do
  LANG="${LANG_ARR[$i]}"
  SEED=$((BASE_SEED + i))
  OUT="${SEED_DIR}/seeds_${LANG}.jsonl"
  if [ -s "$OUT" ] && [ "$(wc -l < "$OUT")" -ge "$RECORDS_PER_LANG" ]; then
    echo "  [skip] ${LANG} already has $(wc -l < "$OUT") records → $OUT"
    continue
  fi
  echo "  [start] ${LANG} (idx ${i}, seed ${SEED}) → ${OUT}"
  python3 -u -m math_generator \
    --n "$RECORDS_PER_LANG" \
    --out "$OUT" \
    --difficulty "$DIFFICULTY" \
    --seed "$SEED" \
    > "${SEED_DIR}/seeds_${LANG}.log" 2>&1 &
  PIDS+=($!)
done

# ─────────────────────────────────────────────────────────────────
# Wait for all generators (they each pin a single core; with 30 in
# parallel on a 288-core node we use ~10 % of cores — still ~10 min total)
# ─────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Waiting for ${#PIDS[@]} generators..."
FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAILED=$((FAILED+1))
  fi
done

# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] DONE  (${FAILED} failed)"
echo ""
printf '%-4s %12s %10s\n' lang records sizeMB
printf '%-4s %12s %10s\n' ---- ------- ------
for L in "${LANG_ARR[@]}"; do
  F="${SEED_DIR}/seeds_${L}.jsonl"
  if [ -s "$F" ]; then
    N=$(wc -l < "$F")
    SZ=$(du -m "$F" | cut -f1)
    printf '%-4s %12d %10s\n' "$L" "$N" "$SZ"
  else
    printf '%-4s %12s %10s\n' "$L" MISSING -
  fi
done
