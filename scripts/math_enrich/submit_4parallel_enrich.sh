#!/bin/bash
# Submit 4 parallel enrich_dp4 sbatch jobs.
# Result: 4 nodes × 4 GPUs = 16 vLLM servers + 16 sharded enrich processes
# Each shard processes 1/16th of the merged seed corpus.
#
# Override params via env: ENRICH_LANGUAGES=en,fr,de bash submit_4parallel_enrich.sh
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
SBATCH="${SCRIPT_DIR}/enrich_dp4.sbatch"

NJOBS="${NJOBS:-4}"
echo "Submitting $NJOBS parallel enrich jobs..."
JOB_IDS=()
for i in $(seq 0 $((NJOBS-1))); do
  JID=$(sbatch \
    --export=ALL,ENRICH_JOB_SHARD_ID=$i,ENRICH_JOB_NUM_SHARDS=$NJOBS \
    --parsable \
    "$SBATCH")
  echo "  job $i/$NJOBS  →  jobid $JID"
  JOB_IDS+=("$JID")
done
echo ""
echo "All submitted. Monitor with:"
echo "  squeue -j $(IFS=,; echo "${JOB_IDS[*]}")"
echo "Per-job logs:"
echo "  ${LOG_DIR}/enrich_<jobid>.out"
echo ""
echo "After all jobs finish, merge across jobs:"
echo "  cat ${ANNEAL_ROOT}/data/maths/math_enrich/enrich_multilang.job*of${NJOBS}.all.jsonl \\"
echo "      > ${ANNEAL_ROOT}/data/maths/math_enrich/enrich_multilang.final.jsonl"
