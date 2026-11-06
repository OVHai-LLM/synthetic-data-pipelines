#!/bin/bash
# Merge per-job outputs from N parallel sbatch jobs into one final JSONL.
# Each job produced  active_reading.goedel_retry.job{J}of{N}.all.jsonl
# Output:           active_reading.goedel_retry.all.jsonl
#
# Usage: bash merge_goedel_outputs.sh [NUM_JOBS]   (default 4)

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

NUM_JOBS="${1:-4}"
BASE=${ANNEAL_ROOT}/proofwiki/active_reading.goedel_retry

echo "Merging $NUM_JOBS per-job outputs..."
files=()
for J in $(seq 0 $((NUM_JOBS-1))); do
  f="${BASE}.job${J}of${NUM_JOBS}.all.jsonl"
  if [ ! -f "$f" ]; then
    echo "  WARNING: missing $f — was this job submitted/completed?"
    continue
  fi
  N=$(wc -l < "$f")
  echo "  job $J: $N records"
  files+=("$f")
done

cat "${files[@]}" > "${BASE}.all.jsonl"
TOTAL=$(wc -l < "${BASE}.all.jsonl")
echo ""
echo "Merged → ${BASE}.all.jsonl ($TOTAL total records)"

python3 -c "
import json, collections
c = collections.Counter()
imp = err = 0
with open('${BASE}.all.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if 'error' in d: err += 1; continue
        c[d.get('final_verdict')] += 1
        if d.get('improved'): imp += 1
total = sum(c.values())
print(f'Final tally ({total} processed, {err} errored):')
for v, n in c.most_common():
    print(f'  {v}: {n} ({100*n/max(total,1):.1f}%)')
print(f'  improved: {imp} ({100*imp/max(total,1):.1f}%)')
"
