#!/bin/bash
#SBATCH --job-name=enrich-dp4-async
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:gh200:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=600G
#SBATCH --time=12:00:00
#SBATCH --output=logs/enrich_%j.out
#SBATCH --error=logs/enrich_%j.err

# Multilingual math-enrich on 4×GH200 with Gemma4-31B FP8 + DP=4 + async client.
#
# Architecture (mirrors goedel_retry_dp4.sbatch):
#   * 4 vLLM servers (DP=4), one per GPU at ports 8002-8005
#   * 4 enrich_async.py processes, sharded across the seed corpus
#   * Each shard: asyncio.Semaphore(512); vLLM continuous batching saturates
#   * vLLM tuned: max-num-seqs=512, max-num-batched-tokens=65536, prefix-caching ON
#     (the 8K default in vanilla `vllm serve` was the actual bottleneck, not GIL)
#
# 2-level sharding so multiple sbatch jobs run in parallel:
#   JOB_SHARD_ID = which sbatch job (0..JOB_NUM_SHARDS-1)
#   JOB_NUM_SHARDS = total parallel sbatch jobs
#   Effective sub-shard for enrich_async: JOB_SHARD_ID * 4 + SHARD
#   out of total JOB_NUM_SHARDS * 4 sub-shards.
#
# Submit one job:           sbatch enrich_dp4.sbatch
# Submit 4 in parallel:     bash submit_4parallel_enrich.sh

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
SEED_GLOB="${ENRICH_SEED_GLOB:-${ANNEAL_ROOT}/part_*/math_simple.jsonl}"
SEED_MERGED="${ENRICH_SEED_MERGED:-${SCRATCH_ROOT}/temp/enrich_seeds_merged.jsonl}"
OUTPUT_BASE="${ENRICH_OUTPUT_BASE:-${ANNEAL_ROOT}/data/maths/math_enrich/enrich_multilang}"
LOG_DIR=${LOG_DIR}
mkdir -p "$LOG_DIR" "$(dirname "$SEED_MERGED")" "$(dirname "$OUTPUT_BASE")"

# ─────────────────────────────────────────────────────────────────
# Tunables (override via env vars when sbatching)
# ─────────────────────────────────────────────────────────────────
MODEL="${ENRICH_MODEL:-RedHatAI/gemma-4-31B-it-FP8-Dynamic}"
MAX_LEN="${ENRICH_MAX_LEN:-32768}"
MAX_SEQS="${ENRICH_MAX_SEQS:-512}"
MAX_BATCHED_TOKENS="${ENRICH_MAX_BATCHED_TOKENS:-65536}"
GPU_MEM_UTIL="${ENRICH_GPU_MEM_UTIL:-0.92}"
CONCURRENCY_PER_SHARD="${ENRICH_CONCURRENCY:-512}"
MAX_TOKENS_PER_REQ="${ENRICH_MAX_TOKENS:-1500}"
TEMPERATURE="${ENRICH_TEMPERATURE:-0.8}"
LANGUAGES="${ENRICH_LANGUAGES:-en,fr,de,es,it,pt,nl,pl,ru,uk,cs,sv,da,fi,no,el,hu,ro,bg,zh,ja,ko,vi,th,id,ar,tr,he,hi,bn}"
PROMPTS="${ENRICH_PROMPTS:-all}"
N_RECORDS="${ENRICH_N:-0}"   # 0 = all in input
RESUME="${ENRICH_RESUME:-1}"  # default to resume so re-submits don't re-do

JOB_SHARD_ID="${ENRICH_JOB_SHARD_ID:-0}"
JOB_NUM_SHARDS="${ENRICH_JOB_NUM_SHARDS:-1}"
TOTAL_SUB_SHARDS=$(( JOB_NUM_SHARDS * 4 ))

# ─────────────────────────────────────────────────────────────────
# Modules + venv
# ─────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Loading modules"
# Stages/2026 must load alone first (lmod silently no-ops the rest otherwise)
module load Stages/2026
module load nvidia-compilers/25.9-CUDA-13 Python NCCL/default-CUDA-13 \
            CUDA/13 CUDA-Python/13 cuDNN/9.19.0.56-CUDA-13

source ${VLLM_ENV}/bin/activate
unset PYTHONPATH

cd "${SRC_DIR}"
echo "[$(date +%H:%M:%S)] Config:"
echo "  MODEL=$MODEL  MAX_LEN=$MAX_LEN  MAX_SEQS=$MAX_SEQS  MAX_BATCHED_TOKENS=$MAX_BATCHED_TOKENS"
echo "  CONCURRENCY=$CONCURRENCY_PER_SHARD  RESUME=$RESUME  N=$N_RECORDS"
echo "  LANGUAGES=$LANGUAGES"
echo "  JOB_SHARD: $JOB_SHARD_ID / $JOB_NUM_SHARDS  (sub-shards $((JOB_SHARD_ID*4))..$((JOB_SHARD_ID*4+3)) of $TOTAL_SUB_SHARDS)"

# ─────────────────────────────────────────────────────────────────
# 0. Build merged seed file once (job 0 does it; others wait).
#    Cheap relative to GPU work and avoids each shard re-globbing.
# ─────────────────────────────────────────────────────────────────
if [ "$JOB_SHARD_ID" = "0" ] && [ ! -s "$SEED_MERGED" ]; then
  echo "[$(date +%H:%M:%S)] Building merged seed file: $SEED_MERGED"
  cat $SEED_GLOB > "$SEED_MERGED"
  echo "  → $(wc -l < "$SEED_MERGED") records"
elif [ ! -s "$SEED_MERGED" ]; then
  echo "[$(date +%H:%M:%S)] Waiting for $SEED_MERGED (job 0 building it)..."
  for i in $(seq 1 60); do
    [ -s "$SEED_MERGED" ] && break
    sleep 10
  done
fi

# ─────────────────────────────────────────────────────────────────
# 1. Launch 4 vLLM servers (DP=4), one per GPU
# ─────────────────────────────────────────────────────────────────
VLLM_PIDS=()
for GPU in 0 1 2 3; do
  PORT=$((8002 + GPU))
  CUDA_VISIBLE_DEVICES=$GPU vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --reasoning-parser gemma4 \
    --tool-call-parser gemma4 \
    --enable-auto-tool-choice \
    --limit-mm-per-prompt '{"image":0,"audio":0}' \
    --served-model-name "$MODEL" \
    > "$LOG_DIR/vllm_enrich_gpu${GPU}_${SLURM_JOB_ID:-local}.log" 2>&1 &
  VLLM_PIDS+=($!)
  echo "  GPU $GPU → vLLM PID $! on port $PORT"
done

# ─────────────────────────────────────────────────────────────────
# 2. Wait for all 4 vLLM /health (Gemma4-31B FP8 ~70 s warmup)
# ─────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Waiting for 4 vLLM endpoints..."
for GPU in 0 1 2 3; do
  PORT=$((8002 + GPU))
  for i in $(seq 1 60); do
    if curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; then
      echo "  port $PORT ready (after $((i*10))s)"
      break
    fi
    sleep 10
    if [ "$i" -eq 60 ]; then
      echo "  ERROR: port $PORT not ready after 600s — see $LOG_DIR/vllm_enrich_gpu${GPU}_*.log"
      kill -9 "${VLLM_PIDS[@]}" 2>/dev/null || true
      exit 1
    fi
  done
done
echo "[$(date +%H:%M:%S)] All 4 endpoints up. Starting enrich shards."

# ─────────────────────────────────────────────────────────────────
# 3. Launch 4 sharded enrich_async processes
# ─────────────────────────────────────────────────────────────────
RESUME_FLAG=""
[ "$RESUME" = "1" ] && RESUME_FLAG="--resume"
N_FLAG=""
[ "$N_RECORDS" -gt 0 ] && N_FLAG="--n $N_RECORDS"

ENRICH_PIDS=()
for SHARD in 0 1 2 3; do
  PORT=$((8002 + SHARD))
  SUB_SHARD_ID=$(( JOB_SHARD_ID * 4 + SHARD ))
  python3 -m math_generator.enrich_async \
    --input       "$SEED_MERGED" \
    --output      "${OUTPUT_BASE}.job${JOB_SHARD_ID}of${JOB_NUM_SHARDS}.shard${SHARD}.jsonl" \
    --endpoint    "http://127.0.0.1:$PORT" \
    --model       "$MODEL" \
    --languages   "$LANGUAGES" \
    --prompts     "$PROMPTS" \
    --shard-id    "$SUB_SHARD_ID" \
    --num-shards  "$TOTAL_SUB_SHARDS" \
    --concurrency "$CONCURRENCY_PER_SHARD" \
    --max-tokens  "$MAX_TOKENS_PER_REQ" \
    --temperature "$TEMPERATURE" \
    --seed        "$SUB_SHARD_ID" \
    $RESUME_FLAG $N_FLAG \
    > "$LOG_DIR/enrich_job${JOB_SHARD_ID}_shard${SHARD}_${SLURM_JOB_ID:-local}.log" 2>&1 &
  ENRICH_PIDS+=($!)
  echo "  shard $SHARD (sub-shard $SUB_SHARD_ID/$TOTAL_SUB_SHARDS) → enrich PID $! → vLLM port $PORT"
done

# ─────────────────────────────────────────────────────────────────
# 4. Wait for enrich shards
# ─────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Waiting for 4 enrich shards..."
wait "${ENRICH_PIDS[@]}"
echo "[$(date +%H:%M:%S)] All shards done."

# ─────────────────────────────────────────────────────────────────
# 5. Per-job merge + summary (judge keep-rate, language distribution)
# ─────────────────────────────────────────────────────────────────
cat "${OUTPUT_BASE}.job${JOB_SHARD_ID}of${JOB_NUM_SHARDS}.shard"{0,1,2,3}".jsonl" \
    > "${OUTPUT_BASE}.job${JOB_SHARD_ID}of${JOB_NUM_SHARDS}.all.jsonl"
TOTAL=$(wc -l < "${OUTPUT_BASE}.job${JOB_SHARD_ID}of${JOB_NUM_SHARDS}.all.jsonl")
echo "[$(date +%H:%M:%S)] Job ${JOB_SHARD_ID}/${JOB_NUM_SHARDS} merged ($TOTAL records)"

python3 -c "
import json, collections
ALL_PATH = '${OUTPUT_BASE}.job${JOB_SHARD_ID}of${JOB_NUM_SHARDS}.all.jsonl'
keep = drop = err = 0
lang = collections.Counter()
prompts = collections.Counter()
reasons = collections.Counter()
with open(ALL_PATH) as f:
    for line in f:
        d = json.loads(line)
        if d.get('error'):
            err += 1; continue
        lang[d.get('rewrite_language','?')] += 1
        prompts[d.get('rewrite_prompt_name','?')] += 1
        if d.get('judge', {}).get('overall_keep'):
            keep += 1
        else:
            drop += 1
            reasons[d.get('judge', {}).get('reason','?')[:40]] += 1
total = keep + drop
print(f'Job ${JOB_SHARD_ID} stats: {total} processed, {err} errored')
print(f'  keep: {keep} ({100*keep/max(total,1):.1f}%)  drop: {drop}')
print(f'  per-language:')
for c, n in lang.most_common():
    print(f'    {c}: {n}')
print(f'  per-prompt-style:')
for p, n in prompts.most_common():
    print(f'    {p}: {n}')
if reasons:
    print(f'  top drop reasons:')
    for r, n in reasons.most_common(5):
        print(f'    {r}: {n}')
"

# ─────────────────────────────────────────────────────────────────
# 6. Stop vLLM
# ─────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Stopping vLLM"
kill "${VLLM_PIDS[@]}" 2>/dev/null || true
sleep 5
kill -9 "${VLLM_PIDS[@]}" 2>/dev/null || true
echo "[$(date +%H:%M:%S)] DONE"
