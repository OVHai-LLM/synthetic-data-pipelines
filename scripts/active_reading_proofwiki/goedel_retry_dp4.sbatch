#!/bin/bash
#SBATCH --job-name=goedel-retry-dp4-async
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:gh200:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=600G
#SBATCH --time=12:00:00
#SBATCH --output=logs/goedel_retry_%j.out
#SBATCH --error=logs/goedel_retry_%j.err

# Goedel-Prover-V2-32B FP8 + DeepGEMM, DP=4 across 4 GH200, async client.
#
# Measured single-GH200 throughput (FP8 + KV-FP8 + DeepGEMM): ~3,350 tok/sec aggregate
# Expected DP=4 aggregate: ~13,000 tok/sec (linear scaling — no inter-GPU comm)
#
# Workload: ~102 K failed Lean snippets, avg 6 K output tokens per Goedel CoT response
#   round-0 only: 612 M tokens → ~13 h on DP=4 (over 12 h limit; --max-snippets 90000 fits)
#   2 rounds avg: 1.22 B tokens → ~26 h (need 2 sbatch submits)
#
# Architecture:
#   * 4 vLLM servers (DP=4), each on its own GPU at ports 8002-8005
#   * 4 goedel_async_retry processes (sharded), each pinned to one endpoint
#   * Each shard: asyncio.Semaphore(256) → ~150 concurrent in-flight per vLLM
#   * vLLM continuous batching saturates KV pool automatically
#
# Submit:  sbatch goedel_retry_dp4.sbatch

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
RUNS_INPUT=${ANNEAL_ROOT}/proofwiki/active_reading.jsonl
CHECK_INPUT=${ANNEAL_ROOT}/proofwiki/active_reading.lean_check.jsonl
OUTPUT_BASE=${ANNEAL_ROOT}/proofwiki/active_reading.goedel_retry
LOG_DIR=${LOG_DIR}
mkdir -p "$LOG_DIR"

# ─────────────────────────────────────────────────────────────────
# Tunables (override via env vars when sbatching)
# ─────────────────────────────────────────────────────────────────
MODEL="${GOEDEL_MODEL:-Goedel-LM/Goedel-Prover-V2-32B}"
MAX_LEN="${GOEDEL_MAX_LEN:-40960}"
MAX_SEQS="${GOEDEL_MAX_SEQS:-256}"
MAX_BATCHED_TOKENS="${GOEDEL_MAX_BATCHED_TOKENS:-65536}"
GPU_MEM_UTIL="${GOEDEL_GPU_MEM_UTIL:-0.92}"
CONCURRENCY_PER_SHARD="${GOEDEL_CONCURRENCY:-256}"
MAX_ROUNDS="${GOEDEL_MAX_ROUNDS:-2}"
MAX_TOKENS_PER_REQ="${GOEDEL_MAX_TOKENS:-32768}"
RESUME="${GOEDEL_RESUME:-0}"   # 1 = append/skip-already-done; useful after sbatch timeout

# 2-level sharding so multiple sbatch jobs can run in parallel:
#   JOB_SHARD_ID = which sbatch job we are (0..JOB_NUM_SHARDS-1)
#   JOB_NUM_SHARDS = total # parallel sbatch jobs
# Within each job, the 4 GPUs further shard via the per-shard SHARD index.
# Effective sub-shard index for goedel_async_retry: JOB_SHARD_ID * 4 + SHARD,
# out of total JOB_NUM_SHARDS * 4 sub-shards.
JOB_SHARD_ID="${GOEDEL_JOB_SHARD_ID:-0}"
JOB_NUM_SHARDS="${GOEDEL_JOB_NUM_SHARDS:-1}"
TOTAL_SUB_SHARDS=$(( JOB_NUM_SHARDS * 4 ))
MAX_SNIPPETS="${GOEDEL_MAX_SNIPPETS:-0}"   # 0 = all
SKIP_LEAN="${GOEDEL_SKIP_LEAN:-0}"          # 1 = no Lean compile (round 0 only)
LEAN_PROJECT="${AR_LEAN_PROJECT:-${LEAN_TOOLCHAIN}/ar_lean_check}"
LEAN_LAKE="${AR_LEAN_LAKE_BIN:-${LEAN_TOOLCHAIN}/.elan/bin/lake}"

# Lean compile single-thread (parallelism is from many concurrent compiles)
export LEAN_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export AR_LEAN_PROJECT="$LEAN_PROJECT"
export AR_LEAN_LAKE_BIN="$LEAN_LAKE"

# ─────────────────────────────────────────────────────────────────
# Modules + venv
# ─────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Loading modules"
# Minimum: Stages/2026 + Python + CUDA/13 (vLLM wheel ships cuDNN/NCCL but
# DeepGEMM needs CUDA_HOME from CUDA/13). Adding nvidia-compilers triggered
# a GCCcore reload + BLIS deactivation that (combined with set -e) killed
# the previous run on the compute node.
module load Stages/2026
module load Python
module load CUDA/13

source ${VLLM_ENV}/bin/activate
unset PYTHONPATH   # prevent Python-bundle-PyPI shadowing

# Compute nodes have no internet — force HF into local-cache-only mode.
# Without this, vLLM hangs validating the repo on hf.co and the script
# eventually exits with "Cannot send a request, as the client has been closed".
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# $HOME/.cache fills the user GPFS quota — vLLM's torch.compile cache (~3 GB
# per model) blew it. Redirect everything cache-y to project space.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torch_inductor_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton_cache}"
export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-${CACHE_ROOT}/flashinfer_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg_cache}"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
         "$FLASHINFER_CACHE_DIR" "$XDG_CACHE_HOME"

cd "${SRC_DIR}"
echo "[$(date +%H:%M:%S)] Config:"
echo "  MODEL=$MODEL  MAX_LEN=$MAX_LEN  MAX_SEQS=$MAX_SEQS"
echo "  CONCURRENCY=$CONCURRENCY_PER_SHARD  MAX_ROUNDS=$MAX_ROUNDS  SKIP_LEAN=$SKIP_LEAN"
echo "  JOB_SHARD: $JOB_SHARD_ID / $JOB_NUM_SHARDS  (sub-shards in this job: $((JOB_SHARD_ID*4))..$((JOB_SHARD_ID*4+3)) of $TOTAL_SUB_SHARDS)"

# ─────────────────────────────────────────────────────────────────
# 1. Launch 4 vLLM servers (DP=4), one per GPU
# ─────────────────────────────────────────────────────────────────
VLLM_PIDS=()
for GPU in 0 1 2 3; do
  PORT=$((8002 + GPU))
  CUDA_VISIBLE_DEVICES=$GPU python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype bfloat16 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --enable-prefix-caching \
    --served-model-name "$MODEL" \
    > "$LOG_DIR/vllm_gpu${GPU}_${SLURM_JOB_ID:-local}.log" 2>&1 &
  VLLM_PIDS+=($!)
  echo "  GPU $GPU → vLLM PID $! on port $PORT"
done

# ─────────────────────────────────────────────────────────────────
# 2. Wait for all 4 vLLM /health
# ─────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Waiting for 4 vLLM endpoints (32B FP8 takes ~3-5 min each)..."
for GPU in 0 1 2 3; do
  PORT=$((8002 + GPU))
  for i in $(seq 1 90); do
    if curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; then
      echo "  port $PORT ready (after $((i*10))s)"
      break
    fi
    sleep 10
    if [ "$i" -eq 90 ]; then
      echo "  ERROR: port $PORT not ready after 900s — see $LOG_DIR/vllm_gpu${GPU}_*.log"
      kill -9 "${VLLM_PIDS[@]}" 2>/dev/null || true
      exit 1
    fi
  done
done
echo "[$(date +%H:%M:%S)] All 4 endpoints up. Starting retry shards."

# ─────────────────────────────────────────────────────────────────
# 3. Launch 4 sharded async retry processes
# ─────────────────────────────────────────────────────────────────
SKIP_FLAG=""
[ "$SKIP_LEAN" = "1" ] && SKIP_FLAG="--skip-lean-check"
MAX_FLAG=""
[ "$MAX_SNIPPETS" -gt 0 ] && MAX_FLAG="--max-snippets $MAX_SNIPPETS"
RESUME_FLAG=""
[ "$RESUME" = "1" ] && RESUME_FLAG="--resume"

RETRY_PIDS=()
for SHARD in 0 1 2 3; do
  PORT=$((8002 + SHARD))
  SUB_SHARD_ID=$(( JOB_SHARD_ID * 4 + SHARD ))
  python3 -m math_lean.active_reading.goedel_async_retry \
    --runs-input  "$RUNS_INPUT" \
    --check-input "$CHECK_INPUT" \
    --output      "${OUTPUT_BASE}.job${JOB_SHARD_ID}of${JOB_NUM_SHARDS}.shard${SHARD}.jsonl" \
    --endpoint    "http://127.0.0.1:$PORT" \
    --model       "$MODEL" \
    --shard-id    "$SUB_SHARD_ID" \
    --num-shards  "$TOTAL_SUB_SHARDS" \
    --max-rounds  "$MAX_ROUNDS" \
    --concurrency "$CONCURRENCY_PER_SHARD" \
    --max-tokens  "$MAX_TOKENS_PER_REQ" \
    --temperature 1.0 \
    --top-p       0.95 \
    $SKIP_FLAG $MAX_FLAG $RESUME_FLAG \
    > "$LOG_DIR/retry_job${JOB_SHARD_ID}_shard${SHARD}_${SLURM_JOB_ID:-local}.log" 2>&1 &
  RETRY_PIDS+=($!)
  echo "  shard $SHARD (sub-shard $SUB_SHARD_ID/$TOTAL_SUB_SHARDS) → retry PID $! → vLLM port $PORT"
done

# ─────────────────────────────────────────────────────────────────
# 4. Wait for retry shards
# ─────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Waiting for 4 retry shards..."
wait "${RETRY_PIDS[@]}"
echo "[$(date +%H:%M:%S)] All shards done."

# ─────────────────────────────────────────────────────────────────
# 5. Merge + summary
# ─────────────────────────────────────────────────────────────────
# Per-job merge (only this job's 4 shards). Cross-job merge happens separately
# after all parallel jobs finish — see scripts/merge_goedel_outputs.sh below.
cat "${OUTPUT_BASE}.job${JOB_SHARD_ID}of${JOB_NUM_SHARDS}.shard"{0,1,2,3}".jsonl" \
    > "${OUTPUT_BASE}.job${JOB_SHARD_ID}of${JOB_NUM_SHARDS}.all.jsonl"
TOTAL=$(wc -l < "${OUTPUT_BASE}.job${JOB_SHARD_ID}of${JOB_NUM_SHARDS}.all.jsonl")
echo "[$(date +%H:%M:%S)] Job ${JOB_SHARD_ID}/${JOB_NUM_SHARDS} merged ($TOTAL records)"

python3 -c "
import json, collections
c = collections.Counter()
imp = err = 0
with open('${OUTPUT_BASE}.job${JOB_SHARD_ID}of${JOB_NUM_SHARDS}.all.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if 'error' in d: err += 1; continue
        c[d.get('final_verdict')] += 1
        if d.get('improved'): imp += 1
total = sum(c.values())
print(f'Job ${JOB_SHARD_ID} tally ({total} processed, {err} errored):')
for v, n in c.most_common():
    print(f'  {v}: {n} ({100*n/max(total,1):.1f}%)')
print(f'  improved: {imp} ({100*imp/max(total,1):.1f}%)')
"

# ─────────────────────────────────────────────────────────────────
# 6. Stop vLLM
# ─────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Stopping vLLM"
kill "${VLLM_PIDS[@]}" 2>/dev/null || true
sleep 5
kill -9 "${VLLM_PIDS[@]}" 2>/dev/null || true
echo "[$(date +%H:%M:%S)] DONE"
