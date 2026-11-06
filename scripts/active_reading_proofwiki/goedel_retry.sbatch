#!/bin/bash
#SBATCH --job-name=goedel-retry-dp4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:gh200:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=600G
#SBATCH --time=24:00:00
#SBATCH --output=logs/goedel_retry_%j.out
#SBATCH --error=logs/goedel_retry_%j.err

# Goedel-Prover-V2 self-correction over 102k failed AR snippets,
# data-parallel across 4 GH200 GPUs on one node.
#
# Architecture:
#   * 4 vLLM instances (one per GPU, ports 8002-8005), each holding the
#     full 32B model in bf16. With KV-cache budget ~400 GB per instance,
#     each can keep ~256 concurrent generations in-flight.
#   * 4 goedel_retry processes (sharded), each pinned to one vLLM endpoint
#     with 128 worker threads (so steady-state ~64 vLLM in-flight + ~64
#     Lean compiles per shard = 256 in-flight + 256 lean = ~512 OS threads
#     and ~256 Lean processes — under the 288-core budget).
#
# Submit with:
#   sbatch ${REPO_ROOT}/scripts/active_reading_proofwiki/goedel_retry.sbatch

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
# Tunables
# ─────────────────────────────────────────────────────────────────
MODEL="${GOEDEL_MODEL:-Goedel-LM/Goedel-Prover-V2-32B}"
MAX_LEN="${GOEDEL_MAX_LEN:-40960}"
MAX_SEQS="${GOEDEL_MAX_SEQS:-256}"
WORKERS_PER_SHARD="${GOEDEL_WORKERS:-128}"
MAX_ROUNDS="${GOEDEL_MAX_ROUNDS:-2}"

# Lean compile is single-threaded per invocation (the worker is the parallelism)
export LEAN_NUM_THREADS=1
# OpenMP / blas threading off — we already saturate via worker concurrency
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Tokenizers internal parallelism off — we get parallelism from the workers
export TOKENIZERS_PARALLELISM=false

# ─────────────────────────────────────────────────────────────────
# Modules + venv
# ─────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Loading modules"
module load Stages/2026
module load Python
module load CUDA-Python/13
module load cuDNN/9.19.0.56-CUDA-13
module load NCCL/default-CUDA-13

source ${VLLM_ENV}/bin/activate
# Critical: Python-bundle-PyPI shadows venv otherwise (regex etc.)
unset PYTHONPATH

# HF token if Goedel needs auth (it's Apache-2.0 but harmless to load)
source ~/.bashrc 2>/dev/null || true

cd "${SRC_DIR}"
# ─────────────────────────────────────────────────────────────────
# 1. Launch 4 vLLM servers, one per GPU
# ─────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Launching 4 vLLM instances of $MODEL"
for GPU in 0 1 2 3; do
  PORT=$((8002 + GPU))
  CUDA_VISIBLE_DEVICES=$GPU python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.92 \
    --dtype bfloat16 \
    --trust-remote-code \
    --enable-prefix-caching \
    --served-model-name "$MODEL" \
    > "$LOG_DIR/vllm_gpu${GPU}_${SLURM_JOB_ID:-local}.log" 2>&1 &
  echo "  GPU $GPU → vLLM PID $! on port $PORT"
done

# ─────────────────────────────────────────────────────────────────
# 2. Wait for all 4 vLLM instances to come up (poll /health)
# ─────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Waiting for vLLM servers to be ready (loading 32B model takes ~3-5 min each)..."
for GPU in 0 1 2 3; do
  PORT=$((8002 + GPU))
  for i in $(seq 1 60); do
    if curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; then
      echo "  port $PORT ready (after ${i}0s)"
      break
    fi
    sleep 10
    if [ "$i" -eq 60 ]; then
      echo "  ERROR: port $PORT not ready after 600s — check $LOG_DIR/vllm_gpu${GPU}_*.log"
      exit 1
    fi
  done
done
echo "[$(date +%H:%M:%S)] All 4 vLLM instances ready."

# ─────────────────────────────────────────────────────────────────
# 3. Launch 4 sharded goedel_retry workers (one per vLLM endpoint)
# ─────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Launching 4 retry shards × $WORKERS_PER_SHARD workers"
for SHARD in 0 1 2 3; do
  PORT=$((8002 + SHARD))
  python3 -m math_lean.active_reading.goedel_retry \
    --runs-input  "$RUNS_INPUT" \
    --check-input "$CHECK_INPUT" \
    --output      "${OUTPUT_BASE}.shard${SHARD}.jsonl" \
    --endpoint    "http://127.0.0.1:$PORT" \
    --model       "$MODEL" \
    --shard-id    "$SHARD" \
    --num-shards  4 \
    --max-rounds  "$MAX_ROUNDS" \
    --workers     "$WORKERS_PER_SHARD" \
    --temperature 1.0 \
    --top-p       0.95 \
    --max-tokens  32768 \
    > "$LOG_DIR/retry_shard${SHARD}_${SLURM_JOB_ID:-local}.log" 2>&1 &
  echo "  shard $SHARD → retry PID $! on port $PORT"
done

# ─────────────────────────────────────────────────────────────────
# 4. Wait for all 4 retry shards to finish
# ─────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Waiting for all 4 retry shards..."
RETRY_PIDS=$(jobs -p | tail -4)
wait $RETRY_PIDS
echo "[$(date +%H:%M:%S)] All retry shards done."

# ─────────────────────────────────────────────────────────────────
# 5. Concatenate shards → one output JSONL + summary
# ─────────────────────────────────────────────────────────────────
cat "${OUTPUT_BASE}.shard"{0,1,2,3}".jsonl" > "${OUTPUT_BASE}.all.jsonl"
TOTAL=$(wc -l < "${OUTPUT_BASE}.all.jsonl")
echo "[$(date +%H:%M:%S)] Merged ${OUTPUT_BASE}.all.jsonl ($TOTAL records)"

# Quick verdict tally
python3 -c "
import json, collections
c = collections.Counter()
imp = 0
with open('${OUTPUT_BASE}.all.jsonl') as f:
    for line in f:
        d = json.loads(line)
        c[d.get('final_verdict')] += 1
        if d.get('improved'): imp += 1
total = sum(c.values())
print(f'Final verdict tally ({total} records):')
for v, n in c.most_common():
    print(f'  {v}: {n} ({100*n/total:.1f}%)')
print(f'  improved (was-fail → ok/sorry): {imp} ({100*imp/total:.1f}%)')
"

# ─────────────────────────────────────────────────────────────────
# 6. Cleanup vLLM servers (background processes)
# ─────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Stopping vLLM servers"
pkill -P $$ -f vllm.entrypoints.openai.api_server || true

echo "[$(date +%H:%M:%S)] DONE"
