#!/bin/bash
#SBATCH --job-name=enrich-lang
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:gh200:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=600G
#SBATCH --time=12:00:00
#SBATCH --output=logs/enrich_lang_%x_%j.out
#SBATCH --error=logs/enrich_lang_%x_%j.err

# Per-language multilingual math-enrich. ONE language, ONE independent seed file.
#
# Reads:  $SEED_DIR/seeds_<lang>.jsonl   (built by generate_30_seeds.sbatch)
# Writes: $OUTPUT_DIR/enrich_<lang>.shard{0..3}.jsonl  → merged to enrich_<lang>.all.jsonl
#
# Architecture:
#   * 4 vLLM servers (DP=4), one per GPU at ports 8002-8005, all loading Gemma4-31B FP8
#   * 4 enrich_async.py shards, each pinned to one GPU, each handling 1/4 of seeds
#   * vLLM tuned for GPU saturation: max-num-batched-tokens=65536, max-num-seqs=512
#
# Required env (set by submit_per_language_enrich.sh):
#   ENRICH_LANG          single language code, e.g. "fr"
#
# Submit one:    ENRICH_LANG=fr sbatch enrich_per_lang.sbatch
# Submit all 30: bash submit_per_language_enrich.sh

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


: "${ENRICH_LANG:?must set ENRICH_LANG (single language code, e.g. fr)}"

# ─────────────────────────────────────────────────────────────────
# Inputs / outputs
# ─────────────────────────────────────────────────────────────────
SEED_DIR="${SEED_DIR:-${ANNEAL_ROOT}/data/maths/math_enrich_seeds}"
SEED_FILE="${ENRICH_SEED_FILE:-${SEED_DIR}/seeds_${ENRICH_LANG}.jsonl}"
OUTPUT_DIR="${ENRICH_OUTPUT_DIR:-${ANNEAL_ROOT}/data/maths/math_enrich}"
OUTPUT_BASE="${OUTPUT_DIR}/enrich_${ENRICH_LANG}"
LOG_DIR=${LOG_DIR}
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [ ! -s "$SEED_FILE" ]; then
  echo "ERROR: seed file missing or empty: $SEED_FILE"
  echo "  Run generate_30_seeds.sbatch first."
  exit 1
fi

# ─────────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────────
MODEL="${ENRICH_MODEL:-RedHatAI/gemma-4-31B-it-FP8-Dynamic}"
MAX_LEN="${ENRICH_MAX_LEN:-32768}"
MAX_SEQS="${ENRICH_MAX_SEQS:-512}"
MAX_BATCHED_TOKENS="${ENRICH_MAX_BATCHED_TOKENS:-65536}"
GPU_MEM_UTIL="${ENRICH_GPU_MEM_UTIL:-0.92}"
CONCURRENCY_PER_SHARD="${ENRICH_CONCURRENCY:-512}"
MAX_TOKENS_PER_REQ="${ENRICH_MAX_TOKENS:-1500}"
TEMPERATURE="${ENRICH_TEMPERATURE:-0.8}"
PROMPTS="${ENRICH_PROMPTS:-all}"
N_RECORDS="${ENRICH_N:-0}"
RESUME="${ENRICH_RESUME:-1}"

# ─────────────────────────────────────────────────────────────────
# Modules + venv
# ─────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Loading modules"
# Minimum: Stages/2026 + Python + CUDA/13. DeepGEMM needs CUDA_HOME from
# CUDA/13; vLLM wheel ships cuDNN/NCCL.
module load Stages/2026
module load Python
module load CUDA/13

source ${VLLM_ENV}/bin/activate
unset PYTHONPATH

# Compute nodes have no internet — force HF into local-cache-only
# mode. Without this, vLLM hangs ~10 min on hf.co validation then exits with
# "Cannot send a request, as the client has been closed".
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# $HOME/.cache fills the user GPFS quota — redirect torch/triton/vllm caches
# to project space (otherwise vLLM dies with "Disk quota exceeded" on
# torch.compile cache writes).
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torch_inductor_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton_cache}"
export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-${CACHE_ROOT}/flashinfer_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg_cache}"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
         "$FLASHINFER_CACHE_DIR" "$XDG_CACHE_HOME"

cd "${SRC_DIR}"
echo "[$(date +%H:%M:%S)] Lang job: ${ENRICH_LANG}"
echo "  SEED_FILE=$SEED_FILE  ($(wc -l < "$SEED_FILE") records)"
echo "  MODEL=$MODEL  CONCURRENCY=$CONCURRENCY_PER_SHARD  RESUME=$RESUME  N=$N_RECORDS"

# ─────────────────────────────────────────────────────────────────
# 1. Launch 4 vLLM servers (DP=4)
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
    > "$LOG_DIR/vllm_enrich_${ENRICH_LANG}_gpu${GPU}_${SLURM_JOB_ID:-local}.log" 2>&1 &
  VLLM_PIDS+=($!)
  echo "  GPU $GPU → vLLM PID $! on port $PORT"
done

# ─────────────────────────────────────────────────────────────────
# 2. Wait for all 4 vLLM /health
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
      echo "  ERROR: port $PORT not ready after 600s"
      kill -9 "${VLLM_PIDS[@]}" 2>/dev/null || true
      exit 1
    fi
  done
done
echo "[$(date +%H:%M:%S)] All endpoints up. Starting 4 enrich shards for lang=${ENRICH_LANG}"

# ─────────────────────────────────────────────────────────────────
# 3. Launch 4 sharded enrich_async processes (single language each).
#    Each shard takes 1/4 of the per-lang seed file.
# ─────────────────────────────────────────────────────────────────
RESUME_FLAG=""
[ "$RESUME" = "1" ] && RESUME_FLAG="--resume"
N_FLAG=""
[ "$N_RECORDS" -gt 0 ] && N_FLAG="--n $N_RECORDS"

ENRICH_PIDS=()
for SHARD in 0 1 2 3; do
  PORT=$((8002 + SHARD))
  python3 -m math_generator.enrich_async \
    --input       "$SEED_FILE" \
    --output      "${OUTPUT_BASE}.shard${SHARD}.jsonl" \
    --endpoint    "http://127.0.0.1:$PORT" \
    --model       "$MODEL" \
    --languages   "$ENRICH_LANG" \
    --prompts     "$PROMPTS" \
    --shard-id    "$SHARD" \
    --num-shards  4 \
    --concurrency "$CONCURRENCY_PER_SHARD" \
    --max-tokens  "$MAX_TOKENS_PER_REQ" \
    --temperature "$TEMPERATURE" \
    --seed        "$SHARD" \
    $RESUME_FLAG $N_FLAG \
    > "$LOG_DIR/enrich_${ENRICH_LANG}_shard${SHARD}_${SLURM_JOB_ID:-local}.log" 2>&1 &
  ENRICH_PIDS+=($!)
  echo "  shard $SHARD → enrich PID $! → vLLM port $PORT"
done

# ─────────────────────────────────────────────────────────────────
# 4. Wait
# ─────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Waiting for 4 enrich shards (lang=${ENRICH_LANG})..."
wait "${ENRICH_PIDS[@]}"
echo "[$(date +%H:%M:%S)] All shards done."

# ─────────────────────────────────────────────────────────────────
# 5. Per-language merge + summary
# ─────────────────────────────────────────────────────────────────
cat "${OUTPUT_BASE}.shard"{0,1,2,3}".jsonl" > "${OUTPUT_BASE}.all.jsonl"
TOTAL=$(wc -l < "${OUTPUT_BASE}.all.jsonl")
echo "[$(date +%H:%M:%S)] Lang ${ENRICH_LANG} merged ($TOTAL records)"

python3 -c "
import json, collections
ALL_PATH = '${OUTPUT_BASE}.all.jsonl'
keep = drop = err = 0
prompts = collections.Counter()
reasons = collections.Counter()
with open(ALL_PATH) as f:
    for line in f:
        d = json.loads(line)
        if d.get('error'):
            err += 1; continue
        prompts[d.get('rewrite_prompt_name','?')] += 1
        if d.get('judge', {}).get('overall_keep'):
            keep += 1
        else:
            drop += 1
            reasons[d.get('judge', {}).get('reason','?')[:40]] += 1
total = keep + drop
print(f'Lang ${ENRICH_LANG}: {total} processed, {err} errored')
print(f'  keep: {keep} ({100*keep/max(total,1):.1f}%)  drop: {drop}')
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
echo "[$(date +%H:%M:%S)] DONE lang=${ENRICH_LANG}"
