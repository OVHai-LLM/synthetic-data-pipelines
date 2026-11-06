#!/bin/bash
#SBATCH --job-name=wiki-ar-dp4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:gh200:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=600G
#SBATCH --time=12:00:00
#SBATCH --output=logs/wiki_dp4_%j.out
#SBATCH --error=logs/wiki_dp4_%j.err

# Active-Reading-on-Wikipedia, DP=4 across 4 GH200, single client process
# using prefix-cache-aware pinning (one rank per article).
#
# Each vLLM serves on its own port (8002-8005). The wiki_main client
# receives the comma-joined endpoint list and uses client.py:VLLMClient
# (multi-endpoint mode) + pipeline.py's pin_key=record["id"] so all 8
# calls per article (2 strategy + 6 apply) land on the same vLLM rank.
# That gives full prefix-cache reuse of the article body.
#
# Per language: ~20 K articles × 8 calls × ~1.2 K output tok ≈ 192 M tok.
# At ~1.4 K output tok/s/GPU → ~190 s/article on DP=4 ≈ 38 K s/lang ≈ 10.5 h/lang.
# Realistically: 1-3 langs per 24 h job; chain submits with afterany.
#
# Resume: per-batch outputs at $LANG_OUT/batch_N.jsonl skip if already
# at full BATCH_SIZE → safe to re-submit after timeout.
#
# Submit defaults (process all langs in $LANGUAGES sequentially):
#   sbatch wiki_dp4.sbatch
#
# Submit with override (just a few langs, smaller cap):
#   LANGUAGES="ga lb" MAX_ARTICLES_PER_LANG=2000 sbatch wiki_dp4.sbatch

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
# Modules + venv
# ─────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Loading modules"
module load Stages/2026
module load Python
module load CUDA/13
module load CUDA-Python/13
module load cuDNN/9.19.0.56-CUDA-13
module load NCCL/default-CUDA-13
module load nvidia-compilers/25.9-CUDA-13

source ${VLLM_ENV}/bin/activate
unset PYTHONPATH

# Compute nodes have no internet — force HF into local-cache-only mode.
# Model must already be in $HF_HOME/hub or vLLM hangs trying to phone home.
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
# ─────────────────────────────────────────────────────────────────
# Configuration (override via env when sbatching)
# ─────────────────────────────────────────────────────────────────
MODEL="${AR_MODEL:-RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic}"  # MoE 4B-active: 2.5× faster than 31B dense, same quality on low-res langs
MAX_LEN="${AR_MAX_LEN:-16384}"
MAX_SEQS="${AR_MAX_SEQS:-256}"
MAX_BATCHED_TOKENS="${AR_MAX_BATCHED_TOKENS:-16384}"
GPU_MEM_UTIL="${AR_GPU_MEM_UTIL:-0.92}"

# Workload knobs (mirror config_wiki.sh; can be overridden)
WIKI_DUMP_ROOT="${WIKI_DUMP_ROOT:-${ANNEAL_ROOT}/data}"
OUT_DIR="${OUT_DIR:-${AR_DATA_DIR}/wiki}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
MAX_ARTICLES_PER_LANG="${MAX_ARTICLES_PER_LANG:-20000}"
SEED="${SEED:-0}"
WORKERS_AR="${WORKERS_AR:-256}"            # 64 in-flight per GPU × 4 GPUs
FLAVOURS="${FLAVOURS:-agnostic,factual}"
APPLY_MAX_TOKENS="${APPLY_MAX_TOKENS:-2048}"
MIN_TOKENS="${MIN_TOKENS:-200}"
MAX_TOKENS="${MAX_TOKENS:-6000}"

# 28 ISO codes — same set as config_wiki.sh; trim/reorder as needed.
LANGUAGES="${LANGUAGES:-ar bs ca cs da et fi ga he hr hu is it ka ko lb lt lv no ro sk sl sq sr sv tr uk zh}"

LOG_DIR=${LOG_DIR}
mkdir -p "$LOG_DIR" "$OUT_DIR"

# Tokenizer threads off — vllm handles its own parallelism
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "[$(date +%H:%M:%S)] Config:"
echo "  MODEL=$MODEL  MAX_LEN=$MAX_LEN  MAX_SEQS=$MAX_SEQS  WORKERS=$WORKERS_AR"
echo "  LANGUAGES=$LANGUAGES"
echo "  MAX_ARTICLES_PER_LANG=$MAX_ARTICLES_PER_LANG  BATCH_SIZE=$BATCH_SIZE"

# ─────────────────────────────────────────────────────────────────
# 1. Launch 4 vLLM servers (DP=4), one per GPU, ports 8002-8005
# ─────────────────────────────────────────────────────────────────
VLLM_PIDS=()
ENDPOINTS=()
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
    --async-scheduling \
    --limit-mm-per-prompt '{"image":0,"audio":0}' \
    > "$LOG_DIR/vllm_gpu${GPU}_${SLURM_JOB_ID:-local}.log" 2>&1 &
  VLLM_PIDS+=($!)
  ENDPOINTS+=("http://127.0.0.1:$PORT")
  echo "  GPU $GPU → vLLM PID $! on port $PORT"
done

# ─────────────────────────────────────────────────────────────────
# 2. Wait for /v1/models on all 4 endpoints (load takes ~3-5 min)
# ─────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Waiting for 4 vLLM endpoints to be ready..."
for GPU in 0 1 2 3; do
  PORT=$((8002 + GPU))
  for i in $(seq 1 90); do
    if curl -sf --max-time 5 -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then
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
ENDPOINT_LIST="$(IFS=, ; echo "${ENDPOINTS[*]}")"
echo "[$(date +%H:%M:%S)] All 4 endpoints up: $ENDPOINT_LIST"

# Trap to kill vLLMs on exit (timeout, error, manual cancel)
cleanup() {
  echo "[$(date +%H:%M:%S)] cleanup: killing vLLM PIDs"
  kill "${VLLM_PIDS[@]}" 2>/dev/null || true
  sleep 5
  kill -9 "${VLLM_PIDS[@]}" 2>/dev/null || true
}
trap cleanup EXIT

# Helper: locate the .jsonl[.gz] dump for a given lang. Tries .gz first
# (config_wiki.sh assumes gzip), falls back to plain .jsonl.
lang_dump_path() {
  local lang="$1"
  for ext in jsonl.gz jsonl; do
    for d in "$WIKI_DUMP_ROOT/$lang"/wikipedia-*; do
      [ -f "$d/$lang.$ext" ] && { echo "$d/$lang.$ext"; return 0; }
    done
  done
  return 1
}

# ─────────────────────────────────────────────────────────────────
# 3. Per-language loop — generate then build pretraining JSONL
# ─────────────────────────────────────────────────────────────────
for LANG_CODE in $LANGUAGES; do
  echo
  echo "[$(date +%H:%M:%S)] === $LANG_CODE ==="

  DUMP=$(lang_dump_path "$LANG_CODE") || {
    echo "  WARN: no dump found for $LANG_CODE under $WIKI_DUMP_ROOT — skipping"
    continue
  }
  LANG_OUT="$OUT_DIR/$LANG_CODE"
  mkdir -p "$LANG_OUT"

  N_BATCHES=$(( (MAX_ARTICLES_PER_LANG + BATCH_SIZE - 1) / BATCH_SIZE ))
  echo "  dump=$DUMP  out=$LANG_OUT  $N_BATCHES batches × $BATCH_SIZE"

  for i in $(seq 0 $((N_BATCHES - 1))); do
    OFFSET=$((i * BATCH_SIZE))
    OUT="$LANG_OUT/batch_${i}.jsonl"
    LOG="$LANG_OUT/batch_${i}.log"

    if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge "$BATCH_SIZE" ]; then
      echo "  [$LANG_CODE] batch $i complete ($(wc -l < "$OUT") records), skipping"
      continue
    fi

    echo "  [$LANG_CODE] batch $i (offset=$OFFSET) → $OUT"
    python3 -u -m math_lean.active_reading.wiki_main \
      --input "$DUMP" \
      --language "$LANG_CODE" \
      --offset "$OFFSET" --n "$BATCH_SIZE" --seed "$SEED" \
      --workers "$WORKERS_AR" \
      --flavours "$FLAVOURS" \
      --endpoint "$ENDPOINT_LIST" \
      --model "$MODEL" \
      --apply-max-tokens "$APPLY_MAX_TOKENS" \
      --min-tokens "$MIN_TOKENS" \
      --max-tokens "$MAX_TOKENS" \
      --out "$OUT" \
      > "$LOG" 2>&1 || {
        echo "  ERROR: [$LANG_CODE] batch $i failed — see $LOG"
        # don't abort the whole run; move on so other batches/langs make progress
        continue
      }
    echo "  [$LANG_CODE] batch $i done — $(wc -l < "$OUT") records"
  done

  # Concat batches into all.jsonl, then build pretraining slice
  ALL="$LANG_OUT/all.jsonl"
  shopt -s nullglob
  BATCHES=("$LANG_OUT"/batch_*.jsonl)
  if [ ${#BATCHES[@]} -gt 0 ]; then
    cat "${BATCHES[@]}" > "$ALL"
    echo "  [$LANG_CODE] all.jsonl: $(wc -l < "$ALL") records"

    PRE="$LANG_OUT/pretraining_medium.jsonl"
    python3 -u -m math_lean.active_reading.to_pretraining \
      --runs-input "$ALL" \
      --out "$PRE" \
      --policy medium \
      > "$LANG_OUT/to_pretraining.log" 2>&1
    echo "  [$LANG_CODE] pretraining: $(wc -l < "$PRE") records"
  fi
  shopt -u nullglob
done

echo
echo "[$(date +%H:%M:%S)] All languages done. cleanup() will stop vLLM."
