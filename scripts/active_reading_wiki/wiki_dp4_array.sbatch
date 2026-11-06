#!/bin/bash
#SBATCH --job-name=wiki-ar-dp4-array
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:gh200:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=600G
#SBATCH --time=12:00:00
#SBATCH --output=logs/wiki_dp4_array_%A_%a.out
#SBATCH --error=logs/wiki_dp4_array_%A_%a.err
#SBATCH --array=0-214

# Active-Reading-on-Wikipedia, FULLY PARALLEL across SHARDS.
#
# Each array task = 1 node × DP4 = 1 shard. SLURM_ARRAY_TASK_ID picks the
# slice of articles. With NUM_SHARDS=215, each shard processes ~41 K
# articles spanning all 28 languages — sized so the AVERAGE shard finishes
# in ~10 h. The 2 h walltime headroom (10 h target + 12 h SLURM cap) is
# there so unlucky shards (longer articles, slow GPU, transient stalls)
# still have room to finish their assigned slice before walltime kills them.
#
# A safety-net TIME_BUDGET_SEC=41400 (11 h 30 m) stops launching the NEXT
# batch only if we're getting close to the wall. The current batch always
# completes. In practice this trips only for the slowest ~1% of shards;
# any batches it skips are picked up by a re-submit (resume logic).
#
# To run on more or fewer nodes, change --array=0-(N-1) and pass
# NUM_SHARDS=N when sbatching:
#   sbatch wiki_dp4_array.sbatch                       # 215 shards (full ~10h each)
#   sbatch --array=0-107 --export=NUM_SHARDS=108 wiki_dp4_array.sbatch   # 108 shards × ~20h
#
# Sharding is via wiki_main's --offset/--n: shard i of N gets a
# CONTIGUOUS slice of the deterministically-shuffled article list per
# lang. With the same SEED across all shards, slices are disjoint.
#
# Within a shard the 4 GPUs share KV via the multi-endpoint client +
# pin_key=article_id (DP4 prefix-cache affinity). vLLM is launched once
# per shard and reused across all 28 langs.

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
module load Stages/2026
module load Python
module load CUDA/13
module load CUDA-Python/13
module load cuDNN/9.19.0.56-CUDA-13
module load NCCL/default-CUDA-13
module load nvidia-compilers/25.9-CUDA-13

source ${VLLM_ENV}/bin/activate
unset PYTHONPATH

# Compute nodes have no internet — force HF into local-cache-only
# mode. Model must already be present at $HF_HOME/hub. Without these flags,
# vLLM hangs ~10 min trying to validate the repo on hf.co before exiting.
# Symptom: "Cannot send a request, as the client has been closed".
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# $HOME/.cache fills the user GPFS quota — vLLM's torch.compile cache (~3 GB
# per model) blew it. Redirect everything cache-y to project space (symptom:
# OSError [Errno 122] Disk quota
# exceeded in ~/.cache/vllm/torch_compile_cache/...).
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torch_inductor_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton_cache}"
export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-${CACHE_ROOT}/flashinfer_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg_cache}"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
         "$FLASHINFER_CACHE_DIR" "$XDG_CACHE_HOME"

cd "${SRC_DIR}"
# ─────────────────────────────────────────────────────────────────
# Sharding
# ─────────────────────────────────────────────────────────────────
SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
NUM_SHARDS="${NUM_SHARDS:-215}"
# Safety net: stop launching new batches at 11h30 so we always exit cleanly
# inside the 12h walltime. Workload is sized for ~10h average per shard.
TIME_BUDGET_SEC="${TIME_BUDGET_SEC:-41400}"
T_START=$(date +%s)

# ─────────────────────────────────────────────────────────────────
# Eligible-article counts per lang (200<tok<6000), measured 2026-05-06.
# Format: lang:count separated by spaces. Update if dumps change.
# ─────────────────────────────────────────────────────────────────
ELIGIBLE_COUNTS="ar:471528 bg:179184 bs:73146 ca:431495 cs:418569 da:154419 et:97864 fi:289769 ga:14354 he:296361 hr:115658 hu:411322 is:23238 it:1207473 ka:79956 ko:398411 lb:25840 lt:150164 lv:64592 no:291578 ro:281520 sk:93396 sl:106293 sq:50603 sr:501581 sv:547156 tr:255745 uk:978480 zh:983869"

_DEFAULT_LANGS="ar bs ca cs da et fi ga he hr hu is ka ko lb lt lv no ro sk sl sq sr sv tr uk zh"
LANGUAGES="${LANGUAGES:-$_DEFAULT_LANGS}"
# Move 'it' to the front ONLY when running the default lang set (its long slice
# amortises the vLLM warmup). If the user overrode LANGUAGES (e.g. to "bg"), don't
# inject 'it' — that's how a run once wasted 11h processing 'it' instead of 'bg'.
if [ "$LANGUAGES" = "$_DEFAULT_LANGS" ]; then
  LANGUAGES="it $LANGUAGES"
fi

# ─────────────────────────────────────────────────────────────────
# vLLM config (mirror of wiki_dp4.sbatch)
# ─────────────────────────────────────────────────────────────────
AR_MODEL_REPO="${AR_MODEL_REPO:-RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic}"
# vLLM's HF-resolver path is brittle under HF_HUB_OFFLINE — it raises
# LocalEntryNotFoundError even when the snapshot is fully present (jobs
# failed this way). When the --model arg is an existing
# directory, vLLM skips HF lookup and reads the safetensors directly.
# Resolve repo → snapshot path now, with a fallback to the repo name if
# resolution fails (e.g. running outside SLURM for testing).
AR_MODEL_CACHE_DIR="${HF_HOME}/hub/models--$(echo "$AR_MODEL_REPO" | sed 's|/|--|g')"
AR_MODEL_REVISION="$(cat "$AR_MODEL_CACHE_DIR/refs/main" 2>/dev/null || true)"
if [ -n "$AR_MODEL_REVISION" ] && [ -d "$AR_MODEL_CACHE_DIR/snapshots/$AR_MODEL_REVISION" ]; then
  MODEL="$AR_MODEL_CACHE_DIR/snapshots/$AR_MODEL_REVISION"
  echo "[$(date +%H:%M:%S)] resolved model to local snapshot: $MODEL"
else
  MODEL="$AR_MODEL_REPO"
  echo "[$(date +%H:%M:%S)] WARN: could not resolve $AR_MODEL_REPO to local snapshot; falling back to repo name"
fi
MAX_LEN="${AR_MAX_LEN:-16384}"
MAX_SEQS="${AR_MAX_SEQS:-256}"
MAX_BATCHED_TOKENS="${AR_MAX_BATCHED_TOKENS:-16384}"
GPU_MEM_UTIL="${AR_GPU_MEM_UTIL:-0.92}"

# Workload knobs
WIKI_DUMP_ROOT="${WIKI_DUMP_ROOT:-${ANNEAL_ROOT}/data}"
OUT_DIR="${OUT_DIR:-${AR_DATA_DIR}/wiki}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
SEED="${SEED:-0}"
WORKERS_AR="${WORKERS_AR:-256}"
FLAVOURS="${FLAVOURS:-agnostic,factual}"
APPLY_MAX_TOKENS="${APPLY_MAX_TOKENS:-2048}"
MIN_TOKENS="${MIN_TOKENS:-200}"
MAX_TOKENS="${MAX_TOKENS:-6000}"

LOG_DIR=${LOG_DIR}
mkdir -p "$LOG_DIR" "$OUT_DIR"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "[$(date +%H:%M:%S)] shard $SHARD_ID / $NUM_SHARDS  job=${SLURM_JOB_ID:-local}  array=${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}"
echo "  model=$MODEL  max_len=$MAX_LEN  workers=$WORKERS_AR"

# ─────────────────────────────────────────────────────────────────
# 1. Boot 4 vLLM (DP4)
# ─────────────────────────────────────────────────────────────────
VLLM_PIDS=()
ENDPOINTS=()
for GPU in 0 1 2 3; do
  PORT=$((8002 + GPU))
  # setsid → vllm runs in its own process group so we can kill the whole
  # tree (engine subprocesses, etc.) with `kill -- -PGID`.
  setsid bash -c "CUDA_VISIBLE_DEVICES=$GPU exec vllm serve '$MODEL' \
    --host 127.0.0.1 --port $PORT \
    --max-model-len $MAX_LEN \
    --max-num-seqs $MAX_SEQS \
    --max-num-batched-tokens $MAX_BATCHED_TOKENS \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --async-scheduling \
    --limit-mm-per-prompt '{\"image\":0,\"audio\":0}'" \
    > "$LOG_DIR/vllm_shard${SHARD_ID}_gpu${GPU}_${SLURM_JOB_ID:-local}.log" 2>&1 &
  VLLM_PIDS+=($!)
  ENDPOINTS+=("http://127.0.0.1:$PORT")
done

# Make sure cleanup runs on any exit (normal, error, SIGTERM from SLURM,
# Ctrl-C). EXIT alone covers most cases; INT/TERM make the intent explicit.
# Cleanup is bounded — never blocks longer than ~10s.
cleanup() {
  rc=$?
  echo "[$(date +%H:%M:%S)] cleanup: stopping vLLM (rc=$rc)"
  # 1. polite SIGTERM to each vLLM process group
  for pid in "${VLLM_PIDS[@]}"; do
    [ -n "$pid" ] && kill -- -"$pid" 2>/dev/null || true
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
  done
  sleep 5
  # 2. SIGKILL anything still alive
  for pid in "${VLLM_PIDS[@]}"; do
    [ -n "$pid" ] && kill -9 -- -"$pid" 2>/dev/null || true
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
  done
  # 3. backup sweep — catch any orphaned vllm/engine workers regardless
  pkill -9 -f "vllm serve" 2>/dev/null || true
  pkill -9 -f "EngineCore" 2>/dev/null || true
  pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
  echo "[$(date +%H:%M:%S)] cleanup done"
  return $rc
}
trap cleanup EXIT INT TERM

# Wait for /v1/models on all 4
echo "[$(date +%H:%M:%S)] waiting for 4 endpoints..."
for GPU in 0 1 2 3; do
  PORT=$((8002 + GPU))
  for i in $(seq 1 90); do
    curl -sf --max-time 5 -o /dev/null "http://127.0.0.1:$PORT/v1/models" && break
    sleep 10
    [ "$i" -eq 90 ] && { echo "ERROR: port $PORT not ready"; exit 1; }
  done
done
ENDPOINT_LIST="$(IFS=, ; echo "${ENDPOINTS[*]}")"
echo "[$(date +%H:%M:%S)] ready: $ENDPOINT_LIST"

# Helper: locate the wiki dump for a given lang. Returns a DIRECTORY
# containing the lang's jsonl(.gz) shards — wiki_loader.iter_articles
# now walks dirs, so multi-shard dumps (it: it_10.jsonl, it_11.jsonl, …,
# sv: sv_1..sv_8, ar: ar_1..ar_7) all get picked up.
# Preference order: wikipedia-1 > wikipedia-1_shuffled > any wikipedia-*
# > wikipedia-<lang> (zh's special-case dir name).
lang_dump_path() {
  local lang="$1"
  local ld="$WIKI_DUMP_ROOT/$lang"
  for cand in "wikipedia-1" "wikipedia-1_shuffled" "wikipedia-$lang" "wikepedia-1"; do
    [ -d "$ld/$cand" ] && [ -n "$(ls -A "$ld/$cand" 2>/dev/null)" ] && {
      echo "$ld/$cand"; return 0;
    }
  done
  for d in "$ld"/wikipedia-* "$ld"/wikepedia-*; do
    [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ] && {
      echo "$d"; return 0;
    }
  done
  return 1
}

# Helper: compute (offset, n) for this shard given total eligible T
shard_slice() {
  local total="$1" shard="$2" nshards="$3"
  # Even-ish split: first (total mod nshards) shards get an extra article
  local base=$(( total / nshards ))
  local rem=$(( total % nshards ))
  local off n
  if [ "$shard" -lt "$rem" ]; then
    n=$(( base + 1 ))
    off=$(( shard * n ))
  else
    n=$base
    off=$(( rem * (base + 1) + (shard - rem) * base ))
  fi
  echo "$off $n"
}

# Look up eligible count for a lang from ELIGIBLE_COUNTS
eligible_count() {
  local lang="$1"
  for kv in $ELIGIBLE_COUNTS; do
    [[ "$kv" == "$lang:"* ]] && { echo "${kv#*:}"; return 0; }
  done
  echo 0
}

# ─────────────────────────────────────────────────────────────────
# 2. Per-language: compute this shard's slice, run wiki_main on it
# ─────────────────────────────────────────────────────────────────
TOTAL_ARTICLES=0
TIME_OUT=0
for LANG_CODE in $LANGUAGES; do
  [ "$TIME_OUT" -eq 1 ] && break

  TOTAL=$(eligible_count "$LANG_CODE")
  [ "$TOTAL" -eq 0 ] && { echo "  WARN: no eligible count for $LANG_CODE — skipping"; continue; }

  read -r OFFSET N_THIS_SHARD < <(shard_slice "$TOTAL" "$SHARD_ID" "$NUM_SHARDS")
  if [ "$N_THIS_SHARD" -le 0 ]; then
    echo "  [$LANG_CODE] shard $SHARD_ID gets 0 articles (total=$TOTAL) — skipping"
    continue
  fi

  DUMP=$(lang_dump_path "$LANG_CODE") || { echo "  WARN: no dump for $LANG_CODE"; continue; }
  LANG_OUT="$OUT_DIR/$LANG_CODE/shard_${SHARD_ID}_of_${NUM_SHARDS}"
  mkdir -p "$LANG_OUT"

  N_BATCHES=$(( (N_THIS_SHARD + BATCH_SIZE - 1) / BATCH_SIZE ))
  echo
  echo "[$(date +%H:%M:%S)] === $LANG_CODE :: shard $SHARD_ID/$NUM_SHARDS ==="
  echo "  total_eligible=$TOTAL  shard_offset=$OFFSET  shard_n=$N_THIS_SHARD  batches=$N_BATCHES × $BATCH_SIZE"

  for i in $(seq 0 $((N_BATCHES - 1))); do
    # Safety-net time budget — checked BEFORE starting each batch.
    # Sizing targets 10h avg per shard; this trips only for the slowest
    # outliers. Current batch always runs to completion once started, so
    # actual wall = TIME_BUDGET_SEC + (one batch worth of work).
    ELAPSED=$(( $(date +%s) - T_START ))
    if [ "$ELAPSED" -ge "$TIME_BUDGET_SEC" ]; then
      echo "  [safety-net] ${ELAPSED}s ≥ ${TIME_BUDGET_SEC}s — current run hit the safety budget; remaining batches resume on re-submit"
      TIME_OUT=1
      break
    fi

    BATCH_OFFSET=$(( OFFSET + i * BATCH_SIZE ))
    BATCH_N=$BATCH_SIZE
    REMAINING=$(( N_THIS_SHARD - i * BATCH_SIZE ))
    [ "$BATCH_N" -gt "$REMAINING" ] && BATCH_N=$REMAINING

    OUT="$LANG_OUT/batch_${i}.jsonl"
    LOG="$LANG_OUT/batch_${i}.log"

    if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge "$BATCH_N" ]; then
      echo "  [$LANG_CODE] s$SHARD_ID b$i complete ($(wc -l < "$OUT") rec), skip"
      continue
    fi

    python3 -u -m math_lean.active_reading.wiki_main \
      --input "$DUMP" \
      --language "$LANG_CODE" \
      --offset "$BATCH_OFFSET" --n "$BATCH_N" --seed "$SEED" \
      --workers "$WORKERS_AR" \
      --flavours "$FLAVOURS" \
      --endpoint "$ENDPOINT_LIST" \
      --model "$MODEL" \
      --apply-max-tokens "$APPLY_MAX_TOKENS" \
      --min-tokens "$MIN_TOKENS" --max-tokens "$MAX_TOKENS" \
      --out "$OUT" \
      > "$LOG" 2>&1 || { echo "  ERROR: $LANG_CODE s$SHARD_ID b$i failed — see $LOG"; continue; }
    DONE_NOW=$(wc -l < "$OUT")
    TOTAL_ARTICLES=$(( TOTAL_ARTICLES + DONE_NOW ))
    echo "  [$LANG_CODE] s$SHARD_ID b$i done — $DONE_NOW rec  elapsed=$(( $(date +%s) - T_START ))s"
  done

  # Concat this shard's batches for the lang
  ALL="$LANG_OUT/all.jsonl"
  shopt -s nullglob
  BATCHES=("$LANG_OUT"/batch_*.jsonl)
  [ ${#BATCHES[@]} -gt 0 ] && cat "${BATCHES[@]}" > "$ALL"
  shopt -u nullglob
done

WALL=$(( $(date +%s) - T_START ))
RATE=$(awk "BEGIN{ if ($WALL>0) printf \"%.2f\", $TOTAL_ARTICLES/$WALL; else print 0 }")
echo
if [ "$TIME_OUT" -eq 1 ]; then
  echo "[$(date +%H:%M:%S)] shard $SHARD_ID/$NUM_SHARDS HIT TIME BUDGET — $TOTAL_ARTICLES articles in ${WALL}s (${RATE}/s); resume next array submit"
else
  echo "[$(date +%H:%M:%S)] shard $SHARD_ID/$NUM_SHARDS COMPLETE — $TOTAL_ARTICLES articles in ${WALL}s (${RATE}/s)"
fi
# trap cleanup runs after this — vLLM gets killed cleanly within ~10s
