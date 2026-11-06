#!/usr/bin/env bash
# Shared configuration for the full Active Reading pipeline.
# Sourced by every step script. Override any var by exporting it before running.

# ─── paths ───────────────────────────────────────────────────────────────

# ── repo bootstrap: locate the checkout (also from a SLURM spool copy) + site config ──
if [ -z "${REPO_ROOT:-}" ]; then
  REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null \
            || git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
fi
export REPO_ROOT
[ -f "$REPO_ROOT/configs/site.env" ] && : "${SITE_ENV:=$REPO_ROOT/configs/site.env}"
source "${SITE_ENV:-$REPO_ROOT/configs/site.env.example}"
# ─────────────────────────────────────────────────────────────────────────────

: "${PROOFWIKI:=$ANNEAL_ROOT/proofwiki_proofs.jsonl}"
: "${OUT_DIR:=$AR_DATA_DIR/proofwiki}"
: "${PYTHON:=python3}"

# ─── AR generation knobs ─────────────────────────────────────────────────
: "${BATCH_SIZE:=1000}"          # theorems per batch — keeps re-runs cheap
: "${N_BATCHES:=20}"             # 20 × 1000 ≈ all theorems passing the filter
: "${SEED:=0}"
: "${WORKERS_AR:=32}"            # safe to push to 48-64 if vLLM has headroom
: "${FLAVOURS:=agnostic,lean}"
: "${APPLY_MAX_TOKENS:=4096}"
: "${MIN_PROOF_CHARS:=200}"
: "${MAX_CHUNK_CHARS:=8000}"

# ─── lean_check knobs ────────────────────────────────────────────────────
: "${WORKERS_CHECK:=8}"
: "${CHECK_TIMEOUT_SEC:=60}"

# ─── lean_retry knobs ────────────────────────────────────────────────────
: "${WORKERS_RETRY:=12}"
: "${RETRY_TEMPERATURE:=0.3}"
: "${RETRY_MAX_TOKENS:=1536}"

# ─── vLLM endpoints ──────────────────────────────────────────────────────
# Endpoint used for AR generation (initial artifacts).
: "${AR_ENDPOINT:=http://localhost:8002}"
: "${AR_MODEL:=RedHatAI/gemma-4-31B-it-FP8-Dynamic}"

# Endpoint used for the lean retry passes. Default to the same endpoint;
# point this at a Lean-specialist model (e.g. on port 8003) if you have one.
: "${RETRY_ENDPOINT:=$AR_ENDPOINT}"
: "${RETRY_MODEL:=$AR_MODEL}"

# ─── filter policy for the pretraining corpus ───────────────────────────
: "${PRETRAIN_POLICY:=medium}"   # strict | medium | loose | all

# ─── derived ─────────────────────────────────────────────────────────────
ALL_AR="$OUT_DIR/all.jsonl"
ALL_CHECK="$OUT_DIR/all.lean_check.jsonl"
ALL_RETRY1="$OUT_DIR/all.lean_retry1.jsonl"
CHECK_AFTER_R1="$OUT_DIR/all.lean_check_after_r1.jsonl"
ALL_RETRY2="$OUT_DIR/all.lean_retry2.jsonl"
CHECK_FINAL="$OUT_DIR/all.lean_check_final.jsonl"
PRETRAIN_AGNOSTIC="$OUT_DIR/pretrain_agnostic.jsonl"
PRETRAIN_LEAN="$OUT_DIR/pretrain_lean.jsonl"
PRETRAIN_OUT="$OUT_DIR/pretraining_${PRETRAIN_POLICY}.jsonl"

mkdir -p "$OUT_DIR"

log() { printf '\033[1;34m[AR]\033[0m %s\n' "$*" >&2; }
ok()  { printf '\033[1;32m[OK]\033[0m %s\n' "$*" >&2; }
err() { printf '\033[1;31m[!!]\033[0m %s\n' "$*" >&2; }
