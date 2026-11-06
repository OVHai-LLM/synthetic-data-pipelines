"""CLI entry point.

Examples:
    # Smoke test: 3 theorems, both flavours, write to /tmp
    python -m math_lean.active_reading \\
        --proofwiki data/proofwiki_proofs.jsonl \\
        --n 3 --workers 4 \\
        --out /tmp/ar_smoke.jsonl

    # 100-theorem prototype run
    python -m math_lean.active_reading \\
        --proofwiki data/proofwiki_proofs.jsonl \\
        --n 100 --workers 16 \\
        --out runs/ar_proto_v1.jsonl

    # Diagnostics-only on an existing run
    python -m math_lean.active_reading --summarise runs/ar_proto_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys

from .client import VLLMClient, env_first, DEFAULT_ENDPOINT, DEFAULT_TIMEOUT
from .diagnostics import format_summary, summarise
from .pipeline import run, sample_theorems


def main():
    ap = argparse.ArgumentParser(
        description="Active Reading prototype on ProofWiki.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--proofwiki", help="path to proofwiki_proofs.jsonl")
    ap.add_argument("--out", help="output JSONL path (one record per source doc)")
    ap.add_argument("--summarise", metavar="JSONL",
                    help="skip generation; print diagnostics for an existing run")

    ap.add_argument("--n", type=int, default=100, help="number of theorems to sample")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N theorems (use with fixed --seed for resumable batches)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)

    ap.add_argument("--flavours", default="agnostic,lean",
                    help="comma-separated subset of {agnostic, lean}")

    ap.add_argument("--strategy-temp", type=float, default=0.8)
    ap.add_argument("--apply-temp", type=float, default=0.8)
    ap.add_argument("--apply-max-tokens", type=int, default=2048)

    ap.add_argument("--min-proof-chars", type=int, default=200)
    ap.add_argument("--max-chunk-chars", type=int, default=8000)

    ap.add_argument("--endpoint",
                    default=env_first("MATH_GEN_VLLM_ENDPOINT", "OPENAI_BASE_URL",
                                      "OPENAI_API_BASE", default=DEFAULT_ENDPOINT))
    ap.add_argument("--model",
                    default=env_first("MATH_GEN_VLLM_MODEL", "OPENAI_MODEL"))
    ap.add_argument("--api-key",
                    default=env_first("MATH_GEN_VLLM_API_KEY", "OPENAI_API_KEY"))
    ap.add_argument("--timeout", type=int,
                    default=int(env_first("MATH_GEN_VLLM_TIMEOUT",
                                          default=str(DEFAULT_TIMEOUT))))
    args = ap.parse_args()

    if args.summarise:
        summary = summarise(args.summarise)
        print(format_summary(summary))
        print()
        print("--- raw summary ---")
        print(json.dumps(summary, indent=2))
        return

    for required in ("proofwiki", "out", "model"):
        if not getattr(args, required.replace("-", "_")):
            ap.error(f"--{required} is required (model can also come from "
                     "MATH_GEN_VLLM_MODEL / OPENAI_MODEL)")

    flavours = [f.strip() for f in args.flavours.split(",") if f.strip()]
    for fl in flavours:
        if fl not in {"agnostic", "lean"}:
            ap.error(f"unknown flavour {fl!r}; expected 'agnostic' or 'lean'")

    print(f"Sampling up to {args.n} theorems from {args.proofwiki} "
          f"(offset={args.offset}, seed={args.seed}) ...", flush=True)
    records = sample_theorems(args.proofwiki, n=args.n, seed=args.seed,
                              min_proof_chars=args.min_proof_chars,
                              max_chunk_chars=args.max_chunk_chars,
                              offset=args.offset)
    print(f"  → {len(records)} candidates pass filters", flush=True)
    if not records:
        sys.exit("No theorems matched the filters; relax --min-proof-chars or --max-chunk-chars.")

    print(f"Endpoint: {args.endpoint}  model={args.model}  workers={args.workers}", flush=True)
    print(f"Flavours: {flavours}  strat_temp={args.strategy_temp}  "
          f"apply_temp={args.apply_temp}  apply_max_tokens={args.apply_max_tokens}",
          flush=True)

    client = VLLMClient(args.endpoint, args.model,
                        api_key=args.api_key, timeout=args.timeout)

    summary_run = run(records, client, args.out,
                      flavours=flavours, workers=args.workers,
                      strategy_temp=args.strategy_temp,
                      apply_temp=args.apply_temp,
                      apply_max_tokens=args.apply_max_tokens)

    print()
    print(f"Wrote {summary_run['written']} docs to {args.out} "
          f"({summary_run['n_strategies']} strategies, "
          f"{summary_run['n_augmentations']} augmentations, "
          f"{summary_run['elapsed_sec']:.1f}s, "
          f"failed_docs={summary_run['failed']})")
    print()
    diag = summarise(args.out)
    print(format_summary(diag))


if __name__ == "__main__":
    main()
