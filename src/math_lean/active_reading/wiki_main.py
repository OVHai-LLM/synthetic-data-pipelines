"""CLI for the Wikipedia variant of the Active Reading pipeline.

Mirror of `__main__.py` but reads gzipped Wikipedia JSONL dumps and
uses the wiki-specific prompts (agnostic + factual). One-language-at-a-
time; the bash scripts under scripts/wiki/ loop this over all 27+
languages.

Examples
--------
Smoke test (50 articles from Lithuanian):

    python -m math_lean.active_reading.wiki_main \\
        --input $WIKI_DUMP_ROOT/lt/wikipedia-4/lt.jsonl.gz \\
        --language lt \\
        --n 50 --workers 8 \\
        --endpoint http://localhost:8002 \\
        --model RedHatAI/gemma-4-31B-it-FP8-Dynamic \\
        --out /tmp/wiki_lt_smoke.jsonl

Resumable batched run (used by the bash scripts):

    python -m math_lean.active_reading.wiki_main \\
        --input ... --language lt \\
        --offset 1000 --n 1000 --seed 0 \\
        --out runs/lt/batch_1.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys

from .client import VLLMClient, env_first, DEFAULT_ENDPOINT, DEFAULT_TIMEOUT
from .diagnostics import format_summary, summarise
from .pipeline import PromptConfig, run
from .prompts_wiki import (
    APPLY_LANGUAGE_SYSTEM_SUFFIX, APPLY_USER_WIKI,
    STRATEGY_FLAVOURS_WIKI,
)
from .prompts import APPLY_SYSTEM
from .wiki_loader import sample_articles


def make_wiki_prompt_config() -> PromptConfig:
    """Wiki prompts: agnostic + factual flavours, language-aware suffix
    on every apply call so the model writes the artifact in the source
    language rather than English."""
    return PromptConfig(
        strategy_flavours=STRATEGY_FLAVOURS_WIKI,
        apply_system=APPLY_SYSTEM,
        apply_user=APPLY_USER_WIKI,
        apply_system_suffix={
            # Same suffix for every flavour — instruct the model to
            # produce same-language output.
            "agnostic": APPLY_LANGUAGE_SYSTEM_SUFFIX,
            "factual":  APPLY_LANGUAGE_SYSTEM_SUFFIX,
        },
        record_format_kwargs=lambda r: {"language": r.get("_language", "en")},
    )


def main():
    ap = argparse.ArgumentParser(
        description="Active Reading on Wikipedia (one language per run).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--input", help="path to <lang>.jsonl.gz")
    ap.add_argument("--language", help="ISO 639 code (e.g. lt). Inferred from filename if omitted.")
    ap.add_argument("--out", help="output JSONL path (one record per source article)")
    ap.add_argument("--summarise", metavar="JSONL",
                    help="skip generation; print diagnostics for an existing run")

    ap.add_argument("--n", type=int, default=100, help="number of articles to sample")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N articles (use with fixed --seed for resumable batches)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)

    ap.add_argument("--flavours", default="agnostic,factual",
                    help="comma-separated subset of {agnostic, factual}")

    ap.add_argument("--strategy-temp", type=float, default=0.8)
    ap.add_argument("--apply-temp", type=float, default=0.8)
    ap.add_argument("--apply-max-tokens", type=int, default=2048)

    ap.add_argument("--min-tokens", type=int, default=200,
                    help="drop articles whose token_count is below this")
    ap.add_argument("--max-tokens", type=int, default=6000,
                    help="drop articles whose token_count is above this")

    ap.add_argument("--endpoint",
                    default=env_first("MATH_GEN_VLLM_ENDPOINT", "OPENAI_BASE_URL",
                                      "OPENAI_API_BASE", default=DEFAULT_ENDPOINT),
                    help="vLLM /v1 endpoint. Pass a single URL for one engine, "
                         "or a comma-separated list (one per DP rank) for "
                         "client-side prefix-affinity routing across DP ranks. "
                         "Example: 'http://h:8002,http://h:8003,http://h:8004,http://h:8005'")
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

    for required in ("input", "out", "model"):
        if not getattr(args, required.replace("-", "_")):
            ap.error(f"--{required} is required (model can also come from "
                     "MATH_GEN_VLLM_MODEL / OPENAI_MODEL)")

    flavours = [f.strip() for f in args.flavours.split(",") if f.strip()]
    for fl in flavours:
        if fl not in STRATEGY_FLAVOURS_WIKI:
            ap.error(f"unknown flavour {fl!r}; expected one of "
                     f"{sorted(STRATEGY_FLAVOURS_WIKI)}")

    print(f"Sampling up to {args.n} articles from {args.input} "
          f"(offset={args.offset}, seed={args.seed}) ...", flush=True)
    records = sample_articles(args.input, n=args.n, seed=args.seed,
                              offset=args.offset,
                              min_tokens=args.min_tokens,
                              max_tokens=args.max_tokens,
                              language=args.language)
    print(f"  → {len(records)} candidates pass filters "
          f"(language={records[0]['_language'] if records else 'n/a'})", flush=True)
    if not records:
        sys.exit("No articles matched the filters; adjust --min-tokens / --max-tokens.")

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
                      apply_max_tokens=args.apply_max_tokens,
                      prompt_config=make_wiki_prompt_config())

    print()
    print(f"Wrote {summary_run['written']} articles to {args.out} "
          f"({summary_run['n_strategies']} strategies, "
          f"{summary_run['n_augmentations']} augmentations, "
          f"{summary_run['elapsed_sec']:.1f}s, "
          f"failed_articles={summary_run['failed']})")
    print()
    diag = summarise(args.out)
    print(format_summary(diag))


if __name__ == "__main__":
    main()
