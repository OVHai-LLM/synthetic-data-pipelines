"""Goedel-Prover-V2 self-correction retry pass for failed Lean snippets.

Differs from `lean_retry.py`:
  * Uses Goedel-Prover-V2's exact prompt format (chat template, single user
    turn for round 0).
  * Multi-round self-correction loop: failed assistant turn stays in chat
    history; new user turn appends compile error wrapped in <error>...</error>.
  * Default 2 correction rounds (matches Goedel-Prover-V2's published
    MiniF2F-Pass@N=90.4% pipeline).
  * Long-CoT reasoning is implicit in Goedel's training (it produces a
    "proof plan" before the Lean code).

Pipeline:
  Round 0: prompt with theorem statement → Goedel produces (plan + proof)
  Compile-check.
  If fail and rounds < max_rounds:
    Round k: append assistant's prior proof, then user msg with the error
    in <error>...</error> wrap → Goedel produces (analysis + revised proof)

Output JSONL: one record per snippet, with all rounds' (code, verdict, error).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from .goedel_client import GoedelClient, ChatMessage, DEFAULT_ENDPOINT, DEFAULT_TIMEOUT
from .lean_check import (
    check_one,
    DEFAULT_PROJECT, DEFAULT_LAKE,
    _FENCE_RE,
)
from .lean_retry import (
    index_runs,
    trim_error,
    select_failing,
    _extract_code,
)


# ─────────────────────────────────────────────────────────────────
# Goedel-Prover-V2 prompts (verbatim from official scripts/pipeline.sh)
# ─────────────────────────────────────────────────────────────────

INITIAL_USER_TEMPLATE = """Complete the following Lean 4 code:

```lean4
{formal_statement}```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."""

CORRECTION_USER_TEMPLATE = """The proof (Round {round_idx}) is not correct. Following is the compilation error message, where we use <error></error> to signal the position of the error.

{error_message}

Please carefully analyze the error and produce a corrected Lean 4 proof.
Before producing the Lean 4 code to formally prove the given theorem, provide a detailed analysis of the error message."""


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

_THEOREM_HEADER_RE = re.compile(
    r"(?:^|\n)\s*(theorem|lemma|example|def|instance)\s+\w+.*?:=\s*by",
    re.DOTALL,
)


def extract_theorem_statement(code: str) -> str:
    """Extract just the theorem statement (everything up to ':= by'),
    so we can hand a clean signature to Goedel for re-proof.

    Falls back to returning the whole snippet if no theorem header found
    (e.g., when the snippet is signature-only or already failed at parse)."""
    m = _THEOREM_HEADER_RE.search(code)
    if not m:
        return code.strip()
    # Keep through the ':= by' opener; Goedel will fill in the proof body.
    end = m.end()
    return code[:end].strip() + " sorry"


def annotate_error(stdout: str, code: str, max_chars: int = 1500) -> str:
    """Wrap the most-relevant portion of the Lean error in <error>...</error>
    tags as Goedel's correction prompt expects.

    Strategy: take the trimmed compiler tail (via lean_retry.trim_error),
    then wrap it as a single <error>...</error> block. Inline-position
    annotation is hard without a structured Lean diagnostic; the wrap is
    enough for the model to attend to it.
    """
    err = trim_error(stdout, max_chars=max_chars)
    return f"<error>\n{err}\n</error>"


def parse_proof_from_response(text: str) -> Optional[str]:
    """Goedel emits ```lean4 ... ``` (or ```lean ... ```) fences."""
    return _extract_code(text)


# ─────────────────────────────────────────────────────────────────
# One snippet's full self-correction loop
# ─────────────────────────────────────────────────────────────────

def goedel_retry_one(
    snippet: dict,
    runs_index: dict,
    *,
    client: GoedelClient,
    max_rounds: int = 2,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_tokens: int = 32768,
    project: str = DEFAULT_PROJECT,
    lake_bin: str = DEFAULT_LAKE,
    timeout_sec: int = 90,
    skip_lean_check: bool = False,
) -> dict:
    """Run Goedel-Prover-V2 self-correction on one failed snippet.

    Returns a dict with all rounds' results.
    """
    doc_id = snippet.get("doc_id")
    info = runs_index.get(doc_id, {})

    formal_statement = extract_theorem_statement(snippet.get("code", ""))

    chat: List[ChatMessage] = [
        ChatMessage(role="user",
                    content=INITIAL_USER_TEMPLATE.format(formal_statement=formal_statement))
    ]

    rounds_log: list = []
    final_verdict = None
    final_code = None

    for round_idx in range(max_rounds + 1):  # 0 = initial, 1..max = corrections
        try:
            completions = client.chat(
                chat,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n=1,
            )
        except Exception as e:
            rounds_log.append({"round": round_idx, "model_error": str(e)})
            break
        response = completions[0]
        code = parse_proof_from_response(response)
        if not code:
            rounds_log.append({"round": round_idx, "no_code_in_response": True,
                               "raw_response": response[:2000]})
            break

        if skip_lean_check:
            check_res = {"verdict": "skipped", "stdout": "", "elapsed_sec": 0.0}
        else:
            check_res = check_one({"code": code}, project=project, lake_bin=lake_bin,
                                  timeout_sec=timeout_sec)
        verdict = check_res.get("verdict")
        rounds_log.append({
            "round": round_idx,
            "code": code,
            "verdict": verdict,
            "stdout_tail": (check_res.get("stdout") or "")[-800:],
            "elapsed_sec": check_res.get("elapsed_sec"),
        })
        final_verdict = verdict
        final_code = code

        if verdict in ("ok", "ok_with_sorry", "timeout", "skipped"):
            break

        if round_idx >= max_rounds:
            break

        # Build correction turn: append assistant + user(error)
        chat.append(ChatMessage(role="assistant", content=response))
        err_msg = annotate_error(check_res.get("stdout") or "", code)
        chat.append(ChatMessage(
            role="user",
            content=CORRECTION_USER_TEMPLATE.format(
                round_idx=round_idx + 1,
                error_message=err_msg,
            ),
        ))

    return {
        "doc_id": doc_id,
        "title": snippet.get("title"),
        "strategy_name": snippet.get("strategy_name"),
        "snippet_index": snippet.get("snippet_index"),
        "original_verdict": snippet.get("verdict_v2") or snippet.get("verdict"),
        "final_verdict": final_verdict,
        "final_code": final_code,
        "n_rounds": len(rounds_log),
        "rounds": rounds_log,
        "improved": (
            final_verdict in ("ok", "ok_with_sorry")
            and (snippet.get("verdict") not in ("ok", "ok_with_sorry"))
        ),
    }


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-input", required=True,
                   help="AR pipeline output (active_reading.jsonl) — for context")
    p.add_argument("--check-input", required=True,
                   help="lean_check JSONL with verdicts + codes")
    p.add_argument("--output", required=True,
                   help="Output JSONL (one record per retried snippet)")
    p.add_argument("--endpoint", default=os.environ.get("GOEDEL_ENDPOINT", DEFAULT_ENDPOINT),
                   help="vLLM OpenAI-API endpoint (default localhost:8002)")
    p.add_argument("--model", default=os.environ.get("GOEDEL_MODEL", "Goedel-LM/Goedel-Prover-V2-32B"))
    p.add_argument("--max-rounds", type=int, default=2,
                   help="Self-correction rounds (Goedel paper uses 2)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--lean-timeout", type=int, default=90)
    p.add_argument("--max-snippets", type=int, default=0,
                   help="Cap number of snippets to process (0 = all)")
    p.add_argument("--shard-id", type=int, default=0,
                   help="Process only snippets[shard_id::num_shards] — for SLURM splitting")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total shards (e.g. 4 to split across 4 SLURM tasks)")
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent in-flight requests to the vLLM server")
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--lake-bin", default=DEFAULT_LAKE)
    p.add_argument("--skip-lean-check", action="store_true",
                   help="Skip Lean compile-check (smoke testing without lake). "
                        "Disables multi-round self-correction since we have no errors to feed back.")
    args = p.parse_args(argv)

    print(f"[{time.strftime('%H:%M:%S')}] indexing runs ...", file=sys.stderr)
    runs_index = index_runs(args.runs_input)

    print(f"[{time.strftime('%H:%M:%S')}] reading check input ...", file=sys.stderr)
    with open(args.check_input) as f:
        check_records = [json.loads(line) for line in f]

    failing = select_failing(check_records)
    if not (0 <= args.shard_id < args.num_shards):
        sys.exit(f"--shard-id must be in [0, {args.num_shards})")
    if args.num_shards > 1:
        n_total = len(failing)
        failing = failing[args.shard_id::args.num_shards]
        print(f"[{time.strftime('%H:%M:%S')}] sharded: {len(failing)} snippets "
              f"(shard {args.shard_id}/{args.num_shards} of {n_total})", file=sys.stderr)
    if args.max_snippets:
        failing = failing[: args.max_snippets]
    print(f"[{time.strftime('%H:%M:%S')}] {len(failing)} snippets to retry "
          f"(of {len(check_records)} total)", file=sys.stderr)

    client = GoedelClient(endpoint=args.endpoint, model=args.model)

    n_done = n_improved = n_ok = 0
    t0 = time.time()
    with open(args.output, "w") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                goedel_retry_one,
                snippet, runs_index,
                client=client,
                max_rounds=args.max_rounds,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                project=args.project,
                lake_bin=args.lake_bin,
                timeout_sec=args.lean_timeout,
                skip_lean_check=args.skip_lean_check,
            ): snippet
            for snippet in failing
        }
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except Exception as e:
                snippet = futures[fut]
                rec = {"doc_id": snippet.get("doc_id"), "error": str(e)}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if rec.get("improved"):
                n_improved += 1
            if rec.get("final_verdict") == "ok":
                n_ok += 1
            if n_done % 50 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(failing) - n_done) / max(rate, 0.001)
                print(f"  {n_done}/{len(failing)} | improved {n_improved} | ok {n_ok} | "
                      f"{rate:.2f} snip/s | ETA {eta/60:.0f} min", file=sys.stderr)

    print(f"\n[{time.strftime('%H:%M:%S')}] DONE. {n_done} snippets, "
          f"{n_improved} improved ({100*n_improved/max(n_done,1):.1f}%), "
          f"{n_ok} fully ok", file=sys.stderr)
    print(f"  Output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
