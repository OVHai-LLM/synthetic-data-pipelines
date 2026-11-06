"""Async high-throughput Goedel-Prover-V2 self-correction.

Replaces the thread-per-worker model in goedel_retry.py with a proper
asyncio loop. Each "task" is one snippet's full self-correction sequence
(round 0 → maybe round 1 → maybe round 2). Many tasks run concurrently;
vLLM's continuous batching gets to fill its KV pool fully.

Lean compile (subprocess, blocking) is offloaded to asyncio.to_thread so
it doesn't stall the event loop.

Why: the synchronous goedel_retry.py only kept ~14 concurrent in-flight
in vLLM even with 64 worker threads (Python HTTP overhead per request).
This async version pushes hundreds of in-flight chats with no thread
overhead, saturating vLLM's batch.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import List, Optional

from openai import AsyncOpenAI

from .lean_check import (
    check_one,
    DEFAULT_PROJECT, DEFAULT_LAKE,
)
from .lean_retry import (
    index_runs,
    trim_error,
    select_failing,
    _extract_code,
)
from .goedel_retry import (
    INITIAL_USER_TEMPLATE,
    CORRECTION_USER_TEMPLATE,
    extract_theorem_statement,
    annotate_error,
    parse_proof_from_response,
)


async def goedel_one_async(
    snippet: dict,
    *,
    client: AsyncOpenAI,
    model: str,
    sem: asyncio.Semaphore,
    max_rounds: int = 2,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_tokens: int = 16384,
    project: str = DEFAULT_PROJECT,
    lake_bin: str = DEFAULT_LAKE,
    timeout_sec: int = 90,
    skip_lean_check: bool = False,
) -> dict:
    """Async version of goedel_retry.goedel_retry_one."""
    doc_id = snippet.get("doc_id")
    formal_statement = extract_theorem_statement(snippet.get("code", ""))
    chat = [
        {"role": "user",
         "content": INITIAL_USER_TEMPLATE.format(formal_statement=formal_statement)}
    ]
    rounds_log: list = []
    final_verdict = None
    final_code = None

    for round_idx in range(max_rounds + 1):
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=model, messages=chat,
                    temperature=temperature, top_p=top_p,
                    max_tokens=max_tokens, n=1,
                )
            except Exception as e:
                rounds_log.append({"round": round_idx, "model_error": str(e)[:300]})
                break
            response_text = resp.choices[0].message.content

        code = parse_proof_from_response(response_text or "")
        if not code:
            rounds_log.append({"round": round_idx, "no_code_in_response": True,
                               "raw_response": (response_text or "")[:1500]})
            break

        if skip_lean_check:
            check_res = {"verdict": "skipped", "stdout": "", "elapsed_sec": 0.0}
        else:
            # blocking subprocess → off-thread so event loop keeps going
            check_res = await asyncio.to_thread(
                check_one, {"code": code},
                project=project, lake_bin=lake_bin, timeout_sec=timeout_sec,
            )
        verdict = check_res.get("verdict")
        rounds_log.append({
            "round": round_idx, "code": code, "verdict": verdict,
            "stdout_tail": (check_res.get("stdout") or "")[-800:],
            "elapsed_sec": check_res.get("elapsed_sec"),
            "completion_tokens": getattr(resp.usage, "completion_tokens", None) if 'resp' in locals() else None,
        })
        final_verdict = verdict
        final_code = code

        if verdict in ("ok", "ok_with_sorry", "timeout", "skipped"):
            break
        if round_idx >= max_rounds:
            break

        # Build correction turn
        chat.append({"role": "assistant", "content": response_text})
        err = annotate_error(check_res.get("stdout") or "", code)
        chat.append({"role": "user",
                     "content": CORRECTION_USER_TEMPLATE.format(
                         round_idx=round_idx + 1, error_message=err)})

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
        "improved": (final_verdict in ("ok", "ok_with_sorry")
                     and snippet.get("verdict") not in ("ok", "ok_with_sorry")),
    }


def _snippet_key(s):
    """Uniquely identify a snippet across re-runs."""
    return (s.get("doc_id"), s.get("strategy_name"), s.get("snippet_index"))


async def main_async(args):
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
        print(f"[{time.strftime('%H:%M:%S')}] sharded: {len(failing)} of {n_total}", file=sys.stderr)
    if args.max_snippets:
        failing = failing[: args.max_snippets]

    # Resume: skip snippets already completed (or errored) in the output file
    already_done = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    already_done.add((r.get("doc_id"), r.get("strategy_name"), r.get("snippet_index")))
                except json.JSONDecodeError:
                    continue
        before = len(failing)
        failing = [s for s in failing if _snippet_key(s) not in already_done]
        print(f"[{time.strftime('%H:%M:%S')}] resume: skipping {before - len(failing)} "
              f"already-done snippets ({len(failing)} remaining)", file=sys.stderr)
    print(f"[{time.strftime('%H:%M:%S')}] {len(failing)} snippets to retry", file=sys.stderr)
    if not failing:
        print(f"[{time.strftime('%H:%M:%S')}] nothing to do — all done.", file=sys.stderr)
        return

    client = AsyncOpenAI(
        base_url=args.endpoint.rstrip("/") + "/v1",
        api_key="EMPTY",
        timeout=900.0,
        max_retries=2,
    )
    sem = asyncio.Semaphore(args.concurrency)

    n_done = n_improved = n_ok = n_err = 0
    t0 = time.time()
    # Append mode if resuming, else fresh write
    fout = open(args.output, "a" if args.resume else "w")

    async def run_and_write(snippet):
        nonlocal n_done, n_improved, n_ok, n_err
        try:
            rec = await goedel_one_async(
                snippet, client=client, model=args.model, sem=sem,
                max_rounds=args.max_rounds,
                temperature=args.temperature, top_p=args.top_p,
                max_tokens=args.max_tokens,
                project=args.project, lake_bin=args.lake_bin,
                timeout_sec=args.lean_timeout,
                skip_lean_check=args.skip_lean_check,
            )
        except Exception as e:
            rec = {"doc_id": snippet.get("doc_id"), "error": str(e)[:300]}
            n_err += 1
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        n_done += 1
        if rec.get("improved"):
            n_improved += 1
        if rec.get("final_verdict") == "ok":
            n_ok += 1
        if n_done % 25 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(failing) - n_done) / max(rate, 0.001)
            print(f"  {n_done}/{len(failing)} | improved {n_improved} | ok {n_ok} | "
                  f"err {n_err} | {rate:.2f} snip/s | ETA {eta/60:.0f} min",
                  file=sys.stderr)

    await asyncio.gather(*(run_and_write(s) for s in failing))
    fout.close()

    print(f"\n[{time.strftime('%H:%M:%S')}] DONE. {n_done} snippets, "
          f"{n_improved} improved ({100*n_improved/max(n_done,1):.1f}%), "
          f"{n_ok} fully ok, {n_err} errors", file=sys.stderr)
    print(f"  Output: {args.output}", file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-input", required=True)
    p.add_argument("--check-input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--endpoint", default=os.environ.get("GOEDEL_ENDPOINT", "http://localhost:8002"))
    p.add_argument("--model", default=os.environ.get("GOEDEL_MODEL", "Goedel-LM/Goedel-Prover-V2-32B"))
    p.add_argument("--max-rounds", type=int, default=2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=32768,
                   help="Max output tokens per request. Goedel's CoT can be 6-15K tokens "
                        "with 6-22%% hitting the cap at 16K — 32K gives them room to finish.")
    p.add_argument("--lean-timeout", type=int, default=90)
    p.add_argument("--max-snippets", type=int, default=0)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=256,
                   help="Max in-flight requests to vLLM (asyncio.Semaphore)")
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--lake-bin", default=DEFAULT_LAKE)
    p.add_argument("--skip-lean-check", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing --output JSONL (append, skip already-done by "
                        "(doc_id, strategy_name, snippet_index)). Use to restart after sbatch timeout.")
    args = p.parse_args(argv)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
