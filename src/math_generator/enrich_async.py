"""enrich_async.py — async high-throughput drop-in for math_generator.enrich.

Mirrors enrich.py exactly (same prompt pool, same judge step, same per-style
diversity hint) but uses asyncio + AsyncOpenAI for client-side concurrency.

Why: the sync ThreadPoolExecutor version saturates around ~200 concurrent
in-flight requests because of synchronous-call bursts (rewrite then judge,
rewrite then judge, …). With async we can keep many more in-flight uniformly,
fully exploiting vLLM's continuous batching.

Schema, output records, env vars and CLI flags match enrich.py 1:1, plus
--concurrency replaces --workers and --resume is added.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from typing import Optional

# pull all the heavy lifting from the original enrich
from .enrich import (
    REWRITE_PROMPTS,
    JUDGE_SYSTEM,
    JUDGE_USER_TEMPLATE,
    LANGUAGES,
    _language_preamble,
    diversity_hint,
    wants_thinking,
    parse_rewrite,
    parse_judge,
    STORY_INCOMPATIBLE_CATEGORIES,
    _NON_NARRATIVE_MODES,
    load_env_file,
    _env_first,
)

from openai import AsyncOpenAI


def _key(rec_id, lang, prompt_name):
    return (rec_id, lang, prompt_name)


async def enrich_one_async(
    record: dict,
    *,
    client: AsyncOpenAI,
    model: str,
    sem: asyncio.Semaphore,
    prompt_pool: list[dict],
    rng: random.Random,
    languages: list[str],
    temperature: float = 0.8,
    max_tokens: int = 1500,
    do_judge: bool = True,
    thinking_policy: str = "off",
    thinking_max_tokens: int = 4096,
) -> dict:
    seed_problem = record["problem"]
    seed_answer = record["answer"]
    seed_meta = record.get("meta", {})
    seed_steps = seed_meta.get("step_text", "(no step text in seed)")

    cat = seed_meta.get("category", "")
    if cat in STORY_INCOMPATIBLE_CATEGORIES:
        safe = [p for p in prompt_pool if p["name"] in _NON_NARRATIVE_MODES]
        if safe:
            prompt_pool = safe

    weights = [p.get("weight", 1.0) for p in prompt_pool]
    prompt = rng.choices(prompt_pool, weights=weights, k=1)[0]
    user = prompt["user"].format(problem=seed_problem, answer=seed_answer, steps=seed_steps)
    user += diversity_hint(prompt["name"], rng)

    lang_code = rng.choice(languages)
    system_msg = prompt["system"] + _language_preamble(lang_code)
    use_thinking = wants_thinking(seed_meta, prompt["name"], thinking_policy)

    out = {
        "seed_id": record.get("id"),
        "seed_problem": seed_problem,
        "seed_answer": seed_answer,
        "seed_meta": seed_meta,
        "rewrite_prompt_name": prompt["name"],
        "rewrite_temperature": temperature,
        "rewrite_thinking_used": use_thinking,
        "rewrite_language": lang_code,
    }

    # 1) rewrite
    async with sem:
        try:
            extra_body = {}
            if use_thinking:
                extra_body = {"chat_template_kwargs": {"enable_thinking": True}}
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_msg},
                          {"role": "user", "content": user}],
                temperature=temperature,
                top_p=0.95,
                max_tokens=max(max_tokens, thinking_max_tokens) if use_thinking else max_tokens,
                extra_body=extra_body or None,
            )
            rewrite_raw = resp.choices[0].message.content or ""
        except Exception as e:
            out["error"] = f"rewrite: {str(e)[:200]}"
            return out

    out["rewrite_raw"] = rewrite_raw
    new_problem, new_solution = parse_rewrite(rewrite_raw)
    out["rewritten_problem"] = new_problem
    out["rewritten_solution"] = new_solution
    if not (new_problem and new_solution):
        out["judge"] = {"overall_keep": False, "reason": "no_problem_parsed"}
        return out

    if not do_judge:
        out["judge"] = {"overall_keep": True, "reason": "judge_disabled"}
        return out

    # 2) judge
    judge_user = JUDGE_USER_TEMPLATE.format(
        original_problem=seed_problem, original_answer=seed_answer,
        rewritten_problem=new_problem, rewritten_solution=new_solution,
    )
    async with sem:
        try:
            jresp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": judge_user}],
                temperature=0.1, top_p=1.0, max_tokens=512,
            )
            judge_raw = jresp.choices[0].message.content or ""
        except Exception as e:
            out["judge_raw"] = ""
            out["judge"] = {"overall_keep": False, "reason": f"judge_error: {str(e)[:120]}"}
            return out

    out["judge_raw"] = judge_raw
    out["judge"] = parse_judge(judge_raw) or {"overall_keep": False, "reason": "judge_parse_failed"}
    return out


async def main_async(args):
    # parse seed records
    print(f"[{time.strftime('%H:%M:%S')}] reading {args.input}", file=sys.stderr)
    records = []
    with open(args.input) as f:
        for i, line in enumerate(f):
            if args.n and i >= args.n:
                break
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if args.shard_id is not None and args.num_shards > 1:
        n_total = len(records)
        records = records[args.shard_id::args.num_shards]
        print(f"[{time.strftime('%H:%M:%S')}] sharded: {len(records)}/{n_total}", file=sys.stderr)

    # filter prompts
    if args.prompts != "all":
        wanted = set(p.strip() for p in args.prompts.split(","))
        prompt_pool = [p for p in REWRITE_PROMPTS if p["name"] in wanted]
        if not prompt_pool:
            sys.exit(f"--prompts={args.prompts} matched nothing")
    else:
        prompt_pool = REWRITE_PROMPTS

    languages = [c.strip() for c in args.languages.split(",") if c.strip()]
    print(f"[{time.strftime('%H:%M:%S')}] {len(records)} records | "
          f"prompts={[p['name'] for p in prompt_pool]} | langs={languages}",
          file=sys.stderr)

    # resume
    already_done = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    already_done.add(_key(r.get("seed_id"), r.get("rewrite_language"),
                                          r.get("rewrite_prompt_name")))
                except json.JSONDecodeError:
                    continue
        before = len(records)
        # we don't know rewrite_language/prompt_name a priori (random per record)
        # so for resume we just skip any record whose seed_id appears in already_done
        seen_seeds = {k[0] for k in already_done}
        records = [r for r in records if r.get("id") not in seen_seeds]
        print(f"[{time.strftime('%H:%M:%S')}] resume: skipping {before - len(records)} "
              f"already-done seed_ids", file=sys.stderr)

    if not records:
        print("Nothing to do.", file=sys.stderr); return

    rng_top = random.Random(args.seed)
    client = AsyncOpenAI(
        base_url=args.endpoint.rstrip("/") + "/v1",
        api_key=args.api_key or "EMPTY",
        timeout=args.timeout,
        max_retries=2,
    )
    # Per-call semaphore inside enrich_one_async — caps concurrent in-flight
    # *requests* (each record makes 2: rewrite + judge). With N workers each
    # making ≤2 in-flight, the semaphore size of 2*N is the right ceiling.
    sem = asyncio.Semaphore(args.concurrency * 2)

    n_done = n_kept = n_err = 0
    t0 = time.time()
    fout = open(args.output, "a" if args.resume else "w")
    fout_lock = asyncio.Lock()

    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 2)
    total = len(records)

    # ── Bounded worker pool (the fix): N workers pull from a queue. The
    # producer feeds the queue with backpressure, so we never have more than
    # ~2N coroutines + 2N pending requests in memory at once. The previous
    # asyncio.gather(*(go(r) for r in 2_000_000_records)) materialised 2 M
    # coroutines on a single event loop and crushed the scheduler.
    async def worker():
        nonlocal n_done, n_kept, n_err
        while True:
            rec = await queue.get()
            if rec is None:
                queue.task_done()
                return
            per_rng = random.Random(rng_top.randrange(1 << 31))
            try:
                out = await enrich_one_async(
                    rec, client=client, model=args.model, sem=sem,
                    prompt_pool=prompt_pool, rng=per_rng, languages=languages,
                    temperature=args.temperature, max_tokens=args.max_tokens,
                    do_judge=not args.no_judge,
                    thinking_policy=args.thinking,
                    thinking_max_tokens=args.thinking_max_tokens,
                )
            except Exception as e:
                out = {"seed_id": rec.get("id"), "error": str(e)[:200]}
                n_err += 1
            line = json.dumps(out, ensure_ascii=False) + "\n"
            async with fout_lock:
                fout.write(line)
                fout.flush()
            n_done += 1
            if out.get("judge", {}).get("overall_keep"):
                n_kept += 1
            if n_done % 50 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (total - n_done) / max(rate, 0.001)
                print(f"  {n_done}/{total} | kept {n_kept} | err {n_err} | "
                      f"{rate:.2f} rec/s | ETA {eta/60:.0f} min", file=sys.stderr)
            queue.task_done()

    async def producer():
        for rec in records:
            await queue.put(rec)
        # poison pills, one per worker
        for _ in range(args.concurrency):
            await queue.put(None)

    workers = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
    await producer()
    await asyncio.gather(*workers)
    fout.close()
    print(f"\n[{time.strftime('%H:%M:%S')}] DONE. {n_done} records, "
          f"{n_kept} kept ({100*n_kept/max(n_done,1):.1f}%), {n_err} errors",
          file=sys.stderr)
    print(f"  Output: {args.output}", file=sys.stderr)


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file")
    pre_args, _ = pre.parse_known_args()
    if pre_args.env_file:
        load_env_file(pre_args.env_file)

    env_endpoint = _env_first("MATH_GEN_VLLM_ENDPOINT", "OPENAI_BASE_URL", "OPENAI_API_BASE",
                              default="http://localhost:8000")
    env_model = _env_first("MATH_GEN_VLLM_MODEL", "OPENAI_MODEL", default=None)
    env_api_key = _env_first("MATH_GEN_VLLM_API_KEY", "OPENAI_API_KEY", default="EMPTY")
    env_timeout = int(_env_first("MATH_GEN_VLLM_TIMEOUT", default="600"))

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--env-file", default=None)
    p.add_argument("--endpoint", default=env_endpoint)
    p.add_argument("--model", default=env_model, required=env_model is None)
    p.add_argument("--api-key", default=env_api_key)
    p.add_argument("--timeout", type=int, default=env_timeout)
    p.add_argument("--n", type=int, default=0, help="cap input records (0 = all)")
    p.add_argument("--concurrency", type=int, default=512,
                   help="asyncio.Semaphore size (max in-flight requests)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=1500)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--prompts", default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--thinking", choices=["off", "auto", "on"], default="off")
    p.add_argument("--thinking-max-tokens", type=int, default=4096)
    p.add_argument("--languages", default="en")
    p.add_argument("--shard-id", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
