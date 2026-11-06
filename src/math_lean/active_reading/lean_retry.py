"""Repair-loop pass for Lean snippets that failed to compile.

Reads two files:
  * `--runs-input`   the AR pipeline output (one record per source theorem,
                     used to recover the source chunk for context)
  * `--check-input`  the lean_check JSONL (used for the snippet code +
                     verdict + Lean stdout)

For every snippet whose verdict is anything but `ok` / `ok_with_sorry` /
`timeout`, builds a repair prompt that gives the model:
  * the source theorem (statement + proofs)
  * the strategy that produced the original snippet
  * the original snippet
  * the Lean compiler error
…and asks for a corrected snippet only. The repaired snippet is then
re-checked through the same `lean_check.check_one` machinery.

Output JSONL contains, per failing snippet, the original code+verdict,
the repair prompt's raw response, the extracted new code, the new
verdict, and whether the verdict actually improved.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from .client import VLLMClient, env_first, DEFAULT_ENDPOINT, DEFAULT_TIMEOUT
from .lean_check import (
    check_one,
    DEFAULT_PROJECT, DEFAULT_LAKE,
    _FENCE_RE,
)


# ─────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────

REPAIR_SYSTEM = """You are a Lean 4 / mathlib4 expert who fixes compile errors. \
Given a source theorem, the study strategy that motivated a Lean snippet, \
the snippet itself, and the exact Lean compiler error, you output a single \
corrected Lean 4 snippet.

House rules for the snippet you produce:
* Toolchain: Lean 4 + mathlib4 master (post-mathlib3 rename).
* Imports: write `import Mathlib` at the top. Do NOT write sub-imports
  like `import Analysis.X` — they will be stripped.
* Names are mathlib4-cased: `Nat.succ`, `Real.log`, `MeasurableSpace`,
  `IsProbabilityMeasure` — never the snake_case mathlib3 forms.
* Tactics you can rely on: `rfl`, `simp`, `simp_all`, `linarith`,
  `nlinarith`, `omega`, `ring`, `ring_nf`, `field_simp`, `norm_num`,
  `decide`, `tauto`, `exact_mod_cast`, `push_neg`, `contradiction`.
* Use real mathlib4 lemma names — do not invent names. If you are not
  sure a name exists, restructure the proof to avoid needing it.
* `sorry` is NEVER acceptable as a shortcut to skip an error. Only
  acceptable when the original strategy is signature-only.
* Make the minimum change needed to compile. Do not change the theorem
  statement unless it is itself ill-formed.

Output format (strict):
<thinking>
Walk through, in 3-6 short bullet points:
- What does the Lean error tell us went wrong?
- Which mathlib lemma / tactic / type would actually work?
- What is the minimal change needed?
</thinking>

```lean
<corrected snippet>
```

Output nothing outside the <thinking> block and the ```lean fence."""

REPAIR_USER_TEMPLATE = """# Source theorem (ProofWiki)
{chunk}

# Study strategy that produced the snippet
{strategy_block}

# Original Lean snippet (failed to compile)
```lean
{code}
```

# Lean compiler error
```
{error}
```

# Task
Diagnose the failure inside <thinking>...</thinking>, then output the \
corrected Lean 4 snippet inside one ```lean ... ``` fence."""


# ─────────────────────────────────────────────────────────────────
# Indexing inputs
# ─────────────────────────────────────────────────────────────────

def index_runs(runs_path: str) -> dict[str, dict]:
    """doc_id → {chunk, title, kind, strategies: dict[str, str]}.

    `strategies` maps strategy name → full markdown body (the
    `## Strategy N: name\\n<paragraph>` block we originally generated).
    Used by the retry to give the model the full intent, not just a
    short label.
    """
    out = {}
    with open(runs_path) as f:
        for line in f:
            r = json.loads(line)
            strategies: dict[str, str] = {}
            for sr in r.get("strategy_responses", []):
                for s in sr.get("strategies", []):
                    # `raw` is the full ## block; fall back to body if missing
                    strategies.setdefault(s["name"], s.get("raw") or s.get("body") or "")
            out[r["id"]] = {
                "chunk": r.get("chunk", ""),
                "title": r.get("title", ""),
                "kind": r.get("kind", "theorem"),
                "strategies": strategies,
            }
    return out


def trim_error(stdout: str, max_chars: int = 1200) -> str:
    """Pull out the most useful part of the Lean compiler output.

    Lean dumps a long search path on import failure; we trim that to
    keep the prompt tight. Otherwise we just take the tail."""
    s = stdout.strip()
    if not s:
        return "(no diagnostic captured)"
    # If the path-search noise is present, slice from the first 'error:' line.
    m = re.search(r"^[^\n]*\berror:\b.*$", s, re.MULTILINE | re.IGNORECASE)
    if m:
        # keep from there to end, capped
        s = s[m.start():]
    return s[-max_chars:]


def select_failing(check_records: list[dict],
                   keep_verdicts: tuple[str, ...] = (
                       "parse_error", "unknown_identifier",
                       "type_error", "typeclass_error",
                       "unfinished_proof", "import_module_missing",
                       "import_misplaced", "truncated_decl",
                       "other", "other_unclassified")) -> list[dict]:
    """Pick check records worth retrying. Keeps the original verdict
    (or v2 if present) so the report can show transitions."""
    out = []
    for r in check_records:
        v = r.get("verdict_v2") or r.get("verdict")
        if v in ("ok", "ok_with_sorry", "timeout"):
            continue
        out.append(r)
    return out


# ─────────────────────────────────────────────────────────────────
# One-snippet repair
# ─────────────────────────────────────────────────────────────────

def _extract_code(text: str) -> Optional[str]:
    blocks = _FENCE_RE.findall(text)
    if blocks:
        return blocks[0].strip()
    # fallback: the model may have ignored the fence — accept anything that
    # has a `theorem`/`lemma`/`example`/`def` keyword in it
    if re.search(r"\b(theorem|lemma|example|def|instance|import)\b", text):
        return text.strip()
    return None


def repair_one(snippet: dict, runs_idx: dict[str, dict],
               client: VLLMClient, *, project: str, lake_bin: str,
               retry_temperature: float, retry_max_tokens: int,
               check_timeout_sec: int) -> dict:
    src = runs_idx.get(snippet["doc_id"])
    if not src:
        return {
            **snippet,
            "retry_skipped": "doc_id not found in runs index",
        }

    name = snippet.get("strategy_name") or "(unknown)"
    body = src["strategies"].get(name, "")
    strategy_block = body if body else f"## {name}\n(strategy body not available)"
    user = REPAIR_USER_TEMPLATE.format(
        chunk=src["chunk"],
        strategy_block=strategy_block,
        code=snippet["code"],
        error=trim_error(snippet.get("stdout") or ""),
    )

    t0 = time.time()
    try:
        raw = client.chat(REPAIR_SYSTEM, user,
                          temperature=retry_temperature,
                          max_tokens=retry_max_tokens)
    except Exception as e:
        return {
            **snippet,
            "retry_error": f"{type(e).__name__}: {e}",
            "retry_elapsed_sec": round(time.time() - t0, 2),
        }

    new_code = _extract_code(raw)
    if not new_code:
        return {
            **snippet,
            "retry_raw": raw,
            "retry_extracted": None,
            "retry_elapsed_sec": round(time.time() - t0, 2),
        }

    # Re-check the repaired snippet through the same machinery as v2.
    new_snip = {
        "doc_id": snippet["doc_id"],
        "title": snippet.get("title"),
        "strategy_name": snippet.get("strategy_name"),
        "snippet_index": snippet.get("snippet_index"),
        "code": new_code,
    }
    new_check = check_one(new_snip, project=project, lake_bin=lake_bin,
                          timeout_sec=check_timeout_sec)

    return {
        **snippet,
        "retry_raw": raw,
        "retry_extracted": new_code,
        "retry_check": new_check,
        "retry_verdict": new_check["verdict"],
        "retry_used_sorry": new_check["used_sorry"],
        "improved": (snippet.get("verdict_v2") or snippet.get("verdict")) not in
                    ("ok", "ok_with_sorry") and
                    new_check["verdict"] in ("ok", "ok_with_sorry"),
        "retry_elapsed_sec": round(time.time() - t0, 2),
    }


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-input", required=True,
                    help="AR pipeline output JSONL (e.g. proto_100.jsonl)")
    ap.add_argument("--check-input", required=True,
                    help="lean_check JSONL (with `verdict` and `stdout`)")
    ap.add_argument("--out", required=True,
                    help="JSONL output: one record per *retried* snippet")

    ap.add_argument("--limit", type=int, default=0,
                    help="only retry the first N failing snippets (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retry-temperature", type=float, default=0.3,
                    help="lower temperature than initial gen — focused repair")
    ap.add_argument("--retry-max-tokens", type=int, default=1536,
                    help="includes ~400 tokens of <thinking> before the snippet")
    ap.add_argument("--check-timeout-sec", type=int, default=60)

    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--lake", default=DEFAULT_LAKE)

    env_endpoint = env_first("MATH_GEN_VLLM_ENDPOINT", "OPENAI_BASE_URL",
                             "OPENAI_API_BASE", default=DEFAULT_ENDPOINT)
    env_model = env_first("MATH_GEN_VLLM_MODEL", "OPENAI_MODEL", default=None)
    env_api_key = env_first("MATH_GEN_VLLM_API_KEY", "OPENAI_API_KEY", default=None)
    ap.add_argument("--endpoint", default=env_endpoint)
    ap.add_argument("--model", default=env_model)
    ap.add_argument("--api-key", default=env_api_key)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = ap.parse_args()

    if not args.model:
        ap.error("--model required (or set MATH_GEN_VLLM_MODEL / OPENAI_MODEL)")

    runs_idx = index_runs(args.runs_input)
    print(f"indexed {len(runs_idx)} source documents from {args.runs_input}",
          flush=True)

    check_records = [json.loads(l) for l in open(args.check_input)]
    failing = select_failing(check_records)
    if args.limit:
        failing = failing[:args.limit]
    if not failing:
        sys.exit("no failing snippets to retry")
    print(f"retrying {len(failing)} failing snippets via {args.endpoint}",
          flush=True)

    client = VLLMClient(args.endpoint, args.model,
                        api_key=args.api_key, timeout=args.timeout)

    n_done = 0
    n_improved = 0
    n_no_extract = 0
    n_retry_err = 0
    t0 = time.time()
    with open(args.out, "w") as fout, \
         ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(repair_one, snip, runs_idx, client,
                      project=args.project, lake_bin=args.lake,
                      retry_temperature=args.retry_temperature,
                      retry_max_tokens=args.retry_max_tokens,
                      check_timeout_sec=args.check_timeout_sec): i
            for i, snip in enumerate(failing)
        }
        for fut in as_completed(futures):
            r = fut.result()
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if r.get("improved"):
                n_improved += 1
            if r.get("retry_extracted") is None and "retry_raw" in r:
                n_no_extract += 1
            if "retry_error" in r:
                n_retry_err += 1
            if n_done % 25 == 0 or n_done == len(failing):
                rate = n_done / max(time.time() - t0, 1e-3)
                print(f"  [{n_done}/{len(failing)}] "
                      f"improved={n_improved} no_extract={n_no_extract} "
                      f"errors={n_retry_err} ({rate:.2f}/s)", flush=True)

    print()
    print(f"wrote {n_done} retry records to {args.out}")
    print(f"improved (failed → ok/ok_with_sorry): {n_improved} / {n_done} "
          f"= {100 * n_improved / max(1, n_done):.1f}%")


if __name__ == "__main__":
    main()
