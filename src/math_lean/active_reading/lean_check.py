"""Compile-check the `lean`-flavoured artifacts in an Active Reading run.

Pulls every ```lean / ```lean4 fenced block out of `lean`-flavour
augmentations, wraps each one in a minimal mathlib4 preamble, and
runs `lake env lean` on it inside a stub project. Tallies one of:

    ok                — exit 0, no `sorry`
    ok_with_sorry     — exit 0, `sorry` present
    parse_error       — Lean parser failure
    elab_error        — type-check / elaboration / unknown-identifier error
    timeout           — exceeded per-snippet wall budget
    other             — non-zero exit but no recognised marker

Outputs:
* a per-snippet JSONL alongside the input (`<input>.lean_check.jsonl`)
* a one-line summary written to stdout

Use the env vars `AR_LEAN_PROJECT` / `AR_LEAN_LAKE_BIN` to override the
stub project path and the `lake` binary respectively.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_PROJECT = os.environ.get("AR_LEAN_PROJECT", "/tmp/ar_lean_check")
DEFAULT_LAKE = os.environ.get("AR_LEAN_LAKE_BIN",
                              os.path.expanduser("~/.elan/bin/lake"))

_FENCE_RE = re.compile(r"```(?:lean4?|Lean4?)\n(.*?)```", re.DOTALL)
_HAS_IMPORT_RE = re.compile(r"^\s*import\s+\S", re.MULTILINE)
_IMPORT_LINE_RE = re.compile(r"^\s*import\s+(\S+)", re.MULTILINE)
_HAS_SORRY_RE = re.compile(r"\bsorry\b")

# Roots that we trust the model to import directly. Anything else is
# either a renamed-since-mathlib3 path (`Analysis.X` instead of
# `Mathlib.Analysis.X`) or a typo, so we rewrite it.
_KNOWN_IMPORT_ROOTS = ("Mathlib", "Init", "Std", "Lean", "Batteries",
                       "Aesop", "Plausible", "ImportGraph", "ProofWidgets",
                       "Qq", "LeanSearchClient")


# ─────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────

def iter_lean_snippets(jsonl_path: str):
    """Yield (doc_id, title, strategy_name, snippet_index, code) tuples."""
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            for a in rec.get("augmentations", []):
                if a.get("flavour") != "lean" or a.get("error"):
                    continue
                code_blocks = _FENCE_RE.findall(a.get("artifact") or "")
                for idx, code in enumerate(code_blocks):
                    yield {
                        "doc_id": rec["id"],
                        "title": rec["title"],
                        "strategy_name": a.get("strategy_name"),
                        "snippet_index": idx,
                        "code": code.strip(),
                    }


# ─────────────────────────────────────────────────────────────────
# One snippet → one lake invocation
# ─────────────────────────────────────────────────────────────────

def _rewrite_imports(code: str) -> tuple[str, dict]:
    """Repair the model's import lines.

    Drops every `import` line whose first path component is not in
    `_KNOWN_IMPORT_ROOTS`, and records each drop. We then unconditionally
    prepend `import Mathlib` (which re-exports the full library).
    Returns (rewritten_code, repair_stats).
    """
    body_lines = []
    dropped: list[str] = []
    rewritten: list[str] = []
    for line in code.splitlines():
        m = _IMPORT_LINE_RE.match(line)
        if not m:
            body_lines.append(line)
            continue
        path = m.group(1).rstrip(";")
        root = path.split(".", 1)[0]
        if root in _KNOWN_IMPORT_ROOTS:
            # Skip the model's `import Mathlib` (we add our own); keep others.
            if path != "Mathlib":
                rewritten.append(line.strip())
            continue
        # unknown root → drop it; record so we can audit
        dropped.append(path)
    new = "import Mathlib\n"
    if rewritten:
        new += "\n".join(rewritten) + "\n"
    new += "\n" + "\n".join(body_lines)
    return new, {"dropped_imports": dropped}


def _wrap(code: str) -> tuple[str, dict]:
    """Force a known-good `import Mathlib` preamble; repair model imports."""
    return _rewrite_imports(code)


def _classify(returncode: int, stderr: str, used_sorry: bool) -> str:
    if returncode == 0:
        return "ok_with_sorry" if used_sorry else "ok"
    s = stderr.lower()
    if "unexpected token" in s or "expected" in s and "got" in s:
        return "parse_error"
    if any(m in s for m in ("unknown identifier", "type mismatch",
                            "function expected", "elaboration",
                            "synthesize", "failed to synth")):
        return "elab_error"
    return "other"


def check_one(snippet: dict, *, project: str, lake_bin: str,
              timeout_sec: int) -> dict:
    code, repair_stats = _wrap(snippet["code"])
    used_sorry = bool(_HAS_SORRY_RE.search(snippet["code"]))

    # Each call gets its own temp .lean inside the project so lake
    # finds the toolchain. We do NOT touch ArLeanCheck/ — keep the
    # main lib clean.
    fd, path = tempfile.mkstemp(prefix="snip_", suffix=".lean", dir=project)
    os.close(fd)
    Path(path).write_text(code)

    t0 = time.time()
    try:
        proc = subprocess.run(
            [lake_bin, "env", "lean", path],
            cwd=project,
            capture_output=True, text=True,
            timeout=timeout_sec,
        )
        result = {
            "verdict": _classify(proc.returncode, proc.stderr, used_sorry),
            "returncode": proc.returncode,
            "stderr": proc.stderr[-2000:],  # cap
            "stdout": proc.stdout[-500:],
        }
    except subprocess.TimeoutExpired:
        result = {"verdict": "timeout", "returncode": None,
                  "stderr": "", "stdout": ""}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    return {
        **snippet,
        **result,
        **repair_stats,
        "elapsed_sec": round(time.time() - t0, 2),
        "used_sorry": used_sorry,
    }


# ─────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────

def run(input_path: str, *, project: str, lake_bin: str,
        workers: int, timeout_sec: int, limit: int = 0,
        out_path: str | None = None) -> dict:
    if not Path(project).exists():
        sys.exit(f"stub project not found: {project} "
                 f"(set AR_LEAN_PROJECT or pass --project)")
    if not shutil.which(lake_bin) and not Path(lake_bin).exists():
        sys.exit(f"lake binary not found: {lake_bin} "
                 f"(set AR_LEAN_LAKE_BIN or pass --lake)")

    snippets = list(iter_lean_snippets(input_path))
    if limit:
        snippets = snippets[:limit]
    if not snippets:
        sys.exit("no lean snippets found in input")

    if out_path is None:
        out_path = input_path + ".lean_check.jsonl"

    print(f"checking {len(snippets)} snippets against {project} "
          f"(workers={workers}, timeout={timeout_sec}s)", flush=True)

    verdicts: dict[str, int] = {}
    results: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(check_one, s, project=project, lake_bin=lake_bin,
                      timeout_sec=timeout_sec): i
            for i, s in enumerate(snippets)
        }
        for done_count, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
            if done_count % 25 == 0 or done_count == len(snippets):
                rate = done_count / max(time.time() - t0, 1e-3)
                print(f"  [{done_count}/{len(snippets)}] "
                      + " ".join(f"{k}={v}" for k, v in sorted(verdicts.items()))
                      + f" ({rate:.2f}/s)", flush=True)

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote per-snippet results to {out_path}")

    summary = {
        "n_snippets": len(snippets),
        "verdicts": verdicts,
        "pct": {k: round(100 * v / len(snippets), 1)
                for k, v in verdicts.items()},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="proto_100.jsonl from the AR pipeline")
    ap.add_argument("--out", default=None,
                    help="per-snippet output JSONL (default: <input>.lean_check.jsonl)")
    ap.add_argument("--project", default=DEFAULT_PROJECT,
                    help=f"path to stub mathlib4 project (default {DEFAULT_PROJECT})")
    ap.add_argument("--lake", default=DEFAULT_LAKE,
                    help=f"lake binary (default {DEFAULT_LAKE})")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel snippet checks (lake env is lightweight)")
    ap.add_argument("--timeout-sec", type=int, default=45,
                    help="per-snippet wall budget")
    ap.add_argument("--limit", type=int, default=0,
                    help="only check the first N snippets (0 = all)")
    args = ap.parse_args()

    summary = run(args.input, project=args.project, lake_bin=args.lake,
                  workers=args.workers, timeout_sec=args.timeout_sec,
                  limit=args.limit, out_path=args.out)
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
