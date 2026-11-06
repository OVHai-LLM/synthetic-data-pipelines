"""Produce a pretraining-ready JSONL from an Active Reading run.

Filtering policy (MEDIUM, the default):

  * agnostic-flavour artifacts: kept unconditionally
  * lean-flavour artifacts: kept iff every ```lean``` block in the
    artifact is verdict `ok` or `ok_with_sorry`. `sorry` is allowed
    because signature-drafting strategies legitimately use it as a
    placeholder for the proof body, and the surrounding prose still
    teaches the formalisation move.

Output JSONL (one record per kept artifact):

    {
      "doc_id":        "proofwiki:Pythagorean_Theorem",
      "title":         "...",
      "kind":          "theorem",
      "categories":    [...],
      "wikilinks":     [...],
      "flavour":       "agnostic" | "lean",
      "strategy_name": "Dependency Mapping",
      "strategy_body": "<full ## block>",
      "source_chunk":  "<statement + proofs>",
      "artifact":      "<the markdown training example>",
      "lean_block_verdicts": ["ok", "ok_with_sorry", ...]   # only for lean
    }

The pretraining loader is expected to flatten this with:

    text = (
        f"# {title}\\n\\n"
        f"## Source\\n{source_chunk}\\n\\n"
        f"## Strategy\\n{strategy_body}\\n\\n"
        f"## Artifact\\n{artifact}"
    )

(or whatever framing your pretraining pipeline already uses).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

_FENCE_RE = re.compile(r"```(?:lean4?|Lean4?)\n(.*?)```", re.DOTALL)


def load_verdicts(check_path: Optional[str]) -> dict[tuple, str]:
    """Read a lean_check JSONL → {(doc_id, strategy_name, snippet_index): verdict}."""
    if not check_path:
        return {}
    out = {}
    with open(check_path) as f:
        for line in f:
            r = json.loads(line)
            key = (r["doc_id"], r.get("strategy_name"), r["snippet_index"])
            out[key] = r["verdict"]
    return out


def keep_lean_artifact(doc_id: str, strategy_name: str, artifact: str,
                       verdicts: dict[tuple, str], policy: str) -> tuple[bool, list[str]]:
    """Return (keep, list of per-block verdicts)."""
    n = len(_FENCE_RE.findall(artifact))
    block_v = [verdicts.get((doc_id, strategy_name, i), "no_verdict") for i in range(n)]
    if policy == "strict":
        keep = n >= 1 and all(v == "ok" for v in block_v)
    elif policy == "medium":
        # n==0 means the lean-flavour artifact has no Lean blocks at all
        # (prose-only: dependency map, lemma list, etc.). Keep it.
        if n == 0:
            keep = True
        else:
            keep = all(v in ("ok", "ok_with_sorry") for v in block_v)
    elif policy == "loose":
        keep = n == 0 or any(v in ("ok", "ok_with_sorry") for v in block_v)
    elif policy == "all":
        keep = True
    else:
        sys.exit(f"unknown policy: {policy}")
    return keep, block_v


def build(runs_path: str, *, lean_check_paths: Iterable[str], out_path: str,
          policy: str = "medium",
          only_flavours: Optional[set[str]] = None) -> dict:
    verdicts: dict[tuple, str] = {}
    for p in lean_check_paths:
        verdicts.update(load_verdicts(p))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    counters = Counter()
    written = 0
    chars_kept = 0

    with open(runs_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            doc_id = rec["id"]
            title = rec.get("title", "")
            kind = rec.get("kind", "theorem")
            chunk = rec.get("chunk", "")
            cats = rec.get("categories", [])
            wlinks = rec.get("wikilinks", [])

            # Index strategy bodies by name (for downstream prompt reconstruction)
            strategy_bodies: dict[str, str] = {}
            for sr in rec.get("strategy_responses", []):
                for s in sr.get("strategies", []):
                    strategy_bodies.setdefault(s["name"], s.get("raw") or s.get("body", ""))

            for a in rec.get("augmentations", []):
                if a.get("error") or not a.get("artifact"):
                    counters["skip_error"] += 1
                    continue
                flavour = a["flavour"]
                if only_flavours is not None and flavour not in only_flavours:
                    counters[f"filtered_out_{flavour}"] += 1
                    continue
                strategy_name = a["strategy_name"]
                artifact = a["artifact"]
                counters[f"total_{flavour}"] += 1

                if flavour == "lean":
                    keep, block_v = keep_lean_artifact(
                        doc_id, strategy_name, artifact, verdicts, policy)
                    if not keep:
                        counters[f"drop_{flavour}"] += 1
                        continue
                else:
                    block_v = []

                out = {
                    "doc_id": doc_id,
                    "title": title,
                    "kind": kind,
                    "categories": cats,
                    "wikilinks": wlinks,
                    "flavour": flavour,
                    "strategy_name": strategy_name,
                    "strategy_body": strategy_bodies.get(strategy_name, ""),
                    "source_chunk": chunk,
                    "artifact": artifact,
                }
                if flavour == "lean":
                    out["lean_block_verdicts"] = block_v
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                written += 1
                chars_kept += len(artifact)
                counters[f"keep_{flavour}"] += 1

    return {
        "written": written,
        "policy": policy,
        "kept_chars": chars_kept,
        "kept_tokens_approx": chars_kept // 4,
        "counters": dict(counters),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-input", required=True,
                    help="AR pipeline JSONL (e.g. proto_100.jsonl or proto_100_v4_leanonly.jsonl)")
    ap.add_argument("--lean-check", action="append", default=[],
                    help="path to a lean_check JSONL (verdicts). Pass once per "
                         "file — typically the final composed verdicts (e.g. "
                         "proto_100_v4_leanonly.jsonl.lean_check_v5_final.jsonl). "
                         "Pass --lean-check multiple times to merge across files.")
    ap.add_argument("--out", required=True, help="pretraining JSONL output path")
    ap.add_argument("--policy", choices=("strict", "medium", "loose", "all"),
                    default="medium",
                    help="lean-artifact filter policy (default: medium)")
    ap.add_argument("--also-runs", action="append", default=[],
                    help="additional AR pipeline JSONL files to merge (e.g. an "
                         "agnostic-flavour run alongside a lean-only run). "
                         "Records from later files overwrite earlier on doc_id.")
    ap.add_argument("--only-flavour", action="append", default=[],
                    help="when given, keep only artifacts of this flavour. "
                         "Pass once per allowed flavour. Applies to ALL inputs.")
    args = ap.parse_args()

    if not args.lean_check:
        print("WARNING: no --lean-check passed; lean-flavour artifacts will all be "
              "treated as 'no_verdict' and (under policy=medium) kept only if they "
              "have no Lean blocks. Pass --lean-check <verdict_jsonl>.", file=sys.stderr)

    inputs = [args.runs_input] + args.also_runs
    summary = {"per_file": []}
    total = Counter()
    written = 0
    chars = 0

    only_flavours = set(args.only_flavour) if args.only_flavour else None

    if len(inputs) == 1:
        s = build(args.runs_input, lean_check_paths=args.lean_check,
                  out_path=args.out, policy=args.policy,
                  only_flavours=only_flavours)
        print(json.dumps(s, indent=2))
        return

    # multi-file: write to temp files then concat
    import tempfile, shutil
    tmpdir = tempfile.mkdtemp(prefix="ar_pretrain_")
    for i, p in enumerate(inputs):
        tp = f"{tmpdir}/part_{i}.jsonl"
        s = build(p, lean_check_paths=args.lean_check,
                  out_path=tp, policy=args.policy,
                  only_flavours=only_flavours)
        s["from"] = p
        summary["per_file"].append(s)
        total += Counter(s["counters"])
        written += s["written"]
        chars += s["kept_chars"]

    # concat into args.out
    with open(args.out, "w") as fout:
        for i in range(len(inputs)):
            with open(f"{tmpdir}/part_{i}.jsonl") as fin:
                shutil.copyfileobj(fin, fout)
    shutil.rmtree(tmpdir)

    summary["total_written"] = written
    summary["total_kept_chars"] = chars
    summary["total_kept_tokens_approx"] = chars // 4
    summary["combined_counters"] = dict(total)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
