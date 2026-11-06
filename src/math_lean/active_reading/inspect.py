"""Render a random sample of Active Reading augmentations as markdown.

Default: 10 augmentations stratified by flavour (5 agnostic + 5 lean),
chosen across distinct source documents so you don't see the same
theorem twice.

Examples
--------
    python -m math_lean.active_reading.inspect \\
        --input data/active_reading/proto_100.jsonl

    # 20 lean-only, no stratification, fixed seed
    python -m math_lean.active_reading.inspect \\
        --input data/active_reading/proto_100.jsonl \\
        --n 20 --flavour lean --seed 42

    # write to a markdown file for offline reading
    python -m math_lean.active_reading.inspect \\
        --input data/active_reading/proto_100.jsonl \\
        --n 30 --out /tmp/ar_sample.md
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict


def _flatten(jsonl_path: str) -> list[dict]:
    """Yield one row per (document, augmentation) with surrounding context."""
    rows = []
    for line in open(jsonl_path):
        rec = json.loads(line)
        for a in rec.get("augmentations", []):
            if a.get("error") or not a.get("artifact"):
                continue
            rows.append({
                "doc_id": rec["id"],
                "title": rec["title"],
                "kind": rec.get("kind"),
                "wikilinks": rec.get("wikilinks", []),
                "categories": rec.get("categories", []),
                "chunk": rec.get("chunk", ""),
                "flavour": a["flavour"],
                "strategy_name": a["strategy_name"],
                "artifact": a["artifact"],
            })
    return rows


def _sample(rows: list[dict], n: int, *, flavour: str | None,
            stratified: bool, distinct_docs: bool, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pool = rows if not flavour else [r for r in rows if r["flavour"] == flavour]
    if not pool:
        sys.exit(f"no rows match flavour={flavour!r}")

    if stratified and not flavour:
        per: dict[str, list[dict]] = defaultdict(list)
        for r in pool:
            per[r["flavour"]].append(r)
        # equal split rounded up; if n is odd, agnostic gets the extra
        out: list[dict] = []
        flavours = sorted(per.keys())
        share = max(1, n // len(flavours))
        for f in flavours:
            cand = per[f][:]
            rng.shuffle(cand)
            taken: list[dict] = []
            seen_docs: set[str] = set()
            for r in cand:
                if distinct_docs and r["doc_id"] in seen_docs:
                    continue
                taken.append(r)
                seen_docs.add(r["doc_id"])
                if len(taken) >= share:
                    break
            out.extend(taken)
        # fill any remainder from full pool
        if len(out) < n:
            rest = [r for r in pool if r not in out]
            rng.shuffle(rest)
            out.extend(rest[:n - len(out)])
        return out[:n]

    # plain sample
    cand = pool[:]
    rng.shuffle(cand)
    out: list[dict] = []
    seen_docs: set[str] = set()
    for r in cand:
        if distinct_docs and r["doc_id"] in seen_docs:
            continue
        out.append(r)
        seen_docs.add(r["doc_id"])
        if len(out) >= n:
            break
    return out


def render(rows: list[dict], *, show_chunk: bool, chunk_chars: int) -> str:
    out: list[str] = []
    for i, r in enumerate(rows, 1):
        out.append(f"# [{i}] {r['title']}  *(flavour: `{r['flavour']}`)*")
        out.append(f"- **id**: `{r['doc_id']}`  &middot; **kind**: {r['kind']}")
        out.append(f"- **strategy**: {r['strategy_name']}")
        if r.get("categories"):
            out.append(f"- **categories**: {', '.join(r['categories'][:6])}"
                       + (" …" if len(r["categories"]) > 6 else ""))
        if r.get("wikilinks"):
            out.append(f"- **wikilinks**: {', '.join(r['wikilinks'][:8])}"
                       + (" …" if len(r["wikilinks"]) > 8 else ""))
        out.append("")
        if show_chunk:
            chunk = r["chunk"][:chunk_chars]
            ellipsis = " …" if len(r["chunk"]) > chunk_chars else ""
            out.append("## Source (truncated)")
            out.append("```")
            out.append(chunk + ellipsis)
            out.append("```")
            out.append("")
        out.append("## Artifact")
        out.append(r["artifact"])
        out.append("")
        out.append("---")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(
        description="Render Active Reading augmentations as markdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--input", required=True,
                    help="JSONL produced by `python -m math_lean.active_reading`")
    ap.add_argument("--n", type=int, default=10,
                    help="number of augmentations to sample (default 10)")
    ap.add_argument("--flavour", choices=("agnostic", "lean"), default=None,
                    help="restrict to one flavour")
    ap.add_argument("--no-stratified", action="store_true",
                    help="don't force an even split across flavours")
    ap.add_argument("--allow-repeat-docs", action="store_true",
                    help="don't require distinct source documents")
    ap.add_argument("--show-chunk", action="store_true",
                    help="include a truncated source chunk above each artifact")
    ap.add_argument("--chunk-chars", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="write to this file instead of stdout")
    args = ap.parse_args()

    rows = _flatten(args.input)
    if not rows:
        sys.exit(f"no augmentations found in {args.input}")
    sample = _sample(rows, args.n,
                     flavour=args.flavour,
                     stratified=not args.no_stratified,
                     distinct_docs=not args.allow_repeat_docs,
                     seed=args.seed)
    md = render(sample, show_chunk=args.show_chunk, chunk_chars=args.chunk_chars)

    if args.out:
        with open(args.out, "w") as f:
            f.write(md)
        print(f"Wrote {len(sample)} augmentations to {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(md)


if __name__ == "__main__":
    main()
