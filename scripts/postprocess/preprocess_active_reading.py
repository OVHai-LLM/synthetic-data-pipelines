"""preprocess_active_reading.py

Convert active_reading.jsonl to standard annealing format ({text,id,metadata}).
Each input record has up to ~10 augmentations[*].artifact strings — one per
strategy/flavour. Each artifact becomes ONE training doc (so ~18.5K input
records → ~150K-180K training docs).

Reads:  $ANNEAL_ROOT/proofwiki/active_reading.jsonl
Writes: $ANNEAL_ROOT/data/maths/active_reading_extracted/
        active_reading_extracted.jsonl
"""
from __future__ import annotations
import os

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

_TOK = None
def _get_tok(name="Qwen/Qwen2.5-7B-Instruct"):
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(name)
    return _TOK


def process_chunk(args):
    chunk_id, lines, tok_name = args
    tok = _get_tok(tok_name)
    out = []
    n_tokens = 0
    for line in lines:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        rec_id = d.get("id", "")
        title = d.get("title", "")
        kind = d.get("kind", "theorem")
        for i, aug in enumerate(d.get("augmentations", [])):
            artifact = aug.get("artifact") or ""
            if aug.get("error") or not artifact.strip() or len(artifact) < 80:
                continue
            n_tok = len(tok.encode(artifact, add_special_tokens=False))
            n_tokens += n_tok
            out.append({
                "text": artifact,
                "id": f"{rec_id}#aug{i}",
                "metadata": {
                    "token_count": n_tok,
                    "source": "active_reading",
                    "kind": kind,
                    "flavour": aug.get("flavour"),
                    "strategy_name": aug.get("strategy_name"),
                    "title": title,
                },
            })
    return chunk_id, out, n_tokens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",
        default=os.environ.get("ANNEAL_ROOT", "data") + "/proofwiki/active_reading.jsonl")
    p.add_argument("--output",
        default=os.environ.get("ANNEAL_ROOT", "data") + "/data/maths/active_reading_extracted/active_reading_extracted.jsonl")
    p.add_argument("--tok-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--workers", type=int, default=min(16, max(1, cpu_count() // 8)))
    p.add_argument("--chunk-size", type=int, default=500)
    args = p.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] reading {args.input}", file=sys.stderr)
    with open(args.input) as f:
        lines = f.readlines()
    print(f"  {len(lines)} input records", file=sys.stderr)

    chunks = [lines[i:i+args.chunk_size] for i in range(0, len(lines), args.chunk_size)]
    print(f"  {len(chunks)} chunks × {args.chunk_size}, {args.workers} workers", file=sys.stderr)

    _get_tok(args.tok_name)  # pre-warm in main proc

    n_done = n_kept = total_tokens = 0
    t0 = time.time()
    with open(args.output, "w") as fout, \
         ProcessPoolExecutor(max_workers=args.workers) as ex:
        jobs = [(i, chunks[i], args.tok_name) for i in range(len(chunks))]
        for chunk_id, out, n_tok in ex.map(process_chunk, jobs):
            for r in out:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_kept += len(out)
            total_tokens += n_tok
            n_done += 1
            if n_done % 10 == 0:
                el = time.time() - t0
                print(f"  {n_done}/{len(chunks)} chunks | kept {n_kept} docs | "
                      f"{total_tokens/1e6:.1f} M tok | {n_kept/el:.0f} docs/s",
                      file=sys.stderr)

    el = time.time() - t0
    print(f"\n[{time.strftime('%H:%M:%S')}] DONE in {el/60:.1f} min", file=sys.stderr)
    print(f"  raw input records: {len(lines)}", file=sys.stderr)
    print(f"  docs produced: {n_kept}", file=sys.stderr)
    print(f"  total Qwen tokens: {total_tokens:,} ({total_tokens/1e6:.1f} M = {total_tokens/1e9:.3f} B)", file=sys.stderr)
    print(f"  output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
