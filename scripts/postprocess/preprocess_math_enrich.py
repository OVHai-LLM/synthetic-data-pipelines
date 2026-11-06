"""preprocess_math_enrich.py

Filter the 30 multilingual math_enrich shards by judge.overall_keep and
convert to standard annealing format ({text, id, metadata{token_count,source,
language,prompt_name,...}}). Tokenize with Qwen.

Reads:  data/maths/math_enrich/enrich_<lang>.shard{0..3}.jsonl   (30 langs)
Writes: data/maths/math_enrich_filtered/enrich_<lang>.jsonl       (one per lang)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

# ── Tokenizer (lazy per-worker) ───────────────────────────────────
_TOK = None
def _get_tok(name="Qwen/Qwen2.5-7B-Instruct"):
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(name)
    return _TOK


def process_lang(args):
    lang, shards, out_path, tok_name = args
    tok = _get_tok(tok_name)
    n_in = n_kept = n_drop = n_err = n_tokens = 0
    with open(out_path, "w") as fout:
        for sh in shards:
            with open(sh) as f:
                for line in f:
                    n_in += 1
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        n_err += 1; continue
                    if d.get("error"):
                        n_err += 1; continue
                    if not d.get("judge", {}).get("overall_keep"):
                        n_drop += 1; continue

                    problem = d.get("rewritten_problem") or ""
                    solution = d.get("rewritten_solution") or ""
                    if not problem.strip() or not solution.strip():
                        n_drop += 1; continue

                    text = f"Problem: {problem}\n\nSolution:\n{solution}"
                    n_tok = len(tok.encode(text, add_special_tokens=False))
                    n_tokens += n_tok
                    out = {
                        "text": text,
                        "id": d.get("seed_id") or "enrich:unknown",
                        "metadata": {
                            "token_count": n_tok,
                            "source": "math_enrich",
                            "language": d.get("rewrite_language", lang),
                            "prompt_name": d.get("rewrite_prompt_name"),
                            "seed_category": d.get("seed_meta", {}).get("category"),
                            "seed_difficulty": d.get("seed_meta", {}).get("difficulty"),
                        },
                    }
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    n_kept += 1
    return lang, n_in, n_kept, n_drop, n_err, n_tokens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",
        default=os.environ.get("ANNEAL_ROOT", "data") + "/data/maths/math_enrich")
    p.add_argument("--output-dir",
        default=os.environ.get("ANNEAL_ROOT", "data") + "/data/maths/math_enrich_filtered")
    p.add_argument("--tok-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--workers", type=int, default=min(30, max(1, cpu_count() // 4)),
                   help="parallel langs (each lang = 4 shards processed sequentially in 1 worker)")
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # group input shards by lang (extract from filename: enrich_<lang>.shard<X>.jsonl)
    by_lang = {}
    for sh in sorted(glob.glob(f"{args.input_dir}/enrich_*.shard*.jsonl")):
        lang = Path(sh).name.replace("enrich_", "").split(".")[0]
        by_lang.setdefault(lang, []).append(sh)
    print(f"[{time.strftime('%H:%M:%S')}] {len(by_lang)} langs, "
          f"{sum(len(v) for v in by_lang.values())} shards total", file=sys.stderr)

    # pre-warm tokenizer in main proc to populate HF cache
    _get_tok(args.tok_name)

    jobs = [(lang, sorted(shards), f"{args.output_dir}/enrich_{lang}.jsonl", args.tok_name)
            for lang, shards in sorted(by_lang.items())]

    grand_in = grand_kept = grand_drop = grand_err = grand_tokens = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(process_lang, j) for j in jobs]):
            lang, n_in, n_kept, n_drop, n_err, n_tokens = fut.result()
            print(f"  [{time.strftime('%H:%M:%S')}] {lang}: in={n_in}, kept={n_kept} "
                  f"({100*n_kept/max(n_in,1):.1f}%), drop={n_drop}, err={n_err}, "
                  f"{n_tokens/1e6:.1f} M tok", file=sys.stderr)
            grand_in += n_in
            grand_kept += n_kept
            grand_drop += n_drop
            grand_err += n_err
            grand_tokens += n_tokens

    el = time.time() - t0
    print(f"\n[{time.strftime('%H:%M:%S')}] DONE in {el/60:.1f} min", file=sys.stderr)
    print(f"  total in:    {grand_in:,}", file=sys.stderr)
    print(f"  kept:        {grand_kept:,} ({100*grand_kept/max(grand_in,1):.1f}%)", file=sys.stderr)
    print(f"  dropped:     {grand_drop:,}", file=sys.stderr)
    print(f"  errors:      {grand_err:,}", file=sys.stderr)
    print(f"  total tokens: {grand_tokens:,} ({grand_tokens/1e9:.3f} B)", file=sys.stderr)
    print(f"  output: {args.output_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
