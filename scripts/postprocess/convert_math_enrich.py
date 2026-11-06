"""
convert_math_enrich.py

Convert math_enrich/*.jsonl (LLM-rewritten math problems with judge scores) into
the standard annealing-corpus jsonl format with text / id / metadata{token_count, ...}.

Source schema fields:
  seed_id, seed_problem, seed_answer, seed_meta{difficulty, category, generator, ...}
  rewrite_prompt_name, rewrite_temperature, rewrite_thinking_used
  rewritten_problem, rewritten_solution
  rewrite_raw, judge_raw
  judge{correctness, naturalness, clarity, difficulty_preservation, overall_keep, ...}

Output schema (per doc):
  text     : "Problem: ...\n\nSolution: ..." formatted document
  id       : seed_id
  metadata : { token_count: int, category, difficulty, generator, rewrite_prompt,
               judge_correctness, naturalness, clarity, difficulty_preservation,
               source: "math_enrich" }

Filter: only docs with judge.overall_keep == True (~81 % of input).
"""

import os
import json
import sys
from pathlib import Path
import click
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count


def format_text(d):
    prob = d.get("rewritten_problem", "").strip()
    sol = d.get("rewritten_solution", "").strip()
    if not prob:
        return None
    if sol:
        return f"Problem: {prob}\n\nSolution: {sol}"
    return f"Problem: {prob}"


def convert_one(args):
    """Worker: read one source jsonl, write one converted jsonl. Returns (in_count, out_count, tokens)."""
    src, dst, tokenizer_name = args
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    in_count = 0
    out_count = 0
    total_tokens = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            in_count += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            j = d.get("judge", {})
            if not j or not j.get("overall_keep"):
                continue
            text = format_text(d)
            if not text:
                continue
            n_tokens = len(tok.encode(text, add_special_tokens=False))
            seed_meta = d.get("seed_meta", {}) or {}
            out = {
                "text": text,
                "id": d.get("seed_id"),
                "metadata": {
                    "token_count": n_tokens,
                    "source": "math_enrich",
                    "category": seed_meta.get("category"),
                    "difficulty": seed_meta.get("difficulty"),
                    "generator": seed_meta.get("generator"),
                    "rewrite_prompt": d.get("rewrite_prompt_name"),
                    "rewrite_temperature": d.get("rewrite_temperature"),
                    "judge_correctness": j.get("correctness"),
                    "judge_naturalness": j.get("naturalness"),
                    "judge_clarity": j.get("clarity"),
                    "judge_difficulty_preservation": j.get("difficulty_preservation"),
                },
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            out_count += 1
            total_tokens += n_tokens
    return src, in_count, out_count, total_tokens


@click.command()
@click.option("--input-dir", type=str,
              default=os.environ.get("ANNEAL_ROOT", "data") + "/math_enrich",
              help="Source dir containing math_enrich_*.jsonl")
@click.option("--output-dir", type=str,
              default=os.environ.get("ANNEAL_ROOT", "data") + "/data/maths/math_enrich",
              help="Output dir for converted jsonl files (will be created)")
@click.option("--tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct",
              help="HF tokenizer name for token_count.")
def main(input_dir, output_dir, tokenizer):
    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    src_files = sorted(inp.glob("math_enrich_*.jsonl"))
    if not src_files:
        print(f"No math_enrich_*.jsonl files in {inp}")
        sys.exit(1)

    jobs = []
    for sf in src_files:
        df = out / sf.name.replace(".jsonl", "_qwen.jsonl")
        jobs.append((str(sf), str(df), tokenizer))

    print(f"Converting {len(src_files)} files → {output_dir}")
    print(f"Workers: {min(len(jobs), cpu_count() // 2)}")
    print()

    grand_in = grand_out = grand_tok = 0
    with ProcessPoolExecutor(max_workers=min(len(jobs), cpu_count() // 2)) as ex:
        for src, n_in, n_out, n_tok in ex.map(convert_one, jobs):
            grand_in += n_in
            grand_out += n_out
            grand_tok += n_tok
            print(f"  {Path(src).name}: in={n_in:,} | out={n_out:,} ({100*n_out/max(n_in,1):.0f}%) | tokens={n_tok/1e6:.1f}M")

    print()
    print(f"TOTAL: {grand_in:,} in → {grand_out:,} out ({100*grand_out/max(grand_in,1):.0f}%) | {grand_tok/1e6:.1f}M Qwen tokens ({grand_tok/1e9:.3f}B)")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
