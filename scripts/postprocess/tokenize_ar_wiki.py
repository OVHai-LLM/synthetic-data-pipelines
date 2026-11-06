#!/usr/bin/env python3
"""Flatten + Qwen-tokenize AR-on-Wiki output into the annealing jsonl format.

Reads:  $AR_DATA_DIR/wiki/<lang>/shard_*_of_215/batch_*.jsonl
        (rich nested: each record has up to 6 augmentations[].artifact)

Writes: $ANNEAL_ROOT/data/ar_wiki_tokenized/<lang>/shard_<N>.jsonl
        (one record per augmentation, with token_count under metadata.token_count)

Each output record:
    {
      "text":      "<artifact>",
      "id":        "ar-<lang>-<src_id>-<flavour>-<short_strategy_name>",
      "metadata": {
        "language":      "<lang>",
        "source":        "ar_wiki",
        "src_id":        "<original record id>",
        "title":         "<source article title>",
        "flavour":       "agnostic" | "factual",
        "strategy_name": "<strategy short name>",
        "token_count":   <int>,
      }
    }

Parallelised over input files. Tokenizer is loaded once per worker (fork-safe).
"""
import argparse
import gzip
import json
import multiprocessing as mp
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

AR_ROOT  = Path(os.environ.get("AR_DATA_DIR", "data/active_reading") + "/wiki")
OUT_ROOT = Path(os.environ.get("ANNEAL_ROOT", "data") + "/data/ar_wiki_tokenized")

# Per-lang Qwen tok/char ratios from a 30-sample tokenization per lang
# (measured 2026-05-13 via count_ar_wiki_tokens.py against
# RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic output). Using these as fixed
# multipliers avoids the HF-tokenizers pyo3 Rust panic that crashes
# subprocesses under both fork and spawn — and keeps the budget within ~5%.
TOK_PER_CHAR = {
    "ar": 0.376, "bg": 0.483, "bs": 0.390, "ca": 0.335, "cs": 0.417,
    "da": 0.342, "et": 0.405, "fi": 0.391, "ga": 0.399, "he": 0.408,
    "hr": 0.398, "hu": 0.440, "is": 0.447, "it": 0.248, "ka": 0.878,
    "ko": 0.610, "lb": 0.360, "lt": 0.428, "lv": 0.447, "no": 0.351,
    "ro": 0.357, "sk": 0.422, "sl": 0.391, "sq": 0.376, "sr": 0.481,
    "sv": 0.349, "tr": 0.351, "uk": 0.490, "zh": 0.599,
}
def estimate_tokens(text, lang):
    return int(len(text) * TOK_PER_CHAR.get(lang, 0.30))


_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")
def slug(s, maxlen=40):
    s = _SLUG_RE.sub("-", (s or "")).strip("-")
    return s[:maxlen] or "x"


def process_file(args):
    in_path, lang, shard_id, out_path = args
    if out_path.exists():
        try:
            n = sum(1 for _ in open(out_path))
            return (str(in_path), n, "skip-existing")
        except Exception:
            pass

    written = 0
    err = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        with open(in_path) as fin, open(tmp, "w") as fout:
            for line in fin:
                try:
                    r = json.loads(line)
                except Exception:
                    err += 1
                    continue
                src_id = r.get("id", "")
                title  = r.get("title", "") or ""
                for a in r.get("augmentations", []) or []:
                    artifact = (a or {}).get("artifact")
                    if not artifact or (a or {}).get("error"):
                        continue
                    flavour       = (a or {}).get("flavour", "")
                    strategy_name = (a or {}).get("strategy_name", "")
                    # Token-count via per-lang Qwen tok/char ratio (see TOK_PER_CHAR)
                    n_tok = estimate_tokens(artifact, lang)
                    out = {
                        "text": artifact,
                        "id":   f"ar-{lang}-{src_id}-{flavour}-{slug(strategy_name)}",
                        "metadata": {
                            "language":      lang,
                            "source":        "ar_wiki",
                            "src_id":        str(src_id),
                            "title":         title,
                            "flavour":       flavour,
                            "strategy_name": strategy_name,
                            "token_count":   n_tok,
                        },
                    }
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    written += 1
        os.replace(tmp, out_path)
    except Exception as e:
        try: tmp.unlink()
        except Exception: pass
        return (str(in_path), 0, f"ERR: {e}")
    return (str(in_path), written, "ok")


def list_jobs():
    jobs = []
    for lang_dir in sorted(AR_ROOT.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        for shard in sorted(lang_dir.glob("shard_*_of_*")):
            shard_id = int(shard.name.split("_")[1])
            for batch in sorted(shard.glob("batch_*.jsonl")):
                # Output file per input batch — keeps fault-tolerance simple
                batch_id = int(batch.stem.split("_")[1])
                out_name = f"shard_{shard_id:03d}_batch_{batch_id:02d}.jsonl"
                out_path = OUT_ROOT / lang / out_name
                jobs.append((str(batch), lang, shard_id, out_path))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=int(os.environ.get("NWORK", 48)))
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N input files (for testing)")
    args = ap.parse_args()

    jobs = list_jobs()
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"# {len(jobs)} input files to tokenize+flatten", file=sys.stderr)
    print(f"# output → {OUT_ROOT}", file=sys.stderr)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    n_done = total_written = total_err = 0
    # spawn (not fork) — pyo3-backed HF tokenizer panics in some forked children.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futs = [ex.submit(process_file, j) for j in jobs]
        for fut in as_completed(futs):
            path, n, status = fut.result()
            n_done += 1
            if status == "ok":
                total_written += n
            elif status == "skip-existing":
                total_written += n
            else:
                total_err += 1
                print(f"  ! {path}: {status}", file=sys.stderr)
            if n_done % 200 == 0:
                print(f"  [{n_done}/{len(jobs)}] total_records={total_written:,} errors={total_err}", flush=True)

    print(f"\nDONE: {n_done} files processed, {total_written:,} records written, {total_err} errors")


if __name__ == "__main__":
    main()
