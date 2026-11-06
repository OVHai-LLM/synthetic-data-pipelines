#!/usr/bin/env python3
"""Count Qwen tokens per language in the AR-on-Wiki output.

For each lang dir under data/wiki/<lang>/shard_*_of_215/batch_*.jsonl:
- count records + total artifact chars (fast pass)
- sample ~30 records to compute Qwen chars-per-token ratio for this lang
- extrapolate total Qwen tokens
"""
import os
import json, os, re, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(os.environ.get("AR_DATA_DIR", "data/active_reading") + "/wiki")
SAMPLE_N = 30  # per-lang sample for tok/char ratio

# Build a list of all .jsonl files grouped by lang
def list_files():
    by_lang = defaultdict(list)
    for lang_dir in sorted(ROOT.iterdir()):
        if not lang_dir.is_dir(): continue
        for shard in lang_dir.glob("shard_*_of_215"):
            for batch in shard.glob("batch_*.jsonl"):
                by_lang[lang_dir.name].append(str(batch))
    return by_lang

def count_records_chars(path):
    """Count records + total artifact chars in one file."""
    n_rec = 0; n_chars = 0
    try:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                n_rec += 1
                for a in r.get("augmentations", []):
                    art = a.get("artifact") or ""
                    n_chars += len(art)
    except Exception as e:
        print(f"  ! {path}: {e}", file=sys.stderr)
    return n_rec, n_chars

def sample_for_tokenization(lang_files, n=SAMPLE_N):
    """Pull n artifact strings from this lang's files, evenly spread."""
    arts = []
    if not lang_files: return arts
    files = lang_files[::max(1, len(lang_files)//5)][:5]  # 5 files
    per_file = max(1, n // max(1, len(files)))
    for fp in files:
        try:
            with open(fp) as f:
                taken = 0
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception: continue
                    for a in r.get("augmentations", []):
                        art = a.get("artifact")
                        if art:
                            arts.append(art)
                            taken += 1
                            if taken >= per_file: break
                    if taken >= per_file: break
        except Exception: pass
    return arts[:n]

def main():
    by_lang = list_files()
    print(f"# {sum(len(v) for v in by_lang.values())} files across {len(by_lang)} langs", file=sys.stderr)

    # Phase 1: fast char/record count per lang in parallel
    lang_totals = {}  # lang -> (n_rec, n_chars)
    with ProcessPoolExecutor(max_workers=int(os.environ.get("NWORK", 48))) as ex:
        per_file_futs = {ex.submit(count_records_chars, fp): (lang, fp)
                         for lang, files in by_lang.items() for fp in files}
        for fut in as_completed(per_file_futs):
            lang, fp = per_file_futs[fut]
            n_rec, n_chars = fut.result()
            if lang not in lang_totals:
                lang_totals[lang] = [0, 0]
            lang_totals[lang][0] += n_rec
            lang_totals[lang][1] += n_chars

    # Phase 2: load Qwen tokenizer once, sample tokenize per lang
    print(f"# loading Qwen tokenizer...", file=sys.stderr)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)

    print(f"\n{'lang':5} {'files':>6} {'records':>10} {'artif_chars':>14} {'sample_n':>8} {'tok/char':>9} {'est_tok':>12} {'B':>6}")
    print("-"*86)
    grand_tok = grand_rec = 0
    for lang in sorted(by_lang):
        n_rec, n_chars = lang_totals[lang]
        n_files = len(by_lang[lang])
        # Sample-tokenize to get tok/char
        samples = sample_for_tokenization(by_lang[lang])
        if samples:
            joined = "\n\n".join(samples[:SAMPLE_N])
            n_tok_sample = len(tok.encode(joined, add_special_tokens=False))
            n_char_sample = sum(len(s) for s in samples[:SAMPLE_N]) + 2 * (len(samples[:SAMPLE_N])-1)  # account for \n\n
            tpc = n_tok_sample / max(n_char_sample, 1)
        else:
            tpc = 0.25  # fallback
            n_tok_sample = 0
        est_tok = int(n_chars * tpc)
        grand_tok += est_tok
        grand_rec += n_rec
        print(f"{lang:5} {n_files:>6} {n_rec:>10,d} {n_chars:>14,d} {len(samples):>8} {tpc:>9.3f} {est_tok:>12,d} {est_tok/1e9:>6.2f}")
    print("-"*86)
    print(f"{'TOT':5} {'':>6} {grand_rec:>10,d} {'':>14} {'':>8} {'':>9} {grand_tok:>12,d} {grand_tok/1e9:>6.2f}")

if __name__ == "__main__":
    main()
