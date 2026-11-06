"""prepare_proofwiki_proofs.py

Convert $ANNEAL_ROOT/proofwiki_proofs.jsonl into the
standard annealing format ({text, id, metadata}) and Qwen-tokenize it.

Records contain:
  - kind: theorem / definition / axiom / corollary
  - title, source_url
  - statement (may be empty)
  - proofs: list of {name, text}  (often empty for definitions)

Output format per line:
  {
    "text": "<formatted theorem text>",
    "id": "<original proofwiki id>",
    "metadata": {
      "token_count": int,
      "source": "proofwiki",
      "kind": "theorem|definition|axiom|corollary",
      "n_proofs": int,
      "categories": [...],
      "source_url": "..."
    }
  }
"""
from __future__ import annotations
import os

import argparse
import json
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count


# ── Wiki / proofwiki markup cleaners ──────────────────────────────────
# We strip the structural noise but keep math notation ($...$, LaTeX) intact.

# [[Definition:Foo|foo]]  →  foo   ; [[Definition:Foo]]  →  Foo
_LINK_PIPE_RE = re.compile(r"\[\[(?:Definition|Theorem|Axiom|Corollary|Symbol|Category|File|Image|Lemma)?:?[^|\]]*\|([^\]]+)\]\]")
_LINK_PLAIN_RE = re.compile(r"\[\[(?:Definition|Theorem|Axiom|Corollary|Symbol|Category|File|Image|Lemma):([^\]]+)\]\]")
_LINK_BARE_RE = re.compile(r"\[\[([^|\]]+)\]\]")

# {{begin-eqn}} ... {{end-eqn}} blocks: keep contents, drop the wrapper tags
_EQN_BEGIN_RE = re.compile(r"\{\{begin-eqn[^}]*\}\}")
_EQN_END_RE = re.compile(r"\{\{end-eqn[^}]*\}\}")
# {{eqn | l = ... | r = ...}}  →  ... = ...   (eqn rows)
_EQN_ROW_RE = re.compile(r"\{\{eqn(.*?)\}\}", re.DOTALL)
# {{anything else}}  →  drop entirely
_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
# <onlyinclude>, </onlyinclude>, <ref>, </ref>, etc.
_HTML_TAG_RE = re.compile(r"</?(?:onlyinclude|ref|sub|sup|small|big|nowiki|code|tt|s|del|u|i|b|em|strong|hr|br/?)\s*[^>]*>", re.IGNORECASE)
# Wikipedia-style footnote refs: [[#Foo|...]] etc. caught above
# == Section headers ==
_HEADER_RE = re.compile(r"^={2,}\s*([^=\n]+?)\s*={2,}\s*$", re.MULTILINE)
# Multiple blank lines
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def _eqn_row_to_text(m: re.Match) -> str:
    body = m.group(1)
    # body is like  | l = ... | r = ... | c = ...
    parts = {}
    for chunk in body.split("|"):
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            parts[k.strip()] = v.strip()
    l = parts.get("l", "")
    o = parts.get("o", "=")  # operator (default =)
    r = parts.get("r", "")
    c = parts.get("c", "")
    s = f"{l} {o} {r}" if l or r else ""
    if c:
        s = f"{s}  ({c})" if s else f"({c})"
    return s.strip()


def clean_wiki_markup(text: str) -> str:
    if not text:
        return ""
    # Order matters: replace links first, then templates
    text = _LINK_PIPE_RE.sub(r"\1", text)
    text = _LINK_PLAIN_RE.sub(r"\1", text)
    text = _LINK_BARE_RE.sub(r"\1", text)
    text = _EQN_ROW_RE.sub(_eqn_row_to_text, text)
    text = _EQN_BEGIN_RE.sub("", text)
    text = _EQN_END_RE.sub("", text)
    text = _TEMPLATE_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    text = _HEADER_RE.sub(r"\n## \1\n", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def format_record(rec: dict) -> dict | None:
    """Build {text,id,metadata} from a raw proofwiki record. Returns None if empty."""
    title = (rec.get("title") or "").strip()
    statement = clean_wiki_markup(rec.get("statement") or "")
    proofs = rec.get("proofs", []) or []

    # If the record is empty (no title + no statement + no proofs), drop
    if not title and not statement and not proofs:
        return None

    kind = rec.get("kind") or "theorem"
    parts = []
    if title:
        parts.append(f"# {kind.capitalize()}: {title}\n")
    if statement:
        parts.append(f"## Statement\n\n{statement}\n")
    for i, p in enumerate(proofs):
        ptext = clean_wiki_markup(p.get("text") or "")
        if not ptext.strip():
            continue
        pname = (p.get("name") or f"Proof {i+1}").strip()
        parts.append(f"## {pname}\n\n{ptext}\n")

    text = "\n".join(parts).strip()
    if not text or len(text) < 30:
        return None

    return {
        "text": text,
        "id": rec.get("id") or "proofwiki:unknown",
        "metadata": {
            "source": "proofwiki",
            "kind": kind,
            "n_proofs": sum(1 for p in proofs if (p.get("text") or "").strip()),
            "categories": rec.get("categories") or [],
            "source_url": rec.get("source_url") or "",
        },
    }


# ── Tokenisation worker (Qwen) ─────────────────────────────────────
_TOK = None
def _get_tok(name: str):
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(name)
    return _TOK


def tokenize_chunk(args):
    chunk_id, records, tok_name = args
    tok = _get_tok(tok_name)
    out = []
    n_tok = 0
    for r in records:
        formatted = format_record(r)
        if formatted is None:
            continue
        t = len(tok.encode(formatted["text"], add_special_tokens=False))
        formatted["metadata"]["token_count"] = t
        n_tok += t
        out.append(formatted)
    return chunk_id, out, n_tok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",
        default=os.environ.get("ANNEAL_ROOT", "data") + "/proofwiki_proofs.jsonl")
    p.add_argument("--output",
        default=os.environ.get("ANNEAL_ROOT", "data") + "/data/maths/proofwiki_proofs/proofwiki_proofs_qwen.jsonl")
    p.add_argument("--tok-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--workers", type=int, default=min(32, max(1, cpu_count() // 4)))
    p.add_argument("--chunk-size", type=int, default=1000)
    args = p.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] reading {args.input}", file=sys.stderr)
    records = []
    with open(args.input) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"[{time.strftime('%H:%M:%S')}] {len(records)} raw records", file=sys.stderr)

    # split into chunks
    chunks = [records[i:i+args.chunk_size] for i in range(0, len(records), args.chunk_size)]
    print(f"[{time.strftime('%H:%M:%S')}] {len(chunks)} chunks × {args.chunk_size}, "
          f"{args.workers} workers", file=sys.stderr)

    # pre-warm tokenizer in main process
    _get_tok(args.tok_name)

    n_done = n_kept = total_tokens = 0
    t0 = time.time()
    with open(args.output, "w") as fout, \
         ProcessPoolExecutor(max_workers=args.workers) as ex:
        jobs = [(i, chunks[i], args.tok_name) for i in range(len(chunks))]
        for chunk_id, out, n_tok in ex.map(tokenize_chunk, jobs):
            for r in out:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_kept += len(out)
            total_tokens += n_tok
            n_done += 1
            if n_done % 5 == 0:
                el = time.time() - t0
                print(f"  {n_done}/{len(chunks)} chunks | kept {n_kept} | "
                      f"{total_tokens/1e6:.1f} M tokens | {n_kept/el:.0f} rec/s",
                      file=sys.stderr)

    print(f"\n[{time.strftime('%H:%M:%S')}] DONE", file=sys.stderr)
    print(f"  raw records: {len(records)}", file=sys.stderr)
    print(f"  kept (non-empty): {n_kept}", file=sys.stderr)
    print(f"  total Qwen tokens: {total_tokens:,} ({total_tokens/1e6:.1f} M = {total_tokens/1e9:.3f} B)",
          file=sys.stderr)
    print(f"  output: {args.output}", file=sys.stderr)
    print(f"  wall: {(time.time()-t0)/60:.1f} min", file=sys.stderr)


if __name__ == "__main__":
    main()
