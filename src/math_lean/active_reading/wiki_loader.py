"""Streaming loader + filter for the Wikipedia jsonl.gz dumps.

The dumps live at:

    $WIKI_DUMP_ROOT/<lang>/wikipedia-*/<lang>.jsonl.gz

Each line is a JSON record:

    {
      "id": <int>,
      "title": "...",
      "text": "# Title\\n\\n<markdown body>",
      "url": "...",
      "categories": [...],
      "infobox": "" | "...",
      "token_count": <int>,
      "entity": "Q...",
      "revdate": "..."
    }

`sample_articles` returns up to `n` records skipping the first `offset`
after a deterministic shuffle keyed by `seed`. Records are filtered by
token_count to drop stubs and giant articles that wouldn't fit a single
LLM call.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Iterator


def _open_maybe_gz(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _iter_one_file(path: str) -> Iterator[dict]:
    with _open_maybe_gz(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_articles(path: str) -> Iterator[dict]:
    """Yield every record in a wikipedia jsonl[.gz] file OR every record
    across every jsonl[.gz] shard inside a directory.

    Many wiki dumps land sharded — `data/{lang}/wikipedia-1/{lang}_1.jsonl,
    {lang}_2.jsonl, …`. Passing the parent dir iterates them all in
    deterministic name-sorted order so the seed-based shuffle stays stable
    across runs.
    """
    p = Path(path)
    if p.is_dir():
        # Sorted glob so iteration order is deterministic across runs.
        files = sorted(
            list(p.glob("*.jsonl")) + list(p.glob("*.jsonl.gz"))
        )
        if not files:
            raise FileNotFoundError(f"no .jsonl(.gz) files in directory {path}")
        for fp in files:
            yield from _iter_one_file(str(fp))
    else:
        yield from _iter_one_file(path)


def sample_articles(path: str, *, n: int, seed: int = 0,
                    offset: int = 0,
                    min_tokens: int = 200,
                    max_tokens: int = 6000,
                    language: str | None = None) -> list[dict]:
    """Pick `n` articles from a Wikipedia dump.

    Filters by `token_count` (when present) so we drop stubs and giant
    articles. The dump's `token_count` is generally a good proxy for
    LLM token cost; cap defaults to ~6k to leave headroom for a 32k
    context with a 1k-token apply prompt and 4k completion.

    Returns records augmented with:
      * `_chunk`    : the article body, ready to feed into a prompt
      * `_language` : the language code (passed in or inferred from path)
      * `id`, `title`, `categories`, `entity`, `url` are kept

    The shape is intentionally a superset of what pipeline.run_one
    expects (it reads `_chunk`, `title`, `id`, optionally `categories`).
    """
    if language is None:
        # Infer from the path. Works for both single files
        # (/…/lt/wikipedia-1/lt.jsonl → "lt") and directories
        # (/…/it/wikipedia-1/ → "it" via the parent-of-parent dir name).
        p = Path(path)
        if p.is_dir():
            language = p.parent.name
        else:
            language = p.stem.split(".")[0]

    candidates: list[dict] = []
    for r in iter_articles(path):
        # Two record schemas in the wild:
        #  - "flat": title/token_count/url/etc. at top level (ga, lb, …)
        #  - "shuffled": same fields nested under r['metadata'] (ko, lt, lv,
        #    no, ro, sk, sl, sq, tr, uk — anything from wikipedia-N_shuffled
        #    dirs). Lift metadata fields onto the top level so the rest of
        #    the pipeline (which reads r['title'] etc.) works for both.
        md = r.get("metadata")
        if isinstance(md, dict):
            for k, v in md.items():
                r.setdefault(k, v)
        tc = r.get("token_count") or 0
        if tc and (tc < min_tokens or tc > max_tokens):
            continue
        text = (r.get("text") or "").strip()
        if not text:
            continue
        # If token_count is missing, fall back to a char heuristic
        if not tc:
            char_est = len(text)
            if char_est < min_tokens * 4 or char_est > max_tokens * 4:
                continue
        r["_chunk"] = text
        r["_language"] = language
        candidates.append(r)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[offset:offset + n]


def count_eligible(path: str, *, min_tokens: int = 200, max_tokens: int = 6000) -> int:
    """How many articles in the dump pass the filter? Useful for sizing
    the run before launching it."""
    n = 0
    for r in iter_articles(path):
        tc = r.get("token_count") or 0
        if tc and (tc < min_tokens or tc > max_tokens):
            continue
        if not (r.get("text") or "").strip():
            continue
        if not tc:
            char_est = len(r.get("text", ""))
            if char_est < min_tokens * 4 or char_est > max_tokens * 4:
                continue
        n += 1
    return n
