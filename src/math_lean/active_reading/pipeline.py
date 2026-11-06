"""Two-stage Active Reading pipeline.

Stage 1: for each source document, generate study strategies under each
         configured "flavour" (currently `agnostic` and `lean`).
Stage 2: for each (document, strategy) pair, generate a study artifact.

Both stages use a thread pool around the vLLM HTTP client. Output is
JSONL, one record per source document; each record carries the raw
strategy responses plus a list of augmentations.
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

from dataclasses import dataclass, field
from typing import Callable

from .client import VLLMClient
from .prompts import (
    APPLY_SYSTEM, APPLY_LEAN_SYSTEM_SUFFIX, APPLY_USER,
    STRATEGY_AGNOSTIC_SYSTEM, STRATEGY_AGNOSTIC_USER,
    STRATEGY_LEAN_SYSTEM, STRATEGY_LEAN_USER,
    parse_strategies,
)


STRATEGY_FLAVOURS = {
    "agnostic": (STRATEGY_AGNOSTIC_SYSTEM, STRATEGY_AGNOSTIC_USER),
    "lean":     (STRATEGY_LEAN_SYSTEM,     STRATEGY_LEAN_USER),
}


@dataclass
class PromptConfig:
    """Pluggable prompt set so the same orchestration can drive
    multiple sources (ProofWiki, Wikipedia, ...).

    `strategy_flavours`        : flavour -> (system, user_template). User
                                 template gets formatted with kwargs
                                 produced by `record_format_kwargs`.
    `apply_system`             : base system prompt for the apply call.
    `apply_user`               : user template for the apply call.
    `apply_system_suffix`      : flavour -> suffix appended to apply_system
                                 (defaults to '' for any unknown flavour).
    `record_format_kwargs`     : callable(record) -> dict, extra kwargs
                                 fed into the user templates. Lets the
                                 wiki variant inject `language` etc.
    """
    strategy_flavours: dict
    apply_system: str
    apply_user: str
    apply_system_suffix: dict = field(default_factory=dict)
    record_format_kwargs: Callable[[dict], dict] = field(
        default_factory=lambda: (lambda r: {"kind": r.get("kind", "theorem")})
    )


PROOFWIKI_PROMPT_CONFIG = PromptConfig(
    strategy_flavours=STRATEGY_FLAVOURS,
    apply_system=APPLY_SYSTEM,
    apply_user=APPLY_USER,
    apply_system_suffix={"lean": APPLY_LEAN_SYSTEM_SUFFIX},
)


# ─────────────────────────────────────────────────────────────────
# Sampling source documents
# ─────────────────────────────────────────────────────────────────

def doc_chunk(record: dict) -> str:
    """Concatenate statement + all proofs into a single source chunk.

    Wikitext is preserved as-is — that's what the paper does, and it
    keeps LaTeX / wikilinks intact for the model to read.
    """
    parts = [f"=== Statement ===\n{record.get('statement','').strip()}"]
    for i, p in enumerate(record.get("proofs") or []):
        name = (p or {}).get("name") or f"Proof {i+1}"
        text = (p or {}).get("text") or ""
        parts.append(f"=== {name} ===\n{text.strip()}")
    return "\n\n".join(parts)


def sample_theorems(path: str, *, n: int, seed: int = 0,
                    min_proof_chars: int = 200,
                    max_chunk_chars: int = 8000,
                    offset: int = 0) -> list[dict]:
    """Pick `n` theorems with at least one substantive proof.

    Filters out records whose total chunk size is too long for the
    context window we are willing to spend per call. Returns records
    augmented with `_chunk` (the joined statement+proofs string).

    `offset` skips the first N theorems in the (deterministically
    shuffled) candidate list — use this with a fixed `seed` to do
    resumable batched runs across the full corpus.
    """
    candidates: list[dict] = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") != "theorem":
                continue
            proofs = r.get("proofs") or []
            if not proofs:
                continue
            longest = max((len((p or {}).get("text", "") or "")
                           for p in proofs), default=0)
            if longest < min_proof_chars:
                continue
            chunk = doc_chunk(r)
            if len(chunk) > max_chunk_chars:
                continue
            r["_chunk"] = chunk
            candidates.append(r)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[offset:offset + n]


# ─────────────────────────────────────────────────────────────────
# Stage 1 — strategies
# ─────────────────────────────────────────────────────────────────

def gen_strategies(client: VLLMClient, record: dict, flavour: str, *,
                   temperature: float = 0.8,
                   prompt_config: PromptConfig = PROOFWIKI_PROMPT_CONFIG) -> dict:
    sys_p, user_tpl = prompt_config.strategy_flavours[flavour]
    fmt_kwargs = {
        "title": record["title"],
        "chunk": record["_chunk"],
        **prompt_config.record_format_kwargs(record),
    }
    user = user_tpl.format(**fmt_kwargs)
    # pin all calls for this article to the same DP rank so vLLM's per-rank
    # prefix cache shares the article's KV blocks across the 8 calls per article
    # (2 strategy + 6 apply). Ignored when client has a single endpoint.
    raw = client.chat(sys_p, user, temperature=temperature, max_tokens=2048,
                      pin_key=str(record["id"]))
    return {
        "flavour": flavour,
        "raw": raw,
        "strategies": parse_strategies(raw),
    }


# ─────────────────────────────────────────────────────────────────
# Stage 2 — apply strategies
# ─────────────────────────────────────────────────────────────────

def apply_strategy(client: VLLMClient, record: dict, strategy: dict,
                   flavour: str, *, temperature: float = 0.8,
                   max_tokens: int = 2048,
                   prompt_config: PromptConfig = PROOFWIKI_PROMPT_CONFIG) -> dict:
    fmt_kwargs = {
        "strategy": strategy["raw"],
        "title": record["title"],
        "chunk": record["_chunk"],
        **prompt_config.record_format_kwargs(record),
    }
    user = prompt_config.apply_user.format(**fmt_kwargs)
    suffix = prompt_config.apply_system_suffix.get(flavour, "")
    if suffix:
        # suffix may itself be a template (e.g. needs `{language}`)
        suffix = suffix.format(**fmt_kwargs)
    system = prompt_config.apply_system + suffix
    artifact = client.chat(system, user,
                           temperature=temperature, max_tokens=max_tokens,
                           pin_key=str(record["id"]))
    return {
        "flavour": flavour,
        "strategy_name": strategy["name"],
        "artifact": artifact,
    }


# ─────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────

def run_one(client: VLLMClient, record: dict, flavours: list[str], *,
            strategy_temp: float, apply_temp: float,
            apply_max_tokens: int,
            prompt_config: PromptConfig = PROOFWIKI_PROMPT_CONFIG) -> dict:
    """Generate strategies (one call per flavour) then apply each one
    (one call per (flavour, strategy) pair). Returns the per-document
    record."""

    strategy_responses: list[dict] = []
    for fl in flavours:
        try:
            strategy_responses.append(
                gen_strategies(client, record, fl,
                               temperature=strategy_temp,
                               prompt_config=prompt_config))
        except Exception as e:
            strategy_responses.append({
                "flavour": fl, "raw": "", "strategies": [],
                "error": f"{type(e).__name__}: {e}",
            })

    augmentations: list[dict] = []
    for sr in strategy_responses:
        fl = sr["flavour"]
        for strat in sr["strategies"]:
            try:
                augmentations.append(apply_strategy(
                    client, record, strat, fl,
                    temperature=apply_temp, max_tokens=apply_max_tokens,
                    prompt_config=prompt_config))
            except Exception as e:
                augmentations.append({
                    "flavour": fl,
                    "strategy_name": strat["name"],
                    "artifact": "",
                    "error": f"{type(e).__name__}: {e}",
                })

    out = {
        "id": record["id"],
        "title": record["title"],
        "kind": record.get("kind"),
        "wikilinks": record.get("wikilinks", []),
        "categories": record.get("categories", []),
        "chunk": record["_chunk"],
        "strategy_responses": strategy_responses,
        "augmentations": augmentations,
    }
    if record.get("_language"):
        out["language"] = record["_language"]
    return out


def run(records: list[dict], client: VLLMClient, out_path: str, *,
        flavours: Iterable[str] = ("agnostic", "lean"),
        workers: int = 8,
        strategy_temp: float = 0.8,
        apply_temp: float = 0.8,
        apply_max_tokens: int = 2048,
        log_every: int = 5,
        prompt_config: PromptConfig = PROOFWIKI_PROMPT_CONFIG) -> dict:
    """Run the pipeline over `records` and stream JSONL to `out_path`.

    Concurrency is at the *document* level: each thread runs Stage 1
    + Stage 2 for one document end-to-end. This keeps strategies and
    their augmentations together in time, simplifying retries.
    Returns a small summary dict.
    """
    flavours = list(flavours)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    written = 0
    failed = 0
    n_strategies = 0
    n_augs = 0
    t0 = time.time()

    with open(out_path, "w") as fout, ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(run_one, client, rec, flavours,
                      strategy_temp=strategy_temp,
                      apply_temp=apply_temp,
                      apply_max_tokens=apply_max_tokens,
                      prompt_config=prompt_config): rec["id"]
            for rec in records
        }
        for fut in as_completed(futures):
            doc_id = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:
                failed += 1
                print(f"  [FAIL] {doc_id}: {type(e).__name__}: {e}", flush=True)
                continue
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
            n_strategies += sum(len(s["strategies"]) for s in rec["strategy_responses"])
            n_augs += len(rec["augmentations"])
            if written % log_every == 0:
                rate = written / max(time.time() - t0, 1e-3)
                print(f"  [{written}/{len(records)}] strategies={n_strategies} "
                      f"augs={n_augs} ({rate:.2f} doc/s)", flush=True)

    elapsed = time.time() - t0
    return {
        "written": written, "failed": failed,
        "n_strategies": n_strategies, "n_augmentations": n_augs,
        "elapsed_sec": elapsed,
    }