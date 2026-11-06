"""Diagnostics for Active Reading output.

Two metrics, mirroring Lin et al. 2025 Section 5.1:

* **self-BLEU** (per source document) — proxy for diversity. We compute
  pairwise corpus-BLEU between every augmentation and the union of its
  siblings as references, then average. Lower = more diverse. Unlike
  sacrebleu we use a clipped n-gram precision with brevity penalty
  rolled up to BLEU-4; the *relative* signal is what matters.

* **Lean-syntax rate** (per flavour) — what fraction of `lean`-flavoured
  augmentations actually contain plausible Lean 4 code. A weak proxy
  for "the strategy was applied as intended"; a real check would
  require `lake build`.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from statistics import mean


# ─────────────────────────────────────────────────────────────────
# Tokenisation
# ─────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenise(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# ─────────────────────────────────────────────────────────────────
# BLEU-4 (clipped n-gram precision + brevity penalty)
# ─────────────────────────────────────────────────────────────────

def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _bleu(hyp: list[str], refs: list[list[str]], max_n: int = 4) -> float:
    """Sentence-level BLEU with smoothing-1 (add-epsilon to zero
    n-gram counts) and standard brevity penalty."""
    if not hyp:
        return 0.0
    log_precisions = []
    for n in range(1, max_n + 1):
        hyp_counts = _ngram_counts(hyp, n)
        if not hyp_counts:
            log_precisions.append(math.log(1e-9))
            continue
        max_ref = Counter()
        for ref in refs:
            for ng, c in _ngram_counts(ref, n).items():
                if c > max_ref[ng]:
                    max_ref[ng] = c
        clipped = sum(min(c, max_ref[ng]) for ng, c in hyp_counts.items())
        total = sum(hyp_counts.values())
        # smoothing: avoid log(0) for sparse high-order n-grams
        p = (clipped + 1e-9) / (total + 1e-9) if clipped > 0 else 1e-9
        log_precisions.append(math.log(p))
    geo = math.exp(sum(log_precisions) / max_n)
    # brevity penalty against closest-length reference
    ref_lens = [len(r) for r in refs] or [0]
    closest = min(ref_lens, key=lambda rl: (abs(rl - len(hyp)), rl))
    bp = 1.0 if len(hyp) > closest else math.exp(1 - closest / max(len(hyp), 1))
    return bp * geo


def self_bleu(texts: list[str]) -> float:
    """Average pairwise BLEU-4: each text scored against the others.

    Returns 0..100 (sacrebleu convention). A handful of texts is enough
    to be meaningful; with <2 texts we return NaN."""
    toks = [tokenise(t) for t in texts if t]
    if len(toks) < 2:
        return float("nan")
    scores = []
    for i, hyp in enumerate(toks):
        refs = [toks[j] for j in range(len(toks)) if j != i]
        scores.append(_bleu(hyp, refs))
    return 100.0 * mean(scores)


# ─────────────────────────────────────────────────────────────────
# Lean-syntax detector
# ─────────────────────────────────────────────────────────────────

_LEAN_FENCE_RE = re.compile(r"```(?:lean4?|Lean4?)\n(.*?)```", re.DOTALL)
_LEAN_DECL_RE = re.compile(r"\b(?:theorem|lemma|def|example|instance)\s+\w+",
                           re.MULTILINE)
_LEAN_TACTICS = {
    "rfl", "rw", "simp", "exact", "apply", "intro", "intros", "constructor",
    "induction", "cases", "use", "refine", "linarith", "nlinarith", "omega",
    "ring", "ring_nf", "field_simp", "norm_num", "decide", "tauto", "by",
    "calc", "have", "show", "obtain", "rcases", "rintro", "ext", "funext",
    "contradiction", "trivial", "assumption",
}
_TACTIC_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in _LEAN_TACTICS) + r")\b")


def lean_signal(text: str) -> dict:
    """Return a small dict of Lean-presence signals for one artifact."""
    fences = _LEAN_FENCE_RE.findall(text)
    has_fence = bool(fences)
    decl_count = len(_LEAN_DECL_RE.findall(text))
    tactic_count = len(_TACTIC_RE.findall(text))
    # "looks like Lean": at least one fenced lean block OR (a decl AND >=2 tactics)
    looks_lean = has_fence or (decl_count >= 1 and tactic_count >= 2)
    return {
        "has_lean_fence": has_fence,
        "lean_decl_count": decl_count,
        "lean_tactic_count": tactic_count,
        "looks_lean": looks_lean,
    }


# ─────────────────────────────────────────────────────────────────
# Top-level summary
# ─────────────────────────────────────────────────────────────────

def summarise(jsonl_path: str) -> dict:
    """Read a pipeline output JSONL and return aggregate diagnostics."""
    n_docs = 0
    n_augs = 0
    n_aug_errors = 0
    n_strat_errors = 0
    aug_chars: list[int] = []
    aug_chars_by_flavour: dict[str, list[int]] = {}
    self_bleu_by_flavour: dict[str, list[float]] = {}
    self_bleu_overall: list[float] = []
    lean_signal_counts: dict[str, dict] = {
        # flavour -> running totals
    }
    n_strats_by_flavour: dict[str, list[int]] = {}

    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            n_docs += 1
            for sr in rec.get("strategy_responses", []):
                fl = sr["flavour"]
                n_strats_by_flavour.setdefault(fl, []).append(len(sr.get("strategies", [])))
                if sr.get("error"):
                    n_strat_errors += 1

            augs = rec.get("augmentations", [])
            by_flavour: dict[str, list[str]] = {}
            for a in augs:
                n_augs += 1
                if a.get("error"):
                    n_aug_errors += 1
                    continue
                art = a.get("artifact") or ""
                aug_chars.append(len(art))
                fl = a["flavour"]
                aug_chars_by_flavour.setdefault(fl, []).append(len(art))
                by_flavour.setdefault(fl, []).append(art)
                # lean signal — only meaningful for lean-flavoured but compute everywhere
                sig = lean_signal(art)
                slot = lean_signal_counts.setdefault(fl, {
                    "n": 0, "fence": 0, "decl": 0, "tactic": 0, "looks_lean": 0,
                })
                slot["n"] += 1
                slot["fence"] += int(sig["has_lean_fence"])
                slot["decl"] += sig["lean_decl_count"]
                slot["tactic"] += sig["lean_tactic_count"]
                slot["looks_lean"] += int(sig["looks_lean"])

            # self-BLEU
            all_arts = [a.get("artifact", "") for a in augs if a.get("artifact")]
            if len(all_arts) >= 2:
                self_bleu_overall.append(self_bleu(all_arts))
            for fl, arts in by_flavour.items():
                if len(arts) >= 2:
                    self_bleu_by_flavour.setdefault(fl, []).append(self_bleu(arts))

    summary = {
        "n_documents": n_docs,
        "n_strategy_errors": n_strat_errors,
        "n_augmentations": n_augs,
        "n_augmentation_errors": n_aug_errors,
        "avg_strategies_per_doc": {
            fl: (sum(v) / len(v) if v else 0.0)
            for fl, v in n_strats_by_flavour.items()
        },
        "avg_artifact_chars": (mean(aug_chars) if aug_chars else 0.0),
        "avg_artifact_chars_by_flavour": {
            fl: (mean(v) if v else 0.0) for fl, v in aug_chars_by_flavour.items()
        },
        "self_bleu_overall_mean": (mean(self_bleu_overall) if self_bleu_overall else float("nan")),
        "self_bleu_by_flavour": {
            fl: (mean(v) if v else float("nan"))
            for fl, v in self_bleu_by_flavour.items()
        },
        "lean_syntax_by_flavour": {
            fl: {
                "n": s["n"],
                "pct_with_fence": 100 * s["fence"] / s["n"] if s["n"] else 0.0,
                "pct_looks_lean": 100 * s["looks_lean"] / s["n"] if s["n"] else 0.0,
                "avg_lean_decls": s["decl"] / s["n"] if s["n"] else 0.0,
                "avg_lean_tactics": s["tactic"] / s["n"] if s["n"] else 0.0,
            }
            for fl, s in lean_signal_counts.items()
        },
    }
    return summary


def format_summary(summary: dict) -> str:
    lines = []
    lines.append(f"Documents:           {summary['n_documents']}")
    lines.append(f"Augmentations:       {summary['n_augmentations']}  "
                 f"(errors: {summary['n_augmentation_errors']})")
    lines.append(f"Strategy errors:     {summary['n_strategy_errors']}")
    lines.append("")
    lines.append("Strategies per doc, by flavour:")
    for fl, v in summary["avg_strategies_per_doc"].items():
        lines.append(f"  {fl:>10}: {v:.2f}")
    lines.append("")
    lines.append(f"Avg artifact chars:  {summary['avg_artifact_chars']:.0f}")
    for fl, v in summary["avg_artifact_chars_by_flavour"].items():
        lines.append(f"  {fl:>10}: {v:.0f}")
    lines.append("")
    lines.append("self-BLEU (lower = more diverse):")
    lines.append(f"  overall:   {summary['self_bleu_overall_mean']:.2f}")
    for fl, v in summary["self_bleu_by_flavour"].items():
        lines.append(f"  {fl:>10}: {v:.2f}")
    lines.append("")
    lines.append("Lean-syntax presence (per flavour):")
    for fl, s in summary["lean_syntax_by_flavour"].items():
        lines.append(f"  {fl:>10}: n={s['n']:<4d} fence={s['pct_with_fence']:5.1f}%  "
                     f"looks_lean={s['pct_looks_lean']:5.1f}%  "
                     f"avg_decls={s['avg_lean_decls']:.2f}  "
                     f"avg_tactics={s['avg_lean_tactics']:.2f}")
    return "\n".join(lines)