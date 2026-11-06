"""Generate plausible-but-wrong answer choices (distractors) for a given gold
answer string.

A *distractor* is a wrong answer that a student or weak model would plausibly
pick. Every problem in the corpus becomes both a free-form item and a 4-way
multiple-choice item once distractors are attached.

Strategy: dispatch on the structural type of the gold answer string.
- "Yes" / "No"             → return ["No"] / ["Yes"] (only one alternative)
- "True" / "False"         → return ["False"] / ["True"]
- pure integer "n"         → off-by-one, sign flip, ×2
- fraction "a/b"           → invert, sign flip, ±1 in numerator
- decimal "12.34"          → ±1, sign flip, last-digit perturbation
- dollar "$12.34" / "$1/2" → as numeric, with $ prepended
- "x = N" or list of vals  → sign-flip the values
- otherwise                → empty list (no distractors generated)

When in doubt, returns fewer distractors rather than nonsense ones — caller
should not assume exactly 3."""

from __future__ import annotations

import random
import re
from fractions import Fraction
from typing import List

YES_NO = {"Yes": "No", "No": "Yes"}
TRUE_FALSE = {"True": "False", "False": "True"}

_INT_RE = re.compile(r"^-?\d+$")
_FRAC_RE = re.compile(r"^(-?\d+)/(\d+)$")
_DEC_RE = re.compile(r"^-?\d+\.\d+$")
_DOLLAR_RE = re.compile(r"^\$(.+)$")
_X_EQ_RE = re.compile(r"^x\s*=\s*(.+)$")


def _int_distractors(n: int) -> List[str]:
    pool = [n + 1, n - 1, -n, n * 2]
    return [str(v) for v in pool if v != n]


def _frac_distractors(num: int, den: int) -> List[str]:
    candidates = [
        Fraction(den, num) if num != 0 else None,    # invert
        Fraction(-num, den),                          # sign flip
        Fraction(num + 1, den),                       # numerator + 1
        Fraction(num - 1, den) if num - 1 != 0 else None,
    ]
    out = []
    seen = {Fraction(num, den)}
    for f in candidates:
        if f is None: continue
        if f in seen: continue
        seen.add(f)
        out.append(f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator))
    return out


def _dec_distractors(x: float, raw: str) -> List[str]:
    decimals = len(raw.split(".")[-1]) if "." in raw else 2
    fmt = f"{{:.{decimals}f}}"
    pool = [x + 1, x - 1, -x]
    # Last-digit perturbation
    step = 10 ** -decimals
    pool.append(x + step)
    out = []
    seen = {fmt.format(x)}
    for v in pool:
        f = fmt.format(v)
        if f not in seen:
            seen.add(f); out.append(f)
    return out


def _x_eq_distractors(rest: str) -> List[str]:
    """For 'x = N' or 'x = a or x = b' style answers."""
    # Single number / fraction / negative
    m_int = _INT_RE.match(rest.strip())
    if m_int:
        n = int(rest.strip())
        return [f"x = {d}" for d in _int_distractors(n)]
    m_frac = _FRAC_RE.match(rest.strip())
    if m_frac:
        return [f"x = {d}" for d in _frac_distractors(int(m_frac.group(1)), int(m_frac.group(2)))]
    return []


def make_distractors(answer: str, n: int = 3) -> List[str]:
    """Return up to `n` plausible wrong answers. May return fewer (or none)
    when the answer's structure isn't in the supported set."""
    if not isinstance(answer, str) or not answer:
        return []
    a = answer.strip()
    out: List[str] = []

    # Yes / No
    if a in YES_NO:
        return [YES_NO[a]]
    if a in TRUE_FALSE:
        return [TRUE_FALSE[a]]

    # Integer
    if _INT_RE.match(a):
        out = _int_distractors(int(a))

    # Fraction
    elif (m := _FRAC_RE.match(a)):
        out = _frac_distractors(int(m.group(1)), int(m.group(2)))

    # Decimal
    elif _DEC_RE.match(a):
        out = _dec_distractors(float(a), a)

    # Dollar wrapper
    elif (m := _DOLLAR_RE.match(a)):
        inner = m.group(1).strip()
        inner_d = make_distractors(inner, n + 2)
        out = [f"${d}" for d in inner_d]

    # x = ... (single root or "x = a or x = b")
    elif (m := _X_EQ_RE.match(a)):
        inner = m.group(1).strip()
        if " or " not in inner:
            out = _x_eq_distractors(inner)
        else:
            parts = [p.strip() for p in inner.split(" or ")]
            # Sign flip both, swap order, pick only one
            try:
                vals = [int(re.sub(r"x\s*=\s*", "", p)) for p in parts]
                out = [f"x = {-vals[0]} or x = {-vals[1]}",
                       f"x = {vals[0]} or x = {-vals[1]}",
                       f"x = {vals[0]}"]
            except (ValueError, IndexError):
                out = []

    # Trim, dedupe, cap at n
    seen = {a}
    final: List[str] = []
    for d in out:
        if d not in seen:
            seen.add(d); final.append(d)
        if len(final) >= n:
            break
    return final


def shuffled_choices(answer: str, distractors: List[str]) -> tuple:
    """Return (lettered_options, correct_letter). Padding the gold answer with
    distractors and shuffling. Letters: A, B, C, D, ..."""
    options = [answer] + list(distractors)
    indices = list(range(len(options)))
    random.shuffle(indices)
    letters = "ABCDEFGH"
    lettered = [(letters[i], options[idx]) for i, idx in enumerate(indices)]
    correct_letter = letters[indices.index(0)]
    return lettered, correct_letter
