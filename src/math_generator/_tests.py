"""Lightweight smoke tests for every registered generator.

Run with `python -m math_generator._tests` from the project root.
Calls every generator N times at each difficulty and asserts the
returned triple is well-formed. Catches the most common refactor
regressions: import errors, missing keys, crashes on an edge case,
empty problem/answer strings.

Not a verification harness — does not check that the answer is
mathematically correct. See the discussion in the project notes for
why a full verifier is overkill here."""

from __future__ import annotations

import random
import sys
import traceback
from collections import defaultdict

from .registry import REGISTRY


REQUIRED_META = {"category"}
N_PER_GEN = 30


def _validate(triple, name):
    if not (isinstance(triple, tuple) and len(triple) == 3):
        return f"return type is not (str, str, dict): {type(triple)}"
    prob, ans, meta = triple
    if not isinstance(prob, str) or not prob:
        return "problem is empty / not a string"
    if not isinstance(ans, str) or not ans:
        return "answer is empty / not a string"
    if not isinstance(meta, dict):
        return f"meta is not a dict: {type(meta)}"
    missing = REQUIRED_META - meta.keys()
    if missing:
        return f"missing meta keys: {missing}"
    return None


def main():
    random.seed(0)
    failures: list[tuple[str, str, str]] = []  # (gen, dif, error)
    seen_difficulties: dict[str, set[str]] = defaultdict(set)
    total = 0

    for dif, pool in REGISTRY.items():
        for name, fn, _w in pool:
            seen_difficulties[name].add(dif)
            for _ in range(N_PER_GEN):
                total += 1
                try:
                    triple = fn(dif)
                except Exception as e:
                    failures.append((name, dif, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
                    break  # don't repeat same crash
                err = _validate(triple, name)
                if err:
                    failures.append((name, dif, err))
                    break

    print(f"Ran {total} generator calls across {sum(len(p) for p in REGISTRY.values())} registry entries.")
    print(f"Distinct generator names: {len(seen_difficulties)}")
    if failures:
        print(f"\n{len(failures)} failures:")
        for name, dif, err in failures:
            print(f"  [{dif:6s}] {name}: {err.splitlines()[0]}")
        sys.exit(1)
    print("All generators OK.")


if __name__ == "__main__":
    main()
