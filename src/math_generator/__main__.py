"""CLI entry point: `python -m math_generator --n 10000 --out math_train.jsonl`."""

import argparse
import json
import random
import uuid

from .distractors import make_distractors, shuffled_choices
from .registry import (
    HARDCODED_LEVELS, CapacityAwareSampler, saturation_report,
)


def main():
    ap = argparse.ArgumentParser(description="Generate diverse math QA JSONL.")
    ap.add_argument("--n", type=int, default=10000, help="number of samples")
    ap.add_argument("--out", type=str, default="math_train.jsonl", help="output JSONL path")
    ap.add_argument("--difficulty", choices=["simple", "medium", "hard", "mix"], default="mix")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--report", action="store_true",
                    help="print per-generator saturation report at the end")
    ap.add_argument("--distractors", action="store_true",
                    help="attach up to 3 distractor answers + a 4-way MCQ form to each record")
    args = ap.parse_args()

    random.seed(args.seed)

    diff_weights = {"simple": 0.25, "medium": 0.40, "hard": 0.35}
    sampler = CapacityAwareSampler()

    seen = set()
    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        attempts = 0
        max_attempts = args.n * 8
        while written < args.n and attempts < max_attempts:
            attempts += 1
            dif = args.difficulty
            if dif == "mix":
                dif = random.choices(
                    list(diff_weights.keys()),
                    weights=list(diff_weights.values()),
                )[0]

            name, gen = sampler.pick(dif)
            try:
                prob, ans, meta = gen(dif)
            except Exception as e:
                continue

            # Track distinct outputs per generator (capacity-aware feedback).
            sampler.record(name, prob)

            # Skip exact duplicates across the whole run.
            if prob in seen:
                continue
            seen.add(prob)

            step = meta.get("step_text")
            if step:
                text = f"Q: {prob}\n{step}\nA: {ans}"
            else:
                text = f"Q: {prob}\nA: {ans}"

            meta["hardcoded"] = HARDCODED_LEVELS.get(name, 0)
            meta["generator"] = name

            if args.distractors:
                ds = make_distractors(ans, n=3)
                if ds:
                    meta["distractors"] = ds
                    options, correct = shuffled_choices(ans, ds)
                    meta["mcq"] = {
                        "options": [{"letter": l, "text": t} for l, t in options],
                        "correct": correct,
                    }

            rec = {
                "id": str(uuid.uuid4()),
                "problem": prob,
                "answer": ans,
                "text": text,
                "meta": {"difficulty": dif, **meta},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} samples to {args.out} (after {attempts} attempts)")

    if args.report:
        print("\nSaturation report (generator, distinct, capacity, % saturated):")
        for name, count, cap, sat in saturation_report(sampler):
            cap_str = "∞" if cap < 0 else str(cap)
            print(f"  {name:32s}  {count:5d}  /  {cap_str:>6s}   ({sat * 100:5.1f}%)")


if __name__ == "__main__":
    main()
