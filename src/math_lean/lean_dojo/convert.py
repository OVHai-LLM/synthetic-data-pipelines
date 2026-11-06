"""Convert LeanDojo Benchmark 4 parquet files into our JSONL schema.

Source: https://huggingface.co/datasets/JohnYang88/lean-dojo-mathlib4

LeanDojo's data is locator-based: each row gives us a file path, theorem
name, line/column range, and (for ~49% of theorems) per-step tactic traces
with `state_before` / `tactic` / `state_after` triples. It does NOT include
the formal statement or proof text — those live in mathlib4 source at the
pinned commit and can be backfilled later by reading the source files.

v0 emits two streams:
- `theorems.jsonl`     metadata only (id, full_name, file_path, source_line, category).
                        formal_statement / formal_proof / imports / dependencies are null.
- `tactic_traces.jsonl` full (state_before, tactic, state_after) triples — the
                        highest-value pretraining payload.

A `manifest.json` is written next to the JSONL with the mathlib commit SHA
so source-text backfill is reproducible.
"""
import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from ..schema import TacticTraceRecord, TheoremRecord


SPLITS = ("train", "validation", "test")


def category_from_path(file_path: str) -> str:
    """`Mathlib/<Cat>/...` -> `<cat>`. Anything else -> 'other'."""
    parts = file_path.split("/")
    if len(parts) >= 2 and parts[0] == "Mathlib":
        return parts[1].lower()
    return "other"


def module_from_path(file_path: str) -> str:
    """`Mathlib/Foo/Bar.lean` -> `Mathlib.Foo.Bar`."""
    stem = file_path[:-len(".lean")] if file_path.endswith(".lean") else file_path
    return stem.replace("/", ".")


def _count_goals(state: str) -> int:
    """Heuristic goal count by '⊢' markers (one per goal)."""
    if not state or state == "no goals":
        return 0
    return state.count("⊢")


def row_to_records(row: dict) -> tuple[TheoremRecord, list[TacticTraceRecord]]:
    """Convert one parquet row to (theorem, [tactic_traces])."""
    file_path = row["file_path"]
    full_name = row["full_name"]
    start = row.get("start") or [0, 0]

    theorem_id = f"mathlib:{full_name}"
    theorem = TheoremRecord(
        id=theorem_id,
        module=module_from_path(file_path),
        category=category_from_path(file_path),
        kind="theorem",
        source_file=file_path,
        source_line=int(start[0]) if start else 0,
    )

    raw = row.get("traced_tactics") or "[]"
    try:
        tactics = json.loads(raw)
    except json.JSONDecodeError:
        return theorem, []

    traces: list[TacticTraceRecord] = []
    n = sum(1 for t in tactics if isinstance(t, dict))
    kept = 0
    for i, t in enumerate(tactics):
        if not isinstance(t, dict):
            continue
        kept += 1
        traces.append(
            TacticTraceRecord(
                theorem_id=theorem_id,
                step_idx=i,
                goal_before=t.get("state_before", ""),
                tactic=t.get("tactic", ""),
                goal_after=t.get("state_after", ""),
                n_goals_before=_count_goals(t.get("state_before", "")),
                n_goals_after=_count_goals(t.get("state_after", "")),
                is_terminal=(kept == n),
            )
        )
    return theorem, traces


def convert_dataset(
    parquet_dir: Path,
    out_dir: Path,
    splits: tuple[str, ...] = SPLITS,
) -> dict:
    """Convert all splits to theorems.jsonl + tactic_traces.jsonl. Returns stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    theorems_path = out_dir / "theorems.jsonl"
    traces_path = out_dir / "tactic_traces.jsonl"

    n_theorems = n_traces = 0
    commits: set[str] = set()

    with theorems_path.open("w", encoding="utf-8") as t_out, \
         traces_path.open("w", encoding="utf-8") as tr_out:
        for split in splits:
            parquet_path = parquet_dir / f"{split}.parquet"
            if not parquet_path.exists():
                print(f"  skipping missing {parquet_path}", file=sys.stderr)
                continue
            print(f"converting {split}: {parquet_path}", file=sys.stderr)
            tbl = pq.read_table(parquet_path)
            split_thm = split_tr = 0
            for row in tbl.to_pylist():
                if commit := row.get("commit"):
                    commits.add(commit)
                theorem, traces = row_to_records(row)
                t_out.write(json.dumps(theorem.to_jsonl_dict(), ensure_ascii=False) + "\n")
                split_thm += 1
                for tr in traces:
                    tr_out.write(json.dumps(tr.to_jsonl_dict(), ensure_ascii=False) + "\n")
                    split_tr += 1
            n_theorems += split_thm
            n_traces += split_tr
            print(f"  {split} done: +{split_thm} theorems, +{split_tr} tactic traces",
                  file=sys.stderr)

    return {
        "theorems": n_theorems,
        "tactic_traces": n_traces,
        "mathlib_commits": sorted(commits),
    }


def write_manifest(out_dir: Path, stats: dict) -> Path:
    manifest = {
        "source": "LeanDojo Benchmark 4 (mathlib4)",
        "huggingface_dataset": "JohnYang88/lean-dojo-mathlib4",
        "licenses": {
            "mathlib4_content": "Apache-2.0",
            "leandojo_extraction": "MIT",
        },
        "schema_version": "0",
        **stats,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="math_lean.lean_dojo.convert")
    p.add_argument(
        "parquet_dir",
        type=Path,
        help="directory with train.parquet / validation.parquet / test.parquet",
    )
    p.add_argument("--out", type=Path, required=True, help="output directory")
    args = p.parse_args(argv)
    stats = convert_dataset(args.parquet_dir, args.out)
    manifest_path = write_manifest(args.out, stats)
    print(f"\ndone: {stats}", file=sys.stderr)
    print(f"manifest: {manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
