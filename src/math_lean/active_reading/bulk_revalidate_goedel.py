"""bulk_revalidate_goedel.py — Lean-check every `final_code` in the goedel_retry
output files, fill in the real verdict (the original run had no working Lean,
so verdicts were all 'other'/'None').

CPU-bound, no GPU needed. Parallelised via ProcessPool.

Reads:  $ANNEAL_ROOT/proofwiki/active_reading.goedel_retry.job*of4.shard*.jsonl
Writes: same path with `.revalidated.jsonl` suffix
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# import the existing Lean check helper (handles import-rewriting + classify)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from math_lean.active_reading.lean_check import check_one


def revalidate_one(args):
    """Worker: rerun Lean on a record's final_code; return enriched record."""
    rec, project, lake_bin, timeout_sec = args
    code = rec.get("final_code") or ""
    if not code.strip():
        # no code generated (model_error); leave verdict as-is
        return {**rec, "revalidate_skipped": "no_final_code"}
    try:
        res = check_one({"code": code}, project=project,
                        lake_bin=lake_bin, timeout_sec=timeout_sec)
    except Exception as e:
        return {**rec, "revalidate_error": str(e)[:200]}
    # overwrite final_verdict with the freshly-computed one
    rec["final_verdict"] = res["verdict"]
    rec["revalidated"] = True
    rec["revalidate_stderr"] = res.get("stderr", "")[:500]
    rec["revalidate_elapsed_sec"] = res.get("elapsed_sec")
    rec["improved"] = (res["verdict"] in ("ok", "ok_with_sorry"))
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs",
        default=os.environ.get("ANNEAL_ROOT", "data") + "/proofwiki/"
                "active_reading.goedel_retry.job*of4.shard*.jsonl",
        help="glob pattern for goedel_retry output files")
    p.add_argument("--project",
        default=os.environ.get("LEAN_TOOLCHAIN", "lean_toolchain") + "/ar_lean_check")
    p.add_argument("--lake-bin",
        default=os.environ.get("LEAN_TOOLCHAIN", "lean_toolchain") + "/.elan/bin/lake")
    p.add_argument("--workers", type=int, default=64,
        help="parallel Lean processes (each is single-threaded; HPC node has 288 cores)")
    p.add_argument("--timeout-sec", type=int, default=120,
        help="per-snippet wall budget for lake env lean")
    p.add_argument("--limit", type=int, default=0, help="cap records (0 = all)")
    p.add_argument("--output-suffix", default=".revalidated.jsonl")
    args = p.parse_args()

    files = sorted(glob.glob(args.inputs))
    if not files:
        sys.exit(f"no files matched: {args.inputs}")
    print(f"[{time.strftime('%H:%M:%S')}] {len(files)} input files")

    # Prepare the project dir for tempfile.mkstemp (used by check_one)
    Path(args.project).mkdir(parents=True, exist_ok=True)
    os.environ["AR_LEAN_PROJECT"] = args.project
    os.environ["AR_LEAN_LAKE_BIN"] = args.lake_bin

    n_done = n_ok = n_err = n_skip = 0
    t0 = time.time()
    for fpath in files:
        out_path = fpath + args.output_suffix
        records = []
        kept = []
        with open(fpath) as f:
            for i, line in enumerate(f):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append(rec)
                code = rec.get("final_code")
                if code and isinstance(code, str) and len(code.strip()) > 5:
                    kept.append(rec)
                if args.limit and len(kept) >= args.limit:
                    break
        # write the no-code records straight to output (marked skipped) so we
        # don't lose them — but don't waste worker time on them.
        with open(out_path, "w") as fout:
            for r in records:
                code = r.get("final_code")
                if not (code and isinstance(code, str) and len(code.strip()) > 5):
                    fout.write(json.dumps({**r, "revalidate_skipped": "no_final_code"},
                                          ensure_ascii=False) + "\n")
                    n_done += 1
                    n_skip += 1
        print(f"[{time.strftime('%H:%M:%S')}] {Path(fpath).name}: "
              f"{len(records)} records read, {len(kept)} have code "
              f"({len(records)-len(kept)} pre-skipped) → {Path(out_path).name}")

        # append-mode for the work that actually requires lake
        with open(out_path, "a") as fout, \
             ProcessPoolExecutor(max_workers=args.workers) as ex:
            jobs = [(r, args.project, args.lake_bin, args.timeout_sec)
                    for r in kept]
            for fut in as_completed([ex.submit(revalidate_one, j) for j in jobs]):
                r = fut.result()
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_done += 1
                if r.get("revalidate_skipped"):
                    n_skip += 1
                elif r.get("revalidate_error"):
                    n_err += 1
                elif r.get("improved"):
                    n_ok += 1
                if n_done % 50 == 0:
                    el = time.time() - t0
                    rate = n_done / el
                    print(f"  {n_done} processed | ok {n_ok} | skip {n_skip} | "
                          f"err {n_err} | {rate:.1f} rec/s | "
                          f"ETA {(len(records) - n_done) / max(rate,0.01) / 60:.0f} min",
                          file=sys.stderr)

    print(f"\n[{time.strftime('%H:%M:%S')}] DONE: {n_done} processed, "
          f"{n_ok} OK ({100*n_ok/max(n_done,1):.1f}%), "
          f"{n_skip} skipped (no code), {n_err} errors")
    print(f"  Wall: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
