"""Compose the final per-snippet verdict file by overlaying retry rounds
onto the original lean_check output.

Output is a JSONL with one row per snippet, holding the *latest* verdict
(check → retry1 → retry2). Same shape as a lean_check output, suitable
to feed into to_pretraining.

Usage:
    python compose_final_verdicts.py <check> <retry1> <retry2> <out>

`retry1` and `retry2` paths may be empty strings to skip that overlay.
"""

from __future__ import annotations

import json
import sys


def k(r):
    return (r["doc_id"], r.get("strategy_name"), r["snippet_index"])


def overlay(rows: list[dict], retry_path: str) -> list[dict]:
    if not retry_path:
        return rows
    idx = {}
    with open(retry_path) as f:
        for line in f:
            rt = json.loads(line)
            idx[k(rt)] = rt
    out = []
    for r in rows:
        rt = idx.get(k(r))
        if rt and rt.get("retry_check"):
            chk = rt["retry_check"]
            new = dict(r)
            new["verdict"] = chk["verdict"]
            new["stdout"] = chk.get("stdout") or ""
            new["code"] = rt.get("retry_extracted") or ""
            out.append(new)
        else:
            out.append(r)
    return out


def main(check_path: str, retry1_path: str, retry2_path: str, out_path: str) -> None:
    with open(check_path) as f:
        rows = [json.loads(l) for l in f]
    print(f"  loaded {len(rows)} initial check rows", file=sys.stderr)

    rows = overlay(rows, retry1_path)
    print(f"  after retry1 overlay: {sum(1 for r in rows if r['verdict'] in ('ok','ok_with_sorry'))}"
          f" / {len(rows)} ok+sorry", file=sys.stderr)

    rows = overlay(rows, retry2_path)
    print(f"  after retry2 overlay: {sum(1 for r in rows if r['verdict'] in ('ok','ok_with_sorry'))}"
          f" / {len(rows)} ok+sorry", file=sys.stderr)

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
