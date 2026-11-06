"""Build a check-style JSONL from a lean_retry JSONL.

For each retry record, emit a row that looks like a lean_check output —
{verdict, stdout, code, ...} — using the retry's `retry_check` if it
exists, otherwise the original failing record copied through. This lets
the next retry round consume the previous round's results as input.

Usage:
    python compose_after_retry.py <retry_jsonl> <out_jsonl>
"""

from __future__ import annotations

import json
import sys


def main(retry_path: str, out_path: str) -> None:
    n_total = 0
    n_ok = 0
    n_no_retry = 0
    with open(retry_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            rt = json.loads(line)
            n_total += 1
            chk = rt.get("retry_check")
            if chk:
                rec = {
                    "doc_id": rt["doc_id"],
                    "title": rt.get("title"),
                    "strategy_name": rt.get("strategy_name"),
                    "snippet_index": rt["snippet_index"],
                    "code": rt.get("retry_extracted") or "",
                    "verdict": chk["verdict"],
                    "stdout": chk.get("stdout") or "",
                }
                if chk["verdict"] in ("ok", "ok_with_sorry"):
                    n_ok += 1
            else:
                n_no_retry += 1
                rec = {
                    "doc_id": rt["doc_id"],
                    "title": rt.get("title"),
                    "strategy_name": rt.get("strategy_name"),
                    "snippet_index": rt["snippet_index"],
                    "code": rt.get("code", ""),
                    "verdict": rt.get("verdict", "other"),
                    "stdout": rt.get("stdout", ""),
                }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  {n_total} retry rows → {out_path}", file=sys.stderr)
    print(f"  ok/ok_with_sorry after this round: {n_ok}", file=sys.stderr)
    print(f"  retry rows with no retry_check (kept as-is): {n_no_retry}",
          file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
