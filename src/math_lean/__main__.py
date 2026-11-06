import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="math_lean")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pw = sub.add_parser("proofwiki", help="ingest the ProofWiki XML dump")
    pw_sub = pw.add_subparsers(dest="proofwiki_cmd", required=True)

    pw_dl = pw_sub.add_parser("download", help="fetch latest.xml.gz")
    pw_dl.add_argument(
        "--out",
        type=Path,
        default=Path("raw/proofwiki/latest.xml.gz"),
        help="destination path",
    )

    pw_parse = pw_sub.add_parser("parse", help="parse XML dump to JSONL")
    pw_parse.add_argument("dump", type=Path, help="path to latest.xml or latest.xml.gz")
    pw_parse.add_argument(
        "--out", type=Path, required=True, help="output JSONL path"
    )
    pw_parse.add_argument(
        "--limit", type=int, default=None, help="stop after N kept records"
    )

    ld = sub.add_parser("lean_dojo", help="convert LeanDojo Benchmark 4 parquet to JSONL")
    ld_sub = ld.add_subparsers(dest="lean_dojo_cmd", required=True)
    ld_conv = ld_sub.add_parser("convert", help="parquet → theorems.jsonl + tactic_traces.jsonl")
    ld_conv.add_argument("parquet_dir", type=Path, help="dir with {train,validation,test}.parquet")
    ld_conv.add_argument("--out", type=Path, required=True, help="output directory")

    pp = sub.add_parser("postprocess", help="filter/dedupe/tag raw Lean extractor output")
    pp.add_argument("raw_dir", help="directory of raw .ndjson from the Lean extractor")
    pp.add_argument("--out", required=True, help="output directory for jsonl streams")

    val = sub.add_parser("validate", help="schema-check existing jsonl files")
    val.add_argument("path", help="jsonl file to validate")

    args = parser.parse_args()

    if args.cmd == "proofwiki":
        if args.proofwiki_cmd == "download":
            from .proofwiki.download import download

            download(args.out)
            return 0
        if args.proofwiki_cmd == "parse":
            from .proofwiki.parse import parse_dump

            n_in, n_out = parse_dump(args.dump, args.out, args.limit)
            print(
                f"done: scanned={n_in} kept={n_out} -> {args.out}", file=sys.stderr
            )
            return 0
    if args.cmd == "lean_dojo":
        if args.lean_dojo_cmd == "convert":
            from .lean_dojo.convert import convert_dataset, write_manifest

            stats = convert_dataset(args.parquet_dir, args.out)
            manifest_path = write_manifest(args.out, stats)
            print(f"\ndone: {stats}", file=sys.stderr)
            print(f"manifest: {manifest_path}", file=sys.stderr)
            return 0
    if args.cmd == "postprocess":
        raise NotImplementedError("postprocess: pending Lean-side extractor")
    if args.cmd == "validate":
        raise NotImplementedError("validate: pending schema enforcement pass")
    return 1


if __name__ == "__main__":
    sys.exit(main())
