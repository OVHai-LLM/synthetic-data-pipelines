"""Fetch the nightly ProofWiki XML dump.

Source: https://proofwiki.org/xmldump/latest.xml.gz
License: CC-BY-SA 3.0 (record `dump_date` in the manifest for attribution).
"""
import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

DUMP_URL = "https://proofwiki.org/xmldump/latest.xml.gz"


def download(out_path: Path, url: str = DUMP_URL) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"GET {url}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "math_lean/0.0.1"})
    with urllib.request.urlopen(req) as resp, out_path.open("wb") as fp:
        shutil.copyfileobj(resp, fp)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"saved {out_path} ({size_mb:.1f} MB)", file=sys.stderr)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="math_lean.proofwiki.download")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("raw/proofwiki/latest.xml.gz"),
        help="destination path (default: raw/proofwiki/latest.xml.gz)",
    )
    p.add_argument("--url", default=DUMP_URL, help="override source URL")
    args = p.parse_args(argv)
    download(args.out, args.url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
