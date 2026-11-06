"""Parse the ProofWiki XML dump into JSONL records.

ProofWiki is a MediaWiki site; the dump is a standard MediaWiki XML export
(one <page> per article, wikitext inside <text>). We stream pages with mwxml
(no full-file load), parse wikitext with wikitextparser, and emit one record
per theorem / definition / axiom / corollary page. Other namespaces and
meta-pages (Help, User, Mathematician, Book, ...) are skipped.

Output schema: see ../../SCHEMA.md, stream `proofwiki_proofs.jsonl`.

Caveats (v0 — explicitly out of scope, document for future passes):
- Wikitext is preserved as-is. Templates like {{eqn|...}}, <math>...</math>
  blocks, and pipe-flattened wikilinks ([[Definition:Group|group]]) are
  kept verbatim. A downstream cleaner can normalize these later.
- We take the *first* heading whose title matches a statement keyword
  (Theorem/Lemma/Corollary/Definition/Axiom). Pages with both "Theorem"
  and "Lemma" sections will use the first one.
- Multi-revision dumps would yield only the latest revision (mwxml gives
  us all in order; we take the last). The standard latest.xml.gz already
  has only current revisions, so this is moot in practice.
"""
import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import IO

import mwxml
import wikitextparser as wtp

from ..schema import Proof, ProofWikiRecord


PROOF_HEADING_RE = re.compile(r"^proof(\s+\d+)?(\s*\(.*\))?$", re.IGNORECASE)
STATEMENT_HEADINGS = {"theorem", "lemma", "corollary"}
DEFINITION_HEADINGS = {"definition"}
AXIOM_HEADINGS = {"axiom", "axioms"}
ALSO_KNOWN_HEADINGS = {"also known as", "also denoted as", "also called"}

# A subpage suffix is a "proof subpage" if it contains the word "proof".
# ProofWiki's convention is to split multi-proof theorems into subpages
# like "Foo/Direct Proof", "Foo/Proof by Induction", "Foo/Proof 1", which
# the parent transcludes via {{:Foo/Direct Proof}}. Without merging these
# back to the parent, ~25% of theorems appear to have zero proofs.
PROOF_SUBPAGE_SUFFIX_RE = re.compile(r"\bproof\b", re.IGNORECASE)

# ProofWiki uses custom MediaWiki namespaces. The numeric IDs come from the
# dump's <siteinfo>; do NOT classify by title prefix — pages display as
# "Definition:Foo" but the namespace is the authoritative signal.
NAMESPACE_KIND: dict[int, str | None] = {
    0: None,         # Main: theorems, lemmas, corollaries — refine by sections
    100: "axiom",    # Axiom:*
    102: "definition",  # Definition:*
}
# Everything else (Talk=1, User=2, ProofWiki=4, File=6, MediaWiki=8,
# Template=10, Help=12, Category=14, Symbols=104, Mathematician=106,
# Book=108, Module=828, plus *_talk variants) is dropped.


def open_dump(path: Path) -> IO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def classify(title: str, namespace: int, sections: list) -> str | None:
    """Return record kind, or None to skip the page entirely."""
    if namespace not in NAMESPACE_KIND:
        return None
    forced = NAMESPACE_KIND[namespace]
    if forced is not None:
        return forced
    # Main namespace: distinguish theorem vs. corollary by sections / title.
    headings = {s.title.strip().lower() for s in sections if s.title}
    if "corollary" in headings or "/Corollary" in title:
        return "corollary"
    if headings & STATEMENT_HEADINGS:
        return "theorem"
    return None


def section_body(section) -> str:
    """Return section text without its heading line."""
    raw = str(section)
    parts = raw.split("\n", 1)
    return parts[1].strip() if len(parts) > 1 else ""


def bullet_items(section) -> list[str]:
    """Extract top-level bullet items from a section body."""
    items = []
    for line in section_body(section).splitlines():
        stripped = line.lstrip()
        if stripped.startswith("* ") or stripped.startswith("*\t"):
            items.append(stripped[2:].strip())
    return items


def page_to_record(title: str, namespace: int, wikitext: str) -> ProofWikiRecord | None:
    """Convert a single MediaWiki page to a ProofWikiRecord, or None to skip."""
    text = (wikitext or "").lstrip()
    if not text:
        return None
    if text.startswith("#REDIRECT") or text.startswith("#redirect"):
        return None

    parsed = wtp.parse(wikitext)
    sections = list(parsed.sections)
    kind = classify(title, namespace, sections)
    if kind is None:
        return None

    statement: str | None = None
    proofs: list[Proof] = []
    sources: list[str] = []
    also_known_as: list[str] = []

    for section in sections:
        if section.title is None:
            continue
        heading = section.title.strip().lower()
        if (
            heading in STATEMENT_HEADINGS
            or heading in DEFINITION_HEADINGS
            or heading in AXIOM_HEADINGS
        ):
            if statement is None:
                statement = section_body(section)
        elif PROOF_HEADING_RE.match(heading):
            proofs.append(Proof(name=section.title.strip(), text=section_body(section)))
        elif heading == "sources":
            sources = bullet_items(section)
        elif heading in ALSO_KNOWN_HEADINGS:
            also_known_as = bullet_items(section)

    categories: list[str] = []
    wikilinks_set: set[str] = set()
    for link in parsed.wikilinks:
        target = link.target.strip()
        if target.startswith("Category:"):
            categories.append(target[len("Category:"):])
        elif not target.startswith(("File:", "Image:")):
            wikilinks_set.add(target)

    return ProofWikiRecord(
        id=f"proofwiki:{title.replace(' ', '_')}",
        title=title,
        kind=kind,
        source_url=f"https://proofwiki.org/wiki/{title.replace(' ', '_')}",
        statement=statement,
        proofs=proofs,
        sources=sources,
        also_known_as=also_known_as,
        categories=categories,
        wikilinks=sorted(wikilinks_set),
    )


def split_proof_subpage(title: str, namespace: int) -> tuple[str, str] | None:
    """If title is a proof subpage like 'Foo/Direct Proof', return (parent, suffix).

    Otherwise return None. Only main-namespace pages with a `/` and a suffix
    matching the proof keyword qualify.
    """
    if namespace != 0 or "/" not in title:
        return None
    parent, _, suffix = title.rpartition("/")
    if not parent or not PROOF_SUBPAGE_SUFFIX_RE.search(suffix):
        return None
    return parent, suffix


def subpage_to_proof(suffix: str, wikitext: str) -> Proof:
    """Extract a Proof from a subpage's wikitext.

    Subpages may either embed an explicit `== Proof ==` section or be
    proof-body-only (often wrapped in <onlyinclude> for transclusion).
    """
    parsed = wtp.parse(wikitext)
    for section in parsed.sections:
        if section.title and PROOF_HEADING_RE.match(section.title.strip().lower()):
            return Proof(name=suffix, text=section_body(section))
    return Proof(name=suffix, text=wikitext.strip())


def parse_dump(
    dump_path: Path, out_path: Path, limit: int | None = None
) -> tuple[int, int]:
    """Stream the dump (two passes) and write JSONL. Returns (pages_scanned, records_kept).

    Pass 1 collects proof subpages keyed by their parent title; pass 2
    streams primary pages, attaches any matching subpage proofs, and emits.
    Two-pass because primaries and their proof subpages are not guaranteed
    to appear in any particular order in the dump.
    """
    print("pass 1/2: collecting proof subpages", file=sys.stderr)
    proof_subpages: dict[str, list[Proof]] = {}
    n_subpages = 0
    with open_dump(dump_path) as fp:
        for page in mwxml.Dump.from_file(fp):
            split = split_proof_subpage(page.title, page.namespace)
            if split is None:
                continue
            parent, suffix = split
            text = ""
            for revision in page:
                text = revision.text or ""
            if not text or text.lstrip().startswith(("#REDIRECT", "#redirect")):
                continue
            try:
                proof = subpage_to_proof(suffix, text)
            except Exception as e:
                print(f"  ERR subpage {page.title!r}: {e}", file=sys.stderr)
                continue
            proof_subpages.setdefault(parent, []).append(proof)
            n_subpages += 1
    print(
        f"pass 1 done: collected {n_subpages} proof subpages "
        f"covering {len(proof_subpages)} parent theorems",
        file=sys.stderr,
    )

    print("pass 2/2: emitting records", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    with open_dump(dump_path) as fp, out_path.open("w", encoding="utf-8") as out:
        for page in mwxml.Dump.from_file(fp):
            n_in += 1
            if split_proof_subpage(page.title, page.namespace) is not None:
                continue  # already absorbed in pass 1
            text = ""
            for revision in page:
                text = revision.text or ""
            try:
                rec = page_to_record(page.title, page.namespace, text)
            except Exception as e:
                print(f"  ERR parsing {page.title!r}: {e}", file=sys.stderr)
                continue
            if rec is None:
                continue
            extras = proof_subpages.pop(rec.title, None)
            if extras:
                rec.proofs.extend(extras)
            out.write(json.dumps(rec.to_jsonl_dict(), ensure_ascii=False) + "\n")
            n_out += 1
            if limit and n_out >= limit:
                break
            if n_in % 5000 == 0:
                print(f"  scanned={n_in} kept={n_out}", file=sys.stderr)
    if proof_subpages and limit is None:
        print(
            f"  note: {len(proof_subpages)} parent titles had proof subpages "
            f"but no primary record (likely subpages of pages whose parent "
            f"got dropped — Definition redirects, deleted parents, etc.)",
            file=sys.stderr,
        )
    return n_in, n_out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="math_lean.proofwiki.parse")
    p.add_argument("dump", type=Path, help="path to latest.xml or latest.xml.gz")
    p.add_argument("--out", type=Path, required=True, help="output JSONL path")
    p.add_argument("--limit", type=int, default=None, help="stop after N kept records")
    args = p.parse_args(argv)
    n_in, n_out = parse_dump(args.dump, args.out, args.limit)
    print(f"done: scanned={n_in} kept={n_out} -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
