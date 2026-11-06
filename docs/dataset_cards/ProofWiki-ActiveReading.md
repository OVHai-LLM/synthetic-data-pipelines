---
license: cc-by-sa-4.0
language:
- en
task_categories:
- text-generation
tags:
- mathematics
- theorem-proving
- lean4
- synthetic
- active-reading
- pretraining
- annealing
pretty_name: ProofWiki Active Reading
size_categories:
- 100K<n<1M
configs:
- config_name: active_reading
  default: true
  data_files:
  - split: train
    path: active_reading/train-*.parquet
- config_name: proofwiki
  data_files:
  - split: train
    path: proofwiki/train-*.parquet
---

# ProofWiki Active Reading

Synthetic mathematical study material produced by applying the **Active Reading** recipe
([Meta, arXiv:2508.09494](https://arxiv.org/abs/2508.09494)) to the full [ProofWiki](https://proofwiki.org) corpus.

For every source theorem the generator first invents *document-specific study strategies*,
then executes each strategy against the source document to produce a **study artifact** —
a worked numerical verification, a proof-dependency tree, a blind reconstruction, a Lean 4
signature draft, a tactic-mode skeleton, and so on. The artifacts, not the strategies, are
the training data.

The corpus was built as the `Active-Reading-WikiProof` slice of the math bucket of a 2 T-token
annealing mix, and is released here as a companion asset to the accompanying tech report.

## Contents

| Config | Rows | Description |
|---|---:|---|
| `active_reading` (default) | **185,319** | One row per study artifact. 18,532 source theorems × 10 artifacts (2 flavours × 5 strategies). **178.5 M Qwen2.5 tokens.** |
| `proofwiki` | **54,318** | The parsed ProofWiki source corpus: theorems, definitions, axioms and corollaries with all their proofs, in lightly-cleaned wikitext. |

```python
from datasets import load_dataset

ar = load_dataset("OVHaiLLM/ProofWiki-ActiveReading", "active_reading", split="train")
pw = load_dataset("OVHaiLLM/ProofWiki-ActiveReading", "proofwiki", split="train")

# artifacts join back onto their source document
ar[0]["doc_id"]   # 'proofwiki:Sine_of_Angle_plus_Full_Angle'
```

## How it was built

A two-stage vLLM pipeline, `google/gemma-4-31b-it` served as
[`RedHatAI/gemma-4-31B-it-FP8-Dynamic`](https://huggingface.co/RedHatAI/gemma-4-31B-it-FP8-Dynamic)
for both stages.

**Stage 1 — strategy generation.** Each source document is shown to the model twice, once per
*flavour*, and the model proposes 5 concrete, document-specific study strategies:

- **`agnostic`** — general mathematical mastery. The prompt asks for a spread of cognitive
  modes (active recall, paraphrase, working a concrete example, weakening a hypothesis,
  mapping lemma dependencies, comparing proof routes, contrasting with neighbouring theorems,
  building a counter-example).
- **`lean`** — strategies aimed specifically at the downstream task of formalising the
  statement in **Lean 4 / mathlib4**: drafting a theorem signature with the right typeclasses,
  mapping each prose step to a mathlib lemma, restating the goal with `apply` / `refine`,
  predicting tactics, building a minimal `import` list.

**Stage 2 — strategy application.** Each (document, strategy) pair is executed independently,
yielding one markdown study artifact (≤ 4096 generated tokens).

Source-document filter: `kind == theorem`, proof text ≥ 200 characters, document truncated to
8000 characters. That selects 18,532 of the 23,936 ProofWiki theorems.

The pipeline code lives in the `math_lean.active_reading` package of the training repo
(`prompts.py`, `pipeline.py`, `client.py`).

## `active_reading` schema

| Field | Type | Notes |
|---|---|---|
| `id` | string | `{doc_id}#aug{i}`, `i` ∈ 0–9 |
| `doc_id` | string | joins onto `proofwiki.id` |
| `title` | string | theorem title |
| `kind` | string | always `theorem` in this config |
| `source_url` | string | ProofWiki page |
| `flavour` | string | `agnostic` (92,660 rows) or `lean` (92,659) |
| `strategy_index` | int32 | 0–4, position within the flavour's 5 strategies |
| `strategy_name` | string | model-invented name, e.g. *Dependency Mapping*, *Typeclass and Signature Drafting* |
| `strategy_body` | string | the stage-1 strategy text that produced this artifact |
| `text` | string | **the study artifact** — the training text |
| `n_chars` | int32 | characters in `text` |
| `token_count` | int32 | Qwen2.5 tokens in `text` |

## `proofwiki` schema

| Field | Type | Notes |
|---|---|---|
| `id`, `title`, `kind`, `source_url` | string | `kind` ∈ definition (29,079), theorem (23,936), corollary (936), axiom (367) |
| `statement` | string | wikitext, LaTeX preserved |
| `proofs` | list&lt;{name, text}&gt; | often several per theorem; empty for most definitions |
| `n_proofs` | int32 | |
| `categories`, `wikilinks`, `sources`, `also_known_as` | list&lt;string&gt; | ProofWiki metadata; `sources` holds `{{BookReference}}` citations |
| `used_for_active_reading` | bool | true for the 18,532 documents that passed the filter |
| `active_reading_chunk` | string | the exact rendered document text fed to the generator (null otherwise) |

## Statistics

- **185,319 artifacts** over **18,532** distinct theorems (one generation failed and is dropped).
- **178,472,691 Qwen2.5 tokens** — 76.98 M `agnostic` + 101.49 M `lean`. Median artifact 958 tokens.
- **49,257 distinct strategy names** — strategies are re-invented per document, so the long tail
  is enormous; the head is dominated by recurring shapes:

| Strategy | Count |
|---|---:|
| Dependency Mapping | 8,577 |
| Typeclass and Signature Drafting | 7,582 |
| Hypothesis Stress-Testing | 3,489 |
| Goal-First Refinement | 2,897 |
| Minimal Import Identification | 2,045 |
| Tactic-to-Step Mapping | 1,617 |
| Concrete Instantiation | 1,407 |
| Concrete Numerical Verification | 1,291 |

## Known limitations

- **The Lean 4 code is mostly not compilable.** In a separate validation pass, 120,801 Lean
  snippets were extracted from the `lean`-flavour artifacts and compiled against Mathlib:
  **10,157 compiled cleanly (8.4 %)**, 7,955 compiled only with `sorry` (6.6 %), and 102,682
  failed (85.0 %). A repair pass over every failing snippet with
  [`Goedel-Prover-V2-32B`](https://huggingface.co/Goedel-LM/Goedel-Prover-V2-32B) recovered
  **none** of them. Treat the `lean`-flavour artifacts as *pedagogical prose about
  formalisation*, useful as pretraining signal, **not** as a verified formal corpus. The
  verification and repair records are not included in this release.
- Artifacts are model-generated and **not fact-checked**; they can contain incorrect
  mathematics, hallucinated mathlib lemma names, and misattributed dependencies.
- Only `theorem`-kind documents were augmented. Definitions, axioms and corollaries appear in
  the `proofwiki` config only.
- Long documents were truncated to 8000 characters before generation, so artifacts for large
  pages may reference only part of the source.
- The source wikitext is lightly cleaned but still contains ProofWiki markup
  (`{{eqn}}`, `[[…]]`, `<onlyinclude>`), by design — that is what the generator saw.

## Licensing

- The `proofwiki` config and all source text are derived from ProofWiki, licensed
  **[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)**. The whole dataset is
  released under the same licence.
- The `active_reading` artifacts are outputs of a Gemma model and are additionally subject to
  the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and Prohibited Use Policy.

## Citation

Active Reading recipe:

```bibtex
@article{activereading2025,
  title   = {Learning Facts at Scale with Active Reading},
  journal = {arXiv preprint arXiv:2508.09494},
  year    = {2025},
  url     = {https://arxiv.org/abs/2508.09494}
}
```

Please also credit ProofWiki as the source corpus.
