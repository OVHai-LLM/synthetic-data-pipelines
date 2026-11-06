---
license: apache-2.0
language: [ar, bg, ca, cs, da, de, el, en, es, fi, fr, he, hi, hr, hu, it, ja, ko, nl, no, pl, pt, ro, ru, sk, sl, sv, tr, uk, zh]
task_categories: [text-generation, question-answering]
tags: [mathematics, synthetic, multilingual, reasoning, llm-judge, pretraining, annealing]
pretty_name: Math-Enrich Multilingual
size_categories: [10M<n<100M]
configs:
- config_name: enrich
  default: true
  data_files:
  - split: train
    path: enrich/*/train-*.parquet
- config_name: seeds
  data_files:
  - split: train
    path: seeds/*/train-*.parquet
- config_name: enrich.ar
  data_files:
  - split: train
    path: enrich/ar/train-*.parquet
- config_name: seeds.ar
  data_files:
  - split: train
    path: seeds/ar/train-*.parquet
- config_name: enrich.bg
  data_files:
  - split: train
    path: enrich/bg/train-*.parquet
- config_name: seeds.bg
  data_files:
  - split: train
    path: seeds/bg/train-*.parquet
- config_name: enrich.ca
  data_files:
  - split: train
    path: enrich/ca/train-*.parquet
- config_name: seeds.ca
  data_files:
  - split: train
    path: seeds/ca/train-*.parquet
- config_name: enrich.cs
  data_files:
  - split: train
    path: enrich/cs/train-*.parquet
- config_name: seeds.cs
  data_files:
  - split: train
    path: seeds/cs/train-*.parquet
- config_name: enrich.da
  data_files:
  - split: train
    path: enrich/da/train-*.parquet
- config_name: seeds.da
  data_files:
  - split: train
    path: seeds/da/train-*.parquet
- config_name: enrich.de
  data_files:
  - split: train
    path: enrich/de/train-*.parquet
- config_name: seeds.de
  data_files:
  - split: train
    path: seeds/de/train-*.parquet
- config_name: enrich.el
  data_files:
  - split: train
    path: enrich/el/train-*.parquet
- config_name: seeds.el
  data_files:
  - split: train
    path: seeds/el/train-*.parquet
- config_name: enrich.en
  data_files:
  - split: train
    path: enrich/en/train-*.parquet
- config_name: seeds.en
  data_files:
  - split: train
    path: seeds/en/train-*.parquet
- config_name: enrich.es
  data_files:
  - split: train
    path: enrich/es/train-*.parquet
- config_name: seeds.es
  data_files:
  - split: train
    path: seeds/es/train-*.parquet
- config_name: enrich.fi
  data_files:
  - split: train
    path: enrich/fi/train-*.parquet
- config_name: seeds.fi
  data_files:
  - split: train
    path: seeds/fi/train-*.parquet
- config_name: enrich.fr
  data_files:
  - split: train
    path: enrich/fr/train-*.parquet
- config_name: seeds.fr
  data_files:
  - split: train
    path: seeds/fr/train-*.parquet
- config_name: enrich.he
  data_files:
  - split: train
    path: enrich/he/train-*.parquet
- config_name: seeds.he
  data_files:
  - split: train
    path: seeds/he/train-*.parquet
- config_name: enrich.hi
  data_files:
  - split: train
    path: enrich/hi/train-*.parquet
- config_name: seeds.hi
  data_files:
  - split: train
    path: seeds/hi/train-*.parquet
- config_name: enrich.hr
  data_files:
  - split: train
    path: enrich/hr/train-*.parquet
- config_name: seeds.hr
  data_files:
  - split: train
    path: seeds/hr/train-*.parquet
- config_name: enrich.hu
  data_files:
  - split: train
    path: enrich/hu/train-*.parquet
- config_name: seeds.hu
  data_files:
  - split: train
    path: seeds/hu/train-*.parquet
- config_name: enrich.it
  data_files:
  - split: train
    path: enrich/it/train-*.parquet
- config_name: seeds.it
  data_files:
  - split: train
    path: seeds/it/train-*.parquet
- config_name: enrich.ja
  data_files:
  - split: train
    path: enrich/ja/train-*.parquet
- config_name: seeds.ja
  data_files:
  - split: train
    path: seeds/ja/train-*.parquet
- config_name: enrich.ko
  data_files:
  - split: train
    path: enrich/ko/train-*.parquet
- config_name: seeds.ko
  data_files:
  - split: train
    path: seeds/ko/train-*.parquet
- config_name: enrich.nl
  data_files:
  - split: train
    path: enrich/nl/train-*.parquet
- config_name: seeds.nl
  data_files:
  - split: train
    path: seeds/nl/train-*.parquet
- config_name: enrich.no
  data_files:
  - split: train
    path: enrich/no/train-*.parquet
- config_name: seeds.no
  data_files:
  - split: train
    path: seeds/no/train-*.parquet
- config_name: enrich.pl
  data_files:
  - split: train
    path: enrich/pl/train-*.parquet
- config_name: seeds.pl
  data_files:
  - split: train
    path: seeds/pl/train-*.parquet
- config_name: enrich.pt
  data_files:
  - split: train
    path: enrich/pt/train-*.parquet
- config_name: seeds.pt
  data_files:
  - split: train
    path: seeds/pt/train-*.parquet
- config_name: enrich.ro
  data_files:
  - split: train
    path: enrich/ro/train-*.parquet
- config_name: seeds.ro
  data_files:
  - split: train
    path: seeds/ro/train-*.parquet
- config_name: enrich.ru
  data_files:
  - split: train
    path: enrich/ru/train-*.parquet
- config_name: seeds.ru
  data_files:
  - split: train
    path: seeds/ru/train-*.parquet
- config_name: enrich.sk
  data_files:
  - split: train
    path: enrich/sk/train-*.parquet
- config_name: seeds.sk
  data_files:
  - split: train
    path: seeds/sk/train-*.parquet
- config_name: enrich.sl
  data_files:
  - split: train
    path: enrich/sl/train-*.parquet
- config_name: seeds.sl
  data_files:
  - split: train
    path: seeds/sl/train-*.parquet
- config_name: enrich.sv
  data_files:
  - split: train
    path: enrich/sv/train-*.parquet
- config_name: seeds.sv
  data_files:
  - split: train
    path: seeds/sv/train-*.parquet
- config_name: enrich.tr
  data_files:
  - split: train
    path: enrich/tr/train-*.parquet
- config_name: seeds.tr
  data_files:
  - split: train
    path: seeds/tr/train-*.parquet
- config_name: enrich.uk
  data_files:
  - split: train
    path: enrich/uk/train-*.parquet
- config_name: seeds.uk
  data_files:
  - split: train
    path: seeds/uk/train-*.parquet
- config_name: enrich.zh
  data_files:
  - split: train
    path: enrich/zh/train-*.parquet
- config_name: seeds.zh
  data_files:
  - split: train
    path: seeds/zh/train-*.parquet
---

# Math-Enrich Multilingual

**21,569,407 LLM-rewritten math problems in 30 languages, each graded by an LLM judge against a
programmatically generated ground-truth answer.** 17,733,053 pass the judge (82.2 %),
worth **7.37 B tokens**.

The distinguishing feature is that **every problem has a verifiable answer**. Problems are not
scraped or free-form generated — they are emitted by 122 deterministic generators (several
Z3-solver-backed) that produce the answer alongside the problem. An LLM then rewrites the bare
problem into natural prose in one of 30 languages, and a second LLM pass judges whether the rewrite
preserved the mathematics and reads well. Both the accepted and the **rejected** rewrites ship,
with their scores.

Built as the math-generator slice of a 2 T-token annealing mix, and released as a companion asset
to the accompanying tech report. See also
[ProofWiki-ActiveReading](https://huggingface.co/datasets/OVHaiLLM/ProofWiki-ActiveReading) and
[Wikipedia-ActiveReading](https://huggingface.co/datasets/OVHaiLLM/Wikipedia-ActiveReading).

## Contents

| Config | Rows | Description |
|---|---:|---|
| `enrich` (default) | **21,569,407** | One row per rewrite: the seed problem and its true answer, the rewrite, and the full judge rubric. |
| `seeds` | **60,000,000** | The generated seed problems: bare statement, ground-truth answer, worked step. |

```python
from datasets import load_dataset

d = load_dataset("OVHaiLLM/Math-Enrich-Multilingual", "enrich.fr", split="train")

# training text for the accepted subset
kept = d.filter(lambda r: r["judge_overall_keep"])
text = f"Problem: {r['rewritten_problem']}\n\nSolution: {r['rewritten_solution']}"

# or mine the rejects: a rewrite the judge marked mathematically wrong
bad = d.filter(lambda r: r["judge_correctness"] == "no")
```

## How it was built

**Stage 1 — seed generation.** 60,000,000 problems from 122 named generators across
165 categories, at three difficulty levels
(medium 24,130,609, hard 18,592,715, simple 17,276,676). Each generator emits the problem, the
exact answer, and a worked `step_text`. 8,573,599 seeds (14.3 %) come from
hardcoded templates; the rest are randomly parameterised. Logic and puzzle categories
(`knights_knaves_z3`, `bool_sat`, …) are solved with Z3, so their answers are machine-checked.

**Stage 2 — rewrite.** `RedHatAI/gemma-4-31B-it-FP8-Dynamic` at temperature 0.8, served on 4×GH200
with DP=4. Each seed is rewritten under one of **7 named styles**, and the model is instructed to
preserve the numbers and the final answer exactly:

| Style | Rewrites |
|---|---:|
| `engineering_physics` | 3,388,385 |
| `hidden_operation` | 3,377,891 |
| `formal` | 3,375,697 |
| `word_problem` | 3,362,579 |
| `red_herring` | 3,355,013 |
| `narrative` | 3,352,685 |
| `mcq` | 1,357,157 |

`word_problem` sets the maths in a real-world scene; `narrative` tells it as a story;
`formal` restates it in strict mathematical language; `engineering_physics` casts it as an applied
problem; `mcq` produces multiple choice with distractors. Two are deliberately adversarial:
**`red_herring`** injects irrelevant-but-plausible numbers, and **`hidden_operation`** hides which
operation is required behind the phrasing.

**Stage 3 — judge.** The same model, at temperature 0.1, scores each rewrite as strict JSON on four
criteria: `correctness` (yes/no, with a free-text reason), and `naturalness`, `clarity` and
`difficulty_preservation` on 1–5 scales. `overall_keep` is the accept decision.

## What the judge found

This is the part we would point a reader at first:

| Judge outcome | Count |
|---|---:|
| Accepted (`overall_keep`) | 17,733,053 (82.2 %) |
| Rejected | 3,836,354 (17.8 %) |
| **Mathematically incorrect** (`correctness: no`) | **608,987** (2.8 %) |
| Scored ≤2 on naturalness | 3,601,520 (16.7 %) |

**The rewriter preserves the mathematics 97.1 % of the time, and
almost all rejections are about prose, not correctness.** Only 608,987 rewrites break
the maths, while 3,601,520 read badly enough to score ≤2 on naturalness. If you want volume over
polish, filtering on `judge_correctness == "yes"` instead of `judge_overall_keep` recovers roughly
15 percentage points of extra data whose answers are still sound.

Naturalness distribution: 1→561,642, 2→3,039,878, 3→2,796,774, 4→6,734,144, 5→8,427,067.

## Per-language breakdown

Splitting the rejections by cause across the 30 languages gives the cleanest result in this
dataset:

| Rejection cause | Range across languages |
|---|---|
| Mathematically incorrect | **2.0 % (`ar`) – 4.5 % (`en`)** |
| Naturalness ≤ 2 | **13.2 % (`ar`) – 25.2 % (`hu`)** |

**Mathematical fidelity is close to language-invariant; prose quality is not.** The spread in
accept rate is almost entirely a style effect — the rewriter carries the numbers across languages
about equally well, and differs mainly in how naturally it writes.

One confounder worth stating: English has the *highest* rate of judge-detected maths errors
(4.5 %), which is unlikely to mean English rewrites are the least accurate. The judge is the same
model doing the rewriting and is strongest in English, so it plausibly *detects* more of its own
errors there. Read the per-language `correctness` column as judge sensitivity as much as rewrite
quality, and do not rank languages by it.

| Code | Language | Seeds | Rewrites | Accepted | Accept % | Tokens (B) |
|---|---|---:|---:|---:|---:|---:|
| `ar` | Arabic | 2,000,000 | 711,875 | 617,251 | 86.7 % | 0.24 |
| `bg` | Bulgarian | 2,000,000 | 659,134 | 542,614 | 82.3 % | 0.26 |
| `ca` | Catalan | 2,000,000 | 691,178 | 568,188 | 82.2 % | 0.22 |
| `cs` | Czech | 2,000,000 | 676,664 | 557,162 | 82.3 % | 0.26 |
| `da` | Danish | 2,000,000 | 727,207 | 605,128 | 83.2 % | 0.23 |
| `de` | German | 2,000,000 | 735,427 | 604,548 | 82.2 % | 0.23 |
| `el` | Greek | 2,000,000 | 604,720 | 469,998 | 77.7 % | 0.37 |
| `en` | English | 2,000,000 | 930,047 | 726,265 | 78.1 % | 0.21 |
| `es` | Spanish | 2,000,000 | 752,687 | 638,811 | 84.9 % | 0.24 |
| `fi` | Finnish | 2,000,000 | 706,099 | 572,185 | 81.0 % | 0.23 |
| `fr` | French | 2,000,000 | 761,924 | 633,191 | 83.1 % | 0.23 |
| `he` | Hebrew | 2,000,000 | 652,760 | 541,900 | 83.0 % | 0.20 |
| `hi` | Hindi | 2,000,000 | 747,653 | 613,936 | 82.1 % | 0.45 |
| `hr` | Croatian | 2,000,000 | 696,707 | 571,791 | 82.1 % | 0.24 |
| `hu` | Hungarian | 2,000,000 | 675,567 | 469,368 | 69.5 % | 0.22 |
| `it` | Italian | 2,000,000 | 705,797 | 592,397 | 83.9 % | 0.23 |
| `ja` | Japanese | 2,000,000 | 799,711 | 655,791 | 82.0 % | 0.23 |
| `ko` | Korean | 2,000,000 | 741,741 | 630,552 | 85.0 % | 0.24 |
| `nl` | Dutch | 2,000,000 | 709,943 | 583,512 | 82.2 % | 0.23 |
| `no` | Norwegian | 2,000,000 | 728,729 | 607,198 | 83.3 % | 0.23 |
| `pl` | Polish | 2,000,000 | 689,687 | 576,174 | 83.5 % | 0.24 |
| `pt` | Portuguese | 2,000,000 | 755,346 | 648,393 | 85.8 % | 0.24 |
| `ro` | Romanian | 2,000,000 | 675,664 | 561,664 | 83.1 % | 0.24 |
| `ru` | Russian | 2,000,000 | 751,285 | 634,413 | 84.4 % | 0.25 |
| `sk` | Slovak | 2,000,000 | 664,833 | 522,616 | 78.6 % | 0.24 |
| `sl` | Slovenian | 2,000,000 | 682,621 | 513,765 | 75.3 % | 0.21 |
| `sv` | Swedish | 2,000,000 | 717,234 | 598,967 | 83.5 % | 0.23 |
| `tr` | Turkish | 2,000,000 | 739,581 | 628,893 | 85.0 % | 0.24 |
| `uk` | Ukrainian | 2,000,000 | 690,570 | 585,794 | 84.8 % | 0.29 |
| `zh` | Chinese | 2,000,000 | 787,016 | 660,588 | 83.9 % | 0.22 |
| | **Total** | **60,000,000** | **21,569,407** | **17,733,053** | **82.2 %** | **7.37** |

## Schemas

### `enrich`

| Field | Type | Notes |
|---|---|---|
| `id` | string | seed id; joins onto `seeds.id` |
| `language` | string | language the rewrite was written in |
| `seed_problem`, `seed_answer` | string | the bare problem and its **ground-truth** answer |
| `seed_step_text` | string | worked step from the generator |
| `seed_category`, `seed_generator`, `seed_difficulty` | string | 165 categories, 122 generators, simple/medium/hard |
| `seed_hardcoded` | bool | template vs randomly parameterised |
| `rewrite_prompt` | string | one of the 7 styles |
| `rewritten_problem`, `rewritten_solution` | string | the rewrite; solution ends `Answer: <value>` |
| `judge_correctness` | string | `yes` / `no` |
| `judge_correctness_reason` | string | the judge's justification |
| `judge_naturalness`, `judge_clarity`, `judge_difficulty_preservation` | int8 | 1–5, `-1` if unparsed |
| `judge_overall_keep` | bool | the accept decision |
| `judge_reject_reason` | string | set on pipeline-level failures (`no_problem_parsed`, `judge_parse_failed`) |
| `token_count` | int32 | Qwen2.5 tokens of the formatted document; `-1` when not accepted |

### `seeds`

| Field | Type | Notes |
|---|---|---|
| `id`, `problem`, `answer`, `step_text` | string | |
| `category`, `generator`, `difficulty` | string | |
| `hardcoded` | bool | |
| `shard_language` | string | which language shard drew this seed — **see the caveat below** |
| `used_in_enrich` | bool | true for the 21,569,407 seeds that were actually rewritten |

Most frequent generators:

| Generator | Seeds |
|---|---:|
| `stats_descriptive` | 2,762,830 |
| `mixed_integer` | 2,555,148 |
| `linear_equation` | 2,461,581 |
| `addition` | 2,236,815 |
| `system_2x2` | 2,143,712 |
| `subtraction` | 1,954,775 |
| `simple_interest` | 1,668,641 |
| `distance_2d` | 1,655,286 |
| `fractions` | 1,561,519 |
| `set_theory` | 1,543,589 |
| `sat_witness_z3` | 1,526,163 |
| `compare` | 1,483,647 |

## Known limitations

- **No diversity or representation evaluation was run — expect social bias and gender
  stereotyping.** Neither the seed problems nor the rewrites were audited for this. The seed
  generators emit bare mathematics, but the rewrite step deliberately injects people and settings:
  story-mode prompts (`word_problem`, `narrative`, `red_herring`, `hidden_operation`) append a
  style constraint drawing a random name from a pool of 259 and a random occupational scenario
  from a pool of 190, and `engineering_physics` draws from 85 engineering subfields. Those pools
  were hand-written to look demographically varied, but that intent was never verified: nobody
  checked the gender balance of the name pool, which names get paired with which occupations
  (pairing is uniformly random per record, so any correlation comes from the model, not the
  sampler), or how the model genders pronouns, roles and authority in the surrounding prose it
  writes freely. The judge rubric scores correctness, naturalness, clarity and difficulty — it has
  **no fairness or representation criterion**, so nothing in the accept/reject decision filters
  stereotyped output. Treat occupational and gender associations in this corpus as unmeasured and
  probably reflective of the generator model's priors. Audit before using it anywhere those
  associations matter.

- **Seed problems are always in English.** `shard_language` records which language shard drew a
  seed, not the language of `problem`. The 30 shards are independent random draws from the same
  English generators; the language only enters at the rewrite step. Do not read `seeds` as a
  multilingual corpus — `enrich` is where the 30 languages live.
- **Most seeds were never rewritten.** 21,569,407 of 60,000,000 seeds (35.9 %)
  have a rewrite; the rest are unused surplus, flagged `used_in_enrich = false`.
- **The judge is the same model as the rewriter** (`gemma-4-31B-it`), so its verdicts are
  self-assessment, not independent review. Expect the usual self-preference bias; the
  608,987 self-caught errors are a lower bound on the true error rate, not an estimate
  of it.
- **`correctness` is judged, not executed.** The seed answer is ground truth, but nothing
  re-derives the rewritten solution symbolically. A rewrite that reaches the right answer by wrong
  reasoning can pass.
- **Answers are exact, solutions are not verified step-by-step.** Treat `seed_answer` as reliable
  and `rewritten_solution` as model output.
- **0 rows have an unparsable judge response** and carry `-1` scores.
- Seeds may repeat across language shards: identical generator plus identical random parameters
  produce identical problem text roughly 18 % of the time between any two shards. Ids are distinct.
- **Not deduplicated against math benchmarks.** The generators are template-driven, so collisions
  with GSM8K/MATH-style items are possible in principle. No decontamination pass was run — check
  before using this for evaluation.

## Relation to the English pilot

An earlier 100 k-problem English-only run (`math_enrich_50k*_Gemma4-31`, 80,910 docs kept after the
same judge filter) preceded this one. It is fully superseded by the `en` shard here and is not
included.

## Licensing

Seed problems are generated by deterministic code and carry no upstream licence; the dataset is
released under **Apache-2.0**. The rewrites and judge verdicts are outputs of a Gemma model and are
additionally subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and Prohibited
Use Policy.
