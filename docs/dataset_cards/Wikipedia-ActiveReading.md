---
license: cc-by-sa-4.0
language: [ar, bg, bs, ca, cs, da, et, fi, ga, he, hr, hu, is, it, ka, ko, lb, lt, lv, no, ro, sk, sl, sq, sr, sv, tr, uk, zh]
task_categories: [text-generation]
tags: [multilingual, wikipedia, synthetic, active-reading, pretraining, annealing]
pretty_name: Wikipedia Active Reading (29 languages)
size_categories: [10M<n<100M]
configs:
- config_name: active_reading
  default: true
  data_files:
  - split: train
    path: active_reading/*/train-*.parquet
- config_name: wikipedia
  data_files:
  - split: train
    path: wikipedia/*/train-*.parquet
- config_name: active_reading.ar
  data_files:
  - split: train
    path: active_reading/ar/train-*.parquet
- config_name: wikipedia.ar
  data_files:
  - split: train
    path: wikipedia/ar/train-*.parquet
- config_name: active_reading.bg
  data_files:
  - split: train
    path: active_reading/bg/train-*.parquet
- config_name: wikipedia.bg
  data_files:
  - split: train
    path: wikipedia/bg/train-*.parquet
- config_name: active_reading.bs
  data_files:
  - split: train
    path: active_reading/bs/train-*.parquet
- config_name: wikipedia.bs
  data_files:
  - split: train
    path: wikipedia/bs/train-*.parquet
- config_name: active_reading.ca
  data_files:
  - split: train
    path: active_reading/ca/train-*.parquet
- config_name: wikipedia.ca
  data_files:
  - split: train
    path: wikipedia/ca/train-*.parquet
- config_name: active_reading.cs
  data_files:
  - split: train
    path: active_reading/cs/train-*.parquet
- config_name: wikipedia.cs
  data_files:
  - split: train
    path: wikipedia/cs/train-*.parquet
- config_name: active_reading.da
  data_files:
  - split: train
    path: active_reading/da/train-*.parquet
- config_name: wikipedia.da
  data_files:
  - split: train
    path: wikipedia/da/train-*.parquet
- config_name: active_reading.et
  data_files:
  - split: train
    path: active_reading/et/train-*.parquet
- config_name: wikipedia.et
  data_files:
  - split: train
    path: wikipedia/et/train-*.parquet
- config_name: active_reading.fi
  data_files:
  - split: train
    path: active_reading/fi/train-*.parquet
- config_name: wikipedia.fi
  data_files:
  - split: train
    path: wikipedia/fi/train-*.parquet
- config_name: active_reading.ga
  data_files:
  - split: train
    path: active_reading/ga/train-*.parquet
- config_name: wikipedia.ga
  data_files:
  - split: train
    path: wikipedia/ga/train-*.parquet
- config_name: active_reading.he
  data_files:
  - split: train
    path: active_reading/he/train-*.parquet
- config_name: wikipedia.he
  data_files:
  - split: train
    path: wikipedia/he/train-*.parquet
- config_name: active_reading.hr
  data_files:
  - split: train
    path: active_reading/hr/train-*.parquet
- config_name: wikipedia.hr
  data_files:
  - split: train
    path: wikipedia/hr/train-*.parquet
- config_name: active_reading.hu
  data_files:
  - split: train
    path: active_reading/hu/train-*.parquet
- config_name: wikipedia.hu
  data_files:
  - split: train
    path: wikipedia/hu/train-*.parquet
- config_name: active_reading.is
  data_files:
  - split: train
    path: active_reading/is/train-*.parquet
- config_name: wikipedia.is
  data_files:
  - split: train
    path: wikipedia/is/train-*.parquet
- config_name: active_reading.it
  data_files:
  - split: train
    path: active_reading/it/train-*.parquet
- config_name: wikipedia.it
  data_files:
  - split: train
    path: wikipedia/it/train-*.parquet
- config_name: active_reading.ka
  data_files:
  - split: train
    path: active_reading/ka/train-*.parquet
- config_name: wikipedia.ka
  data_files:
  - split: train
    path: wikipedia/ka/train-*.parquet
- config_name: active_reading.ko
  data_files:
  - split: train
    path: active_reading/ko/train-*.parquet
- config_name: wikipedia.ko
  data_files:
  - split: train
    path: wikipedia/ko/train-*.parquet
- config_name: active_reading.lb
  data_files:
  - split: train
    path: active_reading/lb/train-*.parquet
- config_name: wikipedia.lb
  data_files:
  - split: train
    path: wikipedia/lb/train-*.parquet
- config_name: active_reading.lt
  data_files:
  - split: train
    path: active_reading/lt/train-*.parquet
- config_name: wikipedia.lt
  data_files:
  - split: train
    path: wikipedia/lt/train-*.parquet
- config_name: active_reading.lv
  data_files:
  - split: train
    path: active_reading/lv/train-*.parquet
- config_name: wikipedia.lv
  data_files:
  - split: train
    path: wikipedia/lv/train-*.parquet
- config_name: active_reading.no
  data_files:
  - split: train
    path: active_reading/no/train-*.parquet
- config_name: wikipedia.no
  data_files:
  - split: train
    path: wikipedia/no/train-*.parquet
- config_name: active_reading.ro
  data_files:
  - split: train
    path: active_reading/ro/train-*.parquet
- config_name: wikipedia.ro
  data_files:
  - split: train
    path: wikipedia/ro/train-*.parquet
- config_name: active_reading.sk
  data_files:
  - split: train
    path: active_reading/sk/train-*.parquet
- config_name: wikipedia.sk
  data_files:
  - split: train
    path: wikipedia/sk/train-*.parquet
- config_name: active_reading.sl
  data_files:
  - split: train
    path: active_reading/sl/train-*.parquet
- config_name: wikipedia.sl
  data_files:
  - split: train
    path: wikipedia/sl/train-*.parquet
- config_name: active_reading.sq
  data_files:
  - split: train
    path: active_reading/sq/train-*.parquet
- config_name: wikipedia.sq
  data_files:
  - split: train
    path: wikipedia/sq/train-*.parquet
- config_name: active_reading.sr
  data_files:
  - split: train
    path: active_reading/sr/train-*.parquet
- config_name: wikipedia.sr
  data_files:
  - split: train
    path: wikipedia/sr/train-*.parquet
- config_name: active_reading.sv
  data_files:
  - split: train
    path: active_reading/sv/train-*.parquet
- config_name: wikipedia.sv
  data_files:
  - split: train
    path: wikipedia/sv/train-*.parquet
- config_name: active_reading.tr
  data_files:
  - split: train
    path: active_reading/tr/train-*.parquet
- config_name: wikipedia.tr
  data_files:
  - split: train
    path: wikipedia/tr/train-*.parquet
- config_name: active_reading.uk
  data_files:
  - split: train
    path: active_reading/uk/train-*.parquet
- config_name: wikipedia.uk
  data_files:
  - split: train
    path: wikipedia/uk/train-*.parquet
- config_name: active_reading.zh
  data_files:
  - split: train
    path: active_reading/zh/train-*.parquet
- config_name: wikipedia.zh
  data_files:
  - split: train
    path: wikipedia/zh/train-*.parquet
---

# Wikipedia Active Reading — 29 languages

**53,960,951 synthetic study artifacts (40.27 B tokens)** generated by applying the
**Active Reading** recipe ([Meta, arXiv:2508.09494](https://arxiv.org/abs/2508.09494)) to
**8,993,564 Wikipedia articles across 29 languages** — none of them English.

Active Reading was published as an English, single-language recipe. This is, as far as we know,
the first multilingual application of it at scale: every strategy and every artifact is written
**in the article's own language**, by a multilingual generator, with no translation step.

Built as the `Active-Reading-multilingual-Wikipedia` slice of the multilingual bucket of a
2 T-token annealing mix, and released as a companion asset to the accompanying tech report.
See also [ProofWiki-ActiveReading](https://huggingface.co/datasets/OVHaiLLM/ProofWiki-ActiveReading),
the mathematical counterpart.

## Contents

| Config | Rows | Size | Description |
|---|---:|---:|---|
| `active_reading` (default) | **53,960,951** | ~35 GB | One row per study artifact: 6 per source article (2 flavours × 3 strategies). |
| `wikipedia` | **10,908,200** | ~14 GB | The source article corpus for the same 29 languages, in the exact preprocessed form the generator saw. |

Every language is also its own config, so you never have to download 50 GB to get one language:

```python
from datasets import load_dataset

# one language
it = load_dataset("OVHaiLLM/Wikipedia-ActiveReading", "active_reading.it", split="train")

# everything (streaming strongly recommended)
ar = load_dataset("OVHaiLLM/Wikipedia-ActiveReading", "active_reading",
                  split="train", streaming=True)

# artifacts join back onto their source article on (language, src_id) -> (language, id)
```

## How it was built

A two-stage vLLM pipeline. Generator: **`RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`** — the MoE
(4 B active) model was chosen over the 31 B dense one after it proved ~2.5× faster at equal
quality on low-resource languages, which is what made 29 languages affordable.

**Stage 1 — strategy generation.** Each article is shown to the model twice, once per *flavour*,
and it proposes 3 article-specific study strategies:

- **`agnostic`** — general "study this article" strategies: internalise the content, recall named
  entities and dates, place facts in context, connect to neighbouring topics.
- **`factual`** — the paper's SimpleWikiQA recipe. The model first *imagines* the downstream
  trivia/QA questions the article could answer, then designs strategies that target exactly that
  material: tail facts, dates, named entities.

The two prompts are byte-identical up to a flavour-specific tail, so vLLM's prefix cache shares
the article's KV blocks across both stage-1 calls.

**Stage 2 — strategy application.** Each (article, strategy) pair is executed independently into
one markdown artifact (≤ 2048 generated tokens). The prompts are in English but carry an explicit
`Write the artifact in {language}.` instruction, so output stays in the source language.

Article filter: 200 ≤ `token_count` ≤ 6000 — drops stubs and articles too long for one call.
That is why the augmented count is ~70–85 % of each language's article count.

## Per-language breakdown

| Code | Language | Articles in corpus | Articles augmented | Artifacts | Tokens (B) |
|---|---|---:|---:|---:|---:|
| `ar` | Arabic | 618,326 | 471,528 | 2,829,143 | 1.99 |
| `bg` | Bulgarian | 217,342 | 179,184 | 1,075,099 | 1.05 |
| `bs` | Bosnian | 80,550 | 73,146 | 438,878 | 0.34 |
| `ca` | Catalan | 494,112 | 431,495 | 2,588,967 | 1.89 |
| `cs` | Czech | 451,039 | 418,569 | 2,511,410 | 2.15 |
| `da` | Danish | 199,299 | 154,419 | 926,512 | 0.62 |
| `et` | Estonian | 115,747 | 97,864 | 587,180 | 0.46 |
| `fi` | Finnish | 317,224 | 289,769 | 1,738,607 | 1.32 |
| `ga` | Irish | 19,984 | 14,354 | 86,124 | 0.06 |
| `he` | Hebrew | 310,756 | 296,361 | 1,778,137 | 1.27 |
| `hr` | Croatian | 132,250 | 115,658 | 693,948 | 0.57 |
| `hu` | Hungarian | 481,730 | 411,322 | 2,467,931 | 2.14 |
| `is` | Icelandic | 27,001 | 23,238 | 139,427 | 0.11 |
| `it` | Italian | 1,590,635 | 1,207,473 | 7,244,827 | 4.00 |
| `ka` | Georgian | 89,644 | 79,956 | 479,723 | 0.72 |
| `ko` | Korean | 481,466 | 398,411 | 2,390,450 | 1.90 |
| `lb` | Luxembourgish | 34,817 | 25,840 | 155,039 | 0.10 |
| `lt` | Lithuanian | 153,978 | 150,164 | 900,983 | 0.73 |
| `lv` | Latvian | 69,990 | 64,592 | 387,550 | 0.34 |
| `no` | Norwegian | 341,039 | 291,578 | 1,749,469 | 1.12 |
| `ro` | Romanian | 298,351 | 281,520 | 1,689,110 | 1.22 |
| `sk` | Slovak | 149,443 | 93,396 | 560,384 | 0.48 |
| `sl` | Slovenian | 126,415 | 106,293 | 637,759 | 0.49 |
| `sq` | Albanian | 72,907 | 50,603 | 303,618 | 0.22 |
| `sr` | Serbian | 612,348 | 501,581 | 3,009,448 | 2.26 |
| `sv` | Swedish | 740,431 | 547,156 | 3,282,935 | 2.03 |
| `tr` | Turkish | 304,861 | 255,745 | 1,534,468 | 1.11 |
| `uk` | Ukrainian | 1,087,947 | 978,480 | 5,870,811 | 5.64 |
| `zh` | Chinese | 1,288,568 | 983,869 | 5,903,014 | 3.94 |
| | **Total** | **10,908,200** | **8,993,564** | **53,960,951** | **40.27** |

## Schemas

### `active_reading`

| Field | Type | Notes |
|---|---|---|
| `id` | string | `ar-{lang}-{src_id}-{flavour}-{strategy-slug}` |
| `language` | string | ISO-639-1 |
| `src_id` | string | Wikipedia page id — joins onto `wikipedia.id` within the same language |
| `title` | string | source article title |
| `flavour` | string | `agnostic` or `factual` (an exact 50/50 split) |
| `strategy_name` | string | model-invented, article-specific |
| `text` | string | **the study artifact** — the training text, in the source language |
| `n_chars` | int32 | |
| `token_count_est` | int32 | see the caveat below |

### `wikipedia`

| Field | Type | Notes |
|---|---|---|
| `id` | string | Wikipedia page id |
| `language` | string | ISO-639-1 |
| `title`, `url` | string | |
| `text` | string | article body rendered to markdown |
| `infobox` | string | markdown table, `""` when the article has none |
| `categories` | list&lt;string&gt; | |
| `entity` | string | Wikidata QID |
| `revdate` | string | last revision timestamp; the snapshot runs to 2025-02-20 |
| `token_count` | int32 | Qwen2.5 tokens, from the dump |
| `used_for_active_reading` | bool | true for the 8,993,564 articles that passed the filter and were augmented |

## Known limitations

- **`token_count_est` is an estimate, not a tokenization.** It is `len(text) × r_lang`, where
  `r_lang` is a per-language Qwen2.5 tokens-per-character ratio measured on a 30-artifact sample
  per language. This was done deliberately: the HF `tokenizers` Rust extension panics and kills
  worker subprocesses under both `fork` and `spawn` at this scale. Budget accuracy is within
  ~5 %; do not use it for anything requiring exact counts. `wikipedia.token_count` **is** a true
  count, taken from the dump.
- **Artifacts are model-generated and not fact-checked.** They can restate errors from the source
  article, hallucinate details, or garble entity names — most likely in the low-resource
  languages (`ga`, `lb`, `is`, `sq`), where the generator is weakest and where, awkwardly, this
  kind of data is most needed.
- **Quality is not uniform across languages.** The same recipe and the same generator were used
  for Italian and for Irish; no per-language prompt tuning or quality gate was applied.
- **No English, and no high-resource languages** (`de`, `es`, `fr`, `ja`, `ru`, `pl`, `pt`, `nl`,
  `el`). Those had enough natural pretraining data and were deliberately left out of the AR run.
- **The `factual` flavour optimises for recall of tail facts**, per the SimpleWikiQA recipe. It
  produces a lot of dense date/entity tables. That is the intent, not a defect, but it does make
  the flavour stylistically narrow — filter on `flavour` if you want the more discursive half.
- Articles are deduplicated by page id within a language, but no cross-language entity dedup was
  applied: the same subject appears independently in many languages.

## Licensing

- All source text is from Wikipedia, **[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)**.
  The whole dataset is released under the same licence.
- The `active_reading` artifacts are outputs of a Gemma model and are additionally subject to the
  [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and Prohibited Use Policy.

## Citation

```bibtex
@article{activereading2025,
  title   = {Learning Facts at Scale with Active Reading},
  journal = {arXiv preprint arXiv:2508.09494},
  year    = {2025},
  url     = {https://arxiv.org/abs/2508.09494}
}
```

Please also credit Wikipedia as the source corpus.
