# olala-synthetic-data-pipelines

This repository contains the data generation pipeline for the synthetic datasets used in the pretrained annealing mix of Olala 7A1B.

| Dataset (built) | Description | Generator model | Scale | Pipeline |
|---|---|---|---|---|
| [`OVHaiLLM/Math-Enrich-Multilingual`](https://huggingface.co/datasets/OVHaiLLM/Math-Enrich-Multilingual) | Programmatic math problems with exact answers, rewritten by an LLM into 30 languages and 7 styles, then graded by an LLM judge | [gemma-4-31B-it](https://huggingface.co/RedHatAI/gemma-4-31B-it-FP8-dynamic) (FP8), rewriter and judge | 21.6 M rewrites, 7.37 B tok (17.7 M accepted) | [Math-Enrich](#1-math-enrich-multilingual) |
| [`OVHaiLLM/Wikipedia-ActiveReading`](https://huggingface.co/datasets/OVHaiLLM/Wikipedia-ActiveReading) | [Active Reading](https://arxiv.org/abs/2508.09494) study artifacts for Wikipedia in 29 non-English languages | [gemma-4-26B-A4B-it](https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic) (FP8) | 54.0 M artifacts, 40.3 B tok | [AR-Wikipedia](#2-active-reading--wikipedia) |
| [`OVHaiLLM/ProofWiki-ActiveReading`](https://huggingface.co/datasets/OVHaiLLM/ProofWiki-ActiveReading) | Active Reading study artifacts for ProofWiki theorems (general and Lean-4-oriented) | [gemma-4-31B-it](https://huggingface.co/RedHatAI/gemma-4-31B-it-FP8-dynamic) (FP8) | 185 k artifacts, 0.18 B tok | [AR-ProofWiki](#3-active-reading--proofwiki) |

The dataset cards as published are in [`docs/dataset_cards/`](docs/dataset_cards/). They hold
the full schemas, statistics and known limitations.

## Layout

```
src/
  math_generator/          seed generators (122 used, Z3/SymPy-backed) + LLM rewrite/judge (enrich*.py)
  math_lean/
    proofwiki/             ProofWiki XML dump → JSONL
    active_reading/        two-stage Active Reading pipeline (ProofWiki + Wikipedia), Lean check, Goedel retry
    lean_dojo/             LeanDojo Benchmark 4 converter (not used by the three datasets)
scripts/
  math_enrich/             SLURM: seed generation, per-language rewrite+judge
  active_reading_wiki/     SLURM: 215-shard array job over 29 languages
  active_reading_proofwiki/  step scripts 01–05 + SLURM Lean-check / Goedel-retry jobs
  postprocess/             raw generations → annealing jsonl {text, id, metadata.token_count}
configs/site.env.example   site paths (copy to configs/site.env, which is gitignored)
docs/                      dataset cards, HPC notes
```

## Setup

```bash
pip install -e '.[proofwiki,postprocess,dev]'
pytest                       # runs all generators at every difficulty level
```

Generation needs an OpenAI-compatible vLLM server. The SLURM jobs activate a venv (`$VLLM_ENV`)
and start the servers themselves (see [`docs/hpc_notes.md`](docs/hpc_notes.md)).

**Site config.** Each shell script locates the checkout from `git` and then sources
`configs/site.env`, or `configs/site.env.example` when that file is missing. The example points
everything under `data/` in the repo. For a real cluster:

```bash
cp configs/site.env.example configs/site.env     # then edit the paths
```

**SLURM.** The `.sbatch` files set no account or partition. Pass them on the command line
(`sbatch -A <account> -p <partition> ...`) or export `SBATCH_ACCOUNT` / `SBATCH_PARTITION` env variables.
The `#SBATCH --output` lines write to `logs/`, relative to the directory you submit from, so
**run `sbatch` from the repo root**. The `module load` lines are for the cluster the data was
built on. Adapt them to yours.

---

## 1. Math-Enrich Multilingual

```
math_generator (122 generators) ──► seeds_<lang>.jsonl ──► vLLM rewrite (7 styles, target lang)
      2 M seeds × 30 langs              60 M seeds            └─► vLLM judge (same model, T=0.1)
                                                                    └─► enrich_<lang>.shard{0..3}.jsonl
```

1. **Seeds.** `python -m math_generator --n N --difficulty mix --seed S --out seeds.jsonl`
   produces `{problem, answer, text, meta{category, generator, difficulty, step_text, hardcoded}}`.
   `registry.CapacityAwareSampler` down-weights generators whose distinct-output space is nearly
   used up. The logic and puzzle generators are solved with Z3. On SLURM:
   `sbatch scripts/math_enrich/generate_30_seeds.sbatch` (30 × 2 M, `BASE_SEED=20260506`, + language index).
2. **Rewrite + judge.** `math_generator.enrich_async` uses the same prompts as `enrich.py`, run as
   async clients. For each seed it picks a style (`word_problem`, `narrative`, `formal`,
   `engineering_physics`, `mcq`, `red_herring`, `hidden_operation`) and asks for a rewrite in the
   target language that keeps every number and the final answer. The same model then grades the
   rewrite as strict JSON: correctness yes/no plus naturalness, clarity and difficulty-preservation
   on 1–5. Model: `RedHatAI/gemma-4-31B-it-FP8-dynamic`, 4 vLLM servers per node (DP4).
   ```bash
   bash scripts/math_enrich/submit_per_language_enrich.sh                 # 30 jobs, 1 node × 12 h each
   ENRICH_N=10000 ENRICH_LANGUAGES_OVERRIDE="fr de" bash scripts/math_enrich/submit_per_language_enrich.sh
   ```
   `enrich_dp4.sbatch` and `submit_4parallel_enrich.sh` are the earlier mixed-language variant.
3. **To annealing format.** `scripts/postprocess/preprocess_math_enrich.py` keeps
   `judge.overall_keep`, formats `Problem: …\n\nSolution: …` and adds Qwen2.5 `token_count`.
   `convert_math_enrich.py` is the version used for the English pilot.

## 2. Active Reading — Wikipedia

Two-stage [Active Reading](https://arxiv.org/abs/2508.09494) (`math_lean/active_reading/pipeline.py`
with the prompts in `prompts_wiki.py`):

* **Stage 1:** for each article and each *flavour* (`agnostic` = general study, `factual` = the
  paper's SimpleWikiQA recipe), the model proposes 3 article-specific strategies.
* **Stage 2:** each (article, strategy) pair is run separately and produces one markdown artifact,
  ≤ 2048 tokens, *written in the article's language*.

The result is 6 artifacts per article. Articles are kept if 200 ≤ `token_count` ≤ 6000.
Generator: `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic`, an MoE about 2.5× faster than the 31B dense
model at equal quality on low-resource languages.

Throughput tricks that made 29 languages affordable:

- The prompt is laid out article-first, so all 8 calls for one article share a prefix.
- `client.py` takes a list of endpoints and **pins every call for one article to one DP rank**
  (`pin_key=article_id`). vLLM's own DP load balancer is round-robin and would leave the prefix
  cache cold. Launch vLLM with `--data-parallel-size 4 --data-parallel-external-lb`.

```bash
sbatch scripts/active_reading_wiki/wiki_dp4_array.sbatch     # 215 shards × 1 node, ~2 h/shard; resumable
# bg was run separately with NUM_SHARDS=4 (and its dumps are misspelt "wikepedia-*")
python scripts/postprocess/tokenize_ar_wiki.py              # flatten → one doc per artifact, est. token counts
```

`wiki_main.py` is the single-language CLI. `wiki_dp4.sbatch`, `dispatch_chains.sh` and
`run_all_languages.sh` are the earlier per-language chain approach. The array job replaced it.

## 3. Active Reading — ProofWiki

```bash
python -m math_lean proofwiki download --out raw/proofwiki/latest.xml.gz
python -m math_lean proofwiki parse raw/proofwiki/latest.xml.gz --out proofwiki_proofs.jsonl   # 54,318 records

cd scripts/active_reading_proofwiki
./run_all.sh          # 01 generate → 02 concat → 03 lean_check → 04 lean_retry → 05 to_pretraining
```

The pipeline is the same two-stage one as for Wikipedia, using `prompts.py`. Settings: flavours
`agnostic` and `lean`, 5 strategies each, theorems with proof ≥ 200 chars, truncated to 8000
chars, `RedHatAI/gemma-4-31B-it-FP8-dynamic` (`launch_gemma4_v2.sh`), apply ≤ 4096 tokens. That
gives 18,532 theorems × 10 artifacts. `postprocess/preprocess_active_reading.py` flattens them
into the annealing jsonl, and `prepare_proofwiki_proofs.py` does the same for the source corpus.

**Lean validation (negative result, documented in the card):**
- `lean_check.py` compiles every ```` ```lean ```` block against a Mathlib stub project. Only
  8.4 % compile cleanly.
- `goedel_retry*.py` with `goedel_retry_dp4.sbatch` runs a repair pass using Goedel-Prover-V2-32B.
  It recovered **0** snippets, so the shipped data does not filter on Lean verdicts.

## License

The code is released under the [Apache License 2.0](LICENSE).

## Licensing of the data

- Math-Enrich: Apache-2.0.
- Wikipedia and ProofWiki derived sets: CC BY-SA 4.0.
- All model outputs are additionally subject to the Gemma Terms of Use.
