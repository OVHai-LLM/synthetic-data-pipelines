# HPC operational notes

These are the lessons from the 2026 generation runs on a SLURM cluster of GH200 nodes. Most of
them first showed up as silent failures.

## Environment

- vLLM venv: `$VLLM_ENV` (vllm 0.20.2rc1.dev63, torch 2.11.0+cu130, Python 3.13.5), built against
  the cluster's CUDA 13, cuDNN 9.19 and NCCL modules. After activating it, `unset PYTHONPATH`: a
  Python module that ships bundled PyPI packages will otherwise shadow the venv.
- "GH200 480GB" exposes only **96 GB HBM** to CUDA. Size the KV budget against 96 GB.
- **Compute nodes have no internet.** Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1`.
  Without them vLLM hangs for about 10 minutes trying to reach hf.co. `snapshot_download` is
  unreliable even with offline mode and a full cache, so resolve
  `$HF_HOME/hub/.../snapshots/<rev>` yourself and pass the absolute path to vLLM.
- **`~/.cache` quota fills up quickly.** Point `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`,
  `TRITON_CACHE_DIR`, `FLASHINFER_CACHE_DIR` and `XDG_CACHE_HOME` at project space (`$CACHE_ROOT`).
  The sbatch scripts already do this.
- SLURM: one node per job with `--gres=gpu:gh200:4`, **12 h maximum walltime**. Write jobs so they
  can resume, and chain them with `--dependency=afterany`.
- Uploads to Hugging Face have to run from a login node, since compute nodes have no network.

## vLLM flags (AR-style workload: 1–5 k input, about 1 k output)

```
--max-model-len 16384 --gpu-memory-utilization 0.92 --kv-cache-dtype fp8 --enable-prefix-caching
--max-num-seqs 256 --max-num-batched-tokens 16384 --async-scheduling --limit-mm-per-prompt '{"image":0,"audio":0}'
```
`--max-num-partial-prefills` raises `NotImplementedError` in this vLLM build.

Measured output throughput per GPU: Gemma-4-26B-A4B is about 3.6 k tok/s on stage 1 and 2.7 k
tok/s on stage 2. Gemma-4-31B dense is about 1.1 k tok/s on both.

Running 4 vLLM servers per node fails about 0.6 % of the time from a port race on
torch.distributed. Resubmit the failed array tasks.

## Wikipedia dump traps (`$WIKI_DUMP_ROOT/<lang>/`)

1. The `wikipedia-N_shuffled` dirs are **not the raw dump**. They hold the annealing rewrite
   `{text, id, metadata}`, with title, url, infobox and entity stripped and `title`/`token_count`
   moved under `metadata`. For `ko lt lv no ro sk sl sq tr uk` the shuffled dir sorts first. The
   first AR run produced **0 records** for those 10 languages until `wiki_loader.sample_articles`
   was changed to flatten `metadata`. For the HF source config, identify the raw dir by checking
   that the first record has a `title` key.
2. Some dirs contain a full nested duplicate of themselves (`it/wikipedia-1/it/`). Glob
   `<dir>/*.jsonl` without recursion and dedup by article id.
3. Bulgarian dirs are spelt **`wikepedia-N`**. Match on `wik*pedia*`.

## Sizing synthetic runs

The AR-Wikipedia estimate was 84.6 B tokens and the run produced 40.3 B. Stage 2 averaged about
700 output tokens per call, well under the `--max-tokens` cap. Estimate output per call from a
small smoke run, not from the cap.

## Token counts

The HF `tokenizers` Rust extension panicked inside worker subprocesses at AR-Wikipedia scale, under
both `fork` and `spawn`. `tokenize_ar_wiki.py` therefore uses per-language tokens-per-char ratios
(`TOK_PER_CHAR`) measured with `count_ar_wiki_tokens.py`. They are accurate to about ±5 %.
