# Stage-A Reproducibility Record

## Frozen input

The Stage-A guard-panel input contains 416 prompts derived from 104 frozen base intents and four formulation conditions.

Input SHA256:

`bb71dc13e3f63f73e8e1d8de8bda51da0431766d593843b5695d7c0eb8a21c11`

Comparison-specific pair eligibility was frozen before guard scoring.

## Evaluation runtime

Guard evaluation used a Kaggle managed notebook with two NVIDIA Tesla T4 GPUs.

The frozen study runtime includes Python 3.12.13, PyTorch 2.10.0+cu128, CUDA runtime 12.8, transformers 5.16.1, accelerate 1.14.0, huggingface-hub 1.29.0, tokenizers 0.23.1, safetensors 0.8.0, sentencepiece 0.2.1, protobuf 5.29.5, numpy 2.0.2, llama-models 0.3.0, fairscale 0.4.13, and tiktoken 0.12.0.

The authoritative runtime specification remains the frozen runtime lock already maintained in the repository.

## Guard identities

### Llama Guard 3-1B

The final Stage-A implementation uses the official Meta raw checkpoint through `llama-models==0.3.0`.

Consolidated checkpoint SHA256:

`10a6c07a5556355be37791a225cdf63352c37de1733910a2ab4b3e70077d8fd3`

The final runtime contract was fixed before any Stage-A study example was scored. The frozen input, pair eligibility, and analysis definitions were unchanged.

### ShieldGemma 2B

Model: `google/shieldgemma-2b`

Frozen revision: `d1dffc...` as recorded in the guard contract and score metadata.

### Granite Guardian 3.3 8B

Model: `ibm-granite/granite-guardian-3.3-8b`

Frozen revision: `b3421e...` as recorded in the guard contract and score metadata.

### Qwen3Guard Gen 4B

Model: `Qwen/Qwen3Guard-Gen-4B`

Frozen revision: `6ec428...` as recorded in the guard contract and score metadata.

The exact immutable revisions are retained in the guard contract and per-guard metadata rather than shortened for execution.

## Scored contract and implementation

Stage-A guard contract SHA256:

`98d8a6bf85685a33a88943076e1d2788d36a4a74eda17e79a9b6e70e57b12b4d`

Stage-A scoring runner SHA256:

`0b24ee4ba54f62ea7d55e4f0fa2ab8924e0f2d874e27bc5130b6072f90022ec4`

These identify the exact contract and implementation used for the reported outputs.

## Completed outputs

Each guard produced 416 scored rows with 416 successfully parsed outputs.

| Guard | Score SHA256 |
|---|---|
| Llama Guard 3-1B | `0b799460f2fb448b28b2b94a3923d87a33912430d1b1fdf6c6a63cae5726daa2` |
| ShieldGemma 2B | `343f10db00271086d67f99f304abfe1fd59c7c3c029cb6c97023ce41107a9a56` |
| Granite Guardian 3.3 8B | `c535600de65f194b45060b81ed369d88da9ac150a787d65972bcde445ddcdcc7` |
| Qwen3Guard Gen 4B | `ea0831fb4984de0b0a16970afc123834887eb63363872e47b5e608c840a9ea4d` |

## Analysis

`scripts/analyze_stage_a_paired_transport.py` reproduces:

- `q`, `q10`, and `q01`;
- exact two-sided 95% Clopper-Pearson intervals for discordance;
- paired FNR differences;
- strict and semantic-only source analyses;
- category- and severity-stratified effects;
- common-mode miss counts;
- pairwise co-miss rates;
- three-of-four failure identities.

Pair eligibility is read from the frozen Stage-A input and is never redefined using guard outcomes.

## Analysis boundary

The Stage-A analysis is development-stage evidence. It does not perform threshold refitting, guard tuning, prompt regeneration, eligibility changes, W0 evaluation, or fresh confirmatory evaluation.
