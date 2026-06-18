# deer-med

A faithful reproduction of **DEER** (Bai et al., 2025, *LLMs are Better Than You Think:
Label-Guided In-Context Learning for Named Entity Recognition*,
[arXiv:2505.23722](https://arxiv.org/abs/2505.23722), EMNLP 2025;
original code: <https://github.com/bflashcp3f/deer>) applied to medical NER on
**NCBI-Disease**. DEER = **D**ata-statistics-grounded nam**E**d **E**ntity **R**ecognition.

This code was extracted from a research project as a standalone baseline. It reproduces the
DEER method (the flat, token-level pipeline), not the original authors' code.

## Method

Three config-driven stages (`orchestrator.py`):

1. **Label-guided retrieval** (`retrieval/label_guided.py`): pick few-shot demonstrations by a
   token-statistics-weighted similarity (DEER eq. 1–5), using per-token entity/context/other
   counts from the training set (`stats/deer_token_stats.py`, the "Step-0" statistics).
2. **Generation** (`generate/generator.py`): few-shot ICL extraction, aligned back to spans
   (`align/aligner.py`).
3. **Error Reflection** (`reflect/flat/`): token-statistics-triggered self-correction in three
   sub-steps — **Unseen** (high-entity tokens left unpredicted), **FN** (false-negative recall
   under θ_FN), and **Boundary** (left/right edge fixes). Triggers live in
   `reflect/flat/triggers.py`.

Method tiers: `m1` = retrieval + generation (no reflection); `m2` = full flat DEER.

DEER is a **flat** method (it predicts a non-overlapping span set). On nested datasets
(CMeEE / GENIA) it therefore cannot recover overlapping gold entities — that shows up as the
`nested-gold recall` gap in the result block and is the expected behaviour of a flat baseline,
not a bug.

## Install

```bash
pip install -e .            # core (numpy) + the package
pip install -e ".[full]"    # + openai, requests, sentence-transformers, torch
```

Python ≥ 3.9. Generation needs an LLM endpoint via `DEEPSEEK_API_KEY` (DeepSeek, OpenAI-compatible);
a `urllib` fallback is built in but the `openai` SDK is the tested path. Retrieval uses local
sentence embeddings (e.g. Qwen3-Embedding-0.6B); a `MockEmbedder` is provided for offline sanity.

## Data

All three datasets are committed and ready to run:

- NCBI-Disease: `data/NCBI-disease/{train,dev,test}.json`
- CMeEE (Chinese clinical, nested): `data/CMeEE-V2/raw/{train,dev,test}.json`
- GENIA (English biomedical, nested): `data/genia_term_corpus/raw/{train,dev,test}.json`

CMeEE originates from the [CBLUE](https://github.com/CBLUEbenchmark/CBLUE) benchmark; GENIA from
the [GENIA term corpus](http://www.geniaproject.org/genia-corpus/term-corpus). (The original DEER
paper evaluates on flat datasets and does not run GENIA; CMeEE/GENIA here are flat-baseline runs.)

## Usage

Offline self-check (no network, no model — Step-0 stats, triggers, full mock pipeline):

```bash
python scripts/selfcheck_deer.py
```

Validate Step-0 statistics against the paper (Figure 2; no API needed):

```bash
python scripts/run_deer.py --validate-stats
```

Full DEER on NCBI test (frozen NCBI runner):

```bash
DEEPSEEK_API_KEY=sk-... python scripts/run_deer.py \
    --method m2 --split test --n-test 1000 --seed 13 \
    --embed-model /path/to/Qwen3-Embedding-0.6B --workers 8
```

Flat DEER on **any** dataset (ncbi / cmeee / genia) via the general runner:

```bash
# offline wiring check first (no API/embedder)
python scripts/run_pipeline.py --dataset cmeee --selfcheck

DEEPSEEK_API_KEY=sk-... python scripts/run_pipeline.py \
    --dataset cmeee --method m2 --split dev --n-test 300 --seed 42 \
    --embed-model /path/to/Qwen3-Embedding-0.6B --workers 8
#   ... --dataset genia ...
```

NCBI hyper-parameters follow the DEER paper (Table 10): λ1=λ2=1, C=2, w_e=w_c=1, w_o=0.01,
θ_FN=0.95, M=1, K=2, N=8 shots. CMeEE/GENIA use the same DEER default grid (`data/datasets.py`).

## Layout

```
src/deer/          # orchestrator + retrieval / generate / reflect(flat) / align / eval / llm / data / stats
scripts/           # run_deer.py (NCBI), run_pipeline.py (ncbi/cmeee/genia), selfcheck_deer.py
data/              # NCBI-disease + CMeEE-V2 + genia_term_corpus (committed)
```

## License

MIT (this reproduction). Please cite the original DEER paper for the method.
