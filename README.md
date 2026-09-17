# Encoder-Based versus LLM-Based Reranking for TFCSR

Supplementary material for **"Encoder-Based versus LLM-Based Reranking for Training-Free Cold-Start Recommendation."**

## Overview

This repository provides a unified and reproducible evaluation of training-free cold-start recommendation (TFCSR) methods, where no task-specific training data are available and the target user has few or no past interactions. We compare BM25 as a lexical baseline, thirteen encoder-based methods (nine embedding models and four dense rerankers), and six LLM-based generative rerankers under a common TFCSR setting across six public datasets.

**Key finding:** encoder-based methods achieve accuracy comparable to, and often exceeding, that of LLM-based generative reranking, at 10–1000× lower estimated inference cost. Concretely, the Qwen3 (8B) embedding is 30–45× cheaper than GPT-5.4 / Sonnet 4.5 at `L=50`, and about 1,000× cheaper when candidate item embeddings are precomputed and reused.

## Models

All encoders are open-weight and evaluated as released, with no task-specific fine-tuning.

| Type | Model | Model ID / source |
|------|-------|-------------------|
| Embedding | SimCSE\* | `princeton-nlp/sup-simcse-roberta-large` |
| Embedding | ModernBERT | `answerdotai/ModernBERT-large` |
| Embedding | RoBERTa\* | `FacebookAI/roberta-large` |
| Embedding | ME5\* | `intfloat/multilingual-e5-large` |
| Embedding | BGE-M3 | `BAAI/bge-m3` |
| Embedding | GTE | `Alibaba-NLP/gte-modernbert-base` |
| Embedding | Qwen3 (0.6B) | `Qwen/Qwen3-Embedding-0.6B` |
| Embedding | Qwen3 (8B) | `Qwen/Qwen3-Embedding-8B` |
| Embedding | Nemotron (8B)† | `nvidia/llama-embed-nemotron-8b` |
| Reranker | BM25 | `rank_bm25` (Python library, `k1=1.2`, `b=0.75`) |
| Reranker | BGE-M3 | `BAAI/bge-reranker-v2-m3` |
| Reranker | GTE | `Alibaba-NLP/gte-reranker-modernbert-base` |
| Reranker | Qwen3 (0.6B) | `Qwen/Qwen3-Reranker-0.6B` |
| Reranker | Qwen3 (8B) | `Qwen/Qwen3-Reranker-8B` |
| LLM | Llama3.3-70B | `us.meta.llama3-3-70b-instruct-v1:0` |
| LLM | Qwen3-235B-A22B | `qwen.qwen3-235b-a22b-2507-v1:0` |
| LLM | GPT-4.1 mini | `gpt-4.1-mini-2025-04-14` |
| LLM | GPT-5.1 | `gpt-5.1-2025-11-13` |
| LLM | GPT-5.4 | `gpt-5.4-2026-03-05` |
| LLM | Sonnet 4.5 | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` |

\* Limited to 512 input tokens. † Academic use only; not licensed for commercial deployment.

Instruction prefixes follow each model's official guidelines. ME5's `query:` / `passage:` prefixes are applied in `embedding.Embedding._encode()`; Qwen3 and Nemotron receive theirs through the SentenceTransformer `query` prompt. The alternative prefixes used in the prefix ablation are defined in `utils.load_inst()`, which covers Qwen3 (8B) and Nemotron only.

## Datasets

Six public datasets are used:

- **ML-1M** — MovieLens-1M (user profile: gender, age, occupation)
- **Job** (with-exp / no-exp) — split by whether the user profile records prior work experience
- **Music** — Amazon CDs & Vinyl
- **Movie** — Amazon Movies & TV
- **Toys** — Amazon Toys & Games
- **Sports** — Amazon Sports & Outdoors

User profile texts exist only for ML-1M and Job, so the profile-only (`N=0`) setting is evaluated on those datasets.

## Evaluation Protocol

- **Users.** 500 users are sampled per dataset, each with at least `N+2` interactions. The two most recent interactions are held out as positives.
- **Candidate set.** `L` items per user: 2 positives and `L-2` negatives sampled from unseen items in the same domain. This evaluates the reranking stage of a retrieve-then-rerank pipeline, not full-catalog ranking.
- **Observed interactions.** `N ∈ {0, 1, 3, 5}` (`N=2, 4` omitted to limit inference cost). At `N=0` the user text is the profile text; at `N≥1` it is built from the texts of the `N` previously selected items.
- **Candidate set size.** `L=10` for the main experiments, `L=50` for the larger-candidate experiments.
- **Metrics.** nDCG@5 at `L=10` and nDCG@10 at `L=50`; Recall is computed as a secondary metric. A rank-based MRR aggregates each method's rank across datasets. As sanity checks, a uniformly random ranking gives an expected nDCG@5 of 0.362 at `L=10` and an expected nDCG@10 of 0.111 at `L=50`.
- **Multi-interaction aggregation.** For `N>1`, encoders use `Concat` (all past item texts joined into one query, the default) or `Separate` (each past item scored independently and averaged). LLMs list past items directly in the prompt.
- **Significance testing.** One-sided Wilcoxon signed-rank test at `p=10⁻³`.
- **Proper-noun masking.** Named entities in user and item texts are replaced with NER tags (spaCy) to probe reliance on memorized entities. Toggled by `flag_replace_NER` at the top of each pipeline notebook.

## Setup

### Requirements

- Python 3.10+
- CUDA-compatible GPU (for encoder and reranker inference). Reported inference times were measured on a single NVIDIA L40S.

### Installation

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121  # match your CUDA version
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Configuration

Set the following variables in `src/utils.py` before running anything:

| Variable | Meaning |
|----------|---------|
| `dir_parent`  | Root directory holding raw datasets, local model weights, and outputs |
| `version_exp` | Experiment version tag, used in all output paths |
| `device`      | GPU device for encoder and reranker inference |

Notebooks assume the following directory layout under `dir_parent`:

```
dir_parent/
├── received_data/           # raw datasets (ml-1m, job-recommendation, amazon)
├── models/
│   ├── embedding_models/    # local weights, mirroring the HuggingFace IDs above
│   └── reranker_models/
└── research/TFCSR/
    ├── preprocessed_data/   # output of Step 1
    ├── embedding_data/      # output of Step 2
    ├── reranking_data/      # output of Step 3
    ├── LLMreranking_data/   # output of Step 4
    ├── score_data/          # cached scores (Step 5)
    ├── figs/                # output figures
    └── tables/              # output LaTeX tables
```

### API Keys

For LLM reranking, set the following environment variables:

```bash
export OPENAI_API_KEY="..."            # GPT-4.1 mini, GPT-5.1, GPT-5.4
export AWS_ACCESS_KEY="..."            # Sonnet 4.5, Qwen3-235B, Llama3.3 (via Bedrock)
export AWS_SECRET_ACCESS_KEY="..."
```

## Quick Start

Two sample notebooks provide a self-contained entry point:

| Notebook | Description |
|----------|-------------|
| `notebook/sample_with_src.ipynb`    | Full pipeline using the `src` modules with the same settings as the paper. Raw-data download and preprocessing are required. |
| `notebook/sample_without_src.ipynb` | Self-contained demo with inline sample data. No additional setup, raw-data download, or preprocessing required; demonstrates the core embedding + LLM reranking flow. |

## Reproducing the Experiments

Run the notebooks in `notebook/` in order.

| Step | Notebook | Description |
|------|----------|-------------|
| 1. Preprocessing    | `1_movielens.ipynb`                         | Parse ML-1M and generate user–item records |
|                     | `1_job.ipynb`                               | Parse the Job dataset (with-exp / no-exp split) |
|                     | `1_amazon.ipynb`                            | Parse Amazon Review Data (Music, Movie, Toys, Sports) |
| 2. Embedding        | `2_1_embedding.ipynb`                       | Compute text embeddings for the nine embedding models |
| 3. Dense reranking  | `2_2_reranker_dense.ipynb`                  | Run BM25 and the four dense cross-encoders |
| 4. LLM reranking    | `2_3_reranker_llm_small_candidates.ipynb`   | LLM reranking, `L=10` |
|                     | `2_4_reranker_llm_large_candidates.ipynb`   | LLM reranking, `L=50` |
| 5. Evaluation       | `3_score.ipynb`                             | Compute nDCG / Recall / MRR and generate all main tables and figures |

Steps 2–4 each read `flag_replace_NER` from the top cell. Run them once with `False` and once with `True`; `3_score.ipynb` loads both result sets and produces the original-text and NER-masked results together.

### Additional Analysis

| Notebook | Description |
|----------|-------------|
| `additional_similarity.ipynb`          | Prefix ablation: nDCG sensitivity to prefix wording (`prefix_sensitivity.pdf`), output-embedding cosine similarity under prefix vs. content perturbations (`content_emb_similarity.pdf`), and the corresponding appendix table |
| `additional_cost.ipynb`                | Inference time and estimated cost per method. Timings are read from `notebook/cost_*.pickle` when present, and re-measured on a GPU otherwise |
| `additional_dataset_statistics.ipynb`  | Token statistics of user and item texts (the last row of the main results table) |

## Reproducibility

**Random seed.** Fixed to `42` via `utils.set_seed()`, called at the top of every notebook. It seeds Python `random`, NumPy, and PyTorch (CPU and CUDA), sets `PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG`, enables deterministic cuDNN and `torch.use_deterministic_algorithms`, and returns a seeded `numpy.random.Generator`.

**LLM sampling.** `temperature=0` for all API calls, with reasoning disabled on the GPT-5 models. Reasoning effort is selected through a `_reasoning_<effort>` suffix on the model name (e.g. `gpt-5.4-2026-03-05_reasoning_none`), which `llm.py` strips before calling the API.

**Candidate ordering.** Item IDs in LLM prompts are assigned randomly per query and the candidate order is shuffled per user, to mitigate position bias.

> **Note.** Some LLM APIs do not guarantee bit-exact reproducibility across runs even at `temperature=0`. We follow best practices but cannot fully eliminate this source of variance.

**Cost figures.** LLM costs are computed from API pricing as of September 15, 2026. Encoder costs are *estimated* by multiplying measured inference time by the hourly price of an AWS EC2 `g6e.xlarge` instance, which carries the same L40S GPU used for the local runs.

## Project Structure

```
src/
├── utils.py                # global config, seed, prefix definitions, LLM factory
├── data_loader.py          # dataset loading and NER masking (spaCy)
├── embedding.py            # bi-encoder wrapper (SentenceTransformer)
├── reranker.py             # reranker dispatcher (BM25 / dense)
├── reranker_bge.py         # BGE / GTE cross-encoder backend
├── reranker_qwen3.py       # Qwen3-Reranker backend (4-bit quantised)
├── reranker_llm.py         # LLM-based listwise reranker
├── llm.py                  # LLM API client (OpenAI / AWS Bedrock)
├── scorer.py               # nDCG@K and Recall@K computation
├── result_loader.py        # aggregate results, statistical tests, LaTeX tables
├── figure_data.py          # data extraction helpers for figures
├── visualizer.py           # matplotlib figure generation
├── attention_analysis.py   # output-embedding and prefix-sensitivity analysis
└── table_utils.py          # LaTeX table formatting utilities

notebook/
├── 1_*.ipynb               # Step 1: preprocessing
├── 2_*.ipynb               # Steps 2–4: model inference
├── 3_score.ipynb           # Step 5: evaluation and visualisation
├── additional_*.ipynb      # supplementary analyses
└── sample_*.ipynb          # quick-start demos
```
