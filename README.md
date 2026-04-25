# ai_engr_sys_design

A research and experimentation repository for AI system design, model benchmarking, and foundational transformer learning.

## Repository Overview

This repository contains three primary experimental projects:

- `dist_vector_db`
  - Lightweight vector store with a mock embedding pipeline
  - HTTP API for vector/document upsert, retrieval, search, snapshotting, and RAG context construction
  - Write-ahead log (WAL) and snapshot persistence for recoverable state

- `model_benchmarking_platform`
  - Config-driven synthetic classification benchmark harness
  - Uses `scikit-learn` for dataset generation, training, evaluation, and logistic regression modeling
  - Tracks run artifacts, metrics, metadata, and model checkpoints

- `transformer_from_scratch`
  - Educational GPT-style transformer implementation in PyTorch
  - Character-level tokenizer and next-token dataset
  - Training loop, checkpoint save/load, generation, and attention inspection utilities

## Getting Started

### Recommended Python Version

- Python 3.10 or newer

### Install dependencies

For `model_benchmarking_platform`:

```bash
pip install -r model_benchmarking_platform/requirements.txt
```

For `transformer_from_scratch` and `dist_vector_db`:

```bash
pip install torch
```

If you plan to run the transformer training on GPU, ensure your PyTorch install matches your CUDA version.

## Subproject Usage

### model_benchmarking_platform

Run an experiment from a JSON config file:

```bash
python model_benchmarking_platform/app.py --config model_benchmarking_platform/configs/baseline.json
```

This will:
- load and validate the experiment config
- generate a synthetic classification dataset
- train a logistic regression model
- compute evaluation metrics
- save `config.json`, `metrics.json`, `metadata.json`, and model artifacts under `runs/` and `artifacts/`

### transformer_from_scratch

Train the mini GPT-style model on the bucketed text corpus:

```bash
python transformer_from_scratch/train.py
```

The script will:
- load character-level training data from `transformer_from_scratch/data/input.txt`
- build a `MiniGPT` model with configurable hyperparameters
- train the model with periodic evaluation
- save the best checkpoint to `checkpoints/mini_gpt.pt`
- print sample generations for fixed prompts

### dist_vector_db

Run the HTTP vector store server:

```bash
python dist_vector_db/vector_store.py
```

Available routes include:
- `GET /vectors`
- `GET /vectors/<id>`
- `POST /documents/upsert`
- `POST /documents/upsert_chunked`
- `POST /documents/search`
- `POST /rag/retrieve`
- `POST /snapshot`

This component is designed as a teaching tool for vector retrieval and RAG-style context building.

## Repository Structure

- `dist_vector_db/`
  - `vector_store.py` — main vector store server and utilities
  - `README.md` — placeholder
  - `vector_store.wal`, `vector_store.snapshot.json` — persisted store state

- `model_benchmarking_platform/`
  - `app.py` — CLI entrypoint
  - `config.py` — config loading and validation
  - `train.py` — training orchestration
  - `data.py`, `models.py`, `metrics.py`, `tracker.py`, `utils.py` — modular benchmarking components
  - `configs/` — experiment definitions
  - `requirements.txt` — package dependencies

- `transformer_from_scratch/`
  - `train.py` — training loop and checkpointing
  - `data.py` — tokenizer and dataset creation
  - `model.py` — transformer architecture implementation
  - `utils.py` — device selection, loss estimation, checkpoint helpers, generation utilities
  - `data/input.txt` — sample corpus for training

## Notes

- The project is primarily educational and experimental.
- `dist_vector_db` uses a deterministic mock embedder, not a production embedding model.
- `model_benchmarking_platform` is focused on synthetic dataset benchmarking rather than real-world datasets.
- `transformer_from_scratch` demonstrates core transformer concepts rather than large-scale production training.
