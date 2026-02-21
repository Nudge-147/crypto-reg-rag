# Crypto Regulation RAG Workspace

This repository compares **baseline retrieval** vs **SAC (Summary-Augmented Chunking)** for legal/regulatory RAG, and serves an API for jurisdiction-filtered queries.

## Core Pipeline

1. Ingest/clean PDFs
   - Baseline: `01_baseline_pipeline.py` -> `cleaned/`
   - SAC: `01_sac_pipeline.py` -> `cleaned_sac/`
2. Build embeddings + FAISS
   - Baseline: `02_baseline_embed.py`
   - SAC/main: `02_embed_build_openai.py` -> `indexes/faiss.index`, `indexes/meta.jsonl`
3. Retrieval + QA
   - `03_retrieve_and_qa.py`
4. API
   - `06_api_server.py` (`GET /health`, `GET /jurisdictions`, `POST /query`)
5. Evaluation
   - `04_baseline_benchmark.py`, `04_benchmark.py`

## Directory Guide

- `raw/`
  - Source PDFs by jurisdiction (`ch`, `hk`, `jp`, `sg`, `kr`, `uk`, `uae`, `us`, `eu`)
- `cleaned/`
  - Baseline chunks
- `cleaned_sac/`
  - SAC chunks (main retrieval source)
- `indexes/`
  - Active FAISS index + metadata (generated artifact)
- `manifests/`
  - Curated source manifests (e.g., SG PDF list)
- `tools/`
  - Utility scripts (e.g., SG downloader)
- `docs/`
  - Project organization and operational notes
- `tests/`
  - `tests/manual/`: hand-curated sets
  - `tests/batches/`: batch-generated sets
- `baseline_data/`
  - Legacy baseline experiment artifacts

## Practical Rule

- If you only run the current production path, focus on:
  - `raw/` -> `01_sac_pipeline.py` -> `02_embed_build_openai.py` -> `03_retrieve_and_qa.py` -> `06_api_server.py`

## Benchmark Quick Usage

- SAC benchmark:
  - `python3 04_benchmark.py --test-set tests/manual/test_set_us_manual.json`
- Baseline benchmark:
  - `python3 04_baseline_benchmark.py --test-set tests/manual/test_set_us_manual.json`
- Export regression CSV (per-case + summary):
  - `python3 04_benchmark.py --test-set tests/batches/test_set_B01_fixed.json --out-csv reports/B01_cases.csv --out-summary-csv reports/B01_summary.csv`
- Core in-corpus benchmark (HK/SG/UK/UAE):
  - `python3 04_benchmark.py --test-set tests/manual/test_set_core_hk_sg_uk_uae.json --out-csv reports/core_hk_sg_uk_uae_cases.csv --out-summary-csv reports/core_hk_sg_uk_uae_summary.csv`

## Incremental Refresh

- Incrementally process new PDFs and append only new chunks to FAISS:
  - `python3 tools/refresh_from_manifest.py --jurisdictions sg,uk,uae`
- If JSONL already exists and you only want append-embedding:
  - `python3 tools/refresh_from_manifest.py --jurisdictions sg,uk,uae --skip-clean`

## API Smoke Regression

- Run 4 key API checks (`/jurisdictions`, HK only, HK+SG mixed, invalid XX=400):
  - `bash tools/api_smoke_regression.sh`

## Runtime Environment

- Active Python interpreter (current working env):
  - `/Users/chenzheyang/anaconda3/envs/crypto_reg/bin/python`
- Note:
  - This env has `faiss` and `openai` available.
  - System `python3` may not have `faiss`.

## API Environment Variables

- Required:
  - `GPTSAPI_API_KEY`
- Optional:
  - `GPTS_BASE_URL` (default: `https://api.gptsapi.net/v1`)
  - `PROXY_URL` (for local proxy, if needed)
  - `HOST` (default: `127.0.0.1`)
  - `PORT` (default: `8000`)
  - `EMBED_MODEL` (default: `text-embedding-3-large`)
  - `QA_MODEL` (default: `gpt-4o-mini`)

## Handoff Quick Start

1. Copy env template:
   - `cp .env.example .env`
2. Load env:
   - `set -a; source .env; set +a`
3. Start API server:
   - `/Users/chenzheyang/anaconda3/envs/crypto_reg/bin/python 06_api_server.py`
4. Run SAC benchmark:
   - `/Users/chenzheyang/anaconda3/envs/crypto_reg/bin/python 04_benchmark.py --test-set tests/manual/test_set_us_manual.json`
   - `/Users/chenzheyang/anaconda3/envs/crypto_reg/bin/python 04_benchmark.py --test-set tests/batches/test_set_B01_fixed.json`
5. Run baseline benchmark:
   - `/Users/chenzheyang/anaconda3/envs/crypto_reg/bin/python 04_baseline_benchmark.py --test-set tests/manual/test_set_us_manual.json`
