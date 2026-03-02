# Structure Notes

## What Was Cleaned

- Removed local-only artifacts:
  - `.DS_Store`
  - `__pycache__/`
  - `.venv/`
  - `Miniforge3-MacOSX-arm64.sh`
- Removed duplicate side workspace:
  - `kr_uk_uae/` (already merged into main `raw/` and main index)

## Current Single Source of Truth

- Source docs: `raw/`
- SAC chunks: `cleaned_sac/`
- Retrieval index: `indexes/faiss.index` + `indexes/meta.jsonl`

## Conventions

- Keep runtime artifacts out of git (`.gitignore` covers `.DS_Store`, `__pycache__`, `.venv`, `indexes/`).
- Keep manifests in `manifests/`.
- Keep utility scripts in `tools/`.
- Keep benchmark datasets in:
  - `tests/manual/` for hand-curated sets
  - `tests/batches/` for generated batches
- Keep pipeline/eval scripts at repo root for discoverability.
