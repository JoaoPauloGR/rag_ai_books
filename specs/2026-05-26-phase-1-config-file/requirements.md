# Phase 1 — Config File

## Context

Phase 0 established CLI stubs that already accept `--config <path>` but ignore the file. Phase 1 makes the flag meaningful by introducing a YAML config and a shared loader. The learning goal is understanding the distinction between an embedding model and a generation model, and introducing `chunk_size`/`chunk_overlap` as tuneable hyperparameters.

## Scope

### What this phase produces

- `config.yaml` at repo root — single source of truth for model names, paths, and chunk parameters
- `src/config.py` — `load_config(path)` function with required-key validation
- Updated `src/ingest.py` — loads config and prints `embedding_model`, `chroma_path`, `chunk_size`, `chunk_overlap`
- Updated `src/query.py` — loads config and prints `generation_model` and `embedding_model`

### What this phase does NOT produce

- Any actual ingestion or query logic (Phase 3/4)
- GPU verification (Phase 2)
- A CLI flag to override individual config values

## Decisions

| Decision | Choice | Reason |
|---|---|---|
| Config format | YAML | Already in `requirements.txt` (`pyyaml`); human-friendly for tuning hyperparameters |
| Validation | Fail fast with `ValueError` listing missing keys | Better DX than a bare `KeyError` three layers deep |
| Default generation model | `llama3.2:3b` | Matches current stack; lighter model for early development |
| Config location | Repo root (`config.yaml`) | Matches the default in both stubs (`--config config.yaml`) |
| Loader location | `src/config.py` | Keeps loader importable as `from src.config import load_config` |

## Out of scope

- Per-environment config overrides
- CLI flags that override individual keys
- Config schema versioning
