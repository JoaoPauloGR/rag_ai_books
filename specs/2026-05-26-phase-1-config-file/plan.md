# Phase 1 — Implementation Plan

## Group 1 — `config.yaml`

1. Create `config.yaml` at repo root with all five required fields and their defaults:
   - `embedding_model: nomic-embed-text`
   - `generation_model: llama3.2:3b`
   - `chroma_path: data/chroma_db`
   - `chunk_size: 512`
   - `chunk_overlap: 64`

## Group 2 — `src/config.py`

2. Create `src/config.py` with a `load_config(path: str = "config.yaml") -> dict` function
3. Loader opens and parses the YAML file with `pyyaml`
4. Validates all five required keys are present; raises `ValueError` listing any missing ones
5. Returns the config dict on success

## Group 3 — Update stubs

6. Update `src/ingest.py`:
   - Import `load_config` from `src.config`
   - Call `load_config(args.config)` after parsing args
   - Print `embedding_model`, `chroma_path`, `chunk_size`, `chunk_overlap` from the loaded config
   - Keep existing `"Ingestion not yet implemented"` message and `sys.exit(0)`

7. Update `src/query.py`:
   - Import `load_config` from `src.config`
   - Call `load_config(args.config)` after parsing args
   - Print `generation_model` and `embedding_model` from the loaded config
   - Keep existing `"Query not yet implemented"` message and `sys.exit(0)`
