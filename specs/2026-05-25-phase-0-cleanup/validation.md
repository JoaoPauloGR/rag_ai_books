# Phase 0 — Validation

## Done when

All of the following pass before merging:

### 1. Help commands exit cleanly

```
python src/ingest.py --help
```
Expected: exits 0, prints usage including `--dir` and `--config` arguments.

```
python src/query.py --help
```
Expected: exits 0, prints usage including `--query` and `--config` arguments.

### 2. Missing required args exits with error

```
python src/ingest.py
```
Expected: exits non-zero with argparse error about missing `--dir`.

```
python src/query.py
```
Expected: exits non-zero with argparse error about missing `--query`.

### 3. Directory structure exists

```
data/books/       # present (contains .gitkeep or moved PDFs)
data/chroma_db/   # present (empty, with .gitkeep)
src/__init__.py   # present
src/ingest.py     # present
src/query.py      # present
requirements.txt  # present
.gitignore        # present
```

### 4. Stub runs without import errors

```
python src/ingest.py --dir data/books/
```
Expected: prints `"Ingestion not yet implemented"`, exits 0.

```
python src/query.py --query "What is a transformer?"
```
Expected: prints `"Query not yet implemented"`, exits 0.

## Not required for merge

- Actual PDF ingestion working
- ChromaDB populated
- Ollama connectivity
- `config.yaml` present
