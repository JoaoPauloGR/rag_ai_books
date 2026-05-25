# Phase 0 — Implementation Plan

## Group 1 — Directory structure

1. Create `src/` directory
2. Create `src/__init__.py` (empty)
3. Create `data/books/` directory
4. Create `data/chroma_db/` directory
5. Add `.gitkeep` to `data/books/` and `data/chroma_db/` so directories are tracked by git

## Group 2 — `src/ingest.py` stub

6. Create `src/ingest.py` with `argparse` setup:
   - `--dir` (required): path to directory of PDFs to ingest
   - `--config` (optional, default `config.yaml`): path to config file
7. `main()` function body: print `"Ingestion not yet implemented"` and exit 0
8. Add `if __name__ == "__main__": main()` guard

## Group 3 — `src/query.py` stub

9. Create `src/query.py` with `argparse` setup:
   - `--query` (required): natural-language question to ask
   - `--config` (optional, default `config.yaml`): path to config file
10. `main()` function body: print `"Query not yet implemented"` and exit 0
11. Add `if __name__ == "__main__": main()` guard

## Group 4 — Project hygiene

12. Create `requirements.txt` with initial dependencies (install-ready for Phase 1+):
    ```
    pymupdf
    chromadb
    ollama
    langchain-text-splitters
    pyyaml
    ```
13. Create `.gitignore` covering: `venv/`, `data/chroma_db/`, `__pycache__/`, `*.pyc`, `.env`
