# Phase 0 — Cleanup / Bootstrap

## Context

The project starts from a clean slate — no prior PoC code exists. The RDC 166 artifact deletion tasks in the roadmap are N/A. The goal is to establish the canonical directory layout and the two entry-point scripts that all future phases will build on.

## Scope

### What this phase produces
- `src/ingest.py` — argparse CLI stub; accepts `--dir` and `--config`; exits 0 with help text
- `src/query.py` — argparse CLI stub; accepts `--query` and `--config`; exits 0 with help text
- `data/books/` — directory for PDF inputs (books remain here going forward)
- `data/chroma_db/` — directory for ChromaDB persistent store
- `src/__init__.py` — empty, makes `src` a package

### What this phase does NOT produce
- Any actual ingestion or query logic (comes in Phase 3 and Phase 4)
- `config.yaml` (comes in Phase 1)
- A config loader (comes in Phase 1)

## Decisions

| Decision | Choice | Reason |
|---|---|---|
| Config flag | Both scripts accept `--config <path>` | Locks in the flag signature early so Phase 1 only adds the YAML file, not a new CLI surface |
| Data layout | `data/books/` + `data/chroma_db/` | All ingestion inputs and outputs under one root; cleaner than mixing books at repo root |
| Stub vs functional | Stubs only | Phase 0 is structural; logic belongs in the phases that introduce the relevant RAG concepts |
| Python package | `src/__init__.py` present | Allows `from src.config import ...` imports in later phases without path hacks |

## Hardcoded defaults (stub placeholders)

The stubs should reference these constants as comments or `NotImplementedError` bodies — they get extracted to `config.yaml` in Phase 1:

- Embedding model: `nomic-embed-text`
- Generation model: `llama3.2:3b`
- ChromaDB path: `data/chroma_db`
- Chunk size: `512`
- Chunk overlap: `64`

## Out of scope

- GPU verification (Phase 2)
- Real chunking or embedding (Phase 3)
- Source attribution in answers (Phase 4)
- Evaluation harness (Phase 5)
