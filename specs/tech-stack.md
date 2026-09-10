# Tech Stack

## Current (Proof of Concept)

| Layer | Tool | Notes |
|---|---|---|
| PDF extraction | PyMuPDF (`fitz`) | Works well; keep |
| Text cleaning | Custom regex | Ad-hoc, RDC 166-specific; replace |
| Chunking | Regex on `Art. X` pattern | Legal-doc specific; replace |
| Embeddings | `nomic-embed-text` via Ollama | Good quality; keep |
| Vector store | ChromaDB (persistent) | Sufficient for local single-user; keep |
| Generation | `llama3.2` (3B) via Ollama | Upgrade model; enable GPU layers |
| Interface | Python CLI (`input()` loop) | Keep; add argparse flags |
| Language | Python 3.x | Keep |

## Target Stack

### Inference Runtime
- **Ollama** — serves both embedding and generation models locally
- **GPU backend**: CUDA via llama.cpp (`OLLAMA_NUM_GPU_LAYERS` or `--num-gpu-layers`)
- **Generation model**: upgrade from `llama3.2:3b` to `llama3.1:8b` or `mistral:7b` depending on VRAM
- **Embedding model**: `nomic-embed-text` (keep) or `mxbai-embed-large` for higher-dim embeddings

### Ingestion Pipeline
- **PyMuPDF** — PDF text extraction (keep)
- **LangChain `RecursiveCharacterTextSplitter`** or custom sliding-window chunker
  - Configurable `chunk_size` (target: 512–1024 tokens) and `overlap` (target: 10–20%)
  - Metadata: source file, page number, chunk index
- **Incremental ingestion** (Phase 5) — chunk IDs are the deterministic
  `source_file::chunk_index`; `ingest` uses `get_or_create_collection` and skips books
  already present. `--force` restores full wipe-and-rebuild. Stable IDs keep the feedback
  log and the evaluation set pointing at the same chunks across re-ingests.
  - Deferred: content-hash detection of a changed PDF, orphan-chunk cleanup

### Interface
- **CLI** — `src/ingest.py`, `src/query.py`, `src/check_gpu.py` (argparse, `--config`); kept
- **Gradio chat UI** (`src/app.py`, Phase 5) — `gr.Blocks`, bound to `127.0.0.1`,
  `share=False`; multi-turn with a follow-up query-rewrite step (`src/rewrite.py`,
  toggled by `rewrite_followups`); per-answer feedback panel (👍/👎 + comment +
  per-cited-source "helpful" checkbox)
- No REST API, no LAN/remote hosting, no auth — single user, single workstation

### Feedback Store
- **SQLite** (`sqlite3`, stdlib) at `feedback_db_path` (`data/feedback.db`, git-ignored),
  written by `src/feedback.py` (Phase 5)
- Tables: `turns` (one per answered question, retrieval params + both model names
  snapshotted), `turn_sources` (one per retrieved chunk, with `chunk_text` snapshot,
  `cited` and `helpful` flags), `feedback` (one per turn, `rating` + `comment`)
- Purpose: seed the Phase 6 evaluation set and export DPO preference pairs later.
  **No model training** (RLHF / DPO / LoRA) is in scope — the store is data collection only

### Vector Store
- **ChromaDB** (persistent) — keep; move to a dedicated `data/` directory
- One collection per book or one shared collection with source metadata for filtering

### Evaluation
- **Custom eval harness** — a JSON file of `(question, expected_chunk_ids)` pairs
- Metrics: hit rate (chunk retrieved in top-k), MRR
- Optional: **RAGAS** library for answer faithfulness and context relevance scoring

### Configuration
- **YAML config file** (`config.yaml`) for model names, paths, chunk parameters
- No environment variables required for normal operation
- Keys: `embedding_model`, `generation_model`, `chroma_path`, `collection_name`,
  `chunk_size`, `chunk_overlap`, `top_k`, and (Phase 5) `feedback_db_path`,
  `rewrite_followups`. `src/config.py` fails loudly on any missing key.

## Known Gaps

Gaps 1–3 and 5 were closed in Phases 2–4; gap 4 is the target of Phase 6.

1. ~~**GPU not verified**~~ — resolved in Phase 2 (`src/check_gpu.py`, `docs/gpu-setup.md`)
2. ~~**Chunking is domain-specific**~~ — resolved in Phase 3 (`RecursiveCharacterTextSplitter`)
3. ~~**Single-document hardcoded**~~ — resolved in Phases 1 & 3 (`config.yaml`, `--dir` ingestion)
4. **No evaluation** — still open; Phase 6 builds the hit-rate / MRR harness, seeded from
   the Phase 5 feedback store
5. ~~**No metadata in retrieved chunks**~~ — resolved in Phase 4 (`[Source: file, p. N]` citations)

### Open considerations

- **Changed / removed PDFs** — incremental ingestion (Phase 5) skips by chunk ID; it does
  not detect that a file's contents changed or that a book was deleted. `--force` is the
  only remedy until content hashing lands.
- **Chunk-param changes invalidate IDs** — changing `chunk_size` / `chunk_overlap` shifts
  chunk boundaries, so `source_file::chunk_index` IDs (and any stored references) only
  stay valid while those params are fixed. Re-tuning in Phase 7 implies a `--force` rebuild.

## Dependencies

Managed with `uv` (`pyproject.toml`), not a `requirements.txt`.

```
pymupdf                    # PDF extraction
chromadb                   # vector store
ollama                     # embedding + generation client
langchain-text-splitters   # RecursiveCharacterTextSplitter
pyyaml                     # config file
gradio                     # chat UI (Phase 5)
# sqlite3 — feedback store (Python standard library, no dependency)
# ragas — optional, considered for answer-faithfulness scoring (not adopted)
```

Dev: `pytest`.
