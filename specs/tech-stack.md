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

## Known Gaps to Address

1. **GPU not verified** — Ollama defaults to CPU if CUDA is not configured correctly; needs explicit validation
2. **Chunking is domain-specific** — regex on legal article numbers won't work on AI books
3. **Single-document hardcoded** — collection name, file paths, and prompt are all specific to RDC 166
4. **No evaluation** — no way to know if retrieval quality is good or bad
5. **No metadata in retrieved chunks** — source book and page number are not surfaced in answers

## Dependencies (target `requirements.txt`)

```
pymupdf
chromadb
ollama
langchain-text-splitters   # for RecursiveCharacterTextSplitter
pyyaml                     # for config file
ragas                      # optional, for eval
```
