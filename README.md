# Local RAG for AI Books

A fully local RAG pipeline for querying AI-related PDFs using GPU-accelerated inference. No cloud, no API keys, no data leaving the machine.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- [Ollama](https://ollama.com) installed and running
- NVIDIA GPU with CUDA (for GPU-accelerated inference)
- Ollama models pulled:
  ```
  ollama pull nomic-embed-text
  ollama pull llama3.2:3b
  ```

## Setup

```bash
uv sync
```

Place your PDF books in `data/books/`.

## Usage

**Ingest a directory of PDFs:**
```bash
uv run python src/ingest.py --dir data/books/
```

**Ask a question:**
```bash
uv run python src/query.py --query "What is the difference between supervised and unsupervised learning?"
```

The answer is followed by deduplicated source citations, one per unique (file, page) pair:

```
Supervised learning uses labelled data ...
[Source: trainingdataformachinelearning.pdf, p. 42]
[Source: trainingdataformachinelearning.pdf, p. 78]
```

The model is instructed to respond with "I don't have enough information in the retrieved passages to answer that." when the context does not contain a relevant answer.

**Check GPU availability:**
```bash
uv run python src/check_gpu.py
```

Both `ingest.py` and `query.py` accept `--config path/to/config.yaml` (default: `config.yaml`).

## Project Structure

```
src/           Python modules
  ingest.py      CLI — orchestrates extract → chunk → embed → store
  extract.py     PDF text extraction (PyMuPDF)
  chunk.py       Text splitting with source metadata
  store.py       ChromaDB collection build/update
  query.py       CLI — retrieve + generate + print deduplicated sources
  retrieve.py    Embed question, query ChromaDB, return top-k chunks
  generate.py    Build grounded prompt, call Ollama, return answer string
  check_gpu.py   GPU availability diagnostic
  config.py      Config loader
data/books/    PDF source documents (not tracked in git)
data/chroma_db/  Persistent vector store (not tracked in git)
specs/         Project specs and roadmap
eval/          Evaluation set and results (Phase 5+)
config.yaml    Model names, paths, chunk parameters
```

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| 0 | Cleanup — domain-agnostic structure | complete |
| 1 | Config file (`config.yaml`) | complete |
| 2 | GPU verification | complete |
| 3 | Multi-document ingestion | complete |
| 4 | Source attribution in answers | complete |
| 5 | Evaluation harness (hit rate, MRR) | planned |
| 6 | Chunking tuning | planned |
| 7 | Model upgrade (optional) | planned |
