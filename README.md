# Local RAG for AI Books

A fully local RAG pipeline for querying AI-related PDFs using GPU-accelerated inference. No cloud, no API keys, no data leaving the machine.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- NVIDIA GPU with CUDA (for GPU-accelerated inference)
- Ollama models pulled:
  ```
  ollama pull nomic-embed-text
  ollama pull llama3.1:8b
  ```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Place your PDF books in `data/books/`.

## Usage

**Ingest a directory of PDFs:**
```bash
python src/ingest.py --dir data/books/
```

**Ask a question:**
```bash
python src/query.py --query "What is the difference between supervised and unsupervised learning?"
```

Both commands accept `--config path/to/config.yaml` (default: `config.yaml`).

## Project Structure

```
src/           Python modules (ingest, query, evaluate)
data/books/    PDF source documents (not tracked in git)
data/chroma_db/  Persistent vector store (not tracked in git)
specs/         Project specs and roadmap
eval/          Evaluation set and results (Phase 5+)
config.yaml    Model names, paths, chunk parameters (Phase 1+)
```

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| 0 | Cleanup — domain-agnostic structure | in progress |
| 1 | Config file (`config.yaml`) | planned |
| 2 | GPU verification | planned |
| 3 | Multi-document ingestion | planned |
| 4 | Source attribution in answers | planned |
| 5 | Evaluation harness (hit rate, MRR) | planned |
| 6 | Chunking tuning | planned |
| 7 | Model upgrade (optional) | planned |
