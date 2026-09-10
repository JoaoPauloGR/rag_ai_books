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

Ingestion is incremental. Chunk IDs are the stable `source_file::chunk_index`, so books
already in the collection are skipped and adding one PDF no longer re-embeds the rest. The
final line reports how many chunks were added, how many were already present, and how many
files were skipped or failed.

**Rebuild from scratch:**
```bash
uv run python src/ingest.py --dir data/books/ --force
```

`--force` wipes the collection and re-embeds every file. Use it after changing
`chunk_size` / `chunk_overlap` (which shift chunk boundaries and invalidate the IDs) or
when a PDF's contents changed — an ID-based skip cannot detect either on its own.

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

All CLIs accept `--config path/to/config.yaml` (default: `config.yaml`).

### Chat UI

A local browser chat interface (Gradio, bound to `127.0.0.1`, `share=False`):

```bash
uv run python src/app.py          # opens http://127.0.0.1:7860
```

`--config` (default `config.yaml`) and `--port` (default `7860`) are accepted.

It is multi-turn: follow-up questions are rewritten to standalone queries before
retrieval (toggle with `rewrite_followups` in `config.yaml`). Under each answer is a
feedback panel — 👍 / 👎, a comment box, and a checkbox per cited source for "which of
these actually helped". Every answered turn and every feedback submission is written to a
local SQLite database (`data/feedback.db`, git-ignored), which a later phase mines for an
evaluation set and preference pairs. No feedback is used for model training in this
project.

See `specs/2026-09-10-phase-5-chat-ui-feedback/` for the full spec.

## Evaluate retrieval

`eval/eval_set.json` is a committed set of hand-verified questions, each tagged with the
source file and page(s) that answer it. `src/evaluate.py` retrieves for every question,
records the rank of the first chunk from the expected file on an expected page, and
(unless `--no-answer-check`) generates an answer and checks that every `answer_keywords`
string appears in it.

```bash
uv run python src/evaluate.py                     # score the committed eval set
uv run python src/evaluate.py --k 3 --no-answer-check   # retrieval-only, top-3
```

Each run prints a per-question table plus `hit@k / hit@1 / hit@3 / MRR / ans_kw`, appends
one row to `eval/results.md`, and writes a full `eval/results/<timestamp>.json` dump.
Needs a populated collection (run ingestion first) and Ollama up.

To regenerate draft questions from sampled chunks (input to manual curation, never
committed as-is):

```bash
uv run python src/build_eval_set.py --n 40        # writes eval/eval_set.draft.json
```

## Project Structure

```
src/           Python modules
  ingest.py      CLI — orchestrates extract → chunk → embed → store (incremental; --force to rebuild)
  extract.py     PDF text extraction (PyMuPDF)
  chunk.py       Text splitting with source metadata
  store.py       ChromaDB collection build/update (stable source_file::chunk_index IDs)
  query.py       CLI — retrieve + generate + print deduplicated sources
  retrieve.py    Embed question, query ChromaDB, return top-k chunks
  generate.py    Build grounded prompt, call Ollama, return answer string
  rewrite.py     Rewrite a follow-up into a standalone query
  app.py         Gradio chat UI — localhost only, with feedback panel
  feedback.py    SQLite persistence for turns + feedback
  build_eval_set.py  CLI — draft candidate eval questions from sampled chunks (Ollama)
  evaluate.py    CLI — score retrieval + answer keywords against eval/eval_set.json
  check_gpu.py   GPU availability diagnostic
  config.py      Config loader
data/books/      PDF source documents (not tracked in git)
data/chroma_db/  Persistent vector store (not tracked in git)
data/feedback.db SQLite feedback + turn log (not tracked in git)
specs/           Project specs and roadmap
eval/
  eval_set.json    Curated, hand-verified question set (tracked)
  results.md       One row per evaluation run (tracked)
  eval_set.draft.json  build_eval_set.py output (git-ignored)
  results/         Per-run JSON dumps (git-ignored)
config.yaml      Model names, paths, chunk parameters, top_k, feedback_db_path, rewrite_followups
```

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| 0 | Cleanup — domain-agnostic structure | complete |
| 1 | Config file (`config.yaml`) | complete |
| 2 | GPU verification | complete |
| 3 | Multi-document ingestion | complete |
| 4 | Source attribution in answers | complete |
| 5 | Chat UI + feedback capture (Gradio) + incremental ingestion | in progress |
| 6 | Evaluation harness (hit rate, MRR) | complete |
| 7 | Chunking tuning | planned |
| 8 | Model upgrade (optional) | planned |
