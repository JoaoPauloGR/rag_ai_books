# Roadmap

Phased from current proof-of-concept to a capable personal RAG system. Each phase is a self-contained unit of work with a clear deliverable, and a set of RAG/ML concepts you will understand by the end of it.

---

## Phase 0 — Cleanup (current code) ✦ start here

**Goal**: remove all RDC 166 artifacts; make the codebase domain-agnostic.

- Delete `rdc166.pdf`, `rdc166_content.txt`, `rdc166_clean.txt`, `rdc166_chunks.json`, `chroma_db/`
- Delete `file_exploration.py`, `debug_ollama.py`, `error.log` (scratch files)
- Remove hardcoded collection name `"rdc166"` and domain-specific prompt from `rag_app.py`
- Reorganize into `src/` (Python modules) and `data/` (books, vector store)

**Done when**: `python src/ingest.py --help` and `python src/query.py --help` run without errors.

**Concepts you will learn:**
- **RAG pipeline overview** — the five steps every RAG system has: ingest → chunk → embed → store → retrieve+generate. You will see exactly where each script fits.
- **Separation of ingestion vs. querying** — why the pipeline is split into an offline phase (building the index) and an online phase (answering questions).

---

## Phase 1 — Config File

**Goal**: replace all magic strings with a single `config.yaml`.

- Create `config.yaml` with: `embedding_model`, `generation_model`, `chroma_path`, `chunk_size`, `chunk_overlap`
- Load config in all scripts via a shared `config.py` loader
- No hardcoded model names or paths anywhere in the source

**Done when**: changing `generation_model` in `config.yaml` changes which model is used.

**Concepts you will learn:**
- **Model names in Ollama** — the difference between an embedding model (converts text to numbers) and a generation model (produces answers), and why they are configured separately.
- **Hyperparameters** — `chunk_size` and `chunk_overlap` are your first RAG hyperparameters: settings that have no single correct value and must be tuned per use case.

---

## Phase 2 — GPU Verification

**Goal**: confirm that Ollama is using the GPU for both embedding and generation.

- Add a `check_gpu.py` script that calls `ollama ps` and parses GPU utilization
- Document the correct Ollama GPU setup for Windows + CUDA in a `docs/gpu-setup.md` file
- Test inference speed: record tokens/sec on CPU vs GPU for the chosen generation model

**Done when**: `ollama ps` shows GPU layers > 0 during inference and tokens/sec is measurably faster.

**Concepts you will learn:**
- **llama.cpp GPU offloading** — LLMs are split into layers; "GPU layers" means how many of those layers run on VRAM instead of RAM. More layers = faster, until you run out of VRAM.
- **VRAM constraints** — why model size (3B, 7B, 8B params) directly determines how much VRAM is needed, and how quantization (Q4, Q8) trades quality for memory.
- **Tokens per second** — the standard unit for measuring LLM inference speed, and why it varies between embedding and generation workloads.

---

## Phase 3 — Multi-Document Ingestion

**Goal**: ingest a directory of PDFs in one command.

- Rewrite `extract_pdf.py` to accept any PDF path (not hardcoded)
- Replace regex-based article chunking with `RecursiveCharacterTextSplitter` (configurable size + overlap)
- Attach metadata to every chunk: `source_file`, `page_number`, `chunk_index`
- CLI: `python src/ingest.py --dir ./books/` processes all PDFs in the folder

**Done when**: two different AI PDFs are ingested and their chunks appear in ChromaDB with correct source metadata.

**Concepts you will learn:**
- **Chunking strategies** — why you can't embed an entire book at once (token limits, retrieval granularity), and the trade-off between large chunks (more context, harder to retrieve precisely) vs. small chunks (easier to retrieve, less context per chunk).
- **Chunk overlap** — why repeating a few sentences between adjacent chunks helps avoid cutting a sentence mid-thought and losing meaning at boundaries.
- **Vector embeddings** — what it means to convert text into a list of numbers (a vector), why semantically similar text produces similar vectors, and why that enables semantic search rather than just keyword matching.
- **Metadata in vector stores** — every chunk stored in ChromaDB can carry a payload (source file, page number). This is how you trace an answer back to its origin.

---

## Phase 4 — Source Attribution in Answers

**Goal**: every answer cites the book and page it came from.

- After retrieval, extract `source_file` and `page_number` from chunk metadata
- Print sources below the answer in the CLI (e.g., `[Source: deep_learning.pdf, p. 42]`)
- Update the generation prompt to instruct the model to only answer from context

**Done when**: asking a question returns an answer plus at least one cited source.

**Concepts you will learn:**
- **Prompt engineering for RAG** — how to structure the system prompt so the LLM uses only the retrieved context and does not hallucinate from its training data. The pattern is: `Context: {retrieved_text} \n Question: {question} \n Answer only using the context above.`
- **Grounding vs. hallucination** — the difference between an answer that is grounded in retrieved text (safe) and one the model invented from its weights (hallucination), and why RAG reduces but does not eliminate hallucination.
- **The retrieval-generation split** — understanding that the LLM in RAG is only responsible for phrasing; correctness depends on what was retrieved, not what the model knows.

---

## Phase 5 — Chat UI + Feedback Capture

**Goal**: a local web chat interface, plus a feedback store that later phases can mine.

- Gradio `gr.Blocks` app (`src/app.py`), localhost only, `share=False`
- Multi-turn chat with a follow-up **query-rewriting** step before retrieval
  (`src/rewrite.py`, toggled by `rewrite_followups` in `config.yaml`)
- Per-answer feedback: 👍 / 👎, a free-text comment, and a "which cited sources helped"
  checkbox per source
- Everything logged to local SQLite (`src/feedback.py`, `feedback_db_path`): one row per
  answered turn with the retrieval params snapshotted, one row per retrieved chunk, one
  feedback row per turn
- `src/query.py` CLI is kept and stays stateless
- **Incremental ingestion** (pulled from `backlog.md`): chunk IDs become the stable
  `source_file::chunk_index`, `ingest` skips books already in the collection, `--force`
  rebuilds. Stable IDs are what keep the feedback log and future eval set pointing at the
  same chunks after a re-ingest.

**Done when**: `uv run python src/app.py` serves a chat that answers with citations, and
submitting feedback writes to `data/feedback.db`. Full checklist in
`specs/2026-09-10-phase-5-chat-ui-feedback/validation.md`.

**Concepts you will learn:**
- **Conversational RAG** — why a follow-up ("what about its downsides?") can't be embedded
  and retrieved as-is, and how a rewrite-to-standalone step fixes it.
- **Feedback collection vs. RLHF** — thumbs up/down is the *data* for preference learning,
  not the training. RLHF = preference data → reward model → RL fine-tune. DPO skips the
  reward model and consumes chosen/rejected pairs directly; this phase's schema is built
  to export those later.
- **Why ratings are stored next to retrieval params** — a judgement is only meaningful
  alongside the `chunk_size` / `chunk_overlap` / `top_k` that produced the answer; Phase 7
  tuning needs that join.

---

## Phase 6 — Evaluation Harness

**Goal**: a repeatable way to measure retrieval quality.

- Create `eval/eval_set.json` — 20 hand-crafted `(question, expected_chunk_ids)` pairs from ingested books
- Write `src/evaluate.py` that runs each question, checks if the expected chunk appears in top-k results, and reports hit rate and MRR
- Baseline: record initial hit rate before any tuning

**Scoring is page-level, not chunk-level** (implemented): entries carry
`expected_source_file` + `expected_pages`, and a hit is a retrieved chunk from that
file on one of those pages. Page numbers survive re-chunking, so the same set stays
valid through Phase 7; `source_file::chunk_index` IDs would not.

**Done when**: `python src/evaluate.py` prints a hit rate score and a per-question pass/fail table.

**Concepts you will learn:**
- **Hit rate (Recall@k)** — the fraction of test questions for which the correct chunk appears somewhere in the top-k retrieved results. The most intuitive RAG retrieval metric.
- **MRR (Mean Reciprocal Rank)** — a stricter metric that rewards finding the right chunk in position 1 more than position 5. Useful when you only pass the top chunk to the LLM.
- **Why you need a test set** — RAG systems can appear to work on the examples you tried while failing on others. A fixed eval set is the only way to know if a change actually helped.
- **The retrieval quality ceiling** — if retrieval is poor (wrong chunks returned), the LLM cannot give a correct answer regardless of how good it is. Eval makes this visible.

---

## Phase 7 — Chunking Tuning

**Goal**: improve retrieval quality by experimenting with chunk parameters.

- Run `evaluate.py` across a grid of `chunk_size` × `chunk_overlap` values (e.g., 256/512/1024 × 0%/10%/20%)
- Record results in `eval/results.md`
- Set `config.yaml` to the best-performing configuration

**Done when**: hit rate on eval set improves vs. Phase 6 baseline.

**Concepts you will learn:**
- **Hyperparameter search** — how to systematically explore a parameter grid rather than guessing, and how to use your eval set as the objective to optimize.
- **The chunking trade-off in practice** — you will directly observe how changing chunk size moves the hit rate up or down, making the abstract trade-off concrete with real data from your books.
- **Index rebuilding** — every change to chunking requires re-embedding from scratch; you will understand why the index is not a live view of the source documents.

---

## Phase 8 — Model Upgrade (optional)

**Goal**: evaluate a larger generation model if VRAM allows.

- Try `llama3.1:8b` or `mistral:7b` and compare answer quality on eval set
- Try `mxbai-embed-large` as an alternative embedding model and compare retrieval hit rate
- Document findings in `eval/results.md`

**Done when**: final model choices are documented with evidence.

**Concepts you will learn:**
- **Embedding model dimensionality** — larger embedding models produce higher-dimensional vectors (e.g., 768 vs. 1536 dims), which can capture more nuance at the cost of more storage and slower embedding.
- **Model capability vs. VRAM trade-off** — why a 7B model often answers better than a 3B model, and exactly how much VRAM each quantization level requires.
- **Decoupling embedding from generation** — the embedding model and the generation model are independent choices; you can improve either one without touching the other.

---

## Deferred / Out of Scope

- REST API or any non-local / LAN hosting (the Phase 5 chat UI is localhost-only)
- Multi-user support
- Model training of any kind (RLHF, DPO, LoRA) — Phase 5 only *collects* feedback
- Document re-indexing / change detection
- Non-PDF formats (EPUB, web pages)
