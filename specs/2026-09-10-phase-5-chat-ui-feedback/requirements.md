# Phase 5 — Chat UI + Feedback Capture

## Context

Phases 0–4 produced a working single-shot RAG pipeline: `src/query.py` retrieves the
`top_k` nearest chunks, generates a grounded answer, and prints deduplicated
`[Source: file.pdf, p. N]` citations. It is CLI-only and stateless.

This phase adds a **local web chat interface** (Gradio) and a **feedback capture
system**. Every answered turn and every piece of user feedback is written to a local
SQLite database. The database schema is designed so a later phase can export two things
from it without re-running anything: eval-set rows (question + quality label) and DPO
preference pairs (chosen vs. rejected answer for a question).

It also pulls in **incremental ingestion** from `specs/backlog.md` (deferred from
Phase 3). Today `src/ingest.py` wipes and rebuilds the whole ChromaDB collection on every
run, and chunk IDs are a positional running integer (`"0"`, `"1"`, …) reassigned from
scratch each time. That makes adding one book cost a full re-embed of all of them, and it
means any stored reference to a chunk ID (this phase's `turn_sources.chunk_id`, a future
eval set's `expected_chunk_ids`) silently points at different text after the next ingest.
This phase switches to deterministic IDs and skips chunks that are already present, so
re-ingestion is cheap and chunk references stay valid.

This is an explicit change to two project non-goals ("CLI only", "Web UI"). The CLI is
kept and stays stateless; the web UI is the new primary surface for interactive use.

### Learning goals

- **Conversational RAG** — why a follow-up like "what about its downsides?" cannot be
  embedded and retrieved directly, and how a query-rewriting step turns it into a
  standalone question before retrieval.
- **Feedback vs. RLHF** — collecting thumbs up/down is *feedback collection*, not RLHF.
  RLHF is: collect preference data → train a reward model → RL-fine-tune the LLM. This
  phase builds only the first step's data store. DPO (preference pairs, no reward model)
  is the simpler path that this schema targets.
- **Why retrieval params are snapshotted with feedback** — a rating is only meaningful
  next to the `chunk_size` / `chunk_overlap` / `top_k` that produced it; Phase 6 tuning
  needs that join.
- **Idempotent ingestion & stable IDs** — deriving a chunk ID from where it came from
  (`source_file::chunk_index`) instead of an auto-incrementing counter makes a re-run a
  near no-op and, more importantly, keeps every external reference to that chunk (a
  feedback row, an eval-set entry) pointing at the same text. The limit: an ID-based skip
  cannot detect that a file's *contents* changed — that needs a content hash or a
  `--force` rebuild.

## Scope

### What this phase produces

- **`src/rewrite.py`** — `rewrite_followup(history: list[dict], question: str, generation_model: str) -> str`
  - If `history` is empty, returns `question` unchanged.
  - Otherwise calls `ollama.chat` with a rewrite prompt (see below) that folds the prior
    turns into a single self-contained question. Returns the rewritten string.

- **`src/feedback.py`** — SQLite persistence layer, stdlib `sqlite3` only:
  - `init_db(db_path: str) -> None` — creates the parent dir and the three tables if
    absent (idempotent).
  - `record_turn(db_path, *, session_id, raw_question, retrieval_question, answer, chunks, cfg) -> int`
    — inserts one `turns` row and one `turn_sources` row per retrieved chunk; returns the
    new `turn_id`.
  - `record_feedback(db_path, *, turn_id, rating, comment, helpful_ranks) -> None`
    — inserts one `feedback` row and sets `turn_sources.helpful` for the given ranks
    (1 for ranks in `helpful_ranks`, 0 for the other cited ranks). Re-submitting for the
    same `turn_id` replaces the prior feedback row.

- **`src/app.py`** — Gradio `gr.Blocks` app, entry point `uv run python src/app.py`
  (`--config`, default `config.yaml`; `--port`, default `7860`). Runs
  `demo.launch(server_name="127.0.0.1", share=False)`.
  - Chatbot + input box. On each user message: rewrite → retrieve → generate → append to
    the transcript → `record_turn` → stash the returned `turn_id` and the turn's
    deduplicated citations in session state.
  - A "Feedback on the last answer" panel: a 👍 / 👎 control, a `CheckboxGroup` populated
    with the last answer's deduplicated `[Source: file, p. N]` strings, a free-text
    comment box, and a Submit button that calls `record_feedback`.

- **`src/retrieve.py`** — extended: each returned dict also carries `chunk_id` (the
  ChromaDB id, from `result["ids"][0]`). Existing keys and callers are unchanged;
  `query.py` ignores the new key. With incremental ingestion in place these IDs are the
  stable `source_file::chunk_index` strings, so `turn_sources.chunk_id` stays meaningful
  across re-ingests.

- **`src/store.py`** — incremental build:
  - Chunk IDs become `f"{chunk['source_file']}::{chunk['chunk_index']}"` (deterministic).
  - `client.get_or_create_collection(collection_name)` instead of delete-then-create.
  - Reads existing IDs once (`collection.get(include=[])["ids"]`) and adds only chunks
    whose ID is not already present.
  - New keyword arg `force: bool = False` — when true, deletes the collection first
    (restores full wipe-and-rebuild).
  - Returns `{"added": int, "skipped": int}`.

- **`src/ingest.py`** — incremental run:
  - New `--force` flag, passed through to `build_collection`.
  - Fast path: before extracting a PDF, skip it entirely if `--force` is not set and a
    chunk with that `source_file` already exists in the collection (avoids re-running
    PyMuPDF and the splitter on unchanged books).
  - Final summary reports chunks added, chunks/files skipped as already present, and
    files that failed extraction.

- **Config** — two new required keys in `config.yaml` and `src/config.py`:
  ```yaml
  feedback_db_path: data/feedback.db
  rewrite_followups: true
  ```

- **`data/feedback.db`** added to `.gitignore`.
- **`gradio`** added to `pyproject.toml` dependencies.
- Tests: `tests/test_rewrite.py`, `tests/test_feedback.py`, and updates to
  `tests/test_retrieve.py` for the new `chunk_id` key.

### What this phase does NOT produce

- Any model training — no RLHF, no DPO, no LoRA.
- The feedback exporter (eval-set JSONL / DPO-pair JSONL) — that is Phase 6+.
- An "edit the answer" / gold-answer field, or a regenerate-for-alternative button.
- Retrieval metrics (hit rate, MRR) — Phase 6.
- Auth, multi-user, or remote/LAN hosting — localhost only, `share=False`.
- Token streaming in the UI (allowed if free; not required).
- Content-hash change detection for modified PDFs — a changed file is only re-ingested via
  `--force`.
- Deleting chunks for a book that was removed from the directory (orphans stay until
  `--force`).
- Changes to chunking or `src/generate.py`'s prompt.
- Persisting the raw chat transcript beyond what the `turns` table records.

## Decisions

| Decision | Choice | Reason |
|---|---|---|
| Frontend | Gradio `gr.Blocks` | Pure Python, one file, local by default; Blocks (not `ChatInterface`) for full control over the feedback panel |
| Conversation | Multi-turn with follow-up rewriting | A standalone-question rewrite step keeps retrieval accurate on follow-ups; the extra LLM call is the cost |
| Rewriting toggle | `rewrite_followups` config key | Lets the rewrite step be disabled for comparison / debugging without code changes |
| CLI | Kept, unchanged, still stateless | Handy for scripting and regression; no rewrite step there (nothing to rewrite) |
| Feedback store | SQLite (`sqlite3` stdlib) at `feedback_db_path` | Queryable, single file, no new dependency; JSONL is write-only and awkward to update |
| Feedback granularity | Rating + comment + per-cited-source `helpful` flag | Per-source flags are the signal Phase 6 retrieval tuning needs |
| Retrieval-param snapshot | `turns` row stores `top_k`, `chunk_size`, `chunk_overlap`, both model names | A rating is only interpretable next to the params that produced it |
| Chunk-text snapshot | `turn_sources.chunk_text` stores the retrieved text | Eval/DPO export must not depend on the ChromaDB index still existing or being unchanged |
| Preference pairs | Assembled later from thumbs-up vs. thumbs-down answers to the same/similar question | No gold-answer editing this phase; the schema keeps enough to pair rows offline |
| One feedback row per turn | Re-submit replaces | Solo use; last judgement wins, keeps queries simple |
| Hosting | `127.0.0.1`, `share=False` | Privacy is a project goal; nothing leaves the machine |
| Chunk ID scheme | `f"{source_file}::{chunk_index}"` | Deterministic and stable across runs; keeps `turn_sources.chunk_id` and future eval-set references valid after a re-ingest |
| Ingestion mode | Incremental by default (skip present IDs), `--force` to wipe | Adding one book no longer re-embeds the whole corpus; `--force` is the escape hatch |
| Skip granularity | Per chunk ID in `store.py`, plus a per-`source_file` fast skip in `ingest.py` | New books add cleanly; an unchanged corpus is a near no-op that also avoids re-extraction |
| Changed / removed files | Not handled — `--force` only | Content hashing and orphan deletion are out of scope while the corpus is still being built up |
| Collection creation | `get_or_create_collection` | Incremental adds need the collection to survive between runs |
| Duplicate filenames across dirs | Not handled — treated as the same book | Two different PDFs with the same filename would collide on ID; documented, not guarded |

## SQLite schema

```sql
CREATE TABLE IF NOT EXISTS turns (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  TEXT NOT NULL,              -- ISO-8601 UTC
    session_id          TEXT NOT NULL,             -- one per app launch
    raw_question        TEXT NOT NULL,             -- what the user typed
    retrieval_question  TEXT NOT NULL,             -- after follow-up rewrite (= raw if none)
    answer              TEXT NOT NULL,
    embedding_model     TEXT NOT NULL,
    generation_model    TEXT NOT NULL,
    top_k               INTEGER NOT NULL,
    chunk_size          INTEGER NOT NULL,
    chunk_overlap       INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS turn_sources (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id      INTEGER NOT NULL REFERENCES turns(id),
    rank         INTEGER NOT NULL,                 -- 0-based retrieval position
    chunk_id     TEXT NOT NULL,                    -- ChromaDB id
    source_file  TEXT NOT NULL,
    page_number  INTEGER NOT NULL,
    chunk_text   TEXT NOT NULL,                    -- snapshot of retrieved text
    cited        INTEGER NOT NULL,                 -- 1 if in the deduped citation list shown
    helpful      INTEGER                           -- NULL = not reviewed, 1 = ticked, 0 = shown-not-ticked
);

CREATE TABLE IF NOT EXISTS feedback (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id   INTEGER NOT NULL UNIQUE REFERENCES turns(id),
    ts        TEXT NOT NULL,
    rating    TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    comment   TEXT
);
```

## Prompts

### Follow-up rewrite (`src/rewrite.py`)

```
Given the conversation so far and a follow-up question, rewrite the follow-up as a
single standalone question that can be understood without the conversation. Keep it
faithful to the user's intent. Do not answer it. Return only the rewritten question.

Conversation:
{history}

Follow-up: {question}

Standalone question:
```

`{history}` is the prior turns rendered as `User: ...` / `Assistant: ...` lines.

The answer-generation prompt in `src/generate.py` is unchanged.

## Out of scope

- Model training of any kind (RLHF, DPO, LoRA) and the feedback exporter.
- Gold-answer editing, regenerate-for-alternative, multi-rating history per turn.
- Hit rate / MRR / retrieval metrics (Phase 6).
- Streaming output, auth, LAN/remote hosting, chat-history export.
- Content-hash change detection, orphan-chunk deletion (incremental ingestion is
  ID-skip + `--force` only).
- Any change to chunking or the generation prompt.
