# Phase 5 — Implementation Plan

Groups are ordered so the suite stays green after each one. Commit per group.

## Group 1 — Config, dependency, and incremental ingestion

### 1a — Config + dependency

1. Add to `config.yaml`:
   ```yaml
   feedback_db_path: data/feedback.db
   rewrite_followups: true
   ```
2. Add `"feedback_db_path"` and `"rewrite_followups"` to `_REQUIRED_KEYS` in `src/config.py`.
3. Add `gradio` to `[project].dependencies` in `pyproject.toml`; run `uv sync`.
4. Add `data/feedback.db` to `.gitignore`.
5. `tests/test_config.py`: assert the two new keys are required (missing-key `ValueError`
   message lists them).

### 1b — Stable chunk IDs + incremental ingestion

6. `src/store.py` — rework `build_collection(chunks, embedding_model, chroma_path,
   collection_name, *, force=False) -> dict`:
   - `client = chromadb.PersistentClient(path=chroma_path)`.
   - If `force`: `try: client.delete_collection(collection_name) except Exception: pass`.
   - `collection = client.get_or_create_collection(collection_name)`.
   - `existing = set(collection.get(include=[])["ids"])`.
   - Define `chunk_id(c) = f'{c["source_file"]}::{c["chunk_index"]}'`.
   - `new_chunks = [c for c in chunks if chunk_id(c) not in existing]`;
     `skipped = len(chunks) - len(new_chunks)`.
   - Batch `new_chunks` (size 100) as today, but `ids = [chunk_id(c) for c in batch]`.
     Keep the metadata keys (`source_file`, `page_number`, `chunk_index`) unchanged.
   - Return `{"added": len(new_chunks), "skipped": skipped}`.
7. `src/ingest.py`:
   - Add `parser.add_argument("--force", action="store_true", help="Wipe the collection and re-embed every file")`.
   - Before the per-file loop, when `not args.force`, open the collection and compute
     `existing_sources = {i.split("::")[0] for i in collection.get(include=[])["ids"]}`
     (use `client.get_or_create_collection`; empty set if the collection is new).
   - In the loop: `if not args.force and pdf_path.name in existing_sources:
     print(f"Skipping (already ingested): {pdf_path.name}"); skipped_files += 1; continue`.
   - Pass `force=args.force` to `build_collection`; capture its return dict.
   - Final line: `f"Done. {result['added']} chunks added, {result['skipped']} chunks already present, {skipped_files} files skipped, {failed} files failed."`
8. `tests/test_store.py` — update:
   - mocks provide `get_or_create_collection` and `collection.get(include=[])` returning
     `{"ids": [...]}`;
   - IDs passed to `add` are now `["book.pdf::0", "book.pdf::1", ...]` (replaces
     `test_ids_are_sequential_strings`);
   - re-running with the same chunks (all IDs already in `existing`) makes no `add` call
     and returns `{"added": 0, "skipped": 5}`;
   - `force=True` calls `delete_collection` once; default does not;
   - `get_or_create_collection` is used, not `create_collection`.
   - While here, fix the stale `ollama.embeddings` patch target — real code calls
     `ollama.embed(...).embeddings[0]`; patch that so the store tests stop hitting a live
     Ollama.
9. `tests/test_ingest.py` — add:
   - second run over the same `--dir` (collection already has those `source_file`s) skips
     every file and calls `build_collection` with an empty `all_chunks` (or with
     `force=False` and everything filtered);
   - a run with one new filename processes only that file;
   - `--force` re-processes all files and `build_collection` receives `force=True`;
   - the mocked client exposes `get_or_create_collection(...).get(include=[])`.

## Group 2 — Retrieve returns `chunk_id`

10. In `src/retrieve.py`, add `chunk_id` to each returned dict from `result["ids"][0]`
    (ids come back without being named in `include`). Zip it alongside documents/metadatas.
11. `tests/test_retrieve.py`: update the mocked Chroma `query` return to include `ids`;
    assert every result dict has a non-empty `chunk_id`. Confirm `query.py` still works
    (it just ignores the key).

## Group 3 — Follow-up rewriting (`src/rewrite.py`)

12. Create `src/rewrite.py` with:
   ```python
   def rewrite_followup(history: list[dict], question: str, generation_model: str) -> str:
   ```
   - `history` is a list of `{"role": "user"|"assistant", "content": str}`.
   - Empty `history` → return `question` unchanged (no LLM call).
   - Else render history as `User: ...` / `Assistant: ...` lines, fill the rewrite prompt
     from requirements.md, call `ollama.chat`, return
     `response["message"]["content"].strip()`.
13. `tests/test_rewrite.py` (mock `ollama.chat`):
    - empty history returns the input verbatim and makes no call;
    - non-empty history calls `ollama.chat` once and returns the stripped content;
    - the rendered prompt contains both the history lines and the follow-up.

## Group 4 — Feedback persistence (`src/feedback.py`)

14. Create `src/feedback.py` (stdlib `sqlite3`) with `init_db`, `record_turn`,
    `record_feedback` per requirements.md. `init_db` makes the parent dir
    (`Path(db_path).parent.mkdir(parents=True, exist_ok=True)`) and runs the three
    `CREATE TABLE IF NOT EXISTS` statements. Every public function calls `init_db` first
    so the app never has to.
15. `record_turn` writes the `turns` row (params pulled from `cfg`), then one
    `turn_sources` row per chunk with `rank` = list index, `cited` = 1 if
    `(source_file, page_number)` is the first occurrence of that pair. Returns
    `cursor.lastrowid`.
16. `record_feedback` does an `INSERT ... ON CONFLICT(turn_id) DO UPDATE` on `feedback`,
    then `UPDATE turn_sources SET helpful = ...` — 1 where `rank IN helpful_ranks`,
    0 for the other `cited = 1` rows of that turn.
17. `tests/test_feedback.py` against a `tmp_path` db:
    - `init_db` twice is a no-op the second time; tables exist;
    - `record_turn` returns an int and writes `len(chunks)` `turn_sources` rows with the
      right `cited` flags and a `chunk_text` snapshot;
    - `record_feedback` writes one row; re-calling replaces it (still one row) and
      updates `helpful` (ticked → 1, cited-but-unticked → 0, uncited → NULL).

## Group 5 — Gradio app (`src/app.py`)

18. Create `src/app.py`:
    - `build_demo(cfg) -> gr.Blocks` so tests can import without launching.
    - `main()`: parse `--config` / `--port`, `load_config`, `init_db(cfg["feedback_db_path"])`,
      `build_demo(cfg).launch(server_name="127.0.0.1", server_port=args.port, share=False)`.
    - State: `gr.State` for `session_id` (a `uuid4` per load) and for `last_turn`
      (`{"turn_id": int, "citations": list[str], "ranks_by_citation": dict[str,list[int]]}`).
    - `on_message(user_msg, chat_history, session_id, last_turn)`:
      1. build `history` dicts from `chat_history`;
      2. `retrieval_q = rewrite_followup(history, user_msg, cfg["generation_model"])`
         if `cfg["rewrite_followups"]` and history else `user_msg`;
      3. `chunks = retrieve_chunks(retrieval_q, cfg["embedding_model"], cfg["chroma_path"], cfg["collection_name"], cfg["top_k"])`
         — wrap `chromadb.errors.NotFoundError` into a friendly chat message, return early;
      4. `answer = generate_answer(retrieval_q, chunks, cfg["generation_model"])`;
      5. dedupe citations preserving first-seen order; build
         `[Source: file, p. N]` strings and a map from each string to the chunk ranks it covers;
      6. `turn_id = record_turn(cfg["feedback_db_path"], session_id=..., raw_question=user_msg, retrieval_question=retrieval_q, answer=answer, chunks=chunks, cfg=cfg)`;
      7. append `answer` + newline-joined citations to `chat_history`;
      8. return updated chat, cleared input, `CheckboxGroup(choices=citations, value=[])`,
         and the new `last_turn` state.
    - `on_feedback(rating, helpful_citations, comment, last_turn)`:
      - no-op with a warning toast if `last_turn` is `None` or `rating` unset;
      - map ticked citation strings → flat rank list via `ranks_by_citation`;
      - `record_feedback(cfg["feedback_db_path"], turn_id=last_turn["turn_id"], rating=..., comment=comment or None, helpful_ranks=ranks)`;
      - `gr.Info("Feedback saved")`; clear the comment box and rating.
19. Layout: `gr.Chatbot(type="messages")`, `gr.Textbox` + send button, then a
    `gr.Accordion("Feedback on the last answer", open=True)` containing
    `gr.Radio(["👍", "👎"], label="Rating")`, `gr.CheckboxGroup([], label="Which sources helped?")`,
    `gr.Textbox(label="Comment", lines=2)`, `gr.Button("Submit feedback")`.
20. `tests/test_app.py` (no server): `build_demo(cfg)` returns a `gr.Blocks`; with
    `retrieve_chunks` / `generate_answer` / `ollama` mocked, calling `on_message`
    produces an answer turn and writes a `turns` row to a `tmp_path` db.

## Group 6 — Docs

21. `README.md`: new "Chat UI" usage block (`uv run python src/app.py`, opens
    `http://127.0.0.1:7860`), note the feedback panel and that everything is stored in
    `data/feedback.db` locally. Document incremental ingestion + `--force` in the ingest
    section. Bump the roadmap table (old 5→6, 6→7, 7→8; new row 5).
22. `specs/roadmap.md`, `specs/mission.md`, and `specs/backlog.md` already updated
    alongside this spec; no action here beyond confirming they match.

## Manual verification

Run every check in `validation.md` end to end against the real ChromaDB collection and
Ollama before marking the phase complete.
