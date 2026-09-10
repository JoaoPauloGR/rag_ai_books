# Phase 5 — Validation

All checks pass before merging. Ollama running and the Phase 3 `books` collection
populated.

## 1. App launches and answers with sources

```
uv run python src/app.py
```

Expected:
- Prints a local URL (`http://127.0.0.1:7860`), no `share` link.
- Asking "What is attention in transformers?" returns a paragraph answer followed by one
  or more `[Source: <file>.pdf, p. <N>]` lines in the chat.

## 2. Follow-up rewriting works

With `rewrite_followups: true`, ask in sequence:
1. "What is a transformer?"
2. "How does it differ from an RNN?"

Then inspect the last `turns` row:

```
sqlite3 data/feedback.db "SELECT raw_question, retrieval_question FROM turns ORDER BY id DESC LIMIT 1;"
```

Expected: `raw_question` is "How does it differ from an RNN?" and `retrieval_question` is
a standalone rewrite that names the transformer (they differ).

Set `rewrite_followups: false`, restart, repeat. Expected: the two columns are now
identical. Restore `true`.

## 3. Every answered turn is persisted

After asking N questions:

```
sqlite3 data/feedback.db "SELECT count(*) FROM turns;"
sqlite3 data/feedback.db "SELECT count(*) FROM turn_sources WHERE turn_id = (SELECT max(id) FROM turns);"
```

Expected: `turns` count == N; `turn_sources` count for the last turn == `top_k` (5).
Spot-check one `turn_sources` row has non-empty `chunk_text`, a `chunk_id`, and the
`turns` row has `top_k`, `chunk_size`, `chunk_overlap`, and both model names filled in.

## 4. Feedback submission writes and updates

On the last answer: pick 👍, tick exactly one of the source checkboxes, type a comment,
click Submit. Expect a "Feedback saved" notice. Then:

```
sqlite3 data/feedback.db "SELECT rating, comment FROM feedback WHERE turn_id = (SELECT max(id) FROM turns);"
sqlite3 data/feedback.db "SELECT rank, cited, helpful FROM turn_sources WHERE turn_id = (SELECT max(id) FROM turns) ORDER BY rank;"
```

Expected:
- one `feedback` row, `rating = 'up'`, comment matches.
- the ticked citation's chunk rank(s) have `helpful = 1`; other `cited = 1` rows have
  `helpful = 0`; any `cited = 0` rows have `helpful = NULL`.

## 5. Re-submitting feedback replaces, not appends

Submit feedback again on the same answer with 👎 and no ticks.

Expected: still exactly one `feedback` row for that `turn_id`, now `rating = 'down'`;
previously-helpful rows are back to `helpful = 0`.

## 6. New config keys are required

Remove `feedback_db_path` from `config.yaml`, run `uv run python src/app.py`.

Expected: `ValueError: config.yaml is missing required keys: feedback_db_path`
(and `rewrite_followups` if both removed). No Gradio server starts. Restore the keys.

## 7. DB is created on first run

Delete `data/feedback.db`, launch the app, ask one question.

Expected: the file is recreated with the three tables; no traceback. If `data/` itself
were missing it would be created too.

## 8. CLI regression

```
uv run python src/query.py --query "What is attention in transformers?"
```

Expected: unchanged Phase 4 behaviour — answer plus deduplicated `[Source: ...]` lines,
exit 0. No dependency on `feedback.db`.

## 9. Out-of-corpus question still refuses

Ask "What is the capital of France?" in the chat.

Expected: "I don't have enough information in the retrieved passages to answer that."
(no confident training-data answer). A `turns` row is still written for it.

## 10. Missing collection is handled in the UI

Rename `data/chroma_db/` temporarily, launch, ask a question.

Expected: a clear in-chat error message ("collection ... not found, run ingestion
first"), the app stays up, no raw traceback in the chat. Restore the directory.

## 11. Tests and gitignore

```
uv run pytest -q
git status --porcelain data/feedback.db
```

Expected: all tests pass, including `test_rewrite.py`, `test_feedback.py`,
`test_app.py`, and the updated `test_retrieve.py`, `test_store.py`, `test_ingest.py`.
`git status` shows nothing for `data/feedback.db` (ignored).

## 12. Chunk IDs are stable and content-derived

After a normal ingest:

```
uv run python -c "import chromadb; c=chromadb.PersistentClient('data/chroma_db').get_collection('books'); print(c.get(include=[])['ids'][:5])"
```

Expected: IDs look like `aiengineering.pdf::0`, `aiengineering.pdf::1`, … — not `0`, `1`.

## 13. Re-running ingest is a near no-op

With the `books` collection already populated:

```
uv run python src/ingest.py --dir data/books/
```

Expected:
- Per-file "Skipping (already ingested): …" lines.
- Summary: `0 chunks added, … files skipped, 0 files failed`.
- No Ollama embedding calls (finishes in seconds).
- ChromaDB count is unchanged:
  `uv run python -c "import chromadb; print(chromadb.PersistentClient('data/chroma_db').get_collection('books').count())"`

## 14. Adding one book ingests only that book

Drop one new small PDF into `data/books/` and re-run the ingest command.

Expected: only the new file prints an "Ingesting" line; the others are skipped; the
collection count grows by exactly that file's chunk count; existing IDs are untouched.

## 15. `--force` rebuilds from scratch

```
uv run python src/ingest.py --dir data/books/ --force
```

Expected: the collection is deleted and every file re-embedded; summary shows all chunks
added, 0 skipped. Count matches a fresh build.

## 16. Feedback survives a re-ingest

Ask a question in the chat and submit feedback (so `turn_sources.chunk_id` rows exist).
Run a normal (non-`--force`) ingest. Re-open the DB:

```
sqlite3 data/feedback.db "SELECT chunk_id FROM turn_sources ORDER BY id DESC LIMIT 3;"
```

Expected: the stored `chunk_id` values still resolve to the same chunks in ChromaDB
(`collection.get(ids=[...])` returns the matching text). A `--force` rebuild keeps the
same IDs too, as long as the PDFs and chunk params are unchanged.

## Not required for merge

- Hit rate / MRR measurement (Phase 6)
- The feedback exporter — eval-set JSONL and DPO-pair JSONL (Phase 6)
- Any model training (RLHF / DPO / LoRA)
- Token streaming, auth, LAN hosting
- Content-hash detection of a changed PDF, orphan-chunk cleanup (`--force` only)
- Chunk-parameter tuning (Phase 7)
