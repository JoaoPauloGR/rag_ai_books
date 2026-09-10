# Backlog

Future improvements deferred from completed phases.

## Incremental ingestion — PULLED INTO PHASE 5 (2026-09-10)

Originally deferred from Phase 3. Now specced in
`specs/2026-09-10-phase-5-chat-ui-feedback/` (Group 1b):

- Chunk IDs become `source_file::chunk_index` (deterministic, stable across runs).
- `ingest` uses `get_or_create_collection` and skips chunks whose ID is already present,
  plus a per-file fast skip; `--force` restores full wipe-and-rebuild.

### Still deferred (residual)

- **Content-hash change detection** — an ID-based skip cannot tell that a PDF's text
  changed (same filename, same chunk count → stale content kept). Needs a per-file
  content hash stored in metadata; re-ingest a file only when its hash differs.
- **Orphan-chunk cleanup** — a book removed from `data/books/` leaves its chunks in the
  collection until a `--force` rebuild. Add a sweep that deletes chunks whose
  `source_file` is no longer on disk.
