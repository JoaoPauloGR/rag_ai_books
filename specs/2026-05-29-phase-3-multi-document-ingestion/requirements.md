# Phase 3 — Multi-Document Ingestion

## Context

Phases 0–2 established the project skeleton, config loading, and GPU verification. The
ingestion stub (`src/ingest.py`) accepts `--dir` and `--config` but does nothing yet.
Phase 3 turns it into a real pipeline: extract text from every PDF in a directory, split
it into overlapping chunks, and store those chunks in ChromaDB with source metadata.

The learning goals are understanding why you can't embed an entire book at once, what
chunk overlap does at boundaries, how semantic embeddings differ from keyword search, and
why metadata in the vector store is the foundation for attribution in Phase 4.

## Scope

### What this phase produces

- `src/extract.py` — `extract_pages(pdf_path) -> list[dict]` returning `{page_number, text}` per page
- `src/chunk.py` — `chunk_pages(pages, chunk_size, chunk_overlap) -> list[dict]` using
  `RecursiveCharacterTextSplitter`; each chunk carries `source_file`, `page_number`, `chunk_index`
- `src/store.py` — `build_collection(chunks, embedding_model, chroma_path, collection_name)`
  that wipes the existing collection and re-embeds from scratch
- Updated `src/ingest.py` — wires extract → chunk → store; logs progress per file; skips
  failed PDFs with a warning; prints a summary at the end

### What this phase does NOT produce

- Source attribution in query output (Phase 4)
- Any changes to `src/query.py` — it still prints "Query not yet implemented"
- An evaluation harness (Phase 5)
- De-duplication or incremental re-ingestion

## Decisions

| Decision | Choice | Reason |
|---|---|---|
| Chunker | `RecursiveCharacterTextSplitter` (LangChain) | Already in `requirements.txt`; configurable via existing `chunk_size`/`chunk_overlap` config keys |
| Collection strategy | Single shared collection | Enables cross-book queries; source is distinguished by `source_file` metadata |
| Re-ingestion behaviour | Wipe and rebuild | Index always reflects current files; no dedup logic required |
| Failed PDF handling | Skip with warning | Unattended batch runs should not abort on a single bad file |
| Collection name | Configured in `config.yaml` as `collection_name` | Avoids a new magic string; consistent with Phase 1 pattern |
| Chunk metadata keys | `source_file`, `page_number`, `chunk_index` | Matches roadmap spec; `source_file` and `page_number` are the minimum needed for Phase 4 attribution |

## New config key

Add to `config.yaml`:

```yaml
collection_name: books
```

`src/config.py` must add `collection_name` to the required-keys list.

## Out of scope

- Incremental ingestion / skip-if-present logic
- Non-PDF formats (EPUB, HTML, plain text)
- Per-collection filtering in queries
- Metadata beyond `source_file`, `page_number`, `chunk_index`
