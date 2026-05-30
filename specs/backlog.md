# Backlog

Future improvements deferred from completed phases.

## Incremental ingestion (deferred from Phase 3)

Instead of wiping and rebuilding the ChromaDB collection on every `ingest` run, only add
chunks whose IDs are not already present in the collection.

**Approach:** Generate chunk IDs as `source_file::chunk_index` so they are stable across
runs. Before calling `collection.add`, query existing IDs and skip any that match.

**Why deferred:** Adds dedup logic and complicates the ID scheme. Wipe-and-rebuild is
correct and simple for Phase 3; incremental is useful once the book corpus stabilises and
re-ingestion time becomes a cost.
