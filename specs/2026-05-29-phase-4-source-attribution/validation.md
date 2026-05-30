# Phase 4 — Validation

## Done when

All of the following pass before merging.

### 1. Basic query returns an answer with sources

Ensure the ChromaDB collection from Phase 3 is populated. Run:

```
python src/query.py --query "What is attention in transformers?"
```

Expected:
- A paragraph-length answer is printed
- One or more source lines follow below the answer: `[Source: <file>.pdf, p. <N>]`
- Exit code 0

### 2. Sources are deduplicated

Ask a question likely to pull multiple chunks from the same page. Inspect the output.

Expected:
- Each unique `(source_file, page_number)` pair appears at most once
- Total source lines ≤ `top_k` (5 by default)

### 3. Hard constraint triggers on an out-of-corpus question

```
python src/query.py --query "What is the capital of France?"
```

Expected:
- The model responds with something close to: "I don't have enough information in the retrieved passages to answer that."
- It does not produce a confident answer drawn from training data

### 4. `top_k` config key is respected

Edit `config.yaml`: set `top_k: 2`. Re-run any query and count the source lines.

Expected: at most 2 unique source citations. Revert to 5 after verifying.

### 5. Missing `top_k` key raises a clear error

Remove `top_k` from `config.yaml` and run query.

Expected: `ValueError: config.yaml is missing required keys: top_k`  
Restore the key after verifying.

### 6. Empty or missing collection raises a useful error

Rename `data/chroma_db/` temporarily so the collection does not exist. Run a query.

Expected: a clear error message is printed and the process exits non-zero. No raw Python traceback.

## Not required for merge

- Hit-rate measurement (Phase 5)
- Chunk tuning experiments (Phase 6)
- A larger generation model (Phase 7)
- Streaming output
