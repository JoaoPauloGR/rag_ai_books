# Phase 3 — Validation

## Done when

All of the following pass before merging.

### 1. Single PDF ingests cleanly

Place one AI-related PDF in `data/books/` and run:

```
python src/ingest.py --dir data/books/
```

Expected:
- Prints `Ingesting: <filename>.pdf` for the file
- Prints batch progress lines (`Stored 100/... chunks...`)
- Ends with `Done. Ingested 1 files, N chunks total.` where N > 0
- Exit code 0

### 2. Two PDFs are both stored with correct metadata

Ingest a directory containing two PDFs. Then open a Python shell:

```python
import chromadb
client = chromadb.PersistentClient(path="data/chroma_db")
col = client.get_collection("books")
print(col.count())  # must be > 0

results = col.get(limit=5, include=["metadatas"])
for m in results["metadatas"]:
    print(m)
```

Expected:
- `col.count()` returns total chunks from both files
- Each metadata dict contains `source_file`, `page_number` (int ≥ 1), and `chunk_index` (int ≥ 0)
- Both source filenames appear across the metadata sample

### 3. Re-running rebuilds from scratch

Run ingest twice on the same directory. After the second run:

```python
col.count()
```

Expected: same count as after the first run (not doubled). The wipe-and-rebuild ensures no duplicates.

### 4. A corrupt or unreadable PDF is skipped

Add a zero-byte file named `bad.pdf` to `data/books/` and re-run ingest.

Expected:
- A warning line is printed for `bad.pdf` (e.g., `Warning: failed to extract bad.pdf — <reason>`)
- The other PDFs are ingested successfully
- Final summary reflects the skipped file count or simply the successful file count
- Exit code 0 (does not abort)

### 5. config.yaml change is respected

Edit `config.yaml`: set `chunk_size: 256`  
Re-run ingest and check `col.count()`.

Expected: chunk count increases compared to the default 512 size (more, smaller chunks).  
Revert `chunk_size` to 512 after verifying.

### 6. Missing `collection_name` key raises a clear error

Remove `collection_name` from `config.yaml` and run ingest.

Expected: `ValueError: config.yaml is missing required keys: collection_name`  
Restore the key after verifying.

## Not required for merge

- `src/query.py` returning real answers (Phase 4)
- Source attribution printed in the CLI (Phase 4)
- An evaluation set or hit-rate metric (Phase 5)
- Ollama GPU usage verified during embedding (already done in Phase 2)
