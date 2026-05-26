# Phase 1 — Validation

## Done when

All of the following pass before merging.

### 1. Config loads cleanly

```
python src/ingest.py --dir data/books/
```

Expected output includes:
```
Embedding model: nomic-embed-text
Chroma path: data/chroma_db
Chunk size: 512
Chunk overlap: 64
Ingestion not yet implemented
```

```
python src/query.py --query "What is a transformer?"
```

Expected output includes:
```
Generation model: llama3.2:3b
Embedding model: nomic-embed-text
Query not yet implemented
```

### 2. Config change is reflected at runtime

Edit `config.yaml`: set `generation_model: mistral:7b`  
Re-run `python src/query.py --query "test"`  
Expected: line reads `Generation model: mistral:7b`  
Revert the change after verifying.

### 3. Missing key raises a clear error

Remove `generation_model` from `config.yaml`, then run either script.  
Expected: `ValueError: config.yaml is missing required keys: generation_model`  
Restore the key after verifying.

### 4. Missing config file raises a clear error

```
python src/query.py --query "test" --config missing.yaml
```

Expected: `FileNotFoundError` (or a clear message that the file was not found), not a cryptic traceback.

## Not required for merge

- Actual PDF ingestion or ChromaDB connectivity
- Ollama running or models pulled
- GPU verification
