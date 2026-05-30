# Phase 4 — Implementation Plan

## Group 1 — Config extension

1. Add `top_k: 5` to `config.yaml`
2. Add `"top_k"` to the required-keys list in `src/config.py`

## Group 2 — Retrieval (`src/retrieve.py`)

3. Create `src/retrieve.py` with a single public function:
   ```python
   def retrieve_chunks(question: str, embedding_model: str, chroma_path: str, collection_name: str, top_k: int) -> list[dict]:
   ```
   - Embeds the question via `ollama.embeddings(model=embedding_model, prompt=question)`
   - Creates a `chromadb.PersistentClient(path=chroma_path)`
   - Calls `collection.query(query_embeddings=[embedding], n_results=top_k, include=["documents", "metadatas"])`
   - Returns a flat list of `{"text": str, "source_file": str, "page_number": int}` dicts

## Group 3 — Generation (`src/generate.py`)

4. Create `src/generate.py` with a single public function:
   ```python
   def generate_answer(question: str, chunks: list[dict], generation_model: str) -> str:
   ```
   - Joins chunk texts with `"\n\n---\n\n"` as a context block
   - Applies the hard-constraint prompt template from requirements.md
   - Calls `ollama.chat(model=generation_model, messages=[{"role": "user", "content": prompt}])`
   - Returns `response["message"]["content"].strip()`

## Group 4 — Wire query (`src/query.py`)

5. Replace the stub body with the full pipeline:
   - Load config (`load_config(args.config)`)
   - Call `retrieve_chunks(...)` with the question
   - Call `generate_answer(...)` with the retrieved chunks
   - Print the answer
   - Collect unique `(source_file, page_number)` pairs from chunks, preserving order of first appearance
   - Print each as `[Source: <source_file>, p. <page_number>]`
