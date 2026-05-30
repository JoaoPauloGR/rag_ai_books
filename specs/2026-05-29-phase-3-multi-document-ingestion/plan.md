# Phase 3 — Implementation Plan

## Group 1 — Config extension

1. Add `collection_name: books` to `config.yaml`
2. Add `"collection_name"` to the required-keys list in `src/config.py`

## Group 2 — PDF extraction (`src/extract.py`)

3. Create `src/extract.py` with a single public function:
   ```python
   def extract_pages(pdf_path: str) -> list[dict]:
   ```
   - Opens the PDF with `fitz.open(pdf_path)`
   - Iterates pages; calls `page.get_text("text")` for each
   - Returns a list of `{"page_number": int, "text": str}` (1-indexed page numbers)
   - Raises `RuntimeError` on any `fitz` error so the caller can catch and skip

## Group 3 — Chunking (`src/chunk.py`)

4. Create `src/chunk.py` with a single public function:
   ```python
   def chunk_pages(pages: list[dict], source_file: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
   ```
   - Instantiates `RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)`
   - For each page dict, splits `page["text"]` into sub-chunks
   - Returns a flat list of:
     ```python
     {"text": str, "source_file": str, "page_number": int, "chunk_index": int}
     ```
   - `chunk_index` is a global counter across all pages for the file (0-based)
   - Pages with no text (blank pages) are skipped silently

## Group 4 — Vector store (`src/store.py`)

5. Create `src/store.py` with a single public function:
   ```python
   def build_collection(chunks: list[dict], embedding_model: str, chroma_path: str, collection_name: str) -> None:
   ```
   - Creates a `chromadb.PersistentClient(path=chroma_path)`
   - Deletes the collection if it already exists (`client.delete_collection(collection_name)`)
   - Creates a fresh collection
   - Embeds each chunk's `"text"` via `ollama.embeddings(model=embedding_model, prompt=text)`
   - Calls `collection.add(ids=[...], embeddings=[...], documents=[...], metadatas=[...])`
   - Batches upserts in groups of 100 to avoid memory spikes on large collections
   - Prints progress every batch (e.g., `Stored 100/843 chunks...`)

## Group 5 — Wire ingest (`src/ingest.py`)

6. Replace the stub body with the full pipeline:
   - Load config (`load_config(args.config)`)
   - Glob all `*.pdf` files under `args.dir`
   - For each PDF:
     - Print `Ingesting: <filename>`
     - Call `extract_pages(pdf_path)`; on `RuntimeError` print a warning and `continue`
     - Call `chunk_pages(pages, source_file=filename, chunk_size=..., chunk_overlap=...)`
     - Accumulate chunks across all files
   - After all files, call `build_collection(all_chunks, ...)`
   - Print final summary: `Done. Ingested N files, M chunks total.`
   - Exit 0 on success
