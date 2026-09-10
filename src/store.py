import chromadb
import ollama


def _chunk_id(chunk: dict) -> str:
    return f'{chunk["source_file"]}::{chunk["chunk_index"]}'


def build_collection(
    chunks: list[dict],
    embedding_model: str,
    chroma_path: str,
    collection_name: str,
    *,
    force: bool = False,
) -> dict:
    client = chromadb.PersistentClient(path=chroma_path)

    if force:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(collection_name)

    existing = set(collection.get(include=[])["ids"])

    new_chunks = [c for c in chunks if _chunk_id(c) not in existing]
    skipped = len(chunks) - len(new_chunks)

    total = len(new_chunks)
    batch_size = 100

    for batch_start in range(0, total, batch_size):
        batch = new_chunks[batch_start : batch_start + batch_size]

        ids = [_chunk_id(chunk) for chunk in batch]
        embeddings = [
            ollama.embed(model=embedding_model, input=chunk["text"]).embeddings[0]
            for chunk in batch
        ]
        documents = [chunk["text"] for chunk in batch]
        metadatas = [
            {
                "source_file": chunk["source_file"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in batch
        ]

        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        print(f"Stored {min(batch_start + batch_size, total)}/{total} chunks...")

    return {"added": len(new_chunks), "skipped": skipped}
