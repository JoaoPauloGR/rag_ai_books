import chromadb
import ollama


def build_collection(
    chunks: list[dict],
    embedding_model: str,
    chroma_path: str,
    collection_name: str,
) -> None:
    client = chromadb.PersistentClient(path=chroma_path)

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(collection_name)

    total = len(chunks)
    batch_size = 100

    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start : batch_start + batch_size]

        ids = [str(batch_start + i) for i in range(len(batch))]
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
