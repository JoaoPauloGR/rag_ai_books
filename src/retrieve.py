import chromadb
import ollama


def retrieve_chunks(
    question: str,
    embedding_model: str,
    chroma_path: str,
    collection_name: str,
    top_k: int,
) -> list[dict]:
    embedding = ollama.embeddings(model=embedding_model, prompt=question)["embedding"]

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(collection_name)
    result = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas"],
    )

    return [
        {
            "text": doc,
            "source_file": meta["source_file"],
            "page_number": meta["page_number"],
            "chunk_id": chunk_id,
        }
        for doc, meta, chunk_id in zip(
            result["documents"][0], result["metadatas"][0], result["ids"][0]
        )
    ]
