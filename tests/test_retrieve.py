# Group 2 — Retrieval (src/retrieve.py)
# Ollama and ChromaDB are fully mocked; no running services required.
from unittest.mock import MagicMock, patch


_CHROMA_RESULT = {
    "documents": [["First chunk.", "Second chunk."]],
    "metadatas": [
        [
            {"source_file": "book_a.pdf", "page_number": 3},
            {"source_file": "book_b.pdf", "page_number": 11},
        ]
    ],
}


def _make_client(chroma_result):
    col = MagicMock()
    col.query.return_value = chroma_result
    client = MagicMock()
    client.get_collection.return_value = col
    return client, col


@patch("src.retrieve.chromadb.PersistentClient")
@patch("src.retrieve.ollama.embeddings")
def test_returns_list_of_chunk_dicts(mock_embed, mock_client_cls):
    mock_embed.return_value = {"embedding": [0.1, 0.2]}
    client, _ = _make_client(_CHROMA_RESULT)
    mock_client_cls.return_value = client

    from src.retrieve import retrieve_chunks
    result = retrieve_chunks("What is attention?", "nomic-embed-text", "data/chroma_db", "books", top_k=2)

    assert len(result) == 2
    assert result[0] == {"text": "First chunk.", "source_file": "book_a.pdf", "page_number": 3}
    assert result[1] == {"text": "Second chunk.", "source_file": "book_b.pdf", "page_number": 11}


@patch("src.retrieve.chromadb.PersistentClient")
@patch("src.retrieve.ollama.embeddings")
def test_embeds_question_with_correct_model(mock_embed, mock_client_cls):
    mock_embed.return_value = {"embedding": [0.1]}
    client, _ = _make_client({"documents": [[]], "metadatas": [[]]})
    mock_client_cls.return_value = client

    from src.retrieve import retrieve_chunks
    retrieve_chunks("test question", "nomic-embed-text", "data/chroma_db", "books", top_k=3)

    mock_embed.assert_called_once_with(model="nomic-embed-text", prompt="test question")


@patch("src.retrieve.chromadb.PersistentClient")
@patch("src.retrieve.ollama.embeddings")
def test_queries_collection_with_top_k(mock_embed, mock_client_cls):
    mock_embed.return_value = {"embedding": [0.5]}
    client, col = _make_client({"documents": [[]], "metadatas": [[]]})
    mock_client_cls.return_value = client

    from src.retrieve import retrieve_chunks
    retrieve_chunks("test question", "nomic-embed-text", "data/chroma_db", "books", top_k=7)

    col.query.assert_called_once()
    assert col.query.call_args[1]["n_results"] == 7


@patch("src.retrieve.chromadb.PersistentClient")
@patch("src.retrieve.ollama.embeddings")
def test_empty_result_returns_empty_list(mock_embed, mock_client_cls):
    mock_embed.return_value = {"embedding": [0.1]}
    client, _ = _make_client({"documents": [[]], "metadatas": [[]]})
    mock_client_cls.return_value = client

    from src.retrieve import retrieve_chunks
    result = retrieve_chunks("anything", "nomic-embed-text", "data/chroma_db", "books", top_k=5)

    assert result == []
