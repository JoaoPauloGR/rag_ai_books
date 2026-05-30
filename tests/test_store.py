from unittest.mock import MagicMock, call, patch
from src.store import build_collection

CHUNKS = [
    {"text": f"chunk {i}", "source_file": "book.pdf", "page_number": 1, "chunk_index": i}
    for i in range(5)
]


def _make_mocks():
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.create_collection.return_value = mock_collection
    return mock_client, mock_collection


def test_deletes_existing_collection_before_creating():
    mock_client, _ = _make_mocks()
    with patch("src.store.chromadb.PersistentClient", return_value=mock_client), \
         patch("src.store.ollama.embeddings", return_value={"embedding": [0.1]}):
        build_collection(CHUNKS, "nomic-embed-text", "data/chroma", "books")
    mock_client.delete_collection.assert_called_once_with("books")
    mock_client.create_collection.assert_called_once_with("books")


def test_delete_failure_does_not_abort():
    mock_client, _ = _make_mocks()
    mock_client.delete_collection.side_effect = Exception("not found")
    with patch("src.store.chromadb.PersistentClient", return_value=mock_client), \
         patch("src.store.ollama.embeddings", return_value={"embedding": [0.1]}):
        build_collection(CHUNKS, "nomic-embed-text", "data/chroma", "books")
    mock_client.create_collection.assert_called_once_with("books")


def test_ids_are_sequential_strings():
    mock_client, mock_collection = _make_mocks()
    with patch("src.store.chromadb.PersistentClient", return_value=mock_client), \
         patch("src.store.ollama.embeddings", return_value={"embedding": [0.1]}):
        build_collection(CHUNKS, "nomic-embed-text", "data/chroma", "books")
    ids_passed = mock_collection.add.call_args[1]["ids"]
    assert ids_passed == ["0", "1", "2", "3", "4"]


def test_metadata_keys_are_correct():
    mock_client, mock_collection = _make_mocks()
    with patch("src.store.chromadb.PersistentClient", return_value=mock_client), \
         patch("src.store.ollama.embeddings", return_value={"embedding": [0.1]}):
        build_collection(CHUNKS, "nomic-embed-text", "data/chroma", "books")
    metadatas = mock_collection.add.call_args[1]["metadatas"]
    assert set(metadatas[0].keys()) == {"source_file", "page_number", "chunk_index"}


def test_batching_splits_into_correct_number_of_calls():
    chunks_150 = [
        {"text": f"chunk {i}", "source_file": "book.pdf", "page_number": 1, "chunk_index": i}
        for i in range(150)
    ]
    mock_client, mock_collection = _make_mocks()
    with patch("src.store.chromadb.PersistentClient", return_value=mock_client), \
         patch("src.store.ollama.embeddings", return_value={"embedding": [0.1]}):
        build_collection(chunks_150, "nomic-embed-text", "data/chroma", "books")
    assert mock_collection.add.call_count == 2


def test_empty_chunks_makes_no_add_calls():
    mock_client, mock_collection = _make_mocks()
    with patch("src.store.chromadb.PersistentClient", return_value=mock_client), \
         patch("src.store.ollama.embeddings", return_value={"embedding": [0.1]}):
        build_collection([], "nomic-embed-text", "data/chroma", "books")
    mock_collection.add.assert_not_called()
