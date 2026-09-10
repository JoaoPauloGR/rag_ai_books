from unittest.mock import MagicMock, call, patch
from src.store import build_collection

CHUNKS = [
    {"text": f"chunk {i}", "source_file": "book.pdf", "page_number": 1, "chunk_index": i}
    for i in range(5)
]

EXPECTED_IDS = [f"book.pdf::{i}" for i in range(5)]


def _make_mocks(existing_ids=None):
    mock_collection = MagicMock()
    mock_collection.get.return_value = {"ids": list(existing_ids or [])}
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    return mock_client, mock_collection


def _embed_patch():
    mock_embed = MagicMock()
    mock_embed.return_value.embeddings = [[0.1]]
    return patch("src.store.ollama.embed", mock_embed)


def test_uses_get_or_create_collection():
    mock_client, _ = _make_mocks()
    with patch("src.store.chromadb.PersistentClient", return_value=mock_client), _embed_patch():
        build_collection(CHUNKS, "nomic-embed-text", "data/chroma", "books")
    mock_client.get_or_create_collection.assert_called_once_with("books")
    mock_client.create_collection.assert_not_called()


def test_default_run_does_not_delete_collection():
    mock_client, _ = _make_mocks()
    with patch("src.store.chromadb.PersistentClient", return_value=mock_client), _embed_patch():
        build_collection(CHUNKS, "nomic-embed-text", "data/chroma", "books")
    mock_client.delete_collection.assert_not_called()


def test_force_deletes_collection_once():
    mock_client, _ = _make_mocks()
    with patch("src.store.chromadb.PersistentClient", return_value=mock_client), _embed_patch():
        build_collection(CHUNKS, "nomic-embed-text", "data/chroma", "books", force=True)
    mock_client.delete_collection.assert_called_once_with("books")


def test_force_delete_failure_does_not_abort():
    mock_client, _ = _make_mocks()
    mock_client.delete_collection.side_effect = Exception("not found")
    with patch("src.store.chromadb.PersistentClient", return_value=mock_client), _embed_patch():
        result = build_collection(CHUNKS, "nomic-embed-text", "data/chroma", "books", force=True)
    assert result == {"added": 5, "skipped": 0}


def test_ids_are_stable_source_index_strings():
    mock_client, mock_collection = _make_mocks()
    with patch("src.store.chromadb.PersistentClient", return_value=mock_client), _embed_patch():
        build_collection(CHUNKS, "nomic-embed-text", "data/chroma", "books")
    ids_passed = mock_collection.add.call_args[1]["ids"]
    assert ids_passed == EXPECTED_IDS


def test_metadata_keys_are_correct():
    mock_client, mock_collection = _make_mocks()
    with patch("src.store.chromadb.PersistentClient", return_value=mock_client), _embed_patch():
        build_collection(CHUNKS, "nomic-embed-text", "data/chroma", "books")
    metadatas = mock_collection.add.call_args[1]["metadatas"]
    assert set(metadatas[0].keys()) == {"source_file", "page_number", "chunk_index"}


def test_batching_splits_into_correct_number_of_calls():
    chunks_150 = [
        {"text": f"chunk {i}", "source_file": "book.pdf", "page_number": 1, "chunk_index": i}
        for i in range(150)
    ]
    mock_client, mock_collection = _make_mocks()
    with patch("src.store.chromadb.PersistentClient", return_value=mock_client), _embed_patch():
        build_collection(chunks_150, "nomic-embed-text", "data/chroma", "books")
    assert mock_collection.add.call_count == 2


def test_empty_chunks_makes_no_add_calls():
    mock_client, mock_collection = _make_mocks()
    with patch("src.store.chromadb.PersistentClient", return_value=mock_client), _embed_patch():
        result = build_collection([], "nomic-embed-text", "data/chroma", "books")
    mock_collection.add.assert_not_called()
    assert result == {"added": 0, "skipped": 0}


def test_rerun_with_all_ids_present_makes_no_add_call():
    mock_client, mock_collection = _make_mocks(existing_ids=EXPECTED_IDS)
    with patch("src.store.chromadb.PersistentClient", return_value=mock_client), _embed_patch():
        result = build_collection(CHUNKS, "nomic-embed-text", "data/chroma", "books")
    mock_collection.add.assert_not_called()
    assert result == {"added": 0, "skipped": 5}


def test_partial_overlap_adds_only_new_chunks():
    mock_client, mock_collection = _make_mocks(existing_ids=["book.pdf::0", "book.pdf::1"])
    with patch("src.store.chromadb.PersistentClient", return_value=mock_client), _embed_patch():
        result = build_collection(CHUNKS, "nomic-embed-text", "data/chroma", "books")
    ids_passed = mock_collection.add.call_args[1]["ids"]
    assert ids_passed == ["book.pdf::2", "book.pdf::3", "book.pdf::4"]
    assert result == {"added": 3, "skipped": 2}
