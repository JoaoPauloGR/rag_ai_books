import pytest
import yaml
from src.config import load_config


def _write_config(tmp_path, data: dict) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data))
    return str(p)


VALID = {
    "embedding_model": "nomic-embed-text",
    "generation_model": "llama3.2:3b",
    "chroma_path": "data/chroma_db",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "collection_name": "books",
}


def test_valid_config_loads(tmp_path):
    cfg = load_config(_write_config(tmp_path, VALID))
    assert cfg["collection_name"] == "books"
    assert cfg["chunk_size"] == 512


def test_missing_collection_name_raises(tmp_path):
    data = {k: v for k, v in VALID.items() if k != "collection_name"}
    with pytest.raises(ValueError, match="collection_name"):
        load_config(_write_config(tmp_path, data))


def test_missing_other_key_raises(tmp_path):
    data = {k: v for k, v in VALID.items() if k != "chunk_size"}
    with pytest.raises(ValueError, match="chunk_size"):
        load_config(_write_config(tmp_path, data))
