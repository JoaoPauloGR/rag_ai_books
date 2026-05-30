# Group 1 — Config extension (top_k required key)
import yaml
import pytest
from src.config import load_config


_BASE = {
    "embedding_model": "nomic-embed-text",
    "generation_model": "llama3.2",
    "chroma_path": "data/chroma_db",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "collection_name": "books",
    "top_k": 5,
}


def _write(tmp_path, cfg):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))
    return str(p)


def test_missing_top_k_raises(tmp_path):
    cfg = {k: v for k, v in _BASE.items() if k != "top_k"}
    with pytest.raises(ValueError, match="top_k"):
        load_config(_write(tmp_path, cfg))


def test_valid_config_with_top_k_loads(tmp_path):
    result = load_config(_write(tmp_path, _BASE))
    assert result["top_k"] == 5
