# Group 4 — Wire query (src/query.py)
# src.retrieve and src.generate are stubbed via sys.modules so this file can be
# collected before those modules are implemented. Once implemented, the real modules
# take precedence and the stubs are ignored.
import sys
import yaml
import importlib
import pytest
from unittest.mock import MagicMock, patch


_CHUNKS_DEDUP = [
    {"text": "A.", "source_file": "attn.pdf", "page_number": 5},
    {"text": "B.", "source_file": "attn.pdf", "page_number": 5},  # same page — should be collapsed
    {"text": "C.", "source_file": "bert.pdf", "page_number": 12},
]

_CHUNKS_UNIQUE = [
    {"text": "A.", "source_file": "attn.pdf", "page_number": 5},
    {"text": "B.", "source_file": "bert.pdf", "page_number": 12},
]


def _write_config(tmp_path):
    cfg = {
        "embedding_model": "nomic-embed-text",
        "generation_model": "llama3.2",
        "chroma_path": "data/chroma_db",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "collection_name": "books",
        "top_k": 3,
        "feedback_db_path": "data/feedback.db",
        "rewrite_followups": True,
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))
    return str(p)


def _run(tmp_path, chunks, answer, capsys):
    config = _write_config(tmp_path)
    import src.query as qmod
    importlib.reload(qmod)
    with patch("src.query.retrieve_chunks", return_value=chunks), \
         patch("src.query.generate_answer", return_value=answer), \
         patch("sys.argv", ["query.py", "--query", "test question", "--config", config]):
        with pytest.raises(SystemExit) as exc:
            qmod.main()
    assert exc.value.code == 0
    return capsys.readouterr().out


def test_answer_is_printed(tmp_path, capsys):
    out = _run(tmp_path, _CHUNKS_UNIQUE, "This is the answer.", capsys)
    assert "This is the answer." in out


def test_sources_are_deduplicated(tmp_path, capsys):
    out = _run(tmp_path, _CHUNKS_DEDUP, "Answer.", capsys)
    assert out.count("[Source: attn.pdf, p. 5]") == 1


def test_all_unique_sources_printed(tmp_path, capsys):
    out = _run(tmp_path, _CHUNKS_UNIQUE, "Answer.", capsys)
    assert "[Source: attn.pdf, p. 5]" in out
    assert "[Source: bert.pdf, p. 12]" in out


def test_source_citation_format(tmp_path, capsys):
    out = _run(tmp_path, _CHUNKS_UNIQUE, "Answer.", capsys)
    assert "[Source: attn.pdf, p. 5]" in out
