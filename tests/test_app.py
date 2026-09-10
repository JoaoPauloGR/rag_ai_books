# Group 5 — Gradio app (src/app.py)
# No server is launched; on_message runs with retrieve/generate/ollama mocked.
import sqlite3
from contextlib import closing
from unittest.mock import patch

import gradio as gr
import pytest

from src.app import build_demo, on_feedback, on_message

CFG = {
    "embedding_model": "nomic-embed-text",
    "generation_model": "llama3.2:3b",
    "chroma_path": "data/chroma_db",
    "collection_name": "books",
    "top_k": 5,
    "chunk_size": 512,
    "chunk_overlap": 64,
    "rewrite_followups": True,
}

CHUNKS = [
    {"chunk_id": "book.pdf::0", "text": "chunk zero", "source_file": "book.pdf", "page_number": 1},
    {"chunk_id": "book.pdf::1", "text": "chunk one", "source_file": "book.pdf", "page_number": 1},
    {"chunk_id": "other.pdf::3", "text": "chunk two", "source_file": "other.pdf", "page_number": 7},
]


def _cfg(tmp_path, **overrides):
    return {**CFG, "feedback_db_path": str(tmp_path / "feedback.db"), **overrides}


def _rows(db_path, query, params=()):
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(query, params).fetchall()]


def test_build_demo_returns_blocks():
    assert isinstance(build_demo(CFG), gr.Blocks)


@patch("src.app.generate_answer", return_value="A grounded answer.")
@patch("src.app.retrieve_chunks", return_value=CHUNKS)
def test_on_message_appends_turn_and_writes_row(mock_retrieve, mock_generate, tmp_path):
    cfg = _cfg(tmp_path, rewrite_followups=False)

    chat, cleared, checkbox, last_turn = on_message(
        cfg, "What is a transformer?", [], "sess-1", None
    )

    assert cleared == ""
    assert chat[-2] == {"role": "user", "content": "What is a transformer?"}
    assert chat[-1]["role"] == "assistant"
    assert "A grounded answer." in chat[-1]["content"]
    assert "[Source: book.pdf, p. 1]" in chat[-1]["content"]

    # retrieval question == raw question when rewriting is off
    mock_retrieve.assert_called_once()
    assert mock_retrieve.call_args[0][0] == "What is a transformer?"

    assert last_turn["turn_id"] == 1
    assert last_turn["citations"] == ["[Source: book.pdf, p. 1]", "[Source: other.pdf, p. 7]"]
    assert last_turn["ranks_by_citation"]["[Source: book.pdf, p. 1]"] == [0, 1]
    assert checkbox.choices == [
        ("[Source: book.pdf, p. 1]", "[Source: book.pdf, p. 1]"),
        ("[Source: other.pdf, p. 7]", "[Source: other.pdf, p. 7]"),
    ]

    turns = _rows(cfg["feedback_db_path"], "SELECT * FROM turns")
    assert len(turns) == 1
    assert turns[0]["raw_question"] == "What is a transformer?"
    assert turns[0]["retrieval_question"] == "What is a transformer?"
    assert len(_rows(cfg["feedback_db_path"], "SELECT * FROM turn_sources")) == 3


@patch("src.app.rewrite_followup", return_value="Standalone rewritten question?")
@patch("src.app.generate_answer", return_value="Answer.")
@patch("src.app.retrieve_chunks", return_value=CHUNKS)
def test_on_message_rewrites_followup_when_history_present(
    mock_retrieve, mock_generate, mock_rewrite, tmp_path
):
    cfg = _cfg(tmp_path)
    history = [
        {"role": "user", "content": "Tell me about transformers."},
        {"role": "assistant", "content": "They use self-attention."},
    ]

    _, _, _, last_turn = on_message(cfg, "And the downsides?", history, "sess-1", None)

    mock_rewrite.assert_called_once()
    assert mock_retrieve.call_args[0][0] == "Standalone rewritten question?"
    turn = _rows(cfg["feedback_db_path"], "SELECT * FROM turns")[0]
    assert turn["raw_question"] == "And the downsides?"
    assert turn["retrieval_question"] == "Standalone rewritten question?"


@patch("src.app.retrieve_chunks")
def test_on_message_handles_missing_collection(mock_retrieve, tmp_path):
    import chromadb.errors

    mock_retrieve.side_effect = chromadb.errors.NotFoundError("nope")
    cfg = _cfg(tmp_path, rewrite_followups=False)

    chat, cleared, checkbox, last_turn = on_message(cfg, "hello?", [], "sess-1", None)

    assert last_turn is None
    assert "ingestion pipeline" in chat[-1]["content"]
    assert _rows(cfg["feedback_db_path"], "SELECT name FROM sqlite_master") == [] or \
        _rows(cfg["feedback_db_path"], "SELECT COUNT(*) AS n FROM turns")[0]["n"] == 0


def test_on_message_ignores_blank_input(tmp_path):
    cfg = _cfg(tmp_path)
    chat, cleared, checkbox, last_turn = on_message(cfg, "   ", [], "sess-1", None)
    assert chat == []
    assert last_turn is None


@patch("src.app.generate_answer", return_value="Answer.")
@patch("src.app.retrieve_chunks", return_value=CHUNKS)
def test_on_feedback_records_and_maps_ranks(mock_retrieve, mock_generate, tmp_path):
    cfg = _cfg(tmp_path, rewrite_followups=False)
    *_, last_turn = on_message(cfg, "q?", [], "sess-1", None)

    comment_out, rating_out = on_feedback(
        cfg, "👍", ["[Source: book.pdf, p. 1]"], "helpful stuff", last_turn
    )
    assert (comment_out, rating_out) == ("", None)

    fb = _rows(cfg["feedback_db_path"], "SELECT * FROM feedback")
    assert len(fb) == 1
    assert fb[0]["rating"] == "up"
    assert fb[0]["comment"] == "helpful stuff"

    helpful = {
        r["rank"]: r["helpful"]
        for r in _rows(cfg["feedback_db_path"], "SELECT rank, helpful FROM turn_sources")
    }
    # book.pdf p.1 covers ranks 0 and 1 (ticked -> 1); other.pdf p.7 cited-not-ticked -> 0
    assert helpful == {0: 1, 1: 1, 2: 0}


def test_on_feedback_noop_without_turn(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.warns(UserWarning):
        out = on_feedback(cfg, "👍", [], "", None)
    assert out == ("", "👍")
