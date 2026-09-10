# Group 4 — Feedback persistence (src/feedback.py)
# Runs against a throwaway tmp_path SQLite file; no external services.
import sqlite3
from contextlib import closing

import pytest

from src.feedback import init_db, record_feedback, record_turn

CFG = {
    "embedding_model": "nomic-embed-text",
    "generation_model": "llama3.2:3b",
    "top_k": 5,
    "chunk_size": 512,
    "chunk_overlap": 64,
}

# ranks 0 and 2 share (book.pdf, 1) -> only rank 0 is cited. Ranks 1 and 3 are
# distinct pairs, so cited = [1, 1, 0, 1].
CHUNKS = [
    {"chunk_id": "book.pdf::0", "text": "chunk zero", "source_file": "book.pdf", "page_number": 1},
    {"chunk_id": "book.pdf::7", "text": "chunk one", "source_file": "book.pdf", "page_number": 2},
    {"chunk_id": "book.pdf::1", "text": "chunk two", "source_file": "book.pdf", "page_number": 1},
    {"chunk_id": "other.pdf::5", "text": "chunk three", "source_file": "other.pdf", "page_number": 9},
]


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "nested" / "feedback.db")


def _rows(db_path, query, params=()):
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(query, params).fetchall()]


def _make_turn(db_path):
    return record_turn(
        db_path,
        session_id="s1",
        raw_question="what about downsides?",
        retrieval_question="what are the downsides of transformers?",
        answer="They are compute-hungry.",
        chunks=CHUNKS,
        cfg=CFG,
    )


def test_init_db_is_idempotent_and_creates_tables(db_path):
    init_db(db_path)
    init_db(db_path)  # second call must not raise

    tables = {
        r["name"]
        for r in _rows(db_path, "SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {"turns", "turn_sources", "feedback"} <= tables


def test_record_turn_returns_int_and_writes_one_source_row_per_chunk(db_path):
    turn_id = _make_turn(db_path)
    assert isinstance(turn_id, int)

    sources = _rows(
        db_path,
        "SELECT * FROM turn_sources WHERE turn_id = ? ORDER BY rank",
        (turn_id,),
    )
    assert len(sources) == len(CHUNKS)
    assert [s["rank"] for s in sources] == [0, 1, 2, 3]
    assert [s["chunk_text"] for s in sources] == [c["text"] for c in CHUNKS]
    # (book.pdf, 1) first seen at rank 0; rank 2 repeats it -> not cited.
    assert [s["cited"] for s in sources] == [1, 1, 0, 1]
    assert all(s["helpful"] is None for s in sources)


def test_record_turn_snapshots_retrieval_params_from_cfg(db_path):
    turn_id = _make_turn(db_path)
    turn = _rows(db_path, "SELECT * FROM turns WHERE id = ?", (turn_id,))[0]
    assert turn["top_k"] == 5
    assert turn["chunk_size"] == 512
    assert turn["chunk_overlap"] == 64
    assert turn["embedding_model"] == "nomic-embed-text"
    assert turn["generation_model"] == "llama3.2:3b"
    assert turn["raw_question"] == "what about downsides?"
    assert turn["retrieval_question"] == "what are the downsides of transformers?"


def test_record_feedback_writes_one_row_and_sets_helpful_flags(db_path):
    turn_id = _make_turn(db_path)
    record_feedback(
        db_path,
        turn_id=turn_id,
        rating="up",
        comment="solid",
        helpful_ranks=[0],
    )

    fb = _rows(db_path, "SELECT * FROM feedback WHERE turn_id = ?", (turn_id,))
    assert len(fb) == 1
    assert fb[0]["rating"] == "up"
    assert fb[0]["comment"] == "solid"

    helpful = {
        s["rank"]: s["helpful"]
        for s in _rows(
            db_path,
            "SELECT rank, helpful FROM turn_sources WHERE turn_id = ?",
            (turn_id,),
        )
    }
    assert helpful == {0: 1, 1: 0, 2: None, 3: 0}


def test_record_feedback_resubmit_replaces_row_and_updates_flags(db_path):
    turn_id = _make_turn(db_path)
    record_feedback(db_path, turn_id=turn_id, rating="up", comment="first", helpful_ranks=[0, 1])
    record_feedback(db_path, turn_id=turn_id, rating="down", comment=None, helpful_ranks=[1])

    fb = _rows(db_path, "SELECT * FROM feedback WHERE turn_id = ?", (turn_id,))
    assert len(fb) == 1
    assert fb[0]["rating"] == "down"
    assert fb[0]["comment"] is None

    helpful = {
        s["rank"]: s["helpful"]
        for s in _rows(
            db_path,
            "SELECT rank, helpful FROM turn_sources WHERE turn_id = ?",
            (turn_id,),
        )
    }
    # rank 0 ticked-then-unticked but still cited -> 0; rank 1 ticked -> 1; rank 2 uncited -> NULL.
    assert helpful == {0: 0, 1: 1, 2: None, 3: 0}


def test_public_functions_do_not_require_manual_init(db_path):
    # record_turn is the first call touching the db; it must create the schema itself.
    turn_id = _make_turn(db_path)
    assert _rows(db_path, "SELECT COUNT(*) AS n FROM turns")[0]["n"] == 1
    record_feedback(db_path, turn_id=turn_id, rating="up", comment=None, helpful_ranks=[])
    assert _rows(db_path, "SELECT COUNT(*) AS n FROM feedback")[0]["n"] == 1
