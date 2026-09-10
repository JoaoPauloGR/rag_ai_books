import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  TEXT NOT NULL,
    session_id          TEXT NOT NULL,
    raw_question        TEXT NOT NULL,
    retrieval_question  TEXT NOT NULL,
    answer              TEXT NOT NULL,
    embedding_model     TEXT NOT NULL,
    generation_model    TEXT NOT NULL,
    top_k               INTEGER NOT NULL,
    chunk_size          INTEGER NOT NULL,
    chunk_overlap       INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS turn_sources (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id      INTEGER NOT NULL REFERENCES turns(id),
    rank         INTEGER NOT NULL,
    chunk_id     TEXT NOT NULL,
    source_file  TEXT NOT NULL,
    page_number  INTEGER NOT NULL,
    chunk_text   TEXT NOT NULL,
    cited        INTEGER NOT NULL,
    helpful      INTEGER
);

CREATE TABLE IF NOT EXISTS feedback (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id   INTEGER NOT NULL UNIQUE REFERENCES turns(id),
    ts        TEXT NOT NULL,
    rating    TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    comment   TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chunk_id(chunk: dict, rank: int) -> str:
    if chunk.get("chunk_id"):
        return chunk["chunk_id"]
    return f'{chunk["source_file"]}::{chunk.get("chunk_index", rank)}'


def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.executescript(_SCHEMA)
        conn.commit()


def record_turn(
    db_path: str,
    *,
    session_id: str,
    raw_question: str,
    retrieval_question: str,
    answer: str,
    chunks: list[dict],
    cfg: dict,
) -> int:
    init_db(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO turns (
                ts, session_id, raw_question, retrieval_question, answer,
                embedding_model, generation_model, top_k, chunk_size, chunk_overlap
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                session_id,
                raw_question,
                retrieval_question,
                answer,
                cfg["embedding_model"],
                cfg["generation_model"],
                cfg["top_k"],
                cfg["chunk_size"],
                cfg["chunk_overlap"],
            ),
        )
        turn_id = cur.lastrowid

        seen: set[tuple[str, int]] = set()
        for rank, chunk in enumerate(chunks):
            pair = (chunk["source_file"], chunk["page_number"])
            cited = 1 if pair not in seen else 0
            seen.add(pair)
            cur.execute(
                """
                INSERT INTO turn_sources (
                    turn_id, rank, chunk_id, source_file, page_number,
                    chunk_text, cited, helpful
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    turn_id,
                    rank,
                    _chunk_id(chunk, rank),
                    chunk["source_file"],
                    chunk["page_number"],
                    chunk["text"],
                    cited,
                ),
            )
        conn.commit()
        return turn_id


def record_feedback(
    db_path: str,
    *,
    turn_id: int,
    rating: str,
    comment: str | None,
    helpful_ranks: list[int],
) -> None:
    init_db(db_path)
    ranks = list(helpful_ranks)
    with closing(sqlite3.connect(db_path)) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO feedback (turn_id, ts, rating, comment)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(turn_id) DO UPDATE SET
                ts = excluded.ts,
                rating = excluded.rating,
                comment = excluded.comment
            """,
            (turn_id, _now(), rating, comment),
        )

        if ranks:
            placeholders = ",".join("?" * len(ranks))
            cur.execute(
                f"UPDATE turn_sources SET helpful = 1 "
                f"WHERE turn_id = ? AND rank IN ({placeholders})",
                (turn_id, *ranks),
            )
            cur.execute(
                f"UPDATE turn_sources SET helpful = 0 "
                f"WHERE turn_id = ? AND cited = 1 AND rank NOT IN ({placeholders})",
                (turn_id, *ranks),
            )
        else:
            cur.execute(
                "UPDATE turn_sources SET helpful = 0 WHERE turn_id = ? AND cited = 1",
                (turn_id,),
            )
        conn.commit()
