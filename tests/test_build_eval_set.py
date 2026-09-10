# Group 2 — Candidate drafter (src/build_eval_set.py)
# chromadb and ollama are fully mocked; no running services required.
import json
import re
from unittest.mock import MagicMock, patch


def _fake_collection_data(count=10):
    ids = [f"book_{i}.pdf::{i}" for i in range(count)]
    docs = [f"Passage text for chunk {i}." for i in range(count)]
    metas = [
        {"source_file": f"book_{i}.pdf", "page_number": i + 1, "chunk_index": i}
        for i in range(count)
    ]
    return ids, docs, metas


def _make_client(ids, docs, metas):
    """Mock a collection whose .get() mirrors real chroma: ids-only when
    include=[], and an id-filtered slice when ids=[...] is passed."""
    by_id = {i: (d, m) for i, d, m in zip(ids, docs, metas)}

    def _get(ids=None, include=None, **kw):
        if ids is None:
            return {"ids": list(by_id), "documents": [], "metadatas": []}
        sel = [i for i in ids if i in by_id]
        return {
            "ids": sel,
            "documents": [by_id[i][0] for i in sel],
            "metadatas": [by_id[i][1] for i in sel],
        }

    col = MagicMock()
    col.get.side_effect = _get
    client = MagicMock()
    client.get_collection.return_value = col
    return client


def _good_reply(model, messages):
    return {
        "message": {
            "content": json.dumps(
                {"question": "What is the thing?", "answer_keywords": ["alpha", "beta"]}
            )
        }
    }


def _bad_for_chunk_3(model, messages):
    if "Passage text for chunk 3." in messages[0]["content"]:
        return {"message": {"content": "not json"}}
    return _good_reply(model, messages)


def _run(tmp_path, argv_extra, chat_side_effect, count=10):
    ids, docs, metas = _fake_collection_data(count)
    client = _make_client(ids, docs, metas)
    out = tmp_path / "draft.json"
    argv = ["build_eval_set.py", "--config", "config.yaml", "--out", str(out), *argv_extra]
    fake_cfg = {
        "chroma_path": "p",
        "collection_name": "books",
        "generation_model": "llama3.2:3b",
    }
    with patch("src.build_eval_set.load_config", return_value=fake_cfg), patch(
        "src.build_eval_set.chromadb.PersistentClient", return_value=client
    ), patch(
        "src.build_eval_set.ollama.chat", side_effect=chat_side_effect
    ) as mock_chat, patch(
        "sys.argv", argv
    ), patch(
        "builtins.print"
    ) as mock_print:
        from src import build_eval_set

        try:
            build_eval_set.main()
        except SystemExit as e:
            assert e.code in (None, 0), e.code

    data = json.loads(out.read_text(encoding="utf-8"))
    printed = [str(c.args[0]) for c in mock_print.call_args_list if c.args]
    return data, printed, mock_chat, {i: m for i, m in zip(ids, metas)}


def _chunk_id_from_notes(entry):
    m = re.search(r"drafted from chunk '(.+)'", entry["notes"])
    assert m, entry["notes"]
    return m.group(1)


def test_n3_produces_3_draft_entries(tmp_path):
    data, _, _, _ = _run(tmp_path, ["--n", "3"], _good_reply)
    assert isinstance(data["entries"], list)
    assert len(data["entries"]) == 3


def test_entries_have_expected_shape_and_metadata(tmp_path):
    data, _, _, meta_by_id = _run(tmp_path, ["--n", "3"], _good_reply)
    for pos, entry in enumerate(data["entries"], start=1):
        assert entry["verified"] is False
        assert entry["id"] == f"q{pos:02d}"
        assert isinstance(entry["question"], str) and entry["question"].strip()
        assert isinstance(entry["answer_keywords"], list) and entry["answer_keywords"]

        meta = meta_by_id[_chunk_id_from_notes(entry)]
        assert entry["expected_source_file"] == meta["source_file"]
        assert entry["expected_pages"] == [meta["page_number"]]


def test_unparseable_reply_is_skipped_run_still_succeeds(tmp_path):
    data, printed, _, _ = _run(tmp_path, ["--n", "10"], _bad_for_chunk_3)

    assert len(data["entries"]) == 9
    assert all(_chunk_id_from_notes(e) != "book_3.pdf::3" for e in data["entries"])
    assert any("Warning" in line for line in printed)
    assert any("drafted 9, skipped 1" in line for line in printed)


def test_same_seed_samples_same_chunks(tmp_path):
    data1, _, _, _ = _run(tmp_path, ["--n", "4", "--seed", "7"], _good_reply)
    data2, _, _, _ = _run(tmp_path, ["--n", "4", "--seed", "7"], _good_reply)

    ids1 = [_chunk_id_from_notes(e) for e in data1["entries"]]
    ids2 = [_chunk_id_from_notes(e) for e in data2["entries"]]
    assert ids1 == ids2
    assert len(set(ids1)) == 4
