# Group 4 — Retrieval scoring (src/evaluate.py)
# retrieve_chunks is mocked; no running services required.
import json
import chromadb.errors
import pytest
from unittest.mock import patch

from src.evaluate import aggregate, load_eval_set, main, score_entry

_CFG = {
    "embedding_model": "nomic-embed-text",
    "generation_model": "llama3.2:3b",
    "chroma_path": "data/chroma_db",
    "collection_name": "books",
    "top_k": 5,
}


def _entry(id="q01", src="a.pdf", pages=(5,), kws=("x",), verified=True, question="Q?"):
    return {
        "id": id,
        "question": question,
        "expected_source_file": src,
        "expected_pages": list(pages),
        "answer_keywords": list(kws),
        "verified": verified,
    }


def _chunk(src, page, cid=None):
    return {
        "text": "t",
        "source_file": src,
        "page_number": page,
        "chunk_id": cid or f"{src}::{page}",
    }


# --- score_entry / aggregate arithmetic --------------------------------------

@patch("src.evaluate.retrieve_chunks")
def test_first_hit_at_rank_2(mock_retrieve):
    mock_retrieve.return_value = [
        _chunk("other.pdf", 1),
        _chunk("a.pdf", 5),
        _chunk("a.pdf", 5),
    ]
    res = score_entry(_entry(src="a.pdf", pages=[5]), _CFG, k=5)
    assert res["first_hit_rank"] == 2

    agg = aggregate([res])
    assert agg["MRR"] == 0.5
    assert agg["hit_rate@1"] == 0.0
    assert agg["hit_rate@3"] == 1.0
    assert agg["hit_rate@k"] == 1.0


@patch("src.evaluate.retrieve_chunks")
def test_no_page_match_anywhere_is_a_miss(mock_retrieve):
    mock_retrieve.return_value = [_chunk("a.pdf", 1), _chunk("a.pdf", 2), _chunk("a.pdf", 3)]
    res = score_entry(_entry(src="a.pdf", pages=[5]), _CFG, k=5)
    assert res["first_hit_rank"] is None

    agg = aggregate([res])
    assert agg["hit_rate@k"] == 0.0
    assert agg["MRR"] == 0.0


@patch("src.evaluate.retrieve_chunks")
def test_right_page_wrong_source_file_is_a_miss(mock_retrieve):
    mock_retrieve.return_value = [_chunk("b.pdf", 5)]  # page in expected_pages, wrong file
    res = score_entry(_entry(src="a.pdf", pages=[5]), _CFG, k=5)
    assert res["first_hit_rank"] is None


@patch("src.evaluate.retrieve_chunks")
def test_aggregate_mixes_hit_and_miss(mock_retrieve):
    mock_retrieve.side_effect = [
        [_chunk("a.pdf", 5)],            # rank 1
        [_chunk("a.pdf", 1)],            # miss
    ]
    r1 = score_entry(_entry(id="q01", src="a.pdf", pages=[5]), _CFG, k=5)
    r2 = score_entry(_entry(id="q02", src="a.pdf", pages=[5]), _CFG, k=5)
    agg = aggregate([r1, r2])
    assert agg["hit_rate@k"] == 0.5
    assert agg["hit_rate@1"] == 0.5
    assert agg["MRR"] == 0.5


@patch("src.evaluate.retrieve_chunks")
def test_retrieved_triples_recorded_for_dump(mock_retrieve):
    mock_retrieve.return_value = [_chunk("a.pdf", 5, cid="a.pdf::42")]
    res = score_entry(_entry(src="a.pdf", pages=[5]), _CFG, k=5)
    assert res["retrieved"] == [("a.pdf::42", "a.pdf", 5)]


# --- eval-set validation ---------------------------------------------------

def _write(tmp_path, entries):
    p = tmp_path / "eval_set.json"
    p.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    return str(p)


def test_validation_rejects_unverified_entry(tmp_path, capsys):
    path = _write(tmp_path, [_entry(id="q01", verified=False)])
    with pytest.raises(SystemExit) as exc:
        load_eval_set(path)
    assert exc.value.code == 2
    out = capsys.readouterr().out
    assert "q01" in out and "verified" in out


def test_validation_rejects_empty_answer_keywords(tmp_path, capsys):
    path = _write(tmp_path, [_entry(id="q07", kws=[])])
    with pytest.raises(SystemExit) as exc:
        load_eval_set(path)
    assert exc.value.code == 2
    out = capsys.readouterr().out
    assert "q07" in out and "answer_keywords" in out


def test_validation_collects_all_failures(tmp_path, capsys):
    path = _write(
        tmp_path,
        [
            _entry(id="q01", verified=False),
            _entry(id="q02", pages=[]),
            _entry(id="q03"),  # valid
        ],
    )
    with pytest.raises(SystemExit) as exc:
        load_eval_set(path)
    assert exc.value.code == 2
    out = capsys.readouterr().out
    assert "q01" in out and "q02" in out


def test_validation_passes_for_good_set(tmp_path):
    path = _write(tmp_path, [_entry(id="q01"), _entry(id="q02", src="b.pdf")])
    assert len(load_eval_set(path)) == 2


# --- main() --------------------------------------------------------------

@patch("src.evaluate.generate_answer", return_value="text")
@patch("src.evaluate.retrieve_chunks")
def test_main_prints_table_and_summary(mock_retrieve, mock_generate, tmp_path, capsys):
    mock_retrieve.side_effect = [
        [_chunk("a.pdf", 5)],                       # q01 -> rank 1
        [_chunk("x.pdf", 1), _chunk("b.pdf", 7)],   # q02 -> rank 2
    ]
    eval_path = _write(
        tmp_path,
        [
            _entry(id="q01", src="a.pdf", pages=[5]),
            _entry(id="q02", src="b.pdf", pages=[7]),
        ],
    )
    argv = ["evaluate.py", "--config", "config.yaml", "--eval-set", eval_path]
    with patch("src.evaluate.load_config", return_value=_CFG), patch("sys.argv", argv):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 0

    out = capsys.readouterr().out
    assert "q01" in out and "q02" in out
    assert "hit@k=1.00" in out
    assert "hit@1=0.50" in out
    assert "MRR=0.75" in out
    assert "k=5" in out


@patch("src.evaluate.generate_answer", return_value="text")
@patch("src.evaluate.retrieve_chunks")
def test_main_k_override(mock_retrieve, mock_generate, tmp_path):
    mock_retrieve.return_value = [_chunk("a.pdf", 5)]
    eval_path = _write(tmp_path, [_entry(id="q01", src="a.pdf", pages=[5])])
    argv = ["evaluate.py", "--eval-set", eval_path, "--k", "3"]
    with patch("src.evaluate.load_config", return_value=_CFG), patch("sys.argv", argv):
        with pytest.raises(SystemExit):
            main()
    assert mock_retrieve.call_args[0][4] == 3


@patch("src.evaluate.retrieve_chunks")
def test_main_missing_collection_exits_1(mock_retrieve, tmp_path, capsys):
    mock_retrieve.side_effect = chromadb.errors.NotFoundError("nope")
    eval_path = _write(tmp_path, [_entry(id="q01")])
    argv = ["evaluate.py", "--eval-set", eval_path]
    with patch("src.evaluate.load_config", return_value=_CFG), patch("sys.argv", argv):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1
    assert "not found" in capsys.readouterr().err


# --- answer keyword check (Group 5) ---------------------------------------

@patch("src.evaluate.generate_answer")
@patch("src.evaluate.retrieve_chunks")
def test_answer_pass_when_all_keywords_present_case_insensitive(mock_retrieve, mock_generate):
    mock_retrieve.return_value = [_chunk("a.pdf", 5)]
    mock_generate.return_value = "It solves TRAINING-serving skew and enables Reuse."
    res = score_entry(
        _entry(src="a.pdf", pages=[5], kws=["training-serving skew", "reuse"]),
        _CFG,
        k=5,
        answer_check=True,
    )
    assert res["answer_pass"] is True

    agg = aggregate([res], answer_checked=True)
    assert agg["answer_keyword_accuracy"] == 1.0


@patch("src.evaluate.generate_answer")
@patch("src.evaluate.retrieve_chunks")
def test_answer_fail_when_one_keyword_missing(mock_retrieve, mock_generate):
    mock_retrieve.return_value = [_chunk("a.pdf", 5)]
    mock_generate.return_value = "It solves training-serving skew."
    res = score_entry(
        _entry(src="a.pdf", pages=[5], kws=["training-serving skew", "reuse"]),
        _CFG,
        k=5,
        answer_check=True,
    )
    assert res["answer_pass"] is False

    agg = aggregate([res], answer_checked=True)
    assert agg["answer_keyword_accuracy"] == 0.0


def test_aggregate_omits_accuracy_key_by_default():
    res = {"id": "q01", "first_hit_rank": 1, "answer_pass": None, "retrieved": []}
    assert "answer_keyword_accuracy" not in aggregate([res])


@patch("src.evaluate.generate_answer")
@patch("src.evaluate.retrieve_chunks")
def test_score_entry_skips_generation_by_default(mock_retrieve, mock_generate):
    mock_retrieve.return_value = [_chunk("a.pdf", 5)]
    res = score_entry(_entry(src="a.pdf", pages=[5]), _CFG, k=5)
    mock_generate.assert_not_called()
    assert res["answer_pass"] is None


@patch("src.evaluate.generate_answer")
@patch("src.evaluate.retrieve_chunks")
def test_main_no_answer_check_skips_generation(mock_retrieve, mock_generate, tmp_path, capsys):
    mock_retrieve.return_value = [_chunk("a.pdf", 5)]
    eval_path = _write(tmp_path, [_entry(id="q01", src="a.pdf", pages=[5])])
    argv = ["evaluate.py", "--eval-set", eval_path, "--no-answer-check"]
    with patch("src.evaluate.load_config", return_value=_CFG), patch("sys.argv", argv):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 0
    mock_generate.assert_not_called()
    out = capsys.readouterr().out
    assert "ans_kw" not in out


@patch("src.evaluate.generate_answer")
@patch("src.evaluate.retrieve_chunks")
def test_main_answer_check_reports_accuracy(mock_retrieve, mock_generate, tmp_path, capsys):
    mock_retrieve.return_value = [_chunk("a.pdf", 5)]
    mock_generate.return_value = "training-serving skew"
    eval_path = _write(
        tmp_path,
        [_entry(id="q01", src="a.pdf", pages=[5], kws=["training-serving skew"])],
    )
    argv = ["evaluate.py", "--eval-set", eval_path]
    with patch("src.evaluate.load_config", return_value=_CFG), patch("sys.argv", argv):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 0
    mock_generate.assert_called_once()
    assert "ans_kw=1.00" in capsys.readouterr().out
