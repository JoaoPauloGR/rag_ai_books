# Group 1 — eval/eval_set.json is present and well-formed
import json
from pathlib import Path

_EVAL_SET = Path(__file__).resolve().parent.parent / "eval" / "eval_set.json"


def test_eval_set_parses_as_json():
    data = json.loads(_EVAL_SET.read_text())
    assert isinstance(data, dict)


def test_eval_set_has_entries_list():
    data = json.loads(_EVAL_SET.read_text())
    assert isinstance(data["entries"], list)
