import pytest
from unittest.mock import MagicMock, patch
from src.extract import extract_pages


def _make_fitz_doc(texts: list[str]):
    pages = []
    for text in texts:
        page = MagicMock()
        page.get_text.return_value = text
        pages.append(page)
    doc = MagicMock()
    doc.__iter__ = MagicMock(return_value=iter(pages))
    doc.__len__ = MagicMock(return_value=len(pages))
    return doc


def test_returns_one_dict_per_page():
    doc = _make_fitz_doc(["Page one text.", "Page two text."])
    with patch("src.extract.fitz.open", return_value=doc):
        result = extract_pages("dummy.pdf")
    assert len(result) == 2


def test_page_numbers_are_one_indexed():
    doc = _make_fitz_doc(["first", "second", "third"])
    with patch("src.extract.fitz.open", return_value=doc):
        result = extract_pages("dummy.pdf")
    assert [r["page_number"] for r in result] == [1, 2, 3]


def test_text_is_preserved():
    doc = _make_fitz_doc(["hello world"])
    with patch("src.extract.fitz.open", return_value=doc):
        result = extract_pages("dummy.pdf")
    assert result[0]["text"] == "hello world"


def test_fitz_error_raises_runtime_error():
    with patch("src.extract.fitz.open", side_effect=Exception("corrupt")):
        with pytest.raises(RuntimeError, match="corrupt"):
            extract_pages("bad.pdf")
