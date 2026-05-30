import sys
from unittest.mock import MagicMock, patch, call
from pathlib import Path


def _run_main(monkeypatch, tmp_path, pdf_files=None, extra_argv=None):
    """Helper: patch filesystem + pipeline, run main(), return captured stdout."""
    argv = ["ingest.py", "--dir", str(tmp_path), "--config", "config.yaml"]
    if extra_argv:
        argv += extra_argv
    monkeypatch.setattr(sys, "argv", argv)

    fake_cfg = {
        "embedding_model": "nomic-embed-text",
        "chroma_path": "data/chroma_db",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "collection_name": "books",
    }

    pdf_files = pdf_files or ["book_a.pdf"]
    found_paths = [tmp_path / f for f in pdf_files]

    with patch("src.ingest.load_config", return_value=fake_cfg), \
         patch("src.ingest.Path.glob", return_value=found_paths), \
         patch("src.ingest.extract_pages", return_value=[{"page_number": 1, "text": "hello"}]) as mock_extract, \
         patch("src.ingest.chunk_pages", return_value=[{"text": "hello", "source_file": "book_a.pdf", "page_number": 1, "chunk_index": 0}]) as mock_chunk, \
         patch("src.ingest.build_collection") as mock_store, \
         patch("builtins.print") as mock_print:
        try:
            from src import ingest
            import importlib
            importlib.reload(ingest)
            ingest.main()
        except SystemExit:
            pass
        return mock_extract, mock_chunk, mock_store, mock_print


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------

def test_extract_called_for_each_pdf(monkeypatch, tmp_path):
    mock_extract, _, _, _ = _run_main(monkeypatch, tmp_path, pdf_files=["a.pdf", "b.pdf"])
    assert mock_extract.call_count == 2


def test_chunk_called_with_source_filename(monkeypatch, tmp_path):
    _, mock_chunk, _, _ = _run_main(monkeypatch, tmp_path, pdf_files=["book_a.pdf"])
    _, kwargs = mock_chunk.call_args
    assert kwargs["source_file"] == "book_a.pdf"


def test_chunk_receives_config_sizes(monkeypatch, tmp_path):
    _, mock_chunk, _, _ = _run_main(monkeypatch, tmp_path, pdf_files=["book_a.pdf"])
    _, kwargs = mock_chunk.call_args
    assert kwargs["chunk_size"] == 512
    assert kwargs["chunk_overlap"] == 64


def test_build_collection_called_once_after_all_files(monkeypatch, tmp_path):
    _, _, mock_store, _ = _run_main(monkeypatch, tmp_path, pdf_files=["a.pdf", "b.pdf"])
    assert mock_store.call_count == 1


def test_all_chunks_accumulated_before_store(monkeypatch, tmp_path):
    """Two PDFs × 1 chunk each → build_collection receives 2 chunks."""
    chunk = {"text": "x", "source_file": "f.pdf", "page_number": 1, "chunk_index": 0}
    with patch("src.ingest.load_config", return_value={
            "embedding_model": "m", "chroma_path": "p", "chunk_size": 512,
            "chunk_overlap": 64, "collection_name": "books"}), \
         patch("src.ingest.Path.glob", return_value=[Path("a.pdf"), Path("b.pdf")]), \
         patch("src.ingest.extract_pages", return_value=[{"page_number": 1, "text": "t"}]), \
         patch("src.ingest.chunk_pages", return_value=[chunk]), \
         patch("src.ingest.build_collection") as mock_store, \
         patch("builtins.print"), \
         patch("sys.argv", ["ingest.py", "--dir", ".", "--config", "config.yaml"]):
        try:
            from src import ingest
            import importlib
            importlib.reload(ingest)
            ingest.main()
        except SystemExit:
            pass
    all_chunks = mock_store.call_args[0][0]
    assert len(all_chunks) == 2


# ---------------------------------------------------------------------------
# Output / progress messages
# ---------------------------------------------------------------------------

def test_prints_ingesting_line_per_file(monkeypatch, tmp_path):
    _, _, _, mock_print = _run_main(monkeypatch, tmp_path, pdf_files=["book_a.pdf"])
    printed = [str(c.args[0]) for c in mock_print.call_args_list if c.args]
    assert any("Ingesting" in line and "book_a.pdf" in line for line in printed)


def test_prints_done_summary(monkeypatch, tmp_path):
    _, _, _, mock_print = _run_main(monkeypatch, tmp_path, pdf_files=["book_a.pdf"])
    printed = [str(c.args[0]) for c in mock_print.call_args_list if c.args]
    assert any("Done" in line for line in printed)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_bad_pdf_is_skipped_and_does_not_abort(monkeypatch, tmp_path):
    fake_cfg = {
        "embedding_model": "m", "chroma_path": "p",
        "chunk_size": 512, "chunk_overlap": 64, "collection_name": "books",
    }
    paths = [tmp_path / "bad.pdf", tmp_path / "good.pdf"]

    def extract_side_effect(path):
        if "bad" in str(path):
            raise RuntimeError("corrupt file")
        return [{"page_number": 1, "text": "ok"}]

    with patch("src.ingest.load_config", return_value=fake_cfg), \
         patch("src.ingest.Path.glob", return_value=paths), \
         patch("src.ingest.extract_pages", side_effect=extract_side_effect), \
         patch("src.ingest.chunk_pages", return_value=[{"text": "ok", "source_file": "good.pdf", "page_number": 1, "chunk_index": 0}]), \
         patch("src.ingest.build_collection"), \
         patch("builtins.print") as mock_print, \
         patch("sys.argv", ["ingest.py", "--dir", str(tmp_path), "--config", "config.yaml"]):
        try:
            from src import ingest
            import importlib
            importlib.reload(ingest)
            ingest.main()
        except SystemExit:
            pass

    printed = [str(c.args[0]) for c in mock_print.call_args_list if c.args]
    assert any("Warning" in line and "bad.pdf" in line for line in printed)
    assert any("Done" in line for line in printed)


def test_exit_code_is_zero_on_success(monkeypatch, tmp_path):
    fake_cfg = {
        "embedding_model": "m", "chroma_path": "p",
        "chunk_size": 512, "chunk_overlap": 64, "collection_name": "books",
    }
    with patch("src.ingest.load_config", return_value=fake_cfg), \
         patch("src.ingest.Path.glob", return_value=[tmp_path / "book.pdf"]), \
         patch("src.ingest.extract_pages", return_value=[{"page_number": 1, "text": "t"}]), \
         patch("src.ingest.chunk_pages", return_value=[{"text": "t", "source_file": "book.pdf", "page_number": 1, "chunk_index": 0}]), \
         patch("src.ingest.build_collection"), \
         patch("builtins.print"), \
         patch("sys.argv", ["ingest.py", "--dir", str(tmp_path), "--config", "config.yaml"]):
        from src import ingest
        import importlib
        importlib.reload(ingest)
        try:
            ingest.main()
            exited_with = 0
        except SystemExit as e:
            exited_with = e.code
    assert exited_with == 0
