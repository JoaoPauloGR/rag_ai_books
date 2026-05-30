from src.chunk import chunk_pages

PAGES = [
    {"page_number": 1, "text": "Alpha " * 200},
    {"page_number": 2, "text": "Beta " * 200},
]


def test_chunks_have_required_keys():
    result = chunk_pages(PAGES, source_file="book.pdf", chunk_size=512, chunk_overlap=64)
    assert result
    for chunk in result:
        assert set(chunk.keys()) == {"text", "source_file", "page_number", "chunk_index"}


def test_source_file_propagated():
    result = chunk_pages(PAGES, source_file="book.pdf", chunk_size=512, chunk_overlap=64)
    assert all(c["source_file"] == "book.pdf" for c in result)


def test_chunk_index_is_globally_sequential():
    result = chunk_pages(PAGES, source_file="book.pdf", chunk_size=512, chunk_overlap=64)
    indices = [c["chunk_index"] for c in result]
    assert indices == list(range(len(result)))


def test_blank_pages_are_skipped():
    pages = [
        {"page_number": 1, "text": "Real content " * 100},
        {"page_number": 2, "text": "   "},
        {"page_number": 3, "text": "More content " * 100},
    ]
    result = chunk_pages(pages, source_file="book.pdf", chunk_size=512, chunk_overlap=64)
    page_numbers = {c["page_number"] for c in result}
    assert 2 not in page_numbers
    assert 1 in page_numbers
    assert 3 in page_numbers


def test_empty_input_returns_empty_list():
    assert chunk_pages([], source_file="book.pdf", chunk_size=512, chunk_overlap=64) == []
