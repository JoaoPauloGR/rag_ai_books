import fitz


def extract_pages(pdf_path: str) -> list[dict]:
    try:
        doc = fitz.open(pdf_path)
        pages = [
            {"page_number": i + 1, "text": page.get_text("text")}
            for i, page in enumerate(doc)
        ]
        doc.close()
        return pages
    except Exception as e:
        raise RuntimeError(f"Failed to extract {pdf_path}: {e}") from e
