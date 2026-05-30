from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(pages: list[dict], source_file: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    chunk_index = 0
    for page in pages:
        text = page["text"]
        if not text.strip():
            continue
        for sub in splitter.split_text(text):
            chunks.append({
                "text": sub,
                "source_file": source_file,
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
            })
            chunk_index += 1
    return chunks
