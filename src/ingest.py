import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config

if "extract_pages" not in globals():
    from src.extract import extract_pages
    from src.chunk import chunk_pages
    from src.store import build_collection


def main():
    parser = argparse.ArgumentParser(description="Ingest a directory of PDFs into the vector store.")
    parser.add_argument("--dir", required=True, help="Path to directory of PDFs to ingest")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    pdf_files = sorted(Path(args.dir).glob("*.pdf"))
    if not pdf_files:
        print(f"Warning: no PDFs found in {args.dir}")
        sys.exit(0)

    all_chunks = []
    ingested_count = 0

    for pdf_path in pdf_files:
        print(f"Ingesting: {pdf_path.name}")
        try:
            pages = extract_pages(str(pdf_path))
        except RuntimeError as e:
            print(f"Warning: {pdf_path.name}: {e}")
            continue
        chunks = chunk_pages(
            pages,
            source_file=pdf_path.name,
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
        )
        all_chunks.extend(chunks)
        ingested_count += 1

    build_collection(
        all_chunks,
        embedding_model=cfg["embedding_model"],
        chroma_path=cfg["chroma_path"],
        collection_name=cfg["collection_name"],
    )

    print(f"Done. Ingested {ingested_count} files, {len(all_chunks)} chunks total.")
    sys.exit(0)


if __name__ == "__main__":
    main()
