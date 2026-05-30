import argparse
import sys
from pathlib import Path

import chromadb.errors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.retrieve import retrieve_chunks
from src.generate import generate_answer


def main():
    parser = argparse.ArgumentParser(description="Query the vector store with a natural-language question.")
    parser.add_argument("--query", required=True, help="Natural-language question to ask")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    try:
        chunks = retrieve_chunks(
            args.query,
            cfg["embedding_model"],
            cfg["chroma_path"],
            cfg["collection_name"],
            cfg["top_k"],
        )
    except chromadb.errors.NotFoundError:
        print(
            f"Error: collection '{cfg['collection_name']}' not found in '{cfg['chroma_path']}'. "
            "Run the ingestion pipeline first.",
            file=sys.stderr,
        )
        sys.exit(1)

    answer = generate_answer(args.query, chunks, cfg["generation_model"])
    print(answer)

    seen = set()
    for chunk in chunks:
        key = (chunk["source_file"], chunk["page_number"])
        if key not in seen:
            seen.add(key)
            print(f"[Source: {chunk['source_file']}, p. {chunk['page_number']}]")

    sys.exit(0)


if __name__ == "__main__":
    main()
