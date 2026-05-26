import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Ingest a directory of PDFs into the vector store.")
    parser.add_argument("--dir", required=True, help="Path to directory of PDFs to ingest")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Embedding model: {cfg['embedding_model']}")
    print(f"Chroma path:     {cfg['chroma_path']}")
    print(f"Chunk size:      {cfg['chunk_size']}")
    print(f"Chunk overlap:   {cfg['chunk_overlap']}")

    print("Ingestion not yet implemented")
    sys.exit(0)


if __name__ == "__main__":
    main()
