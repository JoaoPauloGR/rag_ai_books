import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Query the vector store with a natural-language question.")
    parser.add_argument("--query", required=True, help="Natural-language question to ask")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Generation model: {cfg['generation_model']}")
    print(f"Embedding model:  {cfg['embedding_model']}")

    print("Query not yet implemented")
    sys.exit(0)


if __name__ == "__main__":
    main()
