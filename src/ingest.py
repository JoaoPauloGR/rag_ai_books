import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Ingest a directory of PDFs into the vector store.")
    parser.add_argument("--dir", required=True, help="Path to directory of PDFs to ingest")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.parse_args()

    print("Ingestion not yet implemented")
    sys.exit(0)


if __name__ == "__main__":
    main()
