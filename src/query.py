import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Query the vector store with a natural-language question.")
    parser.add_argument("--query", required=True, help="Natural-language question to ask")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.parse_args()

    print("Query not yet implemented")
    sys.exit(0)


if __name__ == "__main__":
    main()
