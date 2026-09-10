import argparse
import json
import sys
from pathlib import Path

import chromadb.errors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.retrieve import retrieve_chunks


def _validate_entry(entry, idx):
    eid = entry.get("id", f"entry[{idx}]")
    errors = []

    question = entry.get("question")
    if not isinstance(question, str) or not question.strip():
        errors.append(f"{eid}: question must be a non-empty string")

    source_file = entry.get("expected_source_file")
    if not isinstance(source_file, str) or not source_file.strip():
        errors.append(f"{eid}: expected_source_file must be a non-empty string")

    pages = entry.get("expected_pages")
    if (
        not isinstance(pages, list)
        or not pages
        or not all(isinstance(p, int) and not isinstance(p, bool) for p in pages)
    ):
        errors.append(f"{eid}: expected_pages must be a non-empty list of ints")

    keywords = entry.get("answer_keywords")
    if not isinstance(keywords, list) or not keywords:
        errors.append(f"{eid}: answer_keywords must be a non-empty list")

    if entry.get("verified") is not True:
        errors.append(f"{eid}: verified must be true")

    return errors


def load_eval_set(path):
    """Parse and validate the eval set. Print every failure and exit 2 if any."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    entries = data["entries"]
    errors = []
    for idx, entry in enumerate(entries):
        errors.extend(_validate_entry(entry, idx))

    if errors:
        print(f"Eval set '{path}' is invalid:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(2)

    return entries


def score_entry(entry, cfg, k):
    """Retrieve for one question and record the rank of the first page-level hit.

    A hit is a retrieved chunk from the expected source file on one of the
    expected pages.
    """
    chunks = retrieve_chunks(
        entry["question"],
        cfg["embedding_model"],
        cfg["chroma_path"],
        cfg["collection_name"],
        k,
    )

    hits = [
        i
        for i, c in enumerate(chunks, start=1)
        if c["source_file"] == entry["expected_source_file"]
        and c["page_number"] in entry["expected_pages"]
    ]

    return {
        "id": entry["id"],
        "first_hit_rank": hits[0] if hits else None,
        "retrieved": [
            (c["chunk_id"], c["source_file"], c["page_number"]) for c in chunks
        ],
    }


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate(results):
    ranks = [r["first_hit_rank"] for r in results]
    return {
        "hit_rate@k": _mean(rank is not None for rank in ranks),
        "hit_rate@1": _mean(rank == 1 for rank in ranks),
        "hit_rate@3": _mean(rank is not None and rank <= 3 for rank in ranks),
        "MRR": _mean((1 / rank) if rank else 0 for rank in ranks),
    }


def print_report(entries, results, agg, k):
    by_id = {e["id"]: e for e in entries}

    print(f"{'id':<5} {'hit':<4} {'rank':<5} book")
    print(f"{'-' * 5} {'-' * 4} {'-' * 5} {'-' * 4}")
    for r in results:
        rank = r["first_hit_rank"]
        hit = "Y" if rank is not None else "N"
        rank_str = str(rank) if rank is not None else "-"
        book = by_id[r["id"]]["expected_source_file"]
        print(f"{r['id']:<5} {hit:<4} {rank_str:<5} {book}")

    print()
    print(
        f"k={k}  n={len(results)}  "
        f"hit@k={agg['hit_rate@k']:.2f}  hit@1={agg['hit_rate@1']:.2f}  "
        f"hit@3={agg['hit_rate@3']:.2f}  MRR={agg['MRR']:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Score retrieval quality against the curated eval set."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--eval-set", default="eval/eval_set.json", help="Path to the eval set JSON"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Chunks to retrieve per question (default: cfg['top_k'])",
    )
    parser.add_argument(
        "--no-answer-check",
        action="store_true",
        help="Skip answer keyword checking (retrieval-only pass)",
    )
    parser.add_argument(
        "--out-dir", default="eval/results", help="Directory for per-run JSON dumps"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    k = args.k if args.k is not None else cfg["top_k"]

    entries = load_eval_set(args.eval_set)

    try:
        results = [score_entry(entry, cfg, k) for entry in entries]
    except chromadb.errors.NotFoundError:
        print(
            f"Collection '{cfg['collection_name']}' not found in '{cfg['chroma_path']}' "
            "— run ingestion first.",
            file=sys.stderr,
        )
        sys.exit(1)

    agg = aggregate(results)
    print_report(entries, results, agg, k)

    sys.exit(0)


if __name__ == "__main__":
    main()
