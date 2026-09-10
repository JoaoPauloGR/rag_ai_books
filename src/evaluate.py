import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import chromadb.errors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.generate import generate_answer
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


def score_entry(entry, cfg, k, answer_check=False):
    """Retrieve for one question and record the rank of the first page-level hit.

    A hit is a retrieved chunk from the expected source file on one of the
    expected pages. When ``answer_check`` is set, also generate an answer from
    the retrieved chunks and record whether every ``answer_keywords`` string
    occurs (case-insensitive substring) in it.
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

    answer_pass = None
    if answer_check:
        answer = generate_answer(
            entry["question"], chunks, cfg["generation_model"]
        )
        answer_pass = all(
            kw.lower() in answer.lower() for kw in entry["answer_keywords"]
        )

    return {
        "id": entry["id"],
        "first_hit_rank": hits[0] if hits else None,
        "answer_pass": answer_pass,
        "retrieved": [
            (c["chunk_id"], c["source_file"], c["page_number"]) for c in chunks
        ],
    }


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate(results, answer_checked=False):
    ranks = [r["first_hit_rank"] for r in results]
    agg = {
        "hit_rate@k": _mean(rank is not None for rank in ranks),
        "hit_rate@1": _mean(rank == 1 for rank in ranks),
        "hit_rate@3": _mean(rank is not None and rank <= 3 for rank in ranks),
        "MRR": _mean((1 / rank) if rank else 0 for rank in ranks),
    }
    if answer_checked:
        agg["answer_keyword_accuracy"] = _mean(
            bool(r["answer_pass"]) for r in results
        )
    return agg


def print_report(entries, results, agg, k, answer_checked=False):
    by_id = {e["id"]: e for e in entries}

    print(f"{'id':<5} {'hit':<4} {'rank':<5} {'ans':<4} book")
    print(f"{'-' * 5} {'-' * 4} {'-' * 5} {'-' * 4} {'-' * 4}")
    for r in results:
        rank = r["first_hit_rank"]
        hit = "Y" if rank is not None else "N"
        rank_str = str(rank) if rank is not None else "-"
        if not answer_checked:
            ans = "-"
        else:
            ans = "Y" if r["answer_pass"] else "N"
        book = by_id[r["id"]]["expected_source_file"]
        print(f"{r['id']:<5} {hit:<4} {rank_str:<5} {ans:<4} {book}")

    print()
    summary = (
        f"k={k}  n={len(results)}  "
        f"hit@k={agg['hit_rate@k']:.2f}  hit@1={agg['hit_rate@1']:.2f}  "
        f"hit@3={agg['hit_rate@3']:.2f}  MRR={agg['MRR']:.2f}"
    )
    if answer_checked:
        summary += f"  ans_kw={agg['answer_keyword_accuracy']:.2f}"
    print(summary)


_MD_HEADER = (
    "| run (UTC) | models (gen / embed) | chunk | k | n | "
    "hit@k | hit@1 | hit@3 | MRR | ans_kw |\n"
    "|---|---|---|---|---|---|---|---|---|---|\n"
)


def write_results(results, entries, agg, cfg, k, eval_set, out_dir, answer_checked):
    """Persist one run.

    Writes a full JSON dump at ``<out_dir>/<UTC timestamp>.json`` and appends a
    row to the Markdown log at ``<out_dir>.md`` (with a header block when that
    file is new). Returns ``(json_path, md_path)``.
    """
    by_id = {e["id"]: e for e in entries}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    json_path = out_dir / f"{stamp.replace(':', '-')}.json"
    md_path = out_dir.with_suffix(".md")

    payload = {
        "timestamp": stamp,
        "config": {
            "embedding_model": cfg["embedding_model"],
            "generation_model": cfg["generation_model"],
            "chunk_size": cfg["chunk_size"],
            "chunk_overlap": cfg["chunk_overlap"],
            "collection_name": cfg["collection_name"],
            "k": k,
        },
        "eval_set": str(eval_set),
        "n_questions": len(results),
        "aggregate": agg,
        "per_question": [
            {
                "id": r["id"],
                "question": by_id[r["id"]]["question"],
                "expected_source_file": by_id[r["id"]]["expected_source_file"],
                "expected_pages": by_id[r["id"]]["expected_pages"],
                "first_hit_rank": r["first_hit_rank"],
                "answer_pass": r["answer_pass"],
                "retrieved": r["retrieved"],
            }
            for r in results
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    ans_kw = f"{agg['answer_keyword_accuracy']:.2f}" if answer_checked else "-"
    row = (
        f"| {stamp} | {cfg['generation_model']} / {cfg['embedding_model']} "
        f"| {cfg['chunk_size']}/{cfg['chunk_overlap']} | {k} | {len(results)} "
        f"| {agg['hit_rate@k']:.2f} | {agg['hit_rate@1']:.2f} "
        f"| {agg['hit_rate@3']:.2f} | {agg['MRR']:.2f} | {ans_kw} |\n"
    )
    if not md_path.exists():
        md_path.write_text(_MD_HEADER, encoding="utf-8")
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(row)

    return json_path, md_path


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
    answer_check = not args.no_answer_check

    entries = load_eval_set(args.eval_set)

    try:
        results = [score_entry(entry, cfg, k, answer_check) for entry in entries]
    except chromadb.errors.NotFoundError:
        print(
            f"Collection '{cfg['collection_name']}' not found in '{cfg['chroma_path']}' "
            "— run ingestion first.",
            file=sys.stderr,
        )
        sys.exit(1)

    agg = aggregate(results, answer_check)
    print_report(entries, results, agg, k, answer_check)

    json_path, md_path = write_results(
        results, entries, agg, cfg, k, args.eval_set, args.out_dir, answer_check
    )
    print(f"\nwrote    {json_path}")
    print(f"appended {md_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
