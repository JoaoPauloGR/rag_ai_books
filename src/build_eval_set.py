import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
import ollama

from src.config import load_config

_PROMPT_TEMPLATE = """\
You are helping build a retrieval test set for a RAG system over AI/ML books.
Given one passage, write ONE specific question that the passage clearly answers, and
list 1 to 3 short keywords or phrases that a correct answer must contain.
Do not refer to "the passage" or "the text" in the question — it must read as a
standalone question. Prefer concrete, factual questions over vague ones.

Passage:
{chunk_text}

Reply with JSON only:
{"question": "...", "answer_keywords": ["...", "..."]}
"""


def _draft_prompt(chunk_text: str) -> str:
    return _PROMPT_TEMPLATE.replace("{chunk_text}", chunk_text)


def draft_entries(docs, metas, ids, sample_indices, generation_model):
    """Ask the model to draft one question per sampled chunk.

    Returns (entries, skipped). A chunk whose reply fails to parse is skipped
    with a warning, never a crash.
    """
    entries = []
    skipped = 0

    for idx in sample_indices:
        chunk_id = ids[idx]
        meta = metas[idx]
        prompt = _draft_prompt(docs[idx])
        response = ollama.chat(
            model=generation_model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response["message"]["content"]

        try:
            parsed = json.loads(content)
            question = parsed["question"]
            answer_keywords = parsed["answer_keywords"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: chunk '{chunk_id}': could not parse model reply ({e}); skipping")
            skipped += 1
            continue

        if not isinstance(question, str) or not question.strip():
            print(f"Warning: chunk '{chunk_id}': empty question; skipping")
            skipped += 1
            continue

        entries.append(
            {
                "id": f"q{len(entries) + 1:02d}",
                "question": question.strip(),
                "expected_source_file": meta["source_file"],
                "expected_pages": [meta["page_number"]],
                "answer_keywords": answer_keywords,
                "verified": False,
                "notes": f"drafted from chunk '{chunk_id}'",
            }
        )

    return entries, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Draft candidate eval-set questions from sampled collection chunks."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--n", type=int, default=40, help="Number of chunks to sample")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling")
    parser.add_argument(
        "--out",
        default="eval/eval_set.draft.json",
        help="Where to write the draft (overwritten each run)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    client = chromadb.PersistentClient(path=cfg["chroma_path"])
    collection = client.get_collection(cfg["collection_name"])
    result = collection.get(include=["documents", "metadatas"])
    docs = result["documents"]
    metas = result["metadatas"]
    ids = result["ids"]

    if not docs:
        print("Collection is empty — run ingestion first.")
        sys.exit(1)

    n = min(args.n, len(docs))
    sample_indices = random.Random(args.seed).sample(range(len(docs)), n)

    entries, skipped = draft_entries(docs, metas, ids, sample_indices, cfg["generation_model"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"entries": entries}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"drafted {len(entries)}, skipped {skipped} -> {out_path}")


if __name__ == "__main__":
    main()
