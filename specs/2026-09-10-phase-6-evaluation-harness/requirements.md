# Phase 6 — Evaluation Harness

## Context

Phases 0–4 built a working RAG pipeline; Phase 5 adds the chat UI, a feedback store, and
incremental ingestion with stable chunk IDs. Nothing so far measures **retrieval
quality** — whether the right passage is actually reaching the model. This phase builds a
fixed test set and a repeatable scorer so every later change (Phase 7 chunk tuning,
Phase 8 model swaps) can be judged against a number instead of a vibe.

This is mission success criterion #4: "a baseline evaluation set measures retrieval
quality (hit rate ≥ 0.7 on 20 test questions)". Phase 6 delivers the harness and a
recorded baseline; *reaching* 0.7 is Phase 7's job if the baseline falls short.

### Learning goals

- **Hit rate (Recall@k)** — the fraction of questions whose gold passage appears anywhere
  in the top-k retrieved chunks. The most intuitive retrieval metric.
- **MRR (Mean Reciprocal Rank)** — averages `1 / rank_of_first_hit`; rewards putting the
  right chunk at position 1, which matters because the top chunk dominates the prompt.
- **Why a fixed test set** — a RAG system can look fine on the handful of questions you
  tried and fail everywhere else; only a frozen set tells you if a change actually helped.
- **The retrieval ceiling** — if the wrong chunks are retrieved, no model can answer
  correctly. The harness makes that ceiling visible.
- **Cheap answer checking** — a keyword-presence check is a crude but fast, deterministic
  proxy for answer faithfulness, with none of the cost or noise of an LLM judge.
- **LLM-drafted question risk** — a model writing questions from a chunk tends to echo its
  wording (leakage) and stay shallow; human verification is what makes the set trustworthy.

### Dependencies

- Needs a populated `books` collection (Phase 3) and `src/retrieve.py` / `src/generate.py`
  (Phase 4).
- **Does not** need Phase 5 merged: scoring is page-level, and page numbers are stable
  regardless of chunking or ID scheme.
- The Phase 5 feedback store is **not** used here (questions are LLM-drafted, not
  bootstrapped). It stays available for a future expansion of the eval set.

## Scope

### What this phase produces

- **`src/build_eval_set.py`** — re-runnable candidate-question drafter.
  - `chromadb.PersistentClient(cfg["chroma_path"]).get_collection(cfg["collection_name"])`,
    `collection.get(include=["documents", "metadatas"])`, sample `--n` chunks with a
    seeded RNG (`--seed`, default 0).
  - For each sampled chunk, call `ollama.chat(model=cfg["generation_model"], ...)` with the
    drafting prompt (below): produce one question answerable from that chunk plus 1–3
    candidate answer keywords. Parse the model's JSON reply.
  - Write `eval/eval_set.draft.json` (overwrite) — a list of entries in the schema below
    with `verified: false`, `expected_source_file` / `expected_pages` copied from the
    sampled chunk's metadata, and `notes` recording the source chunk id.
  - A chunk whose reply fails to parse is skipped with a warning, not a crash.
  - Args: `--config` (default `config.yaml`), `--n` (default 40), `--seed` (default 0),
    `--out` (default `eval/eval_set.draft.json`).

- **`eval/eval_set.json`** — the committed, hand-verified test set (target: 20 entries,
  all `verified: true`). Produced by curating the draft (Group 3, a manual step).

- **`src/evaluate.py`** — the scorer.
  - Loads `--eval-set` (default `eval/eval_set.json`), **validates** it: every entry has a
    non-empty `question`, `expected_source_file`, non-empty `expected_pages`, non-empty
    `answer_keywords`, and `verified == true`. Any violation → printed error, exit 2.
  - For each entry: `retrieve_chunks(question, ..., k)` where `k = --k` or `cfg["top_k"]`.
    - **Retrieval hit** = some retrieved chunk has `source_file == expected_source_file`
      and `page_number in expected_pages`.
    - Record the 1-based rank of the first hit (or none).
  - Unless `--no-answer-check`: `generate_answer(question, chunks, cfg["generation_model"])`,
    then **answer pass** = every string in `answer_keywords` occurs (case-insensitive
    substring) in the answer.
  - Aggregate: `hit_rate@k`, `hit_rate@1`, `hit_rate@3` (from the same ranked list),
    `MRR` (mean of `1/first_hit_rank`, 0 when no hit), `answer_keyword_accuracy`
    (omitted when `--no-answer-check`).
  - Output (Group 6): a per-question table + summary to the console, an appended row to
    `eval/results.md`, and a full `eval/results/<UTC-timestamp>.json`.
  - Args: `--config`, `--eval-set`, `--k`, `--no-answer-check`, `--out-dir`
    (default `eval/results`). Missing collection → friendly message, exit 1. Successful
    run → exit 0.

- **`eval/results.md`** — committed Markdown log, one row per run (created with a header if
  absent).
- Tests: `tests/test_build_eval_set.py`, `tests/test_evaluate.py`.
- `.gitignore`: `eval/eval_set.draft.json`, `eval/results/`.
- Docs: README "Project Structure" gains `eval/`; README + roadmap status for Phase 6.

### What this phase does NOT produce

- Any change to retrieval, generation, chunking, or config (`src/config.py` untouched —
  no new keys; `evaluate.py` reuses `top_k` and `generation_model`).
- Tuning to hit a target score — that is Phase 7.
- An LLM-as-judge / RAGAS-style faithfulness scorer.
- Bootstrapping questions from the Phase 5 feedback store.
- Chunk-ID-level scoring (see decision below).
- A grid runner over chunk parameters — Phase 7 consumes this harness; it isn't built here.

## Decisions

| Decision | Choice | Reason |
|---|---|---|
| Question source | LLM-drafted from sampled chunks, then human-verified | Fastest route to a full 20; the `verified` gate + hand-written keywords keep it honest |
| Drafting tool | Committed `src/build_eval_set.py`, re-runnable | Reproducible and extendable later; writes a `.draft.json`, never touches the curated file |
| Drafting model | `cfg["generation_model"]` (llama3.2:3b) | No new config; the draft is disposable input to human review anyway |
| Retrieval target | Page-level: `expected_source_file` + `expected_pages` | Matches Phase 4 citations and **survives re-chunking**, so the same set stays valid through Phase 7 (see `tech-stack.md` → Open considerations) |
| Chunk-ID scoring | Not done | `source_file::chunk_index` breaks on any `chunk_size` / `chunk_overlap` change; page numbers don't |
| `k` for scoring | `cfg["top_k"]` by default, `--k` to override; also report @1 and @3 | Measures the system as configured; @1/@3 are free from the same ranked list |
| Answer check | Generate, then require **all** human-written keywords (1–3, case-insensitive substring) | Cheap, deterministic, catches "right chunk, wrong answer"; `--no-answer-check` for a fast retrieval-only pass |
| Keyword authoring | Human, during curation | Model-proposed keywords tend to be loose; the draft may suggest them but the human owns the final list |
| Results | Console + append `eval/results.md` + `eval/results/<ts>.json` | Human-readable history (Phase 7 references `results.md`) plus machine-readable detail per run |
| Committed vs ignored | `eval/eval_set.json` + `eval/results.md` tracked; `eval/eval_set.draft.json` + `eval/results/*.json` ignored | The curated set and the run log are the durable artifacts; drafts and raw run dumps are noise |
| Done bar | Harness runs + baseline row committed, any score | Roadmap "Done when" is "prints a hit rate + per-question table"; hitting 0.7 is Phase 7 |

## `eval/eval_set.json` schema

```json
{
  "entries": [
    {
      "id": "q01",
      "question": "What problem does a feature store solve for ML teams?",
      "expected_source_file": "Building ML Systems with a Feature Store.pdf",
      "expected_pages": [12, 13],
      "answer_keywords": ["training-serving skew", "reuse"],
      "verified": true,
      "notes": "drafted from chunk 'Building ML Systems with a Feature Store.pdf::47'"
    }
  ]
}
```

- `expected_pages` — 1-based page numbers, as produced by `src/extract.py`. A hit on any
  listed page counts.
- `answer_keywords` — 1–3 short strings; **all** must appear in the generated answer for
  the answer check to pass.
- `verified` — must be `true` for `evaluate.py` to accept the entry.

## `eval/results.md` row

```
| run (UTC)            | models (gen / embed)        | chunk | k | n  | hit@k | hit@1 | hit@3 | MRR  | ans_kw |
|---------------------|-----------------------------|-------|---|----|-------|-------|-------|------|--------|
| 2026-09-10T14:03:22Z | llama3.2:3b / nomic-embed-text | 512/64 | 5 | 20 | 0.75  | 0.55  | 0.70  | 0.62 | 0.55   |
```

`ans_kw` is `-` when the run used `--no-answer-check`.

## Drafting prompt (`src/build_eval_set.py`)

```
You are helping build a retrieval test set for a RAG system over AI/ML books.
Given one passage, write ONE specific question that the passage clearly answers, and
list 1 to 3 short keywords or phrases that a correct answer must contain.
Do not refer to "the passage" or "the text" in the question — it must read as a
standalone question. Prefer concrete, factual questions over vague ones.

Passage:
{chunk_text}

Reply with JSON only:
{"question": "...", "answer_keywords": ["...", "..."]}
```

## Out of scope

- LLM-as-judge or RAGAS faithfulness scoring
- Feedback-store bootstrapping of questions
- Chunk-parameter grid runs (Phase 7)
- Any tuning to raise the score (Phase 7)
- Changes to `src/config.py`, retrieval, generation, or chunking
