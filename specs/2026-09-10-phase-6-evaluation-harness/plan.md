# Phase 6 — Implementation Plan

Groups are ordered so `pytest` stays green after each. Commit per group. Group 3 is a
manual curation step; everything else is code + tests.

## Group 1 — `eval/` scaffolding

1. Create `eval/` with a `.gitkeep`.
2. `.gitignore`: add `eval/eval_set.draft.json` and `eval/results/`.
3. Add a stub `eval/eval_set.json` containing `{"entries": []}` so imports/tests have a
   path to read (it gets filled in Group 3).
4. `tests/test_eval_set_file.py`: `eval/eval_set.json` parses as JSON and has an
   `"entries"` list.

## Group 2 — Candidate drafter (`src/build_eval_set.py`)

5. Create `src/build_eval_set.py`:
   - `argparse`: `--config` (`config.yaml`), `--n` (40), `--seed` (0),
     `--out` (`eval/eval_set.draft.json`).
   - `load_config`; open the collection; `collection.get(include=["documents", "metadatas"])`.
   - `random.Random(seed)` → sample `min(n, len(docs))` distinct indices.
   - For each: build the drafting prompt (requirements.md) with the chunk text; call
     `ollama.chat(model=cfg["generation_model"], messages=[{"role": "user", "content": prompt}])`.
   - Parse `response["message"]["content"]` as JSON; pull `question` and `answer_keywords`.
     On `JSONDecodeError` / missing keys / empty question → `print` a warning, skip.
   - Emit an entry: `id` = `f"q{ordinal:02d}"`, `question`, `expected_source_file` +
     `expected_pages` = `[metadata["page_number"]]` from the sampled chunk,
     `answer_keywords` (model's, as a starting point), `verified: false`,
     `notes` = `f"drafted from chunk '{chunk_id}'"` where `chunk_id` is the collection id.
   - Write the list wrapped as `{"entries": [...]}` to `--out` (overwrite); print a summary
     (`drafted X, skipped Y`).
6. `tests/test_build_eval_set.py` (mock `chromadb.PersistentClient` and `ollama.chat`):
   - `--n 3` with 10 fake chunks → 3 draft entries;
   - each entry has `verified is False`, `expected_source_file` / `expected_pages` from the
     matching fake chunk metadata, and a parseable shape;
   - one fake chunk whose mocked reply is `"not json"` → that entry skipped, run still
     succeeds, warning printed;
   - same `--seed` twice → identical sampled chunk ids.

## Group 3 — Curate `eval/eval_set.json` (manual)

7. Run `uv run python src/build_eval_set.py --n 40` against the real collection with Ollama
   up.
8. Review `eval/eval_set.draft.json`. For ~20 keepers: fix the question wording, confirm
   `expected_source_file` by opening the PDF, widen `expected_pages` if the answer spans a
   boundary, **rewrite `answer_keywords` by hand** (1–3 terms a correct answer must
   contain), set `verified: true`, renumber `id`s `q01`..`q20`.
9. Save the curated entries as `eval/eval_set.json` (replacing the Group 1 stub). Aim for
   spread across at least ~10 different books and a mix of definitional / comparative /
   factual questions.
10. `git add eval/eval_set.json` — it is a committed artifact. Do **not** commit
    `eval/eval_set.draft.json`.

## Group 4 — Retrieval scoring (`src/evaluate.py`)

11. Create `src/evaluate.py` with `argparse`: `--config` (`config.yaml`),
    `--eval-set` (`eval/eval_set.json`), `--k` (default `None` → `cfg["top_k"]`),
    `--no-answer-check` (flag), `--out-dir` (`eval/results`).
12. `load_eval_set(path)` → parse, then validate every entry: non-empty `question`,
    `expected_source_file`, `expected_pages` (non-empty list of ints), `answer_keywords`
    (non-empty list), `verified is True`. Collect all failures, print them, `sys.exit(2)`.
13. `score_entry(entry, cfg, k)`:
    - `chunks = retrieve_chunks(entry["question"], cfg["embedding_model"], cfg["chroma_path"], cfg["collection_name"], k)`
      (wrap `chromadb.errors.NotFoundError` at the top level → message + `sys.exit(1)`).
    - `hits = [i for i, c in enumerate(chunks, start=1) if c["source_file"] == entry["expected_source_file"] and c["page_number"] in entry["expected_pages"]]`.
    - return `first_hit_rank = hits[0] if hits else None`, plus the retrieved
      `(chunk_id, source_file, page_number)` list for the JSON dump.
14. Aggregate over entries: `hit_rate@k = mean(rank is not None)`,
    `hit_rate@1 = mean(rank == 1)`, `hit_rate@3 = mean(rank is not None and rank <= 3)`,
    `MRR = mean(1/rank if rank else 0)`.
15. Print a per-question table (`id | hit | rank | book`) and the aggregate summary line.
16. `tests/test_evaluate.py` (mock `retrieve_chunks`):
    - first hit at rank 2 → `MRR` contribution `0.5`, counts for `hit@3` not `hit@1`;
    - no page match anywhere in top-k → miss, `rank None`;
    - page in `expected_pages` but wrong `source_file` → miss;
    - validation: an entry with `verified: false` or `answer_keywords: []` → `SystemExit(2)`.

## Group 5 — Answer keyword check

17. In `evaluate.py`, unless `--no-answer-check`: after retrieval,
    `answer = generate_answer(entry["question"], chunks, cfg["generation_model"])`;
    `answer_pass = all(kw.lower() in answer.lower() for kw in entry["answer_keywords"])`.
18. Add `answer_keyword_accuracy = mean(answer_pass)` to the aggregate and an `ans`
    column to the per-question table. When `--no-answer-check`, skip generation entirely
    and report `ans_kw` as `-`.
19. `tests/test_evaluate.py` (also mock `generate_answer`):
    - all keywords present (case-insensitive) → pass; one missing → fail;
    - `--no-answer-check` → `generate_answer` is never called and the accuracy key is
      absent from the aggregate.

## Group 6 — Results output

20. `write_results(aggregate, cfg, k, out_dir, answer_checked)`:
    - ensure `out_dir` exists; write `<out_dir>/<UTC ISO, ':'→'-'>.json` with
      `{timestamp, config:{embedding_model, generation_model, chunk_size, chunk_overlap, collection_name, k}, eval_set, n_questions, aggregate, per_question:[...]}`.
    - append a row to `eval/results.md` (write the header block from requirements.md first
      if the file does not exist). `chunk` column = `f"{chunk_size}/{chunk_overlap}"`;
      `ans_kw` = `-` when `not answer_checked`.
21. Call it at the end of `main()`; print the two written paths. `main()` returns exit 0.
22. `tests/test_evaluate.py`:
    - a full `main()` run against a `tmp_path` `--out-dir` and a copied 2-entry eval set
      (retrieve/generate mocked) writes one `.json` and appends exactly one line to a
      `tmp_path` `results.md`; a second run appends a second line and does not duplicate
      the header.

## Group 7 — Baseline + docs

23. Run `uv run python src/evaluate.py` against the committed eval set with the current
    `config.yaml`. Commit the resulting `eval/results.md` (baseline row) — **not** the
    `eval/results/*.json`.
24. `README.md`: add an "Evaluate retrieval" usage block
    (`uv run python src/build_eval_set.py`, `uv run python src/evaluate.py`), add `eval/`
    to Project Structure (`eval_set.json` tracked, `results.md` tracked, drafts/results
    ignored), flip the roadmap table row for Phase 6 to "in progress" → "complete" as
    appropriate.
25. `specs/roadmap.md`: note the page-level scoring choice under Phase 6 (one line) so it
    doesn't read as contradicting the old "expected_chunk_ids" wording.

## Manual verification

Work through every check in `validation.md` end to end before merging.
