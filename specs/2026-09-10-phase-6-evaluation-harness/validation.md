# Phase 6 — Validation

All checks pass before merging. Ollama running; the `books` collection populated.

## 1. Drafter produces a usable draft

```
uv run python src/build_eval_set.py --n 20 --seed 1
```

Expected:
- `eval/eval_set.draft.json` is written, parses as JSON, has an `entries` list.
- Each entry has a non-empty `question`, `expected_source_file` matching a real PDF in
  `data/books/`, `expected_pages` a non-empty list of ints, `verified: false`, and a
  `notes` string naming the source chunk.
- The summary prints `drafted N, skipped M` with `N + M == 20` (or `== chunk count` if the
  collection is smaller).
- Re-running with `--seed 1` samples the same chunks (diff the two drafts — question
  wording may vary, chunk ids/pages do not).

## 2. Curated eval set is valid and committed

```
uv run python -c "import json,sys; d=json.load(open('eval/eval_set.json')); e=d['entries']; assert len(e)>=20, len(e); assert all(x['verified'] is True for x in e); assert all(x['answer_keywords'] for x in e); assert all(x['expected_pages'] for x in e); print(len(e),'entries OK')"
git status --porcelain eval/eval_set.json eval/eval_set.draft.json
```

Expected: the assertion prints `>=20 entries OK`; `git status` shows `eval/eval_set.json`
staged/tracked and **nothing** for `eval/eval_set.draft.json` (ignored). Entries span at
least ~10 distinct `expected_source_file` values.

## 3. Evaluate runs and reports

```
uv run python src/evaluate.py
```

Expected:
- A per-question table (`id | hit | rank | book | ans`) followed by a summary line with
  `hit@k`, `hit@1`, `hit@3`, `MRR`, `answer_keyword_accuracy`.
- `k` in the summary equals `top_k` from `config.yaml` (5).
- Exit code 0.
- Two paths printed: `eval/results/<timestamp>.json` and `eval/results.md`.

## 4. Metrics are arithmetically right

Pick one question from the table with `rank = 1` and one with `rank = 3` (or use a
throwaway 2-entry eval set). Verify by hand:
- `hit@1` counts only rank-1 questions.
- `hit@3` counts rank ≤ 3.
- `MRR` = mean of `1/rank` (0 for misses). For a 2-entry set with ranks `1` and `None`,
  `MRR = 0.5`, `hit@k = 0.5`.

## 5. Page match requires the right file

Temporarily edit one eval entry so `expected_source_file` is a different real PDF but
keep its `expected_pages`. Re-run.

Expected: that question flips to a miss (`hit = N`, `rank = -`). Restore the entry.

## 6. Answer keyword check

Inspect a question where `hit = Y` but `ans = N`: read
`eval/results/<timestamp>.json` for that entry and confirm at least one `answer_keywords`
string is genuinely absent from the generated answer (case-insensitive).

```
uv run python src/evaluate.py --no-answer-check
```

Expected: no generation happens (noticeably faster), the summary omits
`answer_keyword_accuracy`, and the new `results.md` row shows `ans_kw = -`.

## 7. Eval-set validation fails loudly

Set `verified: false` on one entry of `eval/eval_set.json` (or blank its
`answer_keywords`). Run `uv run python src/evaluate.py`.

Expected: a message naming the offending entry id and field, exit code 2, no results
files written. Restore the entry.

## 8. Results log accumulates

Run `uv run python src/evaluate.py` twice.

Expected:
- `eval/results.md` has a header block plus **two** data rows (header not duplicated).
- `eval/results/` contains two timestamped JSON files, each with `config`, `aggregate`,
  and a `per_question` array of length = number of entries.
- `git status` shows `eval/results.md` tracked and `eval/results/` ignored.

## 9. Missing collection is handled

Rename `data/chroma_db/` temporarily, run `uv run python src/evaluate.py`.

Expected: a clear "collection not found — run ingestion first" message, non-zero exit, no
traceback, no results files. Restore the directory.

## 10. Baseline recorded

The committed `eval/results.md` contains at least one row produced with the current
`config.yaml` (the baseline). The number itself is not a merge gate — a baseline below
0.7 is Phase 7's problem.

## 11. Nothing else changed

```
uv run pytest -q
git diff --stat main -- src/config.py config.yaml src/retrieve.py src/generate.py
```

Expected: full suite green, including `test_build_eval_set.py` and `test_evaluate.py`.
`config.py`, `config.yaml`, `retrieve.py`, `generate.py` are untouched by this phase.

## Not required for merge

- Hit rate ≥ 0.7 (Phase 7 — chunk tuning)
- A chunk-parameter grid runner (Phase 7)
- LLM-as-judge / RAGAS faithfulness scoring
- Questions sourced from the Phase 5 feedback store
- Chunk-ID-level scoring
