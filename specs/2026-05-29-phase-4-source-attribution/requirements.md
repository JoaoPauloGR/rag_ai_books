# Phase 4 — Source Attribution in Answers

## Context

Phase 3 produced a complete ingestion pipeline that stores chunks in ChromaDB with
`source_file`, `page_number`, and `chunk_index` metadata. Phase 4 activates the query
side of the system: `src/query.py` currently prints "Query not yet implemented." This
phase turns it into a working RAG query loop that retrieves relevant chunks, generates a
grounded answer, and prints the sources below the answer.

The learning goals are: how to structure a retrieval-augmented prompt so the LLM is
forced to cite context rather than hallucinate, and how to trace an answer back to the
specific page it came from.

## Scope

### What this phase produces

- `src/retrieve.py` — `retrieve_chunks(question, embedding_model, chroma_path, collection_name, top_k) -> list[dict]`
  - Embeds the question using `ollama.embeddings`
  - Queries ChromaDB for the `top_k` nearest chunks
  - Returns a list of dicts: `{"text": str, "source_file": str, "page_number": int}`
- `src/generate.py` — `generate_answer(question, chunks, generation_model) -> str`
  - Builds a context block from the retrieved chunk texts
  - Applies the hard-constraint prompt template (see below)
  - Calls `ollama.chat` with the assembled prompt
  - Returns the model's response string
- Updated `src/query.py` — full CLI pipeline:
  - Retrieves chunks → generates answer → prints answer → prints deduplicated sources
  - Sources printed as `[Source: <source_file>, p. <page_number>]`, one per unique (file, page) pair

### What this phase does NOT produce

- An interactive REPL / multi-turn loop (single `--query` per invocation)
- Re-ranking or hybrid search
- An evaluation harness (Phase 5)
- Streaming output
- Any changes to ingestion

## Decisions

| Decision | Choice | Reason |
|---|---|---|
| Source deduplication | Unique (file + page) pairs | Clean output; multiple chunks from the same page collapse into one citation |
| Prompt constraint | Hard — model must say it lacks information if context is insufficient | Core grounding guarantee; prevents silent hallucination from training data |
| Citation format | `[Source: file.pdf, p. 42]` | Human-readable and actionable (open the PDF to that page) |
| Chunk index in citation | Not shown | Internal artifact with no meaning to a human reader |
| top_k | Configured in `config.yaml` as `top_k` (default: 5) | Tunable without code changes |

## New config key

Add to `config.yaml`:

```yaml
top_k: 5
```

`src/config.py` must add `"top_k"` to the required-keys list.

## Prompt template

```
You are a helpful assistant answering questions about AI and machine learning.
Use only the context below to answer. If the answer is not in the context, say exactly:
"I don't have enough information in the retrieved passages to answer that."

Context:
{context}

Question: {question}

Answer:
```

## Out of scope

- Streaming output
- Interactive REPL / multi-turn conversation
- Re-ranking retrieved chunks
- Filtering by source file or page range
- Hybrid search (keyword + vector)
