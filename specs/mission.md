# Mission

Build a personal, fully local RAG system for querying AI-related books using GPU-accelerated inference — no cloud, no API keys, no data leaving the machine.

## Problem

Reading and retaining dense technical books (ML papers, AI textbooks) is slow. Finding answers across hundreds of pages requires manual search. Cloud-based solutions (ChatGPT, Perplexity) require sharing private reading material with external services.

## Solution

A local RAG pipeline that ingests any set of PDFs, embeds them using a local model, and answers natural-language questions by retrieving relevant passages and generating grounded responses — entirely on local hardware via Ollama + GPU.

## Goals

- **Privacy**: all inference runs locally; no data leaves the machine
- **Speed**: GPU-accelerated embedding and generation
- **Accuracy**: retrieval quality that surfaces the right passage, not just keyword matches
- **Learning**: the system itself is a testbed for understanding RAG internals

## Non-goals

- Multi-user support or cloud deployment
- External / REST API, or any non-local hosting — Phase 5 adds a **localhost-only** chat UI (Gradio, `share=False`); nothing is exposed off the machine
- Model training of any kind (RLHF, DPO, LoRA) — Phase 5 collects answer feedback into a local store, but training on it is out of scope
- Real-time document streaming or web scraping
- Support for languages other than English (primary corpus is English AI literature)

## Primary User

Solo use — the developer is the only user. Optimized for a single workstation with a dedicated GPU running Windows with Ollama. Interaction is via the CLI (`src/query.py`) and, from Phase 5, a local browser chat UI (`src/app.py`) bound to `127.0.0.1`.

## Success Criteria

1. Can ingest a directory of AI PDFs in one command
2. GPU is used for both embedding and generation (verified via `ollama ps`)
3. Answers cite the source book and page/chunk
4. A baseline evaluation set measures retrieval quality (hit rate ≥ 0.7 on 20 test questions)
