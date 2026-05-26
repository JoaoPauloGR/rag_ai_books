# Phase 2 — GPU Verification

## Context

Phase 1 wired config loading into the ingest/query stubs — model names are now read from `config.yaml` at runtime, but nothing actually runs yet. Phase 2 verifies that when Ollama does run inference, it uses the GPU. The learning goal is understanding GPU layer offloading, VRAM constraints, and tokens/sec as a measurement unit.

This phase produces no changes to `ingest.py` or `query.py`. It is a setup-verification tool and a baseline benchmark, not a pipeline feature.

## Scope

### What this phase produces

- `src/check_gpu.py` — standalone CLI script that inspects `ollama ps` output and reports whether the active model is running on GPU or CPU
- `--benchmark` flag on `check_gpu.py` — runs a short generation call, measures tokens/sec, and appends a timestamped row to `docs/gpu_benchmarks.csv`
- `docs/gpu_benchmarks.csv` — cumulative record of benchmark runs (created on first use)
- `docs/gpu-setup.md` — step-by-step documentation of the exact Windows 11 + CUDA + Ollama setup required to reach GPU layers > 0

### What this phase does NOT produce

- Any changes to `src/ingest.py` or `src/query.py`
- An automatic GPU check at startup of ingest/query
- CI or automated GPU validation

## Decisions

| Decision | Choice | Reason |
|---|---|---|
| Script placement | `src/check_gpu.py` | Consistent with other source files; importable if needed later |
| GPU detection method | Parse `ollama ps` stdout | Matches what the roadmap specifies; visible to the user with no extra deps |
| Benchmark trigger | `--benchmark` flag | Default run is fast (no inference needed); benchmark is opt-in |
| Results persistence | Append to `docs/gpu_benchmarks.csv` | Survives multiple runs; easy to compare CPU vs GPU rows side-by-side |
| Benchmark metric | `eval_count / (eval_duration / 1e9)` from `ollama.generate()` response | Native Ollama API fields; no external timing code needed |
| Docs scope | Exact setup only (Windows 11 + CUDA + Ollama) | Solo project; a generic multi-GPU guide would not be used |

## Out of scope

- Embedding speed benchmark (only generation tokens/sec is measured)
- Automatic remediation if GPU is not detected
- Config-driven GPU layer count (`OLLAMA_NUM_GPU_LAYERS` is documented, not automated)
