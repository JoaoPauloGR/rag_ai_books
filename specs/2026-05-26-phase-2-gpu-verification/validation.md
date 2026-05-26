# Phase 2 — Validation

## Done when

All of the following pass before merging.

### 1. GPU check passes with a running model

Start a model in one terminal:
```
ollama run llama3.2:3b
```

In another terminal:
```
python src/check_gpu.py
```

Expected output:
```
Model:     llama3.2:3b
Processor: 100% GPU
Status:    PASS
```

The script must exit 0.

### 2. GPU check fails gracefully with no model loaded

With no model running (or after `ollama stop`):
```
python src/check_gpu.py
```

Expected: a clear message like `No model currently loaded in Ollama.` and exit 1.  
Must not raise an unhandled exception.

### 3. Benchmark appends a row to `docs/gpu_benchmarks.csv`

With a model running:
```
python src/check_gpu.py --benchmark
```

Expected:
- Tokens/sec printed to stdout
- `docs/gpu_benchmarks.csv` is created (or updated) with one new row
- Row contains: timestamp, model name, processor value, tokens/sec, eval_count, eval_duration_s

Run a second time and confirm a second row is appended (not overwritten).

### 4. CPU vs GPU rows are distinguishable in the CSV

Disable GPU (e.g., by setting `OLLAMA_NUM_GPU_LAYERS=0` in the Ollama environment) and run:
```
python src/check_gpu.py --benchmark
```

Expected:
- Status prints `FAIL` (processor shows CPU)
- A new row in `docs/gpu_benchmarks.csv` with `processor=100% CPU` and a lower `tokens_per_sec` than the GPU row

### 5. `docs/gpu-setup.md` is complete

The document must cover:
- NVIDIA driver / CUDA requirement
- Ollama install steps
- How to confirm `ollama ps` shows GPU during inference
- At least one troubleshooting entry

## Not required for merge

- Actual PDF ingestion or ChromaDB connectivity
- `ingest.py` or `query.py` invoking `check_gpu.py` automatically
- Embedding model benchmark (generation only)
