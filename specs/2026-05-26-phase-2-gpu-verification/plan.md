# Phase 2 — Implementation Plan

## Group 1 — `docs/` directory and setup guide

1. Create `docs/` directory with a `.gitkeep` placeholder (if the directory does not exist)
2. Write `docs/gpu-setup.md` covering:
   - Prerequisites: NVIDIA driver version, CUDA toolkit, Ollama install on Windows
   - How to verify CUDA is visible to Ollama (`ollama serve` logs, `nvidia-smi`)
   - How to pull the models used in this project (`ollama pull nomic-embed-text`, `ollama pull llama3.2:3b`)
   - How to confirm GPU layers > 0: run `ollama run llama3.2:3b` in one terminal, then `ollama ps` in another — the `PROCESSOR` column must show `GPU`
   - Troubleshooting: common reasons Ollama falls back to CPU (missing CUDA libs, driver mismatch, VRAM too low)

## Group 2 — `src/check_gpu.py` core

3. Add argparse with two flags: `--model` (default: read from `config.yaml`) and `--benchmark`
4. Call `ollama ps` via `subprocess.run` and capture stdout
5. Parse the output table: extract model name and the `PROCESSOR` column value
6. If no model is loaded (empty table), print a clear message: `No model currently loaded in Ollama. Start inference first with: ollama run <model>` and exit 1
7. Print a status summary:
   ```
   Model:     llama3.2:3b
   Processor: 100% GPU
   Status:    PASS
   ```
   Set status to `FAIL` (and exit 1) if the PROCESSOR value contains `CPU` with no GPU component

## Group 3 — Benchmark mode

8. When `--benchmark` is passed, run `ollama.generate(model=..., prompt="Explain what a neural network is in one sentence.")` and capture the response dict
9. Compute `tokens_per_sec = response["eval_count"] / (response["eval_duration"] / 1e9)`
10. Print the result to stdout:
    ```
    Benchmark: 42.3 tokens/sec  (eval_count=38, eval_duration=0.898s)
    ```
11. Append a row to `docs/gpu_benchmarks.csv`; create the file with a header row if it does not exist:
    ```
    timestamp,model,processor,tokens_per_sec,eval_count,eval_duration_s
    ```
    `processor` value comes from the `ollama ps` parse in Group 2 (run the check before the benchmark)
