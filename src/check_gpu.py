import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config


def _run_ollama_ps() -> str:
    result = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ollama ps failed: {result.stderr.strip()}")
        sys.exit(1)
    return result.stdout


_PROCESSOR_RE = re.compile(r"[\d%/]+\s+(?:CPU|GPU)(?:/(?:CPU|GPU))?")


def _parse_ollama_ps(output: str) -> list[dict]:
    lines = output.strip().splitlines()
    if len(lines) < 2:
        return []

    header = lines[0]
    proc_start = header.index("PROCESSOR")

    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        name = line.split()[0]
        # Use a regex on the processor region rather than slicing to the next
        # column header — Ollama added a context-size column between PROCESSOR
        # and UNTIL that shifts the boundary.
        match = _PROCESSOR_RE.search(line[proc_start:])
        processor = match.group(0).strip() if match else line[proc_start:].split()[0]
        rows.append({"name": name, "processor": processor})

    return rows


def _check(model_name: str | None) -> tuple[str, str, bool]:
    output = _run_ollama_ps()
    rows = _parse_ollama_ps(output)

    if not rows:
        print("No model currently loaded in Ollama.")
        print("Start inference first with: ollama run <model>")
        sys.exit(1)

    if model_name:
        match = next((r for r in rows if r["name"].startswith(model_name)), None)
        if match is None:
            loaded = ", ".join(r["name"] for r in rows)
            print(f"Model '{model_name}' is not loaded. Currently loaded: {loaded}")
            sys.exit(1)
        row = match
    else:
        row = rows[0]

    name = row["name"]
    processor = row["processor"]
    passed = "GPU" in processor.upper()
    status = "PASS" if passed else "FAIL"

    print(f"Model:     {name}")
    print(f"Processor: {processor}")
    print(f"Status:    {status}")

    return name, processor, passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Ollama GPU usage.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None, help="Model name to check (default: first loaded model)")
    parser.add_argument("--benchmark", action="store_true", help="Run a generation benchmark and record tokens/sec")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = args.model or cfg["generation_model"]

    name, processor, passed = _check(model_name)

    if args.benchmark:
        # Implemented in Group 3
        print("\n--benchmark not yet implemented")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
