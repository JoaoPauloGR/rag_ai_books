# GPU Setup — Windows 11 + NVIDIA + Ollama

## Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| NVIDIA driver | 452.39 | Adds CUDA 11.0 support. Check: `nvidia-smi` — version shown top-right |
| Windows 11 | Any | No WSL required; Ollama runs natively |
| VRAM | 4 GB | Enough for `llama3.2:3b` (Q4, ~2 GB) + `nomic-embed-text` (~274 MB) |
| Ollama | Latest | Download from https://ollama.com/download — installs as a Windows service |

Ollama ships with its own CUDA runtime. **No separate CUDA Toolkit install is needed.**

---

## Step 1 — Verify the NVIDIA driver

```powershell
nvidia-smi
```

Look for:
- CUDA Version (top-right corner) — must be 11.0 or higher
- GPU name and VRAM size

If `nvidia-smi` is not found, the driver is missing or not on PATH. Install the latest Game Ready or Studio Driver from https://www.nvidia.com/drivers.

---

## Step 2 — Install Ollama

1. Download the Windows installer from https://ollama.com/download
2. Run the installer — Ollama starts as a background service automatically
3. Verify it's running:

```powershell
ollama --version
```

Ollama logs appear in `%LOCALAPPDATA%\Ollama\`. If Ollama detected the GPU at startup, the log will contain a line like:

```
msg="inference compute" id=GPU-... library=cuda compute=8.6 driver=12.x name="NVIDIA GeForce ..."
```

If you see `library=cpu` instead, see the Troubleshooting section below.

---

## Step 3 — Pull the project models

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

Check download sizes: `nomic-embed-text` is ~274 MB; `llama3.2:3b` is ~2 GB.

---

## Step 4 — Confirm GPU layers during inference

Open two terminals.

**Terminal 1** — start a model:
```powershell
ollama run llama3.2:3b
```

**Terminal 2** — inspect the running model:
```powershell
ollama ps
```

Expected output:
```
NAME            ID              SIZE      PROCESSOR    UNTIL
llama3.2:3b     ...             2.0 GB    100% GPU     4 minutes from now
```

The `PROCESSOR` column must contain `GPU`. If it shows `100% CPU`, the GPU is not being used — see Troubleshooting.

---

## Troubleshooting

### `ollama ps` shows `100% CPU`

Ollama fell back to CPU. Most common causes:

**1. Driver too old**  
`nvidia-smi` shows CUDA Version below 11.0. Update the NVIDIA driver.

**2. Ollama started before the driver was installed**  
Restart the Ollama service:
```powershell
Stop-Process -Name "ollama" -Force
ollama serve
```
Or restart it from the system tray icon.

**3. Not enough VRAM for the model**  
`llama3.2:3b` (Q4) needs ~2 GB of VRAM free. Check `nvidia-smi` for free VRAM. Close other GPU-heavy apps. If VRAM is consistently too low, reduce `OLLAMA_NUM_GPU_LAYERS` to split the model between GPU and CPU:
```powershell
$env:OLLAMA_NUM_GPU_LAYERS = 20   # partial offload
ollama serve
```

**4. Multiple GPUs / integrated graphics**  
Ollama may pick the wrong GPU. Set explicitly:
```powershell
$env:CUDA_VISIBLE_DEVICES = 0    # use first GPU
ollama serve
```

### `nvidia-smi` is not found

The driver is installed but not on PATH, or not installed at all.  
Add `C:\Windows\System32` to PATH (it's usually already there), or reinstall the NVIDIA driver.
