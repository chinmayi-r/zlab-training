#!/usr/bin/env bash
# Local setup for the Diffusers DDPM report on Windows via WSL2 (Ubuntu) + NVIDIA GPU.
# Run INSIDE the WSL Ubuntu shell. Reuses the same miniconda + cu124 torch approach that
# worked for the AI-Scientist week. The torch cu124 wheels bundle CUDA; no toolkit needed.
set -eo pipefail

ENV_NAME="${ENV_NAME:-diffusers}"

echo "==> miniconda"
if ! command -v conda >/dev/null 2>&1; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$HOME/miniconda3"
fi
source "$($HOME/miniconda3/bin/conda info --base 2>/dev/null || conda info --base)/etc/profile.d/conda.sh"

# Accept default-channel ToS non-interactively (no-op if already accepted).
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

echo "==> conda env + torch (cu124 wheels) + diffusers stack"
conda create -n "$ENV_NAME" python=3.11 -y
conda activate "$ENV_NAME"
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install "diffusers[training]" "transformers" "datasets" "accelerate" "safetensors" "Pillow"

echo "==> GPU check"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("WARNING: GPU not visible to WSL. Update the NVIDIA *Windows* driver, then run")
    print("         'wsl --shutdown' in PowerShell and reopen Ubuntu. (CPU still works, just slow.)")
PY

echo
echo "Done. Next:"
echo "  bash run_local.sh          # trains the butterflies DDPM (downloads dataset on first run)"
