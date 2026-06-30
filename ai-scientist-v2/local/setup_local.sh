#!/usr/bin/env bash
# Local setup for AI-Scientist-v2 on Windows via WSL2 (Ubuntu) + NVIDIA GPU (RTX 3050).
# Run INSIDE the WSL Ubuntu shell. See local/README.md for the Windows prep (wsl --install
# + NVIDIA Windows driver). The torch cu124 wheels bundle CUDA; no CUDA toolkit needed.
set -eo pipefail

REPO_DIR="${REPO_DIR:-$HOME/AI-Scientist-v2}"
ENV_NAME="${ENV_NAME:-ai_scientist}"

echo "==> system libs (poppler/chktex are optional; used only by writeup)"
sudo apt-get update -y && sudo apt-get install -y git wget poppler-utils chktex || true

echo "==> miniconda"
if ! command -v conda >/dev/null 2>&1; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$HOME/miniconda3"
fi
source "$($HOME/miniconda3/bin/conda info --base 2>/dev/null || conda info --base)/etc/profile.d/conda.sh"

echo "==> clone repo into $REPO_DIR"
[ -d "$REPO_DIR" ] || git clone https://github.com/SakanaAI/AI-Scientist-v2.git "$REPO_DIR"

echo "==> conda env + torch (cu124 wheels) + requirements"
conda create -n "$ENV_NAME" python=3.11 -y
conda activate "$ENV_NAME"
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
cd "$REPO_DIR"
pip install -r requirements.txt
pip install "ruamel.yaml" "sympy==1.13.1"

echo "==> GPU check"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("WARNING: GPU not visible to WSL. Update the NVIDIA *Windows* driver, then run")
    print("         'wsl --shutdown' in PowerShell and reopen Ubuntu.")
PY

echo
echo "Done. Next:"
echo "  export OPENAI_API_KEY=sk-...    # your key"
echo "  bash local/run_local.sh"
