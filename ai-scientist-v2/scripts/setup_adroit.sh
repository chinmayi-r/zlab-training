#!/usr/bin/env bash
# setup_adroit.sh - one-time environment setup for AI-Scientist-v2 on Adroit.
#
# RUN THIS ON THE LOGIN NODE (adroit-vis).  <-- verified June 2026
# On Adroit the LOGIN node has general outbound internet (github/PyPI/conda work);
# COMPUTE nodes do NOT -- `module load proxy/default` there only opens a small
# pre-approved API allowlist (github clone gets 403). So clone + build the env on
# the login node; only the GPU *run* goes on a compute node (see slurm/).
#
# Quota note: a conda env + the repo is several GB. /home is small (~20GB). Put both
# on scratch:   export REPO_DIR=/scratch/network/$USER/AI-Scientist-v2
#               export CONDA_ENVS_DIR=/scratch/network/$USER/conda-envs
set -eo pipefail   # NOTE: no `-u` -- conda's activate script references unset $PS1

REPO_DIR="${REPO_DIR:-$HOME/AI-Scientist-v2}"
ENV_NAME="${ENV_NAME:-ai_scientist}"
ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda3/2024.6}"   # `module avail anaconda3` for options
CONDA_ENVS_DIR="${CONDA_ENVS_DIR:-}"

echo "==> 1. Sanity: can we reach github (what we actually need here)?"
if git ls-remote https://github.com/SakanaAI/AI-Scientist-v2.git -h HEAD >/dev/null 2>&1; then
  echo "    github reachable -- good, you're on a node with general internet (login)."
else
  echo "    github NOT reachable. You are probably on a COMPUTE node." >&2
  echo "    Run this on the LOGIN node (adroit-vis) instead." >&2
  exit 1
fi

echo "==> 2. Clone the repo into $REPO_DIR (the ONLY source of truth for flags/keys)"
if [[ ! -d "$REPO_DIR" ]]; then
  git clone https://github.com/SakanaAI/AI-Scientist-v2.git "$REPO_DIR"
else
  echo "    already present at $REPO_DIR"
fi

echo "==> 3. Load conda ($ANACONDA_MODULE)"
module load "$ANACONDA_MODULE"
# Make `conda activate` work in a non-interactive shell, and survive set -e.
source "$(conda info --base)/etc/profile.d/conda.sh"
if [[ -n "$CONDA_ENVS_DIR" ]]; then
  mkdir -p "$CONDA_ENVS_DIR"
  conda config --add envs_dirs "$CONDA_ENVS_DIR"
  echo "    envs will live in $CONDA_ENVS_DIR"
fi

echo "==> 4. Create + activate env '$ENV_NAME' (per README)"
conda create -n "$ENV_NAME" python=3.11 -y
conda activate "$ENV_NAME"   # top of script omits `set -u`, so conda's $PS1 ref is safe

# PyTorch via the self-contained pip cu124 wheels, NOT conda. The conda pytorch build
# pulls MKL 2025 (breaks `import torch`: undefined symbol iJIT_NotifyEvent), and even
# pinning mkl=2024 leaves torchvision ABI-broken (torchvision::nms does not exist). The
# pip wheels bundle their own CUDA/MKL libs and are mutually ABI-matched -> both fixed.
pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
conda install anaconda::poppler -y          # PDF tools
conda install conda-forge::chktex -y        # LaTeX lint

echo "==> 5. Python requirements"
cd "$REPO_DIR"
pip install -r requirements.txt
pip install "sympy==1.13.1"                  # torch 2.5.1 pins this; requirements pulls 1.14
pip install "ruamel.yaml" || true           # keeps comments when generating config variants
# Only if you keep the DEFAULT Bedrock-Claude experiment model (NOT the $0 Sandbox route):
# pip install "anthropic[bedrock]"

echo "==> 6. GPU check (will say False on the login node -- that's fine, real check is in the job)"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
PY

echo
echo "Done (login node). Next:"
echo "  1. cp scripts/sandbox_secrets.env.example ~/.ai_scientist_secrets  (chmod 600, fill, NEVER git-add)"
echo "  2. python scripts/wire_sandbox.py --repo $REPO_DIR"
echo "  3. Pre-fetch datasets here on the login node (compute nodes can't download) -- see experiments.md"
echo "  4. Verify the Sandbox is reachable FROM A COMPUTE NODE with the AzureOpenAI smoke test"
echo "     (a curl to '/' returns 502 even when it works -- only a real call proves it)."
