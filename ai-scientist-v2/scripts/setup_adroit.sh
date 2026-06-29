#!/usr/bin/env bash
# setup_adroit.sh - one-time environment setup for AI-Scientist-v2 on Adroit.
# Commands mirror the repo README's Installation section verbatim (verified).
# Run this on a COMPUTE node (module load proxy/default first for outbound internet),
# NOT a login node.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/AI-Scientist-v2}"
ENV_NAME="${ENV_NAME:-ai_scientist}"

echo "==> 1. Sanity: outbound internet (needs 'module load proxy/default' on compute node)"
if curl -sSf https://api.openai.com/v1/models >/dev/null 2>&1; then
  echo "    internet OK"
else
  echo "    NO INTERNET -- run 'module load proxy/default' (compute node only), then re-run." >&2
  echo "    If it still fails, that's a real setup finding: document it and ask RC/Taiming." >&2
fi

echo "==> 2. Clone the repo (the ONLY source of truth for flags/keys)"
if [[ ! -d "$REPO_DIR" ]]; then
  git clone https://github.com/SakanaAI/AI-Scientist-v2.git "$REPO_DIR"
else
  echo "    already present at $REPO_DIR"
fi

echo "==> 3. Conda env (per README)"
# Load whatever anaconda module Adroit actually provides; adjust the name.
module load anaconda3/2024.6 2>/dev/null || module load anaconda3 2>/dev/null || true
conda create -n "$ENV_NAME" python=3.11 -y
# shellcheck disable=SC1091
source activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install anaconda::poppler -y          # PDF tools
conda install conda-forge::chktex -y        # LaTeX lint

echo "==> 4. Python requirements"
cd "$REPO_DIR"
pip install -r requirements.txt
# ruamel keeps comments when we generate config variants; harmless if already present.
pip install "ruamel.yaml" || true
# Only needed if you keep the DEFAULT experiment model (Bedrock Claude). The $0
# Sandbox route does NOT need this -- it uses AzureOpenAI via the openai package.
# pip install "anthropic[bedrock]"

echo "==> 5. GPU check"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
PY

echo
echo "Done. Next:"
echo "  1. cp scripts/sandbox_secrets.env.example ~/.ai_scientist_secrets  (chmod 600, fill in, NEVER git-add)"
echo "  2. python scripts/wire_sandbox.py --repo $REPO_DIR     # \$0 Sandbox route"
echo "  3. see experiments.md for ideation -> run -> mine"
