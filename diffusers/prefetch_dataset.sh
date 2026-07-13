#!/usr/bin/env bash
# Run this ON THE ADROIT LOGIN NODE (which has general internet). It downloads the HF
# dataset into HF_HOME on scratch so the compute node — where the proxy BLOCKS HuggingFace —
# can train fully offline. This is the diffusion-week version of last week's CIFAR pre-cache.
set -eo pipefail

ENV_NAME="${ENV_NAME:-diffusers}"
DATASET="${DATASET:-huggan/smithsonian_butterflies_subset}"
export HF_HOME="${HF_HOME:-/scratch/network/$USER/hf_cache}"

module load anaconda3/2024.6
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "$ENV_NAME"; set -u

mkdir -p "$HF_HOME"
echo "==> caching $DATASET into $HF_HOME (login node, open internet)"
python - <<PY
import os
from datasets import load_dataset
ds = load_dataset(os.environ.get("DATASET", "$DATASET"), split="train")
print("cached", len(ds), "images. columns:", ds.column_names)
PY
echo "==> done. On the compute node, set HF_HOME=$HF_HOME and HF_HUB_OFFLINE=1"
