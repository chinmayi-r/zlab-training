#!/usr/bin/env bash
# Train the small butterflies DDPM locally (WSL2 + GPU). Open internet, so the HF dataset
# downloads normally. Run from this diffusers/ dir after setup_local.sh.
set -eo pipefail

ENV_NAME="${ENV_NAME:-diffusers}"
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_BASE="$($HOME/miniconda3/bin/conda info --base 2>/dev/null || conda info --base 2>/dev/null)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Cache HF data/models under the repo so re-runs don't re-download.
export HF_HOME="${HF_HOME:-$KIT/hf_cache}"

# Fast/small defaults for a 4 GB RTX 3050. Override any via env, e.g. IMAGE_SIZE=64 EPOCHS=50.
python "$KIT/train_ddpm.py" \
  --dataset "${DATASET:-huggan/smithsonian_butterflies_subset}" \
  --image_size "${IMAGE_SIZE:-32}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --epochs "${EPOCHS:-10}" \
  --sample_every "${SAMPLE_EVERY:-5}" \
  --out_dir "${OUT_DIR:-$KIT/ddpm-out}"

echo
echo "DONE. Trained pipeline + sample grids are in ${OUT_DIR:-$KIT/ddpm-out}/"
echo "Open samples_final.png to see the generated butterflies."
