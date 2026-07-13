#!/bin/bash
#SBATCH --job-name=ddpm
#SBATCH --output=ddpm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3g.20gb:1
# One MIG A100 slice (~20 GB) is plenty for 64px DDPM. For a full A100: --gres=gpu:1 --constraint=a100
#
# Submit from the diffusers/ kit dir AFTER running prefetch_dataset.sh on the login node:
#   sbatch slurm/train_ddpm.slurm
set -eo pipefail

ENV_NAME="${ENV_NAME:-diffusers}"
export HF_HOME="${HF_HOME:-/scratch/network/$USER/hf_cache}"
export HF_HUB_OFFLINE=1          # dataset is pre-cached; compute node can't reach HF anyway
export DATASET="${DATASET:-huggan/smithsonian_butterflies_subset}"

module load anaconda3/2024.6
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "$ENV_NAME"; set -u

nvidia-smi -L || true
python train_ddpm.py \
  --dataset "$DATASET" \
  --image_size "${IMAGE_SIZE:-64}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --epochs "${EPOCHS:-50}" \
  --sample_every "${SAMPLE_EVERY:-10}" \
  --out_dir "${OUT_DIR:-ddpm-out}"

echo "DONE. Samples + pipeline in ${OUT_DIR:-ddpm-out}/"
