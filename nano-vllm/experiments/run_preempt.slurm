#!/bin/bash
#SBATCH --job-name=nanovllm-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=mig
#SBATCH --output=preempt_%j.out

module load anaconda3/2024.2
conda activate nanovllm
cd "$SLURM_SUBMIT_DIR"

# Log which GPU we landed on. flash-attn needs compute capability >= 8.0
# (A100/Ampere or newer); older cards (V100=7.0, T4=7.5) fail with
# "FlashAttention only supports Ampere GPUs or newer".
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv || true
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))" || true

# Compute nodes are offline; don't let HF phone home.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# nano-vllm source on path (the conda env was moved, so the editable install / pip
# console scripts are unreliable — PYTHONPATH avoids needing the install at all).
export PYTHONPATH=${NANOVLLM_SRC:-/scratch/network/cr7998/nano-vllm}

MODEL=${MODEL:-/scratch/network/cr7998/hf/Qwen3-0.6B}

# Sweep N under tight memory so preemption is forced. If preempt=0 everywhere,
# lower --gpu-mem-util (e.g. 0.20) and resubmit.
python exp_preempt.py \
    --model "$MODEL" \
    --n 4 8 16 32 64 96 \
    --gpu-mem-util 0.30 \
    --prompt-tokens 600 \
    --max-tokens 600 \
    --out results.csv \
    --verbose
