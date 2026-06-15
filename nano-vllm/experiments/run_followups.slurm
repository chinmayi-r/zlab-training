#!/bin/bash
#SBATCH --job-name=nanovllm-followups
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodelist=adroit-h11g2
#SBATCH --output=followups_%j.out

module load anaconda3/2024.2
conda activate nanovllm
cd "$SLURM_SUBMIT_DIR"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=${NANOVLLM_SRC:-/scratch/network/cr7998/nano-vllm}
MODEL=${MODEL:-/scratch/network/cr7998/hf/Qwen3-0.6B}
NS="4 8 16 32 64 96"
UTIL=0.30

nvidia-smi --query-gpu=name,compute_cap --format=csv || true

echo "=== A1. admission=optimistic (baseline) ==="
python exp_preempt.py --model "$MODEL" --n $NS --gpu-mem-util $UTIL \
    --admission optimistic --out admit_optimistic.csv

echo "=== A2. admission=conservative (zero preemption, may queue) ==="
python exp_preempt.py --model "$MODEL" --n $NS --gpu-mem-util $UTIL \
    --admission conservative --out admit_conservative.csv

echo "=== B. policy=cheapest-to-redo (compare vs optimistic baseline = LIFO) ==="
python exp_preempt.py --model "$MODEL" --n $NS --gpu-mem-util $UTIL \
    --policy cheapest --out policy_cheapest.csv

echo "=== C1. variable-length, policy=lifo ==="
python exp_varlen.py --model "$MODEL" --n 64 --gpu-mem-util $UTIL \
    --policy lifo --out varlen_lifo.csv

echo "=== C2. variable-length, policy=cheapest ==="
python exp_varlen.py --model "$MODEL" --n 64 --gpu-mem-util $UTIL \
    --policy cheapest --out varlen_cheapest.csv

echo "=== DONE. CSVs: admit_optimistic admit_conservative policy_cheapest varlen_lifo varlen_cheapest ==="
