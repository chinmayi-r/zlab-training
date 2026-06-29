#!/bin/bash
# E2 -- task-spec hypothesis: "the failure is the idea, not the agent".
# The ONLY variable that changes vs the E2 baseline is the idea .md: topic_concrete.md
# pins dataset + model size + exact metric + compute budget; topic_baseline.md is vague.
# Same config (configs/E2_baseline.yaml), same seed. Doubles as the CUDA-OOM fix.
#
# VERIFY the #SBATCH lines against Adroit's current docs.
#SBATCH --job-name=ai_sci_E2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1            # MIG A100 slice or V100; verify
#SBATCH --time=06:00:00
#SBATCH --partition=mig         # verify partition name
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

module purge
module load proxy/default
module load anaconda3/2024.6 2>/dev/null || module load anaconda3
conda activate ai_scientist

source ~/.ai_scientist_secrets

REPO="$HOME/AI-Scientist-v2"
cd "$REPO"

cp configs/E2_baseline.yaml bfts_config.yaml

# Switch IDEA between the two arms of the A/B (run twice; change only this line):
#   A (vague)    : topic_baseline.json
#   B (concrete) : topic_concrete.json
IDEA="${IDEA:-ai_scientist/ideas/topic_concrete.json}"

python launch_scientist_bfts.py \
  --load_ideas "$IDEA" \
  --load_code \
  --add_dataset_ref \
  --model_writeup gpt-4o \
  --model_writeup_small gpt-4o-mini \
  --model_citation gpt-4o-mini \
  --model_review gpt-4o-mini \
  --model_agg_plots o3-mini \
  --num_cite_rounds 20

echo "DONE. Compare nodes-to-first-working-node + #OOM/setup errors across the two arms:"
echo "  python scripts/mine_journal.py experiments/<run>/logs/0-run/ -o E2_<arm>_nodes.csv"
