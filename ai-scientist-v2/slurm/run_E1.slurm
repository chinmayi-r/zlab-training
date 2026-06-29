#!/bin/bash
# E1 -- tool-use hypothesis: "many failures are abandonment, not incapability".
# A/B partner of run_E2 / a baseline run. Only ONE variable changes vs baseline:
# agent.search.max_debug_depth 3->8 and debug_prob 0.5->0.9 (see configs/E1_debug8.yaml).
#
# VERIFY the #SBATCH lines against Adroit's current docs (partition name, gres
# syntax, undergrad time limits). These mirror the lab's nano-vllm setup.
#SBATCH --job-name=ai_sci_E1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1            # MIG A100 slice or V100; check Adroit's exact gres/constraint
#SBATCH --time=06:00:00         # runs take hours; check undergrad limits
#SBATCH --partition=mig         # verify partition name
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

module purge
module load proxy/default               # outbound internet + Sandbox access (COMPUTE node only)
module load anaconda3/2024.6 2>/dev/null || module load anaconda3
conda activate ai_scientist

source ~/.ai_scientist_secrets          # keys incl. AI_SANDBOX_KEY + USE_AI_SANDBOX

REPO="$HOME/AI-Scientist-v2"
cd "$REPO"

# Sanity before burning a job slot
curl -sSf https://api-ai-sandbox.princeton.edu >/dev/null 2>&1 \
  && echo "sandbox reachable" || echo "WARN: sandbox not reachable from this node"

# Use the E1 variant (generate it once with apply_experiment_config.py --preset E1 --small --sandbox)
cp configs/E1_debug8.yaml bfts_config.yaml

# Reuse the SAME idea JSON as the baseline so only the debug knobs differ.
python launch_scientist_bfts.py \
  --load_ideas "ai_scientist/ideas/topic_baseline.json" \
  --load_code \
  --add_dataset_ref \
  --model_writeup gpt-4o \
  --model_writeup_small gpt-4o-mini \
  --model_citation gpt-4o-mini \
  --model_review gpt-4o-mini \
  --model_agg_plots o3-mini \
  --num_cite_rounds 20
# All model flags use Sandbox model names. For pure tree-analysis you can add
# --skip_writeup --skip_review to stop before the (paid-elsewhere) writeup stage.

echo "DONE. Mine the tree:"
echo "  python scripts/mine_journal.py experiments/<run>/logs/0-run/ -o E1_nodes.csv"
