#!/bin/bash
# E2 -- task-spec hypothesis: "the failure is the idea, not the agent".
# The ONLY variable that changes vs the E2 baseline is the idea .md: topic_concrete.md
# pins dataset + model size + exact metric + compute budget; topic_baseline.md is vague.
# Same config (configs/E2_baseline.yaml), same seed. Doubles as the CUDA-OOM fix.
#
# #SBATCH lines verified against Adroit `snodes`/`sinfo` (June 2026).
#SBATCH --job-name=ai_sci_E2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu                 # Adroit GPUs live here (NOT 'mig')
#SBATCH --gres=gpu:3g.20gb:1            # one MIG A100 slice (~20GB). Full A100: --gres=gpu:1 --constraint=a100
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

module purge
module load proxy/default                  # compute-node API allowlist (incl. the Sandbox)
module load "${ANACONDA_MODULE:-anaconda3/2024.6}"
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "${ENV_NAME:-ai_scientist}"; set -u   # set +u avoids PS1-unbound

source ~/.ai_scientist_secrets             # keys incl. AI_SANDBOX_KEY + USE_AI_SANDBOX

REPO="${REPO_DIR:-$HOME/AI-Scientist-v2}"
cd "$REPO"

# Real Sandbox check from this compute node (curl '/' gives a misleading 502).
python - <<'PY' || { echo "ABORT: Sandbox unreachable from compute node; ask RC to allowlist it"; exit 1; }
import os
from openai import AzureOpenAI
c = AzureOpenAI(api_key=os.environ["AI_SANDBOX_KEY"],
                azure_endpoint=os.environ.get("SANDBOX_ENDPOINT","https://api-ai-sandbox.princeton.edu/"),
                api_version=os.environ.get("SANDBOX_API_VERSION","2025-03-01-preview"))
print("sandbox OK:", c.chat.completions.create(model="gpt-4o-mini", max_tokens=5,
        messages=[{"role":"user","content":"ok"}]).choices[0].message.content)
PY

# Datasets must be pre-fetched on the LOGIN node -- compute nodes can't download.
# See experiments.md step 4b. (CIFAR-10 example for topic_concrete.)
export CIFAR_DIR="${CIFAR_DIR:-/scratch/network/$USER/data}"

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
