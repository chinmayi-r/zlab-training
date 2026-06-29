#!/bin/bash
# E1 -- tool-use hypothesis: "many failures are abandonment, not incapability".
# A/B partner of run_E2 / a baseline run. Only ONE variable changes vs baseline:
# agent.search.max_debug_depth 3->8 and debug_prob 0.5->0.9 (see configs/E1_debug8.yaml).
#
# #SBATCH lines verified against Adroit `snodes`/`sinfo` (June 2026): GPUs are in the
# 'gpu' partition; MIG A100 slices are gres `3g.20gb` (~20GB) on adroit-h11g2.
#SBATCH --job-name=ai_sci_E1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu                 # Adroit GPUs live here (NOT 'mig')
#SBATCH --gres=gpu:3g.20gb:1            # one MIG A100 slice (~20GB). Full A100: --gres=gpu:1 --constraint=a100
#SBATCH --time=06:00:00                 # gpu partition allows up to 7 days; keep runs small
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

module purge
module load proxy/default               # compute-node API allowlist (incl. the Sandbox)
module load "${ANACONDA_MODULE:-anaconda3/2024.6}"
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "${ENV_NAME:-ai_scientist}"; set -u   # set +u avoids PS1-unbound

source ~/.ai_scientist_secrets          # keys incl. AI_SANDBOX_KEY + USE_AI_SANDBOX

REPO="${REPO_DIR:-$HOME/AI-Scientist-v2}"
cd "$REPO"

# Real Sandbox reachability test from THIS compute node. A `curl /` returns 502 even
# when the Sandbox works (bare root isn't a valid endpoint) -- only a real call proves it.
python - <<'PY' || { echo "ABORT: Sandbox not reachable from compute node; ask RC to allowlist api-ai-sandbox.princeton.edu"; exit 1; }
import os
from openai import AzureOpenAI
c = AzureOpenAI(api_key=os.environ["AI_SANDBOX_KEY"],
                azure_endpoint=os.environ.get("SANDBOX_ENDPOINT","https://api-ai-sandbox.princeton.edu/"),
                api_version=os.environ.get("SANDBOX_API_VERSION","2025-03-01-preview"))
r = c.chat.completions.create(model="gpt-4o-mini", max_tokens=5,
        messages=[{"role":"user","content":"ok"}])
print("sandbox OK:", r.choices[0].message.content)
PY

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
