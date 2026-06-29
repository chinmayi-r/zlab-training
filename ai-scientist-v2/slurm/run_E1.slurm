#!/bin/bash
# E1 -- tool-use hypothesis: "many failures are abandonment, not incapability".
# Only ONE variable changes vs the E1 baseline: agent.search.max_debug_depth 3->8 and
# debug_prob 0.5->0.9 (configs/E1_debug8.yaml vs configs/E1_baseline.yaml).
# Backend: real OpenAI via the proxy (the Princeton Sandbox is currently unreachable).
#
# #SBATCH lines verified against Adroit snodes/sinfo (June 2026): GPUs are in 'gpu';
# MIG A100 slices are gres 3g.20gb (~20GB) on adroit-h11g2.
#SBATCH --job-name=ai_sci_E1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu                 # Adroit GPUs live here (NOT 'mig')
#SBATCH --gres=gpu:3g.20gb:1            # one MIG A100 slice (~20GB). Full A100: --gres=gpu:1 --constraint=a100
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

set -eo pipefail
module purge
module load proxy/default               # compute-node API allowlist (reaches OpenAI)
module load "${ANACONDA_MODULE:-anaconda3/2024.6}"
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "${ENV_NAME:-ai_scientist}"; set -u

source ~/.ai_scientist_secrets
unset USE_AI_SANDBOX OPENAI_BASE_URL    # force the real OpenAI endpoint + key

REPO="${REPO_DIR:-$HOME/AI-Scientist-v2}"
cd "$REPO"
export CIFAR_DIR="${CIFAR_DIR:-/scratch/network/$USER/data}"

# Pre-flight: confirm OpenAI is reachable from this node before burning the slot.
python - <<'PY' || { echo "ABORT: OpenAI unreachable from compute node"; exit 1; }
from openai import OpenAI
print("openai OK:", OpenAI().chat.completions.create(model="gpt-4o-mini", max_tokens=5,
        messages=[{"role":"user","content":"ok"}]).choices[0].message.content)
PY

# Pick the variant. Default = the deep-debug arm; set BASELINE=1 for the A side.
CFG="configs/E1_debug8.yaml"; [ "${BASELINE:-0}" = "1" ] && CFG="configs/E1_baseline.yaml"
echo "using config: $CFG"
cp "$CFG" bfts_config.yaml

# Same idea for both arms so only the debug knobs differ. Skip writeup/review:
# Semantic Scholar (citations) is proxy-blocked, and we only need the experiment tree.
python launch_scientist_bfts.py \
  --load_ideas "ai_scientist/ideas/topic_baseline.json" \
  --load_code \
  --skip_writeup --skip_review

echo "DONE. Mine the tree:"
echo "  python <kit>/scripts/mine_journal.py experiments/<run>/logs/0-run/ -o E1_nodes.csv"
