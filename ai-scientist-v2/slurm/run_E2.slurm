#!/bin/bash
# E2 -- task-spec hypothesis: "the failure is the idea, not the agent".
# The ONLY variable that changes is the idea .md: topic_concrete (pinned) vs
# topic_baseline (vague). Same config (configs/E2_baseline.yaml) for both arms.
# Backend: real OpenAI via the proxy.
#SBATCH --job-name=ai_sci_E2
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
module load proxy/default
module load "${ANACONDA_MODULE:-anaconda3/2024.6}"
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "${ENV_NAME:-ai_scientist}"; set -u

source ~/.ai_scientist_secrets
unset USE_AI_SANDBOX OPENAI_BASE_URL

REPO="${REPO_DIR:-$HOME/AI-Scientist-v2}"
cd "$REPO"
export CIFAR_DIR="${CIFAR_DIR:-/scratch/network/$USER/data}"

python - <<'PY' || { echo "ABORT: OpenAI unreachable from compute node"; exit 1; }
from openai import OpenAI
print("openai OK:", OpenAI().chat.completions.create(model="gpt-4o-mini", max_tokens=5,
        messages=[{"role":"user","content":"ok"}]).choices[0].message.content)
PY

cp configs/E2_baseline.yaml bfts_config.yaml

# Switch IDEA between the two arms (run twice; change only this):
#   A (vague)    : topic_baseline.json
#   B (concrete) : topic_concrete.json   <- start here (CIFAR pre-fetched, can actually run)
IDEA="${IDEA:-ai_scientist/ideas/topic_concrete.json}"
echo "using idea: $IDEA"

python launch_scientist_bfts.py \
  --load_ideas "$IDEA" \
  --skip_writeup --skip_review

echo "DONE. Compare nodes-to-first-working-node + #OOM/setup errors across the two arms."
