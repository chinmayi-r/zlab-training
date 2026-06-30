#!/usr/bin/env bash
# Launch one AI-Scientist-v2 experiment locally (WSL2 + GPU). No SLURM, no proxy, no
# Sandbox -- open internet + your OpenAI key. Run from the kit dir after setup_local.sh.
set -eo pipefail

: "${OPENAI_API_KEY:?export your OpenAI key first: export OPENAI_API_KEY=sk-...}"
REPO_DIR="${REPO_DIR:-$HOME/AI-Scientist-v2}"
ENV_NAME="${ENV_NAME:-ai_scientist}"
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the ai-scientist-v2 kit dir
export CIFAR_DIR="${CIFAR_DIR:-$HOME/data}"              # shared dataset cache across nodes

CONDA_BASE="$($HOME/miniconda3/bin/conda info --base 2>/dev/null || conda info --base 2>/dev/null)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$REPO_DIR"

# Pre-cache CIFAR once (open internet here) so the 14 nodes don't each re-download.
python -c "import torchvision as tv,os; r=os.environ['CIFAR_DIR']; tv.datasets.CIFAR10(r,train=True,download=True); tv.datasets.CIFAR10(r,train=False,download=True); print('CIFAR cached at',r)"

# Idea (default = the fast/GPU-explicit smoke idea so nodes finish in seconds, not hours;
# override with IDEA_FILE=topic_concrete_local.json for the full 30-epoch protocol).
IDEA_FILE="${IDEA_FILE:-topic_concrete_fast.json}"
cp "$KIT/ideas/$IDEA_FILE" ai_scientist/ideas/

# Cheap config (gpt-4o coder; switch all to gpt-4o-mini to cut cost to ~cents).
COMMON="--set agent.code.model=gpt-4o --set agent.feedback.model=gpt-4o-mini --set agent.vlm_feedback.model=gpt-4o-mini --set report.model=gpt-4o-mini --set generate_report=false"
python "$KIT/scripts/apply_experiment_config.py" --config bfts_config.yaml --preset baseline --small $COMMON --out /tmp/bfts_local.yaml
cp /tmp/bfts_local.yaml bfts_config.yaml

python launch_scientist_bfts.py \
  --load_ideas "ai_scientist/ideas/$IDEA_FILE" \
  --skip_writeup --skip_review

echo
echo "DONE. Mine the newest run:"
echo "  RUN=\$(ls -dt $REPO_DIR/experiments/*/ | head -1)"
echo "  python $KIT/scripts/mine_journal.py \"\$RUN/logs/0-run/\" -o local_nodes.csv"
