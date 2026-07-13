#!/usr/bin/env bash
# Launch the 2D-diffusion smoke study locally (WSL2). Toy datasets are generated in-code
# (moons/circle/line) -- NO download, so this works with or without internet and needs no
# GPU (the MLP denoiser is tiny). Run from the kit dir after local/setup_local.sh.
set -eo pipefail

: "${OPENAI_API_KEY:?export your OpenAI key first: export OPENAI_API_KEY=sk-...}"
REPO_DIR="${REPO_DIR:-$HOME/AI-Scientist-v2}"
ENV_NAME="${ENV_NAME:-ai_scientist}"
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the ai-scientist-v2 kit dir

CONDA_BASE="$($HOME/miniconda3/bin/conda info --base 2>/dev/null || conda info --base 2>/dev/null)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$REPO_DIR"

# No dataset pre-cache needed -- the idea generates its 2D data procedurally.
IDEA_FILE="${IDEA_FILE:-topic_diffusion_fast.json}"
cp "$KIT/ideas/$IDEA_FILE" ai_scientist/ideas/

# Cheap config (gpt-4o coder; switch all to gpt-4o-mini to cut cost to ~cents).
# exec.timeout=120 -> a hung node fails fast instead of blocking.
COMMON="--set agent.code.model=gpt-4o --set agent.feedback.model=gpt-4o-mini --set agent.vlm_feedback.model=gpt-4o-mini --set report.model=gpt-4o-mini --set generate_report=false --set exec.timeout=120"
python "$KIT/scripts/apply_experiment_config.py" --config bfts_config.yaml --preset baseline --small $COMMON --out /tmp/bfts_diffusion.yaml
cp /tmp/bfts_diffusion.yaml bfts_config.yaml

python launch_scientist_bfts.py \
  --load_ideas "ai_scientist/ideas/$IDEA_FILE" \
  --skip_writeup --skip_review

echo
echo "DONE. Mine the newest run:"
echo "  RUN=\$(ls -dt $REPO_DIR/experiments/*/ | head -1)"
echo "  python $KIT/scripts/mine_journal.py \"\$RUN/logs/0-run/\" -o diffusion_nodes.csv"
