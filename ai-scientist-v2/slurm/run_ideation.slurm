#!/bin/bash
# run_ideation.slurm - generate idea JSONs from the topic .md files. Ideation calls
# the LLM API, so it must run on a compute node with proxy/default (no GPU needed).
# Prereq (do once on the LOGIN node): copy the topic files into the repo:
#   cp <kit>/ideas/topic_baseline.md <kit>/ideas/topic_concrete.md $REPO_DIR/ai_scientist/ideas/
#   sbatch slurm/run_ideation.slurm   &&  tail -f ideation-<jobid>.out
#SBATCH --job-name=ideation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=ideation-%j.out
#SBATCH --error=ideation-%j.out

module purge
module load proxy/default
module load "${ANACONDA_MODULE:-anaconda3/2024.6}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-ai_scientist}"
source ~/.ai_scientist_secrets
unset USE_AI_SANDBOX OPENAI_BASE_URL          # real OpenAI

cd "${REPO_DIR:-$HOME/AI-Scientist-v2}"

for t in topic_baseline topic_concrete; do
  echo "=== ideating: $t ==="
  python ai_scientist/perform_ideation_temp_free.py \
    --workshop-file "ai_scientist/ideas/$t.md" \
    --model gpt-4o-mini \
    --max-num-generations 6 \
    --num-reflections 3
done
echo "DONE. Check: ls -la ai_scientist/ideas/topic_*.json"
