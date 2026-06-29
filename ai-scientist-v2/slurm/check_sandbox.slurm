#!/bin/bash
# check_sandbox.slurm - prove the AI Sandbox is reachable FROM A COMPUTE NODE.
# Submit from the login node:   sbatch check_sandbox.slurm
# Then:                         cat sbx-<jobid>.out
# Needs no GPU, so it schedules fast. The whole point is that `module load
# proxy/default` ONLY works inside a job (it aborts on login nodes), and that's
# the only route to api-ai-sandbox.princeton.edu.
#SBATCH --job-name=sbx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=sbx-%j.out
#SBATCH --error=sbx-%j.out

module purge
module load proxy/default                  # compute-node only; this is the route to the Sandbox
module load "${ANACONDA_MODULE:-anaconda3/2024.6}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-ai_scientist}"
source ~/.ai_scientist_secrets

echo "running on host: $(hostname)"        # should be a compute node, NOT adroit-vis
python - <<'PY'
import os
from openai import AzureOpenAI
c = AzureOpenAI(api_key=os.environ["AI_SANDBOX_KEY"],
                azure_endpoint=os.environ.get("SANDBOX_ENDPOINT","https://api-ai-sandbox.princeton.edu/"),
                api_version=os.environ.get("SANDBOX_API_VERSION","2025-03-01-preview"))
print("sandbox says:", c.chat.completions.create(model="gpt-4o-mini", max_tokens=10,
        messages=[{"role":"user","content":"reply with: ok"}]).choices[0].message.content)
PY
