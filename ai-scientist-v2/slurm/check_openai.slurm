#!/bin/bash
# check_openai.slurm - confirm the real OpenAI API is reachable through Adroit's
# proxy with your key, BEFORE burning a real run. Needs OPENAI_API_KEY (real key)
# in ~/.ai_scientist_secrets, and USE_AI_SANDBOX off / OPENAI_BASE_URL unset.
#   sbatch slurm/check_openai.slurm  &&  ls -t oai-*.out | head -1 | xargs cat
#SBATCH --job-name=oai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=oai-%j.out
#SBATCH --error=oai-%j.out

module purge
module load proxy/default
module load "${ANACONDA_MODULE:-anaconda3/2024.6}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-ai_scientist}"
source ~/.ai_scientist_secrets
unset USE_AI_SANDBOX OPENAI_BASE_URL          # force the real OpenAI endpoint

echo "host: $(hostname)   https_proxy=${https_proxy:-<unset>}   base_url=${OPENAI_BASE_URL:-<default>}"
python - <<'PY'
import os
from openai import OpenAI
c = OpenAI()  # reads OPENAI_API_KEY, honors https_proxy automatically
for m in ["gpt-4o-mini", "gpt-4o"]:
    try:
        r = c.chat.completions.create(model=m, max_tokens=10,
                messages=[{"role":"user","content":"reply with the single word: ok"}])
        print(f"{m}: SUCCESS ->", r.choices[0].message.content)
    except Exception as e:
        print(f"{m}: FAIL ->", type(e).__name__, str(e)[:200])
PY
