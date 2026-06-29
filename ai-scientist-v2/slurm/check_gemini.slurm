#!/bin/bash
# check_gemini.slurm - verify a REAL Gemini call works through Adroit's proxy.
# This mirrors exactly how AI-Scientist-v2's llm.py talks to Gemini (OpenAI SDK
# pointed at Google's OpenAI-compatible endpoint), so success here means the repo
# will work too. Needs GEMINI_API_KEY in ~/.ai_scientist_secrets.
#   sbatch slurm/check_gemini.slurm  &&  cat gem-<jobid>.out
#SBATCH --job-name=gem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=gem-%j.out
#SBATCH --error=gem-%j.out

module purge
module load proxy/default
module load "${ANACONDA_MODULE:-anaconda3/2024.6}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-ai_scientist}"
source ~/.ai_scientist_secrets

echo "host: $(hostname)   https_proxy=${https_proxy:-<unset>}"
python - <<'PY'
import os
from openai import OpenAI
c = OpenAI(api_key=os.environ["GEMINI_API_KEY"],
           base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
for model in ["gemini-2.0-flash", "gemini-2.5-flash"]:
    try:
        r = c.chat.completions.create(model=model, max_tokens=10,
                messages=[{"role":"user","content":"reply with the single word: ok"}])
        print(f"{model}: SUCCESS ->", r.choices[0].message.content)
    except Exception as e:
        print(f"{model}: FAIL ->", type(e).__name__, str(e)[:200])
PY
