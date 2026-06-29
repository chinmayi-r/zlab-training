#!/bin/bash
# diagnose_sandbox.slurm - figure out HOW (or whether) the AI Sandbox is reachable
# from an Adroit compute node. Tests, in order: DNS, direct AzureOpenAI (no proxy),
# AzureOpenAI via proxy, and a raw curl POST via proxy. Submit from login:
#   sbatch slurm/diagnose_sandbox.slurm   &&  cat sbxdiag-<jobid>.out
#SBATCH --job-name=sbxdiag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=sbxdiag-%j.out
#SBATCH --error=sbxdiag-%j.out

module purge
module load "${ANACONDA_MODULE:-anaconda3/2024.6}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-ai_scientist}"
source ~/.ai_scientist_secrets

HOST=api-ai-sandbox.princeton.edu
echo "=== host: $(hostname) ==="

echo; echo "=== [1] DNS resolution (direct) ==="
python - <<'PY'
import socket
try: print("resolves to", socket.gethostbyname("api-ai-sandbox.princeton.edu"))
except Exception as e: print("DNS FAIL:", repr(e))
PY

call() {
python - <<'PY'
import os, traceback
from openai import AzureOpenAI
try:
    c = AzureOpenAI(api_key=os.environ["AI_SANDBOX_KEY"],
        azure_endpoint=os.environ.get("SANDBOX_ENDPOINT","https://api-ai-sandbox.princeton.edu/"),
        api_version=os.environ.get("SANDBOX_API_VERSION","2025-03-01-preview"))
    r = c.chat.completions.create(model="gpt-4o-mini", max_tokens=5,
        messages=[{"role":"user","content":"ok"}])
    print("SUCCESS:", r.choices[0].message.content)
except Exception as e:
    print("FAIL:", type(e).__name__, str(e)[:300])
PY
}

echo; echo "=== [2] AzureOpenAI DIRECT (proxy NOT loaded yet) ==="
echo "HTTPS_PROXY=${HTTPS_PROXY:-<unset>}"
call

echo; echo "=== loading proxy/default ==="
module load proxy/default
echo "HTTPS_PROXY=${HTTPS_PROXY:-<unset>}  https_proxy=${https_proxy:-<unset>}"

echo; echo "=== [3] AzureOpenAI VIA PROXY ==="
call

echo; echo "=== [4] raw curl POST via proxy (real endpoint, shows HTTP code) ==="
curl -sS -X POST \
  "https://${HOST}/openai/deployments/gpt-4o-mini/chat/completions?api-version=${SANDBOX_API_VERSION:-2025-03-01-preview}" \
  -H "api-key: ${AI_SANDBOX_KEY}" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"ok"}],"max_tokens":5}' \
  -w "\nHTTP %{http_code}\n" || true

echo; echo "=== [5] curl DIRECT via --noproxy (bypass proxy) ==="
curl -sS --noproxy '*' -X POST \
  "https://${HOST}/openai/deployments/gpt-4o-mini/chat/completions?api-version=${SANDBOX_API_VERSION:-2025-03-01-preview}" \
  -H "api-key: ${AI_SANDBOX_KEY}" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"ok"}],"max_tokens":5}' \
  -w "\nHTTP %{http_code}\n" || true
