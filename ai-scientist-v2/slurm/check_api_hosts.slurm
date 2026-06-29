#!/bin/bash
# check_api_hosts.slurm - which LLM API hosts are reachable through Adroit's
# proxy/default? The Sandbox (api-ai-sandbox.princeton.edu) returns 502; this tells
# us which alternative backend works so we're not blocked. Any HTTP code that is NOT
# 000/502/403 means "reachable" (401/404/405 are fine -- they prove we got through).
#   sbatch slurm/check_api_hosts.slurm  &&  cat apihosts-<jobid>.out
#SBATCH --job-name=apihosts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=apihosts-%j.out
#SBATCH --error=apihosts-%j.out

module purge
module load proxy/default
echo "host: $(hostname)   https_proxy=${https_proxy:-<unset>}"
echo

probe () {  # $1 = label, $2 = url
  code=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 20 "$2" 2>/dev/null || echo "ERR")
  printf "  %-34s -> HTTP %s\n" "$1" "$code"
}

echo "=== reachable through proxy? (401/404/405 = reachable; 000/502 = blocked) ==="
probe "Princeton Sandbox"   "https://api-ai-sandbox.princeton.edu/"
probe "OpenAI"              "https://api.openai.com/v1/models"
probe "Gemini (Google)"    "https://generativelanguage.googleapis.com/v1beta/models"
probe "Anthropic"          "https://api.anthropic.com/v1/models"
probe "HuggingFace"        "https://huggingface.co"
probe "Semantic Scholar"   "https://api.semanticscholar.org/graph/v1/paper/search?query=test"
