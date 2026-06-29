# AI-Scientist-v2 on Adroit — Run Guide

End-to-end command sequence for the deep-dive. Designed for a **MIG A100 slice (or
V100)** on Adroit with the **Princeton AI Sandbox** as the $0 model backend. Every
flag/config key here was verified against the real repo — if the repo's README ever
disagrees, **the repo wins**.

> **Adroit network topology (verified June 2026 — important, and the opposite of what
> you might assume):**
> - **LOGIN node (`adroit-vis`)** has general outbound internet → `git clone`, `pip`,
>   `conda` all work here. **Do all setup + dataset downloads on the login node.**
> - **COMPUTE nodes** have *no* general internet; `module load proxy/default` opens only
>   a small **pre-approved API allowlist**. A github clone there returns **403**. The
>   Sandbox host returns **502** to a bare `curl /` — that means "reached but wrong
>   path", not "blocked" (blocked = 403), so it is *probably* allow-listed; prove it
>   with a real `chat.completions.create`, not curl.
> - GPUs are in the **`gpu`** partition (there is no `mig` partition). MIG A100 slices
>   are gres `3g.20gb` (~20GB) on `adroit-h11g2`; full A100s are `--gres=gpu:1
>   --constraint=a100`.

---

## 0. Get the kit onto Adroit (login node)

```bash
git clone https://github.com/chinmayi-r/zlab-training.git    # or pull if you have it
cd zlab-training/ai-scientist-v2
git checkout claude/ai-scientist-v2-failures-l9vej5
chmod +x scripts/*.sh scripts/*.py slurm/*.slurm
```

## 1. One-time setup — ON THE LOGIN NODE (general internet lives here)

Do NOT salloc for setup. Clone + env build need general internet, which compute nodes
lack. Put the repo and conda env on **scratch** (small /home quota):

```bash
export REPO_DIR=/scratch/network/$USER/AI-Scientist-v2
export CONDA_ENVS_DIR=/scratch/network/$USER/conda-envs
export ANACONDA_MODULE=anaconda3/2024.6        # `module avail anaconda3` for the exact name

bash scripts/setup_adroit.sh                   # clones into $REPO_DIR, builds env on scratch
```

The script pins the anaconda module, sources conda properly, and wraps `conda activate`
in `set +u` (Adroit's conda trips the `PS1: unbound variable` error otherwise).

## 2. Secrets (never commit these)

```bash
cp scripts/sandbox_secrets.env.example ~/.ai_scientist_secrets
chmod 600 ~/.ai_scientist_secrets
# edit ~/.ai_scientist_secrets: paste AI_SANDBOX_KEY (USE_AI_SANDBOX=1 already set)
source ~/.ai_scientist_secrets
```

## 3. Wire the Sandbox into the repo (the $0 patch)

```bash
python scripts/wire_sandbox.py --repo $REPO_DIR
# patches ai_scientist/llm.py AND treesearch/backend/backend_openai.py (idempotent).
# If an anchor isn't found, it tells you to patch by hand (see report.md Part D).
```

Smoke test that calls actually route to the Sandbox. **The Sandbox is reachable ONLY
from a compute node with `module load proxy/default`** — the login node cannot resolve
`api-ai-sandbox.princeton.edu` (per Princeton RC docs). So grab a quick session:

```bash
salloc --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:20:00   # CPU node is fine
module load proxy/default                  # compute-node only -- this is what reaches the Sandbox
module load anaconda3/2024.6
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ai_scientist
source ~/.ai_scientist_secrets

python - <<'PY'
import os
from openai import AzureOpenAI
c = AzureOpenAI(api_key=os.environ["AI_SANDBOX_KEY"],
                azure_endpoint=os.environ.get("SANDBOX_ENDPOINT","https://api-ai-sandbox.princeton.edu/"),
                api_version=os.environ.get("SANDBOX_API_VERSION","2025-03-01-preview"))
r = c.chat.completions.create(model="gpt-4o-mini", max_tokens=10,
        messages=[{"role":"user","content":"say ok"}])
print("sandbox says:", r.choices[0].message.content)
PY
```

> `Name or service not known` / `ConnectError` = you're on the login node, or forgot
> `module load proxy/default`. A model-not-found (404) is *fine* — it proves the route
> works; just use a current model name (`gpt-4o`).

## 4. Generate idea JSONs — ALSO on a compute node (ideation calls the Sandbox)

Ideation makes Sandbox calls, so it must run on a compute node with `proxy/default`
loaded (no GPU needed — a CPU session is fine). Do this in the *same* `salloc` session
as the smoke test above:

```bash
cd $REPO_DIR
cp ~/zlab-training/ai-scientist-v2/ideas/topic_baseline.md ai_scientist/ideas/
cp ~/zlab-training/ai-scientist-v2/ideas/topic_concrete.md ai_scientist/ideas/

for t in topic_baseline topic_concrete; do
  python ai_scientist/perform_ideation_temp_free.py \
    --workshop-file "ai_scientist/ideas/$t.md" \
    --model gpt-4o \
    --max-num-generations 8 \
    --num-reflections 3
done
# -> writes ai_scientist/ideas/topic_baseline.json and topic_concrete.json
```

> For **E1** the idea is held constant — reuse `topic_baseline.json` for both arms so
> only the debug knobs differ. For **E2** the idea *is* the variable — `topic_baseline`
> (vague) vs `topic_concrete` (pinned).

### 4b. Pre-fetch datasets ON THE LOGIN NODE (compute nodes can't download)

The LLM-written experiment code will try to download datasets at runtime (e.g.
torchvision CIFAR-10). On a compute node that download is blocked (no general internet).
So fetch them once on the login node into a scratch path the job will reuse:

```bash
export CIFAR_DIR=/scratch/network/$USER/data
python - <<PY
import os, torchvision
root = os.environ["CIFAR_DIR"]
torchvision.datasets.CIFAR10(root=root, train=True,  download=True)
torchvision.datasets.CIFAR10(root=root, train=False, download=True)
print("CIFAR-10 cached under", root)
PY
```

The SLURM scripts export `CIFAR_DIR`; `topic_concrete.md` tells the agent to use
torchvision CIFAR-10, which will find the cached copy instead of downloading. If your
idea needs a different dataset, pre-fetch it here too. Also export `HF_HOME` /
`HF_DATASETS_CACHE` to a scratch path and pre-pull any HuggingFace datasets on login.

## 5. Build the config variants

```bash
cd ~/zlab-training/ai-scientist-v2
mkdir -p $REPO_DIR/configs

# E1: deeper/greedier debugging (the only change vs baseline) + cheap + sandbox
python scripts/apply_experiment_config.py --config $REPO_DIR/bfts_config.yaml \
  --preset E1 --small --sandbox --out $REPO_DIR/configs/E1_debug8.yaml
python scripts/apply_experiment_config.py --config $REPO_DIR/bfts_config.yaml \
  --preset baseline --small --sandbox --out $REPO_DIR/configs/E1_baseline.yaml

# E2: identical config for both arms (the idea .md is the variable)
python scripts/apply_experiment_config.py --config $REPO_DIR/bfts_config.yaml \
  --preset baseline --small --sandbox --out $REPO_DIR/configs/E2_baseline.yaml
```

`--small` lowers stage iters and sets `num_drafts=1`; `--sandbox` moves the experiment
model off Bedrock-Claude onto a Sandbox model (default `gpt-4o`). Keep each variant so
"what I changed" is reproducible.

## 6. Submit the runs

The SLURM scripts copy the right variant over `bfts_config.yaml`, source your secrets,
and use Sandbox model names for all launcher flags.

```bash
cp ~/zlab-training/ai-scientist-v2/slurm/run_E1.slurm $REPO_DIR/
cp ~/zlab-training/ai-scientist-v2/slurm/run_E2.slurm $REPO_DIR/
cd $REPO_DIR

# E1 baseline vs deep-debug: run twice, swapping the cp line / config (edit in-script).
sbatch run_E1.slurm

# E2 two arms (only IDEA changes):
IDEA=ai_scientist/ideas/topic_baseline.json  sbatch run_E2.slurm
IDEA=ai_scientist/ideas/topic_concrete.json  sbatch run_E2.slurm

squeue -u $USER
tail -f slurm-<jobid>.out
```

> **Cost tip:** for failure analysis you mostly need the *tree*, not the PDF. Add
> `--skip_writeup --skip_review` to the launcher to stop after the experiment stage.
> On the Sandbox the writeup is free too, but skipping it makes runs much shorter.

## 7. Mine the trees (Part B)

```bash
cd $REPO_DIR

# success/failure + node counts across all runs:
~/zlab-training/ai-scientist-v2/scripts/triage_runs.sh experiments

# per-run CSV for the report:
python ~/zlab-training/ai-scientist-v2/scripts/mine_journal.py \
       experiments/<run>/logs/0-run/ -o E1_nodes.csv
```

Open `experiments/<run>/logs/0-run/unified_tree_viz.html` in a browser (scp it back, or
use a port-forward) to see working vs buggy nodes visually. Cross-reference the CSV:
**buggy leaves with `debug_depth` below `max_debug_depth`** are abandoned-with-budget —
the tool-use signal E1 targets.

## 8. Compare A/B and fill the tables

- **E1:** did buggy nodes convert to working with the deeper budget? (count
  `is_buggy=True` leaves in baseline that have working counterparts in E1)
- **E2:** nodes-to-first-working-node and #OOM/setup errors, vague vs concrete.

Put two filled failure-node rows + the A/B result into the Part F tables in
`report.md`. That is the deliverable.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `module load anaconda3` → "No default version" | pin a version: `module load anaconda3/2024.6` (`module avail anaconda3` to list) |
| `PS1: unbound variable` during conda activate | `set +u` before `conda activate`, `set -u` after — the kit scripts already do this |
| `import torch` → `undefined symbol: iJIT_NotifyEvent` | conda pulled MKL 2025; pin it back: `conda install "mkl=2024.0" -y` (kit script now does this) |
| pip warns `torch requires sympy==1.13.1` | `pip install "sympy==1.13.1"` (kit script now does this) |
| github clone `403` on a compute node | expected — clone on the **login node** (general internet); compute nodes are allowlist-only |
| `curl https://api-ai-sandbox.princeton.edu/` → 502 | not a failure — a bare `/` isn't a valid path. Test with the AzureOpenAI smoke call instead |
| `salloc` fails with `--partition=mig` | there is no `mig` partition; use `--partition=gpu --gres=gpu:3g.20gb:1` (MIG slice) |
| dataset download hangs/fails inside the job | pre-fetch it on the login node (step 4b); compute nodes can't download |
| `KeyError: AI_SANDBOX_KEY` | `source ~/.ai_scientist_secrets` in the same shell / SLURM script |
| calls hit public OpenAI / 401 | `USE_AI_SANDBOX` not exported, or `wire_sandbox.py` not run, or a `claude-` model name slipped through (those skip the Sandbox by design) |
| `wire_sandbox.py` "anchor not found" | repo drifted; patch the two functions by hand per report.md Part D |
| CUDA OOM | use `topic_concrete.json` (pins batch/model) — that's E2; or drop batch to 64 in the idea `.md` |
| Bedrock/AWS error | you left the experiment model on Claude — rebuild the config with `--sandbox` |
| ideation/launcher rejects a model name | the Sandbox name must reach the OpenAI branch (no `claude-` substring); `gpt-4o`/`o3-mini` are safe |
| `$REPO_DIR` empty in a new shell | re-`export REPO_DIR=...` (and `CONDA_ENVS_DIR`, `ANACONDA_MODULE`) — or add them to `~/.ai_scientist_secrets` |
