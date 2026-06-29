# AI-Scientist-v2 on Adroit — Run Guide

End-to-end command sequence for the deep-dive. Designed for a **MIG A100 slice (or
V100)** on Adroit with the **Princeton AI Sandbox** as the $0 model backend. Every
flag/config key here was verified against the real repo — if the repo's README ever
disagrees, **the repo wins**.

> Adroit specifics (`--partition`, `--gres`, `module load` names, undergrad time
> limits) vary; the placeholders below mirror the lab's nano-vllm setup. **Verify them
> against the current Princeton RC / Adroit docs.**

---

## 0. Get the kit onto Adroit

```bash
# on Adroit
git clone https://github.com/chinmayi-r/zlab-training.git    # or pull if you have it
cd zlab-training/ai-scientist-v2
chmod +x scripts/*.sh scripts/*.py slurm/*.slurm
```

## 1. One-time setup (on a COMPUTE node)

Outbound internet and the Sandbox only work after `module load proxy/default`, which
only works on a compute node — so grab an interactive session first:

```bash
salloc --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G \
       --gres=gpu:1 --time=01:00:00 --partition=mig      # verify syntax for Adroit
module load proxy/default
curl -sSf https://api.openai.com/v1/models >/dev/null && echo "internet OK" || echo "NO INTERNET"

# clone + conda env + requirements + GPU check (mirrors the repo README verbatim)
REPO_DIR=$HOME/AI-Scientist-v2 bash scripts/setup_adroit.sh
```

## 2. Secrets (never commit these)

```bash
cp scripts/sandbox_secrets.env.example ~/.ai_scientist_secrets
chmod 600 ~/.ai_scientist_secrets
# edit ~/.ai_scientist_secrets: paste AI_SANDBOX_KEY (USE_AI_SANDBOX=1 already set)
source ~/.ai_scientist_secrets
```

## 3. Wire the Sandbox into the repo (the $0 patch)

```bash
python scripts/wire_sandbox.py --repo $HOME/AI-Scientist-v2
# patches ai_scientist/llm.py AND treesearch/backend/backend_openai.py (idempotent).
# If an anchor isn't found, it tells you to patch by hand (see report.md Part D).
```

Quick smoke test that calls actually route to the Sandbox:

```bash
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

## 4. Copy the two idea topics into the repo and generate idea JSONs

The launcher consumes a `*.json` idea file (and, with `--load_code`, a same-named
`*.py`). Generate the JSON once per topic with the ideation script (cheap, paid once —
or free via the Sandbox since ideation also goes through the patched OpenAI path):

```bash
cd $HOME/AI-Scientist-v2
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

## 5. Build the config variants

```bash
cd ~/zlab-training/ai-scientist-v2
mkdir -p $HOME/AI-Scientist-v2/configs

# E1: deeper/greedier debugging (the only change vs baseline) + cheap + sandbox
python scripts/apply_experiment_config.py --config $HOME/AI-Scientist-v2/bfts_config.yaml \
  --preset E1 --small --sandbox --out $HOME/AI-Scientist-v2/configs/E1_debug8.yaml
python scripts/apply_experiment_config.py --config $HOME/AI-Scientist-v2/bfts_config.yaml \
  --preset baseline --small --sandbox --out $HOME/AI-Scientist-v2/configs/E1_baseline.yaml

# E2: identical config for both arms (the idea .md is the variable)
python scripts/apply_experiment_config.py --config $HOME/AI-Scientist-v2/bfts_config.yaml \
  --preset baseline --small --sandbox --out $HOME/AI-Scientist-v2/configs/E2_baseline.yaml
```

`--small` lowers stage iters and sets `num_drafts=1`; `--sandbox` moves the experiment
model off Bedrock-Claude onto a Sandbox model (default `gpt-4o`). Keep each variant so
"what I changed" is reproducible.

## 6. Submit the runs

The SLURM scripts copy the right variant over `bfts_config.yaml`, source your secrets,
and use Sandbox model names for all launcher flags.

```bash
cp ~/zlab-training/ai-scientist-v2/slurm/run_E1.slurm $HOME/AI-Scientist-v2/
cp ~/zlab-training/ai-scientist-v2/slurm/run_E2.slurm $HOME/AI-Scientist-v2/
cd $HOME/AI-Scientist-v2

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
cd $HOME/AI-Scientist-v2

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
| `NO INTERNET` after salloc | you're on a login node, or forgot `module load proxy/default` (compute node only) |
| `KeyError: AI_SANDBOX_KEY` | `source ~/.ai_scientist_secrets` in the same shell / SLURM script |
| calls hit public OpenAI / 401 | `USE_AI_SANDBOX` not exported, or `wire_sandbox.py` not run, or a `claude-` model name slipped through (those skip the Sandbox by design) |
| `wire_sandbox.py` "anchor not found" | repo drifted; patch the two functions by hand per report.md Part D |
| CUDA OOM | use `topic_concrete.json` (pins batch/model) — that's E2; or drop batch to 64 in the idea `.md` |
| Bedrock/AWS error | you left the experiment model on Claude — rebuild the config with `--sandbox` |
| ideation/launcher rejects a model name | the Sandbox name must reach the OpenAI branch (no `claude-` substring); `gpt-4o`/`o3-mini` are safe |
