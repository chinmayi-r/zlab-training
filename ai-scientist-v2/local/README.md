# Running AI-Scientist-v2 locally (Windows + RTX 3050, via WSL2)

Why local: your home internet is open, so the whole class of Adroit failures vanishes
(the agent's `download=True` works, ideation works, no proxy/allowlist). You still need
an LLM key — use your **OpenAI** key (works directly over normal internet; the Princeton
Sandbox is retired and not needed here).

## Step 0 — Windows prep (once, in PowerShell as admin)

```powershell
wsl --install            # installs WSL2 + Ubuntu; reboot if asked
```
Also install/update the **NVIDIA Windows driver** (GeForce Game Ready or Studio). That
driver is what exposes the RTX 3050 to WSL — do NOT install a separate CUDA *driver*
inside Ubuntu. (The torch pip wheels bundle the CUDA runtime, so no CUDA toolkit needed.)

Open "Ubuntu" from the Start menu to get a Linux shell. Everything below runs there.

## Step 1 — get the kit + run setup (inside Ubuntu)

```bash
cd ~
git clone https://github.com/chinmayi-r/zlab-training.git
cd zlab-training/ai-scientist-v2
git checkout claude/ai-scientist-v2-failures-l9vej5
bash local/setup_local.sh          # miniconda + repo + torch(cu124) + requirements
```
At the end it prints `cuda available: True` if the GPU is visible. If it says `False`,
update the NVIDIA Windows driver and run `wsl --shutdown` in PowerShell, then reopen
Ubuntu and re-check with:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Step 2 — set your OpenAI key (per shell, or add to ~/.bashrc)

```bash
export OPENAI_API_KEY="sk-...your key..."
```

## Step 3 — run

```bash
bash local/run_local.sh            # builds a cheap config + launches the concrete idea
```
This uses `ideas/topic_concrete_local.json` (lets the agent download CIFAR normally,
caches it under `$CIFAR_DIR`), `gpt-4o` as the coder, and skips writeup/review. Watch the
console; when it finishes, mine the tree:
```bash
RUN=$(ls -dt $HOME/AI-Scientist-v2/experiments/*/ | head -1)
python scripts/mine_journal.py "$RUN/logs/0-run/" -o local_nodes.csv
column -s, -t local_nodes.csv | cut -c1-200
```

## Notes
- RTX 3050 is 4 GB (laptop) or 8 GB (desktop). ResNet-18 on CIFAR at batch 128 fits; if
  you hit CUDA OOM, the idea already tells the agent to drop to batch 64.
- Cost: a small-iteration run on `gpt-4o` is a few dollars. Use `gpt-4o-mini` everywhere
  (edit `run_local.sh`) to make it ~cents.
- No SLURM, no `proxy/default`, no Sandbox — just `python launch_scientist_bfts.py`.
