# RAW OBSERVATIONS — no interpretation, just what was seen

This file records the raw facts observed while running AI-Scientist-v2 for the
failure-mode report. No opinions, no classification — those live in `report.md`.
Everything here was either printed to a terminal during a run or is a config value.

**IMPORTANT — the underlying files are NOT in this repo yet.** They are on two
machines. This file is a transcript of what was observed plus a map of where the
real output lives. Use `data/collect_outputs.sh` on each machine to pull the actual
journals/CSVs in here so the evidence is preserved. Until you do, the only record of
these numbers is this file + chat scrollback.

---

## 0. Where the raw output physically lives (must be collected from BOTH machines)

| Machine | Run(s) | Output dir | Reachable now? |
|---|---|---|---|
| Adroit (login/compute) | R1, R2, R3 | `/scratch/network/$USER/AI-Scientist-v2/experiments/<timestamp_idea>/` | only from Adroit |
| Local PC (Windows/WSL2) | L1 | `~/AI-Scientist-v2/experiments/<timestamp_idea>/` | only from your PC |

Each run dir contains:
- `logs/0-run/stage_*/journal*.json`  <- per-node plan/code/output/errors (the evidence)
- `logs/0-run/unified_tree_viz.html`  <- green/red node tree (screenshot-able)
- `logs/0-run/tree_data.json`         <- tree edges
- `<timestamp_idea>.pdf`              <- ONLY if a writeup succeeded (none of ours did; we used --skip_writeup)

To get them into the repo: run `data/collect_outputs.sh` on Adroit, then again on
your PC, then commit the `data/adroit/` and `data/local/` folders it produces.

---

## 1. ADROIT runs (no-internet compute node, OpenAI key via `module load proxy/default`)

Common settings: idea = `topic_concrete` (CIFAR-10 / ResNet-18), 1 draft,
stage1 = 14 iters, `max_debug_depth=3`, `debug_prob=0.5`, GPU = 1x MIG A100 `3g.20gb`,
`--skip_writeup --skip_review`, `generate_report=false`.

### R1 — coder = gpt-4o-mini
- total nodes: 14
- working / buggy: **0 / 14**
- error classes observed: 7 `URLError`, 3 `ModuleNotFoundError`, 2 `SystemExit`,
  1 `NameError`, 1 `FileNotFoundError`
- the URLError text: `URLError: Tunnel connection failed: 403 Forbidden`
  (from the agent's code calling `torchvision.datasets.CIFAR10(..., download=True)`)

### R2 — coder = gpt-4o (only variable changed vs R1: the coder model)
- total nodes: 14
- working / buggy: **0 / 14**
- error classes observed: 8 `URLError`, 4 `RuntimeError`, 1 `FileNotFoundError`,
  1 `ModuleNotFoundError`
- verbatim feedback-model output on a buggy node (paraphrase from console):
  *"set `download=True` ... ensure the environment has internet access"* — advice that
  cannot succeed on the compute node; where the agent had already set download=True this
  produced a repeat `URLError 403`.
- representative node ids seen in console: `a0f9641...` (n0, draft, URLError),
  `263be77...` (n9, debug, URLError again)
- crash at end with 0 working nodes: best-node selector printed
  *"all metrics are nan ... selecting the first by default"*, then
  `log_summarization.overall_summarize` raised `ValueError: not enough values to unpack`
  / `shutil.rmtree(...experiment_results)` raised `FileNotFoundError`. Setting
  `generate_report=false` avoided this.

### R3 — coder = gpt-4o, offline note moved into the idea **Abstract**
- STATUS: **NEVER RUN.** Scratch has only two AI-Scientist-v2 experiment dirs
  (`2026-06-29_13-18-30` = R1, `2026-06-30_00-51-48` = R2). There is no third run.
  The report's R3 row is a planned-but-not-executed variant; do not cite numbers for it.

### Preflight logs (verbatim, in `data/adroit/slurm_preflight_logs/`)
- `sbx-3290184.out` (Sandbox reachability): `httpx.ProxyError: 502 Proxy Error` →
  `openai.APIConnectionError: Connection error.` — the retired Princeton Sandbox is
  unreachable through `proxy/default`.
- `oai-3290212.out` (OpenAI reachability through the proxy):
  `host: adroit-h11n2  https_proxy=http://adroit-proxy:8080` then
  `gpt-4o-mini: SUCCESS -> ok` and `gpt-4o: SUCCESS -> Ok`. OpenAI works through the proxy;
  this is why the runs used a personal OpenAI key, not the Sandbox.
- The proxy banner it printed: *"Http proxy settings allow access to a very limited and
  pre-approved list of internet API servers ... General internet access is not supported."*

### R2 per-node (from the committed CSV; all 14 are `draft`, all `is_buggy=True`)
- node ids + exc_type: `a0f9641…` URLError, `6949afae…` RuntimeError, `fa89047f…` URLError,
  `f2bc5eb9…` FileNotFoundError, `9c1a8512…` URLError, `4628c940…` URLError,
  `7dc77b13…` RuntimeError, `fd03f120…` URLError, `151744d3…` ModuleNotFoundError,
  `263be772…` URLError, `69431232…` URLError, `5308fae8…` RuntimeError,
  `cf2fa248…` URLError, `01675466…` RuntimeError.
- every traceback's first frame is
  `ai_scientist/treesearch/interpreter.py` (the sandboxed exec wrapper).

---

## 2. LOCAL run L1 (Windows + RTX-3050, WSL2, open home internet, OpenAI key direct)

Settings: idea = `topic_concrete_fast.json` (3 epochs, 1 seed, 3 arms:
baseline / dropout p=0.3 / weight-decay 5e-4), coder = gpt-4o,
feedback/VLM = gpt-4o-mini, `exec.timeout=120`, `generate_report=false`,
`--skip_writeup --skip_review`, CIFAR pre-cached under `$CIFAR_DIR`.

### Stage mined: `stage_2_baseline_tuning_1_first_attempt/journal.json`
- total nodes: 9
- working / buggy: **2 / 7**  (7 buggy = 7 abandoned leaves)
- top error classes: 2 `HfUriError`, 1 `DatasetNotFoundError`
- working node ids:
  - `2af67383418246f995c7e0b0e9613ad1` (n0) — is_buggy = False
  - `3852733109ac4558b6a5abb4aa42e3cb` (n2) — is_buggy = False
- real metric observed on a working node: **28.61% top-1 validation accuracy**
  (ResNet-18 trained from scratch on the CIFAR-10 subset)
- the buggy nodes failed reaching for a HuggingFace dataset path (`HfUriError` /
  `DatasetNotFoundError`) instead of the cached torchvision CIFAR-10 that the idea pins.

### Other stages of L1 (NOT yet mined — go pull them)
- `stage_1_*`, `stage_3_*`, `stage_4_*` journals exist in the same run dir but were not
  mined. Stage 4 (multi-model ablation) was observed **looping / re-running** and hitting
  the 120s `exec.timeout`; the run was Ctrl-C'd there. Earlier stages had already produced
  the working nodes above.

---

## 3. Config values in effect (from `bfts_config.yaml`, verified defaults)

| key | default | used in Adroit runs | used in L1 |
|---|---|---|---|
| `agent.search.max_debug_depth` | 3 | 3 | 3 |
| `agent.search.debug_prob` | 0.5 | 0.5 | 0.5 |
| `agent.search.num_drafts` | 3 | 1 | (baseline preset) |
| `agent.num_workers` | 4 | — | — |
| `agent.stages.stage1_max_iters` | 20 | 14 | (small preset) |
| `agent.code.model` | `anthropic.claude-3-5-sonnet-20241022-v2:0` | gpt-4o-mini (R1), gpt-4o (R2/R3) | gpt-4o |
| `agent.feedback.model` | `gpt-4o-2024-11-20` | (default) | gpt-4o-mini |
| `agent.vlm_feedback.model` | `gpt-4o-2024-11-20` | (default) | gpt-4o-mini |
| `exec.timeout` | (repo default) | (default) | 120 |
| `generate_report` | true | false | false |

---

## 4. Collection status (what is now backed by real files in the repo)

- [x] Pulled R1/R2 run dirs off Adroit -> `data/adroit/` (journals + CSVs + tree_viz.html)
- [x] Confirmed R3 was NEVER run (only 2 Adroit experiment dirs exist on scratch)
- [x] Pulled L1 run dir off the PC -> `data/local/` (all 4 stage journals + tree_viz.html)
- [x] Mined every journal with `scripts/mine_journal.py` -> CSVs committed alongside
- [x] Saved the Adroit SLURM preflight logs -> `data/adroit/slurm_preflight_logs/*.out`
- [ ] (nice-to-have) screenshot each `unified_tree_viz.html` -> `data/*/tree.png`

Every node count / error class / metric quoted in `report.md` is now reproducible from a
committed file: re-mine any `data/{adroit,local}/**/journal*.json` with
`python scripts/mine_journal.py <file>` to regenerate the CSV and the printed totals.

### File inventory (committed)
```
data/adroit/2026-06-29_13-18-30_.../  R1: stage_1 journal.json + .csv + tree viz  (0/14, 7 URLError)
data/adroit/2026-06-30_00-51-48_.../  R2: stage_1 journal.json + .csv + tree viz  (0/14, 8 URLError)
data/adroit/slurm_preflight_logs/     sbx*/oai*/apihosts*/ideation* .out (proxy 502, OpenAI SUCCESS)
data/local/2026-06-30_04-33-04_.../   L1: stage_1..4 journals + .csv + tree viz  (5 working / 25 buggy)
data/local/2026-06-30_04-27-45_.../   partial local run: stage_1 journal only
```
