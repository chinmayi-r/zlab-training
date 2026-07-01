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
- STATUS: launched to test whether surfacing the constraint helps; final node
  counts NOT captured in chat. **Go pull this off Adroit** (it may or may not have
  finished). If it never ran to completion, mark it incomplete in the report.

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

## 4. What is STILL MISSING (checklist before this report is "backed by data")

- [ ] Pull R1/R2/R3 run dirs off Adroit -> `data/adroit/` (journals + tree_viz.html + console log)
- [ ] Confirm R3's final node counts (was it finished?)
- [ ] Pull L1 run dir off your PC -> `data/local/` (all stage journals + the 28.61% node)
- [ ] Mine each journal with `scripts/mine_journal.py` -> commit the per-run CSVs
- [ ] (nice-to-have) screenshot each `unified_tree_viz.html` -> `data/*/tree.png`
- [ ] Save the two console logs (Adroit `.out` from SLURM, PC terminal scrollback)

Once `data/adroit/` and `data/local/` contain the real journals + CSVs, every number in
`report.md` is reproducible from files in the repo instead of from chat memory.
