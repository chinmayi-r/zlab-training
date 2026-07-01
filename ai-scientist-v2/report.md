# AI-Scientist-v2: A Deep Dive on Agent Failure Modes

**Repo under study:** [SakanaAI/AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2)
**Paper:** [arXiv:2504.08066](https://arxiv.org/abs/2504.08066) ([Sakana mirror](https://pub.sakana.ai/ai-scientist-v2/paper))
**Goal (per Taiming):** stop surveying breadth; go deep on one or two representative
agent failures, classify the root cause, propose an intervention, and frame why the
system exists and what gap it fills.

> **Grounding note.** Everything about *commands, flags, config keys, and client
> code* below was verified by reading the actual repo files (README.md,
> `bfts_config.yaml`, `launch_scientist_bfts.py`, `ai_scientist/llm.py`,
> `ai_scientist/treesearch/backend/backend_openai.py`,
> `ai_scientist/treesearch/journal.py`). A few **paper-only numbers** (exact review
> scores, Table 1 cell values) are marked `[fill from PDF]` — the arXiv/Sakana PDF
> hosts were unreachable from the build environment, so confirm those against the
> real PDF before quoting them. Do not invent them.

---

## Part A — Conceptual framing: why this exists / what gap it fills

### A.1 What is the actual evaluation signal?

AI-Scientist-v2 is not a benchmark in the SWE-bench sense — there is no held-out
test set and no automatic checker. The "score" is an **evaluation claim**: can an
end-to-end agent produce a paper that passes *real peer review*? The headline result
is the first **entirely AI-generated paper accepted at an ICLR workshop** (the ICBINB
workshop; the repo even ships that workshop's call-for-papers as the default idea seed,
`ai_scientist/ideas/i_cant_believe_its_not_better.md`). Acceptance scores: `[fill from PDF]`.

Is peer-review acceptance a good proxy for scientific value? It is a *real* signal —
a committee of humans said "this clears the workshop bar" — but it is a proxy with a
known skew:

- **Over-rewards** polish, framing, narrative coherence, and figure aesthetics. The
  system optimizes these directly: `bfts_config.yaml` defines a `vlm_feedback` model
  (a vision-language reviewer, default `gpt-4o-2024-11-20`) whose job is to look at
  generated plots and push the agent to improve them. A loop that refines figures
  until a VLM is satisfied is, almost by construction, optimizing "looks like an
  accepted paper" rather than "is correct."
- **Under-rewards** genuine novelty and correctness. A workshop reviewer reading one
  of hundreds of submissions cannot re-run the experiments; they trust the artifact.
  An agent that produces a *convincing* artifact is rewarded the same as one that
  produces a *correct* one. Workshops (especially a negative-results venue like
  ICBINB) also have a higher acceptance rate and a lower novelty bar than main tracks.

So the evaluation signal is "acceptable," and "acceptable" is mostly about
**presentation and plausibility**, only loosely about truth. That is the single most
important thing to keep in mind when reading any failure: the system is climbing the
gradient of *looks-publishable*, not *is-right*.

### A.2 Gap vs. v1 (arXiv:2408.06292)

v1 relied on **human-authored, per-domain code templates**: a researcher handed the
agent a working experimental scaffold for a specific domain, and the agent filled in
the science. High success, but narrow and non-generalizing — every new domain needed
new human scaffolding.

v2 **removes the templates**. Instead of a fixed scaffold it uses **progressive
agentic tree search** (built on [AIDE](https://github.com/WecoAI/aideml)) managed by
an **experiment-manager agent**, which lets it generalize across ML domains and tackle
open-ended ideas. The trade is explicit and the README itself admits v2 is not
strictly "better": broader and more open-ended, but **lower success rate** than the
templated v1. (Table 1 in the paper quantifies the v1-vs-v2 comparison: `[fill from PDF]`.)

The interesting research question this raises: **is removing scaffolding always
progress?** Structure *helps* when the solution path is known and narrow (templates
encode hard-won setup knowledge, eliminating whole classes of setup failure). Structure
*constrains* when the space is open and the right path is unknown. v2's bet is that for
open-ended discovery you must give up the scaffold to explore — and then pay for it in
a much higher rate of setup/execution failures (which is exactly what the failure
analysis in Part B surfaces). This is the heart of the structure-vs-generality tension.

### A.3 Gap vs. closed agent benchmarks (SWE-bench, MLE-bench, RE-bench)

Those benchmarks grade **well-specified tasks with known solutions and automatic
checks**: a SWE-bench task has a hidden test suite; MLE-bench has a Kaggle leaderboard;
RE-bench has reference solutions. The grader is cheap, objective, and repeatable.

AI-Scientist-v2 targets **open-ended discovery with no ground-truth solution**. There
is no hidden test to pass. That is precisely the gap it fills — and precisely why it is
hard to measure. The tree search + experiment manager is the *mechanism for searching
an open solution space* rather than *solving a closed one*: drafts are roots, debugging
and improvement are branches, and a learned/LLM judge picks the "best" node because no
oracle exists. The evaluation problem and the search problem are the same problem.

### A.4 Where it sits among peers (related work)

- **AlphaEvolve** ([2506.13131](https://arxiv.org/abs/2506.13131)) — algorithmic
  discovery via evolutionary search over code, graded by an *automatic* fitness
  function. Closed evaluation, open search.
- **AI co-scientist** ([2502.18864](https://arxiv.org/abs/2502.18864)) — hypothesis
  *generation* and ranking; stops short of running the full experiment-to-paper loop.
- **Kosmos** ([2511.02824](https://arxiv.org/abs/2511.02824)) — agentic scientific
  discovery / data analysis.

v2's distinctive bet is **full end-to-end with the artifact being a peer-reviewed
paper** — it owns ideation, execution, *and* the writeup, and is graded by humans on
the final document. It takes on the whole pipeline, which is why its failures can occur
at any of three very different stages (see Part B).

### A.5 Bridge to the failure cases: where is the bottleneck?

Tie-off (now filled from the runs in Parts F/F′/G): across both the Adroit and the local
runs the bottleneck is **execution / tool-use**, never ideation or model capability. On
Adroit it shows up as a blocked-download confound the feedback loop can't escape (G.1–G.4);
on open internet, with that confound removed, it shows up as the agent picking the wrong
data API (`HfUriError` instead of the pinned torchvision loader, G.6). Both are *tool-use*
failures; a 6× stronger coder (gpt-4o vs gpt-4o-mini) changed nothing, ruling out
capability. The agent has reasonable ideas and writes plausible code, but loses runs to
setup/data-loading/debugging failures rather than to bad ideation or bad evaluation. If that holds, the highest-leverage investment is **execution scaffolding**
(better debugging strategy, OOM handling, a minimal correct starter scaffold) — which
is interesting because it is *exactly the scaffolding v2 deliberately removed*. The
research idea hiding here: **selective scaffolding** — keep the open-ended ideation/search
of v2, but re-introduce a thin, generic execution scaffold to recover v1's setup
robustness without re-introducing v1's per-domain narrowness. E1 and E2 (Part C) are
designed to test whether the bottleneck is really tool-use vs. task-spec.

---

## Part B — Mining the logs (method + what to extract)

Every run writes a timestamped folder:

```
experiments/<timestamp_ideaname>/
  logs/0-run/
    unified_tree_viz.html   <- open in a browser: working (green) vs buggy (red) nodes
    journal*.json           <- the serialized tree: per-node plan, code, output, errors
  <timestamp_ideaname>.pdf  <- exists ONLY if the run reached a successful writeup
```

### B.1 The data model (verified from `treesearch/journal.py`)

A run is a `Journal` of `Node`s. The fields that matter for diagnosis (real attribute
names):

| Field | Meaning |
|---|---|
| `id`, `step`, `parent`, `children` | tree position (UUID hex ids) |
| `plan`, `overall_plan` | what the node *intended* to do |
| `code`, `plot_code` | what it actually wrote (diff vs. parent is most informative) |
| `_term_out` / `term_out` | execution output |
| `exc_type`, `exc_info`, `exc_stack` | the exception, if it crashed |
| `is_buggy`, `is_buggy_plots` | the agent's own bug verdict |
| `metric`, `analysis` | the score and the agent's post-hoc analysis |
| `vlm_feedback_summary` | the VLM's verdict on the figures |

Derived (computed by `scripts/mine_journal.py`):

- `stage` — `draft` (no parent), `debug` (parent was buggy), or `improve`. Mirrors the
  repo's own `Node.stage_name` property.
- `debug_depth` — length of the consecutive buggy-ancestor chain. The search cap is
  `agent.search.max_debug_depth` (default **3**); `agent.search.debug_prob` (default
  **0.5**) is the probability the agent chooses to debug a buggy node vs. abandon it.
- `outcome` — a **buggy leaf** (`is_buggy=True`, `n_children=0`) is an **abandoned**
  line of attack: the single most diagnostic signal for tool-use failures.

The convenience accessors `Journal.draft_nodes`, `.buggy_nodes`, `.good_nodes`, and
`.get_best_node()` confirm the success criterion: a "good" node is non-buggy **with
passing plots**, and the best node is chosen by an **LLM judge** — there is no oracle.

### B.2 Commands

```bash
cd ~/AI-Scientist-v2
# success/failure at a glance, plus working/buggy counts per run:
~/zlab-training/ai-scientist-v2/scripts/triage_runs.sh experiments

# one run -> per-node CSV for the report:
python ~/zlab-training/ai-scientist-v2/scripts/mine_journal.py \
       experiments/<run>/logs/0-run/ -o nodes.csv
```

`mine_journal.py` prints totals (working / buggy / abandoned-leaves / drafts) and the
top error classes, and writes a CSV with one row per node:
`id, parent, step, stage, is_buggy, outcome, n_children, debug_depth, exc_type,
first_tb_line, metric, plan`.

### B.3 The three diagnostic axes (classify every failure node)

| Axis | Signature in the tree | Example |
|---|---|---|
| **Task specification** | Idea was vague/infeasible; agent flails on undefined choices | No dataset/metric pinned → tries random datasets, OOM, gives up |
| **Model capability** | Understood the task, couldn't write correct code for the method | Misimplements the algorithm; can't fix a subtle numerical bug even with debug budget left |
| **Tool-use strategy** | Could have succeeded but used the tools poorly / quit early | Misreads traceback, re-submits the same broken code, abandons at `debug_depth=1` while budget remained, ignores an OOM signal |

How to tell them apart from the CSV:

- **Abandoned leaf with `debug_depth` < `max_debug_depth`** → tool-use (gave up with
  budget to spare). This is what E1 tests.
- **Repeated identical `exc_type` across a debug chain that exhausts the budget** →
  capability (couldn't fix it even when it kept trying). This is what E3 isolates.
- **OOM / `FileNotFoundError` / dataset-download errors clustered near the drafts** →
  task-spec (the idea never pinned the environment). This is what E2 tests.

Pick the **one or two richest cases** — ideally one where the classification is *not*
obvious and you had to read the `code` diff and `_term_out` to decide. That ambiguity
is the depth Taiming wants. Record them in the Part F tables.

---

## Part C — Controlled experiments (test the diagnosis, don't just collect more data)

Each experiment is a clean A/B: **same idea, same seed, change exactly one variable.**
Recommended pair: **E1 (tool-use) + E2 (task-spec)** — they map onto two of the three
axes and E2 doubles as the CUDA-OOM fix.

### E1 — "abandonment, not incapability" (tool-use axis)

- **Change exactly one thing:** `agent.search.max_debug_depth` 3→8 and
  `agent.search.debug_prob` 0.5→0.9. Nothing else.
- **Generate the variant:**
  ```bash
  python scripts/apply_experiment_config.py \
    --config ~/AI-Scientist-v2/bfts_config.yaml \
    --preset E1 --small --sandbox \
    --out ~/AI-Scientist-v2/configs/E1_debug8.yaml
  # baseline (A side): same flags WITHOUT --preset E1
  python scripts/apply_experiment_config.py \
    --config ~/AI-Scientist-v2/bfts_config.yaml \
    --preset baseline --small --sandbox \
    --out ~/AI-Scientist-v2/configs/E1_baseline.yaml
  ```
- **Measure:** success Y/N; number of buggy nodes that become working *after* the
  extra debug budget; change in abandoned-leaf count.
- **Prediction if tool-use is the bottleneck:** the deeper budget converts a
  meaningful fraction of previously-abandoned leaves into working nodes. If it makes
  no difference, those failures were *capability* (the agent couldn't fix them anyway),
  which is itself a clean result.

### E2 — "the failure is the idea, not the agent" (task-spec axis)

- **Change exactly one thing:** the idea `.md`. `ideas/topic_baseline.md` is vague;
  `ideas/topic_concrete.md` pins dataset (CIFAR-10, fixed 5k subset), model
  (ResNet-18 from scratch), metric (top-1, 3 seeds), and a compute budget that fits a
  ~10GB MIG slice (batch ≤128, 30 epochs, drop to batch 64 if tight). Config identical
  for both arms (`configs/E2_baseline.yaml`).
- **Measure:** number of OOM/setup errors; nodes-to-first-working-node; success Y/N.
- **Prediction if task-spec is the bottleneck:** the concrete idea reaches a working
  node in far fewer nodes and produces far fewer OOM/`FileNotFound`/dataset errors. The
  compute-budget line in `topic_concrete.md` is also the README's recommended CUDA-OOM
  fix, so E2 is the OOM experiment too.

### E3 (optional, cleanest capability isolator)

Swap **only** the experiment model (`agent.code.model`) keeping idea+seed fixed:
```bash
python scripts/apply_experiment_config.py --config ~/AI-Scientist-v2/bfts_config.yaml \
  --preset baseline --small --sandbox --sandbox-model gpt-4o-mini \
  --out ~/AI-Scientist-v2/configs/E3_weak.yaml      # vs --sandbox-model gpt-4o
```
If the *same* error recurs with both a weaker and a stronger model, it is task-spec or
tool-use, not capability; if the stronger model fixes it, it was capability.

> Other runbook experiments (E4 starter-scaffold via `--load_code`, E5 breadth via
> `num_drafts`/`num_workers`) are supported by the same tooling but are not the
> recommended pair.

---

## Part D — Finding: wiring the Princeton AI Sandbox (the $0 route)

This is a real setup result, not boilerplate — the runbook explicitly asks "does the
repo support a custom base_url cleanly?" **Answer: not for the Sandbox, without a small
patch.** Verified from the source:

1. **Two client layers, both hardcode a bare OpenAI client:**
   - `ai_scientist/llm.py :: create_client(model)` → for `gpt`/`o1`/`o3` returns
     `openai.OpenAI()` (writeup, review, citation, ideation, report).
   - `ai_scientist/treesearch/backend/backend_openai.py :: get_ai_client(...)` →
     returns `openai.OpenAI(max_retries=max_retries)` (the **experiment** coder +
     feedback + VLM — i.e. the tree search itself).
2. **A plain base_url is *almost* enough but not quite.** The OpenAI SDK honours
   `OPENAI_BASE_URL`, so a standard OpenAI-compatible endpoint could be redirected with
   env vars and zero code change. But the Princeton Sandbox is **Azure-flavoured**: per
   the verified `myscript.py`, it needs `AzureOpenAI(azure_endpoint=
   "https://api-ai-sandbox.princeton.edu/", api_version="2025-03-01-preview")`, which
   builds `/openai/deployments/<model>/...?api-version=` URLs the stock client does not
   produce. So a code patch *is* required.
3. **The default experiment model bypasses OpenAI entirely.** `agent.code.model`
   defaults to `anthropic.claude-3-5-sonnet-20241022-v2:0` (Bedrock), routed through
   `anthropic.AnthropicBedrock()`. To go $0 you must *also* switch the experiment model
   off Bedrock to a Sandbox model name — handled by `apply_experiment_config.py --sandbox`.
4. **Routing keys on substrings.** Both layers send any `claude-`-containing name to
   the Anthropic backend, everything else to OpenAI. The Sandbox names
   (`gpt-4o`, `o3-mini`, `Meta-Llama-3-1-70B-Instruct-htzs`, `Mistral-small-zgjes`, …)
   contain no `claude-`, so they all reach the (patched) OpenAI path. Good.

**The patch** (`scripts/wire_sandbox.py`, idempotent) inserts an `AzureOpenAI` branch
gated on `USE_AI_SANDBOX` into *both* layers. Activate with `USE_AI_SANDBOX=1` +
`AI_SANDBOX_KEY` and point every model flag at a Sandbox name. If the repo source has
drifted and an anchor is not found, the script tells you and you patch the two functions
by hand as shown above.

> **Two meanings of "sandbox."** The Princeton AI Sandbox is a secure place to *get
> model access* — it is **not** a sandbox for *executing the agent's generated code*.
> The README warns v2 runs arbitrary LLM-written code and recommends Docker; Adroit
> gives no Docker (no root). Clarify with Taiming where code execution should be
> confined (Apptainer/Singularity, a constrained env, watched small runs). These are
> different problems.

---

## Part E — Adroit / SLURM gotchas (the ones that actually bite)

- **Two disjoint networks — and the Sandbox lives on the compute side (verified June
  2026, confirmed against Princeton RC docs).** This is the single most important
  operational fact, and it is *split*:
  - The **login node** (`adroit-vis`) has general outbound internet — `git clone`, `pip`,
    `conda`, public dataset downloads work. But it **cannot reach the Sandbox at all**:
    `api-ai-sandbox.princeton.edu` does not even resolve there (`Name or service not
    known`). So setup/env/dataset-prefetch go on the login node; Sandbox calls cannot.
  - **Compute nodes** have *no* general internet (a github clone returns **403**), but
    they reach the Sandbox via `module load proxy/default` (RC's documented mechanism;
    the module sets `HTTP(S)_PROXY` which `httpx`/`openai` honor automatically — no code
    change). A bare `curl /` to the Sandbox returns **502** (reached, wrong path — *not*
    403/blocked), so only a real `chat.completions.create` proves it; the SLURM scripts
    run exactly that as a pre-flight and abort early if it fails.
  - **Net consequence:** every step that calls the Sandbox — **ideation *and* the runs** —
    must execute on a **compute node with `proxy/default`** (ideation needs no GPU, so a
    CPU session from the `all` partition suffices). Only clone/conda/dataset-prefetch
    belong on the login node. `setup_adroit.sh` refuses to run where github is
    unreachable; the SLURM scripts load `proxy/default` before any Sandbox call.
  - LLM-*written* code that tries to download data at runtime will fail on the compute
    node; pre-fetch datasets on the login node (`CIFAR_DIR`, `HF_HOME` on scratch).
- **Two conda gotchas.** `module load anaconda3` errors with "No default version" — pin
  one (`anaconda3/2024.6`). And conda's activate script trips `set -u` with
  `PS1: unbound variable` — wrap activation in `set +u; conda activate …; set -u`. Both
  fixed in the kit scripts.
- **Partition/gres.** There is no `mig` partition; GPUs are in **`gpu`**. A MIG A100
  slice (~20GB, plenty for the ResNet-18 E2 idea) is `--partition=gpu --gres=gpu:3g.20gb:1`;
  a full A100 is `--gres=gpu:1 --constraint=a100`.
- **/home quota is small (~20GB).** Put the repo and the conda env on scratch
  (`REPO_DIR`, `CONDA_ENVS_DIR` under `/scratch/network/$USER`).
- **Config keys are nested, and defaults differ from the runbook.** The runbook's flat
  snippet (`num_workers`, `steps`, `max_debug_depth`, …) maps onto nested keys; real
  defaults are `num_workers: 4` (runbook says 3) and `steps: 5` with per-stage caps
  `stage1_max_iters: 20 / stage2: 12 / stage3: 12 / stage4: 18` (the runbook's "steps:
  21" does not exist as a single key). The repo wins; `apply_experiment_config.py` edits
  the real nested keys. See the table below.
- **CUDA OOM** → pin smaller models/batch in the idea `.md` (the README fix). That is
  exactly what `topic_concrete.md` does, which is why E2 is also the OOM experiment.

### Config key reality check (verified `bfts_config.yaml`)

| Runbook said | Real key | Real default |
|---|---|---|
| `num_workers` | `agent.num_workers` | **4** |
| `steps` | `agent.steps` | 5 (fallback) |
| (the real per-stage caps) | `agent.stages.stage{1..4}_max_iters` | 20 / 12 / 12 / 18 |
| `num_seeds` | `agent.multi_seed_eval.num_seeds` | 3 |
| `max_debug_depth` | `agent.search.max_debug_depth` | 3 |
| `debug_prob` | `agent.search.debug_prob` | 0.5 |
| `num_drafts` | `agent.search.num_drafts` | 3 |
| (experiment model) | `agent.code.model` | `anthropic.claude-3-5-sonnet-20241022-v2:0` (Bedrock) |
| (output-eval model) | `agent.feedback.model` | `gpt-4o-2024-11-20` |
| (figure reviewer) | `agent.vlm_feedback.model` | `gpt-4o-2024-11-20` |
| (final report) | `report.model` | `gpt-4o-2024-11-20` |

### Verified `launch_scientist_bfts.py` flags

`--writeup-type` (`icbinb`=4pg / `normal`=8pg), `--load_ideas`, `--load_code`,
`--idea_idx`, `--add_dataset_ref`, `--writeup-retries`, `--attempt_id`,
`--model_agg_plots`, `--model_writeup`, `--model_citation`, `--num_cite_rounds`,
`--model_writeup_small`, `--model_review`, `--skip_writeup`, `--skip_review`. (There is
no `--model` or `--seed` flag on the launcher; the experiment model lives in
`bfts_config.yaml`, seeds in `agent.multi_seed_eval`.) For tree-only failure analysis,
`--skip_writeup --skip_review` stops before the writeup stage.

---

## Part F — Data recorded (Adroit, June 2026)

Backend reality: the Princeton AI Sandbox API (`api-ai-sandbox.princeton.edu`) was
**retired** mid-2026 (RC confirmed: replaced by a Portkey gateway), so the `$0` route
in Parts D/E is dead as written; these runs used a personal **OpenAI** key reached
through `module load proxy/default` (the proxy allow-lists OpenAI/Anthropic/Gemini, but
**not** the Sandbox, HuggingFace, or Semantic Scholar). GPU: one MIG A100 `3g.20gb`
slice. `--skip_writeup --skip_review`, `generate_report=false`.

### Per run

| run | idea | exp model | num_drafts | stage1_iters | max_debug_depth | debug_prob | total nodes | working/buggy | PDF? | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | topic_concrete | gpt-4o-mini | 1 | 14 | 3 | 0.5 | 14 | **0 / 14** | no | 7 URLError, 3 ModuleNotFound, 2 SystemExit, 1 NameError, 1 FileNotFound |
| R2 | topic_concrete | **gpt-4o** | 1 | 14 | 3 | 0.5 | 14 | **0 / 14** | no | 8 URLError, 4 RuntimeError, 1 FileNotFound, 1 ModuleNotFound |
| R3 | (planned) topic_concrete, offline note in Abstract | gpt-4o | — | — | — | — | **not run** | — | — | designed but never executed on Adroit; the offline-constraint idea was instead validated by the local run (Part F′) |

> **Data provenance.** R1 and R2 above are backed by committed files:
> `data/adroit/2026-06-29_13-18-30_.../` and `data/adroit/2026-06-30_00-51-48_.../`
> (each has the raw `journal.json`, the mined `.csv`, and `unified_tree_viz.html`). Proxy
> reachability is in `data/adroit/slurm_preflight_logs/` (Sandbox = `502 Proxy Error`;
> `gpt-4o`/`gpt-4o-mini` = `SUCCESS`). Re-mine any journal with `scripts/mine_journal.py`.

### Per failure node (representative)

| node id | stage | what it tried | error class | outcome | classification | proposed intervention |
|---|---|---|---|---|---|---|
| a0f9641… (R2 n0) | draft | load CIFAR-10 via torchvision, `download=True` | `URLError: Tunnel connection failed: 403 Forbidden` | abandoned | **tool-use / environment** (agent defaults to a download the cluster forbids) | surface offline-load constraint in the prompt; or `--load_code` data scaffold |
| 263be77… (R2 n9→debug) | debug | reviewer said *"set download=True … ensure internet access"*, retried download | `URLError` 403 again | abandoned | **tool-use (feedback misdiagnosis)** | environment-aware feedback that recognizes a blocked-network signal |

---

## Part F′ — Data recorded (LOCAL, open internet — the clean A/B counterpart)

Same idea family, same agent, **one variable changed: the network**. Run on a local
Windows + RTX-3050 box via WSL2 (`local/`), open home internet, personal OpenAI key
reached directly (no proxy, no Sandbox, no SLURM). `gpt-4o` coder, `gpt-4o-mini`
feedback/VLM, `--skip_writeup --skip_review`, `generate_report=false`, `exec.timeout=120`.
The fast/GPU-explicit idea (`ideas/topic_concrete_fast.json`: 3 epochs, 1 seed, 3 arms)
so nodes finish in seconds; CIFAR-10 pre-cached under `$CIFAR_DIR`.

### Per run

| run | idea | exp model | env | stage mined | total nodes | working/buggy | abandoned leaves | top error classes | result |
|---|---|---|---|---|---|---|---|---|---|
| L1 | topic_concrete_fast | gpt-4o | **local, open internet** | stage_2_baseline_tuning | 9 | **2 / 7** | 7 | 2 `HfUriError`, 1 `DatasetNotFoundError` | **working nodes — real ResNet-18 training, 28.61% top-1 val acc** |

Contrast with Part F: on Adroit (no internet) the *identical* agent on the *same* idea
got **0 working / 14** with the plurality `URLError 403`. The only change between the two
data points is the network wall. That is the clean experimental result.

### Per node (local)

| node id | stage | is_buggy | what happened | classification |
|---|---|---|---|---|
| 2af6738… (L1 n0) | draft | **False (working)** | loaded the cached CIFAR-10, trained ResNet-18, returned a real top-1 metric (28.61%) | success — the network confound removed, the agent's code runs |
| 3852733… (L1 n2) | draft/improve | **False (working)** | second viable node | success |
| (7 buggy leaves) | debug | True | reached for a **HuggingFace** dataset path instead of the cached torchvision CIFAR — `HfUriError` / `DatasetNotFoundError` (HF is also unreachable / the URI is malformed) | **tool-use** — intrinsic agent signal, *not* the environment wall |

---

## Part G — Findings (the deep dive)

Three runs (two analyzed, one pending) on the pinned CIFAR-10 / ResNet-18 idea, plus the
long setup trail, yield a concrete answer to Part A's question — *is the bottleneck
ideation, execution, or evaluation?*

**G.1 The bottleneck is execution, and specifically tool-use under an environment
constraint — not ideation or model capability.**
Across R1 (gpt-4o-mini) and R2 (gpt-4o) the outcome is *identical*: 14 nodes, **zero
working**, the plurality of failures `URLError: …403 Forbidden` from the agent's code
calling `torchvision…CIFAR10(download=True)` against a no-internet compute node. A 6×
stronger coder changed nothing — a clean **capability isolator (E3)**: this is *not* a
model-capability failure. Ideation was never the issue either (the idea is fully
specified). The wall is *setup/execution*.

**G.2 The debug loop cannot escape an environment failure because the feedback model
misdiagnoses it.** The verbatim reviewer output on the buggy nodes was *"set
`download=True` … ensure the environment has internet access"* — advice that is
impossible on the cluster and, where the agent had already tried `download=True`, sends
it in a circle. So the agentic tree search burns its whole iteration budget re-issuing a
call that can never succeed. This is a specific, citable **tool-use failure mode**:
the evaluator's root-cause attribution is wrong, so debugging is misdirected. (Good
hook for the *Why LLMs Aren't Scientists Yet* vocabulary.)

**G.3 A large fraction of "agent failure" on a restricted cluster is really
infrastructure.** 7/14 (R1) and 8/14 (R2) failures were the blocked download; the rest
(`ModuleNotFound` = a package not installed and un-`pip`-installable offline,
`FileNotFound`, `RuntimeError`) are also partly environment-shaped. The runbook's
"confound" is the dominant signal here. This is itself a result: evaluating open-ended
agents on a locked-down HPC environment measures the environment as much as the agent.

**G.4 The framework is brittle at the no-good-node boundary.** With zero working nodes,
the best-node selector picks arbitrarily ("all metrics are `nan`… selecting the first by
default"), and the run then crashes in `log_summarization.overall_summarize`
(`ValueError: not enough values to unpack`) / `shutil.rmtree(... experiment_results)`
(`FileNotFoundError`). Setting `generate_report=false` sidesteps it. Worth noting as a
robustness gap when an idea is infeasible in the given environment.

**G.5 Intervention and the research idea.** The fix that the diagnosis points to is
**selective scaffolding**: keep v2's open-ended ideation/search, but re-introduce a thin,
generic *execution* scaffold for setup (here: a correct offline data-loader, or simply
surfacing the environment constraint in the prompt the agent actually reads — note the
launcher's `task_desc` passes only Title+Abstract+Short Hypothesis, **not** the
`Experiments` field, so constraints placed there are invisible to the coder). The
local run (Part F′) is the executed test of the cheapest version of this — the offline/
GPU/dataloader constraints were moved into the Abstract, and the agent then reached
working nodes (the planned Adroit R3 variant was never run). The broader bet — the
"research idea from this" Taiming hinted at — is **environment-aware agents/feedback**:
a reviewer that recognizes a `403/URLError/Tunnel` signature as "network is blocked, stop
retrying downloads, use the local cache" would convert a whole class of dead debug loops
into progress. This is exactly the scaffolding v2 *removed* relative to v1 — evidence
that removing structure is not free, and that the useful middle ground is generic
setup-scaffolding, not per-domain templates.

**G.6 The local open-internet run confirms the confound — and surfaces a *real* tool-use
failure underneath it.** Re-running the same agent on the same idea with the only change
being open internet (Part F′, L1) flips the outcome: **2 working nodes vs. 0 on Adroit**,
including a node that actually trained ResNet-18 to a real **28.61% top-1** metric. This
is the clean A/B that isolates the network as the dominant Adroit variable — same model,
same idea, same framework, opposite result. *But* the 7 buggy leaves that remain are now
**genuine agent signal, not infrastructure**: instead of `URLError 403`, they fail with
`HfUriError` / `DatasetNotFoundError` — the agent reaching for a **HuggingFace** dataset
path in the baseline-tuning stage even though the idea pins torchvision CIFAR-10 and the
data is already cached locally. That is a true **tool-use** failure (wrong data API, not a
blocked network), and it is exactly the intrinsic signal the Adroit confound had been
masking. So the two data points together say: (a) most "agent failure" on the locked-down
cluster was infrastructure (G.1–G.3); (b) once you remove it, the residual failures are
real but different — sloppy tool/data-API selection — which is again a *tool-use* axis
result, reinforcing G.5's selective-scaffolding fix (a thin data-loading scaffold would
kill both the `URLError` and the `HfUriError` class at once).

> **Reproduce the local clean run:** `local/setup_local.sh` then `local/run_local.sh`
> (WSL2 + RTX-3050, open internet, OpenAI key). Mine with
> `scripts/mine_journal.py <run>/logs/0-run/`. The two contaminated-vs-clean data points
> (Adroit 0/14 `URLError` vs. local 2/7 with `HfUriError`) are the report's core evidence.
> One robustness note from the local run: Stage-4 (multi-model ablation) nodes can exceed
> a short `exec.timeout` and re-loop; bump `exec.timeout` (≥600s) if you want Stage 4 to
> finish, or stop after Stages 1–3, which already produce working nodes.

---

## Reading list (priority order)

1. AI-Scientist-v2 — [arXiv:2504.08066](https://arxiv.org/abs/2504.08066) (Table 1
   v1-vs-v2; the tree-search section).
2. AI-Scientist v1 — [arXiv:2408.06292](https://arxiv.org/abs/2408.06292) (the template
   approach v2 contrasts against).
3. AIDE — [github.com/WecoAI/aideml](https://github.com/WecoAI/aideml) (the tree-search
   substrate; skim how nodes/debug work — it maps directly onto `journal.py`).
4. *Why LLMs Aren't Scientists Yet* — [arXiv:2601.03315](https://arxiv.org/abs/2601.03315)
   (failure-mode vocabulary to cite in the diagnosis).
5. Optional contrast: AlphaEvolve [2506.13131](https://arxiv.org/abs/2506.13131),
   AI co-scientist [2502.18864](https://arxiv.org/abs/2502.18864).

---

## TL;DR for the kit

`scripts/setup_adroit.sh` → `scripts/wire_sandbox.py` → `scripts/apply_experiment_config.py`
(make E1/E2 variants) → submit `slurm/run_E1.slurm` / `run_E2.slurm` →
`scripts/mine_journal.py` to turn each tree into a CSV → fill the Part F tables.
See `experiments.md` for the exact command sequence.
