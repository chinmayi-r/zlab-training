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

Tie-off (filled in once Part B/C are done): across the runs I mined, the bottleneck is
predominantly **execution / tool-use** — the agent has reasonable ideas and writes
plausible code, but loses runs to setup and debugging failures (environment, data
loading, OOM, repeated re-submission of broken code) rather than to bad ideation or bad
evaluation. If that holds, the highest-leverage investment is **execution scaffolding**
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

- **Outbound internet is off by default on compute nodes.** `module load proxy/default`
  on the *compute* node (not login). The scripts sanity-check this with a `curl`. If it
  still fails after the module load, that is a legitimate finding to document and raise
  with RC/Taiming — and note that some traffic from the LLM-*written* code may not honor
  the proxy env vars, another reason to keep runs small and watched.
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

## Part F — Data to record

### Per run

| run | idea | exp model | num_workers | stage1_iters | num_drafts | max_debug_depth | debug_prob | total nodes | working/buggy | PDF? | wall-clock | approx $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | | | | | | 3 | 0.5 | | | | | |
| E1 | | | | | | 8 | 0.9 | | | | | |
| E2-vague | topic_baseline | | | | | | | | | | | |
| E2-concrete | topic_concrete | | | | | | | | | | | |

### Per failure node (the heart of the report — aim for ≥2 filled rows)

| node id | stage | what it tried (plan) | error class | # debug attempts | outcome | my classification | proposed intervention |
|---|---|---|---|---|---|---|---|
| | | | | | abandoned / fixed | task-spec / capability / tool-use | |
| | | | | | | | |

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
