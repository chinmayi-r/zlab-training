# zlab Training

Training materials and weekly reports for the [Princeton ZLab Warmup Program](https://github.com/zlab-princeton-internal/warmup-program).

## Structure

```
nano-vllm/
├── report.md            # the report: one question (KV-cache preemption cost), traced through the code
├── setup.md             # Adroit setup + the flash-attn install fix
└── experiments/
    ├── README.md        # the experiment: what it measures, how to run, how to read results
    ├── exp_preempt.py   # instrumented driver (monkeypatches nano-vllm; never edits the repo)
    └── run_preempt.slurm
```

## Reports

| Week | Question | Folder |
|------|----------|--------|
| 1–2 | nano-vllm: what does it cost when the KV cache runs out mid-generation? | `nano-vllm/` |

The report follows the warmup-program style: not "I ran inference", but one
question traced to the bottom of the code, with a hypothesis tested by experiment.
