#!/usr/bin/env python3
"""
apply_experiment_config.py - Produce a bfts_config.yaml variant for a controlled
experiment, editing only the keys you ask for.

IMPORTANT: the real bfts_config.yaml is NESTED (verified). The runbook's flat
snippet (num_workers / steps / max_debug_depth / debug_prob / num_drafts) maps onto:

    agent.num_workers
    agent.steps                     # global fallback; per-stage caps below usually win
    agent.stages.stage1_max_iters   # <- this is the "steps per stage" knob
    agent.stages.stage2_max_iters
    agent.stages.stage3_max_iters
    agent.stages.stage4_max_iters
    agent.multi_seed_eval.num_seeds
    agent.search.max_debug_depth    # E1
    agent.search.debug_prob         # E1
    agent.search.num_drafts         # E5 / "breadth"
    agent.code.model                # EXPERIMENT model (default: bedrock claude-3-5-sonnet)
    agent.feedback.model
    agent.vlm_feedback.model
    report.model

Defaults in the real repo: num_workers=4, steps=5, stage1=20/stage2=12/stage3=12/
stage4=18, num_seeds=3, max_debug_depth=3, debug_prob=0.5, num_drafts=3.

Presets:
  E1        tool-use hypothesis: max_debug_depth 3->8, debug_prob 0.5->0.9
  baseline  no search change (use as the A side of E1, and for E2)
  --small   cheap A/B: lower stage iters + num_drafts=1 (pair with either preset)
  --sandbox swap experiment/feedback/vlm/report models to a Sandbox model name
            (REQUIRED for the $0 route -- moves the experiment model off Bedrock-Claude)

You can also set any nested key directly:  --set agent.search.debug_prob=0.9

Examples:
  # E1 cheap, Sandbox-routed:
  python apply_experiment_config.py --config ~/AI-Scientist-v2/bfts_config.yaml \\
      --preset E1 --small --sandbox --out ~/AI-Scientist-v2/configs/E1_debug8.yaml
  # E2 baseline (the idea .md is the variable, config unchanged except cheap+sandbox):
  python apply_experiment_config.py --config ~/AI-Scientist-v2/bfts_config.yaml \\
      --preset baseline --small --sandbox --out ~/AI-Scientist-v2/configs/E2_baseline.yaml
"""
import argparse
import os
import sys

try:
    from ruamel.yaml import YAML  # preserves comments/order
    _yaml = YAML()
    _yaml.preserve_quotes = True

    def load(path):
        with open(path) as f:
            return _yaml.load(f)

    def dump(data, path):
        with open(path, "w") as f:
            _yaml.dump(data, f)
    BACKEND = "ruamel (comments preserved)"
except Exception:  # pragma: no cover
    import yaml as _pyyaml

    def load(path):
        with open(path) as f:
            return _pyyaml.safe_load(f)

    def dump(data, path):
        with open(path, "w") as f:
            _pyyaml.safe_dump(data, f, sort_keys=False)
    BACKEND = "pyyaml (comments dropped)"


def coerce(v):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def set_nested(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d or d[k] is None:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.expanduser("~/AI-Scientist-v2/bfts_config.yaml"))
    ap.add_argument("--out", required=True, help="where to write the variant")
    ap.add_argument("--preset", choices=["E1", "baseline"], default="baseline")
    ap.add_argument("--small", action="store_true", help="cheap A/B knobs (low iters, 1 draft)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--sandbox", action="store_true", help="point models at Sandbox")
    ap.add_argument("--sandbox-model", default="gpt-4o", help="Sandbox model name (default gpt-4o)")
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    help="extra nested key, e.g. agent.search.debug_prob=0.9 (repeatable)")
    args = ap.parse_args()

    if not os.path.isfile(args.config):
        sys.exit(f"config not found: {args.config}")
    cfg = load(args.config)

    changes = []

    def apply(k, v):
        set_nested(cfg, k, v)
        changes.append(f"{k} = {v!r}")

    if args.preset == "E1":
        apply("agent.search.max_debug_depth", 8)
        apply("agent.search.debug_prob", 0.9)

    if args.small:
        apply("agent.search.num_drafts", 1)
        apply("agent.stages.stage1_max_iters", 14)
        apply("agent.stages.stage2_max_iters", 8)
        apply("agent.stages.stage3_max_iters", 8)
        apply("agent.stages.stage4_max_iters", 10)

    if args.workers is not None:
        apply("agent.num_workers", args.workers)
        # num_seeds should equal num_workers when workers < 3, else 3 (per the config comment)
        apply("agent.multi_seed_eval.num_seeds", args.workers if args.workers < 3 else 3)

    if args.sandbox:
        m = args.sandbox_model
        # The experiment coder MUST move off Bedrock-Claude for Sandbox routing.
        apply("agent.code.model", m)
        apply("agent.feedback.model", m)
        apply("agent.vlm_feedback.model", m)
        apply("report.model", m)

    for s in args.sets:
        if "=" not in s:
            sys.exit(f"--set expects key=value, got {s!r}")
        k, v = s.split("=", 1)
        apply(k.strip(), coerce(v.strip()))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dump(cfg, args.out)

    print(f"YAML backend: {BACKEND}")
    print(f"base   : {args.config}")
    print(f"variant: {args.out}")
    print("changes:")
    for c in changes:
        print(f"  {c}")
    if not changes:
        print("  (none -- pure copy)")
    if args.sandbox and args.sandbox_model.lower().startswith(("o1", "o3")):
        print("\nNOTE: o1/o3 models ignore temperature; the backend already special-cases them.")


if __name__ == "__main__":
    main()
