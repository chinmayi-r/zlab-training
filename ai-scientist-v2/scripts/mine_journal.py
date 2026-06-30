#!/usr/bin/env python3
"""
mine_journal.py - Turn an AI-Scientist-v2 tree-search journal into a per-node CSV
for failure analysis (Part B of the runbook).

Each run writes a journal under:
    experiments/<timestamp_ideaname>/logs/0-run/   (look for journal*.json / *.json)

The journal is a serialized `Journal` of `Node`s. Fields below are taken from the
REAL Node.to_dict() in ai_scientist/treesearch/journal.py (verified), so we only
read keys that actually exist; everything is accessed defensively with .get() so a
schema drift degrades to blanks instead of crashing.

Real Node fields we use:
    id, step, parent (id ref), children (id refs), is_buggy, is_buggy_plots,
    plan, code, _term_out / term_out, exc_type, exc_info, exc_stack,
    metric, analysis, plots_generated

Derived columns:
    stage         draft / debug / improve   (draft = no parent; debug = parent is buggy)
    debug_depth   length of the consecutive buggy-ancestor chain ending at this node
    first_tb_line first traceback-ish line from exc_stack or terminal output
    n_children    out-degree (a leaf that is buggy = abandoned line of attack)

Usage:
    python mine_journal.py experiments/<run>/logs/0-run/journal.json -o nodes.csv
    python mine_journal.py experiments/<run>/logs/0-run/            # auto-finds json
"""
import argparse
import csv
import glob
import json
import os
import sys


def load_journal(path):
    """Accept a journal file or a directory; return the list of node dicts."""
    if os.path.isdir(path):
        # Look at the top level, then recurse (the journal often lives in a
        # stage_*/ subdir, e.g. logs/0-run/stage_1_.../journal.json).
        cands = (glob.glob(os.path.join(path, "journal*.json"))
                 or glob.glob(os.path.join(path, "*.json"))
                 or glob.glob(os.path.join(path, "**", "journal*.json"), recursive=True))
        if not cands:
            sys.exit(f"No journal.json found under {path} (searched recursively)")
        path = cands[0]
        print(f"Using journal: {path}", file=sys.stderr)
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        nodes = data.get("nodes", data.get("journal", []))
    else:
        nodes = data
    if not isinstance(nodes, list):
        sys.exit(f"Could not find a node list in {path}")
    return nodes, path


def node_id(n):
    return n.get("id") or n.get("node_id") or ""


def parent_id(n):
    # Real journal.json uses "parent_id"; older/alt forms may use "parent".
    p = n.get("parent_id")
    if p is None:
        p = n.get("parent")
    if isinstance(p, dict):
        return p.get("id", "")
    return p or ""


def first_tb_line(n):
    """Best-effort first informative traceback line."""
    stack = n.get("exc_stack")
    if isinstance(stack, list) and stack:
        first = stack[0]
        return (str(first[0]) if isinstance(first, (list, tuple)) else str(first))[:200]
    out = n.get("term_out") or n.get("_term_out") or ""
    if isinstance(out, list):
        out = "".join(str(x) for x in out)
    for line in reversed(str(out).splitlines()):
        s = line.strip()
        if any(k in s for k in ("Error", "Exception", "Traceback", "error:")):
            return s[:200]
    return ""


def oneliner(text, n=90):
    if not text:
        return ""
    s = " ".join(str(text).split())
    return s[:n] + ("…" if len(s) > n else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("journal", help="journal.json file OR the logs/0-run/ directory")
    ap.add_argument("-o", "--out", default="nodes.csv", help="output CSV (default nodes.csv)")
    args = ap.parse_args()

    nodes, src = load_journal(args.journal)
    by_id = {node_id(n): n for n in nodes}

    def buggy(n):
        v = n.get("is_buggy")
        return bool(v) if v is not None else None

    def debug_depth(n):
        """Consecutive buggy ancestors (the agent's debug chain)."""
        depth, cur, seen = 0, n, set()
        while True:
            pid = parent_id(cur)
            if not pid or pid in seen or pid not in by_id:
                break
            seen.add(pid)
            parent = by_id[pid]
            if buggy(parent):
                depth += 1
                cur = parent
            else:
                break
        return depth

    child_count = {}
    for n in nodes:
        pid = parent_id(n)
        if pid:
            child_count[pid] = child_count.get(pid, 0) + 1

    rows = []
    for n in nodes:
        nid = node_id(n)
        pid = parent_id(n)
        is_b = buggy(n)
        nch = child_count.get(nid, 0)
        if not pid:
            stage = "draft"
        elif by_id.get(pid) is not None and buggy(by_id[pid]):
            stage = "debug"
        else:
            stage = "improve"
        # abandoned = buggy leaf (no child ever fixed it)
        outcome = "?"
        if is_b is True:
            outcome = "abandoned(leaf)" if nch == 0 else "buggy(retried)"
        elif is_b is False:
            outcome = "working"
        rows.append({
            "id": nid,
            "parent": pid,
            "step": n.get("step", ""),
            "stage": stage,
            "is_buggy": is_b,
            "outcome": outcome,
            "n_children": nch,
            "debug_depth": debug_depth(n),
            "exc_type": oneliner(n.get("exc_type"), 60),
            "first_tb_line": first_tb_line(n),
            "metric": oneliner(n.get("metric"), 40),
            "plan": oneliner(n.get("plan")),
        })

    cols = ["id", "parent", "step", "stage", "is_buggy", "outcome", "n_children",
            "debug_depth", "exc_type", "first_tb_line", "metric", "plan"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    total = len(rows)
    nb = sum(1 for r in rows if r["is_buggy"] is True)
    ng = sum(1 for r in rows if r["is_buggy"] is False)
    drafts = sum(1 for r in rows if r["stage"] == "draft")
    abandoned = sum(1 for r in rows if r["outcome"] == "abandoned(leaf)")
    print(f"\nsource     : {src}")
    print(f"total nodes: {total}")
    print(f"  working  : {ng}")
    print(f"  buggy    : {nb}   (abandoned leaves: {abandoned})")
    print(f"  drafts   : {drafts}")
    print(f"wrote      : {args.out}")
    # top error classes
    from collections import Counter
    errs = Counter(r["exc_type"] for r in rows if r["exc_type"])
    if errs:
        print("top error classes:")
        for e, c in errs.most_common(8):
            print(f"  {c:3d}  {e}")


if __name__ == "__main__":
    main()
