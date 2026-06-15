"""
exp_varlen.py — does LIFO eviction treat long requests unfairly?

The main experiment used a uniform 600/600 workload. Real traffic mixes short and
long requests. Hypothesis: under memory pressure, LIFO eviction disproportionately
hurts LONG requests, because (a) short requests finish quickly and leave the
running set before the squeeze, while (b) long requests stay resident longer, are
more likely to be the ones crossing block boundaries when the cache is full, and
(c) cost more to re-prefill when evicted. If true, long-request latency blows up
while short-request latency is barely touched — an eviction-policy fairness problem.

We send a 50/50 mix of short (max_tokens=100) and long (max_tokens=1000) requests
and measure, per group: how often each request is evicted and its end-to-end
latency. Run with --policy lifo and --policy cheapest to compare the two policies'
fairness. No edits to the repo — we patch the live instances.

Usage (GPU node):
  python exp_varlen.py --model /scratch/.../Qwen3-0.6B --n 64 \\
      --gpu-mem-util 0.30 --policy lifo --out varlen_lifo.csv
  python exp_varlen.py ... --policy cheapest --out varlen_cheapest.csv
"""

import argparse
import csv
import time
from collections import deque, Counter

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.scheduler import Scheduler


# --- per-request instrumentation (monkeypatch) ------------------------------
T0 = 0.0
FINISH = {}                 # seq_id -> (latency_s, max_tokens)
EVICT = Counter()           # seq_id -> number of times preempted

_orig_preempt = Scheduler.preempt
_orig_postprocess = Scheduler.postprocess


def _preempt(self, seq):
    EVICT[seq.seq_id] += 1
    return _orig_preempt(self, seq)


def _postprocess(self, seqs, token_ids, is_prefill):
    out = _orig_postprocess(self, seqs, token_ids, is_prefill)
    now = time.perf_counter()
    for seq in seqs:
        if seq.is_finished and seq.seq_id not in FINISH:
            FINISH[seq.seq_id] = (now - T0, seq.max_tokens)
    return out


Scheduler.preempt = _preempt
Scheduler.postprocess = _postprocess


# --- the cheapest-to-redo policy (same trick as exp_preempt) ----------------
class CheapestDeque(deque):
    def pop(self):
        victim = min(self, key=lambda s: s.num_tokens)
        self.remove(victim)
        return victim


def build_prompts(llm, n, prompt_tokens):
    tok = llm.tokenizer
    filler = "the quick brown fox jumps over the lazy dog and then keeps running "
    prompts = []
    for i in range(n):
        ids = tok.encode(f"Request {i}: " + filler * (prompt_tokens // 8 + 4))
        while len(ids) < prompt_tokens:
            ids += tok.encode(filler)
        prompts.append(ids[:prompt_tokens])
    return prompts


def summarize(group_label, lat, evicts):
    """lat: list of latencies (s); evicts: list of eviction counts (same order)."""
    n = len(lat)
    if n == 0:
        return {}
    lat_sorted = sorted(lat)
    evicted = sum(1 for e in evicts if e > 0)
    return {
        "group": group_label,
        "count": n,
        "requests_evicted": evicted,
        "frac_evicted": round(evicted / n, 3),
        "total_evictions": sum(evicts),
        "mean_evictions": round(sum(evicts) / n, 2),
        "mean_latency_s": round(sum(lat) / n, 2),
        "p50_latency_s": round(lat_sorted[n // 2], 2),
        "max_latency_s": round(max(lat), 2),
    }


def main():
    global T0
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=64, help="total requests (half short, half long)")
    p.add_argument("--gpu-mem-util", type=float, default=0.30)
    p.add_argument("--prompt-tokens", type=int, default=200)
    p.add_argument("--short-max", type=int, default=100)
    p.add_argument("--long-max", type=int, default=1000)
    p.add_argument("--policy", choices=["lifo", "cheapest"], default="lifo")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--out", default="varlen.csv")
    args = p.parse_args()

    import os
    model = os.path.expanduser(args.model)

    llm = LLM(model, enforce_eager=True, gpu_memory_utilization=args.gpu_mem_util,
              max_num_seqs=max(256, args.n), max_model_len=args.max_model_len)
    if args.policy == "cheapest":
        llm.scheduler.running = CheapestDeque(llm.scheduler.running)

    n = args.n
    prompts = build_prompts(llm, n, args.prompt_tokens)
    # interleave short/long so admission order doesn't bias which group is newest
    sps = []
    for i in range(n):
        mt = args.short_max if i % 2 == 0 else args.long_max
        sps.append(SamplingParams(temperature=0.6, max_tokens=mt, ignore_eos=True))

    FINISH.clear(); EVICT.clear()
    print(f"# model={model}")
    print(f"# n={n} policy={args.policy} gpu_mem_util={args.gpu_mem_util} "
          f"short_max={args.short_max} long_max={args.long_max}")

    T0 = time.perf_counter()
    outputs = llm.generate(prompts, sps, use_tqdm=False)
    wall = time.perf_counter() - T0

    # group results by short vs long (classified by the request's max_tokens)
    short_lat, short_ev, long_lat, long_ev = [], [], [], []
    for seq_id, (lat, mt) in FINISH.items():
        ev = EVICT[seq_id]
        if mt == args.short_max:
            short_lat.append(lat); short_ev.append(ev)
        else:
            long_lat.append(lat); long_ev.append(ev)

    rows = [summarize("short", short_lat, short_ev),
            summarize("long", long_lat, long_ev)]
    for r in rows:
        r["policy"] = args.policy
        r["wall_s"] = round(wall, 2)
        r["total_out_tokens"] = sum(len(o["token_ids"]) for o in outputs)

    print(f"\nwall={wall:.2f}s  total preemptions={sum(EVICT.values())}")
    for r in rows:
        print(f"  {r['group']:5s}: n={r['count']:3d} "
              f"evicted={r['requests_evicted']:3d} ({r['frac_evicted']:.0%}) "
              f"mean_evicts={r['mean_evictions']:.2f} "
              f"mean_lat={r['mean_latency_s']:6.2f}s "
              f"p50={r['p50_latency_s']:6.2f}s max={r['max_latency_s']:6.2f}s")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
