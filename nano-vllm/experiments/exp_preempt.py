"""
exp_preempt.py — measure the cost of KV-cache preemption in nano-vllm.

ONE question, three measurements:
  (1) How many preemptions occur as a function of N (number of concurrent requests)?
  (2) How does wall-clock time scale with N — graceful, or a cliff?
  (3) When a preempted request re-prefills, does prefix caching recover any of
      its lost KV cache? (i.e. does block_manager.allocate report num_cached_blocks > 0)

We never edit the nano-vllm source. Instead we monkeypatch two methods at import
time so every preemption and every (re-)allocation is logged:
  - Scheduler.preempt        -> counts evictions, records which seq_id was evicted
  - BlockManager.allocate    -> records num_cached_blocks on each allocation

Because each prompt is given a UNIQUE prefix ("Request <i>:"), two different
requests never share a prefix. So any num_cached_blocks > 0 observed during the
run can ONLY come from a preempted request re-prefilling and finding ITS OWN old
blocks still alive in the hash table. That is exactly the recovery effect we want
to measure (hypothesis (b)).

Usage (on a GPU node):
  python exp_preempt.py --model ~/huggingface/Qwen3-0.6B \\
      --n 4 8 16 32 64 --gpu-mem-util 0.30 \\
      --prompt-tokens 600 --max-tokens 600 --out results.csv

Tips to actually trigger preemption:
  - Lower --gpu-mem-util (0.20-0.35) so the KV cache holds few blocks.
  - Raise --n so many sequences decode at once.
  - Keep --prompt-tokens and --max-tokens > block_size (256) so each sequence
    crosses block boundaries during decode and must allocate new blocks.
  If you see "preemptions=0" for every N, the cache is too big: lower --gpu-mem-util.
"""

import argparse
import csv
import time
from collections import deque

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.block_manager import BlockManager


# ----------------------------------------------------------------------------
# Instrumentation: monkeypatch, do not edit the repo.
# ----------------------------------------------------------------------------
class Stats:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.reset()

    def reset(self):
        self.num_preemptions = 0
        self.preempted_seq_ids = []         # one entry per preemption event
        self.allocate_calls = 0
        self.cached_blocks_total = 0        # sum of num_cached_blocks across allocate()
        self.cached_hits = []               # (seq_id, num_cached_blocks) where > 0


STATS = Stats()

_orig_preempt = Scheduler.preempt
_orig_allocate = BlockManager.allocate


def _patched_preempt(self, seq):
    STATS.num_preemptions += 1
    STATS.preempted_seq_ids.append(seq.seq_id)
    if STATS.verbose:
        print(f"  PREEMPT seq {seq.seq_id}  "
              f"(blocks held={len(seq.block_table)}, tokens={seq.num_tokens})")
    return _orig_preempt(self, seq)


def _patched_allocate(self, seq, num_cached_blocks):
    STATS.allocate_calls += 1
    STATS.cached_blocks_total += num_cached_blocks
    if num_cached_blocks > 0:
        STATS.cached_hits.append((seq.seq_id, num_cached_blocks))
        if STATS.verbose:
            print(f"  CACHE-HIT seq {seq.seq_id}  num_cached_blocks={num_cached_blocks} "
                  f"(re-prefill recovered {num_cached_blocks} block(s))")
    return _orig_allocate(self, seq, num_cached_blocks)


Scheduler.preempt = _patched_preempt
BlockManager.allocate = _patched_allocate


# ----------------------------------------------------------------------------
# Workload
# ----------------------------------------------------------------------------
def build_unique_prompts(llm, n, prompt_tokens):
    """n prompts, each with a unique prefix so no cross-request prefix sharing.
    Each is trimmed to exactly `prompt_tokens` token ids."""
    tok = llm.tokenizer
    filler = "the quick brown fox jumps over the lazy dog and then keeps running "
    prompts = []
    for i in range(n):
        text = f"Request {i}: " + filler * (prompt_tokens // 8 + 4)
        ids = tok.encode(text)[:prompt_tokens]
        # pad up if the unique prefix made it short
        while len(ids) < prompt_tokens:
            ids += tok.encode(filler)
        prompts.append(ids[:prompt_tokens])
    return prompts


def build_llm(model, gpu_mem_util, max_num_seqs, max_model_len):
    return LLM(
        model,
        enforce_eager=True,            # avoid CUDA-graph capture noise during a stress test
        gpu_memory_utilization=gpu_mem_util,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
    )


# ----------------------------------------------------------------------------
# Follow-up knobs (still no edits to the repo — we patch the live instances)
# ----------------------------------------------------------------------------
class CheapestDeque(deque):
    """A running-queue whose .pop() evicts the CHEAPEST-to-redo sequence (fewest
    total tokens) instead of the newest. In scheduler.py, .pop() is used in exactly
    one place — the eviction at line 62 `self.preempt(self.running.pop())` — so
    swapping the deque changes the eviction policy without touching schedule().
    Every other operation (popleft/append/extendleft/remove) is inherited."""
    def pop(self):
        victim = min(self, key=lambda s: s.num_tokens)
        self.remove(victim)
        return victim


def install_policy(llm, policy):
    """policy='lifo' (default nano-vllm) or 'cheapest' (fewest-token victim)."""
    if policy == "cheapest":
        llm.scheduler.running = CheapestDeque(llm.scheduler.running)
    return policy


def install_admission(llm, mode):
    """mode='optimistic' (default) or 'conservative'.

    Conservative admission reserves WORST-CASE blocks (prompt + max_tokens) for
    every live request, so the sum of reservations never exceeds the cache. Because
    current allocation <= reservation for every seq, a running seq that crosses a
    block boundary always finds a free block -> preemption can never happen. The
    per-seq check alone is not enough: already-running seqs keep growing into the
    free pool, so the reservation must be a GLOBAL running counter."""
    if mode != "conservative":
        return mode
    bm = llm.scheduler.block_manager
    total = len(bm.blocks)
    bm._reserved = 0
    bsz = bm.block_size
    orig_can_allocate = bm.can_allocate
    orig_allocate = bm.allocate
    orig_deallocate = bm.deallocate

    def worst_case(seq):
        return (seq.num_prompt_tokens + seq.max_tokens + bsz - 1) // bsz

    def can_allocate(seq):
        if bm._reserved + worst_case(seq) > total:
            return -1                      # not enough worst-case room -> queue it
        return orig_can_allocate(seq)

    def allocate(seq, num_cached_blocks):
        seq._wc = worst_case(seq)
        bm._reserved += seq._wc
        return orig_allocate(seq, num_cached_blocks)

    def deallocate(seq):
        bm._reserved -= getattr(seq, "_wc", 0)
        seq._wc = 0
        return orig_deallocate(seq)

    bm.can_allocate = can_allocate
    bm.allocate = allocate
    bm.deallocate = deallocate
    return mode


def reset_prefix_cache(llm):
    """Clear cross-run prefix-cache state so each N is measured in isolation.

    nano-vllm initializes a global process group in ModelRunner.__init__ and never
    tears it down, so we build ONE LLM and reuse it for every N. But after a
    generate() completes, all blocks are freed yet their content hashes linger in
    hash_to_block_id (deallocate never clears them). Without this reset, the
    repeated "Request 0..k" prompts across different N values would produce
    prefix-cache hits unrelated to preemption and pollute the recovery measurement.
    """
    bm = llm.scheduler.block_manager
    bm.hash_to_block_id.clear()
    for blk in bm.blocks:
        blk.hash = -1
        blk.token_ids = []


def measure_n(llm, n, gpu_mem_util, prompt_tokens, max_tokens, verbose,
              policy="lifo", admission="optimistic"):
    STATS.reset()
    STATS.verbose = verbose
    reset_prefix_cache(llm)

    bm = llm.scheduler.block_manager
    if hasattr(bm, "_reserved"):
        bm._reserved = 0                   # clear conservative-admission accounting
    num_blocks = len(bm.blocks)

    prompts = build_unique_prompts(llm, n, prompt_tokens)
    # nano-vllm forbids temperature <= 1e-10 (greedy sampling). The exact value is
    # irrelevant to this stress test — ignore_eos=True forces every sequence to
    # generate the full max_tokens regardless of which token is sampled.
    sp = SamplingParams(temperature=0.6, max_tokens=max_tokens, ignore_eos=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, [sp] * n, use_tqdm=False)
    wall = time.perf_counter() - t0

    total_tokens = sum(len(o["token_ids"]) for o in outputs)
    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    row = {
        "N": n,
        "policy": policy,
        "admission": admission,
        "num_kvcache_blocks": num_blocks,
        "gpu_mem_util": gpu_mem_util,
        "prompt_tokens": prompt_tokens,
        "max_tokens": max_tokens,
        "num_preemptions": STATS.num_preemptions,
        "unique_seqs_preempted": len(set(STATS.preempted_seq_ids)),
        "allocate_calls": STATS.allocate_calls,
        "cached_blocks_on_realloc": STATS.cached_blocks_total,
        "cache_hit_events": len(STATS.cached_hits),
        "wall_s": round(wall, 3),
        "total_out_tokens": total_tokens,
        "throughput_tok_s": round(total_tokens / wall, 1) if wall > 0 else 0,
        "peak_mem_mb": round(peak_mb, 1),
    }

    print(f"[{policy}/{admission}] N={n:4d} blocks={num_blocks:4d} "
          f"preempt={row['num_preemptions']:4d} "
          f"(uniq={row['unique_seqs_preempted']:3d}) "
          f"reprefill_cached_blocks={row['cached_blocks_on_realloc']:4d} "
          f"wall={row['wall_s']:7.2f}s "
          f"tok/s={row['throughput_tok_s']:7.1f}")

    torch.cuda.empty_cache()
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to model dir (e.g. ~/huggingface/Qwen3-0.6B)")
    p.add_argument("--n", type=int, nargs="+", default=[4, 8, 16, 32, 64],
                   help="list of concurrent-request counts to sweep")
    p.add_argument("--gpu-mem-util", type=float, default=0.30,
                   help="lower = smaller KV cache = more preemption pressure")
    p.add_argument("--prompt-tokens", type=int, default=600)
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--policy", choices=["lifo", "cheapest"], default="lifo",
                   help="eviction victim: lifo=newest (nano-vllm default), "
                        "cheapest=fewest-token (cheapest to re-prefill)")
    p.add_argument("--admission", choices=["optimistic", "conservative"],
                   default="optimistic",
                   help="optimistic=admit on current block need (default); "
                        "conservative=reserve worst-case prompt+max_tokens blocks "
                        "(provably zero preemption)")
    p.add_argument("--out", default="results.csv")
    p.add_argument("--verbose", action="store_true",
                   help="print every PREEMPT and CACHE-HIT event live")
    args = p.parse_args()

    import os
    model = os.path.expanduser(args.model)

    print(f"# model={model}")
    print(f"# gpu_mem_util={args.gpu_mem_util} prompt_tokens={args.prompt_tokens} "
          f"max_tokens={args.max_tokens}")
    print(f"# policy={args.policy} admission={args.admission}")
    print(f"# sweeping N over {args.n}\n")

    # Build the engine ONCE (nano-vllm can't re-init its process group) and reuse
    # it for every N. max_num_seqs must admit the largest N so they all decode
    # concurrently and can actually exhaust the cache.
    max_num_seqs = max(args.max_num_seqs, max(args.n))
    llm = build_llm(model, args.gpu_mem_util, max_num_seqs, args.max_model_len)
    install_policy(llm, args.policy)
    install_admission(llm, args.admission)

    rows = []
    for n in args.n:
        rows.append(measure_n(
            llm, n, args.gpu_mem_util, args.prompt_tokens, args.max_tokens,
            args.verbose, policy=args.policy, admission=args.admission,
        ))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
