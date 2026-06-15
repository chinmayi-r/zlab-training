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


def run_one(model, n, gpu_mem_util, prompt_tokens, max_tokens,
            max_num_seqs, max_model_len, verbose):
    STATS.reset()
    STATS.verbose = verbose

    llm = LLM(
        model,
        enforce_eager=True,            # avoid CUDA-graph capture noise during a stress test
        gpu_memory_utilization=gpu_mem_util,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
    )
    num_blocks = len(llm.scheduler.block_manager.blocks)

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

    print(f"N={n:4d} blocks={num_blocks:4d} "
          f"preempt={row['num_preemptions']:4d} "
          f"(uniq={row['unique_seqs_preempted']:3d}) "
          f"reprefill_cached_blocks={row['cached_blocks_on_realloc']:4d} "
          f"wall={row['wall_s']:7.2f}s "
          f"tok/s={row['throughput_tok_s']:7.1f}")

    del llm
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
    p.add_argument("--out", default="results.csv")
    p.add_argument("--verbose", action="store_true",
                   help="print every PREEMPT and CACHE-HIT event live")
    args = p.parse_args()

    import os
    model = os.path.expanduser(args.model)

    print(f"# model={model}")
    print(f"# gpu_mem_util={args.gpu_mem_util} prompt_tokens={args.prompt_tokens} "
          f"max_tokens={args.max_tokens}")
    print(f"# sweeping N over {args.n}\n")

    rows = []
    for n in args.n:
        rows.append(run_one(
            model, n, args.gpu_mem_util, args.prompt_tokens, args.max_tokens,
            args.max_num_seqs, args.max_model_len, args.verbose,
        ))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
