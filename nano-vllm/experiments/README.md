# Experiment: the cost of KV-cache preemption in nano-vllm

## The one question

**What happens when nano-vllm runs out of KV-cache memory mid-generation, and what does it cost?**

When the decode loop needs a fresh block for a running sequence but none are free,
the scheduler **preempts**: it evicts a running sequence, deallocates all its KV
blocks, and pushes it back to the waiting queue to be re-prefilled from scratch
later (`scheduler.py:58-79`). This experiment measures how expensive that is.

## What we measure (three numbers, one question)

1. **Preemptions vs N** — how many evictions happen as the number of concurrent
   requests N grows.
2. **Wall-clock vs N** — does latency degrade gracefully or fall off a cliff once
   preemption kicks in.
3. **Prefix-cache recovery** — when a preempted request re-prefills, does it get
   any blocks back from the prefix cache (`block_manager.allocate`'s
   `num_cached_blocks`), or is the re-prefill pure wasted compute.

## The hypothesis (state it before you run)

Preemption under nano-vllm's LIFO policy (`scheduler.py:62`, `self.running.pop()`)
is expensive because:
- **(a)** re-prefill recomputes every token the evicted sequence had already
  processed (prompt + generated-so-far) — see `model_runner.prepare_prefill`
  (`model_runner.py:129`), and
- **(b)** prefix caching is unlikely to recover the cost, because the same memory
  pressure that caused the eviction means those physical blocks get re-allocated
  to other sequences, which clears their hash (`block_manager.py:47-48`).

Prediction: throughput degrades sharply past the N where preemptions begin, and
`cached_blocks_on_realloc` stays near zero. The prediction can be wrong — maybe
prefix caching helps more than expected — which is what makes it worth running.

## How it's instrumented

`exp_preempt.py` **monkeypatches** two methods at import time, so you never edit
the cloned nano-vllm repo:
- `Scheduler.preempt` → counts evictions and logs the evicted `seq_id`.
- `BlockManager.allocate` → logs `num_cached_blocks` on every allocation.

Each prompt gets a **unique prefix** (`"Request <i>:"`), so two different requests
never share a prefix. Therefore any `num_cached_blocks > 0` during the run can
only mean a preempted request re-prefilled and found *its own* old blocks still
alive — exactly the recovery effect in hypothesis (b).

## How to run (Adroit)

See `../setup.md` first (env + flash-attn). Then, on a **GPU compute node**:

```bash
# interactive
salloc --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --gres=gpu:1 --time=01:00:00 --partition=mig
conda activate nanovllm
cd ~/zlab-training/nano-vllm/experiments   # or wherever you put these scripts

python exp_preempt.py --model ~/huggingface/Qwen3-0.6B \
    --n 4 8 16 32 64 96 --gpu-mem-util 0.30 \
    --prompt-tokens 600 --max-tokens 600 --out results.csv --verbose
```

or submit the batch job:

```bash
sbatch run_preempt.slurm
tail -f preempt_*.out
```

### Tuning so you actually SEE preemption

Preemption only happens in the **decode** phase when free blocks hit zero. If you
get `preempt=0` for every N, the cache is too big — **lower `--gpu-mem-util`**
(try 0.25, then 0.20). Keep `--prompt-tokens` and `--max-tokens` above the block
size (256) so each sequence crosses block boundaries during decode and is forced
to allocate. Raise `--n` to add pressure.

## What you get out

`results.csv` with one row per N:

| column | meaning |
|--------|---------|
| `N` | concurrent requests |
| `num_kvcache_blocks` | physical blocks the cache was sized to (depends on `--gpu-mem-util`) |
| `num_preemptions` | total eviction events |
| `unique_seqs_preempted` | how many distinct sequences got evicted (a seq can be evicted repeatedly) |
| `cached_blocks_on_realloc` | total blocks recovered via prefix cache during (re-)prefill — **the key number for hypothesis (b)** |
| `wall_s` | wall-clock for the batch |
| `throughput_tok_s` | output tokens / wall |
| `peak_mem_mb` | peak GPU memory |

## Turning the CSV into the report

1. **Plot 1: preemptions vs N.** Expect ~0 until the cache saturates, then it
   climbs. The knee is where N exceeds what the cache can hold.
2. **Plot 2: throughput (tok/s) vs N.** Expect it to rise, then fall once
   preemption starts thrashing. Mark the knee from Plot 1 on this curve — does the
   throughput cliff line up with the onset of preemption?
3. **Read `cached_blocks_on_realloc`.** Near zero → hypothesis (b) holds, prefix
   caching does not rescue preempted requests under pressure. Large → hypothesis
   (b) is wrong; explain why (probably blocks survived because the victim's blocks
   weren't immediately re-popped from the FIFO free list).
4. **Quantify wasted compute.** A preempted sequence that had generated `g` tokens
   on top of a `p`-token prompt re-prefills `p + g` tokens. Multiply by
   `num_preemptions` (weighted by where each eviction happened) for an upper bound
   on tokens recomputed. Compare to `total_out_tokens` to express waste as a %.

To plot without a display, write a CSV and plot locally, or add matplotlib with
`plt.savefig("preempt.png")` and `scp` it back.
