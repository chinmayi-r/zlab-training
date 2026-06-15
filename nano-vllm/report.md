# nano-vllm: what does it cost when the KV cache runs out mid-generation?

**Repo under study:** [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm/tree/main) — a ~1,450-line reimplementation of vLLM's three core ideas (paged KV cache, continuous batching, prefix caching).

**Why this repo exists:** vLLM is 100k+ lines and growing; you cannot trace a
single request through it in an afternoon. nano-vllm implements the same ideas in
a codebase you can read end to end, so every scheduling and memory decision is
legible. The gap it fills is pedagogical — it is vLLM's ideas made readable.

**The question this report traces to the bottom:**
> When nano-vllm runs out of KV-cache memory mid-generation, it *preempts* — it
> evicts a running request, throws away its KV cache, and re-prefills it later
> from scratch. How expensive is that, does prefix caching recover any of the
> cost, and is the eviction policy (always evict the most-recently-added request)
> the right one?

All line numbers below refer to the `main` branch as of this report.

**TL;DR in plain terms.** To generate text, the model keeps "notes" (the KV cache)
about everything read and written so far, so it never re-reads from the start.
Those notes live in limited GPU memory. When too many requests run at once, the
notes don't fit, and nano-vllm *preempts*: it picks one request, **deletes all its
notes, and sends it back to the start of the line to redo everything**. This report
asks how much that costs. Findings: (1) it kicks in sharply once the cache
saturates (here ~27 concurrent requests); (2) each eviction forces exactly one full
re-do of that request's work — ~36% more prefill passes and roughly half-again the
useful token-work at heavy load; (3) the "reuse old notes" feature (prefix caching)
recovers almost nothing (~3%, one block per victim), because the freed memory is
reused immediately under the very pressure that caused the eviction; and (4) the
surprise — overall tokens/sec does **not** crash, it just stops improving. The cost
hides in *latency*, not throughput. So watching throughput alone makes preemption
look free when it isn't.

---

## Part 1 — Background: the three ideas (so the question makes sense)

### Attention, briefly

Each token's embedding $x_i$ is projected to a query, key, and value:
$Q_i=W_Q x_i,\; K_i=W_K x_i,\; V_i=W_V x_i$. The output for token $i$ is
$\text{softmax}(Q_iK^T/\sqrt{d_k})\,V$ — token $i$ attends to every token $j$ via
the dot product $Q_i\cdot K_j$. The $\sqrt{d_k}$ scaling keeps softmax out of
saturation. Multi-head attention runs $h$ of these in parallel; causal masking
zeros the upper triangle so token $i$ can't see the future. Qwen3 uses
**grouped-query attention** — fewer K/V heads than Q heads — which directly shrinks
the KV cache (`num_key_value_heads` in `model_runner.py:110`).

### The three inference problems and nano-vllm's answers

| Problem | Solution | Where in nano-vllm |
|---|---|---|
| Recomputing K/V for past tokens every step is pure waste | **KV cache**: store K/V, compute only the new token's | `model_runner.allocate_kv_cache` (`model_runner.py:103-115`) |
| Variable-length requests fragment contiguous memory | **PagedAttention**: fixed-size blocks, a per-seq block table, non-contiguous physical pages, hash-addressed sharing | `engine/block_manager.py` |
| Static batching idles the GPU waiting on the longest seq | **Continuous batching**: every step re-schedules; finished seqs leave immediately, new ones slot in | `engine/scheduler.py` |

The KV cache is one pre-allocated slab sized *after* a warmup pass measures peak
memory (`model_runner.py:103-115`):

```python
block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype.itemsize   # :112
config.num_kvcache_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes  # :113
```

So `gpu_memory_utilization` is the dial that sets how many blocks exist — which is
exactly the dial the experiment turns down to force preemption.

### Codebase map (one line each)

| File | What it does |
|------|-------------|
| `config.py` | Dataclass of all engine config; loads HF config, asserts `kvcache_block_size % 256 == 0` |
| `sampling_params.py` | Per-request generation controls (temperature, max_tokens, ignore_eos) |
| `engine/sequence.py` | One request: token stream, block table, scheduling state, custom pickling for TP |
| `engine/block_manager.py` | PagedAttention: alloc/free fixed-size blocks; xxhash content-hash prefix cache |
| `engine/scheduler.py` | Continuous batching: prefill/decode dispatch, token budgets, **preemption** |
| `engine/model_runner.py` | Builds CUDA tensors, runs the model, CUDA-graph capture, sizes the KV cache, TP IPC |
| `engine/llm_engine.py` | Glue: tokenizer + scheduler + model_runner; drives the step loop |
| `layers/attention.py` | Triton scatter kernel writes K/V into the paged cache; FlashAttention varlen/kvcache |
| `layers/linear.py` | Column/Row/QKV/Merged tensor-parallel linears with sharding weight loaders |
| `layers/rotary_embedding.py` | Builds RoPE tables, applies them to Q and K |
| `layers/layernorm.py` | RMSNorm with fused residual add |
| `layers/activation.py` | SiluAndMul (SwiGLU): SiLU one half, multiply the other |
| `layers/embed_head.py` | Vocab-parallel embedding lookup and LM head |
| `layers/sampler.py` | Temperature-scaled sampling from logits |
| `models/qwen3.py` | Full Qwen3 decoder wired to the parallel layers (GQA + SwiGLU) |
| `utils/context.py` | Module-global context passing slot mapping / block tables / seqlens to attention |
| `utils/loader.py` | safetensors loader with packed-module remap (q/k/v_proj → qkv_proj) |
| `bench.py` | Throughput benchmark over 256 random-length sequences |

---

## Part 2 — The preemption path, annotated line by line

### Step 1: the trigger (`scheduler.py:58-73`)

The decode branch of `schedule()` pulls each running sequence and asks the block
manager whether it can grow by one token:

```python
while self.running and len(scheduled_seqs) < self.max_num_seqs:   # :58
    seq = self.running.popleft()                                  # :59
    while not self.block_manager.can_append(seq):                 # :60
        if self.running:
            self.preempt(self.running.pop())                      # :62  evict SOMEONE ELSE
        else:
            self.preempt(seq)                                     # :64  evict YOURSELF
            break
    else:
        seq.num_scheduled_tokens = 1                              # :67
        seq.is_prefill = False
        self.block_manager.may_append(seq)                        # :69
        scheduled_seqs.append(seq)
```

`can_append` is tiny (`block_manager.py:103-104`):

```python
def can_append(self, seq):
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

It only needs a *new* block when the sequence just crossed a block boundary
(`len(seq) % block_size == 1`, i.e. the next token starts a fresh block). So
preemption fires only at block boundaries — with `block_size = 256`, roughly once
every 256 generated tokens per sequence. **This is why the experiment uses long
prompts and long generations**: short requests never cross a boundary and never
trigger preemption.

### Step 2: the eviction policy is LIFO (`scheduler.py:62`)

The victim is `self.running.pop()` — the **right** end of the running deque, i.e.
the **most-recently-added** running sequence. Sequences enter `running` in the
order they finished prefill, so the victim is the *newest* admitted request.

Is LIFO right? The argument *for*: evicting the newest request preserves the
sequences that have generated the most tokens, so you protect the most invested
work and avoid re-prefilling long sequences. The argument *against*: it's
arbitrary with respect to remaining work — it can repeatedly evict the same
newly-admitted sequence (starvation), and it makes no attempt to evict the
sequence cheapest to recompute. A policy aware of *invested compute* (evict the
one with the fewest generated tokens) or *remaining work* would waste less. The
experiment's `unique_seqs_preempted` column reveals whether the same victims keep
getting hit (starvation) or eviction spreads out.

### Step 3: what `preempt` throws away (`scheduler.py:75-79`)

```python
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True                  # it will re-PREFILL, not resume decode
    self.block_manager.deallocate(seq)     # give back every block
    self.waiting.appendleft(seq)           # front of the queue -> re-admitted soon
```

It is re-marked `is_prefill = True`. There is **no CPU swap buffer** in nano-vllm —
the only way the evicted sequence comes back is by recomputing. When it is
re-admitted, `postprocess` (`scheduler.py:86-87`) and `prepare_prefill`
(`model_runner.py:129`) process **all** its tokens again: the original prompt plus
every token it had already generated. That recomputation is the headline cost.

### Step 4: deallocation preserves the hash (`block_manager.py:94-101`)

```python
def deallocate(self, seq):
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)   # :99
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

`_deallocate_block` (`block_manager.py:53-56`) only moves the id to the free list —
it does **not** clear `block.hash` or `block.token_ids`, and it leaves the entry in
`hash_to_block_id`. So immediately after eviction the victim's blocks are still
*content-addressable*. If the victim re-prefills before anyone reuses those
physical blocks, it can get them back as prefix-cache hits. **This is the one way
preemption cost could be partly recovered.**

### Step 5: but the hash dies on reuse (`block_manager.py:43-51`)

```python
def _allocate_block(self):
    block_id = self.free_block_ids.popleft()      # FIFO
    block = self.blocks[block_id]
    assert block.ref_count == 0
    if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
        del self.hash_to_block_id[block.hash]      # :48  hash destroyed here
    block.reset()
    ...
```

The hash survives *only until that physical block is popped from the free list for
someone else*. Under the memory pressure that caused the preemption, free blocks
are scarce and get re-popped almost immediately — destroying the victim's hashes
before it can re-prefill. So the recovery in Step 4 is real but fragile: **it works
when there's slack and fails exactly when there's pressure.** That is the crux of
the hypothesis.

### Step 6: where recovery would show up (`block_manager.py:58-92`)

When the victim re-prefills, `can_allocate` (`:58-73`) walks its completed blocks,
re-derives the chained hash, and checks `hash_to_block_id`:

```python
block_id = self.hash_to_block_id.get(h, -1)                       # :65
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    break                                                          # cache miss -> stop counting
num_cached_blocks += 1                                             # :68
```

`allocate` (`:75-92`) then re-refs the cached blocks and only allocates fresh ones
for the tail, setting `seq.num_cached_tokens = num_cached_blocks * block_size`
(`:92`). Those cached tokens are skipped in the re-prefill (`scheduler.py:39`). So
**`num_cached_blocks` returned to `allocate` is the exact recovery signal** — and
it is the value the experiment logs.

### Cost summary

| Cost | Mechanism | Line |
|---|---|---|
| Lost KV cache | `deallocate` frees all of the victim's blocks | `block_manager.py:94-101` |
| Wasted compute | re-prefill recomputes prompt + all generated-so-far tokens | `model_runner.py:129`, `scheduler.py:77` |
| Possible recovery | freed blocks keep their hash until re-popped | `block_manager.py:53-56`, `47-48` |
| Recovery defeated by pressure | scarce free blocks are re-popped immediately, clearing hashes | `block_manager.py:44,48` |
| Policy risk | LIFO can re-evict the same newest seq (starvation) | `scheduler.py:62` |

---

## Part 3 — Experiment

Full driver, instrumentation, and run instructions in
[`experiments/`](experiments/README.md). Summary:

- **Setup:** Qwen3-0.6B on a MIG A100 slice, `gpu_memory_utilization` forced low
  (0.30 or less) to make a small KV cache.
- **Instrumentation (monkeypatch, repo untouched):** wrap `Scheduler.preempt` to
  count/log evictions; wrap `BlockManager.allocate` to log `num_cached_blocks`.
- **Workload:** N concurrent requests with **unique** 600-token prompts (unique so
  no cross-request prefix sharing — any cache hit is genuine preemption recovery),
  600 generated tokens each, `ignore_eos=True`.
- **Sweep N** ∈ {4, 8, 16, 32, 64, 96} and record, per N: preemptions,
  unique sequences preempted, blocks recovered on re-prefill, wall-clock,
  throughput.

### Predictions (to be confirmed or falsified)

1. Preemptions ≈ 0 until N saturates the cache, then climb.
2. Throughput rises with N, then **cliffs** at the N where preemption begins —
   degradation is not graceful in the overload regime.
3. `cached_blocks_on_realloc` stays near zero under pressure → prefix caching does
   **not** rescue preempted requests (hypothesis (b)).

### Results

Qwen3-0.6B on an A100 MIG `3g.20gb` slice, `gpu_memory_utilization=0.30`
(→ 136 KV-cache blocks), 600-token unique prompts, 600 generated tokens,
`ignore_eos=True`. One engine reused across all N; prefix cache cleared between
sweep points so every recovery is genuine within-run preemption recovery.

| N | blocks | preemptions | uniq preempted | reprefill cached blocks | allocate_calls | wall (s) | tok/s |
|---|--------|-------------|----------------|-------------------------|----------------|----------|-------|
| 4 | 136 | 0 | 0 | 0 | 4 | 17.48 | 137 |
| 8 | 136 | 0 | 0 | 0 | 8 | 14.03 | 342 |
| 16 | 136 | 0 | 0 | 0 | 16 | 13.98 | 687 |
| 32 | 136 | 5 | 5 | 1 | 37 | 18.28 | 1050 |
| 64 | 136 | 26 | 23 | 3 | 90 | 28.84 | 1332 |
| 96 | 136 | 35 | 32 | 4 | 131 | 42.71 | 1349 |

**Each preemption = exactly one extra full re-prefill.** The data shows the clean
identity `allocate_calls = N + num_preemptions` at every point (32+5=37, 64+26=90,
96+35=131). So preemption's cost is precisely "redo the prefill," with no hidden
multiplier. `total_out_tokens` is also exactly N×600 throughout, meaning every
request still finished its full output — preemption *delayed* work, it never
*dropped* any. Peak GPU memory is flat at ~5.68 GB from N=32 on, confirming the KV
cache is a fixed pre-allocated slab (`model_runner.py:115`): memory doesn't grow
with load, the cache just runs out of free blocks to hand out.

**Knee (onset of preemption):** N = 32. The cache holds ~27 concurrent 1200-token
sequences (136 blocks ÷ 5 blocks/seq); preemption begins as soon as N exceeds that.

**Throughput:** rises steeply while the cache has room (137 → 342 → 687 → 1050)
then **plateaus** at ~1330–1350 tok/s for N=64 and N=96. There is **no cliff** —
see analysis below.

**Recovery:** total blocks recovered on re-prefill = 4 (at N=96), against ~120
blocks lost to eviction — about **3%**. `cache_hit_events` equals
`cached_blocks_on_realloc` at every N (1, 3, 4), which means each recovering
re-prefill got back **exactly one block**, never more — the chained hash breaks
after block 0 (see Part 4).

**Wasted compute (N=96):** 35 extra full re-prefills (`allocate_calls` 131 vs N=96,
i.e. **36% more prefill passes than the no-preemption baseline**). Each victim held
~770–1025 tokens (3–4 blocks), so ≈ 35 × ~900 ≈ **31,500 tokens re-prefilled from
scratch**, about **~55% of the 57,600 output tokens** generated. Substantial waste that the
throughput number does not reveal.

See `experiments/exp_plot.py` for the two plots (preemptions-vs-N, throughput-vs-N).

---

## Part 4 — What I learned & the policy I'd try next

**Prediction 1 (preemption onset) — confirmed.** Evictions are zero until the cache
saturates at N≈27, then climb monotonically (5 → 26 → 35). The LIFO policy is
directly visible in the logs: victims are evicted in descending seq_id order
(`seq 67, 66, 65, 64, 63…`), i.e. newest-first, exactly `self.running.pop()`
(`scheduler.py:62`).

**Prediction 2 (throughput cliff) — falsified, and this is the headline.**
Throughput does not collapse when preemption begins; it *plateaus*. The system
degrades **gracefully**, not off a cliff. The reason: throughput is measured in
*output* tokens/sec, and re-prefill is a single highly-parallel forward pass —
cheap per token compared to the memory-bound decode loop. So the recomputed work
(≈55% extra tokens at N=96) barely moves tok/s; it surfaces as **wall-clock /
latency** instead (wall triples from N=16 to N=96 while throughput is flat). The
lesson: aggregate throughput is the wrong lens for preemption cost — it hides the
waste. Per-request latency and total tokens-processed are the honest metrics.

**Prediction 3 (prefix caching can't rescue) — strongly confirmed.** Recovery was
~3% of lost blocks, and when a re-prefill recovered anything it recovered **exactly
one block** even though victims held 3–4. There is a precise reason, and it comes
from the interaction of two FIFO details:

- The free list is FIFO: freed blocks go to the **back** (`block_manager.py:56`),
  allocation takes from the **front** (`:44`).
- `deallocate` frees a victim's blocks in **reverse** order
  (`reversed(seq.block_table)`, `:95`). So the victim's *first* block `b0` lands at
  the very back of the free queue (reused **last**), while its later blocks
  `b1, b2, b3` sit nearer the front (reused **first**).

When the victim re-prefills, `can_allocate` walks its blocks 0,1,2… re-deriving the
*chained* hash and stops at the first miss (`:62-68`). Under load `b0` is usually
still cached (reused last) so block 0 matches — but `b1` has already been grabbed
by another request, which deleted its hash (`:47-48`), so block 1 misses and the
chain breaks. **Hence recovery is capped at exactly one block, and specifically
block 0**, no matter how much the victim had generated. It is even harsher in
aggregate: `cache_hit_events` was 4 of 35 preemptions at N=96, so **~89% of evicted
requests recovered nothing at all** — under heavy demand even `b0` is reused before
the victim returns. Prefix caching helps exactly when there is memory slack — and
fails exactly under the pressure that causes preemption.

It's worth being fair to the feature: prefix caching was never *designed* to rescue
preempted requests. Its job is to skip recomputation when two requests genuinely
share a prefix (a shared system prompt, or the same prompt re-asked) — and at that
job it works well. The fact that `deallocate` leaves hashes intact
(`block_manager.py:53-56`) makes *some* preemption recovery possible as a side
effect, which is why a naive reading of the code ("the hashes are kept, so a victim
should get everything back!") predicts much higher recovery than the ~1 block we
measured. The experiment's value is showing precisely why that naive expectation is
wrong under load.

**On the LIFO policy.** It is *defensible* but not *optimal*. Defensible: evicting
the newest sequence protects the older sequences that have generated the most
tokens, so it tends to preempt victims that have less invested (the logs show
victims with 769–1025 tokens, often the shorter ones). Not optimal: it is blind to
invested compute and re-evicts recently re-admitted sequences (N=96: 35 evictions,
32 unique → 3 sequences thrashed twice), wasting their re-prefill repeatedly.

**The policy I'd try next.** Evict by *least invested compute* — choose the running
sequence with the fewest tokens generated so far (cheapest to recompute) rather
than the newest. That directly minimizes wasted re-prefill per eviction and avoids
thrashing a sequence that just paid to come back. It's a small change: replace
`self.running.pop()` (`scheduler.py:62`) with an `argmin` over `running` by
`seq.num_completion_tokens`, backed by a heap if the linear scan matters. The
production-grade answer is a true CPU **swap** (vLLM's `swap_out`/`swap_in`): copy
the victim's blocks to host memory instead of dropping them, so it *resumes* decode
instead of re-prefilling — trading PCIe bandwidth for zero recompute. With more
time I'd implement least-invested-compute, re-run this exact sweep, and compare
total tokens-recomputed and wall-clock at N=64 and N=96.

---

## Part 5 — Follow-ups: admission, eviction policy, and fairness

Three extensions, all implemented by patching the live engine (the repo is still
never edited): conservative admission, a cheapest-to-redo eviction policy, and a
mixed short/long workload. Drivers and flags in
[`experiments/`](experiments/README.md) (`--admission`, `--policy`, `exp_varlen.py`).

> *Confound to keep in mind:* each configuration runs as a fresh process, and the
> KV cache sized to 136 blocks for the baseline run and 143 for later ones (startup
> free-memory variation). That's a ~5% shift in absolute block/preemption counts;
> trends are unaffected.

### 5A. Conservative admission — reserve worst-case blocks, never preempt

Gate admission on a **global reserved-blocks counter** (worst-case
`prompt + max_tokens` per live request) so the cache can't be oversubscribed.

| N | optimistic wall / tok/s | conservative wall / tok/s | conservative preemptions |
|---|--------------------------|----------------------------|--------------------------|
| 4 | 17.52 / 137 | 15.25 / 157 | 0 |
| 8 | 13.96 / 344 | 13.90 / 345 | 0 |
| 16 | 13.93 / 689 | 13.84 / 694 | 0 |
| 32 | 18.26 / 1052 | **29.82 / 644** | 0 |
| 64 | 28.79 / 1334 | **41.86 / 917** | 0 |
| 96 | 42.59 / 1353 | **55.77 / 1033** | 0 |

**Result: it does exactly what it promises — zero preemptions at every N — and is
strictly *slower* for it.** Below saturation (N≤16) the two are identical; once the
cache saturates conservative is 1.6× slower at N=32, narrowing to 1.3× at N=96. The
predicted crossover where conservative *beats* optimistic never occurs within
N≤96. Why: conservative caps concurrency at ~28 sequences (143 blocks ÷ 5) and
serializes the overflow, paying in idle parallelism and a low-batch tail;
optimistic admits everyone and absorbs the overflow through *cheap, graceful*
preemption (Part 4). **Because preemption is cheap, paying to avoid it costs more
than it saves.** The gap shrinks with N (1.6×→1.3×), suggesting a crossover only at
much heavier overload — a sweep to N=256+ would test that.

### 5B. Eviction policy — LIFO vs cheapest-to-redo

Swap `scheduler.running` for a deque whose `.pop()` evicts the fewest-token request
(cheapest to re-prefill) instead of the newest.

| N | LIFO preempt (uniq) | cheapest preempt (uniq) | LIFO wall / tok/s | cheapest wall / tok/s |
|---|---------------------|--------------------------|-------------------|-----------------------|
| 32 | 5 (5) | 4 (4) | 18.26 / 1052 | 18.20 / 1055 |
| 64 | 26 (23) | 22 (21) | 28.79 / 1334 | 28.65 / 1340 |
| 96 | 35 (32) | 38 (34) | 42.59 / 1353 | 42.43 / 1358 |

**Result: outcome (i) — the policies are indistinguishable** in wall-clock and
throughput. This confirms the prediction that for a *uniform* workload "newest" ≈
"fewest tokens": the most-recently-admitted request has generated the least, so
LIFO already approximates cheapest-to-redo. Neither thrashed (preempt-vs-uniq gaps
are small for both). Cheapest recovered marginally more prefix blocks (3/6/9 vs
1/3/4) but the magnitude is trivial. The takeaway: **LIFO is a reasonable default
for uniform traffic — the policy only matters when newest ≠ cheapest**, i.e. under
non-uniform workloads, which motivates 5C.

### 5C. Fairness under mixed short/long requests (`exp_varlen.py`)

64 requests, 50/50 split of `max_tokens=100` (short) and `max_tokens=1000` (long),
200-token prompts, `gpu_memory_utilization=0.30`.

| policy | group | frac evicted | mean latency (s) | max latency (s) |
|--------|-------|--------------|------------------|-----------------|
| LIFO | short | 0% | 3.64 | 3.64 |
| LIFO | long | 12% | 25.36 | 29.02 |
| cheapest | short | 0% | 3.58 | 3.58 |
| cheapest | long | 12% | 25.04 | 28.68 |

**Results:**
1. **The eviction burden falls entirely on long requests** — 0% of short requests
   were ever evicted, 12% of long were. Short requests finish in ~3.6 s and leave
   the running set before the cache fills, so they're never eviction candidates;
   long requests are resident during the squeeze and hold the most blocks, so they
   are both the trigger and the victim. The policy is **structurally unfair to long
   requests** — confirmed.
2. **But most of the latency gap is inherent, not eviction-induced.** Long requests
   take ~25 s vs ~3.6 s mostly because they generate 10× more tokens. Eviction adds
   only a modest tail: evicted long requests reach ~29 s vs a ~24.8 s median — a
   ~4–5 s penalty on the unlucky 12%. Eviction worsens long-request tail latency
   but is not the main driver of the short/long gap.
3. **Changing the policy doesn't fix fairness.** LIFO and cheapest give
   near-identical eviction rates and latencies, because in this mix the long
   requests are the only viable eviction targets regardless of victim heuristic.
   Fairness needs a fairness-aware mechanism (priority/aging, or CPU swap so long
   requests *pause* instead of restart), not a smarter victim choice.

This config produced only mild pressure (4 evictions); a harsher setting (lower
util or a higher long-request fraction) would amplify the eviction tail and is the
natural next step.

### What the follow-ups add up to

- **Avoiding preemption is not worth it here:** conservative reservation eliminates
  preemption but is strictly slower up to N=96, because preemption is cheap.
- **The eviction policy barely matters** for uniform or mildly-mixed workloads;
  LIFO is a fine default.
- **The real lever is fairness:** long requests bear all the eviction cost — a
  structural property of length-mixed traffic, fixable only by fairness-aware
  scheduling or true swap, not by tweaking which victim you pick.

---

## Connection to research

KV-cache layout matters for interpretability probes on intermediate activations.
Note that nano-vllm's cache stores only K and V projections — hidden states are
ephemeral (one forward pass). To extract intermediate activations you'd add
forward hooks on the attention/MLP modules in `models/qwen3.py`, writing to a side
buffer without touching the inference path.
