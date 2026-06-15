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

| N | blocks | preemptions | uniq preempted | reprefill cached blocks | wall (s) | tok/s |
|---|--------|-------------|----------------|-------------------------|----------|-------|
| 4 | 136 | 0 | 0 | 0 | 17.48 | 137 |
| 8 | 136 | 0 | 0 | 0 | 14.03 | 342 |
| 16 | 136 | 0 | 0 | 0 | 13.98 | 687 |
| 32 | 136 | 5 | 5 | 1 | 18.28 | 1050 |
| 64 | 136 | 26 | 23 | 3 | 28.84 | 1332 |
| 96 | 136 | 35 | 32 | 4 | 42.71 | 1349 |

**Knee (onset of preemption):** N = 32. The cache holds ~27 concurrent 1200-token
sequences (136 blocks ÷ 5 blocks/seq); preemption begins as soon as N exceeds that.

**Throughput:** rises steeply while the cache has room (137 → 342 → 687 → 1050)
then **plateaus** at ~1330–1350 tok/s for N=64 and N=96. There is **no cliff** —
see analysis below.

**Recovery:** total blocks recovered on re-prefill = 4 (at N=96), against ~120
blocks lost to eviction — about **3%**. Every single cache hit recovered **exactly
1 block**, never more.

**Wasted compute (N=96):** 35 preemptions, each victim holding ~770–1025 tokens
(3–4 blocks) → roughly 35 × ~900 ≈ **31,500 tokens re-prefilled from scratch**,
about **55% of the 57,600 output tokens** generated. Substantial waste that the
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
~3% of lost blocks, and crucially **every cache hit recovered exactly one block**
even though victims held 3–4. This is a clean confirmation of the Step 5 mechanism:
when a victim re-prefills, `can_allocate` walks its blocks re-deriving the chained
hash (`block_manager.py:62-68`); block 0's hash sometimes survives, but blocks 1+
have already had their physical pages re-popped from the FIFO free list and their
hashes deleted (`block_manager.py:44,47-48`). The chain breaks after the first
block, so recovery is capped at one block regardless of how much the victim had
generated. Prefix caching helps exactly when there is memory slack — and fails
exactly under the pressure that causes preemption.

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

## Connection to research

KV-cache layout matters for interpretability probes on intermediate activations.
Note that nano-vllm's cache stores only K and V projections — hidden states are
ephemeral (one forward pass). To extract intermediate activations you'd add
forward hooks on the attention/MLP modules in `models/qwen3.py`, writing to a side
buffer without touching the inference path.
