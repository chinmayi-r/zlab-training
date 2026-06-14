# nano-vllm: A Complete Code Walkthrough

**Repo under study:** [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm/tree/main)  
**Total size:** ~1,200 lines of Python  
**Purpose:** Clean implementation of the three core vLLM ideas — KV cache, PagedAttention, and continuous batching — small enough to read end to end.

---

## Week 1: Theory

### Day 1–2: Attention from Scratch

#### Why Attention Exists

An RNN processes tokens one at a time. To get information from token 1 to token 50, you pass through 49 hidden state updates, each of which can corrupt or drop information. Attention solves this by letting every token query every other token directly in O(1) hops.

#### Q/K/V

Each token's embedding $x_i \in \mathbb{R}^{d_\text{model}}$ is projected three ways via learned matrices:

$$Q_i = W_Q x_i, \quad K_i = W_K x_i, \quad V_i = W_V x_i$$

- **Query**: what this token is looking for
- **Key**: what this token contains / advertises
- **Value**: what this token contributes if selected

The attention score between token $i$ and token $j$ is $Q_i \cdot K_j$. High dot product = token $i$ should "attend to" token $j$. The full output for token $i$:

$$\text{output}_i = \text{softmax}\!\left(\frac{Q_i K^T}{\sqrt{d_k}}\right) V$$

The $\sqrt{d_k}$ scaling prevents large dot products from collapsing softmax into a one-hot distribution (which would zero out gradients and stop learning).

#### Multi-Head Attention

Instead of one set of Q/K/V projections, run $h$ sets in parallel, each in dimension $d_k = d_\text{model}/h$. Each head can specialize — one for local syntax, one for long-range semantics, etc. Outputs are concatenated and projected back:

$$\text{MHA}(x) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

Qwen3 uses **grouped-query attention (GQA)**: fewer K/V heads than Q heads. Multiple query heads share one K/V pair. This cuts KV cache size by a factor of `num_q_heads / num_kv_heads` with minimal quality loss.

#### Causal Masking

For autoregressive generation, token $i$ must not see tokens $i+1, i+2, \ldots$ — that would be cheating. We mask the upper triangle of the $N \times N$ attention matrix with $-\infty$ before softmax, so those positions get zero weight. FlashAttention handles this with `causal=True`.

---

### Day 3–4: The Inference Problem

Three problems arise when you scale from single-request inference to a serving system:

#### Problem 1: Redundant Computation → KV Cache

During generation, when you produce token $t+1$, you need $K_j$ and $V_j$ for all previous tokens $j = 1 \ldots t$. But those haven't changed. Recomputing them is pure waste.

**Solution:** store K and V as you compute them. On each decode step, only compute K/V for the one new token.

In nano-vllm, the cache lives in `model_runner.py`:
```python
self.kv_cache = torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
```
Shape breakdown: `2` = K and V; `num_blocks` = physical pages; `block_size` = tokens per page.

#### Problem 2: Memory Fragmentation → PagedAttention

With multiple requests, each with a different number of cached tokens, contiguous allocation wastes memory (fragmentation) and prevents sharing common prefixes. The vLLM paper borrows the OS virtual memory idea:

- Divide KV cache into fixed-size **blocks** (pages), here `block_size = 256` tokens
- Each sequence has a **block table** mapping logical block index → physical block id
- Blocks can be allocated non-contiguously
- Blocks holding the same token sequence (same hash) can be **shared** across requests (prefix caching)

#### Problem 3: GPU Underutilization → Continuous Batching

In static batching, the GPU waits for the longest sequence to finish before starting new ones. With continuous batching, each step of the generation loop can add new requests as soon as a slot opens. nano-vllm's scheduler does exactly this: `schedule()` returns a mix of prefill and decode sequences every step.

---

## Week 2: Code Reading

### Day 5: Codebase Map

```
nano-vllm/
├── bench.py                    # benchmark: 256 random seqs, measure tok/s
├── example.py                  # minimal usage example
├── pyproject.toml
└── nanovllm/
    ├── __init__.py             # exports LLM, SamplingParams
    ├── config.py               # all engine config as a dataclass
    ├── sampling_params.py      # per-request generation params
    ├── llm.py                  # user-facing LLM wrapper
    ├── engine/
    │   ├── sequence.py         # one request: token ids, block table, state machine
    │   ├── block_manager.py    # PagedAttention: block alloc/free + prefix hash cache
    │   ├── scheduler.py        # continuous batching: prefill/decode dispatch + preemption
    │   ├── llm_engine.py       # glue: tokenizer + scheduler + model_runner
    │   └── model_runner.py     # CUDA tensor prep, model forward, CUDA graphs, TP comm
    ├── layers/
    │   ├── attention.py        # Triton KV scatter kernel + FlashAttention calls
    │   ├── linear.py           # tensor-parallel linear layers with weight sharding
    │   ├── rotary_embedding.py # RoPE positional embeddings for Q and K
    │   ├── layernorm.py        # RMSNorm with fused residual add
    │   ├── activation.py       # SiluAndMul for SwiGLU MLP
    │   ├── embed_head.py       # vocab-parallel embedding + LM head
    │   └── sampler.py          # temperature sampling from logits
    ├── models/
    │   └── qwen3.py            # full Qwen3 decoder (attention + MLP + norms)
    └── utils/
        ├── context.py          # global context object: prefill/decode metadata for attn
        └── loader.py           # safetensors loader with packed-module weight remapping
```

**File-by-file one-liners:**

| File | What it does |
|------|-------------|
| `config.py` | Dataclass holding all engine config; validates and loads HuggingFace config on init |
| `sampling_params.py` | Per-request generation controls (temperature, max_tokens, ignore_eos) |
| `engine/sequence.py` | Represents one request: token stream, block table, scheduling state; custom pickling for multiprocessing |
| `engine/block_manager.py` | PagedAttention: allocates/frees fixed-size KV blocks, uses xxhash content hashing for prefix-cache reuse |
| `engine/scheduler.py` | Continuous batching: each step picks prefill or decode sequences, enforces token budgets, preempts when KV cache is full |
| `engine/model_runner.py` | Prepares CUDA tensors, runs the model, handles CUDA graph capture for decode, broadcasts to tensor-parallel ranks via shared memory |
| `engine/llm_engine.py` | Glues tokenizer + scheduler + model_runner; drives the step loop and reports throughput |
| `layers/attention.py` | Writes K/V into paged cache via Triton scatter kernel; calls FlashAttention (varlen prefill / kvcache decode) |
| `layers/linear.py` | Column/Row/QKV/Merged parallel linears with weight loaders that shard across TP ranks on load |
| `layers/rotary_embedding.py` | Builds RoPE frequency tables and applies them to Q and K |
| `layers/layernorm.py` | RMSNorm with fused residual-add (saves one read/write of the residual stream) |
| `layers/activation.py` | SiluAndMul: split gate-up output, apply SiLU to one half, multiply |
| `layers/embed_head.py` | Vocab-parallel embedding lookup and LM head projection |
| `layers/sampler.py` | Temperature-scaled sampling: divide logits, softmax, multinomial |
| `models/qwen3.py` | Full Qwen3 transformer wired to all parallel layers (GQA + SwiGLU MLP) |
| `utils/context.py` | Module-global context set before each forward pass; passes slot mapping, block tables, seqlens to attention |
| `utils/loader.py` | Loads safetensors weights with packed-module remapping (q_proj/k_proj/v_proj → qkv_proj) |
| `bench.py` | Benchmark: 256 random-length sequences, prints total throughput in tok/s |

---

### Component 1: KV Cache Allocation (`model_runner.py`)

**Where:** `ModelRunner.allocate_kv_cache()`, lines ~60–80

```python
self.kv_cache = torch.empty(
    2,                          # K and V
    hf_config.num_hidden_layers,
    config.num_kvcache_blocks,  # physical pages
    self.block_size,            # tokens per page (256)
    num_kv_heads,               # per TP rank
    head_dim
)
```

The number of blocks is calculated dynamically: after a warmup forward pass measures peak GPU memory, the remaining free memory (at `gpu_memory_utilization=0.9` of total) is divided by `block_bytes` — the bytes for one block across all layers:

```python
block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype.itemsize
config.num_kvcache_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes
```

After allocation, each attention layer gets pointers into this single tensor:
```python
for module in self.model.modules():
    if hasattr(module, "k_cache"):
        module.k_cache = self.kv_cache[0, layer_id]
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id += 1
```

This is one pre-allocated slab — no dynamic allocation during inference.

---

### Component 2: PagedAttention / Block Manager (`engine/block_manager.py`)

**Block size:** `256` tokens (set as `Sequence.block_size` in `config.py`, must be a multiple of 256 per the assertion).

**Data structures:**
- `free_block_ids: deque[int]` — available physical block ids (FIFO)
- `used_block_ids: set[int]` — currently allocated block ids
- `hash_to_block_id: dict[int, int]` — content hash → physical block (prefix cache)
- Each `Block` has: `block_id`, `ref_count`, `hash`, `token_ids`

**Allocation flow (`can_allocate` + `allocate`):**

When a new sequence arrives, `can_allocate` walks its completed blocks (all but the last, which may be partially filled) and checks whether their content hash exists in `hash_to_block_id`. If yes, those blocks are cached — no allocation needed. Only non-cached blocks consume free slots.

```python
# can_allocate: count how many blocks are already cached
h = -1
for i in range(seq.num_blocks - 1):
    token_ids = seq.block(i)
    h = BlockManager.compute_hash(token_ids, h)
    block_id = self.hash_to_block_id.get(h, -1)
    if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
        break
    num_cached_blocks += 1
```

The hash is chained: `hash(block_i)` depends on `hash(block_{i-1})`, making it a rolling hash of the entire prefix. This ensures two sequences with the same first $k$ blocks always get the same hash chain.

**Reference counting:**
- Each block has `ref_count`
- When a sequence is allocated, cached blocks get `ref_count += 1`
- On deallocation (`deallocate`), blocks are walked in reverse: `ref_count -= 1`, and only freed when `ref_count == 0`
- This allows multiple sequences to safely share prefix blocks

**What happens when memory is full:**

In `scheduler.py`, if `can_allocate` returns `-1` (not enough free blocks), the scheduler tries to preempt:
```python
while not self.block_manager.can_append(seq):
    if self.running:
        self.preempt(self.running.pop())   # evict lowest-priority running seq
    else:
        self.preempt(seq)
        break
```

`preempt` deallocates the sequence's blocks, puts it back in the waiting queue, and marks it for re-prefill. This is **swap-to-CPU** style eviction by re-computation — nano-vllm does not implement a separate CPU swap buffer. There is no LRU or other eviction policy for the prefix cache; eviction is implicit (free blocks are reused FIFO, and the hash entry is deleted when the block is reallocated).

**Hash blocks (`hash_blocks`):**

After each scheduler step, `hash_blocks` is called to record the content hash of newly completed full blocks:
```python
for i in range(start, end):
    block = self.blocks[seq.block_table[i]]
    h = self.compute_hash(seq.block(i), h)
    block.update(h, token_ids)
    self.hash_to_block_id[h] = block.block_id
```

This is how blocks become available for future prefix reuse.

---

### Component 3: Scheduler (`engine/scheduler.py`)

**Continuous batching principle:** every call to `schedule()` returns a fresh batch. There is no "wait for batch to finish" — finished sequences are removed immediately in `postprocess()` and new ones fill the slot next step.

**Two phases per step:**

1. **Prefill priority:** The scheduler first tries to admit sequences from `waiting`. It respects:
   - `max_num_seqs`: total sequences in flight
   - `max_num_batched_tokens`: total tokens in one step
   - Chunked prefill: a single long prompt can be split across multiple steps (only for the first sequence in a chunk)
   - Prefix cache: if `num_cached_blocks > 0`, those tokens are skipped in the first prefill pass

2. **Decode fallback:** If no waiting sequences were admitted, all running sequences each contribute 1 token (their last generated token). This is the decode step.

The two phases are mutually exclusive per step — a step is either all-prefill or all-decode. This avoids the complexity of mixed batches while still allowing continuous request ingestion between decode steps.

**Preemption:** occurs when a decode sequence needs a new block but none are free. The scheduler evicts the most-recently-added running sequence (LIFO), which is an approximation of shortest-remaining-work.

---

### Component 4: Attention Kernel (`layers/attention.py`)

**The Triton scatter kernel (`store_kvcache_kernel`):**

```python
@triton.jit
def store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride,
                          k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D: tl.constexpr):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key = tl.load(key_ptr + idx * key_stride + tl.arange(0, D))
    tl.store(k_cache_ptr + slot * D + tl.arange(0, D), key)
    # same for value
```

`slot_mapping` is prepared by `model_runner.py` before each forward pass. It maps each input token position (in the packed input tensor) to a physical cache slot: `block_table[block_id] * block_size + offset_in_block`. The Triton kernel writes each token's K and V directly to its physical location — a scatter operation with no intermediate buffer.

**FlashAttention calls:**
- **Prefill:** `flash_attn_varlen_func` — handles variable-length sequences in one kernel call using cumulative sequence length arrays (`cu_seqlens_q`, `cu_seqlens_k`). With prefix caching, `cu_seqlens_k > cu_seqlens_q` because cached tokens are attended to but not recomputed; `k, v` are replaced by `k_cache, v_cache` and a block table is passed.
- **Decode:** `flash_attn_with_kvcache` — single-token queries attending over the full KV cache. Block tables route attention to non-contiguous physical pages.

---

### Component 5: Tensor Parallelism (`engine/model_runner.py`, `layers/linear.py`)

For multi-GPU setups, nano-vllm uses **tensor parallelism** (splitting each layer's weights across GPUs):

- **Column-parallel:** each GPU holds `output_size / tp_size` output rows. Q/K/V projections split this way so each GPU handles its own heads.
- **Row-parallel:** each GPU holds `input_size / tp_size` input columns; outputs are summed via `dist.all_reduce`.
- **Communication:** for `tp_size > 1`, the scheduler lives on rank 0 and broadcasts work to other ranks via a `SharedMemory` segment. The method name and args are pickled into shared memory; worker ranks `Event.wait()` for a signal, read the method, execute it, and loop.
- **CUDA graphs:** captured once at startup for decode batch sizes 1, 2, 4, 8, 16, 32, ... up to `max_num_seqs`. Replaying a CUDA graph eliminates Python overhead and CPU-GPU synchronization for the common decode path. Prefill always runs eagerly (variable length, can't be captured).

---

## Experiments (run on Adroit MIG A100)

See `experiments.md` for step-by-step instructions and results template.

---

## Deep Dive: The Paging System

**Block size:** 256 tokens, hardcoded in `Sequence.block_size` and asserted to be a multiple of 256 in `Config.__post_init__`. Not configurable without code change. The 256 constraint exists because the Triton kernel's `D` (head dimension times num_heads) is a `tl.constexpr` that must be a power of two.

**Block freeing:** reference-counted. `deallocate(seq)` walks the block table in reverse, decrements each block's `ref_count`, and calls `_deallocate_block` only when it hits zero. This means a prefix block shared by 3 sequences isn't freed until all 3 finish.

**When memory is full:** the scheduler preempts (re-queues and deallocates) the most-recently-added running sequence. No LRU, no CPU offload. The preempted sequence will be re-prefilled from scratch on its next admission — unless its prefix blocks are still in cache (which they might be, if nobody evicted them).

**Alternative eviction policy:** You could implement LRU by maintaining a doubly-linked list of cached (ref_count == 0) blocks ordered by last-access time, and evicting the least-recently-used one when `free_block_ids` is empty. The current FIFO `deque` would be replaced with this structure. This is what production vLLM does.

---

## Connection to Research

KV cache memory layout matters for interpretability work. When running probes on intermediate hidden states during inference, you need to know where activations live. In nano-vllm, hidden states are ephemeral — they exist only for one forward pass and aren't stored in the KV cache (only K and V projections are). To extract intermediate activations for analysis, you would add forward hooks on the attention or MLP layers, which fire during the standard forward pass and can write activations to a separate buffer without disrupting the inference pipeline.

---

## Summary

nano-vllm implements three ideas from the vLLM paper in ~1,200 lines:

| Idea | Code location | Key insight |
|------|--------------|-------------|
| KV Cache | `model_runner.py:allocate_kv_cache` | One pre-allocated slab; per-layer views assigned at startup |
| PagedAttention | `engine/block_manager.py` | Hash-chained content addressing enables prefix sharing with ref-counting |
| Continuous Batching | `engine/scheduler.py` | Prefill-first scheduling with chunked prefill; preemption on OOM |
| Flash Attention | `layers/attention.py` | Triton scatter kernel to write K/V; FA varlen (prefill) / FA kvcache (decode) |
| Tensor Parallelism | `layers/linear.py`, `model_runner.py` | Column/row split with shared-memory IPC between ranks |
