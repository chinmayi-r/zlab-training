# Adroit Experiments: nano-vllm

Run these on a MIG A100 slice on Adroit. Each experiment is self-contained.

## Setup

```bash
# On Adroit login node
ssh adroit.princeton.edu

# Request an interactive GPU session (MIG A100 slice)
salloc --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G \
       --gres=gpu:a100:1 --time=02:00:00 --partition=mig

# Once on the compute node:
module load anaconda3/2024.2
conda create -n nanovllm python=3.11 -y
conda activate nanovllm

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
pip install transformers triton xxhash safetensors numpy tqdm

# Clone nano-vllm
git clone https://github.com/GeeeekExplorer/nano-vllm.git
cd nano-vllm
pip install -e .

# Download a small model (Qwen2.5-0.5B fits easily on a MIG slice)
# Option A: from HuggingFace hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='~/huggingface/Qwen2.5-0.5B-Instruct')
"
# Option B: if already on /scratch/gpfs, just set MODEL_PATH to that directory
MODEL_PATH=~/huggingface/Qwen2.5-0.5B-Instruct
```

> **Note:** nano-vllm's `models/` only has `qwen3.py` but Qwen2.5 uses the same architecture.
> If the import fails, either use Qwen3-0.6B or patch `llm_engine.py` to import the right model class.
> Qwen3-0.6B is the cleanest choice since it's the repo's native model.

---

## Experiment 0: Sanity Check — Basic Generation

```python
# save as test_basic.py
import os
from nanovllm import LLM, SamplingParams

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B")
llm = LLM(MODEL_PATH, enforce_eager=True, max_model_len=2048)

outputs = llm.generate(
    ["The capital of France is", "The speed of light is"],
    SamplingParams(temperature=0.7, max_tokens=32)
)
for o in outputs:
    print(o["text"])
```

Expected: coherent short completions. If this works, the install is good.

---

## Experiment 1: Prefix Cache Hit Rate

**Hypothesis:** sending the same prompt twice should be faster on the second call because prompt tokens are already in the KV cache.

```python
# save as exp1_prefix_cache.py
import os, time
from nanovllm import LLM, SamplingParams

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B")
llm = LLM(MODEL_PATH, enforce_eager=True, max_model_len=2048)

long_prompt = "Explain the theory of relativity in detail. " * 20   # ~200 tokens
sp = SamplingParams(temperature=0.7, max_tokens=64)

# First call: cold cache
t0 = time.perf_counter()
llm.generate([long_prompt], sp, use_tqdm=False)
t_cold = time.perf_counter() - t0

# Second call: warm cache (prefix should be cached)
t0 = time.perf_counter()
llm.generate([long_prompt], sp, use_tqdm=False)
t_warm = time.perf_counter() - t0

print(f"Cold (no cache): {t_cold:.3f}s")
print(f"Warm (prefix cached): {t_warm:.3f}s")
print(f"Speedup: {t_cold/t_warm:.2f}x")
```

**What to look for:** The warm call skips prefill for all complete blocks of the prompt. `block_manager.can_allocate` returns `num_cached_blocks > 0`, so `scheduler.py` sets `seq.num_scheduled_tokens` to only the uncached tail. You should see a speedup proportional to `(prompt_tokens / block_size)` blocks being skipped.

**If there's no speedup:** check whether `num_cached_tokens` is being set correctly after the first call by adding a print in `scheduler.postprocess`.

---

## Experiment 2: Batch Size vs. Throughput

**Hypothesis:** throughput (tokens/sec) should increase with batch size because attention is memory-bandwidth-bound during decode, and batching amortizes the memory reads of K/V.

```python
# save as exp2_batch_throughput.py
import os, time
from nanovllm import LLM, SamplingParams

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B")
llm = LLM(MODEL_PATH, enforce_eager=True, max_model_len=2048)

results = []
for batch_size in [1, 2, 4, 8, 16]:
    prompts = ["Write a short poem about the ocean."] * batch_size
    sp = SamplingParams(temperature=0.7, max_tokens=128, ignore_eos=True)

    # warmup
    llm.generate(prompts[:1], SamplingParams(max_tokens=8), use_tqdm=False)

    t0 = time.perf_counter()
    llm.generate(prompts, [sp] * batch_size, use_tqdm=False)
    elapsed = time.perf_counter() - t0

    total_tokens = batch_size * 128
    throughput = total_tokens / elapsed
    results.append((batch_size, throughput, elapsed))
    print(f"batch={batch_size:3d}  tokens={total_tokens:6d}  time={elapsed:.2f}s  throughput={throughput:.1f} tok/s")

# Plot (optional, requires matplotlib)
try:
    import matplotlib.pyplot as plt
    bs = [r[0] for r in results]
    tps = [r[1] for r in results]
    plt.figure(figsize=(7, 4))
    plt.plot(bs, tps, 'o-')
    plt.xlabel("Batch size")
    plt.ylabel("Throughput (tok/s)")
    plt.title("Batch size vs. decode throughput")
    plt.tight_layout()
    plt.savefig("batch_throughput.png", dpi=150)
    print("Saved batch_throughput.png")
except ImportError:
    pass
```

**What to look for:** near-linear scaling at small batch sizes (GPU underutilized), then diminishing returns as memory bandwidth saturates. The curve flattening point tells you where the GPU is fully utilized.

---

## Experiment 3: Sequence Length vs. GPU Memory

**Hypothesis:** KV cache grows linearly with sequence length. At each decode step, one new K/V pair per layer per head is added to the cache.

```python
# save as exp3_memory_scaling.py
import os, time
import torch
from nanovllm import LLM, SamplingParams

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B")

results = []
for target_len in [128, 256, 512, 1024]:
    llm = LLM(MODEL_PATH, enforce_eager=True, max_model_len=target_len + 64)
    torch.cuda.reset_peak_memory_stats()

    prompts = ["Write a very long essay about artificial intelligence."]
    sp = SamplingParams(temperature=0.7, max_tokens=target_len, ignore_eos=True)
    llm.generate(prompts, sp, use_tqdm=False)

    mem_mb = torch.cuda.max_memory_allocated() / 1e6
    results.append((target_len, mem_mb))
    print(f"seq_len={target_len:5d}  peak_mem={mem_mb:.1f} MB")
    del llm
    torch.cuda.empty_cache()

# Check linearity
for i in range(1, len(results)):
    prev_len, prev_mem = results[i-1]
    curr_len, curr_mem = results[i]
    ratio = curr_len / prev_len
    mem_ratio = curr_mem / prev_mem
    print(f"Length {prev_len}→{curr_len} ({ratio:.1f}x): memory {prev_mem:.0f}→{curr_mem:.0f} MB ({mem_ratio:.2f}x)")
```

**What to look for:** memory ratio should closely track length ratio once the KV cache dominates. Model weights are fixed; only the cache grows. Divergence at short lengths = model weight overhead is significant relative to cache.

**Theoretical expectation for Qwen3-0.6B:**  
`block_bytes = 2 * 28_layers * 256_block_size * 2_kv_heads * 64_head_dim * 2_bytes_bf16 = 14,680,064 bytes ≈ 14 MB per block`  
At 1024 tokens = 4 full blocks → ~56 MB of KV cache alone.

---

## Experiment 4 (Optional): CUDA Graph Speedup

**Hypothesis:** CUDA graphs (default `enforce_eager=False`) should make decode faster by eliminating Python dispatch overhead.

```python
# save as exp4_cuda_graph.py
import os, time
from nanovllm import LLM, SamplingParams

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B")

for eager in [True, False]:
    label = "eager" if eager else "cuda_graph"
    llm = LLM(MODEL_PATH, enforce_eager=eager, max_model_len=2048)
    prompts = ["Tell me about quantum computing."] * 8
    sp = SamplingParams(temperature=0.7, max_tokens=128, ignore_eos=True)

    # warmup
    llm.generate(prompts[:1], SamplingParams(max_tokens=4), use_tqdm=False)

    t0 = time.perf_counter()
    llm.generate(prompts, [sp]*8, use_tqdm=False)
    elapsed = time.perf_counter() - t0
    tps = 8 * 128 / elapsed
    print(f"{label:12s}  time={elapsed:.2f}s  throughput={tps:.1f} tok/s")
    del llm
```

---

## Recording Results

Fill in this table after running experiments:

| Experiment | Key metric | Result |
|-----------|------------|--------|
| Exp 1: prefix cache | Speedup 2nd call | ___ x |
| Exp 2: batch=1 | Decode throughput | ___ tok/s |
| Exp 2: batch=8 | Decode throughput | ___ tok/s |
| Exp 2: batch=16 | Decode throughput | ___ tok/s |
| Exp 3: seq_len=256 | Peak GPU mem | ___ MB |
| Exp 3: seq_len=1024 | Peak GPU mem | ___ MB |
| Exp 3: linearity | 4x length → ?x memory | ___ x |
| Exp 4: eager | Throughput | ___ tok/s |
| Exp 4: cuda graph | Throughput | ___ tok/s |

---

## Adroit-specific Notes

- MIG slices on Adroit give you a fraction of an A100 (typically 1/7th = ~10 GB). Qwen3-0.6B weights are ~1.2 GB in bf16, leaving ~8 GB for KV cache — enough for hundreds of blocks.
- If you get OOM on the MIG slice, reduce `gpu_memory_utilization` to 0.7.
- Use `squeue -u $USER` to check your job status.
- Transfer results back: `scp adroit:/path/to/results.txt .`
- If matplotlib isn't available, write CSVs and plot locally.

## Slurm Batch Script (for longer runs)

```bash
#!/bin/bash
#SBATCH --job-name=nanovllm-exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --partition=mig
#SBATCH --output=nanovllm_%j.out

module load anaconda3/2024.2
conda activate nanovllm
cd ~/nano-vllm

python exp1_prefix_cache.py
python exp2_batch_throughput.py
python exp3_memory_scaling.py
python exp4_cuda_graph.py
```

Submit with: `sbatch run_experiments.sh`
