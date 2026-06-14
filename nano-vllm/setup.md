# Adroit setup for nano-vllm

## The error you hit

```
ERROR: Failed to build 'flash-attn' when getting requirements to build wheel
...
ModuleNotFoundError: No module named 'torch'
```

Two things went wrong, both fixable:

1. **Build isolation hides torch.** `pip install -e .` (and `pip install flash-attn`)
   build flash-attn in an *isolated* environment that does not contain torch, but
   flash-attn's `setup.py` does `import torch` at build time. Fix: build with
   `--no-build-isolation` so it can see the torch already in your env.
2. **`adroit-vis` has no GPU / no CUDA compiler.** Building flash-attn from source
   needs `nvcc` and is a ~30-minute compile. Do it on a **GPU compute node**, not
   the login/visualization node — or skip the compile entirely with a prebuilt wheel.

## Recommended path: prebuilt wheel (fast, no compile)

Get on a GPU node first:

```bash
salloc --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --gres=gpu:1 --time=01:00:00 --partition=mig
module load anaconda3/2024.2
conda activate nanovllm
```

Check what your torch needs so you pick the matching wheel:

```bash
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| cxx11abi', torch._C._GLIBCXX_USE_CXX11_ABI)"
# you reported: torch 2.5.1+cu121  -> cuda 12.1, abi almost certainly False (pip cu121 wheels)
```

Then install the matching prebuilt wheel from the flash-attn releases
(https://github.com/Dao-AILab/flash-attention/releases). For torch 2.5 / cu12 /
python 3.11 / abiFALSE the file looks like:

```bash
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
```

> If that exact tag 404s, open the releases page, find the newest
> `...+cu12torch2.5cxx11abiFALSE-cp311-...whl`, and use its URL. The four things
> that MUST match: `torch2.5`, `cu12`, `cp311`, and `cxx11abiFALSE` (use `TRUE`
> only if the check above printed `True`).

## Fallback: build from source (slow, GPU node only)

```bash
module load cudatoolkit/12.4   # provides nvcc; pick the 12.x available via `module avail cudatoolkit`
pip install ninja psutil       # ninja makes the build parallel; without it the build can take hours
MAX_JOBS=4 pip install flash-attn==2.8.3 --no-build-isolation
```

The `--no-build-isolation` flag is the key fix for your `No module named 'torch'`
error.

## Then install nano-vllm itself

Once flash-attn is installed and importable, nano-vllm's editable install will see
the dependency as satisfied and will **not** try to rebuild it:

```bash
python -c "import flash_attn; print('flash-attn OK', flash_attn.__version__)"
cd ~/nano-vllm
pip install -e . --no-build-isolation
```

## Full clean recipe (copy-paste)

```bash
# --- on a GPU compute node ---
module load anaconda3/2024.2
conda create -n nanovllm python=3.11 -y && conda activate nanovllm
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers triton xxhash safetensors numpy tqdm
# flash-attn: prebuilt wheel matching your torch (see above)
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
git clone https://github.com/GeeeekExplorer/nano-vllm.git
cd nano-vllm && pip install -e . --no-build-isolation
```

## Get the model

Qwen3-0.6B is nano-vllm's native model (the repo only ships `models/qwen3.py`).

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='$HOME/huggingface/Qwen3-0.6B')"
```

## Sanity check before the experiment

```python
# test_basic.py
import os
from nanovllm import LLM, SamplingParams
llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B"), enforce_eager=True, max_model_len=2048)
for o in llm.generate(["The capital of France is"], SamplingParams(max_tokens=16), use_tqdm=False):
    print(o["text"])
```

If that prints a coherent completion, you're ready for `experiments/exp_preempt.py`.

## Adroit notes

- A MIG A100 slice is ~1/7 of an A100 (~10 GB). Qwen3-0.6B weights are ~1.2 GB in
  bf16. For the preemption experiment we *want* a small KV cache, so a MIG slice
  plus `--gpu-mem-util 0.30` is ideal for forcing eviction.
- `squeue -u $USER` to check your job; `scp adroit:~/.../results.csv .` to pull results back.
