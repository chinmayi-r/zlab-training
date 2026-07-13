# Diffusers — beginner report kit (unconditional DDPM)

Warmup-program report on the **[HuggingFace Diffusers](https://github.com/huggingface/diffusers)**
library (the "Beginner: Diffusion model library" entry). It reproduces the official
["Train a diffusion model"](https://huggingface.co/docs/diffusers/tutorials/basic_training)
tutorial — an **unconditional DDPM** (`UNet2DModel` + `DDPMScheduler`) trained from scratch —
shrunk so it runs on a 4 GB laptop GPU in minutes, and offline on Adroit.

Unlike the AI-Scientist week, **there is no LLM and no API key** — Diffusers is a plain
training library, so the only cost is GPU time (both machines are $0).

## The beginner diffusion datasets (from Diffusers' own tutorials/examples)

| Dataset id | Where it's used | Res |
|---|---|---|
| `huggan/smithsonian_butterflies_subset` | the "Train a diffusion model" tutorial (**default here**) | 128 |
| `huggan/flowers-102-categories` | `examples/unconditional_image_generation` | 64 |
| `huggan/pokemon` | `examples/unconditional_image_generation` | 64 |
| `lambdalabs/pokemon-blip-captions` | `examples/text_to_image` (SD fine-tune) | 512 |

Swap any in with `DATASET=...` (the loader just reads the `image` column). All are HF
downloads — which matters for *where* you run (see below).

## Files

| File | What it does |
|---|---|
| `train_ddpm.py` | self-contained DDPM training (real Diffusers API, plain torch loop). Saves the pipeline + `samples_final.png`. |
| `sample.py` | generate images from your trained model **or** a pretrained Hub pipeline. |
| `setup_local.sh` | WSL2 miniconda + cu124 torch + `diffusers`/`datasets`/`accelerate`. |
| `run_local.sh` | trains with 4 GB-friendly defaults (32px, batch 16, 10 epochs). |
| `prefetch_dataset.sh` | **Adroit login node** — caches the HF dataset onto scratch. |
| `slurm/train_ddpm.slurm` | **Adroit compute node** — trains offline on a MIG A100 slice. |

---

## Path A — Local PC (recommended: simplest, HF download just works)

```bash
cd ~/zlab-training/zlab-training/diffusers   # your checkout of this branch
bash setup_local.sh                          # miniconda + torch + diffusers
bash run_local.sh                            # downloads butterflies, trains, saves samples
```
Output lands in `ddpm-out/`: `samples_epoch5.png`, `samples_final.png`, and the saved
pipeline. Open `samples_final.png` to see generated butterflies. Knobs (env vars):
`IMAGE_SIZE` (32→64 if VRAM allows), `EPOCHS`, `BATCH_SIZE`, `DATASET`.

**4 GB VRAM note:** defaults (32px, batch 16) fit comfortably. If you hit CUDA OOM, drop
`BATCH_SIZE=8`. CPU works too (no GPU needed) — just slower.

## Path B — Adroit (faster A100, but HF is blocked on compute nodes)

HuggingFace downloads are blocked by the compute-node proxy (same wall as last week's
CIFAR). So **pre-cache on the login node first**, then train offline:

```bash
# 1) LOGIN node (has internet) — set up env once + cache the dataset:
cd /scratch/network/$USER/zlab-training/diffusers
ENV_NAME=diffusers bash setup_local.sh        # same setup script works on Adroit
bash prefetch_dataset.sh                       # -> caches into /scratch/network/$USER/hf_cache

# 2) COMPUTE node — train offline (dataset already cached, HF_HUB_OFFLINE=1 in the script):
sbatch slurm/train_ddpm.slurm
```

## After the run — collect results into the repo

```bash
# from the diffusers/ kit dir, copy the deliverables in and commit:
git add diffusers/ddpm-out/samples_final.png diffusers/ddpm-out/*.png
git commit -m "diffusers: DDPM sample grids + loss log"
git push origin claude/ai-scientist-v2-failures-l9vej5
```
(The full pipeline weights are large — commit the **sample PNGs** and the console loss log,
not the multi-hundred-MB `.safetensors`, unless you want them.)

## For the report

- **What it is:** DDPM = forward process adds Gaussian noise over `num_train_timesteps`;
  the `UNet2DModel` learns to predict that noise; sampling runs the reverse process.
- **What to record:** per-epoch loss (printed), the sample grids over epochs (quality
  improving), final samples, VRAM/time, and any failure you hit (OOM, the Adroit HF wall).
- **Contrast with last week:** no agent, no LLM — you drive the library directly, so
  failures are *your* config (OOM, resolution, offline data), not an agent's tool-use.
