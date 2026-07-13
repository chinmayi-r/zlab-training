# Diffusers — Running an Unconditional Diffusion Model (Warmup Report)

**Repo studied:** [huggingface/diffusers](https://github.com/huggingface/diffusers) (Beginner track)
**Task:** reproduce the official ["Train a diffusion model"](https://huggingface.co/docs/diffusers/tutorials/basic_training)
tutorial — train an unconditional **DDPM** from scratch on a small image dataset, generate
samples, and document how the system works and what I observed.
**Author:** [you] · **Date:** [fill]

> Fill markers: `[fill]` = drop in your real numbers/observations after the run. Everything
> conceptual is written out; the results sections are scaffolded and waiting for data.

---

## Part A — What this is and why it exists

**The gap it fills.** Generative image models learn a distribution over images and let you
sample new ones. Earlier families had known pain: **GANs** generate in one shot but are
unstable to train (mode collapse, delicate min–max balance); **VAEs** are stable but tend to
produce blurry samples. **Diffusion models** (Ho et al. 2020, DDPM) hit a different point on
the trade-off: stable, likelihood-style training with a simple regression loss, and
high-sample-quality — at the cost of *slow, many-step sampling*. They are now the backbone of
Stable Diffusion / DALL·E-class systems.

**What Diffusers is.** A HuggingFace library that packages diffusion models as composable
parts — **models** (`UNet2DModel`, …), **schedulers** (`DDPMScheduler`, `DDIMScheduler`, …),
and **pipelines** (`DDPMPipeline`, `StableDiffusionPipeline`, …) — so you can train from
scratch, fine-tune, or just do inference without reimplementing the math.

**Contrast with the AI-Scientist week.** That was an autonomous *agent* (LLM + tool use);
the report was about diagnosing its failure modes. Diffusers is a *library* you drive
directly — no LLM, no API key, no agent. So "failures" here are your own engineering
choices (out-of-memory, resolution, offline data), not an agent's tool-use mistakes.

## Part B — How the system works (DDPM)

**Forward process (fixed, no learning).** Given a clean image `x₀`, repeatedly add a little
Gaussian noise over `T` steps (`num_train_timesteps`, here 1000) following a variance
schedule `βₜ`. A closed form lets you jump straight to any step:
`xₜ = √(ᾱₜ)·x₀ + √(1−ᾱₜ)·ε`, where `ε ~ N(0, I)` and `ᾱₜ` is the cumulative product of
`(1−βₜ)`. In code this is `scheduler.add_noise(x0, noise, t)`.

**The model.** `UNet2DModel` takes the noised image `xₜ` and the timestep `t` and predicts
the noise `ε` that was added. It's a U-Net (down/up sampling with skip connections) plus a
sinusoidal timestep embedding so one network handles every noise level.

**Training objective.** Just a regression: sample a random `t` per image, noise the image,
predict the noise, minimize `MSE(ε_pred, ε)`. That single simple loss is the whole training
signal — this is why diffusion training is so stable.

**Reverse process (sampling).** Start from pure noise `x_T ~ N(0, I)` and iteratively apply
the model to remove a step of noise, `T → 0`, ending at a fresh generated image. In code this
is `DDPMPipeline(unet, scheduler)(...)`.

| Component | Role in this run |
|---|---|
| `UNet2DModel` | the denoiser ε-predictor (shrunk here: `block_out_channels=(64,128,128,256)`) |
| `DDPMScheduler` | defines `βₜ` schedule, `add_noise`, and the reverse step (`num_train_timesteps=1000`) |
| `DDPMPipeline` | wraps model+scheduler to generate sample grids |

## Part C — What I ran (method)

| Setting | Value |
|---|---|
| Dataset | `huggan/smithsonian_butterflies_subset` (~1000 imgs) [or: `[fill]`] |
| Resolution | `[fill]` (32 local / 64 Adroit) |
| Model | `UNet2DModel`, `block_out_channels=(64,128,128,256)`, `layers_per_block=2` |
| Scheduler | `DDPMScheduler`, `num_train_timesteps=1000` |
| Optimizer / lr | AdamW / 1e-4 |
| Batch size | `[fill]` |
| Epochs | `[fill]` |
| Seed | 0 |
| Machine | `[fill]` (RTX 3050 4 GB local / MIG A100 20 GB Adroit) |
| Script | `train_ddpm.py` (this kit) |

Reproduce: `bash setup_local.sh && bash run_local.sh` (local), or `prefetch_dataset.sh` on
the Adroit login node then `sbatch slurm/train_ddpm.slurm`.

## Part D — Results

**Loss curve** (per-epoch MSE, from console):

| epoch | loss |
|---|---|
| 1 | `[fill]` |
| … | … |
| final | `[fill]` |

**Samples over training** — paste the grids the run saved (`ddpm-out/samples_*.png`):

- `samples_epoch5.png` — `[fill: mostly noise? blobs of color?]`
- `samples_final.png` — `[fill: recognizable butterflies? blurry? mode-collapsed?]`

**Resource use:** peak VRAM `[fill]`, wall-clock time `[fill]` for `[fill]` epochs.

## Part E — Failures / gotchas hit

- CUDA OOM at `[fill]` resolution/batch → fixed by `[fill]` (lower batch / lower res).
- Adroit: HuggingFace download blocked on compute node → solved with login-node
  `prefetch_dataset.sh` + `HF_HUB_OFFLINE=1` (documented, `[fill: did you use Adroit?]`).
- WSL DataLoader deadlock avoided with `num_workers=0`.
- `[fill: anything else]`

## Part F — Reflection

- **What I learned:** `[fill — e.g., how the noise-prediction objective works, how sample
  quality tracks training, why sampling is slow (T steps)]`.
- **Contrast with the agent week:** `[fill — you control the library directly; failures are
  config/hardware, not agent tool-use]`.
- **Limitations:** tiny model + few epochs + one dataset → samples are a smoke-test, not
  publication quality; no FID/quantitative sample metric, only visual + training loss.
- **Natural next steps:** more epochs / higher res; swap `DDPMScheduler`→`DDIMScheduler`
  for faster sampling; try `huggan/flowers-102-categories`; add an FID score.

## References / reading

- Ho, Jain, Abbeel — *Denoising Diffusion Probabilistic Models* (DDPM), 2020, arXiv:2006.11239
- Song et al. — *Denoising Diffusion Implicit Models* (DDIM, faster sampling), arXiv:2010.02502
- Diffusers docs — "Train a diffusion model": https://huggingface.co/docs/diffusers/tutorials/basic_training
- Diffusers repo: https://github.com/huggingface/diffusers
- (Optional) Rombach et al. — *Latent Diffusion / Stable Diffusion*, arXiv:2112.10752
