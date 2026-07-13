# Diffusers — Running an Unconditional Diffusion Model (Warmup Report)

**Repo studied:** [huggingface/diffusers](https://github.com/huggingface/diffusers) (Beginner track)
**Task:** reproduce the official ["Train a diffusion model"](https://huggingface.co/docs/diffusers/tutorials/basic_training)
tutorial — train an unconditional **DDPM** from scratch on a small image dataset, generate
samples, and document how the system works and what I observed.
**Machine:** NVIDIA RTX 3060 Laptop GPU (6 GB), Windows 11, native Python 3.10 + torch 2.5.1+cu124.
**Run date:** 2026-07-12. **Wall-clock:** ~29.5 min end-to-end (train + all sample grids).

> This is the *actual* run, not a scaffold — every number below comes from the console log
> ([`ddpm-out/` samples](ddpm-out/) + [`loss_log.csv`](loss_log.csv)).

---

## Part A — What this is and why it exists

**The gap it fills.** Generative image models learn a distribution over images and let you
sample new ones. Earlier families had known pain: **GANs** generate in one shot but are
unstable to train (mode collapse, delicate min–max balance); **VAEs** are stable but tend to
produce blurry samples. **Diffusion models** (Ho et al. 2020, DDPM) hit a different point on
the trade-off: stable, likelihood-style training with a simple regression loss, and
high sample quality — at the cost of *slow, many-step sampling*. They are now the backbone of
Stable Diffusion / DALL·E-class systems.

**What Diffusers is.** A HuggingFace library that packages diffusion models as composable
parts — **models** (`UNet2DModel`, …), **schedulers** (`DDPMScheduler`, `DDIMScheduler`, …),
and **pipelines** (`DDPMPipeline`, `StableDiffusionPipeline`, …) — so you can train from
scratch, fine-tune, or just do inference without reimplementing the math. **I did not rewrite
any of that library code**: `train_ddpm.py` is a plain PyTorch loop that *calls* the real
Diffusers API (`UNet2DModel` + `DDPMScheduler` + `DDPMPipeline`). The diffusion math lives in
the library, untouched.

**Contrast with the AI-Scientist week.** That was an autonomous *agent* (LLM + tool use);
the report was about diagnosing its failure modes. Diffusers is a *library* you drive
directly — no LLM, no API key, no agent. So "failures" here are your own engineering
choices (out-of-memory, resolution, offline data, OS/env mismatch), not an agent's tool-use
mistakes. See Part E for the ones I actually hit.

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
signal — this is why diffusion training is so stable (and why my loss curve, Part D, is
monotone-ish with no GAN-style oscillation).

**Reverse process (sampling).** Start from pure noise `x_T ~ N(0, I)` and iteratively apply
the model to remove a step of noise, `T → 0`, ending at a fresh generated image. In code this
is `DDPMPipeline(unet, scheduler)(...)`. This is the slow part: **1000 model evaluations per
image**, ≈70 s to make one 16-image grid on this GPU.

| Component | Role in this run |
|---|---|
| `UNet2DModel` | the denoiser ε-predictor. `block_out_channels=(64,128,128,256)`, `layers_per_block=2` → **17.2 M params** |
| `DDPMScheduler` | defines `βₜ` schedule, `add_noise`, and the reverse step (`num_train_timesteps=1000`) |
| `DDPMPipeline` | wraps model+scheduler to generate the sample grids |

## Part C — What I ran (method)

| Setting | Value |
|---|---|
| Dataset | `huggan/smithsonian_butterflies_subset` — **1000 images** (the tutorial default) |
| Resolution | **64 × 64** (downloaded, resized, normalized to `[-1,1]`) |
| Model | `UNet2DModel`, `(64,128,128,256)`, `layers_per_block=2`, 17.2 M params |
| Scheduler | `DDPMScheduler`, `num_train_timesteps=1000` |
| Optimizer / lr | AdamW / 1e-4 |
| Batch size | 16 (→ 63 steps/epoch) |
| Epochs | 50 |
| Seed | 0 |
| Machine | RTX 3060 Laptop GPU 6 GB · Windows 11 native · torch 2.5.1+cu124 |
| Sample cadence | grid every 10 epochs + final (seed-0, all 16 shown — **not** cherry-picked) |
| Script | `train_ddpm.py` (this kit; Diffusers API unmodified) |

**Reproduce (Windows native, what I actually did):**
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python -m pip install "diffusers[training]" transformers datasets accelerate safetensors Pillow
$env:HF_HOME="$PWD\hf_cache"
.venv\Scripts\python train_ddpm.py --image_size 64 --batch_size 16 --epochs 50 --sample_every 10 --out_dir ddpm-out
```
(The committed `setup_local.sh`/`run_local.sh` do the same on WSL/Linux+conda; see Part E for
why I bypassed them on native Windows.)

## Part D — Results

**Loss curve** (per-epoch MSE noise-prediction loss, full log in [`loss_log.csv`](loss_log.csv)):

| epoch | 1 | 5 | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|---|
| loss | 0.264 | 0.054 | 0.040 | 0.030 | 0.026 | 0.025 | **0.018** |

Shape: a fast drop in the first ~5 epochs (0.26 → 0.05), then a long slow decline into a
noisy plateau around **0.024–0.025** from epoch ~25 on (individual epochs bounce between
0.018 and 0.027 — expected, since each epoch's loss averages random timesteps `t`, and high-`t`
batches are intrinsically harder). No divergence, no oscillation — the hallmark of the simple
regression objective.

**Samples over training** — [`ddpm-out/evolution.png`](ddpm-out/evolution.png) stitches these:

| grid | what it shows |
|---|---|
| [`samples_epoch10.png`](ddpm-out/samples_epoch10.png) | recognizable butterfly *silhouettes* already, but muddy, desaturated, grainy wing texture |
| [`samples_epoch30.png`](ddpm-out/samples_epoch30.png) | clean bilaterally-symmetric wings, colors separating, white background emerging |
| [`samples_epoch50.png` / `samples_final.png`](ddpm-out/samples_final.png) | vivid, clearly butterfly-shaped: symmetric wings, visible body, distinct greens/blues/oranges/reds/yellows on a clean background |

**Resource use:** single-forward peak (batch 16, 64px) = **332 MiB**; total GPU memory in use
during training ≈ **2.5 GB** (incl. the Windows desktop) — comfortably inside 6 GB, no OOM.
**Timing:** ≈**23 s/epoch** (≈19.5 min for 50 epochs) + ≈**70 s per sample grid** (6 grids ≈ 7 min).

## Part E — Failures / gotchas actually hit (and fixes)

1. **`setup_local.sh` doesn't run on native Windows.** It assumes WSL2 + miniconda + `wget`
   (line 1 `wget … Miniconda3-Linux-x86_64.sh`). This machine is native Windows Python 3.10,
   no conda. **Fix:** built the *same* stack (torch cu124 + diffusers/datasets/accelerate) in a
   `venv` via pip. Same libraries, different packaging — model code untouched.
2. **Only ~1.5 GB VRAM free at first** (Brave + Epic + Fortnite held ~4.5 GB of the 6 GB).
   That would have forced 32px. It freed to ~5 GB before the real run, so I ran **64px**. If it
   recurs, `BATCH_SIZE=8 IMAGE_SIZE=32` is the fallback.
3. **`expandable_segments:True` not supported on Windows** — torch prints a warning and
   ignores it. Harmless; left in (it *is* honored on Linux/Adroit).
4. **`DDPMPipeline` floods stdout** with a 1000-step tqdm bar per grid. I piped training to a
   log and `grep`-ed the epoch/loss/sample lines rather than editing the library or the script.
5. **No CUDA OOM** at 64px/batch-16 — the peak (332 MiB fwd) left lots of headroom, contra my
   initial worry.
6. `num_workers=0` in the DataLoader (kit default) avoids the multi-worker deadlock seen under
   WSL last week; it's also the safe choice on Windows.

## Part F — Reflection & self-critique

**What I learned.** The noise-prediction objective really is the whole story: one MSE loss,
no adversary, no KL term, and it trains monotonically. Sample quality visibly tracks the loss
— epoch 10 (loss 0.04) already has butterfly *shape*; the slow grind from 0.04 → 0.024 is
where *color fidelity and sharpness* come in (compare epoch-10 vs epoch-50 grids). And the
slowness is structural: sampling is 1000 sequential UNet passes, which dominated my wall-clock
far more than training did.

**Where this run can be critiqued — and my response** (per-decision):

- *"64px + 17 M params + 50 epochs is a toy; these aren't publication-quality samples."*
  Correct, and intended — this is a smoke test to show the pipeline works end-to-end on a
  laptop, not SOTA. The samples are honestly labeled (seed-0, all 16 shown, not cherry-picked).
- *"Loss ≠ sample quality; you report no FID/IS."* True — MSE noise-loss is only a proxy. I
  lean on the *visual* evolution grid as the qualitative check. Adding a real **FID** against
  the training set is the single highest-value next step and is called out below.
- *"One seed, one dataset — no error bars, no generality claim."* Agreed; I make no
  generality claim. Depth-over-breadth (per Taimeng's past note) is deliberate: I ran **one**
  dataset thoroughly (loss curve + 5-point sample evolution + resource profile) rather than
  sampling butterflies/flowers/pokemon shallowly. Breadth is a knob (`DATASET=…`) I left off.
- *"White backgrounds everywhere — did it mode-collapse?"* No — wing shapes/colors vary widely
  across the 16 samples. The white background is a **dataset bias** (Smithsonian specimen cards
  are photographed on white), faithfully learned, not collapse.
- *"You changed the environment from the committed scripts."* Only the *packaging* (venv vs
  conda) to match native Windows; the training script and the Diffusers library are byte-for-
  byte the originals. The env delta is documented in Part C/E so the result is reproducible.
- *"~30 min is slow for this."* ~2/3 of that is the six 1000-step sample grids, not training.
  Switching the sampler to **DDIM** (e.g. 50 steps) would cut sampling ~20× with minor quality
  loss — the obvious efficiency win, listed next.

**Limitations.** Tiny model + few epochs + one dataset + one seed → a smoke-test, not a
benchmark; no quantitative sample metric (visual + training loss only); sampling is slow by
construction (full 1000-step DDPM).

**Natural next steps** (ranked): (1) add an **FID** vs. the training set for a real number;
(2) swap `DDPMScheduler`→`DDIMScheduler` for ~20× faster sampling; (3) push resolution to
128px / more epochs now that VRAM headroom is confirmed; (4) try `huggan/flowers-102-categories`
as a second dataset to test generality.

## References / reading

- Ho, Jain, Abbeel — *Denoising Diffusion Probabilistic Models* (DDPM), 2020, arXiv:2006.11239
- Song et al. — *Denoising Diffusion Implicit Models* (DDIM, faster sampling), arXiv:2010.02502
- Diffusers docs — "Train a diffusion model": https://huggingface.co/docs/diffusers/tutorials/basic_training
- Diffusers repo: https://github.com/huggingface/diffusers
- (Optional) Rombach et al. — *Latent Diffusion / Stable Diffusion*, arXiv:2112.10752
