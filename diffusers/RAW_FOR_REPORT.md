# RAW MATERIALS — hand this to Claude to write the final Diffusers report

Everything below is fact, pulled from the actual run. Give this whole file to a fresh Claude
chat and say: *"Write my warmup report from these raw materials."* Attach the PNGs listed in
§8 as image uploads so the writer can describe them. Numbers here are ground truth — the
writer should not invent any.

---

## 1. Assignment context

- **Program:** Princeton ZLab warmup (bi-weekly reports; explore an open-source ML repo, run
  it, write up how it works + what you observed). Repo: https://github.com/zlab-princeton-internal/warmup-program
- **This report's repo:** HuggingFace **Diffusers** (listed under "Beginner: Diffusion model
  library"). https://github.com/huggingface/diffusers
- **Task chosen:** reproduce the official "Train a diffusion model" tutorial — train an
  unconditional **DDPM** from scratch, generate samples, document it.
  Tutorial: https://huggingface.co/docs/diffusers/tutorials/basic_training
- **Reviewer guidance to honor (Taimeng, prior round):** *depth in a few things over breadth.*
  Go deep on one or two things — reconstruct what happened, find the root cause, propose/apply
  a fix — rather than surveying many datasets shallowly.
- **Constraint honored:** did NOT rewrite any Diffusers library/core math. The training script
  (`train_ddpm.py`) is a plain PyTorch loop that *calls* the real Diffusers API
  (`UNet2DModel` + `DDPMScheduler` + `DDPMPipeline`).

## 2. How DDPM works (for the concept section)

- **Forward (fixed, no learning):** add Gaussian noise to a clean image `x0` over T=1000 steps
  via schedule βt. Closed form: `xt = √(ᾱt)·x0 + √(1−ᾱt)·ε`, ε~N(0,I), ᾱt = ∏(1−βt).
  Code: `scheduler.add_noise(x0, noise, t)`.
- **Model:** `UNet2DModel` takes noised image `xt` + timestep `t`, predicts the added noise ε.
  U-Net (down/up + skips) with sinusoidal timestep embedding → one net handles all noise levels.
- **Training objective:** pick random t per image, noise it, predict noise, minimize
  `MSE(ε_pred, ε)`. One regression loss, no adversary/KL → stable training.
- **Reverse (sampling):** start from pure noise x_T~N(0,I), apply model to denoise T→0 → new
  image. Code: `DDPMPipeline(unet, scheduler)(...)`. Slow: 1000 sequential UNet passes/image.
- Family context: GANs = one-shot but unstable (mode collapse); VAEs = stable but blurry;
  diffusion = stable regression training + high quality, at the cost of slow many-step sampling.
  Backbone of Stable Diffusion / DALL·E. Papers: DDPM (Ho et al. 2020, arXiv:2006.11239),
  DDIM (Song et al. 2020, arXiv:2010.02502).

## 3. Exact run configuration

| Setting | 64px run (primary) | 128px run (Part G) |
|---|---|---|
| Dataset | `huggan/smithsonian_butterflies_subset` (1000 images) | same |
| Resolution | 64×64 | 128×128 |
| Model | `UNet2DModel`, block_out_channels=(64,128,128,256), layers_per_block=2 | same arch |
| Params | **17.2 M** | 17.2 M |
| Scheduler | `DDPMScheduler`, num_train_timesteps=1000 | same |
| Optimizer / lr | AdamW / 1e-4 | AdamW / 1e-4 |
| Batch size | 16 (→63 steps/epoch) | 8 (batch 16 OOMs at 128px) |
| Epochs | 50 | 30 |
| Seed | 0 | 0 |
| Normalization | images resized + normalized to [-1,1] | same |

- **Machine:** NVIDIA **RTX 3060 Laptop GPU (6 GB)**, Windows 11, **native** Python 3.10,
  torch 2.5.1+cu124. (Ran in a `venv`, NOT the committed WSL/conda `setup_local.sh` — see §6.)
- **Sampling cadence:** grid every 10 epochs + final; seed-0, **all 16 samples shown, not
  cherry-picked.**

**Exact commands used (Windows native):**
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python -m pip install "diffusers[training]" transformers datasets accelerate safetensors Pillow
$env:HF_HOME="$PWD\hf_cache"
.venv\Scripts\python train_ddpm.py --image_size 64 --batch_size 16 --epochs 50 --sample_every 10 --out_dir ddpm-out
```

## 4. Results — full numbers

### 4a. 64px loss curve (per-epoch MSE noise-prediction loss)
```
epoch,loss
1,0.2640  2,0.0962  3,0.0732  4,0.0540  5,0.0540  6,0.0473  7,0.0403  8,0.0358
9,0.0411  10,0.0403 11,0.0399 12,0.0343 13,0.0347 14,0.0318 15,0.0343 16,0.0318
17,0.0289 18,0.0317 19,0.0275 20,0.0295 21,0.0264 22,0.0281 23,0.0281 24,0.0267
25,0.0259 26,0.0301 27,0.0277 28,0.0252 29,0.0256 30,0.0255 31,0.0239 32,0.0250
33,0.0249 34,0.0267 35,0.0258 36,0.0207 37,0.0275 38,0.0234 39,0.0219 40,0.0249
41,0.0241 42,0.0234 43,0.0237 44,0.0240 45,0.0239 46,0.0230 47,0.0255 48,0.0260
49,0.0270 50,0.0179
```
Shape: fast drop epochs 1–5 (0.264→0.054), then slow decline to a noisy plateau ≈0.024–0.026
from ~epoch 25 on (bounces 0.018–0.027 because each epoch averages random timesteps t; high-t
batches are intrinsically harder). No divergence, no oscillation.

### 4b. 128px loss curve (30 epochs)
```
1,0.1740 2,0.0645 3,0.0468 4,0.0358 5,0.0313 6,0.0336 7,0.0293 8,0.0261 9,0.0257 10,0.0228
11,0.0291 12,0.0204 13,0.0193 14,0.0204 15,0.0205 16,0.0237 17,0.0202 18,0.0199 19,0.0186
20,0.0175 21,0.0207 22,0.0182 23,0.0211 24,0.0184 25,0.0180 26,0.0168 27,0.0191 28,0.0167
29,0.0195 30,0.0207
```
Final ≈0.017–0.021.

### 4c. Sample quality progression (from the grids)
- epoch 10 (loss ~0.040): recognizable butterfly *silhouettes*, but muddy, desaturated, grainy.
- epoch 30 (loss ~0.026): clean bilaterally-symmetric wings, colors separating, white bg emerging.
- epoch 50 / final (loss 0.018): vivid, clearly butterfly-shaped — symmetric wings, visible
  body, distinct greens/blues/oranges/reds/yellows on clean white background.

### 4d. Resource use & timing (64px)
- Single-forward peak VRAM (batch 16, 64px): **332 MiB**. Total GPU mem in use during training
  ≈ **2.5 GB** (incl. Windows desktop) — comfortably inside 6 GB, no OOM.
- **≈23 s/epoch** (≈19.5 min for 50 epochs) + **≈70 s per 16-image sample grid** (6 grids ≈7 min).

### 4e. 128px resource/timing
- 128px train step peaks at **5498 MiB** at batch 16 (OOM, ~all 6 GB) → dropped to batch 8,
  which peaks at **2935 MiB**. ~4 min/epoch → ~2 h for 30 epochs.
- 128px full-DDPM samples are the sharpest of the whole report (better wing venation + symmetry
  than 64px). Side-by-side: `resolution_compare.png`.

## 5. Part G experiment — DDIM fast sampling: naive swap FAILS, and why

DDPM sampling = 1000 sequential UNet passes: measured **61.6 s** for a 16-image grid at 64px.
DDIM (Song et al.) subsamples that chain. Textbook advice: "swap DDPMScheduler→DDIMScheduler
for ~10–20× speedup." Tried it — it broke. 4-arm diagnostic sweep (`diag_ddim.py`):

| Sampler | 64px time | 128px result |
|---|---|---|
| DDPM, 1000 steps (stochastic) | 61.6 s | clean butterflies (reference) |
| DDIM, 100 steps, **eta=0** (default, deterministic) | 6.7 s | **gray mush — total collapse** |
| DDIM, 250 steps, eta=0 | — | still gray mush (more steps ≠ fix) |
| DDIM, 100 steps, **eta=1.0** (stochastic) | 6.7 s | **clean butterflies — recovered** ✅ |

**Diagnosis:** changing timestep spacing (leading→trailing) did nothing; raising steps
100→250 did nothing; flipping **eta 0→1 fixed it completely.** So the variable is *determinism*.
Deterministic DDIM (eta=0) follows a fixed ODE trajectory with no per-step noise; this small,
lightly-trained UNet's ε-prediction errors compound with nothing to correct them → path drifts
off the data manifold to a gray attractor (worse at 128px, where the model is relatively less
trained). Stochastic sampling (eta=1.0, or full DDPM) re-injects noise each step, staying on
the manifold. **Payoff: eta=1.0 DDIM = DDPM-quality at ~9× speed (6.7 s vs 61.6 s at 64px).**
Committed `ddim_sample.py` now defaults to eta=1.0 with a comment on the trap.
Tie-in: in Part F "swap to DDIM" was listed as an easy win; running it showed the *naive* form
is wrong for a small model — "standard trick" ≠ "free lunch."

## 6. Failures / gotchas actually hit (and fixes)

1. Committed `setup_local.sh` is WSL2+miniconda+wget; this machine is native Windows, no conda.
   Fix: same stack (torch cu124 + diffusers/datasets/accelerate) via `venv`+pip. Model code
   untouched; only packaging differs.
2. Only ~1.5 GB VRAM free at first (Brave+Epic+Fortnite held ~4.5 GB of 6 GB) → would've forced
   32px. Freed to ~5 GB before the run → ran 64px. Fallback if it recurs: BATCH_SIZE=8 IMAGE_SIZE=32.
3. `expandable_segments:True` not supported on Windows — torch warns and ignores. Harmless.
4. `DDPMPipeline` floods stdout with a 1000-step tqdm bar per grid → piped to a log and grepped
   epoch/loss/sample lines instead of editing the library.
5. No CUDA OOM at 64px/batch-16 (peak 332 MiB fwd) — contra initial worry.
6. `num_workers=0` in DataLoader avoids the multi-worker deadlock (seen under WSL last week);
   also the safe choice on Windows.

## 7. Self-critique points already worked out (reviewer will ask these)

- "64px/17M/50ep is a toy, not publication quality." → Correct & intended; smoke test to prove
  the pipeline end-to-end on a laptop. Samples honestly labeled (seed-0, all 16, no cherry-pick).
- "Loss ≠ sample quality, no FID/IS." → True; MSE noise-loss is a proxy; lean on visual
  evolution grid. **FID is the top gap — see §9, now computable via `fid.py`.**
- "One seed, one dataset, no error bars." → Agreed; no generality claim. Depth-over-breadth is
  deliberate (Taimeng) — one dataset deep (loss + 5-pt evolution + resource profile + DDIM
  study) vs many shallow. Breadth left as a `DATASET=` knob.
- "White backgrounds — mode collapse?" → No; wing shapes/colors vary widely across the 16.
  White bg is a dataset bias (Smithsonian specimen cards on white), faithfully learned.
- "Changed env from committed scripts." → Only packaging (venv vs conda); training script +
  Diffusers lib byte-for-byte original; delta documented.
- "~30 min is slow." → ~2/3 is the six 1000-step grids, not training. DDIM eta=1 cuts sampling
  ~9× (§5).

## 8. Figures to ATTACH to the Claude chat (image uploads) + captions

| File | Caption / what it shows |
|---|---|
| `ddpm-out/evolution.png` | sample grids stitched across epochs 10→50 — noise→butterflies |
| `ddpm-out/samples_epoch10.png` | epoch 10: muddy silhouettes |
| `ddpm-out/samples_epoch30.png` | epoch 30: clean symmetric wings, color separating |
| `ddpm-out/samples_final.png` | epoch 50: vivid, clearly butterfly-shaped |
| `ddpm128/samples_final.png` | 128px full-DDPM: sharpest samples of the report |
| `resolution_compare.png` | 64px vs 128px side by side |
| `ddpm128/sampler_compare.png` | DDPM vs DDIM eta=0 (mush) vs eta=1 (clean) |
| `ddpm-out/samples_ddim100_eta1.png` | 64px DDIM-100 eta=1: DDPM quality at ~9× speed |

(Loss-curve plots: the writer can render from the CSVs in §4, or you can plot `loss_log.csv`.)

## 9. FID — the one missing number (compute it, then paste in)

FID = Frechet Inception Distance: run generated + real images through Inception-v3, model each
set's features as a Gaussian, measure the distance between them. **Lower = closer to real**;
penalizes both bad quality and low diversity. Indicative only with ~1000 reals — compare FIDs
computed the SAME way.

Run on your desktop (GPU + cached dataset), then paste the two numbers:
```
python -m pip install torchmetrics            # one-time
python fid.py --model ddpm-out --n_samples 500 --image_size 64     # 64px model
python fid.py --model ddpm128  --n_samples 500 --image_size 128    # 128px model
```
- FID(64px)  = __________   (fill)
- FID(128px) = __________   (fill)
- Expected/hypothesis: 128px FID lower than 64px would quantitatively back the "128px is
  sharper" claim in §4e/Part G.1. If it's NOT lower, that's an honest, reportable surprise.

## 10. Reading list

- Ho, Jain, Abbeel — DDPM, 2020, arXiv:2006.11239
- Song, Meng, Ermon — DDIM, 2020, arXiv:2010.02502
- Diffusers "Train a diffusion model": https://huggingface.co/docs/diffusers/tutorials/basic_training
- Diffusers repo: https://github.com/huggingface/diffusers
- (optional) Rombach et al. — Latent/Stable Diffusion, arXiv:2112.10752
