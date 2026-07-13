#!/usr/bin/env python
"""Diagnose why naive DDIM sampling degrades. Tries variants on the 128px model."""
import sys, time, torch
from diffusers import UNet2DModel, DDIMScheduler, DDIMPipeline
from torchvision.utils import make_grid, save_image

model = sys.argv[1] if len(sys.argv) > 1 else "ddpm128"
device = "cuda"
unet = UNet2DModel.from_pretrained(f"{model}/unet").to(device)

def run(tag, sched, steps, eta):
    pipe = DDIMPipeline(unet=unet, scheduler=sched).to(device)
    pipe.set_progress_bar_config(disable=True)
    g = torch.Generator(device=device).manual_seed(0)
    torch.cuda.synchronize(); t0 = time.time()
    out = pipe(batch_size=16, generator=g, num_inference_steps=steps, eta=eta, output_type="np")
    torch.cuda.synchronize(); dt = time.time() - t0
    imgs = torch.from_numpy(out.images).permute(0, 3, 1, 2)
    p = f"{model}/diag_{tag}.png"
    save_image(make_grid(imgs, nrow=4), p)
    print(f"{tag}: {dt:.1f}s -> {p}", flush=True)

# A: library-default DDIM (leading spacing, eta 0) — the naive path that failed
run("A_default100", DDIMScheduler(num_train_timesteps=1000), 100, 0.0)
# B: trailing timestep spacing (the standard low-step-quality fix)
run("B_trailing100", DDIMScheduler(num_train_timesteps=1000, timestep_spacing="trailing"), 100, 0.0)
# C: stochastic DDIM (eta=1 ~ closer to DDPM), trailing
run("C_trailing_eta1", DDIMScheduler(num_train_timesteps=1000, timestep_spacing="trailing"), 100, 1.0)
# D: more steps, trailing
run("D_trailing250", DDIMScheduler(num_train_timesteps=1000, timestep_spacing="trailing"), 250, 0.0)
