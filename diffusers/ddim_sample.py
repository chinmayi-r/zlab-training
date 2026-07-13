#!/usr/bin/env python
"""
Fast sampling from a trained DDPM using the DDIM sampler (Song et al. 2020).

A model trained with the DDPM epsilon-objective can be sampled by *any* compatible
scheduler. DDPMScheduler runs the full reverse chain (num_train_timesteps=1000 model
evals). DDIMScheduler is deterministic and skips steps, so ~20-50 evals give similar
quality — a ~20-50x sampling speedup. This is pure Diffusers API; no model code changed.

    python ddim_sample.py --model ddpm-out --steps 50 --out ddpm-out/samples_ddim50.png
"""
import argparse, time, torch
from diffusers import UNet2DModel, DDIMScheduler, DDIMPipeline
from torchvision.utils import make_grid, save_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="trained pipeline dir (expects <model>/unet)")
    ap.add_argument("--steps", type=int, default=100, help="DDIM inference steps (<< 1000)")
    ap.add_argument("--eta", type=float, default=1.0,
                    help="0=deterministic DDIM, 1=stochastic (~DDPM). NOTE: eta=0 collapses to "
                         "gray mush on this small/undertrained model — eta=1.0 recovers quality.")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="samples_ddim.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet = UNet2DModel.from_pretrained(f"{args.model}/unet").to(device)
    # DDIMScheduler defaults (linear betas, epsilon prediction) match DDPMScheduler's,
    # so it's drop-in compatible with a DDPM-trained UNet.
    pipe = DDIMPipeline(unet=unet, scheduler=DDIMScheduler(num_train_timesteps=1000)).to(device)
    pipe.set_progress_bar_config(disable=True)

    g = torch.Generator(device=device).manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    out = pipe(batch_size=args.n, generator=g, num_inference_steps=args.steps,
               eta=args.eta, output_type="np")
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    imgs = torch.from_numpy(out.images).permute(0, 3, 1, 2)
    save_image(make_grid(imgs, nrow=int(args.n ** 0.5) or 4), args.out)
    print(f"DDIM {args.steps} steps: {dt:.1f}s for {args.n} imgs -> {args.out}")


if __name__ == "__main__":
    main()
