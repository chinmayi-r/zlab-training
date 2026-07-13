#!/usr/bin/env python
"""
Generate images from a diffusion pipeline — either your trained DDPM (from train_ddpm.py's
--out_dir) or a pretrained pipeline off the Hub. Near-zero compute; good for the report's
"how inference works" section.

    # sample from your own trained model:
    python sample.py --model ddpm-out --n 16 --out my_samples.png

    # sample from a pretrained DDPM on the Hub (downloads weights):
    python sample.py --model google/ddpm-cifar10-32 --n 16 --out cifar_samples.png
"""
import argparse, torch
from diffusers import DiffusionPipeline
from torchvision.utils import make_grid, save_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local pipeline dir (train_ddpm out_dir) or a Hub id")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="samples.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(args.model).to(device)
    g = torch.Generator(device=device).manual_seed(args.seed)
    out = pipe(batch_size=args.n, generator=g, output_type="np")
    imgs = torch.from_numpy(out.images).permute(0, 3, 1, 2)
    save_image(make_grid(imgs, nrow=int(args.n ** 0.5) or 4), args.out)
    print(f"saved {args.n} samples -> {args.out}")


if __name__ == "__main__":
    main()
