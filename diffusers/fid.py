#!/usr/bin/env python
"""
Compute FID (Frechet Inception Distance) between generated samples and the real training
butterflies. Lower FID = generated distribution is closer to the real one (captures both
image quality AND diversity, so a mode-collapsed model scores badly).

Run on your desktop (needs the GPU + the cached HF dataset). One command:

    python fid.py --model ddpm-out --n_samples 500 --image_size 64
    python fid.py --model ddpm128 --n_samples 500 --image_size 128

Notes / caveats to put in the report:
  * FID is noisy with few images. The butterflies set is only 1000 reals, and generating
    thousands of samples at 1000 DDPM steps is slow, so this uses DDIM eta=1.0 (your Part G
    finding: fast AND clean) and a few hundred samples. Treat the number as INDICATIVE and
    only compare FIDs computed the SAME way (same n_samples, same sampler).
  * Two numbers are worth reporting: the 64px model vs the 128px model (does higher-res
    training actually lower FID, backing the qualitative "sharper" claim in Part G.1?).
"""
import argparse, os, torch
from diffusers import DiffusionPipeline, DDIMScheduler


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="trained pipeline dir (e.g. ddpm-out or ddpm128)")
    ap.add_argument("--dataset", default=os.environ.get("DATASET", "huggan/smithsonian_butterflies_subset"))
    ap.add_argument("--n_samples", type=int, default=500, help="generated images to score")
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--ddim_steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from torchmetrics.image.fid import FrechetInceptionDistance
    from datasets import load_dataset
    from torchvision import transforms

    # --- real images -> uint8 tensors at the model's resolution ---
    ds = load_dataset(args.dataset, split="train")
    col = "image" if "image" in ds.column_names else ds.column_names[0]
    to_uint8 = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.PILToTensor(),           # uint8 [0,255], (C,H,W)
    ])
    reals = torch.stack([to_uint8(r[col].convert("RGB")) for r in ds]).to(device)

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid.update(reals, real=True)

    # --- generated images via DDIM eta=1.0 (fast + clean, per Part G) ---
    pipe = DiffusionPipeline.from_pretrained(args.model).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    g = torch.Generator(device=device).manual_seed(args.seed)
    made = 0
    while made < args.n_samples:
        b = min(args.batch, args.n_samples - made)
        out = pipe(batch_size=b, num_inference_steps=args.ddim_steps, eta=1.0,
                   generator=g, output_type="np")
        imgs = torch.from_numpy(out.images).permute(0, 3, 1, 2)      # (N,C,H,W) in [0,1]
        imgs = (imgs * 255).clamp(0, 255).to(torch.uint8).to(device)
        fid.update(imgs, real=False)
        made += b
        print(f"  generated {made}/{args.n_samples}", flush=True)

    score = fid.compute().item()
    print(f"\nFID ({args.model}, {args.image_size}px, {args.n_samples} samples, "
          f"DDIM-{args.ddim_steps} eta=1) = {score:.2f}   (lower is better)")


if __name__ == "__main__":
    main()
