#!/usr/bin/env python
"""
Train a small unconditional DDPM with HuggingFace Diffusers — the beginner
"Train a diffusion model" tutorial, shrunk to run on a 4 GB laptop GPU (RTX 3050) in
minutes, and to run offline on Adroit once the dataset is pre-cached.

Grounded in the real Diffusers API: UNet2DModel + DDPMScheduler + DDPMPipeline.
Plain PyTorch loop (no `accelerate` config prompt) so a beginner can just run it.

Examples
--------
    # local PC, tiny/fast smoke run (downloads butterflies from HF):
    python train_ddpm.py --image_size 32 --epochs 10 --batch_size 16

    # Adroit compute node (dataset already cached under $HF_HOME on scratch):
    HF_HUB_OFFLINE=1 python train_ddpm.py --image_size 64 --epochs 50 --batch_size 32
"""
import argparse, os, math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.environ.get("DATASET", "huggan/smithsonian_butterflies_subset"),
                   help="HF dataset id (butterflies subset is the tutorial default).")
    p.add_argument("--image_size", type=int, default=int(os.environ.get("IMAGE_SIZE", 32)),
                   help="Square resolution. 32 fits 4GB easily; 64 needs the MIG slice.")
    p.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", 16)))
    p.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 10)))
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_train_timesteps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=os.environ.get("OUT_DIR", "ddpm-out"))
    p.add_argument("--sample_every", type=int, default=0,
                   help="Also save a sample grid every N epochs (0 = only at the end).")
    p.add_argument("--max_images", type=int, default=0,
                   help="Cap dataset size for a faster smoke run (0 = use all).")
    return p.parse_args()


def build_model(image_size, num_train_timesteps):
    from diffusers import UNet2DModel, DDPMScheduler
    # A deliberately small UNet so it fits 4 GB. The tutorial uses a much larger one
    # (block_out_channels up to 512) at 128px; we shrink both to keep it beginner-fast.
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    return model, scheduler


def make_loader(args):
    from datasets import load_dataset
    from torchvision import transforms
    ds = load_dataset(args.dataset, split="train")
    if args.max_images and args.max_images < len(ds):
        ds = ds.select(range(args.max_images))
    # HF image datasets expose an "image" column (PIL). Butterflies/flowers/pokemon all do.
    img_col = "image" if "image" in ds.column_names else ds.column_names[0]
    tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),   # -> [-1, 1]
    ])

    def collate(batch):
        imgs = [tfm(rec[img_col].convert("RGB")) for rec in batch]
        return torch.stack(imgs)

    # num_workers=0: multi-worker loaders deadlock under WSL (learned the hard way last week).
    return DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)


@torch.no_grad()
def save_samples(model, scheduler, args, device, tag):
    from diffusers import DDPMPipeline
    from torchvision.utils import make_grid, save_image
    model.eval()
    pipe = DDPMPipeline(unet=model, scheduler=scheduler).to(device)
    g = torch.Generator(device=device).manual_seed(args.seed)
    out = pipe(batch_size=16, generator=g, output_type="np")
    imgs = torch.from_numpy(out.images).permute(0, 3, 1, 2)  # (N,H,W,C)->(N,C,H,W), in [0,1]
    grid = make_grid(imgs, nrow=4)
    path = os.path.join(args.out_dir, f"samples_{tag}.png")
    save_image(grid, path)
    model.train()
    print(f"  saved sample grid -> {path}", flush=True)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | dataset={args.dataset} | image_size={args.image_size} "
          f"| batch={args.batch_size} | epochs={args.epochs}", flush=True)

    loader = make_loader(args)
    model, scheduler = build_model(args.image_size, args.num_train_timesteps)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps_per_epoch = len(loader)
    print(f"{len(loader.dataset)} images | {steps_per_epoch} steps/epoch", flush=True)

    for epoch in range(args.epochs):
        running = 0.0
        for i, clean in enumerate(loader):
            clean = clean.to(device)
            noise = torch.randn_like(clean)
            bs = clean.shape[0]
            t = torch.randint(0, scheduler.config.num_train_timesteps, (bs,), device=device).long()
            noisy = scheduler.add_noise(clean, noise, t)
            noise_pred = model(noisy, t, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            opt.step()
            opt.zero_grad()
            running += loss.item()
        avg = running / max(1, steps_per_epoch)
        print(f"epoch {epoch+1}/{args.epochs}  loss={avg:.4f}", flush=True)
        if args.sample_every and (epoch + 1) % args.sample_every == 0:
            save_samples(model, scheduler, args, device, f"epoch{epoch+1}")

    # Save the trained pipeline + a final sample grid (the deliverable).
    from diffusers import DDPMPipeline
    DDPMPipeline(unet=model, scheduler=scheduler).save_pretrained(args.out_dir)
    save_samples(model, scheduler, args, device, "final")
    print(f"DONE. Pipeline + samples in {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
