# Report-only montages. No diffusion code touched.
from PIL import Image, ImageDraw
import os
def montage(items, out, cell=300):
    imgs=[(Image.open(p).convert("RGB").resize((cell,cell)),lab) for p,lab in items]
    pad,lh=10,24; W=len(imgs)*cell+(len(imgs)+1)*pad; H=cell+lh+2*pad
    c=Image.new("RGB",(W,H),"white"); d=ImageDraw.Draw(c); x=pad
    for im,lab in imgs:
        c.paste(im,(x,lh+pad)); d.text((x+4,6),lab,fill="black"); x+=cell+pad
    c.save(out); print("saved",out,c.size)

# 1) Sampler comparison @128px: DDPM-1000 vs DDIM-100 eta1 vs DDIM-100 eta0 (deterministic, fails)
montage([
    ("ddpm128/samples_final.png","DDPM 1000 steps  (61s@64px)"),
    ("ddpm128/samples_ddim100_eta1.png","DDIM 100, eta=1  (24s)  GOOD"),
    ("ddpm128/samples_ddim100.png","DDIM 100, eta=0  (deterministic)  FAILS"),
], "ddpm128/sampler_compare.png")

# 2) Resolution comparison: 64px vs 128px (both DDPM full)
montage([
    ("ddpm-out/samples_final.png","64px  (50 epochs)"),
    ("ddpm128/samples_final.png","128px (30 epochs)"),
], "resolution_compare.png")
