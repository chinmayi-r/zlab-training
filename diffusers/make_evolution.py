# Report helper only: stitches the saved sample grids into one labeled evolution strip.
# Does NOT touch any diffusion/model code.
from PIL import Image, ImageDraw
import os
tags = [("epoch10","epoch 10"),("epoch20","epoch 20"),("epoch30","epoch 30"),
        ("epoch40","epoch 40"),("epoch50","epoch 50 (final)")]
d = "ddpm-out"
imgs = [(Image.open(os.path.join(d,f"samples_{t}.png")).convert("RGB"), lab) for t,lab in tags]
w,h = imgs[0][0].size
pad, lab_h = 8, 22
strip = Image.new("RGB", (len(imgs)*w + (len(imgs)+1)*pad, h + lab_h + 2*pad), "white")
dr = ImageDraw.Draw(strip)
x = pad
for im,lab in imgs:
    strip.paste(im, (x, lab_h+pad))
    dr.text((x+4, 4), lab, fill="black")
    x += w + pad
strip.save(os.path.join(d,"evolution.png"))
print("saved", os.path.join(d,"evolution.png"), strip.size)
