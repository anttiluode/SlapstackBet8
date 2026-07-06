#!/usr/bin/env python3
"""
Build the studio's built-in layers: the Bet-6 tractor and star, plus a boat,
drawn on NEUTRAL MID-GRAY (127,127,127). sigmoid(0) = 0.5, so a mid-gray
background needs NO atoms — the fit spends its whole budget on the object
and the layer comes out spatially compact by construction. This is the
object-layer recipe the Space UI also recommends for uploaded images.
"""

import json
import math

import numpy as np
from PIL import Image, ImageDraw

from oracle import fit_image, render_atoms

GRAY = (127, 127, 127)


def draw_tractor(px=96):
    # fills ~90% of the frame: template extent IS the object extent
    im = Image.new("RGB", (px, px), GRAY)
    d = ImageDraw.Draw(im)
    d.rectangle([10, 34, 74, 72], fill=(200, 40, 30))
    d.rectangle([42, 8, 74, 34], fill=(180, 30, 25))
    d.ellipse([4, 58, 42, 92], fill=(20, 20, 20))
    d.ellipse([57, 66, 82, 90], fill=(25, 25, 25))
    return im


def draw_star(px=96):
    im = Image.new("RGB", (px, px), GRAY)
    d = ImageDraw.Draw(im)
    c, r1, r2 = px / 2, px * 0.47, px * 0.19
    pts = []
    for i in range(10):
        r = r1 if i % 2 == 0 else r2
        a = math.pi / 2 + i * math.pi / 5
        pts.append((c + r * math.cos(a), c - r * math.sin(a)))
    d.polygon(pts, fill=(240, 200, 60))
    d.ellipse([c - 9, c - 9, c + 9, c + 9], fill=(200, 90, 30))
    return im


def draw_boat(px=96):
    im = Image.new("RGB", (px, px), GRAY)
    d = ImageDraw.Draw(im)
    d.polygon([(6, 64), (90, 64), (74, 88), (20, 88)], fill=(160, 80, 40))
    d.rectangle([45, 6, 50, 64], fill=(230, 220, 200))
    d.polygon([(50, 8), (86, 56), (50, 56)], fill=(250, 245, 225))
    return im


if __name__ == "__main__":
    out, stats, panels = {}, {}, []
    for name, img, seed in [("tractor", draw_tractor(), 0),
                            ("star", draw_star(), 1),
                            ("boat", draw_boat(), 3)]:
        atoms, ledger = fit_image(img, n_atoms=140, iters=400, seed=seed)
        r = np.hypot(atoms[:, 0], atoms[:, 1])
        stats[name] = {"atoms": len(atoms),
                       "psnr_db": round(ledger["final_psnr_db"], 1),
                       "radius_p95": round(float(np.percentile(r, 95)), 2)}
        out[name] = np.asarray(atoms).round(5).tolist()
        panels.append((render_atoms(np.asarray(atoms), 160)
                       .transpose(1, 2, 0) * 255).astype(np.uint8))
        print(name, stats[name])
    json.dump(out, open("studio/builtins.json", "w"))
    Image.fromarray(np.concatenate(panels, 1)).save("builtins_preview.png")
    print("wrote studio/builtins.json")
