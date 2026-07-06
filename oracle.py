#!/usr/bin/env python3
"""
Slapstack Studio oracle — turns pixels or text into Gabor-atom LAYERS.

Two paths, honestly separated:

  fit_image(img)   CPU, minutes.  Bet-5 recon loop (verified math): fit a
                   GaborPacketImage to the image by MSE, export hard-open
                   atoms in the Bet-6 field layout. This is exactly
                   bet6_open._train_recon generalized to any image.

  sds_layer(text)  GPU, minutes.  Bet-5 run_sds loop from scratch on a
                   fresh atom population, Stable Diffusion 2.1 as the
                   score oracle. UNTESTED on GPU in this build environment
                   (no CUDA, no SD weights here) — the loop is a line-for-
                   line adaptation of the Bet-5 SDS loop that WAS verified
                   on GPU, but treat the first run as a smoke test. Known
                   open risk carries over: SD 2.1 mode-seeking
                   oversaturation at high CFG.

Atom layout (FIELDS): [x, y, theta, sigma_u, sigma_v, freq, phase, r, g, b]
with amplitude folded into signed color and xy centered (canonical frame).
"""

import io
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from bet5_gabor_sds import GaborPacketImage

FIELDS = ["x", "y", "theta", "su", "sv", "f", "phase", "r", "g", "b"]


# ---------------------------------------------------------------------------
# shared: model -> Bet-6 atom array  (port of bet6_open.atoms_from_model)
# ---------------------------------------------------------------------------

def atoms_from_model(m):
    keep = np.where(m.gates.hard_open().numpy())[0]
    arr = np.zeros((len(keep), len(FIELDS)))
    arr[:, 0:2] = torch.tanh(m.xy_raw).detach().numpy()[keep]
    arr[:, 0:2] -= arr[:, 0:2].mean(0)
    arr[:, 2] = m.theta.detach().numpy()[keep]
    arr[:, 3] = np.clip(np.exp(m.log_sigma_u.detach().numpy()[keep]), 5e-3, 2.0)
    arr[:, 4] = np.clip(np.exp(m.log_sigma_v.detach().numpy()[keep]), 5e-3, 2.0)
    arr[:, 5] = np.log1p(np.exp(m.freq_raw.detach().numpy()[keep]))
    arr[:, 6] = np.mod(m.phase.detach().numpy()[keep], 2 * math.pi)
    amp = m.amp.detach().numpy()[keep]
    arr[:, 7:10] = m.color.detach().numpy()[keep] * amp[:, None]
    return arr


# ---------------------------------------------------------------------------
# preview renderer — same formula the verified JS core uses
# (pre[c] += color_c * env * carrier ; sigmoid(2*pre))
# ---------------------------------------------------------------------------

def render_atoms(atoms, H=192):
    ys = np.linspace(-1, 1, H)
    X, Y = np.meshgrid(ys, ys)
    pre = np.zeros((3, H, H), np.float32)
    for a in atoms:
        dx, dy = X - a[0], Y - a[1]
        ct, st = math.cos(a[2]), math.sin(a[2])
        u = ct * dx + st * dy
        v = -st * dx + ct * dy
        env = np.exp(-0.5 * ((u / a[3]) ** 2 + (v / a[4]) ** 2))
        car = np.cos(2 * np.pi * a[5] * u + a[6])
        ec = (env * car).astype(np.float32)
        for c in range(3):
            pre[c] += a[7 + c] * ec
    return 1 / (1 + np.exp(-2 * pre))


def preview_png_bytes(atoms, H=192):
    img = (render_atoms(atoms, H).transpose(1, 2, 0) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# path 1: image -> layer  (CPU, the Bet-5 recon loop)
# ---------------------------------------------------------------------------

def fit_image(pil_img, n_atoms=140, iters=400, size=96, seed=0,
              l0_weight=1e-3, gate_warmup=60, progress=None):
    """Fit a Gabor packet layer to an image by MSE. CPU, deterministic."""
    dev = torch.device("cpu")
    torch.manual_seed(seed)
    pil_img = pil_img.convert("RGB")
    tgt = torch.from_numpy(
        np.asarray(pil_img.resize((size, size)), dtype=np.float32).copy()
    ).permute(2, 0, 1) / 255.0

    m = GaborPacketImage(n_atoms, seed=seed)
    # bg_bias is DROPPED by atoms_from_model on export, so it must not be
    # learned: atoms alone must carry the image, background stays sigmoid-
    # neutral. (Letting it train and then dropping it leaves compensating
    # haze smeared across the frame — found by the coverage battery.)
    m.bg_bias.requires_grad_(False)
    m.bg_bias.zero_()
    m.train()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad],
                           lr=1e-2)
    t0 = time.time()
    log = []
    for it in range(iters):
        opt.zero_grad()
        mse = F.mse_loss(m.render(size, size, dev), tgt)
        loss = mse + l0_weight * m.gates.l0().sum() / n_atoms
        loss.backward()
        if it < gate_warmup and m.gates.logits.grad is not None:
            m.gates.logits.grad.zero_()
        opt.step()
        if it % max(1, iters // 10) == 0 or it == iters - 1:
            psnr = -10 * math.log10(max(mse.item(), 1e-12))
            log.append({"it": it, "mse": float(mse.item()), "psnr_db": psnr})
            if progress:
                progress(it / iters, f"fit {it}/{iters}  psnr {psnr:.1f} dB")
    atoms = atoms_from_model(m)
    ledger = {
        "path": "fit_image", "status": "verified-CPU",
        "n_atoms_model": n_atoms, "n_atoms_open": int(len(atoms)),
        "iters": iters, "size": size, "seed": seed,
        "final_psnr_db": log[-1]["psnr_db"], "seconds": round(time.time() - t0, 1),
        "log": log,
    }
    return atoms, ledger


# ---------------------------------------------------------------------------
# path 2: text -> layer  (GPU, the Bet-5 SDS loop from scratch)
# ---------------------------------------------------------------------------

def sds_layer(prompt, negative_prompt="blurry, low quality, deformed",
              n_atoms=192, iters=900, render_size=256, cfg=30.0, seed=0,
              l0_weight=3e-3, gate_warmup=150,
              t_min=0.02, t_max_start=0.98, t_max_end=0.50,
              sd_model="sd2-community/stable-diffusion-2-1-base",
              progress=None):
    """Text -> Gabor layer via score distillation. Line-for-line adaptation
    of the verified Bet-5 SDS loop, run FROM SCRATCH (no init atoms) and
    without camera jitter (a layer is a single canonical view).

    HONESTY: this function has NOT been executed in the build environment
    (no GPU, no SD weights). The first run on a GPU Space is a smoke test."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "text->layer needs a GPU (Stable Diffusion score distillation). "
            "This Space is on CPU hardware: use image->layer instead, or "
            "duplicate the Space onto GPU hardware.")
    from diffusers import StableDiffusionPipeline, DDPMScheduler

    device = torch.device("cuda")
    torch.manual_seed(seed)
    dtype = torch.float16

    pipe = StableDiffusionPipeline.from_pretrained(
        sd_model, torch_dtype=dtype, safety_checker=None,
        requires_safety_checker=False)
    pipe.to(device)
    vae, unet, tok, te = pipe.vae, pipe.unet, pipe.tokenizer, pipe.text_encoder
    for mod in (vae, unet, te):
        mod.requires_grad_(False)
    sched = DDPMScheduler.from_pretrained(sd_model, subfolder="scheduler")
    alphas = sched.alphas_cumprod.to(device)
    T = sched.config.num_train_timesteps

    def embed(text):
        ids = tok(text, padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to(device)
        return te(ids)[0]

    with torch.no_grad():
        emb = torch.cat([embed(negative_prompt), embed(prompt)])

    m = GaborPacketImage(n_atoms, seed=seed)
    m.train().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)

    t0 = time.time()
    log = []
    for it in range(iters):
        opt.zero_grad()
        img = m.render(render_size, render_size, device, chunk=64)
        x = img[None] * 2 - 1
        if render_size != 512:
            x = F.interpolate(x, (512, 512), mode="bilinear",
                              align_corners=False)
        latents = vae.encode(x.to(dtype)).latent_dist.sample() \
            * vae.config.scaling_factor
        latents = latents.float()

        frac = it / max(1, iters - 1)
        t_max = t_max_start + (t_max_end - t_max_start) * frac
        t = torch.randint(int(t_min * T), int(t_max * T), (1,), device=device)
        noise = torch.randn_like(latents)
        noisy = sched.add_noise(latents, noise, t)
        with torch.no_grad():
            eps = unet(torch.cat([noisy] * 2).to(dtype), torch.cat([t] * 2),
                       encoder_hidden_states=emb).sample.float()
            eps_un, eps_tx = eps.chunk(2)
            eps_hat = eps_un + cfg * (eps_tx - eps_un)
        w = (1 - alphas[t]).view(-1, 1, 1, 1)
        grad = (w * (eps_hat - noise)).detach()
        sds_loss = (grad * latents).sum() / latents.numel()
        loss = sds_loss + l0_weight * m.gates.l0().sum() / n_atoms
        loss.backward()
        if it < gate_warmup and m.gates.logits.grad is not None:
            m.gates.logits.grad.zero_()
        torch.nn.utils.clip_grad_norm_(list(m.parameters()), 1.0)
        opt.step()
        if it % max(1, iters // 20) == 0 or it == iters - 1:
            log.append({"it": it, "sds": float(sds_loss.item()),
                        "t_max": t_max})
            if progress:
                progress(it / iters, f"sds {it}/{iters}")

    m.eval().cpu()
    atoms = atoms_from_model(m)
    ledger = {
        "path": "sds_layer", "status": "UNTESTED-GPU (adapted from verified Bet-5 loop)",
        "prompt": prompt, "negative_prompt": negative_prompt,
        "n_atoms_model": n_atoms, "n_atoms_open": int(len(atoms)),
        "iters": iters, "render_size": render_size, "cfg": cfg, "seed": seed,
        "sd_model": sd_model, "seconds": round(time.time() - t0, 1),
        "known_risk": "SD2.1 mode-seeking oversaturation at high CFG (Bet-5 ledger)",
        "log": log,
    }
    return atoms, ledger
