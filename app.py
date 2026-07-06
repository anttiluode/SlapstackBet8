#!/usr/bin/env python3
"""
Slapstack Studio — HuggingFace Space server.

Layout:
  /                 Gradio app: the Studio (iframe) + generation tabs + ledger
  /studio/…         static: the single-file interactive client (verified JS BP)
  /gradio_api/…     Gradio REST API, called by the client JS:
                      layer_from_image_b64(png_b64, n_atoms, iters) -> JSON
                      layer_from_text(prompt, negative, n_atoms, iters, cfg) -> JSON

Division of labor (the whole point):
  the SERVER knows what things look like (SD oracle / image fitting),
  the CLIENT knows what is where and how sure (verified BP in the browser).
"""

import base64
import io
import json
import os

import gradio as gr
import numpy as np
from PIL import Image

from oracle import fit_image, sds_layer, preview_png_bytes

MAX_ATOMS = 256
MAX_ITERS_CPU = 800
MAX_ITERS_GPU = 1500


def _layer_payload(atoms, ledger):
    png = preview_png_bytes(atoms, 192)
    return json.dumps({
        "atoms": np.asarray(atoms).round(5).tolist(),
        "preview_png_b64": base64.b64encode(png).decode(),
        "ledger": ledger,
    })


# ---------------- endpoints (also used by the studio client JS) -------------

def layer_from_image_b64(png_b64: str, n_atoms: float, iters: float) -> str:
    """b64 PNG/JPEG -> Gabor layer JSON. CPU path, verified."""
    raw = base64.b64decode(png_b64.split(",")[-1])
    img = Image.open(io.BytesIO(raw))
    n_atoms = int(min(max(n_atoms, 16), MAX_ATOMS))
    iters = int(min(max(iters, 50), MAX_ITERS_CPU))
    atoms, ledger = fit_image(img, n_atoms=n_atoms, iters=iters)
    return _layer_payload(atoms, ledger)


def layer_from_text(prompt: str, negative: str, n_atoms: float,
                    iters: float, cfg: float) -> str:
    """text -> Gabor layer JSON via SDS. GPU only; honest error on CPU."""
    n_atoms = int(min(max(n_atoms, 32), MAX_ATOMS))
    iters = int(min(max(iters, 100), MAX_ITERS_GPU))
    atoms, ledger = sds_layer(prompt, negative_prompt=negative or
                              "blurry, low quality, deformed",
                              n_atoms=n_atoms, iters=iters, cfg=float(cfg))
    return _layer_payload(atoms, ledger)


# ---------------- human-facing wrappers for the Gradio tabs -----------------

def ui_from_image(img, n_atoms, iters):
    if img is None:
        raise gr.Error("upload an image first")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    out = layer_from_image_b64(base64.b64encode(buf.getvalue()).decode(),
                               n_atoms, iters)
    d = json.loads(out)
    prev = Image.open(io.BytesIO(base64.b64decode(d["preview_png_b64"])))
    led = dict(d["ledger"]); led.pop("log", None)
    return prev, json.dumps(led, indent=2), out


def ui_from_text(prompt, negative, n_atoms, iters, cfg):
    if not (prompt or "").strip():
        raise gr.Error("write a prompt first")
    out = layer_from_text(prompt, negative, n_atoms, iters, cfg)
    d = json.loads(out)
    prev = Image.open(io.BytesIO(base64.b64decode(d["preview_png_b64"])))
    led = dict(d["ledger"]); led.pop("log", None)
    return prev, json.dumps(led, indent=2), out


CSS = """
.studio-frame iframe { width: 100%; height: 860px; border: 0; border-radius: 8px; }
"""

with gr.Blocks(title="Slapstack Studio", css=CSS) as demo:
    gr.Markdown(
        "# Slapstack Studio\n"
        "**Generate Gabor-atom layers with AI, then move them, occlude them, "
        "and watch belief propagation keep track.** Every entity in the "
        "studio is a posterior: layers are recovered from an unlabeled atom "
        "soup by BP, a drag is a pose clamp, occlusion honestly widens the "
        "belief. The interactive engine below is a JS port verified against "
        "the SlapstackBet6 Python to 2e-16 (transform), 8.6e-8 (render MSE), "
        "identical BP accuracy.")
    with gr.Tab("Studio"):
        gr.HTML('<div class="studio-frame">'
                '<iframe src="/studio/studio.html"></iframe></div>')
    with gr.Tab("Layer from image (CPU, verified)"):
        with gr.Row():
            with gr.Column():
                in_img = gr.Image(type="pil", label="image")
                in_na = gr.Slider(32, MAX_ATOMS, 140, step=4, label="atom budget")
                in_it = gr.Slider(100, MAX_ITERS_CPU, 400, step=50, label="fit iterations")
                btn_i = gr.Button("Fit layer", variant="primary")
            with gr.Column():
                out_prev_i = gr.Image(label="layer preview (atoms only)")
                out_led_i = gr.Textbox(label="ledger", lines=8)
                out_json_i = gr.Textbox(label="layer JSON (paste into the Studio)",
                                        lines=4, max_lines=4)
        btn_i.click(ui_from_image, [in_img, in_na, in_it],
                    [out_prev_i, out_led_i, out_json_i])
    with gr.Tab("Layer from text (GPU, untested)"):
        gr.Markdown(
            "Score-distillation of a fresh atom population against Stable "
            "Diffusion 2.1 — a line-for-line adaptation of the Bet-5 SDS "
            "loop that was verified on GPU, but **this exact function has "
            "not been executed yet**; the first run is a smoke test. On CPU "
            "hardware this tab refuses honestly. Known carried-over risk: "
            "SD2.1 mode-seeking oversaturation at high CFG.")
        with gr.Row():
            with gr.Column():
                in_pr = gr.Textbox(label="prompt", placeholder="a red tractor, side view, flat background")
                in_ng = gr.Textbox(label="negative prompt", value="blurry, low quality, deformed")
                in_na2 = gr.Slider(32, MAX_ATOMS, 192, step=4, label="atom budget")
                in_it2 = gr.Slider(100, MAX_ITERS_GPU, 900, step=50, label="SDS iterations")
                in_cfg = gr.Slider(5, 60, 30, step=1, label="CFG")
                btn_t = gr.Button("Distill layer", variant="primary")
            with gr.Column():
                out_prev_t = gr.Image(label="layer preview (atoms only)")
                out_led_t = gr.Textbox(label="ledger", lines=8)
                out_json_t = gr.Textbox(label="layer JSON (paste into the Studio)",
                                        lines=4, max_lines=4)
        btn_t.click(ui_from_text, [in_pr, in_ng, in_na2, in_it2, in_cfg],
                    [out_prev_t, out_led_t, out_json_t])

    # API-only endpoints for the studio client (string in/out, no FileData)
    api_b64_in = gr.Textbox(visible=False)
    api_na = gr.Number(visible=False, value=140)
    api_it = gr.Number(visible=False, value=400)
    api_out = gr.Textbox(visible=False)
    gr.Button(visible=False).click(layer_from_image_b64,
                                   [api_b64_in, api_na, api_it], api_out,
                                   api_name="layer_from_image_b64")
    api_pr = gr.Textbox(visible=False)
    api_ng = gr.Textbox(visible=False)
    api_na2 = gr.Number(visible=False, value=192)
    api_it2 = gr.Number(visible=False, value=900)
    api_cfg = gr.Number(visible=False, value=30)
    api_out2 = gr.Textbox(visible=False)
    gr.Button(visible=False).click(layer_from_text,
                                   [api_pr, api_ng, api_na2, api_it2, api_cfg],
                                   api_out2, api_name="layer_from_text")

# ---------------- FastAPI mount: static studio + gradio ---------------------
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/studio", StaticFiles(directory=os.path.join(
    os.path.dirname(__file__), "studio")), name="studio")
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
