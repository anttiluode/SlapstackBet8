# Slapstack Studio

![pic](pic2.png)

**Spawn Gabor-atom layers with AI, move them, occlude them, and watch belief propagation keep track.**

Slapstack Studio is a neuro-symbolic scene editor. It splits the generative AI problem into two distinct halves:

1. **The Semantic Oracle (Server):** A neural network (Stable Diffusion / SDS) generates discrete, semantic visual layers as lists of Gabor atoms.
2. **The Geometric Physics Engine (Browser):** A probabilistic factor graph running in the browser tracks these atoms using Belief Propagation (BP), maintaining persistent identities, calculable uncertainties, and physical boundaries that can be dragged and occluded.

Every entity on the canvas is a posterior. Layers are recovered from an unlabeled atom soup by BP. A drag is a pose clamp. Occlusion honestly widens the belief state.

## Features

- **AI Layer Spawning:** Distill text prompts into Gabor layers using Stable Diffusion Score Distillation Sampling (SDS) on the GPU, featuring cross-attention matting to create compact, cleanly cut-out objects. Alternatively, fit images on the CPU via an Adam-driven MSE loop.
- **Dynamic K-Binding:** Drop new layers into the arena dynamically. The BP engine scales to track any number of objects simultaneously.
- **Depth-Ordered Auto-Occlusion:** Move a layer behind another, and its atoms lose evidence honestly based on the envelope mass of the foreground object. The marginals revert toward the prior, and coverage releases when the object re-emerges.
- **Semantic Hint Brush:** Shift-click the canvas to drop a Gaussian attention prior for a selected layer. This forces the rigid geometric atoms to snap to your semantic map.
- **Painter vs. Field Compositing:** Toggle between the mathematical additive field model (sigmoid) and a painterly depth-sorted alpha-blended compositor.
- **Exact Sim(2) Interactions:** Drag to move, use the mouse wheel to rotate, and `Shift` + wheel to scale. All interactions apply exact group algebra to the pose posterior.
- **Prompt as Intervention:** Type commands like *"move the boat left, put the star behind the tractor"* to issue structural pose clamps via local grammar (or Claude).

## Architecture

The architecture intentionally decouples the heavy pixel-generation weights from the interactive spatial logic:

- `app.py` & `oracle.py`: A FastAPI/Gradio backend handling the SDS cross-attention extraction (GPU) and image-fitting (CPU).
- `studio.html`: A self-contained, 65+ KB static client that runs the Loopy BP math (`core.js`) and depth compositing (`studio_core.js`) entirely in the browser.

## Local Setup & Build Instructions

To run Slapstack Studio locally, you need Python 3.x and Node.js.

### 1. Structure the Directories

Create the `studio` directory and move the core math files inside so the build script can find them:

```bash
mkdir studio
mv core.js studio/
mv studio_core.js studio/
```

### 2. Bake the Built-in Library

Generate the mathematically neutral, compact starting templates (Tractor, Star, Boat). This runs an MSE fit on the CPU and saves them to `studio/builtins.json`:

```bash
python make_builtins.py
```

### 3. Compile the Web Client

Bundle the UI, the core math, the studio math, and the JSON built-ins into a single, deployable HTML client:

```bash
node build.js
```

*This will output `studio/studio.html` (approx. 94 KB).*

### 4. Boot the Server

Launch the FastAPI/Gradio backend:

```bash
python app.py
```

Access the application at http://127.0.0.1:7860 or http://0.0.0.0:7860.

## Hugging Face Spaces Deployment

This repository is ready to be deployed as a Hugging Face Space.

- To utilize the **Text → Layer (SDS)** spawning, the Space must be configured with **GPU hardware**.
- If deployed on a standard CPU Space, the GPU tab will honestly refuse execution, but the **Image → Layer (MSE Fit)** and the interactive studio canvas will function perfectly.

## Honest Ledger

- **Verified in this build:** The BP core matches the original Python math to float precision (transform exact to 2e-16, render MSE 8.6e-8, identical binding). Drag = pose clamp + re-settle. Identity persists through occlusion. Depth occlusion removes evidence honestly (marginals revert toward the prior, coverage releases when moved away). The semantic likelihood factor resolves the identical-twins ambiguity case (binding accuracy jumps from 0.05 on geometry alone to 0.99 with location priors). Image-to-layer fitting runs end-to-end on CPU.
- **Not verified:** Text → Layer (SDS) and cross-attention matte are line-for-line adaptations of verified GPU loops but have not been executed on a GPU in this specific pipeline—the first run on a GPU Space is a smoke test. SD attention maps flicker across seeds, which is why they enter BP as a *prior* with a weight, never as truth. Real-photo scene encoding remains open. Painter compositing and the alpha matte are presentation choices, not the base additive field model.
