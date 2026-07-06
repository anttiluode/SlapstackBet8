/* Slapstack Studio layer math — NEW mechanics on top of the verified core.
   core.js is untouched; everything here is presentation-adjacent inference
   plumbing, tested by tests_studio.js:
     renderEnv      per-layer envelope-energy buffer (for alpha + coverage)
     coverageOf     how much front-stack mass sits on each atom center
     autoHidden     coverage -> per-atom evidence mask (depth occlusion)
     compositePainter  back-to-front alpha compositing of layer buffers
     composePose    Sim(2) composition in vote coordinates [tx,ty,rho,lam]
*/
"use strict";

const ALPHA_C = 2.0;        // alpha = 1 - exp(-ALPHA_C * bounded envelope energy)
const COVER_THRESH = 0.55;  // atom loses evidence when this covered

/* Opacity of a layer = where it actually PAINTS: per-pixel deviation of
   the pre-sigmoid field from neutral. Cancelling atom pairs (high energy,
   zero sum) are correctly transparent — envelope-based opacity is not used
   because it saturates on exactly those invisible pairs.
   a = 1 - exp(-ALPHA_C * max(0, |preR|+|preG|+|preB| - FLOOR)). */
const ALPHA_FLOOR = 0.08;
function alphaFromPre(pre, H, out) {
  out = out || new Float32Array(H * H);
  const n = H * H;
  for (let i = 0; i < n; i++) {
    const m = Math.abs(pre[i]) + Math.abs(pre[n + i]) + Math.abs(pre[2 * n + i]);
    out[i] = m > ALPHA_FLOOR ? 1 - Math.exp(-ALPHA_C * (m - ALPHA_FLOOR)) : 0;
  }
  return out;
}

/* coverage of each atom center by a front stack of alpha buffers:
   cov_i = 1 - prod_front (1 - a_m(x_i)). */
function coverageOf(atoms, frontAlphaBufs, H) {
  const out = new Float32Array(atoms.length);
  for (let i = 0; i < atoms.length; i++) {
    const px = Math.max(0, Math.min(H - 1, Math.round((atoms[i][0] + 1) / 2 * (H - 1))));
    const py = Math.max(0, Math.min(H - 1, Math.round((atoms[i][1] + 1) / 2 * (H - 1))));
    let keep = 1;
    for (const buf of frontAlphaBufs) keep *= 1 - buf[py * H + px];
    out[i] = 1 - keep;
  }
  return out;
}

/* Per-atom evidence mask from the depth stack.
   alphaBufs aligned with layers. An atom of layer k is evidence-free if the
   strictly-in-front stack covers it beyond COVER_THRESH, or its layer is
   user-hidden. */
function autoHidden(obs, layerOf, layers, alphaBufs, H) {
  const mask = new Array(obs.length).fill(false);
  const covFrac = layers.map(() => 0);
  const counts = layers.map(() => 0);
  for (let i = 0; i < obs.length; i++) {
    const k = layerOf[i];
    if (k < 0) continue;                       // clutter: always evidenced
    counts[k]++;
    if (layers[k].hidden) { mask[i] = true; covFrac[k]++; continue; }
    const front = [];
    for (let m = 0; m < layers.length; m++)
      if (m !== k && !layers[m].hidden && layers[m].depth > layers[k].depth)
        front.push(alphaBufs[m]);
    if (!front.length) continue;
    const cov = coverageOf([obs[i]], front, H)[0];
    if (cov > COVER_THRESH) { mask[i] = true; covFrac[k]++; }
  }
  for (let k = 0; k < layers.length; k++)
    covFrac[k] = counts[k] ? covFrac[k] / counts[k] : 0;
  return { mask, covFrac };
}

/* Painter compositing: layers back-to-front by depth.
   rgbBufs[k]: Float32Array(3*H*H) pre-sigmoid; alphaBufs[k]: from
   alphaFromPre. Base is mid-gray (sigmoid(0)). */
function compositePainter(order, rgbBufs, alphaBufs, H, out) {
  out = out || new Uint8ClampedArray(4 * H * H);
  const n = H * H;
  const acc = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) { acc[i] = 127.5; acc[n + i] = 127.5; acc[2 * n + i] = 127.5; }
  for (const k of order) {
    const pre = rgbBufs[k], al = alphaBufs[k];
    for (let i = 0; i < n; i++) {
      const a = al[i];
      if (a < 1e-3) continue;
      acc[i]         += a * (255 / (1 + Math.exp(-2 * pre[i]))         - acc[i]);
      acc[n + i]     += a * (255 / (1 + Math.exp(-2 * pre[n + i]))     - acc[n + i]);
      acc[2 * n + i] += a * (255 / (1 + Math.exp(-2 * pre[2 * n + i])) - acc[2 * n + i]);
    }
  }
  for (let i = 0; i < n; i++) {
    out[4 * i] = acc[i]; out[4 * i + 1] = acc[n + i];
    out[4 * i + 2] = acc[2 * n + i]; out[4 * i + 3] = 255;
  }
  return out;
}

/* Backdrop trim: whole-image fits carry large-sigma background pads that
   cover the full frame; for OBJECT layers, drop atoms whose envelope is
   wider than sigMax. HEURISTIC, honestly: it also kills any genuinely
   large object parts. The untrimmed layer is kept in the library. */
function trimAtoms(atoms, sigMax = 0.35) {
  const kept = atoms.filter(a => Math.max(a[3], a[4]) <= sigMax);
  if (!kept.length) return atoms.slice();
  // re-center xy so the trimmed set is a canonical template again
  let mx = 0, my = 0;
  for (const a of kept) { mx += a[0]; my += a[1]; }
  mx /= kept.length; my /= kept.length;
  return kept.map(a => { const b = a.slice(); b[0] -= mx; b[1] -= my; return b; });
}

/* Sim(2) composition in vote coordinates: (g2 ∘ g1). */
function composePose(g2, g1) {
  const s2 = Math.exp(g2[3]);
  const c = Math.cos(g2[2]), s = Math.sin(g2[2]);
  const wrap = d => ((d + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  return [s2 * (c * g1[0] - s * g1[1]) + g2[0],
          s2 * (s * g1[0] + c * g1[1]) + g2[1],
          wrap(g1[2] + g2[2]), g1[3] + g2[3]];
}
function gestureTranslate(dx, dy) { return [dx, dy, 0, 0]; }
function gestureRotateAbout(c, drho) {
  const co = Math.cos(drho), si = Math.sin(drho);
  return [c[0] - (co * c[0] - si * c[1]), c[1] - (si * c[0] + co * c[1]), drho, 0];
}
function gestureScaleAbout(c, ds) {
  const s = Math.exp(ds);
  return [c[0] - s * c[0], c[1] - s * c[1], 0, ds];
}

if (typeof module !== "undefined") {
  module.exports = {
    ALPHA_C, COVER_THRESH, alphaFromPre, coverageOf, autoHidden,
    compositePainter, composePose, gestureTranslate, gestureRotateAbout,
    gestureScaleAbout, trimAtoms,
  };
}
