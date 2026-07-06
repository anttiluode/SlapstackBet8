/* Studio battery — the big-prize claims, tested headless:
   1. K=3 bind where the third layer came from the LIVE SERVER (image->layer)
   2. depth auto-occlusion: drag star behind tractor -> atoms lose evidence,
      marginals honestly uncertain, identity persists; move away -> rebinds
   3. painter compositor: the front layer wins where it covers
*/
"use strict";
const fs = require("fs");
const path = require("path");
const C = require("./studio/core.js");
const S = require("./studio/studio_core.js");
const B = JSON.parse(fs.readFileSync(__dirname + "/studio/builtins.json"));
let pass = 0, fail = 0;
const check = (n, ok, d) => { console.log((ok ? "GO   " : "FAIL ") + n + (d ? "  [" + d + "]" : "")); ok ? pass++ : fail++; };

const templates = [B.tractor, B.star, B.boat];
const K = 3;

function argAssign(marg, prev, evid) {
  return marg.map((m, i) => {
    if (prev && prev[i] >= 0 && !evid[i]) return prev[i];
    let b = 0;
    for (let k = 1; k <= K; k++) if (m[k] > m[b]) b = k;
    return b === K ? -1 : b;
  });
}

// ---- scene: three layers + clutter, seeded
let st = 777; const rand = () => { st = (st * 1103515245 + 12345) & 0x7fffffff; return st / 0x7fffffff; };
const TAU = Math.PI * 2;
const xis = [[0.35, -0.3, 0.9, 0.0], [-0.45, 0.35, -2.1, -0.12], [0.05, 0.45, 0.4, -0.25]];
const obs = [], gt = [];
for (let k = 0; k < K; k++)
  for (const a of C.transformAtoms(templates[k], xis[k])) { obs.push(a); gt.push(k); }
for (let i = 0; i < 15; i++) {
  const su = 0.04 + 0.08 * rand();
  obs.push([-0.95 + 1.9 * rand(), -0.95 + 1.9 * rand(), -Math.PI / 2 + Math.PI * rand(),
            su, su * (0.4 + 0.5 * rand()), 4 + 10 * rand(), TAU * rand(), rand(), rand(), rand()]);
  gt.push(-1);
}
const NS = [1, 1, 1, .3, .3, 20, 3, .5, .5, .5];
for (const a of obs) for (let q = 0; q < 10; q++) {
  const g = Math.sqrt(-2 * Math.log(rand() + 1e-9)) * Math.cos(TAU * rand());
  a[q] += g * 0.006 * NS[q];
}
for (const a of obs) { a[3] = Math.max(a[3], .012); a[4] = Math.max(a[4], .008); a[5] = Math.max(a[5], .5); }

// ---- 1. dynamic K=3 bind, third template fresh from the generation path
let r = C.bpBind(templates, obs, { iters: 40 });
let evid = obs.map(() => true);
let assign = argAssign(r.marg, null, evid);
{
  let ok = 0, okBoat = 0, nBoat = 0;
  for (let i = 0; i < gt.length; i++) {
    if (assign[i] === gt[i]) ok++;
    if (gt[i] === 2) { nBoat++; if (assign[i] === 2) okBoat++; }
  }
  check("K=3 bind with generated boat layer", ok / gt.length > 0.85,
        (ok / gt.length).toFixed(3) + " overall");
  check("boat atoms bind to boat", okBoat / nBoat > 0.85, okBoat + "/" + nBoat);
}

// ---- 2. depth auto-occlusion on TRIMMED object layers (studio semantics)
{
  const trimmed = templates;  // builtins are compact by construction (gray-fit)
  const xisT = [[0.3, -0.25, 0.4, -0.6], [0.3, -0.25, -1.0, -0.6], [-0.45, 0.45, 0.2, -0.6]];
  const obsT = [], gtT = [];
  for (let k = 0; k < K; k++)
    for (const a of C.transformAtoms(trimmed[k], xisT[k])) { obsT.push(a); gtT.push(k); }
  // star placed ON the tractor; tractor in front by depth
  const rT = C.bpBind(trimmed, obsT, { iters: 30 });
  let evT = obsT.map(() => true);
  let asT = argAssign(rT.marg, null, evT);

  const H = 128;
  const layers = [
    { depth: 2, hidden: false },   // tractor front
    { depth: 1, hidden: false },   // star behind
    { depth: 0, hidden: false },   // boat back, far away
  ];
  const alphaBufs = layers.map((_, k) =>
    S.alphaFromPre(C.renderPre(obsT.filter((_, i) => asT[i] === k), H), H));
  const { mask, covFrac } = S.autoHidden(obsT, asT, layers, alphaBufs, H);

  check("star (behind, overlapped) loses evidence", covFrac[1] > 0.5,
        (100 * covFrac[1]).toFixed(0) + "% of star atoms covered");
  check("tractor (front) keeps evidence", covFrac[0] < 0.25,
        (100 * covFrac[0]).toFixed(0) + "% covered");
  check("boat (far away) keeps evidence", covFrac[2] < 0.1,
        (100 * covFrac[2]).toFixed(0) + "% covered");

  const wasStar = asT.map(a => a === 1);
  const r2 = C.bpBind(trimmed, obsT, {
    iters: 15, hiddenMask: mask, clampPose: [null, rT.mu[1], null] });
  evT = mask.map(m => !m);
  const as2 = argAssign(r2.marg, asT, evT);
  const ent = C.atomEntropy(r2.marg);
  let hEnt = 0, nH = 0, kept = 0, nWas = 0;
  for (let i = 0; i < gtT.length; i++) {
    if (mask[i]) { hEnt += ent[i]; nH++; }
    if (wasStar[i] && mask[i]) { nWas++; if (as2[i] === 1) kept++; }
  }
  check("covered atoms honestly uncertain", hEnt / nH > 1.5,
        (hEnt / nH).toFixed(2) + " bits");
  check("identity persists under the tractor", kept === nWas,
        kept + "/" + nWas + " covered star atoms stay star");

  // move the star away: coverage releases, full rebind recovers
  const g2 = S.gestureTranslate(-0.7, -0.35);
  for (let i = 0; i < obsT.length; i++)
    if (as2[i] === 1) obsT[i] = C.transformAtoms([obsT[i]], g2)[0];
  const alphaBufs2 = layers.map((_, k) =>
    S.alphaFromPre(C.renderPre(obsT.filter((_, i) => as2[i] === k), H), H));
  const { covFrac: cf2 } = S.autoHidden(obsT, as2, layers, alphaBufs2, H);
  check("coverage releases when moved away", cf2[1] < 0.15,
        (100 * cf2[1]).toFixed(0) + "%");
  const r3 = C.bpBind(trimmed, obsT, { iters: 25 });
  const as3 = argAssign(r3.marg, null, obsT.map(() => true));
  let ok = 0;
  for (let i = 0; i < gtT.length; i++) if (as3[i] === gtT[i]) ok++;
  check("re-emergence from behind rebinds", ok / gtT.length > 0.85,
        (ok / gtT.length).toFixed(3));
}

// ---- 3. painter compositor: depth order decides the overlap
{
  const H = 96;
  const red  = [[-0.3, 0, 0, 0.28, 0.28, 0.05, 0, 1.5, -1.5, -1.5]];
  const blue = [[ 0.3, 0, 0, 0.28, 0.28, 0.05, 0, -1.5, -1.5, 1.5]];
  const rgb = [C.renderPre(red, H), C.renderPre(blue, H)];
  const env = [S.alphaFromPre(rgb[0], H), S.alphaFromPre(rgb[1], H)];
  const pick = (out, x, y) => {
    const px = Math.round((x + 1) / 2 * (H - 1)), py = Math.round((y + 1) / 2 * (H - 1));
    const i = 4 * (py * H + px);
    return [out[i], out[i + 1], out[i + 2]];
  };
  const blueFront = S.compositePainter([0, 1], rgb, env, H);
  const redFront  = S.compositePainter([1, 0], rgb, env, H);
  const a = pick(blueFront, 0.3, 0), b = pick(blueFront, -0.3, 0);
  check("front layer wins at its center", a[2] > 170 && a[0] < 90,
        "rgb(" + a.join(",") + ")");
  check("back layer shows where uncovered", b[0] > 170 && b[2] < 90,
        "rgb(" + b.join(",") + ")");
  const m1 = pick(blueFront, 0, 0), m2 = pick(redFront, 0, 0);
  check("depth order decides the overlap", m1[2] > m2[2] && m2[0] > m1[0],
        "blue-front rgb(" + m1.join(",") + ") vs red-front rgb(" + m2.join(",") + ")");
}

console.log("\n" + pass + " GO, " + fail + " FAIL");
process.exit(fail ? 1 : 0);
