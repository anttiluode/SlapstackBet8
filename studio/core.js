/* Slapstack Playroom core — faithful JS port of SlapstackBet6 math.
   FIELDS: [x, y, theta, su, sv, f, phase, r, g, b]
   Verified against bet6_open.py / bet6_bp_binding.py / bet6_multimodal.py
   by tests_node.js before being embedded in the playroom. */
"use strict";

const TAU = Math.PI * 2;

function wrapPi(d) {
  return ((d + Math.PI) % TAU + TAU) % TAU - Math.PI;
}

function rotApply(rho, x, y) {
  const c = Math.cos(rho), s = Math.sin(rho);
  return [c * x - s * y, s * x + c * y];
}

/* Exact Sim(2) action on atom parameters (the Bet 5 algebra):
   xy -> s R xy + t, theta -> theta + rho, sigma -> s sigma, f -> f/s.
   Envelope-relative phase and color are INVARIANT. */
function transformAtoms(atoms, xi) {
  const [tx, ty, rho, lam] = xi;
  const s = Math.exp(lam);
  const out = new Array(atoms.length);
  for (let i = 0; i < atoms.length; i++) {
    const a = atoms[i];
    const [rx, ry] = rotApply(rho, a[0], a[1]);
    out[i] = [
      s * rx + tx, s * ry + ty,
      a[2] + rho,
      a[3] * s, a[4] * s,
      a[5] / s,
      a[6],
      a[7], a[8], a[9],
    ];
  }
  return out;
}

/* Sim(2)-invariant intrinsic signature: identity lives here. */
function signature(atoms) {
  return atoms.map(a => [
    Math.log(a[3] * a[5]),
    Math.log(a[3] / a[4]),
    Math.cos(a[6]), Math.sin(a[6]),
    a[7], a[8], a[9],
  ]);
}

/* Two pose-vote hypotheses per correspondence (pi-ambiguity fix):
   H0: rho = d_theta,      phi_obs ==  phi_tmpl
   H1: rho = d_theta + pi, phi_obs == -phi_tmpl  */
function poseVotes2pi(obs, tmpl, sigPhase = 0.35) {
  const s = Math.pow((obs[3] / tmpl[3]) * (obs[4] / tmpl[4]) * (tmpl[5] / obs[5]), 1 / 3);
  const dTheta = obs[2] - tmpl[2];
  const out = [];
  for (let H = 0; H < 2; H++) {
    const rho = wrapPi(dTheta + H * Math.PI);
    const phiExp = H === 0 ? tmpl[6] : -tmpl[6];
    const dphi = wrapPi(obs[6] - phiExp);
    const pc = -0.5 * dphi * dphi / (sigPhase * sigPhase);
    const [rx, ry] = rotApply(rho, tmpl[0], tmpl[1]);
    out.push([[obs[0] - s * rx, obs[1] - s * ry, rho, Math.log(s)], pc]);
  }
  return out;
}

/* ------- small dense linear algebra on 4x4 (row-major flat arrays) ------- */
function mat4Inv(m) {
  // Gauss-Jordan, fine for well-conditioned SPD 4x4s here.
  const a = m.map(r => r.slice());
  const inv = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  for (let col = 0; col < 4; col++) {
    let piv = col;
    for (let r = col + 1; r < 4; r++)
      if (Math.abs(a[r][col]) > Math.abs(a[piv][col])) piv = r;
    [a[col], a[piv]] = [a[piv], a[col]];
    [inv[col], inv[piv]] = [inv[piv], inv[col]];
    const d = a[col][col];
    for (let j = 0; j < 4; j++) { a[col][j] /= d; inv[col][j] /= d; }
    for (let r = 0; r < 4; r++) {
      if (r === col) continue;
      const f = a[r][col];
      for (let j = 0; j < 4; j++) { a[r][j] -= f * a[col][j]; inv[r][j] -= f * inv[col][j]; }
    }
  }
  return inv;
}
function mat4Det(m) {
  const a = m.map(r => r.slice());
  let det = 1;
  for (let col = 0; col < 4; col++) {
    let piv = col;
    for (let r = col + 1; r < 4; r++)
      if (Math.abs(a[r][col]) > Math.abs(a[piv][col])) piv = r;
    if (piv !== col) { [a[col], a[piv]] = [a[piv], a[col]]; det = -det; }
    det *= a[col][col];
    if (a[col][col] === 0) return 0;
    for (let r = col + 1; r < 4; r++) {
      const f = a[r][col] / a[col][col];
      for (let j = col; j < 4; j++) a[r][j] -= f * a[col][j];
    }
  }
  return det;
}
function mat4Vec(m, v) {
  return [0,1,2,3].map(i => m[i][0]*v[0]+m[i][1]*v[1]+m[i][2]*v[2]+m[i][3]*v[3]);
}
function matAdd(A, B, wB = 1) {
  return A.map((row, i) => row.map((x, j) => x + wB * B[i][j]));
}

/* Greedy mode-seeking init (angle-aware), port of _density_peaks. */
function densityPeaks(votes, weights, M, radius = 0.45) {
  const scale = [0.15, 0.15, 0.30, 0.20];
  const n = votes.length;
  const dens = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let acc = 0;
    for (let k = 0; k < n; k++) {
      const d0 = (votes[k][0] - votes[i][0]) / scale[0];
      const d1 = (votes[k][1] - votes[i][1]) / scale[1];
      const d2 = wrapPi(votes[k][2] - votes[i][2]) / scale[2];
      const d3 = (votes[k][3] - votes[i][3]) / scale[3];
      acc += weights[k] * Math.exp(-0.5 * (d0*d0 + d1*d1 + d2*d2 + d3*d3));
    }
    dens[i] = acc;
  }
  const peaks = [];
  const alive = new Uint8Array(n).fill(1);
  for (let m = 0; m < M; m++) {
    let best = -1, bestD = -Infinity;
    for (let i = 0; i < n; i++)
      if (alive[i] && dens[i] > bestD) { bestD = dens[i]; best = i; }
    if (best < 0) break;
    peaks.push(votes[best].slice());
    for (let i = 0; i < n; i++) {
      const dxy = Math.hypot(votes[i][0] - votes[best][0], votes[i][1] - votes[best][1]);
      const dr = Math.abs(wrapPi(votes[i][2] - votes[best][2]));
      if (dxy + dr <= radius) alive[i] = 0;
    }
  }
  return peaks;
}

/* Loopy BP binding of obs atoms to K templates at unknown poses.
   Port of bet6_open.bp_bind: candidates carry both pi-hypotheses,
   cavity messages, damping, branch-aligned rotation fusion.
   Options: clampPose — array of length K; if clampPose[k] is a pose xi,
   object k's pose is held fixed (conditioning-as-intervention) and only
   the assignment beliefs re-equilibrate around it. */
function bpBind(templates, obs, opts = {}) {
  const iters = opts.iters ?? 40;
  const damping = opts.damping ?? 0.5;
  const cavity = opts.cavity ?? true;
  const sigVar = opts.sigVar ?? 0.08;
  const outLL = opts.outLL ?? -14.0;
  const clampPose = opts.clampPose ?? null;
  const hiddenMask = opts.hiddenMask ?? null;  // per-atom: true = no evidence
  const onIter = opts.onIter ?? null;

  const K = templates.length;
  const sigT = templates.map(signature);
  const sigO = signature(obs);
  const N = obs.length;

  const Vdiag = [0.03 * 0.03, 0.03 * 0.03, 0.05 * 0.05, 0.05 * 0.05];
  const Vinv = [[1/Vdiag[0],0,0,0],[0,1/Vdiag[1],0,0],[0,0,1/Vdiag[2],0],[0,0,0,1/Vdiag[3]]];
  const Vmat = [[Vdiag[0],0,0,0],[0,Vdiag[1],0,0],[0,0,Vdiag[2],0],[0,0,0,Vdiag[3]]];
  const P0inv = [[1e-2,0,0,0],[0,1e-2,0,0],[0,0,1e-2,0],[0,0,0,1e-2]];

  // ---- candidate generation: 3 signature-nearest per template, 2 hypotheses
  const cands = [], votes = [], base = [];
  for (let i = 0; i < N; i++) {
    const c = [], v = [], b = [];
    if (!(hiddenMask && hiddenMask[i])) {
      for (let k = 0; k < K; k++) {
        const d2 = sigT[k].map(st => {
          let acc = 0;
          for (let q = 0; q < 7; q++) { const d = st[q] - sigO[i][q]; acc += d * d; }
          return acc;
        });
        const order = d2.map((d, j) => [d, j]).sort((p, q) => p[0] - q[0]).slice(0, 3);
        for (const [dj, j] of order) {
          for (const [xi, pc] of poseVotes2pi(obs[i], templates[k][j])) {
            c.push([k, j]); v.push(xi); b.push(-0.5 * dj / sigVar + pc);
          }
        }
      }
    }
    cands.push(c); votes.push(v); base.push(b);
  }

  // ---- beliefs seeded from the identity+phase channel
  let B = [];
  for (let i = 0; i < N; i++) {
    const ll = base[i].concat([outLL]);
    const mx = Math.max(...ll);
    let e = ll.map(x => Math.exp(x - mx));
    const s = e.reduce((a, x) => a + x, 0);
    B.push(e.map(x => x / s));
  }

  // ---- pose init: density peak of each object's votes (or the clamp)
  let mu = [];
  for (let k = 0; k < K; k++) {
    if (clampPose && clampPose[k]) { mu.push(clampPose[k].slice()); continue; }
    const vk = [], wk = [];
    for (let i = 0; i < N; i++)
      for (let ci = 0; ci < cands[i].length; ci++)
        if (cands[i][ci][0] === k) { vk.push(votes[i][ci]); wk.push(B[i][ci]); }
    mu.push(vk.length ? densityPeaks(vk, wk, 1)[0] : [0, 0, 0, 0]);
  }
  let Sig = mu.map(() => [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]);

  for (let it = 0; it < iters; it++) {
    // pose fusion
    const Lam = [], eta = [];
    for (let k = 0; k < K; k++) { Lam.push(P0inv.map(r => r.slice())); eta.push([0,0,0,0]); }
    for (let i = 0; i < N; i++) {
      for (let ci = 0; ci < cands[i].length; ci++) {
        const k = cands[i][ci][0];
        const v = votes[i][ci].slice();
        v[2] = mu[k][2] + wrapPi(v[2] - mu[k][2]);
        const w = B[i][ci];
        Lam[k] = matAdd(Lam[k], Vinv, w);
        const Vv = mat4Vec(Vinv, v);
        for (let q = 0; q < 4; q++) eta[k][q] += w * Vv[q];
      }
    }
    Sig = Lam.map(mat4Inv);
    for (let k = 0; k < K; k++) {
      if (clampPose && clampPose[k]) {
        mu[k] = clampPose[k].slice();
        Sig[k] = [[1e-6,0,0,0],[0,1e-6,0,0],[0,0,1e-6,0],[0,0,0,1e-6]];
      } else {
        mu[k] = mat4Vec(Sig[k], eta[k]);
        mu[k][2] = wrapPi(mu[k][2]);
      }
    }

    // assignment update with cavity
    const newB = [];
    for (let i = 0; i < N; i++) {
      const nc = cands[i].length;
      const ll = new Array(nc + 1);
      for (let ci = 0; ci < nc; ci++) {
        const k = cands[i][ci][0];
        const v = votes[i][ci].slice();
        v[2] = mu[k][2] + wrapPi(v[2] - mu[k][2]);
        let mC, Sk;
        if (cavity && !(clampPose && clampPose[k])) {
          const Lc = matAdd(Lam[k], Vinv, -B[i][ci]);
          const Vv = mat4Vec(Vinv, v);
          const ecav = [0,1,2,3].map(q => eta[k][q] - B[i][ci] * Vv[q]);
          const Sc = mat4Inv(Lc);
          mC = mat4Vec(Sc, ecav);
          Sk = Sc;
        } else {
          mC = mu[k]; Sk = Sig[k];
        }
        const r = [v[0] - mC[0], v[1] - mC[1], wrapPi(v[2] - mC[2]), v[3] - mC[3]];
        const Cov = matAdd(Sk, Vmat, 1);
        const Ci = mat4Inv(Cov);
        const Cr = mat4Vec(Ci, r);
        const quad = r[0]*Cr[0] + r[1]*Cr[1] + r[2]*Cr[2] + r[3]*Cr[3];
        ll[ci] = base[i][ci] - 0.5 * quad - 0.5 * Math.log(mat4Det(Cov));
      }
      ll[nc] = outLL;
      const mx = Math.max(...ll);
      let e = ll.map(x => Math.exp(x - mx));
      const s = e.reduce((a, x) => a + x, 0);
      e = e.map(x => x / s);
      newB.push(e.map((x, q) => damping * x + (1 - damping) * B[i][q]));
    }
    B = newB;
    if (onIter) onIter(it, marginals(), mu, Sig);
  }

  function marginals() {
    const marg = [];
    for (let i = 0; i < N; i++) {
      if (cands[i].length === 0) {
        // no evidence: assignment belief reverts to the prior (uniform),
        // not to a confident "outlier" — this is what permanence means.
        marg.push(new Array(K + 1).fill(1 / (K + 1)));
        continue;
      }
      const m = new Array(K + 1).fill(0);
      for (let ci = 0; ci < cands[i].length; ci++) m[cands[i][ci][0]] += B[i][ci];
      m[K] = B[i][cands[i].length];
      marg.push(m);
    }
    return marg;
  }

  return { marg: marginals(), mu, Sig, cands, votes, B };
}

/* Numpy-matching reference render (verification + full-res compositor).
   pre[c] += color_c * env * carrier ; out = sigmoid(2 * pre).
   Returns {pre: Float32Array(3*H*H)} pre-sigmoid field. */
function renderPre(atoms, H, pre) {
  pre = pre || new Float32Array(3 * H * H);
  const lim = 3.2; // envelope support cut, in sigmas
  for (const a of atoms) {
    const [ax, ay, th, su, sv, f, ph, r, g, b] = a;
    const ct = Math.cos(th), st = Math.sin(th);
    const rad = lim * Math.max(su, sv);
    // pixel bbox: x in [-1,1] maps to col (H-1)*(x+1)/2
    const x0 = Math.max(0, Math.floor((ax - rad + 1) / 2 * (H - 1)));
    const x1 = Math.min(H - 1, Math.ceil((ax + rad + 1) / 2 * (H - 1)));
    const y0 = Math.max(0, Math.floor((ay - rad + 1) / 2 * (H - 1)));
    const y1 = Math.min(H - 1, Math.ceil((ay + rad + 1) / 2 * (H - 1)));
    for (let py = y0; py <= y1; py++) {
      const Y = -1 + 2 * py / (H - 1);
      const dy = Y - ay;
      for (let px = x0; px <= x1; px++) {
        const X = -1 + 2 * px / (H - 1);
        const dx = X - ax;
        const u = ct * dx + st * dy;
        const v = -st * dx + ct * dy;
        const eArg = 0.5 * ((u / su) * (u / su) + (v / sv) * (v / sv));
        if (eArg > lim * lim / 2 * 1.6) continue;
        const env = Math.exp(-eArg);
        const car = Math.cos(TAU * f * u + ph);
        const ec = env * car;
        const idx = py * H + px;
        pre[idx] += r * ec;
        pre[H * H + idx] += g * ec;
        pre[2 * H * H + idx] += b * ec;
      }
    }
  }
  return pre;
}

function sigmoidField(pre, H, out) {
  out = out || new Uint8ClampedArray(4 * H * H);
  const n = H * H;
  for (let i = 0; i < n; i++) {
    out[4 * i]     = 255 / (1 + Math.exp(-2 * pre[i]));
    out[4 * i + 1] = 255 / (1 + Math.exp(-2 * pre[n + i]));
    out[4 * i + 2] = 255 / (1 + Math.exp(-2 * pre[2 * n + i]));
    out[4 * i + 3] = 255;
  }
  return out;
}

/* Ownership field: P(k|pixel) through the atoms' actual envelopes,
   energy-weighted. Port of bet6_open.ownership_field. */
function ownershipField(obs, marg, K, H) {
  const O = [];
  for (let k = 0; k <= K; k++) O.push(new Float32Array(H * H));
  const lim = 3.2;
  for (let i = 0; i < obs.length; i++) {
    const a = obs[i];
    const energy = Math.hypot(a[7], a[8], a[9]);
    const ct = Math.cos(a[2]), st = Math.sin(a[2]);
    const rad = lim * Math.max(a[3], a[4]);
    const x0 = Math.max(0, Math.floor((a[0] - rad + 1) / 2 * (H - 1)));
    const x1 = Math.min(H - 1, Math.ceil((a[0] + rad + 1) / 2 * (H - 1)));
    const y0 = Math.max(0, Math.floor((a[1] - rad + 1) / 2 * (H - 1)));
    const y1 = Math.min(H - 1, Math.ceil((a[1] + rad + 1) / 2 * (H - 1)));
    for (let py = y0; py <= y1; py++) {
      const Y = -1 + 2 * py / (H - 1);
      const dy = Y - a[1];
      for (let px = x0; px <= x1; px++) {
        const X = -1 + 2 * px / (H - 1);
        const dx = X - a[0];
        const u = ct * dx + st * dy;
        const v = -st * dx + ct * dy;
        const env = Math.exp(-0.5 * ((u / a[3]) ** 2 + (v / a[4]) ** 2));
        const idx = py * H + px;
        for (let k = 0; k <= K; k++) O[k][idx] += marg[i][k] * energy * env;
      }
    }
  }
  const P = O.map(() => new Float32Array(H * H));
  const ent = new Float32Array(H * H);
  const support = new Uint8Array(H * H);
  for (let idx = 0; idx < H * H; idx++) {
    let tot = 0;
    for (let k = 0; k <= K; k++) tot += O[k][idx];
    support[idx] = tot > 0.05 ? 1 : 0;
    let e = 0;
    for (let k = 0; k <= K; k++) {
      const p = O[k][idx] / (tot + 1e-6);
      P[k][idx] = p;
      e -= p * Math.log2(p + 1e-12);
    }
    ent[idx] = e;
  }
  return { P, ent, support };
}

/* Assignment entropy per atom, in bits. */
function atomEntropy(marg) {
  return marg.map(m => {
    let e = 0;
    for (const p of m) e -= p * Math.log2(p + 1e-12);
    return e;
  });
}

if (typeof module !== "undefined") {
  module.exports = {
    wrapPi, transformAtoms, signature, poseVotes2pi, densityPeaks,
    bpBind, renderPre, sigmoidField, ownershipField, atomEntropy,
    mat4Inv, mat4Det,
  };
}
