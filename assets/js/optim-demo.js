// Section 2's optimisation demos.
//
// The loss surfaces are sampled offline by tools/precompute_landscapes.py using
// the paper's own synthesiser and losses (L = 0.05 L_MSS + 1.0 L_SOT, Eq. 12).
// Here we interpolate that surface, take finite differences for the gradient,
// and descend. The divergence between the time-domain and frequency-sampling
// markers is therefore real: it is the difference between two genuinely
// different landscapes, one of which has been distorted by time-aliasing.

import { renderFrequencySampling } from './ksa-freq.js';
import { ExtendedKsaProcessor } from './interactive-ks.js';
import { stft } from './fft.js';
import { whenVisible } from './scrollytelling.js';

let manifest = null;
let blob = null;

// Resolved against this module rather than the host page, so the demos and the
// test harnesses under tests/ both find the data.
const DATA_BASE = new URL('../data/', import.meta.url);

async function loadData() {
  if (manifest) return manifest;
  const get = async (name, as) => {
    const response = await fetch(new URL(name, DATA_BASE));
    if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
    return as === 'json' ? response.json() : response.arrayBuffer();
  };
  const [meta, bin] = await Promise.all([
    get('landscapes.json', 'json'),
    get('landscapes.bin', 'bin'),
  ]);
  manifest = meta;
  blob = bin;
  return manifest;
}

/** Dequantise one stored slice into physical loss units. */
function dequantise(group, key) {
  const meta = manifest[group];
  const slice = meta.slices[key];
  if (!slice) throw new Error(`no slice ${key}`);
  const [rows, cols] = meta.shape;
  const raw = new Uint16Array(blob, slice.offset, rows * cols);
  const out = new Float32Array(rows * cols);
  const span = (slice.max - slice.min) / 65535;
  for (let i = 0; i < raw.length; i += 1) out[i] = slice.min + raw[i] * span;
  return out;
}

/**
 * A loss surface over the two axes of a landscape.
 *
 * Landscape B stores L_MSS and L_SOT separately, because they behave very
 * differently on f0 and the training objective is dominated by one of them.
 * Passing `mix` (0 = pure MSS, 1 = pure SOT) blends them; each term is first
 * normalised to its own range, since their absolute scales are unrelated.
 */
class Surface {
  constructor(group, key, mix = null) {
    const meta = manifest[group];
    const [rows, cols] = meta.shape;
    this.rows = rows;
    this.cols = cols;
    this.xValues = meta.axes.x.values;
    this.yValues = meta.axes.y.values;

    if (mix === null) {
      const slice = meta.slices[key];
      if (!slice) throw new Error(`no slice ${key}`);
      this.data = dequantise(group, key);
      this.min = slice.min;
      this.max = slice.max;
    } else {
      const mss = dequantise(group, `${key}_mss`);
      const sot = dequantise(group, `${key}_sot`);
      const range = buf => {
        let lo = Infinity, hi = -Infinity;
        for (let i = 0; i < buf.length; i += 1) {
          if (buf[i] < lo) lo = buf[i];
          if (buf[i] > hi) hi = buf[i];
        }
        return [lo, (hi - lo) || 1];
      };
      const [mLo, mSpan] = range(mss);
      const [sLo, sSpan] = range(sot);
      this.data = new Float32Array(rows * cols);
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < this.data.length; i += 1) {
        const v = (1 - mix) * ((mss[i] - mLo) / mSpan) + mix * ((sot[i] - sLo) / sSpan);
        this.data[i] = v;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      this.min = lo;
      this.max = hi;
    }
  }

  /** Loss at fractional grid coordinates (col, row), bilinearly interpolated. */
  at(col, row) {
    const c = Math.max(0, Math.min(this.cols - 1.001, col));
    const r = Math.max(0, Math.min(this.rows - 1.001, row));
    const c0 = Math.floor(c);
    const r0 = Math.floor(r);
    const fc = c - c0;
    const fr = r - r0;
    const i = (rr, cc) => this.data[rr * this.cols + cc];
    return (
      i(r0, c0) * (1 - fc) * (1 - fr) +
      i(r0, c0 + 1) * fc * (1 - fr) +
      i(r0 + 1, c0) * (1 - fc) * fr +
      i(r0 + 1, c0 + 1) * fc * fr
    );
  }

  /** Central-difference gradient in grid units. */
  gradient(col, row, h = 0.75) {
    return [
      (this.at(col + h, row) - this.at(col - h, row)) / (2 * h),
      (this.at(col, row + h) - this.at(col, row - h)) / (2 * h),
    ];
  }

  /** Grid coordinates of the global minimum. */
  argmin() {
    let best = Infinity;
    let bc = 0;
    let br = 0;
    for (let r = 0; r < this.rows; r += 1) {
      for (let c = 0; c < this.cols; c += 1) {
        const v = this.data[r * this.cols + c];
        if (v < best) { best = v; bc = c; br = r; }
      }
    }
    return { col: bc, row: br, value: best };
  }
}

/** Perceptually ordered blue -> yellow ramp for the loss heatmap. */
function lossColour(t) {
  const stops = [
    [ 26,  32,  62],
    [ 44,  84, 130],
    [ 47, 138, 138],
    [123, 184,  99],
    [237, 199,  74],
    [252, 246, 197],
  ];
  const x = Math.max(0, Math.min(0.9999, t)) * (stops.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = stops[i];
  const b = stops[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

function paintSurface(canvas, surface) {
  const { rows, cols } = surface;
  const image = new ImageData(cols, rows);
  const span = surface.max - surface.min || 1;
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      // Normalised to this slice's own range, per the brief.
      const t = (surface.data[r * cols + c] - surface.min) / span;
      const [red, green, blue] = lossColour(1 - t); // bright = low loss
      const o = ((rows - 1 - r) * cols + c) * 4;
      image.data[o] = red;
      image.data[o + 1] = green;
      image.data[o + 2] = blue;
      image.data[o + 3] = 255;
    }
  }
  const off = document.createElement('canvas');
  off.width = cols;
  off.height = rows;
  off.getContext('2d').putImageData(image, 0, 0);
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  return ctx;
}

/** One descending marker on a surface. */
class Walker {
  constructor(surface, colour, label, { stepSize = 1.4 } = {}) {
    this.surface = surface;
    this.colour = colour;
    this.label = label;
    this.stepSize = stepSize;
    this.trail = [];
  }

  reset(col, row) {
    this.col = col;
    this.row = row;
    this.trail = [[col, row]];
    this.done = false;
  }

  step() {
    if (this.done) return;
    const [gc, gr] = this.surface.gradient(this.col, this.row);
    const norm = Math.hypot(gc, gr);
    if (norm < 1e-9) { this.done = true; return; }
    // Normalised step: the two surfaces have very different loss scales, and
    // comparing their *paths* is the point, not their raw gradient magnitudes.
    this.col = Math.max(0, Math.min(this.surface.cols - 1, this.col - this.stepSize * gc / norm));
    this.row = Math.max(0, Math.min(this.surface.rows - 1, this.row - this.stepSize * gr / norm));
    this.trail.push([this.col, this.row]);
    if (this.trail.length > 400) this.done = true;
  }
}

function drawWalkers(ctx, canvas, walkers, surface) {
  const toX = col => (col / (surface.cols - 1)) * canvas.width;
  const toY = row => canvas.height - (row / (surface.rows - 1)) * canvas.height;

  walkers.forEach(walker => {
    if (!walker.trail.length) return;
    ctx.strokeStyle = walker.colour;
    ctx.lineWidth = 2;
    ctx.beginPath();
    walker.trail.forEach(([c, r], i) => {
      const x = toX(c);
      const y = toY(r);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Direction arrows every so often, so a still frame still reads as motion.
    for (let i = 8; i < walker.trail.length; i += 24) {
      const [c0, r0] = walker.trail[i - 4];
      const [c1, r1] = walker.trail[i];
      const x0 = toX(c0), y0 = toY(r0), x1 = toX(c1), y1 = toY(r1);
      const angle = Math.atan2(y1 - y0, x1 - x0);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - 7 * Math.cos(angle - 0.4), y1 - 7 * Math.sin(angle - 0.4));
      ctx.lineTo(x1 - 7 * Math.cos(angle + 0.4), y1 - 7 * Math.sin(angle + 0.4));
      ctx.closePath();
      ctx.fillStyle = walker.colour;
      ctx.fill();
    }

    const [c, r] = walker.trail[walker.trail.length - 1];
    ctx.beginPath();
    ctx.arc(toX(c), toY(r), 6, 0, Math.PI * 2);
    ctx.fillStyle = walker.colour;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
  });
}

function drawTarget(ctx, canvas, surface, col, row) {
  const x = (col / (surface.cols - 1)) * canvas.width;
  const y = canvas.height - (row / (surface.rows - 1)) * canvas.height;
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x - 9, y); ctx.lineTo(x + 9, y);
  ctx.moveTo(x, y - 9); ctx.lineTo(x, y + 9);
  ctx.stroke();
}

const nearestIndex = (values, target) => {
  let best = 0;
  let bestDistance = Infinity;
  values.forEach((v, i) => {
    const d = Math.abs(v - target);
    if (d < bestDistance) { bestDistance = d; best = i; }
  });
  return best;
};

// ── Spectrogram rendering ───────────────────────────────────────────────────

function renderTimeDomain({ f0, fs, decay, damping, pluck, dynamic, onsetSamples, length }) {
  const processor = new ExtendedKsaProcessor(fs, f0);
  const out = new Float64Array(length);
  const burstLength = Math.floor(fs / f0);
  const start = Math.max(0, Math.round(onsetSamples));
  for (let i = start; i < length; i += 1) {
    const n = i - start;
    const excitation = n < burstLength ? (Math.sin(n * 12.9898) * 43758.5453 % 1) * 2 - 1 : 0;
    out[i] = processor.process(excitation, { pluck, dynamic, damping, decay });
  }
  return out;
}

function paintSpectrogram(canvas, signal, { fftSize = 256, hop = 64, tint = null }) {
  const { frames, bins, data } = stft(signal, { fftSize, hop });
  const image = new ImageData(frames, bins);
  for (let f = 0; f < frames; f += 1) {
    for (let k = 0; k < bins; k += 1) {
      const v = Math.max(0, Math.min(1, data[f * bins + k]));
      const o = ((bins - 1 - k) * frames + f) * 4;
      if (tint) {
        image.data[o] = Math.round(255 - (255 - tint[0]) * v);
        image.data[o + 1] = Math.round(255 - (255 - tint[1]) * v);
        image.data[o + 2] = Math.round(255 - (255 - tint[2]) * v);
        image.data[o + 3] = 255;
      } else {
        const [r, g, b] = lossColour(v);
        image.data[o] = r;
        image.data[o + 1] = g;
        image.data[o + 2] = b;
        image.data[o + 3] = 255;
      }
    }
  }
  const off = document.createElement('canvas');
  off.width = frames;
  off.height = bins;
  off.getContext('2d').putImageData(image, 0, 0);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
}

/** Overlay two spectrograms: target in red, prediction in blue. */
function paintOverlay(canvas, target, prediction, options = {}) {
  const { fftSize = 256, hop = 64 } = options;
  const a = stft(target, { fftSize, hop });
  const b = stft(prediction, { fftSize, hop });
  const frames = Math.min(a.frames, b.frames);
  const bins = a.bins;
  const image = new ImageData(frames, bins);
  for (let f = 0; f < frames; f += 1) {
    for (let k = 0; k < bins; k += 1) {
      const t = Math.max(0, Math.min(1, a.data[f * bins + k]));
      const p = Math.max(0, Math.min(1, b.data[f * bins + k]));
      const o = ((bins - 1 - k) * frames + f) * 4;
      image.data[o] = Math.round(255 - 200 * p);          // red channel kept by target
      image.data[o + 1] = Math.round(255 - 200 * Math.max(t, p));
      image.data[o + 2] = Math.round(255 - 200 * t);      // blue channel kept by prediction
      image.data[o + 3] = 255;
    }
  }
  const off = document.createElement('canvas');
  off.width = frames;
  off.height = bins;
  off.getContext('2d').putImageData(image, 0, 0);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
}

export {
  loadData, Surface, Walker, paintSurface, drawWalkers, drawTarget,
  paintSpectrogram, paintOverlay, renderTimeDomain, renderFrequencySampling,
  nearestIndex, whenVisible, lossColour,
};
