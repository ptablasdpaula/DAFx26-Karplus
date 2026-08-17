// Section 2.1's matching widget.
//
// A three-event extended KSA against the NSynth recording shown above it. All
// events sit on one string at one pitch; each carries its own decay, damping,
// pluck position and dynamics, held from its onset until the next — the same
// piecewise-constant treatment the synthesiser gives a set of events.
//
// Events are selected and moved on the spectrogram itself rather than through a
// list of controls, so the thing you are editing is the thing you are looking
// at, and only the selected event's four parameters are on screen at once.

import { ExtendedKsaProcessor } from './interactive-ks.js';
import { paintSpectrogram } from './optim-demo.js';
import { whenVisible } from './scrollytelling.js';

const FS = 16000;
const DURATION = 4;
const LENGTH = FS * DURATION;
const SPEC = { fftSize: 512, hop: 128 };
const MAX_ONSET = 3.85;

// The target's measured pitch, so matching is actually reachable.
const F0 = 89.3;

const DEFAULTS = [
  { onset: 0, decay: 0.995, damping: 0.32, pluck: 0.25, dynamic: 0.85 },
  { onset: 1.5, decay: 0.99, damping: 0.45, pluck: 0.25, dynamic: 0.12 },
  { onset: 3.0, decay: 0.93, damping: 0.62, pluck: 0.25, dynamic: 0.2 },
];
let events = DEFAULTS.map(event => ({ ...event }));
let selected = 0;

let context = null;
let targetBuffer = null;
let renderedBuffer = null;
let playing = null;
let root = null;
let pending = false;
let drag = null;
let hover = null;

const audio = () => {
  if (!context) context = new AudioContext();
  return context;
};
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** Deterministic zero-mean burst, so the picture does not shimmer as you drag. */
function makeBurst(length, seed) {
  const out = new Float64Array(length);
  let state = seed >>> 0;
  let mean = 0;
  for (let i = 0; i < length; i += 1) {
    state = (state * 1664525 + 1013904223) >>> 0;
    out[i] = (state / 0xffffffff) * 2 - 1;
    mean += out[i];
  }
  mean /= length;
  for (let i = 0; i < length; i += 1) out[i] -= mean;
  return out;
}

const burstLength = Math.floor(FS / F0);
const bursts = [makeBurst(burstLength, 12345), makeBurst(burstLength, 67890), makeBurst(burstLength, 24680)];

/** Events sorted by onset, which is the order the string actually sees them. */
const ordered = () => events
  .map((event, index) => ({ ...event, index }))
  .sort((a, b) => a.onset - b.onset);

/** Render the string at the current settings. */
export function renderMatchSignal() {
  const timeline = ordered();
  const processor = new ExtendedKsaProcessor(FS, F0);
  const out = new Float64Array(LENGTH);
  const starts = timeline.map(event => Math.round(event.onset * FS));

  let current = 0;
  for (let n = 0; n < LENGTH; n += 1) {
    while (current + 1 < timeline.length && n >= starts[current + 1]) current += 1;
    const event = timeline[n >= starts[0] ? current : 0];

    let excitation = 0;
    for (let k = 0; k < timeline.length; k += 1) {
      const offset = n - starts[k];
      if (offset >= 0 && offset < burstLength) {
        excitation += bursts[timeline[k].index][offset] * timeline[k].dynamic;
      }
    }
    out[n] = processor.process(excitation, {
      f0: F0,
      pluck: event.pluck,
      dynamic: event.dynamic,
      damping: event.damping,
      decay: event.decay,
    });
  }

  let peak = 1e-9;
  for (let n = 0; n < LENGTH; n += 1) peak = Math.max(peak, Math.abs(out[n]));
  const buffer = audio().createBuffer(1, LENGTH, FS);
  const channel = buffer.getChannelData(0);
  for (let n = 0; n < LENGTH; n += 1) channel[n] = (out[n] / peak) * 0.9;
  renderedBuffer = buffer;
  return out;
}

/** Event markers, drawn on a transparent canvas over the spectrogram. */
function drawMarkers() {
  const canvas = root.querySelector('[data-match-overlay]');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);

  events.forEach((event, index) => {
    const x = (event.onset / DURATION) * width;
    const active = index === selected;
    const lit = active || index === hover;
    ctx.strokeStyle = active ? '#4da3ff' : lit ? '#8fc7ff' : 'rgba(141,199,255,.55)';
    ctx.lineWidth = active ? 3 : 2;
    ctx.beginPath();
    ctx.moveTo(x, 14);
    ctx.lineTo(x, height);
    ctx.stroke();

    // A grab bar at the top, so it reads as draggable.
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillRect(x - 16, 2, 32, 12);
    ctx.fillStyle = active ? '#04203c' : '#0b2b49';
    ctx.font = 'bold 9px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(String(index + 1), x, 11);
  });
}

function syncControls() {
  const event = events[selected];
  root.querySelectorAll('[data-match-param]').forEach(input => {
    input.value = String(event[input.dataset.matchParam]);
  });
  root.querySelectorAll('[data-match-readout]').forEach(element => {
    const key = element.dataset.matchReadout;
    const value = event[key];
    element.textContent = key === 'decay' ? value.toFixed(4) : value.toFixed(2);
  });
  const label = root.querySelector('[data-match-selected]');
  if (label) label.textContent = `Event ${selected + 1} · onset ${event.onset.toFixed(2)} s`;
  root.querySelectorAll('[data-match-chip]').forEach(chip =>
    chip.classList.toggle('active', Number(chip.dataset.matchChip) === selected));
}

function draw() {
  pending = false;
  paintSpectrogram(root.querySelector('[data-match-spec="yours"]'), renderMatchSignal(), SPEC);
  drawMarkers();
  syncControls();
}

const schedule = () => {
  if (pending) return;
  pending = true;
  requestAnimationFrame(draw);
};

function stop() {
  if (playing) {
    try { playing.stop(); } catch { /* already ended */ }
    playing = null;
  }
  root?.querySelectorAll('[data-match-cell]').forEach(cell => cell.classList.remove('playing'));
}

function play(which, cell) {
  const ctx = audio();
  if (ctx.state === 'suspended') ctx.resume();
  const wasPlaying = cell.classList.contains('playing');
  stop();
  if (wasPlaying) return;

  const buffer = which === 'target' ? targetBuffer : renderedBuffer;
  if (!buffer) return;
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  source.onended = () => cell.classList.remove('playing');
  source.start();
  playing = source;
  cell.classList.add('playing');
}

/** Which event marker is nearest a pointer position, in seconds. */
function nearestEvent(seconds) {
  let best = 0;
  let bestDistance = Infinity;
  events.forEach((event, index) => {
    const distance = Math.abs(event.onset - seconds);
    if (distance < bestDistance) { bestDistance = distance; best = index; }
  });
  return { index: best, distance: bestDistance };
}

function bindOverlay() {
  const canvas = root.querySelector('[data-match-overlay]');
  if (!canvas) return;
  const secondsAt = event => {
    const rect = canvas.getBoundingClientRect();
    return clamp(((event.clientX - rect.left) / rect.width) * DURATION, 0, MAX_ONSET);
  };

  canvas.addEventListener('pointermove', event => {
    if (drag !== null) {
      events[drag].onset = secondsAt(event);
      schedule();
      return;
    }
    const { index, distance } = nearestEvent(secondsAt(event));
    const near = distance < 0.18 ? index : null;
    if (near !== hover) {
      hover = near;
      canvas.style.cursor = near === null ? 'pointer' : 'ew-resize';
      drawMarkers();
    }
  });

  canvas.addEventListener('pointerleave', () => {
    if (drag === null && hover !== null) { hover = null; drawMarkers(); }
  });

  canvas.addEventListener('pointerdown', event => {
    const seconds = secondsAt(event);
    const { index, distance } = nearestEvent(seconds);
    if (distance < 0.18) {
      selected = index;
      drag = index;
      canvas.setPointerCapture(event.pointerId);
      syncControls();
      drawMarkers();
    } else {
      // A click away from any marker plays the attempt, like the target cell.
      play('yours', root.querySelector('[data-match-cell="yours"]'));
    }
  });

  const release = () => {
    if (drag === null) return;
    drag = null;
    schedule();
  };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
}

async function loadTarget(url) {
  const response = await fetch(url);
  targetBuffer = await audio().decodeAudioData(await response.arrayBuffer());
  const channel = targetBuffer.getChannelData(0);
  // Onto the render grid, so both spectrograms use the same rate and window and
  // are therefore actually comparable.
  const ratio = targetBuffer.sampleRate / FS;
  const signal = new Float64Array(LENGTH);
  for (let n = 0; n < LENGTH; n += 1) {
    signal[n] = channel[Math.min(channel.length - 1, Math.round(n * ratio))];
  }
  paintSpectrogram(root.querySelector('[data-match-spec="target"]'), signal, SPEC);
}

export function initMatchWidget() {
  root = document.querySelector('[data-match-widget]');
  if (!root) return;

  root.querySelectorAll('[data-match-param]').forEach(input => {
    input.addEventListener('input', () => {
      events[selected][input.dataset.matchParam] = Number(input.value);
      schedule();
    });
  });

  root.querySelectorAll('[data-match-chip]').forEach(chip => {
    chip.addEventListener('click', () => {
      selected = Number(chip.dataset.matchChip);
      syncControls();
      drawMarkers();
    });
  });

  const targetCell = root.querySelector('[data-match-cell="target"]');
  if (targetCell) targetCell.addEventListener('click', () => play('target', targetCell));

  const reset = root.querySelector('[data-action="match-reset"]');
  if (reset) reset.addEventListener('click', () => {
    events = DEFAULTS.map(event => ({ ...event }));
    selected = 0;
    schedule();
  });

  bindOverlay();

  whenVisible(root, () => {
    draw();
    loadTarget(root.dataset.matchTarget).catch(error =>
      console.warn('match widget: could not load the target', error));
  }, '300px');

  const idle = new IntersectionObserver(entries => {
    if (entries.every(entry => !entry.isIntersecting)) stop();
  }, { threshold: 0.05 });
  idle.observe(root);
}

export { stop as stopMatchWidget, events as matchEvents, F0 as MATCH_F0 };
