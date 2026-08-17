// Section 2.1's matching widget.
//
// A two-event extended KSA against a real NSynth recording. Both events sit on
// one string at one pitch, with onsets fixed at 0 s and 3 s — which is the
// structure of the target: a pluck, a long decay, then a release. Parameters are
// piecewise constant, held from each onset until the next, exactly as the
// synthesiser's _expand_sparse_events_to_dense does.
//
// The point is not that anyone will match it exactly. It is that trying makes
// the parameterisation concrete before Section 2.2 starts asking whether a
// gradient could do the same job.

import { ExtendedKsaProcessor } from './interactive-ks.js';
import { paintSpectrogram } from './optim-demo.js';
import { whenVisible } from './scrollytelling.js';

const FS = 16000;
const DURATION = 4;
const LENGTH = FS * DURATION;
const SECOND_ONSET = 3;
const SPEC = { fftSize: 512, hop: 128 };

// The target's measured pitch, so matching is actually reachable.
const F0 = 89.3;

const events = [
  { decay: 0.995, damping: 0.32, pluck: 0.25, dynamic: 0.85 },
  { decay: 0.93, damping: 0.62, pluck: 0.25, dynamic: 0.25 },
];

let context = null;
let targetBuffer = null;
let renderedBuffer = null;
let playing = null;
let root = null;
let pending = false;

const audio = () => {
  if (!context) context = new AudioContext();
  return context;
};

/** Deterministic zero-mean burst, so the picture does not shimmer as you drag. */
function burst(length, seed) {
  const out = new Float64Array(length);
  let state = seed;
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
const bursts = [burst(burstLength, 12345), burst(burstLength, 67890)];

/** Render the two-event string at the current settings. */
function render() {
  const processor = new ExtendedKsaProcessor(FS, F0);
  const out = new Float64Array(LENGTH);
  const switchAt = SECOND_ONSET * FS;
  for (let n = 0; n < LENGTH; n += 1) {
    const event = n < switchAt ? events[0] : events[1];
    let excitation = 0;
    if (n < burstLength) excitation = bursts[0][n] * events[0].dynamic;
    else if (n >= switchAt && n < switchAt + burstLength) {
      excitation = bursts[1][n - switchAt] * events[1].dynamic;
    }
    out[n] = processor.process(excitation, {
      f0: F0,
      pluck: event.pluck,
      dynamic: event.dynamic,
      damping: event.damping,
      decay: event.decay,
    });
  }
  // Normalise for playback only; the spectrogram normalises itself.
  let peak = 1e-9;
  for (let n = 0; n < LENGTH; n += 1) peak = Math.max(peak, Math.abs(out[n]));
  const buffer = audio().createBuffer(1, LENGTH, FS);
  const channel = buffer.getChannelData(0);
  for (let n = 0; n < LENGTH; n += 1) channel[n] = (out[n] / peak) * 0.9;
  renderedBuffer = buffer;
  return out;
}

function draw() {
  pending = false;
  const signal = render();
  paintSpectrogram(root.querySelector('[data-match-spec="yours"]'), signal, SPEC);
  root.querySelectorAll('[data-match-readout]').forEach(element => {
    const [index, key] = element.dataset.matchReadout.split('.');
    const value = events[Number(index)][key];
    element.textContent = key === 'decay' ? value.toFixed(4) : value.toFixed(2);
  });
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
  const context_ = audio();
  if (context_.state === 'suspended') context_.resume();
  const wasPlaying = cell.classList.contains('playing');
  stop();
  if (wasPlaying) return;

  const buffer = which === 'target' ? targetBuffer : renderedBuffer;
  if (!buffer) return;
  const source = context_.createBufferSource();
  source.buffer = buffer;
  source.connect(context_.destination);
  source.onended = () => cell.classList.remove('playing');
  source.start();
  playing = source;
  cell.classList.add('playing');
}

async function loadTarget(url) {
  const response = await fetch(url);
  const bytes = await response.arrayBuffer();
  targetBuffer = await audio().decodeAudioData(bytes);
  const channel = targetBuffer.getChannelData(0);
  // Resample by simple decimation onto the render grid so both spectrograms
  // are computed at the same rate with the same window — otherwise they are
  // not comparable, which is the whole point.
  const ratio = targetBuffer.sampleRate / FS;
  const signal = new Float64Array(LENGTH);
  for (let n = 0; n < LENGTH; n += 1) {
    const i = Math.min(channel.length - 1, Math.round(n * ratio));
    signal[n] = channel[i];
  }
  paintSpectrogram(root.querySelector('[data-match-spec="target"]'), signal, SPEC);
}

export function initMatchWidget() {
  root = document.querySelector('[data-match-widget]');
  if (!root) return;

  root.querySelectorAll('[data-match-param]').forEach(input => {
    const [index, key] = input.dataset.matchParam.split('.');
    input.value = String(events[Number(index)][key]);
    input.addEventListener('input', () => {
      events[Number(index)][key] = Number(input.value);
      schedule();
    });
  });

  root.querySelectorAll('[data-match-cell]').forEach(cell => {
    cell.addEventListener('click', () => play(cell.dataset.matchCell, cell));
  });

  const reset = root.querySelector('[data-action="match-reset"]');
  if (reset) reset.addEventListener('click', () => {
    events[0] = { decay: 0.995, damping: 0.32, pluck: 0.25, dynamic: 0.85 };
    events[1] = { decay: 0.93, damping: 0.62, pluck: 0.25, dynamic: 0.25 };
    root.querySelectorAll('[data-match-param]').forEach(input => {
      const [index, key] = input.dataset.matchParam.split('.');
      input.value = String(events[Number(index)][key]);
    });
    schedule();
  });

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

export { stop as stopMatchWidget, render as renderMatchSignal, events as matchEvents, F0 as MATCH_F0 };
