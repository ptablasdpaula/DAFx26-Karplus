// Section 1's sticky figure.
//
// Two layers, one picture:
//   top     the analytic |H(e^jw)| of whichever algorithm is on screen, straight
//           from the z-domain. It responds to the knobs and to nothing else —
//           no audio goes into it.
//   bottom  the measured spectrum of the ringing string, decaying after each
//           excitation.
//
// The comb teeth on top say where energy is allowed to live; the trace below
// shows it draining out of them at the rate the loop filter sets.

import { ExtendedKsaProcessor, OriginalKsaProcessor } from './interactive-ks.js';
import { centsError, extendedResponse, originalResponse } from './ks-response.js';
import { observeSteps } from './scrollytelling.js';

const RESPONSE_POINTS = 900;
const FLOOR_DB = -66;
const MAX_HZ = 4000; // the interesting comb structure lives well below Nyquist

const state = {
  mode: 'original',
  f0: 220,
  decay: 0.99,
  damping: 0.28,
  pluck: 0.25,
  dynamic: 0.6,
};

let audioContext = null;
let analyser = null;
let voice = null;
let canvas = null;
let ctx = null;
let spectrumBuffer = null;
let running = false;
let root = null;

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const sampleRate = () => (audioContext ? audioContext.sampleRate : 48000);

function ensureAudio() {
  if (!audioContext) {
    audioContext = new AudioContext();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 4096;
    analyser.smoothingTimeConstant = 0.55;
    analyser.minDecibels = -110;
    analyser.maxDecibels = -10;
    spectrumBuffer = new Float32Array(analyser.frequencyBinCount);
    analyser.connect(audioContext.destination);
  }
  return audioContext;
}

/** Trigger one pluck through whichever algorithm is currently on screen. */
export function pluck() {
  const context = ensureAudio();
  if (context.state === 'suspended') context.resume();
  if (voice) voice.stop();

  const fs = context.sampleRate;
  const original = state.mode === 'original';
  const originalProcessor = new OriginalKsaProcessor(fs, state.f0);
  const extendedProcessor = new ExtendedKsaProcessor(fs, state.f0);
  const burstLength = original ? originalProcessor.delayLength : Math.floor(fs / state.f0);

  const burst = new Float64Array(burstLength);
  for (let i = 0; i < burstLength; i += 1) burst[i] = Math.random() * 2 - 1;

  const node = context.createScriptProcessor(512, 0, 1);
  const gain = context.createGain();
  gain.gain.value = 0.25;
  let index = 0;
  let alive = true;

  node.onaudioprocess = event => {
    const out = event.outputBuffer.getChannelData(0);
    for (let i = 0; i < out.length; i += 1) {
      const excitation = index < burstLength ? burst[index++] : 0;
      const sample = original
        ? originalProcessor.process(excitation)
        : extendedProcessor.process(excitation, {
            pluck: state.pluck,
            dynamic: state.dynamic,
            damping: state.damping,
            decay: state.decay,
          });
      out[i] = alive ? sample : 0;
    }
  };

  node.connect(gain).connect(analyser);
  voice = {
    stop() {
      if (!alive) return;
      alive = false;
      node.disconnect();
      gain.disconnect();
    },
  };
  const current = voice;
  setTimeout(() => {
    current.stop();
    if (voice === current) voice = null;
  }, 8000);
}

export function stopKsaFigure() {
  if (voice) voice.stop();
  voice = null;
}

/** Analytic magnitude in dB over the plotted frequency range. */
function analyticCurve(fs) {
  const out = new Float64Array(RESPONSE_POINTS);
  const top = Math.min(MAX_HZ, fs / 2);
  let peak = 1e-12;
  for (let i = 0; i < RESPONSE_POINTS; i += 1) {
    const hz = (i / (RESPONSE_POINTS - 1)) * top;
    const omega = (2 * Math.PI * hz) / fs;
    const h = state.mode === 'original'
      ? originalResponse(omega, {
          delayLength: OriginalKsaProcessor.delayLengthFor(state.f0, fs),
          decayGain: 0.996,
        })
      : extendedResponse(omega, { ...state, fs });
    out[i] = Math.hypot(h.re, h.im);
    if (out[i] > peak) peak = out[i];
  }
  for (let i = 0; i < RESPONSE_POINTS; i += 1) {
    out[i] = clamp(20 * Math.log10(Math.max(out[i], 1e-12) / peak), FLOOR_DB, 0);
  }
  return out;
}

function draw() {
  if (!ctx || !canvas) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
    canvas.width = width * dpr;
    canvas.height = height * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const fs = sampleRate();
  const top = Math.min(MAX_HZ, fs / 2);
  const padL = 46;
  const padR = 12;
  const padT = 14;
  const padB = 26;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const xOf = hz => padL + (hz / top) * plotW;
  const yOf = db => padT + (1 - (db - FLOOR_DB) / -FLOOR_DB) * plotH;

  // Grid
  ctx.strokeStyle = '#e6e9ec';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#8a929a';
  ctx.font = '10px system-ui, sans-serif';
  ctx.textAlign = 'right';
  for (let db = 0; db >= FLOOR_DB; db -= 20) {
    const y = yOf(db);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(width - padR, y);
    ctx.stroke();
    ctx.fillText(`${db}`, padL - 6, y + 3);
  }
  ctx.textAlign = 'center';
  const hzStep = top <= 4000 ? 500 : 1000;
  for (let hz = 0; hz <= top; hz += hzStep) {
    const x = xOf(hz);
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, height - padB);
    ctx.stroke();
    ctx.fillText(hz >= 1000 ? `${hz / 1000}k` : `${hz}`, x, height - padB + 14);
  }

  // Layer 2 (beneath): the measured spectrum, filled.
  if (analyser && voice) {
    analyser.getFloatFrequencyData(spectrumBuffer);
    const binHz = fs / analyser.fftSize;
    ctx.beginPath();
    ctx.moveTo(padL, height - padB);
    let peak = -Infinity;
    for (let i = 0; i < spectrumBuffer.length; i += 1) {
      if (spectrumBuffer[i] > peak) peak = spectrumBuffer[i];
    }
    for (let i = 0; i < spectrumBuffer.length; i += 1) {
      const hz = i * binHz;
      if (hz > top) break;
      const db = clamp(spectrumBuffer[i] - Math.max(peak, -100), FLOOR_DB, 0);
      ctx.lineTo(xOf(hz), yOf(db));
    }
    ctx.lineTo(width - padR, height - padB);
    ctx.closePath();
    ctx.fillStyle = 'rgba(111, 151, 178, 0.32)';
    ctx.fill();
  }

  // Layer 1 (on top): the analytic response.
  const curve = analyticCurve(fs);
  ctx.beginPath();
  for (let i = 0; i < curve.length; i += 1) {
    const hz = (i / (curve.length - 1)) * top;
    const x = xOf(hz);
    const y = yOf(curve[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = '#d55b37';
  ctx.lineWidth = 1.6;
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = '#5f6871';
  ctx.textAlign = 'left';
  ctx.fillText('dB', 8, padT + 4);
  ctx.textAlign = 'right';
  ctx.fillText('Hz', width - padR, height - 4);
}

function tick() {
  if (!running) return;
  draw();
  requestAnimationFrame(tick);
}

function updateReadouts() {
  if (!root) return;
  const fs = sampleRate();
  const n = OriginalKsaProcessor.delayLengthFor(state.f0, fs);
  const sounding = OriginalKsaProcessor.soundingFrequency(n, fs);
  const set = (name, value) => {
    const element = root.querySelector(`[data-readout="${name}"]`);
    if (element) element.textContent = value;
  };
  set('f0', `${state.f0.toFixed(1)} Hz`);
  set('delay', `${n} samples`);
  set('sounding', `${sounding.toFixed(2)} Hz`);
  const cents = centsError(state.f0, sounding);
  set('cents', `${cents >= 0 ? '+' : ''}${cents.toFixed(1)} cents`);
  const centsElement = root.querySelector('[data-readout="cents"]');
  if (centsElement) centsElement.classList.toggle('off-pitch', Math.abs(cents) > 5);
  set('decay', state.decay.toFixed(4));
  set('damping', state.damping.toFixed(3));
  set('pluck', state.pluck.toFixed(2));
  set('dynamic', state.dynamic.toFixed(2));
}

function setMode(mode) {
  if (mode === state.mode) return;
  state.mode = mode;
  root.dataset.mode = mode;
  root.querySelectorAll('[data-ksa-extended]').forEach(element =>
    element.toggleAttribute('hidden', mode !== 'extended'));
  updateReadouts();
}

export function initKsaFigure() {
  root = document.querySelector('[data-ksa-figure]');
  if (!root) return;
  canvas = root.querySelector('canvas');
  ctx = canvas.getContext('2d');
  root.dataset.mode = state.mode;

  // Build the context up front (it starts suspended, which needs no gesture) so
  // the analytic curve and the measured spectrum are computed at the same
  // sample rate from the very first frame rather than after the first pluck.
  try {
    ensureAudio();
  } catch (error) {
    console.warn('ksa figure: no AudioContext', error);
  }

  root.querySelectorAll('[data-ksa-param]').forEach(input => {
    const key = input.dataset.ksaParam;
    const apply = () => {
      const value = Number(input.value);
      if (key === 'delay') {
        // Two views of one quantity: dragging the integer delay sets the pitch.
        const fs = sampleRate();
        state.f0 = OriginalKsaProcessor.soundingFrequency(Math.round(value), fs);
        const f0Input = root.querySelector('[data-ksa-param="f0"]');
        if (f0Input) f0Input.value = state.f0;
      } else {
        state[key] = value;
        if (key === 'f0') {
          const fs = sampleRate();
          const delayInput = root.querySelector('[data-ksa-param="delay"]');
          if (delayInput) delayInput.value = OriginalKsaProcessor.delayLengthFor(value, fs);
        }
      }
      updateReadouts();
    };
    input.addEventListener('input', apply);
    apply();
  });

  root.querySelectorAll('[data-ksa-mode]').forEach(button =>
    button.addEventListener('click', () => setMode(button.dataset.ksaMode)));
  const pluckButton = root.querySelector('[data-action="pluck"]');
  if (pluckButton) pluckButton.addEventListener('click', () => pluck());

  const section = document.querySelector('[data-ksa-steps]');
  if (section) observeSteps(section, step => setMode(step === 'extended' ? 'extended' : 'original'));

  updateReadouts();
  running = true;
  requestAnimationFrame(tick);
}

export { state as ksaFigureState };
