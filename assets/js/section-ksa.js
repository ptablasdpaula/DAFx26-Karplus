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
import { observeSteps, whenVisible } from './scrollytelling.js';

const RESPONSE_POINTS = 900;
const FLOOR_DB = -84;
// The analyser reports dBFS; this is the level mapped to the top of the plot.
const MEASURED_REF_DB = -18;
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

/**
 * How hard the string was plucked, as a linear amplitude.
 *
 * The dynamics filter H_D models intensity as spectral tilt at constant DC
 * gain, which is the perceptually right thing for timbre but leaves the overall
 * level untouched. Physically a gentler pluck also puts less energy into the
 * string, so the excitation carries that. Applied to both the audio and the
 * analytic curve, so the two keep agreeing.
 */
const excitationGain = dynamic => dynamic;

// The frequency control is logarithmic, so a given drag distance is the same
// musical interval anywhere on the range.
const F0_MIN = 60;
const F0_MAX = 1200;
const sliderToF0 = v => F0_MIN * (F0_MAX / F0_MIN) ** v;
const f0ToSlider = f => Math.log(f / F0_MIN) / Math.log(F0_MAX / F0_MIN);

/**
 * In original mode the only reachable pitches are fs/(N + 1/2) for whole N, so
 * snap to the nearest one and push it back to the slider. The thumb then steps
 * between reachable values instead of gliding over unreachable ones — you can
 * feel the quantisation widen as the delay line gets shorter. The extended
 * algorithm interpolates, so there it stays continuous.
 */
function snapF0(f0, fs, mode) {
  if (mode !== 'original') return f0;
  return OriginalKsaProcessor.soundingFrequency(
    OriginalKsaProcessor.delayLengthFor(f0, fs), fs);
}

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

/**
 * A continuously replucked string.
 *
 * One persistent node rather than a voice per click: every control is read
 * inside the audio callback, so dragging a slider retunes or reshapes the
 * string as you move it. The burst is regenerated on each repluck, which is
 * also when a change of delay length or pluck strength takes hold — everything
 * else applies immediately.
 */
const REPLUCK_SECONDS = 1.5;

function startEngine() {
  const context = ensureAudio();
  if (context.state === 'suspended') context.resume();
  if (voice) return;

  const fs = context.sampleRate;
  const originalProcessor = new OriginalKsaProcessor(fs, state.f0);
  const extendedProcessor = new ExtendedKsaProcessor(fs, state.f0);

  let burst = new Float64Array(1);
  let index = 1;
  let countdown = 0;
  let alive = true;

  const repluck = () => {
    const original = state.mode === 'original';
    const length = original
      ? OriginalKsaProcessor.delayLengthFor(state.f0, fs)
      : Math.max(2, Math.floor(fs / state.f0));
    // Zero-mean, as the research code's no_dc_burst does: the loop passes DC
    // almost untouched, so any offset would accumulate.
    burst = new Float64Array(length);
    let mean = 0;
    for (let i = 0; i < length; i += 1) { burst[i] = Math.random() * 2 - 1; mean += burst[i]; }
    mean /= length;
    // A softer pluck is quieter as well as duller. H_D has unity DC gain, so on
    // its own it only tilts the spectrum; the level belongs to the excitation,
    // as it does in the synthesiser's burst_gain.
    const level = original ? 1 : excitationGain(state.dynamic);
    for (let i = 0; i < length; i += 1) burst[i] = (burst[i] - mean) * level;
    index = 0;
    countdown = Math.round(REPLUCK_SECONDS * fs);
  };
  repluck();

  const node = context.createScriptProcessor(512, 0, 1);
  const gain = context.createGain();
  gain.gain.value = 0.25;

  node.onaudioprocess = event => {
    const out = event.outputBuffer.getChannelData(0);
    for (let i = 0; i < out.length; i += 1) {
      if (countdown-- <= 0) repluck();
      const excitation = index < burst.length ? burst[index++] : 0;
      const sample = state.mode === 'original'
        ? originalProcessor.process(excitation, {
            delayLength: OriginalKsaProcessor.delayLengthFor(state.f0, fs),
          })
        : extendedProcessor.process(excitation, {
            f0: state.f0,
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
}

/** Toggle the string on or off. Returns the new state. */
export function toggleKsaSound() {
  if (voice) {
    stopKsaFigure();
    return false;
  }
  startEngine();
  return true;
}

export function stopKsaFigure() {
  if (voice) voice.stop();
  voice = null;
}

/**
 * Analytic magnitude in dB over the plotted frequency range.
 *
 * Shape comes from the transfer function; the overall level comes from the
 * pluck strength, so a gentler pluck moves the whole curve down rather than
 * only tilting it. Exported for tests/page-check.html.
 */
export function responseCurveDb(config, fs, points = RESPONSE_POINTS) {
  const out = new Float64Array(points);
  const top = Math.min(MAX_HZ, fs / 2);
  let peak = 1e-12;
  for (let i = 0; i < points; i += 1) {
    const hz = (i / (points - 1)) * top;
    const omega = (2 * Math.PI * hz) / fs;
    const h = config.mode === 'original'
      ? originalResponse(omega, {
          delayLength: OriginalKsaProcessor.delayLengthFor(config.f0, fs),
        })
      : extendedResponse(omega, { ...config, fs });
    out[i] = Math.hypot(h.re, h.im);
    if (out[i] > peak) peak = out[i];
  }
  const levelDb = config.mode === 'original'
    ? 0
    : 20 * Math.log10(Math.max(excitationGain(config.dynamic), 1e-6));
  for (let i = 0; i < points; i += 1) {
    out[i] = clamp(20 * Math.log10(Math.max(out[i], 1e-12) / peak) + levelDb, FLOOR_DB, 0);
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
    // Fixed reference, not a per-frame peak: normalising each frame would hide
    // both the decay we are here to show and any change in pluck strength.
    for (let i = 0; i < spectrumBuffer.length; i += 1) {
      const hz = i * binHz;
      if (hz > top) break;
      const db = clamp(spectrumBuffer[i] - MEASURED_REF_DB, FLOOR_DB, 0);
      ctx.lineTo(xOf(hz), yOf(db));
    }
    ctx.lineTo(width - padR, height - padB);
    ctx.closePath();
    ctx.fillStyle = 'rgba(111, 151, 178, 0.32)';
    ctx.fill();
  }

  // Layer 1 (on top): the analytic response.
  const curve = responseCurveDb(state, fs);
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

  // Show the arithmetic, not just the answer: sample rate divided by the
  // requested pitch, rounded to whole samples, and the pitch that really comes
  // back out. The gap between the two is the whole point of Section 1.1.
  const requested = state.f0Requested ?? state.f0;
  const exact = fs / requested;
  const cents = centsError(requested, sounding);
  const substitution = root.querySelector('[data-readout="substitution"]');
  if (substitution) {
    const sign = cents >= 0 ? '+' : '';
    const off = Math.abs(cents) > 5 ? ' off-pitch' : '';
    substitution.innerHTML = state.mode === 'original'
      ? `<span class="sub-line">${fs} &divide; ${requested.toFixed(1)} = ` +
        `${exact.toFixed(2)} &rarr; rounded to <b>${n} samples</b></span>` +
        `<span class="sub-line">sounds at <b>${sounding.toFixed(2)} Hz</b>` +
        `<b class="cents${off}">${sign}${cents.toFixed(1)} cents</b></span>`
      : `<span class="sub-line">${fs} &divide; ${state.f0.toFixed(1)} = ` +
        `<b>${exact.toFixed(2)} samples</b>, fractional delay and all</span>` +
        `<span class="sub-line">sounds at <b>${state.f0.toFixed(2)} Hz</b>` +
        `<b class="cents">exact</b></span>`;
  }
  set('decay', state.decay.toFixed(4));
  set('damping', state.damping.toFixed(3));
  set('pluck', state.pluck.toFixed(2));
  set('dynamic', state.dynamic.toFixed(2));
}

const applyHandlers = [];

function setMode(mode) {
  if (mode === state.mode || !root) return;
  state.mode = mode;
  // Re-run the controls so the frequency snaps on entering original mode and
  // relaxes on leaving it.
  setTimeout(() => applyHandlers.forEach(fn => fn()), 0);
  root.dataset.mode = mode;
  // Scrolling drives this too, so the controls have to follow — otherwise the
  // toggle claims one algorithm while the curve shows the other.
  root.querySelectorAll('[data-ksa-mode]').forEach(button =>
    button.classList.toggle('active', button.dataset.ksaMode === mode));
  root.querySelectorAll('[data-ksa-extended]').forEach(element =>
    element.toggleAttribute('hidden', mode !== 'extended'));
  root.querySelectorAll('[data-diagram]').forEach(figure =>
    figure.toggleAttribute('hidden', figure.dataset.diagram !== mode));
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
      if (key === 'f0') {
        const requested = sliderToF0(Number(input.value));
        const snapped = snapF0(requested, sampleRate(), state.mode);
        state.f0 = snapped;
        state.f0Requested = requested;
        // Write the snapped value back so the thumb visibly lands on it.
        if (state.mode === 'original') input.value = f0ToSlider(snapped);
      } else {
        state[key] = Number(input.value);
      }
      updateReadouts();
    };
    input.addEventListener('input', apply);
    input.dataset.apply = '';
    applyHandlers.push(apply);
    apply();
  });

  root.querySelectorAll('[data-ksa-mode]').forEach(button =>
    button.addEventListener('click', () => setMode(button.dataset.ksaMode)));
  const soundButton = root.querySelector('[data-action="sound"]');
  if (soundButton) {
    soundButton.addEventListener('click', () => {
      const on = toggleKsaSound();
      soundButton.setAttribute('aria-pressed', String(on));
      soundButton.classList.toggle('btn-dark', on);
      soundButton.classList.toggle('btn-outline-dark', !on);
      soundButton.innerHTML = on
        ? '<i class="bi bi-volume-up"></i> Sound on'
        : '<i class="bi bi-volume-mute"></i> Sound off';
    });
  }

  const section = document.querySelector('[data-ksa-steps]');
  if (section) observeSteps(section, step => setMode(step === 'extended' ? 'extended' : 'original'));

  // Never leave a string ringing in a figure the reader has scrolled past.
  const idle = new IntersectionObserver(entries => {
    if (entries.every(entry => !entry.isIntersecting)) stopKsaFigure();
  }, { threshold: 0.05 });
  idle.observe(root);

  updateReadouts();
  running = true;
  requestAnimationFrame(tick);
}

export { state as ksaFigureState };
