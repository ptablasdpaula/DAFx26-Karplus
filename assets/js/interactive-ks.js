let pauseExamples = () => {};

const SAMPLE_F0_MIN = 55;
const STEPS = 64;
const FIRST_MIDI = 48;
const LAST_MIDI = 60;
const KNOB_FULL_SCALE_PIXELS = 320;
const FINE_KNOB_FULL_SCALE_PIXELS = 1000;
const PROCESS_BUFFER_SIZE = 1024;

const KS_PARAMS = [
  { id: 'f0', label: 'Fundamental frequency', min: 110, max: 880, value: 440, decimals: 0, unit: ' Hz' },
  { id: 'dynamic', label: 'Dynamics', min: 0, max: 1, value: 0.62, decimals: 2 },
  { id: 'pluck', label: 'Pluck position', min: 0.01, max: 0.5, value: 0.22, decimals: 2 },
  { id: 'a1', label: 'Damping', min: 0.00001, max: 1, value: 0.48, decimals: 5, scale: 'log' },
  { id: 'decay', label: 'Decay', min: 0.00001, max: 0.99999, value: 0.985, decimals: 5, scale: 'antilog' }
];

const COMPONENT_ARROWS = {
  f0: 'M190 110 C270 120 280 130 360 160',
  dynamic: 'M810 122 C720 125 705 138 640 170',
  pluck: 'M185 380 C278 345 300 330 388 305',
  a1: 'M810 388 C725 360 705 345 622 326',
  decay: 'M500 630 C500 555 506 502 510 432'
};

const paramState = Object.fromEntries(KS_PARAMS.map(param => [param.id, { base: param.value, live: param.value }]));
const lfos = [
  { id: 'lfo1', label: 'LFO 1', freq: 0.35, amp: 0 },
  { id: 'lfo2', label: 'LFO 2', freq: 0.9, amp: 0 }
];
const modConnections = new Map(KS_PARAMS.map(param => [param.id, new Set()]));
const midiNotes = Array.from({ length: LAST_MIDI - FIRST_MIDI + 1 }, (_, i) => LAST_MIDI - i);
const pattern = new Map(midiNotes.map(note => [note, new Set()]));

[
  [48, 0], [52, 4], [55, 8], [60, 12],
  [50, 16], [53, 20], [57, 24], [60, 28],
  [52, 32], [55, 36], [59, 40], [60, 44],
  [48, 48], [55, 52], [57, 56], [52, 60]
].forEach(([note, step]) => pattern.get(note)?.add(step));

let audioContext = null;
let synthNode = null;
let synthOutputGain = null;
let synthState = null;
let sequenceTimer = null;
let currentStep = 0;
let draggingCable = null;
let animationFrame = null;

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const normalize = (value, min, max) => (value - min) / (max - min);
const valueToNorm = (value, param) => {
  const clipped = clamp(value, param.min, param.max);
  if (param.scale === 'log') {
    return Math.log(clipped / param.min) / Math.log(param.max / param.min);
  }
  if (param.scale === 'antilog') {
    const complement = param.max + param.min - clipped;
    return 1 - Math.log(complement / param.min) / Math.log(param.max / param.min);
  }
  return normalize(clipped, param.min, param.max);
};
const normToValue = (norm, param) => {
  const clipped = clamp(norm, 0, 1);
  if (param.scale === 'log') {
    return param.min * (param.max / param.min) ** clipped;
  }
  if (param.scale === 'antilog') {
    return param.max + param.min - param.min * (param.max / param.min) ** (1 - clipped);
  }
  return param.min + clipped * (param.max - param.min);
};
const midiToFreq = midi => 440 * 2 ** ((midi - 69) / 12);
const noteName = midi => {
  const names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  return `${names[midi % 12]}${Math.floor(midi / 12) - 1}`;
};

export function initInteractiveKarplus({ pauseExamples: pauseExamplesCallback } = {}) {
  pauseExamples = pauseExamplesCallback || (() => {});
  buildPatchControls();
  buildLfos();
  buildPianoRoll();
  drawComponentArrows();
  updatePatchCables();
  document.getElementById('ks-play').addEventListener('click', startSequence);
  document.getElementById('ks-stop').addEventListener('click', stopSequence);
  document.getElementById('ks-tempo').addEventListener('change', () => {
    if (!sequenceTimer) return;
    stopSequence();
    startSequence();
  });
  window.addEventListener('resize', updatePatchCables);
  startSynthAnimation();
}

function buildPatchControls() {
  const holder = document.getElementById('ks-knobs');
  holder.innerHTML = '';
  KS_PARAMS.forEach(param => {
    const node = document.createElement('div');
    node.className = 'patch-node';
    node.dataset.paramNode = param.id;
    node.innerHTML = `
      <div class="knob-card">
        <div class="knob-head"><span>${param.label}</span><span class="knob-readout" data-readout="${param.id}"></span></div>
        <button class="knob-shell" type="button" data-knob="${param.id}" aria-label="${param.label}">
          <span class="knob-indicator"></span>
        </button>
        <button class="mod-jack" type="button" data-mod-input="${param.id}" aria-label="${param.label} modulation input">in</button>
      </div>`;
    holder.appendChild(node);
  });
  document.querySelectorAll('[data-knob]').forEach(attachKnobGestures);
}

function attachKnobGestures(knob) {
  const param = KS_PARAMS.find(item => item.id === knob.dataset.knob);
  let drag = null;

  knob.addEventListener('pointerdown', event => {
    event.preventDefault();
    knob.setPointerCapture(event.pointerId);
    drag = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      startNorm: valueToNorm(paramState[param.id].base, param)
    };
  });

  knob.addEventListener('pointermove', event => {
    if (!drag || drag.pointerId !== event.pointerId) return;
    const fullScale = event.shiftKey ? FINE_KNOB_FULL_SCALE_PIXELS : KNOB_FULL_SCALE_PIXELS;
    const motion = (drag.startY - event.clientY) + (event.clientX - drag.startX) * 0.45;
    setParamBase(param.id, normToValue(drag.startNorm + motion / fullScale, param));
  });

  knob.addEventListener('pointerup', event => {
    if (drag && drag.pointerId === event.pointerId) drag = null;
  });

  knob.addEventListener('pointercancel', () => {
    drag = null;
  });

  knob.addEventListener('wheel', event => {
    event.preventDefault();
    const fullScale = event.shiftKey ? FINE_KNOB_FULL_SCALE_PIXELS : KNOB_FULL_SCALE_PIXELS;
    setParamBase(param.id, normToValue(valueToNorm(paramState[param.id].base, param) - event.deltaY / fullScale, param));
  }, { passive: false });

  knob.addEventListener('dblclick', () => {
    setParamBase(param.id, param.value);
  });
}

function setParamBase(paramId, value) {
  const param = KS_PARAMS.find(item => item.id === paramId);
  paramState[paramId].base = clamp(value, param.min, param.max);
  updateSynthReadouts(performance.now() / 1000);
}

function buildLfos() {
  const holder = document.getElementById('lfo-grid');
  holder.innerHTML = '';
  lfos.forEach(lfo => {
    const card = document.createElement('div');
    card.className = 'lfo-card';
    card.innerHTML = `
      <h3 style="font-size:1rem;margin:0 0 0.75rem">${lfo.label}</h3>
      <button class="lfo-output" type="button" data-lfo-output="${lfo.id}" aria-label="${lfo.label} output">out</button>
      <label class="lfo-row">Rate <input type="range" min="0.05" max="8" step="0.01" value="${lfo.freq}" data-lfo-rate="${lfo.id}"><span data-lfo-rate-readout="${lfo.id}"></span></label>
      <label class="lfo-row">Amount <input type="range" min="0" max="1" step="0.01" value="${lfo.amp}" data-lfo-amp="${lfo.id}"><span data-lfo-amp-readout="${lfo.id}"></span></label>`;
    holder.appendChild(card);
  });
  document.querySelectorAll('[data-lfo-rate]').forEach(input => {
    input.addEventListener('input', () => {
      lfos.find(lfo => lfo.id === input.dataset.lfoRate).freq = Number(input.value);
    });
  });
  document.querySelectorAll('[data-lfo-amp]').forEach(input => {
    input.addEventListener('input', () => {
      lfos.find(lfo => lfo.id === input.dataset.lfoAmp).amp = Number(input.value);
    });
  });
  document.querySelectorAll('[data-lfo-output]').forEach(output => {
    output.addEventListener('pointerdown', event => startCableDrag(event, output.dataset.lfoOutput, output));
  });
}

function buildPianoRoll() {
  const grid = document.getElementById('roll-grid');
  grid.innerHTML = '';
  midiNotes.forEach(note => {
    const label = document.createElement('div');
    label.className = 'note-label';
    label.textContent = noteName(note);
    grid.appendChild(label);
    for (let step = 0; step < STEPS; step += 1) {
      const cell = document.createElement('button');
      cell.type = 'button';
      cell.className = `roll-cell${step % 16 === 0 ? ' barline' : ''}${step % 4 === 0 ? ' beat' : ''}`;
      cell.dataset.note = String(note);
      cell.dataset.step = String(step);
      cell.classList.toggle('active', pattern.get(note).has(step));
      cell.addEventListener('click', () => {
        const notes = pattern.get(note);
        if (notes.has(step)) notes.delete(step);
        else notes.add(step);
        cell.classList.toggle('active', notes.has(step));
      });
      grid.appendChild(cell);
    }
  });
}

function drawComponentArrows() {
  const svg = document.getElementById('component-arrows');
  if (!svg) return;
  Array.from(svg.querySelectorAll('path[data-arrow]')).forEach(path => path.remove());
  Object.entries(COMPONENT_ARROWS).forEach(([paramId, d]) => {
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.dataset.arrow = paramId;
    path.setAttribute('d', d);
    svg.appendChild(path);
  });
}

function startCableDrag(event, lfoId, output) {
  event.preventDefault();
  output.setPointerCapture(event.pointerId);
  draggingCable = { lfoId, output, pointerId: event.pointerId, x: event.clientX, y: event.clientY };
  output.addEventListener('pointermove', moveCableDrag);
  output.addEventListener('pointerup', finishCableDrag);
  output.addEventListener('pointercancel', finishCableDrag);
  updatePatchCables();
}

function moveCableDrag(event) {
  if (!draggingCable || draggingCable.pointerId !== event.pointerId) return;
  draggingCable.x = event.clientX;
  draggingCable.y = event.clientY;
  updatePatchCables();
}

function finishCableDrag(event) {
  if (!draggingCable || draggingCable.pointerId !== event.pointerId) return;
  const dropTarget = document.elementFromPoint(event.clientX, event.clientY)?.closest('[data-mod-input]');
  if (dropTarget) toggleConnection(draggingCable.lfoId, dropTarget.dataset.modInput);
  endCableDrag();
}

function endCableDrag() {
  if (draggingCable) {
    draggingCable.output.removeEventListener('pointermove', moveCableDrag);
    draggingCable.output.removeEventListener('pointerup', finishCableDrag);
    draggingCable.output.removeEventListener('pointercancel', finishCableDrag);
  }
  draggingCable = null;
  updatePatchCables();
}

function toggleConnection(lfoId, paramId) {
  const targets = modConnections.get(paramId);
  if (targets.has(lfoId)) targets.delete(lfoId);
  else targets.add(lfoId);
  updatePatchCables();
}

function updatePatchCables() {
  const svg = document.getElementById('patch-cables');
  const stage = document.getElementById('patch-stage');
  if (!svg || !stage) return;
  const stageBox = stage.getBoundingClientRect();
  svg.setAttribute('viewBox', `0 0 ${stageBox.width} ${stageBox.height}`);
  svg.innerHTML = '';
  modConnections.forEach((lfoIds, paramId) => {
    lfoIds.forEach(lfoId => {
      const start = jackCenter(document.querySelector(`[data-lfo-output="${lfoId}"]`), stageBox);
      const end = jackCenter(document.querySelector(`[data-mod-input="${paramId}"]`), stageBox);
      drawCable(svg, start, end);
    });
  });
  if (draggingCable) {
    const start = jackCenter(draggingCable.output, stageBox);
    const end = { x: draggingCable.x - stageBox.left, y: draggingCable.y - stageBox.top };
    drawCable(svg, start, end);
  }
  document.querySelectorAll('[data-mod-input]').forEach(input => {
    input.classList.toggle('connected', modConnections.get(input.dataset.modInput).size > 0);
  });
  document.querySelectorAll('[data-lfo-output]').forEach(output => {
    const lfoId = output.dataset.lfoOutput;
    output.classList.toggle('connected', Array.from(modConnections.values()).some(targets => targets.has(lfoId)));
  });
}

function jackCenter(element, stageBox) {
  const box = element.getBoundingClientRect();
  return {
    x: box.left + box.width / 2 - stageBox.left,
    y: box.top + box.height / 2 - stageBox.top
  };
}

function drawCable(svg, start, end) {
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  const dx = Math.max(80, Math.abs(end.x - start.x) * 0.45);
  path.setAttribute('d', `M${start.x},${start.y} C${start.x + dx},${start.y} ${end.x - dx},${end.y} ${end.x},${end.y}`);
  svg.appendChild(path);
}

function lfoValueFor(param, timeSeconds) {
  const definition = KS_PARAMS.find(item => item.id === param);
  let value = paramState[param].base;
  modConnections.get(param).forEach(lfoId => {
    const lfo = lfos.find(item => item.id === lfoId);
    if (!lfo || lfo.amp === 0) return;
    value += Math.sin(2 * Math.PI * lfo.freq * timeSeconds) * lfo.amp * (definition.max - definition.min) * 0.5;
  });
  return clamp(value, definition.min, definition.max);
}

function updateSynthReadouts(timeSeconds = performance.now() / 1000) {
  KS_PARAMS.forEach(param => {
    const base = paramState[param.id].base;
    const live = lfoValueFor(param.id, timeSeconds);
    paramState[param.id].live = live;
    const baseNorm = valueToNorm(base, param);
    const shell = document.querySelector(`[data-knob="${param.id}"]`);
    const readout = document.querySelector(`[data-readout="${param.id}"]`);
    if (shell) {
      shell.style.setProperty('--base', `${baseNorm * 270}deg`);
      shell.style.setProperty('--angle', `${-135 + baseNorm * 270}deg`);
    }
    if (readout) readout.textContent = `${live.toFixed(param.decimals)}${param.unit || ''}`;
  });
  lfos.forEach(lfo => {
    const rate = document.querySelector(`[data-lfo-rate-readout="${lfo.id}"]`);
    const amp = document.querySelector(`[data-lfo-amp-readout="${lfo.id}"]`);
    if (rate) rate.textContent = `${lfo.freq.toFixed(2)} Hz`;
    if (amp) amp.textContent = lfo.amp.toFixed(2);
  });
  updatePatchCables();
}

function startSynthAnimation() {
  if (animationFrame) return;
  const tick = () => {
    updateSynthReadouts();
    animationFrame = requestAnimationFrame(tick);
  };
  tick();
}

function computeDynamicsR(f0, dynamicLevel, fs) {
  const minBw = SAMPLE_F0_MIN;
  const maxBw = fs / 2;
  const bwHz = minBw * (maxBw / minBw) ** dynamicLevel;
  const fm = Math.sqrt(minBw * maxBw);
  const Ts = 1 / fs;
  const rL = Math.exp(-bwHz * Math.PI * Ts);
  const cosFm = Math.cos(2 * Math.PI * fm * Ts);
  const sinFm = Math.sin(2 * Math.PI * fm * Ts);
  const denom = Math.hypot(1 - rL * cosFm, rL * sinFm);
  const gL = (1 - rL) / Math.max(denom, 1e-9);
  const left = (1 - gL ** 2 * Math.cos(2 * Math.PI * f0 * Ts)) / Math.max(1 - gL ** 2, 1e-9);
  const right = 2 * gL * Math.sin(Math.PI * f0 * Ts) * (Math.sqrt(Math.max(0, 1 - gL ** 2 * Math.cos(Math.PI * f0 * Ts) ** 2)) / Math.max(1 - gL ** 2, 1e-9));
  const rPlus = left + right;
  const rMinus = left - right;
  return Math.abs(rPlus) < 1 ? rPlus : rMinus;
}

function onePolePhaseDelay(f0, a1, fs) {
  const omega = 2 * Math.PI * f0 / fs;
  const denomReal = 1 - a1 * Math.cos(omega);
  const denomImag = a1 * Math.sin(omega);
  return -Math.atan2(denomImag, denomReal) / omega;
}

function initContinuousSynth() {
  if (synthNode) return;
  const fs = audioContext.sampleRate;
  synthState = {
    fs,
    delayBuffer: new Float32Array(Math.ceil(fs / SAMPLE_F0_MIN) + 8),
    excitationDelay: new Float32Array(Math.ceil(fs / SAMPLE_F0_MIN) + 8),
    writeIdx: 0,
    excitationWriteIdx: 0,
    filterState: 0,
    dynamicsState: 0,
    pendingBursts: [],
    activeBursts: [],
    outputDc: 0,
    noteCenterFreq: 440,
    targetNoteCenterFreq: 440
  };
  synthNode = audioContext.createScriptProcessor(PROCESS_BUFFER_SIZE, 0, 1);
  synthOutputGain = audioContext.createGain();
  synthOutputGain.gain.value = 0.42;
  synthNode.onaudioprocess = processContinuousSynth;
  synthNode.connect(synthOutputGain).connect(audioContext.destination);
}

function triggerKsNote(midi, triggerTimeSeconds) {
  if (!synthState) return;
  const values = Object.fromEntries(KS_PARAMS.map(param => [param.id, lfoValueFor(param.id, performance.now() / 1000)]));
  const noteFreq = midiToFreq(midi);
  const freq = clamp(noteFreq * (values.f0 / 440), 40, 5000);
  const sampleTime = Math.max(0, Math.floor(triggerTimeSeconds * synthState.fs));
  const burstLength = Math.max(2, Math.floor(synthState.fs / freq));
  synthState.pendingBursts.push({ sampleTime, burstLength, index: 0, seed: Math.random() * 2 ** 31, noteFreq });
}

function processContinuousSynth(event) {
  const out = event.outputBuffer.getChannelData(0);
  if (!synthState) {
    out.fill(0);
    return;
  }
  const state = synthState;
  const fs = state.fs;
  const now = audioContext.currentTime;
  for (let n = 0; n < out.length; n += 1) {
    const absoluteSample = Math.floor((now + n / fs) * fs);
    while (state.pendingBursts.length && state.pendingBursts[0].sampleTime <= absoluteSample) {
      const burst = state.pendingBursts.shift();
      state.targetNoteCenterFreq = burst.noteFreq;
      state.activeBursts.push(burst);
    }

    const values = Object.fromEntries(KS_PARAMS.map(param => [param.id, lfoValueFor(param.id, (performance.now() / 1000) + n / fs)]));
    state.noteCenterFreq += 0.0015 * (state.targetNoteCenterFreq - state.noteCenterFreq);
    const freq = clamp(state.noteCenterFreq * (values.f0 / 440), 40, 5000);
    let excitation = 0;
    for (let i = state.activeBursts.length - 1; i >= 0; i -= 1) {
      const burst = state.activeBursts[i];
      if (burst.index >= burst.burstLength) {
        state.activeBursts.splice(i, 1);
        continue;
      }
      excitation += seededNoise(burst.seed + burst.index) * 0.85;
      burst.index += 1;
    }

    const combDelay = clamp((fs / freq) * values.pluck, 1, state.excitationDelay.length - 2);
    const combed = excitation - readDelay(state.excitationDelay, state.excitationWriteIdx, combDelay);
    state.excitationDelay[state.excitationWriteIdx] = excitation;
    state.excitationWriteIdx = (state.excitationWriteIdx + 1) % state.excitationDelay.length;

    const r = clamp(computeDynamicsR(freq, values.dynamic, fs), -0.999, 0.999);
    const shaped = (1 - r) * combed + r * state.dynamicsState;
    state.dynamicsState = shaped;

    const loopDelay = clamp(fs / freq + onePolePhaseDelay(freq, values.a1, fs), 2, state.delayBuffer.length - 4);
    const delayed = readDelay(state.delayBuffer, state.writeIdx, loopDelay);
    const filtered = values.decay * (1 - values.a1) * delayed + values.a1 * state.filterState;
    state.filterState = filtered;

    const sample = shaped + filtered;
    state.delayBuffer[state.writeIdx] = sample;
    state.writeIdx = (state.writeIdx + 1) % state.delayBuffer.length;
    state.outputDc = 0.995 * state.outputDc + 0.005 * sample;
    out[n] = Math.tanh((sample - state.outputDc) * 0.9);
  }
}

function readDelay(buffer, writeIdx, delay) {
  const read = writeIdx - delay;
  const wrapped = read < 0 ? read + buffer.length : read;
  const i0 = Math.floor(wrapped) % buffer.length;
  const i1 = (i0 + 1) % buffer.length;
  const frac = wrapped - Math.floor(wrapped);
  return buffer[i0] * (1 - frac) + buffer[i1] * frac;
}

function seededNoise(seed) {
  const x = Math.sin(seed * 12.9898) * 43758.5453;
  return 2 * (x - Math.floor(x)) - 1;
}

function clearPlayhead() {
  document.querySelectorAll('.roll-cell.playhead').forEach(cell => cell.classList.remove('playhead'));
}

function setPlayhead(step) {
  clearPlayhead();
  document.querySelectorAll(`.roll-cell[data-step="${step}"]`).forEach(cell => cell.classList.add('playhead'));
  const bar = Math.floor(step / 16) + 1;
  const beat = Math.floor((step % 16) / 4) + 1;
  const sixteenth = (step % 4) + 1;
  document.getElementById('ks-step-readout').textContent = `${bar}.${beat}.${sixteenth}`;
}

export function stopSequence() {
  if (sequenceTimer) {
    clearInterval(sequenceTimer);
    sequenceTimer = null;
  }
  if (synthState) {
    synthState.pendingBursts = [];
    synthState.activeBursts = [];
    synthState.delayBuffer.fill(0);
    synthState.excitationDelay.fill(0);
    synthState.filterState = 0;
    synthState.dynamicsState = 0;
    synthState.outputDc = 0;
    synthState.noteCenterFreq = 440;
    synthState.targetNoteCenterFreq = 440;
  }
  clearPlayhead();
  const playButton = document.getElementById('ks-play');
  if (playButton) playButton.innerHTML = '<i class="bi bi-play-fill"></i> Play';
}

async function startSequence() {
  pauseExamples();
  if (!audioContext) audioContext = new AudioContext();
  if (audioContext.state === 'suspended') await audioContext.resume();
  initContinuousSynth();
  if (sequenceTimer) {
    stopSequence();
    return;
  }
  currentStep = 0;
  const playButton = document.getElementById('ks-play');
  playButton.innerHTML = '<i class="bi bi-pause-fill"></i> Pause';
  const tick = () => {
    const bpm = clamp(Number(document.getElementById('ks-tempo').value) || 110, 60, 180);
    setPlayhead(currentStep);
    midiNotes.forEach(note => {
      if (pattern.get(note).has(currentStep)) triggerKsNote(note, audioContext.currentTime + 0.005);
    });
    currentStep = (currentStep + 1) % STEPS;
  };
  tick();
  sequenceTimer = setInterval(tick, 60 / (clamp(Number(document.getElementById('ks-tempo').value) || 110, 60, 180) * 4) * 1000);
}
