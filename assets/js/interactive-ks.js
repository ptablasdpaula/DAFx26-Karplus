let pauseExamples = () => {};

const STEPS = 32;
const FIRST_MIDI = 48;
const LAST_MIDI = 60;
const STEP_WIDTH = 22;
const KSA_F0_MIN = 20;
const LAGRANGE_ORDER = 5;
const controllers = [];
let audioContext;
let lastActiveController = null;
let extendedParams = { dynamic: 0.62, pluck: 0.22, decay: 0.994, damping: 0.34, noteOn: true, noteOff: true };

// Two-bar, one-octave monophonic adaptation of the climactic rolled-chord
// contour from Debussy's Clair de lune.
const clairContour = [56, 53, 49, 53, 56, 60, 56, 53, 58, 54, 51, 54, 58, 60, 58, 54];
const defaultNotes = clairContour.map((midi, index) => ({
  midi,
  start: index * 2,
  length: 2,
  velocity: 0.5 + 0.45 * Math.sin((index / (clairContour.length - 1)) * Math.PI)
}));

const midiToFreq = midi => 440 * 2 ** ((midi - 69) / 12);
const noteName = midi => `${['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][midi % 12]}${Math.floor(midi / 12) - 1}`;
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const modulo = (value, length) => ((value % length) + length) % length;

function computeDynamicsR(f0, dynamicLevel, fs) {
  const minBw = KSA_F0_MIN;
  const maxBw = fs / 2;
  const bwHz = minBw * (maxBw / minBw) ** dynamicLevel;
  const fm = Math.sqrt(minBw * maxBw);
  const samplePeriod = 1 / fs;
  const rL = Math.exp(-bwHz * Math.PI * samplePeriod);
  const real = 1 - rL * Math.cos(2 * Math.PI * fm * samplePeriod);
  const imag = rL * Math.sin(2 * Math.PI * fm * samplePeriod);
  const gL = (1 - rL) / Math.hypot(real, imag);
  const denominator = 1 - gL ** 2;
  const left = (1 - gL ** 2 * Math.cos(2 * Math.PI * f0 * samplePeriod)) / denominator;
  const right = 2 * gL * Math.sin(Math.PI * f0 * samplePeriod) * Math.sqrt(1 - gL ** 2 * Math.cos(Math.PI * f0 * samplePeriod) ** 2) / denominator;
  const plus = left + right;
  const minus = left - right;
  return Math.abs(plus) < 1 ? plus : minus;
}

function onePolePhaseDelay(f0, a1, fs) {
  const omega = 2 * Math.PI * f0 / fs;
  return -Math.atan2(a1 * Math.sin(omega), 1 - a1 * Math.cos(omega)) / omega;
}

function lagrangeFractionalDelay(totalDelay) {
  const offset = Math.floor(LAGRANGE_ORDER / 2);
  const adjusted = totalDelay - offset;
  const integerDelay = Math.floor(adjusted);
  const centeredFraction = adjusted - integerDelay + offset;
  const coefficients = new Float64Array(LAGRANGE_ORDER + 1);
  for (let n = 0; n <= LAGRANGE_ORDER; n += 1) {
    let coefficient = 1;
    for (let k = 0; k <= LAGRANGE_ORDER; k += 1) {
      if (k !== n) coefficient *= (centeredFraction - k) / (n - k);
    }
    coefficients[n] = coefficient;
  }
  return { integerDelay, coefficients };
}

export class ExtendedKsaProcessor {
  constructor(sampleRate, frequency) {
    this.fs = sampleRate;
    this.frequency = frequency;
    this.period = sampleRate / frequency;
    const maxDelay = Math.floor(sampleRate / KSA_F0_MIN);
    this.delay = new Float64Array(maxDelay);
    this.excitationDelay = new Float64Array(maxDelay);
    this.write = 0;
    this.excitationWrite = 0;
    this.loopFilterState = 0;
    this.dynamicsState = 0;
  }

  process(excitation, { pluck, dynamic, damping, decay }) {
    const combDelay = this.period * pluck;
    const combInteger = Math.floor(combDelay);
    const combFraction = combDelay - combInteger;
    const combIndex0 = modulo(this.excitationWrite - combInteger, this.excitationDelay.length);
    const combIndex1 = modulo(this.excitationWrite - combInteger - 1, this.excitationDelay.length);
    const delayedExcitation = (1 - combFraction) * this.excitationDelay[combIndex0] + combFraction * this.excitationDelay[combIndex1];
    this.excitationDelay[this.excitationWrite] = excitation;
    this.excitationWrite = (this.excitationWrite + 1) % this.excitationDelay.length;
    let shaped = excitation - delayedExcitation;
    const dynamicsR = computeDynamicsR(this.frequency, dynamic, this.fs);
    shaped = (1 - dynamicsR) * shaped + dynamicsR * this.dynamicsState;
    this.dynamicsState = shaped;

    const correctedDelay = this.period + onePolePhaseDelay(this.frequency, damping, this.fs);
    const { integerDelay, coefficients } = lagrangeFractionalDelay(correctedDelay);
    let delayedSample = 0;
    for (let k = 0; k <= LAGRANGE_ORDER; k += 1) {
      delayedSample += coefficients[k] * this.delay[modulo(this.write - integerDelay - k, this.delay.length)];
    }
    this.loopFilterState = decay * (1 - damping) * delayedSample + damping * this.loopFilterState;
    const output = shaped + this.loopFilterState;
    this.delay[this.write] = output;
    this.write = (this.write + 1) % this.delay.length;
    return output;
  }
}

export function initInteractiveKarplus({ pauseExamples: callback } = {}) {
  pauseExamples = callback || (() => {});
  document.querySelectorAll('[data-roll]').forEach(root => controllers.push(new PianoRoll(root, root.dataset.roll)));
  document.querySelectorAll('[data-ks-param]').forEach(input => {
    const physicalMin = Number(input.dataset.physicalMin);
    const physicalMax = Number(input.dataset.physicalMax);
    const scale = input.dataset.scale;
    const toPhysical = normalized => {
      if (scale === 'log') return physicalMin * (physicalMax / physicalMin) ** normalized;
      if (scale === 'reverse-log') return physicalMax + physicalMin - physicalMin * (physicalMax / physicalMin) ** (1 - normalized);
      return Number(normalized);
    };
    const toNormalized = physical => {
      if (scale === 'log') return Math.log(physical / physicalMin) / Math.log(physicalMax / physicalMin);
      if (scale === 'reverse-log') {
        const complement = physicalMax + physicalMin - physical;
        return 1 - Math.log(complement / physicalMin) / Math.log(physicalMax / physicalMin);
      }
      return physical;
    };
    if (input.dataset.initial) input.value = toNormalized(Number(input.dataset.initial));
    const update = () => {
      const value = toPhysical(Number(input.value));
      extendedParams[input.dataset.ksParam] = value;
      input.parentElement.querySelector('output').value = scale ? value.toFixed(5) : value.toFixed(2);
    };
    input.addEventListener('input', update);
    update();
  });
  document.querySelectorAll('[data-event-toggle]').forEach(input => input.addEventListener('change', () => {
    extendedParams[input.dataset.eventToggle === 'on' ? 'noteOn' : 'noteOff'] = input.checked;
  }));
  const observer = new IntersectionObserver(entries => entries.forEach(entry => {
    if (!entry.isIntersecting) controllers.find(controller => controller.root === entry.target)?.stop();
  }), { threshold: 0.02 });
  controllers.forEach(controller => observer.observe(controller.root));
  document.addEventListener('keydown', event => {
    if (event.metaKey && !event.shiftKey && event.key.toLowerCase() === 'z' && lastActiveController?.history.length) {
      event.preventDefault();
      lastActiveController.undo();
    }
    if ((event.key === 'Backspace' || event.key === 'Delete') && lastActiveController?.selectedNote) {
      event.preventDefault();
      lastActiveController.deleteSelected();
    }
  });
}

class PianoRoll {
  constructor(root, mode) {
    this.root = root;
    this.mode = mode;
    this.notes = defaultNotes.map(note => ({ ...note }));
    this.timer = null;
    this.step = 0;
    this.voices = [];
    this.gesture = null;
    this.history = [];
    this.selectedNote = null;
    this.build();
  }

  build() {
    this.root.innerHTML = `
      <div class="piano-toolbar">
        <div class="transport"><button class="btn btn-sm btn-dark" type="button" data-action="play"><i class="bi bi-play-fill"></i> Play</button><label class="tempo-control">Tempo <input class="form-control form-control-sm" type="number" min="60" max="180" value="110"></label></div>
        <div><span class="mode-badge">${this.mode === 'original' ? 'Original KSA' : 'Extended KSA'}</span><span data-readout>1.1.1</span></div>
      </div>
      <p class="roll-help">Drag empty space to draw · drag notes sideways for time or up/down for pitch · drag an edge to resize · double-click to delete${this.mode === 'extended' ? ' · <kbd>⌘</kbd>-drag vertically for velocity' : ''}</p>
      <div class="roll-grid"></div>`;
    this.grid = this.root.querySelector('.roll-grid');
    for (let midi = LAST_MIDI; midi >= FIRST_MIDI; midi -= 1) {
      const label = document.createElement('div');
      label.className = `note-label${[1,3,6,8,10].includes(midi % 12) ? ' black-key' : ''}`;
      label.textContent = noteName(midi);
      this.grid.appendChild(label);
      const lane = document.createElement('div');
      lane.className = 'roll-lane';
      lane.dataset.midi = midi;
      for (let beat = 0; beat <= STEPS; beat += 4) {
        const line = document.createElement('i');
        line.style.left = `${beat * STEP_WIDTH}px`;
        line.className = beat % 16 === 0 ? 'bar-line' : 'beat-line';
        lane.appendChild(line);
      }
      lane.addEventListener('pointerdown', event => this.beginDraw(event));
      this.grid.appendChild(lane);
    }
    this.root.querySelector('[data-action="play"]').addEventListener('click', () => this.play());
    this.renderNotes();
  }

  renderNotes() {
    this.root.querySelectorAll('.midi-note').forEach(element => element.remove());
    this.notes.forEach(note => {
      const lane = this.root.querySelector(`.roll-lane[data-midi="${note.midi}"]`);
      const block = document.createElement('div');
      block.className = 'midi-note';
      block.classList.toggle('selected', note === this.selectedNote);
      block.style.left = `${note.start * STEP_WIDTH + 1}px`;
      block.style.width = `${note.length * STEP_WIDTH - 2}px`;
      block.style.setProperty('--velocity', this.mode === 'extended' ? note.velocity : 0.72);
      block.title = this.mode === 'extended' ? `${noteName(note.midi)} · velocity ${Math.round(note.velocity * 127)}` : noteName(note.midi);
      block.innerHTML = `<span>${noteName(note.midi)}</span>${this.mode === 'extended' ? '<b class="velocity-meter"></b>' : ''}<b class="resize-handle left"></b><b class="resize-handle right"></b>`;
      block.addEventListener('pointerdown', event => this.beginEdit(event, note));
      block.addEventListener('dblclick', event => {
        event.stopPropagation();
        lastActiveController = this;
        this.recordHistory();
        this.notes = this.notes.filter(item => item !== note);
        this.selectedNote = null;
        this.renderNotes();
      });
      lane.appendChild(block);
    });
  }

  beginDraw(event) {
    if (event.target.closest('.midi-note')) return;
    lastActiveController = this;
    this.recordHistory();
    const start = clamp(Math.floor(event.offsetX / STEP_WIDTH), 0, STEPS - 1);
    const note = { midi: Number(event.currentTarget.dataset.midi), start, length: 1, velocity: 0.72 };
    this.notes.push(note);
    this.selectedNote = note;
    this.gesture = { kind: 'draw', note, originX: event.clientX };
    this.bindGesture();
    this.renderNotes();
  }

  beginEdit(event, note) {
    event.stopPropagation();
    lastActiveController = this;
    this.selectedNote = note;
    this.root.querySelectorAll('.midi-note.selected').forEach(block => block.classList.remove('selected'));
    event.currentTarget.classList.add('selected');
    const handle = event.target.closest('.resize-handle');
    const velocityEdit = this.mode === 'extended' && event.metaKey;
    this.gesture = { kind: velocityEdit ? 'velocity' : handle?.classList.contains('left') ? 'resize-left' : handle ? 'resize-right' : 'move', note, originX: event.clientX, originY: event.clientY, originStart: note.start, originLength: note.length, originMidi: note.midi, originVelocity: note.velocity, recorded: false };
    this.bindGesture();
  }

  bindGesture() {
    this.moveListener = event => this.continueGesture(event);
    this.upListener = () => this.endGesture();
    document.addEventListener('pointermove', this.moveListener);
    document.addEventListener('pointerup', this.upListener, { once: true });
  }

  continueGesture(event) {
    const gesture = this.gesture;
    if (!gesture) return;
    if (!gesture.recorded && gesture.kind !== 'draw') {
      this.recordHistory();
      gesture.recorded = true;
    }
    const delta = Math.round((event.clientX - gesture.originX) / STEP_WIDTH);
    const note = gesture.note;
    if (gesture.kind === 'draw') note.length = clamp(delta + 1, 1, STEPS - note.start);
    if (gesture.kind === 'move') {
      note.start = clamp(gesture.originStart + delta, 0, STEPS - note.length);
      note.midi = clamp(gesture.originMidi + Math.round((gesture.originY - event.clientY) / 22), FIRST_MIDI, LAST_MIDI);
    }
    if (gesture.kind === 'resize-right') note.length = clamp(gesture.originLength + delta, 1, STEPS - note.start);
    if (gesture.kind === 'resize-left') {
      const end = gesture.originStart + gesture.originLength;
      note.start = clamp(gesture.originStart + delta, 0, end - 1);
      note.length = end - note.start;
    }
    if (gesture.kind === 'velocity') note.velocity = clamp(gesture.originVelocity + (gesture.originY - event.clientY) / 120, 0.05, 1);
    this.renderNotes();
  }

  endGesture() {
    document.removeEventListener('pointermove', this.moveListener);
    this.gesture = null;
  }

  recordHistory() {
    this.history.push(this.notes.map(note => ({ ...note })));
    if (this.history.length > 50) this.history.shift();
  }

  undo() {
    const previous = this.history.pop();
    if (!previous) return;
    this.notes = previous.map(note => ({ ...note }));
    this.selectedNote = null;
    this.renderNotes();
  }

  deleteSelected() {
    if (!this.selectedNote) return;
    this.recordHistory();
    this.notes = this.notes.filter(note => note !== this.selectedNote);
    this.selectedNote = null;
    this.renderNotes();
  }

  async play() {
    lastActiveController = this;
    pauseExamples();
    controllers.filter(controller => controller !== this).forEach(controller => controller.stop());
    if (!audioContext) audioContext = new AudioContext();
    if (audioContext.state === 'suspended') await audioContext.resume();
    if (this.timer) { this.stop(); return; }
    this.step = 0;
    this.root.querySelector('[data-action="play"]').innerHTML = '<i class="bi bi-pause-fill"></i> Pause';
    const tick = () => {
      const bpm = clamp(Number(this.root.querySelector('input[type="number"]').value) || 110, 60, 180);
      const secondsPerStep = 60 / (bpm * 4);
      this.setPlayhead(this.step);
      const note = this.notes.find(item => item.start === this.step);
      if (note && (this.mode === 'original' || extendedParams.noteOn)) {
        this.voices.forEach(voice => voice.stop());
        this.voices = [];
        this.trigger(note, note.length * secondsPerStep);
      }
      this.step = (this.step + 1) % STEPS;
      this.timer = setTimeout(tick, secondsPerStep * 1000);
    };
    tick();
  }

  trigger(note, duration) {
    const fs = audioContext.sampleRate;
    const frequency = midiToFreq(note.midi);
    const period = fs / frequency;
    const maxDelay = Math.floor(fs / KSA_F0_MIN);
    const delay = new Float64Array(maxDelay);
    let write = 0, originalPrevious = 0, alive = true, off = false;
    const extendedProcessor = new ExtendedKsaProcessor(fs, frequency);
    const processor = audioContext.createScriptProcessor(512, 0, 1);
    const gain = audioContext.createGain();
    const burstLength = Math.floor(period);
    const burst = new Float64Array(burstLength);
    let burstMax = 0;
    for (let i = 0; i < burstLength; i += 1) { burst[i] = Math.random(); burstMax = Math.max(burstMax, burst[i]); }
    for (let i = 0; i < burstLength; i += 1) burst[i] = ((burst[i] / burstMax) - 0.5) * 2 * note.velocity;
    let burstIndex = 0;
    const offAt = audioContext.currentTime + duration;
    processor.onaudioprocess = event => {
      const out = event.outputBuffer.getChannelData(0);
      for (let i = 0; i < out.length; i += 1) {
        const original = this.mode === 'original';
        if (!original && extendedParams.noteOff && audioContext.currentTime + i / fs >= offAt) off = true;
        let excitation = burstIndex < burstLength ? burst[burstIndex++] : 0;
        if (original) {
          const delayed = delay[modulo(write - burstLength, delay.length)];
          const filtered = 0.996 * 0.5 * (delayed + originalPrevious);
          originalPrevious = delayed;
          const outputSample = excitation + filtered;
          delay[write] = outputSample;
          write = (write + 1) % delay.length;
          out[i] = alive ? outputSample : 0;
          continue;
        }
        const damping = off ? Math.min(1, extendedParams.damping * 1.1) : extendedParams.damping;
        const decay = off ? extendedParams.decay * 0.9 : extendedParams.decay;
        const outputSample = extendedProcessor.process(excitation, {
          pluck: extendedParams.pluck,
          dynamic: extendedParams.dynamic,
          damping,
          decay
        });
        out[i] = alive ? outputSample : 0;
      }
    };
    gain.gain.value = 0.24;
    processor.connect(gain).connect(audioContext.destination);
    const voice = { stop() { if (!alive) return; alive = false; processor.disconnect(); gain.disconnect(); } };
    this.voices.push(voice);
    const lifetime = this.mode === 'original' || !extendedParams.noteOff ? 6000 : Math.max(3500, duration * 1000 + 1600);
    setTimeout(() => { voice.stop(); this.voices = this.voices.filter(item => item !== voice); }, lifetime);
  }

  setPlayhead(step) {
    this.root.querySelectorAll('.roll-lane').forEach(lane => lane.style.setProperty('--playhead', `${step * STEP_WIDTH}px`));
    this.root.querySelector('[data-readout]').textContent = `${Math.floor(step / 16) + 1}.${Math.floor((step % 16) / 4) + 1}.${step % 4 + 1}`;
  }

  stop() {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    this.voices.forEach(voice => voice.stop());
    this.voices = [];
    this.root.querySelectorAll('.roll-lane').forEach(lane => lane.style.removeProperty('--playhead'));
    const button = this.root.querySelector('[data-action="play"]');
    if (button) button.innerHTML = '<i class="bi bi-play-fill"></i> Play';
  }
}

export function stopSequence() {
  controllers.forEach(controller => controller.stop());
}
