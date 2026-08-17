let pauseExamples = () => {};

const STEPS = 64;
const BAR_STEPS = 8;
const FIRST_MIDI = 36;
const LAST_MIDI = 64;
const STEP_WIDTH = 11;
const KSA_F0_MIN = 20;
const LAGRANGE_ORDER = 5;
const MAX_VOICES = 24;
const VELOCITY_GAIN_MIN = 0.001;
const VELOCITY_GAIN_MAX = 1;
const C_MAJOR_PITCH_CLASSES = new Set([0, 2, 4, 5, 7, 9, 11]);
const ROLL_MIDI_NOTES = Array.from(
  { length: LAST_MIDI - FIRST_MIDI + 1 },
  (_, index) => LAST_MIDI - index
).filter(midi => C_MAJOR_PITCH_CLASSES.has(midi % 12));
const controllers = [];
let audioContext;
let lastActiveController = null;
let extendedParams = { dynamic: 0.62, pluck: 0.22, decay: 0.99999999, damping: 0.7, noteOn: true, noteOff: true };

// Upper-staff transcription supplied by the user: eight 2/4 measures,
// transposed down one octave to fit the compact C3-E4 demonstration range.
const defaultNotes = [
  { midi: 53, start: 0, length: 2, velocity: 0.82 },
  { midi: 57, start: 2, length: 2, velocity: 0.88 },
  { midi: 59, start: 4, length: 4, velocity: 0.92 },
  { midi: 53, start: 8, length: 2, velocity: 0.78 },
  { midi: 57, start: 10, length: 2, velocity: 0.86 },
  { midi: 59, start: 12, length: 4, velocity: 0.90 },
  { midi: 53, start: 16, length: 2, velocity: 0.80 },
  { midi: 57, start: 18, length: 2, velocity: 0.86 },
  { midi: 59, start: 20, length: 2, velocity: 0.90 },
  { midi: 64, start: 22, length: 2, velocity: 0.94 },
  { midi: 62, start: 24, length: 4, velocity: 0.96 },
  { midi: 59, start: 28, length: 2, velocity: 0.86 },
  { midi: 60, start: 30, length: 2, velocity: 0.88 },
  { midi: 59, start: 32, length: 2, velocity: 0.86 },
  { midi: 55, start: 34, length: 2, velocity: 0.78 },
  { midi: 52, start: 36, length: 8, velocity: 0.72 },
  { midi: 50, start: 46, length: 2, velocity: 0.68 },
  { midi: 52, start: 48, length: 2, velocity: 0.74 },
  { midi: 55, start: 50, length: 2, velocity: 0.82 },
  { midi: 52, start: 52, length: 12, velocity: 0.70 }
];

// Lower-staff accompaniment from the supplied score: F-(A,C) for the first
// four bars, then C-(E,G). The bass root is followed by three repeated dyads.
const accompaniment = [
  { root: 41, dyad: [45, 48] }, // F2, then A2-C3
  { root: 36, dyad: [40, 43] }  // C2, then E2-G2
];
for (let bar = 0; bar < 8; bar += 1) {
  const chord = accompaniment[bar < 4 ? 0 : 1];
  defaultNotes.push({ midi: chord.root, start: bar * BAR_STEPS, length: 2, velocity: 0.01, voice: 'bass' });
  for (let beat = 1; beat < 4; beat += 1) {
    chord.dyad.forEach(midi => defaultNotes.push({ midi, start: bar * BAR_STEPS + beat * 2, length: 2, velocity: 0.01, voice: 'bass' }));
  }
}

const midiToFreq = midi => 440 * 2 ** ((midi - 69) / 12);
const noteName = midi => `${['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][midi % 12]}${Math.floor(midi / 12) - 1}`;
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const modulo = (value, length) => ((value % length) + length) % length;
const gainToDecibels = gain => 20 * Math.log10(gain);

export function computeDynamicsR(f0, dynamicLevel, fs) {
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

export function onePolePhaseDelay(f0, a1, fs) {
  const omega = 2 * Math.PI * f0 / fs;
  return -Math.atan2(a1 * Math.sin(omega), 1 - a1 * Math.cos(omega)) / omega;
}

export function lagrangeFractionalDelay(totalDelay) {
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

/**
 * Karplus & Strong (1983), in Smith's signal-processing recasting with explicit
 * additive input:  y[n] = x[n] + 1/2 * (y[n-N] + y[n-N-1]).
 *
 * As published there is no gain term: the two-point average is the only loss in
 * the loop. Written that way the algorithm is only marginally stable, because
 * |H_L| is exactly 1 at DC — the pole sits on the unit circle and any DC in the
 * excitation rings forever. Implementations therefore keep a loop gain a hair
 * below one, as here. It also matters that the averager's magnitude is
 * |cos(w/2)|, which at a modern sample rate is very close to 1 for a low note:
 * the 1983 algorithm decays far more slowly at 48 kHz than at the 8-25 kHz it
 * was written for, and that is exactly what motivated the explicit decay
 * control the extended version adds.
 *
 * The two-point averager H_L(z) = 1/2 (1 + z^-1) is linear phase with a phase
 * delay of exactly half a sample at every frequency, so the loop rings at
 * fs / (N + 1/2), not fs / N. Karplus and Strong's own tuning rule is therefore
 *
 *     N = fs / f0 - 1/2
 *
 * and because N must be an integer, pitch lands on a quantised grid — there is
 * no interpolation filter here to take up the remainder. That residual error is
 * what the fractional delay in the extended algorithm exists to remove.
 */
export class OriginalKsaProcessor {
  static delayLengthFor(f0, fs) {
    return Math.max(2, Math.round(fs / f0 - 0.5));
  }

  static soundingFrequency(delayLength, fs) {
    return fs / (delayLength + 0.5);
  }

  constructor(sampleRate, frequency, { decayGain = 0.996 } = {}) {
    this.fs = sampleRate;
    this.decayGain = decayGain;
    this.delayLength = OriginalKsaProcessor.delayLengthFor(frequency, sampleRate);
    this.frequency = OriginalKsaProcessor.soundingFrequency(this.delayLength, sampleRate);
    this.delay = new Float64Array(Math.floor(sampleRate / KSA_F0_MIN));
    this.write = 0;
    this.previous = 0;
  }

  /** `delayLength` may be supplied per sample so the pitch can track a slider. */
  process(excitation, { delayLength } = {}) {
    if (delayLength && delayLength !== this.delayLength) {
      this.delayLength = delayLength;
      this.frequency = OriginalKsaProcessor.soundingFrequency(delayLength, this.fs);
    }
    const delayed = this.delay[modulo(this.write - this.delayLength, this.delay.length)];
    const filtered = this.decayGain * 0.5 * (delayed + this.previous);
    this.previous = delayed;
    const output = excitation + filtered;
    this.delay[this.write] = output;
    this.write = (this.write + 1) % this.delay.length;
    return output;
  }
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

  process(excitation, { pluck, dynamic, damping, decay, f0 }) {
    // f0 may change per sample, so a slider drag retunes the string as you move
    // it rather than at the next pluck.
    if (f0 && f0 !== this.frequency) {
      this.frequency = f0;
      this.period = this.fs / f0;
    }
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
      input.parentElement.querySelector('output').value = input.dataset.ksParam === 'decay' ? value.toFixed(8) : scale ? value.toFixed(5) : value.toFixed(2);
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

  setMode(mode) {
    if (mode === this.mode) return;
    this.stop();
    this.mode = mode;
    this.root.dataset.mode = mode;
    this.root.querySelectorAll('[data-mode-option]').forEach(button =>
      button.classList.toggle('active', button.dataset.modeOption === mode));
    document.querySelectorAll('[data-extended-only]').forEach(element =>
      element.toggleAttribute('hidden', mode !== 'extended'));
    this.renderNotes();
  }

  build() {
    this.root.dataset.mode = this.mode;
    this.root.innerHTML = `
      <div class="piano-toolbar">
        <div class="transport"><button class="btn btn-sm btn-dark" type="button" data-action="play"><i class="bi bi-play-fill"></i> Play</button><label class="tempo-control">Tempo <input class="form-control form-control-sm" type="number" min="60" max="180" value="160"></label></div>
        <div class="mode-switch" role="group" aria-label="Algorithm">
          <button type="button" data-mode-option="original"${this.mode === 'original' ? ' class="active"' : ''}>Original KSA</button>
          <button type="button" data-mode-option="extended"${this.mode === 'extended' ? ' class="active"' : ''}>Extended KSA</button>
        </div>
        <span data-readout>1.1.1</span>
      </div>
      <p class="roll-help">C-major notes · drag empty space to draw · drag notes sideways for time or up/down for pitch · drag an edge to resize · double-click to delete · <kbd>⌘</kbd>-drag vertically for burst gain</p>
      <div class="roll-grid"></div>`;
    this.grid = this.root.querySelector('.roll-grid');
    ROLL_MIDI_NOTES.forEach(midi => {
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
        line.className = beat % BAR_STEPS === 0 ? 'bar-line' : 'beat-line';
        lane.appendChild(line);
      }
      lane.addEventListener('pointerdown', event => this.beginDraw(event));
      this.grid.appendChild(lane);
    });
    this.root.querySelector('[data-action="play"]').addEventListener('click', () => this.play());
    this.root.querySelectorAll('[data-mode-option]').forEach(button =>
      button.addEventListener('click', () => this.setMode(button.dataset.modeOption)));
    this.renderNotes();
  }

  renderNotes() {
    this.root.querySelectorAll('.midi-note').forEach(element => element.remove());
    this.notes.forEach(note => {
      const lane = this.root.querySelector(`.roll-lane[data-midi="${note.midi}"]`);
      const block = document.createElement('div');
      block.className = 'midi-note';
      block.classList.toggle('selected', note === this.selectedNote);
      block.classList.toggle('bass-note', note.voice === 'bass');
      block.style.left = `${note.start * STEP_WIDTH + 1}px`;
      block.style.width = `${note.length * STEP_WIDTH - 2}px`;
      block.style.setProperty('--velocity', note.velocity);
      block.title = `${noteName(note.midi)} · burst gain ${note.velocity.toFixed(3)} (${gainToDecibels(note.velocity).toFixed(1)} dB)`;
      block.innerHTML = `<span>${noteName(note.midi)}</span><b class="velocity-meter"></b><b class="resize-handle left"></b><b class="resize-handle right"></b>`;
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
    const note = { midi: Number(event.currentTarget.dataset.midi), start, length: 1, velocity: 0.72, voice: 'melody' };
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
    const velocityEdit = event.metaKey;
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
      const originPitchIndex = ROLL_MIDI_NOTES.indexOf(gesture.originMidi);
      const rowDelta = Math.round((event.clientY - gesture.originY) / 22);
      note.midi = ROLL_MIDI_NOTES[clamp(originPitchIndex + rowDelta, 0, ROLL_MIDI_NOTES.length - 1)];
    }
    if (gesture.kind === 'resize-right') note.length = clamp(gesture.originLength + delta, 1, STEPS - note.start);
    if (gesture.kind === 'resize-left') {
      const end = gesture.originStart + gesture.originLength;
      note.start = clamp(gesture.originStart + delta, 0, end - 1);
      note.length = end - note.start;
    }
    if (gesture.kind === 'velocity') note.velocity = clamp(gesture.originVelocity + (gesture.originY - event.clientY) / 120, VELOCITY_GAIN_MIN, VELOCITY_GAIN_MAX);
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
      const bpm = clamp(Number(this.root.querySelector('input[type="number"]').value) || 160, 60, 180);
      const secondsPerStep = 60 / (bpm * 4);
      this.setPlayhead(this.step);
      const notesAtStep = this.notes.filter(item => item.start === this.step);
      if (this.mode === 'original' || extendedParams.noteOn) notesAtStep.forEach(note => this.trigger(note, note.length * secondsPerStep));
      this.step = (this.step + 1) % STEPS;
      this.timer = setTimeout(tick, secondsPerStep * 1000);
    };
    tick();
  }

  trigger(note, duration) {
    while (this.voices.length >= MAX_VOICES) this.voices.shift().stop();
    const fs = audioContext.sampleRate;
    const frequency = midiToFreq(note.midi);
    const period = fs / frequency;
    let alive = true, off = false;
    const originalProcessor = new OriginalKsaProcessor(fs, frequency);
    const extendedProcessor = new ExtendedKsaProcessor(fs, frequency);
    const processor = audioContext.createScriptProcessor(512, 0, 1);
    const gain = audioContext.createGain();
    // The excitation is one delay-line period long. In original mode that is the
    // quantised N the loop actually uses, so burst and loop stay commensurate.
    const burstLength = this.mode === 'original' ? originalProcessor.delayLength : Math.floor(period);
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
          out[i] = alive ? originalProcessor.process(excitation) : 0;
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
    this.root.querySelector('[data-readout]').textContent = `${Math.floor(step / BAR_STEPS) + 1}.${Math.floor((step % BAR_STEPS) / 4) + 1}.${step % 4 + 1}`;
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
