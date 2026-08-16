// Wires the two Section 2 demos to the precomputed landscapes.
//
//   Demo A (2.2)  decay x damping, tKSA vs fKSA at a reader-chosen FFT size.
//   Demo B (2.3)  onset time x f0, showing that nothing tells an onset which
//                 way to move.

import {
  Surface, Walker, drawTarget, drawWalkers, loadData, nearestIndex,
  paintOverlay, paintSpectrogram, paintSurface, renderFrequencySampling,
  renderTimeDomain, whenVisible,
} from './optim-demo.js';

const TKSA_COLOUR = '#2f6fb0';
const FKSA_COLOUR = '#d55b37';

function bindCanvasDrag(canvas, surface, onPick) {
  const pick = event => {
    const rect = canvas.getBoundingClientRect();
    const col = ((event.clientX - rect.left) / rect.width) * (surface.cols - 1);
    const row = (1 - (event.clientY - rect.top) / rect.height) * (surface.rows - 1);
    onPick(
      Math.max(0, Math.min(surface.cols - 1, col)),
      Math.max(0, Math.min(surface.rows - 1, row)),
    );
  };
  canvas.addEventListener('pointerdown', event => {
    canvas.setPointerCapture(event.pointerId);
    pick(event);
    const move = e => pick(e);
    const up = () => {
      canvas.removeEventListener('pointermove', move);
    };
    canvas.addEventListener('pointermove', move);
    canvas.addEventListener('pointerup', up, { once: true });
  });
}

// ── Demo A ──────────────────────────────────────────────────────────────────
function initDemoA(root, meta) {
  const canvas = root.querySelector('[data-landscape]');
  const specTarget = root.querySelector('[data-spec="target"]');
  const specT = root.querySelector('[data-spec="tksa"]');
  const specF = root.querySelector('[data-spec="fksa"]');
  const fftSelect = root.querySelector('[data-fft]');
  const onsetSelect = root.querySelector('[data-onset]');
  const playButton = root.querySelector('[data-action="play"]');
  const stepButton = root.querySelector('[data-action="step"]');
  const resetButton = root.querySelector('[data-action="reset"]');
  const readout = root.querySelector('[data-readout="status"]');

  const fs = meta.fs;
  const length = meta.numSamples;
  const fixed = meta.fixed;

  let surfaceT = null;
  let surfaceF = null;
  let walkerT = null;
  let walkerF = null;
  let timer = null;
  let startCol = 20;
  let startRow = 60;

  const load = () => {
    const onset = onsetSelect.value;
    const fft = fftSelect.value;
    surfaceT = new Surface('A', `A_${onset}_tksa`);
    surfaceF = new Surface('A', `A_${onset}_fksa${fft}`);
    walkerT = new Walker(surfaceT, TKSA_COLOUR, 'tKSA');
    walkerF = new Walker(surfaceF, FKSA_COLOUR, 'fKSA');
    reset();
  };

  const paramsAt = (surface, col, row) => ({
    f0: fixed.f0,
    fs,
    decay: surface.xValues[Math.round(col)],
    damping: surface.yValues[Math.round(row)],
    pluck: fixed.pluck_position,
    dynamic: fixed.dynamic_level,
    onsetSamples: (onsetSelect.value === '3s' ? 3 : 0) * fs,
    length,
  });

  const drawSpectrograms = () => {
    const onsetSamples = (onsetSelect.value === '3s' ? 3 : 0) * fs;
    paintSpectrogram(specTarget, renderTimeDomain({
      f0: fixed.f0, fs, decay: fixed.decay, damping: fixed.a1,
      pluck: fixed.pluck_position, dynamic: fixed.dynamic_level, onsetSamples, length,
    }), {});
    paintSpectrogram(specT, renderTimeDomain(paramsAt(surfaceT, walkerT.col, walkerT.row)), {});
    paintSpectrogram(specF, renderFrequencySampling({
      ...paramsAt(surfaceF, walkerF.col, walkerF.row),
      nFft: Number(fftSelect.value),
    }), {});
  };

  const render = () => {
    const ctx = paintSurface(canvas, surfaceT);
    const targetCol = nearestIndex(surfaceT.xValues, meta.A.target.decay);
    const targetRow = nearestIndex(surfaceT.yValues, meta.A.target.a1);
    drawTarget(ctx, canvas, surfaceT, targetCol, targetRow);
    drawWalkers(ctx, canvas, [walkerT, walkerF], surfaceT);
    if (readout) {
      readout.innerHTML =
        `<span style="color:${TKSA_COLOUR}">tKSA</span> g=${surfaceT.xValues[Math.round(walkerT.col)].toFixed(4)}, ` +
        `a=${surfaceT.yValues[Math.round(walkerT.row)].toFixed(3)} &nbsp;·&nbsp; ` +
        `<span style="color:${FKSA_COLOUR}">fKSA</span> g=${surfaceF.xValues[Math.round(walkerF.col)].toFixed(4)}, ` +
        `a=${surfaceF.yValues[Math.round(walkerF.row)].toFixed(3)} &nbsp;·&nbsp; ` +
        `target g=${meta.A.target.decay}, a=${meta.A.target.a1}`;
    }
    drawSpectrograms();
  };

  function reset() {
    stop();
    walkerT.reset(startCol, startRow);
    walkerF.reset(startCol, startRow);
    render();
  }

  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    if (playButton) playButton.textContent = 'Run descent';
  }

  const stepOnce = () => {
    walkerT.step();
    walkerF.step();
    render();
    if (walkerT.done && walkerF.done) stop();
  };

  if (playButton) playButton.addEventListener('click', () => {
    if (timer) { stop(); return; }
    playButton.textContent = 'Pause';
    timer = setInterval(stepOnce, 60);
  });
  if (stepButton) stepButton.addEventListener('click', () => { stop(); stepOnce(); });
  if (resetButton) resetButton.addEventListener('click', reset);
  if (fftSelect) fftSelect.addEventListener('change', load);
  if (onsetSelect) onsetSelect.addEventListener('change', load);

  load();
  bindCanvasDrag(canvas, surfaceT, (col, row) => {
    startCol = col;
    startRow = row;
    reset();
  });
}

// ── Demo B ──────────────────────────────────────────────────────────────────
function initDemoB(root, meta) {
  const canvas = root.querySelector('[data-landscape]');
  const overlay = root.querySelector('[data-spec="overlay"]');
  const targetSelect = root.querySelector('[data-target-choice]');
  const playButton = root.querySelector('[data-action="play"]');
  const stepButton = root.querySelector('[data-action="step"]');
  const resetButton = root.querySelector('[data-action="reset"]');
  const readout = root.querySelector('[data-readout="status"]');

  const fs = meta.fs;
  const length = meta.numSamples;
  const fixed = meta.fixed;
  const targets = meta.B.targets;

  let surfaceT = null;
  let surfaceF = null;
  let walkerT = null;
  let walkerF = null;
  let timer = null;
  let startCol = 70;
  let startRow = 20;
  let targetIndex = [1, 1];

  const load = () => {
    const [ti, fi] = targetIndex;
    surfaceT = new Surface('B', `B_t${ti}_f${fi}_tksa`);
    surfaceF = new Surface('B', `B_t${ti}_f${fi}_fksa`);
    walkerT = new Walker(surfaceT, TKSA_COLOUR, 'tKSA');
    walkerF = new Walker(surfaceF, FKSA_COLOUR, 'fKSA');
    reset();
  };

  const signalAt = (surface, col, row) => renderTimeDomain({
    f0: surface.yValues[Math.round(row)],
    fs,
    decay: fixed.decay,
    damping: fixed.a1,
    pluck: fixed.pluck_position,
    dynamic: fixed.dynamic_level,
    onsetSamples: surface.xValues[Math.round(col)] * fs,
    length,
  });

  const render = () => {
    const targetOnset = targets.onsets[targetIndex[0]];
    const targetF0 = targets.f0s[targetIndex[1]];
    const ctx = paintSurface(canvas, surfaceT);
    drawTarget(ctx, canvas, surfaceT,
      nearestIndex(surfaceT.xValues, targetOnset),
      nearestIndex(surfaceT.yValues, targetF0));
    drawWalkers(ctx, canvas, [walkerT, walkerF], surfaceT);

    const target = renderTimeDomain({
      f0: targetF0, fs, decay: fixed.decay, damping: fixed.a1,
      pluck: fixed.pluck_position, dynamic: fixed.dynamic_level,
      onsetSamples: targetOnset * fs, length,
    });
    paintOverlay(overlay, target, signalAt(surfaceT, walkerT.col, walkerT.row));

    if (readout) {
      readout.innerHTML =
        `<span style="color:${TKSA_COLOUR}">tKSA</span> t=${surfaceT.xValues[Math.round(walkerT.col)].toFixed(2)}s, ` +
        `f0=${surfaceT.yValues[Math.round(walkerT.row)].toFixed(0)}Hz &nbsp;·&nbsp; ` +
        `<span style="color:${FKSA_COLOUR}">fKSA</span> t=${surfaceF.xValues[Math.round(walkerF.col)].toFixed(2)}s, ` +
        `f0=${surfaceF.yValues[Math.round(walkerF.row)].toFixed(0)}Hz &nbsp;·&nbsp; ` +
        `target t=${targetOnset}s, f0=${targetF0}Hz`;
    }
  };

  function reset() {
    stop();
    walkerT.reset(startCol, startRow);
    walkerF.reset(startCol, startRow);
    render();
  }

  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    if (playButton) playButton.textContent = 'Run descent';
  }

  const stepOnce = () => {
    walkerT.step();
    walkerF.step();
    render();
    if (walkerT.done && walkerF.done) stop();
  };

  if (playButton) playButton.addEventListener('click', () => {
    if (timer) { stop(); return; }
    playButton.textContent = 'Pause';
    timer = setInterval(stepOnce, 60);
  });
  if (stepButton) stepButton.addEventListener('click', () => { stop(); stepOnce(); });
  if (resetButton) resetButton.addEventListener('click', reset);
  if (targetSelect) {
    targets.onsets.forEach((t, ti) => targets.f0s.forEach((f, fi) => {
      const option = document.createElement('option');
      option.value = `${ti},${fi}`;
      option.textContent = `${t} s, ${f} Hz`;
      if (ti === 1 && fi === 1) option.selected = true;
      targetSelect.appendChild(option);
    }));
    targetSelect.addEventListener('change', () => {
      targetIndex = targetSelect.value.split(',').map(Number);
      load();
    });
  }

  load();
  bindCanvasDrag(canvas, surfaceT, (col, row) => {
    startCol = col;
    startRow = row;
    reset();
  });
}

export function initOptimDemos() {
  const a = document.querySelector('[data-demo="a"]');
  const b = document.querySelector('[data-demo="b"]');
  if (!a && !b) return;

  const start = async () => {
    let meta;
    try {
      meta = await loadData();
    } catch (error) {
      [a, b].forEach(root => {
        if (!root) return;
        const note = root.querySelector('[data-readout="status"]');
        if (note) note.textContent =
          'Precomputed landscapes are not available — run tools/precompute_landscapes.py.';
      });
      console.warn('optim demos: could not load landscapes', error);
      return;
    }
    if (a) initDemoA(a, meta);
    if (b) initDemoB(b, meta);
  };

  whenVisible(a || b, start, '400px');
}
