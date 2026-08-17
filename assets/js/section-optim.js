// Wires the two Section 2 demos to the precomputed landscapes.
//
//   Demo A (2.2)  decay x damping, tKSA vs fKSA at a reader-chosen FFT size.
//   Demo B (2.3)  onset time x f0, showing that nothing tells an onset which
//                 way to move.

import {
  GradientField, Surface, Walker, drawTarget, drawWalkers, loadData, nearestIndex,
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
//
// Both events are optimised at once and both are always on screen: a loss
// surface each on the left, and on the right the spectrograms of the whole
// two-event signal. That pairing is the point — a frame short enough to
// truncate the decay of the pluck at 0 s does something quite different to the
// one at 3 s, and you can only see that if you are looking at both.
function initDemoA(root, meta) {
  // Four squares: a surface per (event, implementation). Each walker is drawn
  // on the surface it is actually descending — the two implementations do not
  // share a landscape, which is the whole point of Section 2.2.
  const canvasFor = (event, impl) => root.querySelector(`[data-landscape="${event}-${impl}"]`);
  const specTarget = root.querySelector('[data-spec="target"]');
  const specT = root.querySelector('[data-spec="tksa"]');
  const specF = root.querySelector('[data-spec="fksa"]');
  const fftSelect = root.querySelector('[data-fft]');
  const playButton = root.querySelector('[data-action="play"]');
  const stepButton = root.querySelector('[data-action="step"]');
  const resetButton = root.querySelector('[data-action="reset"]');
  const readout = root.querySelector('[data-readout="status"]');
  const optimiserSelect = root.querySelector('[data-optimiser]');

  const fs = meta.fs;
  const length = meta.numSamples;
  const fixed = meta.fixed;
  const ONSETS = [0, 3];
  const SLICES = ['0s', '3s'];

  // One entry per event, each with its own pair of surfaces and walkers.
  const tracks = ONSETS.map(() => ({ surfaceT: null, surfaceF: null, walkerT: null, walkerF: null }));
  let timer = null;
  let starts = [[20, 60], [20, 60]];

  if (fftSelect && !fftSelect.options.length) {
    meta.A.fftSizes.forEach((n, i) => {
      const option = document.createElement('option');
      option.value = String(n);
      option.textContent = `${n} (${(1000 * n / fs).toFixed(0)} ms)`;
      if (i === Math.floor(meta.A.fftSizes.length / 2)) option.selected = true;
      fftSelect.appendChild(option);
    });
  }

  const load = () => {
    const fft = fftSelect.value;
    tracks.forEach((track, i) => {
      track.surfaceT = new Surface('A', `A_${SLICES[i]}_tksa`);
      track.surfaceF = new Surface('A', `A_${SLICES[i]}_fksa${fft}`);
      const options = { optimiser: optimiserSelect ? optimiserSelect.value : 'adam' };
      track.walkerT = new Walker(track.surfaceT, TKSA_COLOUR, 'tKSA', options);
      track.walkerF = new Walker(track.surfaceF, FKSA_COLOUR, 'fKSA', options);
    });
    reset();
  };

  const paramsFor = (surface, walker) => ({
    decay: surface.xValues[Math.round(walker.col)],
    damping: surface.yValues[Math.round(walker.row)],
  });

  /** Render a two-event signal, one implementation, current walker positions. */
  const twoEvents = (kind) => {
    const out = new Float64Array(length);
    tracks.forEach((track, i) => {
      const params = kind === 'target'
        ? { decay: fixed.decay, damping: fixed.a1 }
        : paramsFor(kind === 'tksa' ? track.surfaceT : track.surfaceF,
                    kind === 'tksa' ? track.walkerT : track.walkerF);
      const common = {
        f0: fixed.f0, fs, pluck: fixed.pluck_position, dynamic: fixed.dynamic_level,
        onsetSamples: ONSETS[i] * fs, length, ...params,
      };
      // Each event is a separate excitation of an LTI system, so superposing the
      // two responses is what either implementation actually produces.
      const piece = kind === 'fksa'
        ? renderFrequencySampling({ ...common, nFft: Number(fftSelect.value) })
        : renderTimeDomain(common);
      for (let n = 0; n < length; n += 1) out[n] += piece[n];
    });
    return out;
  };

  let lastCells = '';
  const drawSpectrograms = (force = false) => {
    const cells = tracks.flatMap(track => [
      Math.round(track.walkerT.col), Math.round(track.walkerT.row),
      Math.round(track.walkerF.col), Math.round(track.walkerF.row),
    ]).join(',');
    if (!force && cells === lastCells) return;
    lastCells = cells;
    paintSpectrogram(specT, twoEvents('tksa'), {});
    paintSpectrogram(specF, twoEvents('fksa'), {});
  };

  const render = () => {
    tracks.forEach((track, i) => {
      [['tksa', track.surfaceT, track.walkerT, TKSA_COLOUR],
       ['fksa', track.surfaceF, track.walkerF, FKSA_COLOUR]].forEach(([impl, surface, walker]) => {
        const canvas = canvasFor(i, impl);
        if (!canvas) return;
        const ctx = paintSurface(canvas, surface);
        drawTarget(ctx, canvas, surface,
          nearestIndex(surface.xValues, meta.A.target.decay),
          nearestIndex(surface.yValues, meta.A.target.a1));
        drawWalkers(ctx, canvas, [walker], surface);
      });
    });
    if (readout) {
      readout.innerHTML = tracks.map((track, i) => {
        const t = paramsFor(track.surfaceT, track.walkerT);
        const f = paramsFor(track.surfaceF, track.walkerF);
        return `<span class="readout-row">Event ${i + 1} @ ${ONSETS[i]} s &nbsp; ` +
          `<span style="color:${TKSA_COLOUR}">tKSA</span> g=${t.decay.toFixed(4)}, a=${t.damping.toFixed(3)} &nbsp;·&nbsp; ` +
          `<span style="color:${FKSA_COLOUR}">fKSA</span> g=${f.decay.toFixed(4)}, a=${f.damping.toFixed(3)}</span>`;
      }).join('') +
      `<span class="readout-row">target g=${meta.A.target.decay}, a=${meta.A.target.a1} for both</span>`;
    }
    drawSpectrograms();
  };

  function reset() {
    stop();
    tracks.forEach((track, i) => {
      track.walkerT.reset(starts[i][0], starts[i][1]);
      track.walkerF.reset(starts[i][0], starts[i][1]);
    });
    lastCells = '';
    paintSpectrogram(specTarget, twoEvents('target'), {});
    render();
  }

  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    if (playButton) playButton.textContent = 'Run descent';
  }

  const stepOnce = () => {
    tracks.forEach(track => { track.walkerT.step(); track.walkerF.step(); });
    render();
    if (tracks.every(track => track.walkerT.done && track.walkerF.done)) stop();
  };

  if (playButton) playButton.addEventListener('click', () => {
    if (timer) { stop(); return; }
    playButton.textContent = 'Pause';
    timer = setInterval(stepOnce, 110);
  });
  if (stepButton) stepButton.addEventListener('click', () => { stop(); stepOnce(); });
  if (resetButton) resetButton.addEventListener('click', reset);
  if (fftSelect) fftSelect.addEventListener('change', load);

  if (optimiserSelect) optimiserSelect.addEventListener('change', load);

  load();
  tracks.forEach((track, i) => ['tksa', 'fksa'].forEach(impl => {
    const canvas = canvasFor(i, impl);
    if (canvas) bindCanvasDrag(canvas, track.surfaceT, (col, row) => {
      starts[i] = [col, row];
      reset();
    });
  }));
}

// ── Demo B ──────────────────────────────────────────────────────────────────
function initDemoB(root, meta) {
  const canvas = root.querySelector('[data-landscape]');
  const overlay = root.querySelector('[data-spec="overlay"]');
  const targetSelect = root.querySelector('[data-target-choice]');
  const mixSlider = root.querySelector('[data-loss-mix]');
  const mixLabel = root.querySelector('[data-readout="mix"]');
  const playButton = root.querySelector('[data-action="play"]');
  const stepButton = root.querySelector('[data-action="step"]');
  const resetButton = root.querySelector('[data-action="reset"]');
  const readout = root.querySelector('[data-readout="status"]');
  const optimiserSelect = root.querySelector('[data-optimiser]');

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

  // Slider position is the SOT share. The paper trains at w_MSS = 0.05 against
  // w_SOT = 1.0, i.e. a 95% SOT share, which is where the slider starts.
  const trainingMix = meta.B.weights.sot / (meta.B.weights.sot + meta.B.weights.mss);
  const currentMix = () => (mixSlider ? Number(mixSlider.value) / 100 : trainingMix);

  const load = () => {
    const [ti, fi] = targetIndex;
    const mix = currentMix();
    if (mixLabel) {
      const pct = Math.round(mix * 100);
      mixLabel.textContent = Math.abs(mix - trainingMix) < 0.02
        ? 'training weights'
        : pct === 0 ? 'pure MSS' : pct === 100 ? 'pure SOT' : `${100 - pct}% MSS / ${pct}% SOT`;
    }
    surfaceT = new Surface('B', `B_t${ti}_f${fi}_tksa`, mix);
    surfaceF = new Surface('B', `B_t${ti}_f${fi}_fksa`, mix);
    // The markers follow the real autograd gradient, not the slope of the
    // heatmap under them — which is exactly why the onset axis misbehaves.
    let fieldT = null;
    let fieldF = null;
    try {
      fieldT = new GradientField(`G_t${ti}_f${fi}_tksa`);
      fieldF = new GradientField(`G_t${ti}_f${fi}_fksa`);
    } catch (error) {
      console.warn('optim demo B: no gradient field, falling back to surface slope', error);
    }
    walkerT = new Walker(surfaceT, TKSA_COLOUR, 'tKSA', { field: fieldT });
    walkerF = new Walker(surfaceF, FKSA_COLOUR, 'fKSA', { field: fieldF });
    targetSignal = null;
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

  // The target signal only changes when the reader picks a different target,
  // so synthesise it there rather than on every descent tick.
  let targetSignal = null;
  let lastCell = '';

  const render = () => {
    const targetOnset = targets.onsets[targetIndex[0]];
    const targetF0 = targets.f0s[targetIndex[1]];
    const ctx = paintSurface(canvas, surfaceT);
    drawTarget(ctx, canvas, surfaceT,
      nearestIndex(surfaceT.xValues, targetOnset),
      nearestIndex(surfaceT.yValues, targetF0));
    drawWalkers(ctx, canvas, [walkerT, walkerF], surfaceT);

    if (!targetSignal) {
      targetSignal = renderTimeDomain({
        f0: targetF0, fs, decay: fixed.decay, damping: fixed.a1,
        pluck: fixed.pluck_position, dynamic: fixed.dynamic_level,
        onsetSamples: targetOnset * fs, length,
      });
    }
    const cell = `${Math.round(walkerT.col)},${Math.round(walkerT.row)}`;
    if (cell !== lastCell) {
      lastCell = cell;
      paintOverlay(overlay, targetSignal, signalAt(surfaceT, walkerT.col, walkerT.row));
    }

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
    lastCell = '';
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
  if (mixSlider) mixSlider.addEventListener('input', load);
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

  // ?eager skips the visibility gate so headless checks exercise the real path.
  if (new URLSearchParams(location.search).has('eager')) start();
  else whenVisible(a || b, start, '400px');
}
