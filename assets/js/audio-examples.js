import WaveSurfer from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/wavesurfer.esm.js';
import Spectrogram from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/spectrogram.js';

const AUDIO_EXT = '.ogg';
const EXAMPLES_TO_SHOW = 5;
const MAX_FILES = 290;
const BASE_PATH = 'main/experiments/evaluation/audio';

const allFileIDs = Array.from({length: MAX_FILES}, (_, i) => String(i + 1).padStart(3, '0'));
const rand = (arr, n) => arr.slice().sort(() => 0.5 - Math.random()).slice(0, Math.min(n, arr.length));
const icon = (btn, on) => { btn.innerHTML = on ? '<i class="fas fa-pause"></i>' : '<i class="fas fa-play"></i>'; };
const loadingIcon = btn => { btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>'; };

const SYNTH_MODELS = [
  { label: 'Target', path: `${BASE_PATH}/synthetic/target`, domain: 'always' },
  { label: '$\\mathcal{P}$-Only', path: `${BASE_PATH}/synthetic/pred/oksa/p_only`, domain: 'always' },
  { label: 'Audio-Only<br><small>(freq.)</small>', path: `${BASE_PATH}/synthetic/pred/fksa/audio_only`, domain: 'freq' },
  { label: 'Audio-Only<br><small>(time)</small>', path: `${BASE_PATH}/synthetic/pred/tksa/audio_only`, domain: 'time' },
  { label: '$\\mathcal{P}$+Audio<br><small>(freq.)</small>', path: `${BASE_PATH}/synthetic/pred/fksa/p_audio`, domain: 'freq' },
  { label: '↪ detach $f_0$<br><small>(freq.)</small>', path: `${BASE_PATH}/synthetic/pred/fksa/p_audio/detach_f0`, domain: 'freq', ablation: true },
  { label: '↪ detach onset<br><small>(freq.)</small>', path: `${BASE_PATH}/synthetic/pred/fksa/p_audio/detach_onset`, domain: 'freq', ablation: true },
  { label: '↪ detach both<br><small>(freq.)</small>', path: `${BASE_PATH}/synthetic/pred/fksa/p_audio/detach_both`, domain: 'freq', ablation: true },
  { label: '$\\mathcal{P}$+Audio<br><small>(time)</small>', path: `${BASE_PATH}/synthetic/pred/tksa/p_audio`, domain: 'time' },
  { label: '↪ detach $f_0$<br><small>(time)</small>', path: `${BASE_PATH}/synthetic/pred/tksa/p_audio/detach_f0`, domain: 'time', ablation: true },
  { label: '↪ detach onset<br><small>(time)</small>', path: `${BASE_PATH}/synthetic/pred/tksa/p_audio/detach_onset`, domain: 'time', ablation: true },
  { label: '↪ detach both<br><small>(time)</small>', path: `${BASE_PATH}/synthetic/pred/tksa/p_audio/detach_both`, domain: 'time', ablation: true }
];

const NSYNTH_MODELS = [
  { label: 'Target', path: `${BASE_PATH}/real/target`, domain: 'always' },
  { label: 'HpN', path: `${BASE_PATH}/real/pred/hpn/audio_only`, domain: 'always', hpn: true },
  { label: 'HpN+', path: `${BASE_PATH}/real/pred/hpn_p/audio_only`, domain: 'always', hpn: true },
  { label: '$\\mathcal{P}$-Only', path: `${BASE_PATH}/real/pred/oksa/p_only`, domain: 'always' },
  { label: 'Audio-Only<br><small>(freq.)</small>', path: `${BASE_PATH}/real/pred/fksa/audio_only`, domain: 'freq' },
  { label: 'Audio-Only<br><small>(time)</small>', path: `${BASE_PATH}/real/pred/tksa/audio_only`, domain: 'time' },
  { label: '$\\mathcal{P}$+Audio<br><small>(freq.)</small>', path: `${BASE_PATH}/real/pred/fksa/p_audio`, domain: 'freq' },
  { label: '↪ detach $f_0$<br><small>(freq.)</small>', path: `${BASE_PATH}/real/pred/fksa/p_audio/detach_f0`, domain: 'freq', ablation: true },
  { label: '↪ detach onset<br><small>(freq.)</small>', path: `${BASE_PATH}/real/pred/fksa/p_audio/detach_onset`, domain: 'freq', ablation: true },
  { label: '↪ detach both<br><small>(freq.)</small>', path: `${BASE_PATH}/real/pred/fksa/p_audio/detach_both`, domain: 'freq', ablation: true },
  { label: '$\\mathcal{P}$+Audio<br><small>(time)</small>', path: `${BASE_PATH}/real/pred/tksa/p_audio`, domain: 'time' },
  { label: '↪ detach $f_0$<br><small>(time)</small>', path: `${BASE_PATH}/real/pred/tksa/p_audio/detach_f0`, domain: 'time', ablation: true },
  { label: '↪ detach onset<br><small>(time)</small>', path: `${BASE_PATH}/real/pred/tksa/p_audio/detach_onset`, domain: 'time', ablation: true },
  { label: '↪ detach both<br><small>(time)</small>', path: `${BASE_PATH}/real/pred/tksa/p_audio/detach_both`, domain: 'time', ablation: true }
];

const players = new Map();
const playerCleanups = new Map();
const MAX_ACTIVE_AUDIO_LOADS = 8;
let activeAudioLoads = 0;
const audioLoadQueue = [];
export const pauseExamplePlayers = (keep) => players.forEach((ws, btn) => {
  if (ws !== keep && ws.isPlaying()) {
    ws.pause();
    ws.seekTo(0);
    icon(btn, false);
    btn.closest('.cell')?.classList.remove('playing');
  }
});
const state = {
  synth: {
    view: 'both',
    showAblations: false,
    ids: rand(allFileIDs, EXAMPLES_TO_SHOW)
  },
  real: {
    view: 'both',
    showAblations: false,
    showHpn: false,
    ids: rand(allFileIDs, EXAMPLES_TO_SHOW)
  }
};

function makePlayer(td) {
  return WaveSurfer.create({
    container: td.querySelector('.waveform'),
    backend: 'MediaElement', height: 60, waveColor: '#999', progressColor: '#007BFF',
    plugins: [ Spectrogram.create({ container: td.querySelector('.spectrogram'), scale: 'mel', labels: false, height: 60 }) ]
  });
}

function enqueueAudioLoad(start, priority = false) {
  const job = { start, cancelled: false, done: false };
  if (priority) audioLoadQueue.unshift(job);
  else audioLoadQueue.push(job);
  pumpAudioLoadQueue();
  return () => { job.cancelled = true; };
}

function pumpAudioLoadQueue() {
  while (activeAudioLoads < MAX_ACTIVE_AUDIO_LOADS && audioLoadQueue.length) {
    const job = audioLoadQueue.shift();
    if (job.cancelled) continue;
    activeAudioLoads += 1;
    job.start(() => {
      if (!job.done) {
        job.done = true;
        activeAudioLoads -= 1;
        pumpAudioLoadQueue();
      }
    });
  }
}

function visibleModels(modelsArray, sectionState) {
  return modelsArray.filter(model => {
    const matchesView = model.domain === 'always' || sectionState.view === 'both' || model.domain === sectionState.view;
    const matchesAblations = !model.ablation || sectionState.showAblations;
    const matchesHpn = !model.hpn || sectionState.showHpn;
    return matchesView && matchesAblations && matchesHpn;
  });
}

function clearTablePlayers(tbody) {
  Array.from(playerCleanups.entries()).forEach(([btn, destroy]) => {
    if (tbody.contains(btn)) {
      destroy();
    }
  });
}

function renderTable(tableSelector, modelsArray, sectionState) {
  const thead = document.querySelector(`${tableSelector} thead`);
  const tbody = document.querySelector(`${tableSelector} tbody`);
  let headerHTML = '<tr><th>Model</th>';
  const ids = sectionState.ids;
  ids.forEach(id => {
    headerHTML += `<th>Example ${Number(id)}</th>`;
  });
  clearTablePlayers(tbody);
  thead.innerHTML = headerHTML + '</tr>';
  tbody.innerHTML = '';

  visibleModels(modelsArray, sectionState).forEach(model => {
    const row = document.createElement('tr');

    const labelCell = document.createElement('td');
    labelCell.className = 'row-label';
    labelCell.innerHTML = model.label;
    row.appendChild(labelCell);

    ids.forEach(id => {
      const td = document.createElement('td');
      td.className = 'cell';
      td.innerHTML = `<div class="media-stack"><div class="waveform"></div><div class="spectrogram"></div><button class="overlay-play" aria-label="Play audio"><i class="fas fa-play"></i></button></div>`;
      const url = `${model.path}/${id}${AUDIO_EXT}`;
      const playBtn = td.querySelector('.overlay-play');
      let ws = null;
      let ready = false;
      let loadStarted = false;
      let pendingPlay = false;
      let cancelQueuedLoad = null;
      let finishActiveLoad = null;
      const setReady = (isReady) => {
        ready = isReady;
        playBtn.disabled = false;
      };
      const releaseLoadSlot = () => {
        if (finishActiveLoad) {
          const finish = finishActiveLoad;
          finishActiveLoad = null;
          finish();
        }
      };
      const attemptPlay = async () => {
        if (!ws) return;
        pauseExamplePlayers(ws);
        try {
          await ws.play();
        } catch (error) {
          console.warn(`Could not play ${url}`, error);
          icon(playBtn, false);
        }
      };
      const startLoad = finish => {
        if (loadStarted) {
          if (finish) finish();
          return;
        }
        loadStarted = true;
        finishActiveLoad = finish || null;
        const loadResult = ws.load(url);
        if (loadResult && typeof loadResult.catch === 'function') {
          loadResult.catch(error => {
            console.warn(`Could not load ${url}`, error);
            releaseLoadSlot();
          });
        }
      };
      const promoteLoad = () => {
        if (cancelQueuedLoad) {
          cancelQueuedLoad();
          cancelQueuedLoad = null;
          startLoad(null);
        }
      };
      const cell = {
        init() {
          if (!ws) {
            setReady(false);
            loadingIcon(playBtn);
            td.classList.add('loaded');
            ws = makePlayer(td);
            players.set(playBtn, ws);
            playerCleanups.set(playBtn, cell.destroy);
            ws.on('ready', () => {
              setReady(true);
              releaseLoadSlot();
              if (pendingPlay) {
                pendingPlay = false;
                attemptPlay();
              } else {
                icon(playBtn, false);
              }
            });
            ws.on('play', () => {
              icon(playBtn, true);
              playBtn.setAttribute('aria-label', 'Pause audio');
              td.classList.add('playing');
              td.classList.remove('loading');
            });
            ws.on('pause', () => {
              icon(playBtn, false);
              playBtn.setAttribute('aria-label', 'Play audio');
              td.classList.remove('playing');
            });
            ws.on('finish', () => {
              icon(playBtn, false);
              playBtn.setAttribute('aria-label', 'Play audio');
              td.classList.remove('playing');
            });
            ws.on('error', error => {
              console.warn(`Could not load ${url}`, error);
              setReady(false);
              icon(playBtn, false);
              td.classList.remove('loading');
              releaseLoadSlot();
            });
            cancelQueuedLoad = enqueueAudioLoad(finish => {
              cancelQueuedLoad = null;
              if (!ws) finish();
              else startLoad(finish);
            });
          }
          return ws;
        },
        destroy() {
          pendingPlay = false;
          if (cancelQueuedLoad) {
            cancelQueuedLoad();
            cancelQueuedLoad = null;
          }
          releaseLoadSlot();
          if (ws) {
            try { ws.destroy(); } catch (e) {}
            players.delete(playBtn);
            playerCleanups.delete(playBtn);
            ws = null;
          }
          loadStarted = false;
          setReady(false);
          td.classList.remove('loaded');
          td.classList.remove('loading');
          td.classList.remove('playing');
          icon(playBtn, false);
          playBtn.setAttribute('aria-label', 'Play audio');
        },
      };
      playBtn.onclick = async () => {
        const w = cell.init();
        if (w.isPlaying()) {
          pendingPlay = false;
          w.pause();
          return;
        }
        if (!ready) {
          pendingPlay = true;
          loadingIcon(playBtn);
          td.classList.add('loading');
          promoteLoad();
          try {
            await w.play();
          } catch (error) {
            // MediaElement playback will retry from the ready handler if the source is still loading.
          }
          return;
        }
        await attemptPlay();
      };
      cell.init();
      row.appendChild(td);
    });

    tbody.appendChild(row);
  });
  if (window.MathJax && window.MathJax.typesetPromise) window.MathJax.typesetPromise([thead.parentElement]);
}

function setActive(kind, section, activeButton) {
  document.querySelectorAll(`[data-section="${section}"][${kind}]`).forEach(button => {
    button.classList.toggle('active', button === activeButton);
  });
}

function renderAll() {
  renderTable('#table-synth', SYNTH_MODELS, state.synth);
  renderTable('#table-nsynth', NSYNTH_MODELS, state.real);
}

function initFeaturedAudio() {
  const container = document.querySelector('[data-featured-audio]');
  if (!container) return;
  const button = container.querySelector('.overlay-play');
  const ws = makePlayer(container);
  players.set(button, ws);
  container.classList.add('loaded', 'loading');
  loadingIcon(button);
  button.disabled = true;
  ws.load(container.dataset.featuredAudio);
  ws.on('ready', () => {
    container.classList.remove('loading');
    button.disabled = false;
    icon(button, false);
  });
  ws.on('play', () => { icon(button, true); container.classList.add('playing'); });
  ws.on('pause', () => { icon(button, false); container.classList.remove('playing'); });
  ws.on('finish', () => { icon(button, false); container.classList.remove('playing'); });
  ws.on('error', error => {
    console.warn(`Could not load ${container.dataset.featuredAudio}`, error);
    button.disabled = false;
    icon(button, false);
  });
  button.addEventListener('click', async () => {
    if (ws.isPlaying()) ws.pause();
    else { pauseExamplePlayers(ws); await ws.play(); }
  });
}


export function initAudioExamples() {
  initFeaturedAudio();
  document.querySelectorAll('[data-view]').forEach(button => {
    button.onclick = () => {
      const section = button.dataset.section;
      state[section].view = button.dataset.view;
      setActive('data-view', section, button);
      renderTable(section === 'synth' ? '#table-synth' : '#table-nsynth', section === 'synth' ? SYNTH_MODELS : NSYNTH_MODELS, state[section]);
    };
  });
  document.querySelectorAll('[data-ablation]').forEach(button => {
    button.onclick = () => {
      const section = button.dataset.section;
      state[section].showAblations = button.dataset.ablation === 'on';
      setActive('data-ablation', section, button);
      renderTable(section === 'synth' ? '#table-synth' : '#table-nsynth', section === 'synth' ? SYNTH_MODELS : NSYNTH_MODELS, state[section]);
    };
  });
  document.querySelectorAll('[data-hpn]').forEach(button => {
    button.onclick = () => {
      const section = button.dataset.section;
      state[section].showHpn = button.dataset.hpn === 'on';
      setActive('data-hpn', section, button);
      renderTable('#table-nsynth', NSYNTH_MODELS, state.real);
    };
  });
  document.getElementById('shuffle-synth').onclick = () => {
    state.synth.ids = rand(allFileIDs, EXAMPLES_TO_SHOW);
    renderTable('#table-synth', SYNTH_MODELS, state.synth);
  };
  document.getElementById('shuffle-nsynth').onclick = () => {
    state.real.ids = rand(allFileIDs, EXAMPLES_TO_SHOW);
    renderTable('#table-nsynth', NSYNTH_MODELS, state.real);
  };
  renderAll();
}
