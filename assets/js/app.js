import { initAudioExamples, pauseExamplePlayers } from './audio-examples.js';
import { initInteractiveKarplus, stopSequence } from './interactive-ks.js';
import { initTabs } from './tabs.js';
import { renderPdfCanvases } from './pdf-renderer.js';

const refreshPdfs = () => requestAnimationFrame(() => renderPdfCanvases());

refreshPdfs();
initAudioExamples();
initInteractiveKarplus({ pauseExamples: () => pauseExamplePlayers(null) });
initTabs({
  onTabChange: tabName => {
    if (tabName !== 'interactive') stopSequence();
    refreshPdfs();
  }
});
window.addEventListener('resize', refreshPdfs);
