import { initAudioExamples, pauseExamplePlayers } from './audio-examples.js';
import { initInteractiveKarplus, stopSequence } from './interactive-ks.js';
import { initKsaFigure, stopKsaFigure } from './section-ksa.js';
import { initOptimDemos } from './section-optim.js';
import { initTabs } from './tabs.js';

initAudioExamples();
initInteractiveKarplus({ pauseExamples: () => pauseExamplePlayers(null) });
initKsaFigure();
initOptimDemos();

initTabs({
  onTabChange: tabName => {
    // Never leave a string ringing in a panel the reader has left.
    if (tabName !== 'synth') stopSequence();
    if (tabName !== 'story') stopKsaFigure();
  },
});

// In-prose links that jump to the synth tab.
document.querySelectorAll('[data-goto-tab]').forEach(button =>
  button.addEventListener('click', () => {
    const target = document.querySelector(`[data-tab-target="${button.dataset.gotoTab}"]`);
    if (target) target.click();
  }));
