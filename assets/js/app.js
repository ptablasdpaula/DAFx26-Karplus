import { initAudioExamples, pauseExamplePlayers } from './audio-examples.js';
import { initInteractiveKarplus, stopSequence } from './interactive-ks.js';
import { initTabs } from './tabs.js';
initAudioExamples();
initInteractiveKarplus({ pauseExamples: () => pauseExamplePlayers(null) });
initTabs({
  onTabChange: tabName => {
    if (tabName !== 'interactive') stopSequence();
  }
});
