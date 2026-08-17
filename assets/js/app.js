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

// Section nav: jump to a heading, and keep the current one highlighted.
const sectionNav = document.querySelector('[data-section-nav]');
if (sectionNav) {
  const links = Array.from(sectionNav.querySelectorAll('a[href^="#"]'));
  const targets = links
    .map(link => ({ link, element: document.getElementById(link.hash.slice(1)) }))
    .filter(entry => entry.element);

  links.forEach(link => link.addEventListener('click', event => {
    event.preventDefault();
    // The story tab has to be showing before a heading inside it can be reached.
    const storyTab = document.querySelector('[data-tab-target="story"]');
    if (storyTab && !storyTab.classList.contains('active')) storyTab.click();
    document.getElementById(link.hash.slice(1))
      ?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }));

  const mark = () => {
    let current = null;
    targets.forEach(entry => {
      if (entry.element.getBoundingClientRect().top <= window.innerHeight * 0.35) current = entry;
    });
    links.forEach(link => link.classList.toggle('current', current?.link === link));
  };
  window.addEventListener('scroll', mark, { passive: true });
  mark();
}

// In-prose links that jump to the synth tab.
document.querySelectorAll('[data-goto-tab]').forEach(button =>
  button.addEventListener('click', () => {
    const target = document.querySelector(`[data-tab-target="${button.dataset.gotoTab}"]`);
    if (target) target.click();
  }));
