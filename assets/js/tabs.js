export function initTabs({ onTabChange } = {}) {
  const tabButtons = Array.from(document.querySelectorAll('[data-tab-target]'));
  const tabPanels = new Map(
    Array.from(document.querySelectorAll('.tab-panel')).map(panel => [panel.id.replace('tab-', ''), panel])
  );
  const typeset = new Set();

  function setTab(tabName) {
    tabButtons.forEach(button => button.classList.toggle('active', button.dataset.tabTarget === tabName));
    tabPanels.forEach((panel, name) => panel.classList.toggle('active', name === tabName));
    document.body.classList.toggle('synth-open', tabName === 'synth');
    if (onTabChange) onTabChange(tabName);

    // The story panel is one long scroll and gets typeset once on load; the
    // synth panel only needs it the first time it is revealed.
    if (!typeset.has(tabName) && window.MathJax?.typesetPromise) {
      const panel = tabPanels.get(tabName);
      if (panel) {
        typeset.add(tabName);
        window.MathJax.typesetPromise([panel]);
      }
    }
    window.scrollTo({ top: 0, behavior: 'auto' });
  }

  tabButtons.forEach(button => {
    button.addEventListener('click', () => setTab(button.dataset.tabTarget));
  });

  setTab('story');
}
