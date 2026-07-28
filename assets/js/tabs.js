export function initTabs({ onTabChange } = {}) {
const tabButtons = Array.from(document.querySelectorAll('[data-tab-target]'));
const tabPanels = new Map(Array.from(document.querySelectorAll('.tab-panel')).map(panel => [panel.id.replace('tab-', ''), panel]));
function setTab(tabName) {
  tabButtons.forEach(button => button.classList.toggle('active', button.dataset.tabTarget === tabName));
  tabPanels.forEach((panel, name) => panel.classList.toggle('active', name === tabName));
  document.body.classList.toggle('synth-open', tabName === 'interactive');
  if (onTabChange) onTabChange(tabName);
  if (window.MathJax && window.MathJax.typesetPromise) {
    const panel = tabPanels.get(tabName);
    if (panel) window.MathJax.typesetPromise([panel]);
  }
}
tabButtons.forEach(button => {
  button.addEventListener('click', () => setTab(button.dataset.tabTarget));
});


  setTab('intro');
}
