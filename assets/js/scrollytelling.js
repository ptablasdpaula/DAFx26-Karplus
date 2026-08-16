// Drives the sticky figures in Sections 1 and 2.
//
// The panel itself is pinned with CSS `position: sticky`; all this does is
// decide which prose step is currently "in charge" and tell the figure, so the
// figure can morph as the argument moves rather than jumping between separate
// widgets.

/**
 * Watch `[data-step]` elements inside `root` and report the active one.
 *
 * @param {Element} root      container holding the steps
 * @param {(step: string, element: Element) => void} onStep
 * @returns {() => void} teardown
 */
export function observeSteps(root, onStep) {
  const steps = Array.from(root.querySelectorAll('[data-step]'));
  if (!steps.length) return () => {};

  let active = null;
  const setActive = element => {
    if (!element || element === active) return;
    active = element;
    steps.forEach(step => step.classList.toggle('is-active', step === element));
    onStep(element.dataset.step, element);
  };

  // Pick whichever step is nearest the middle of the viewport. Simply taking
  // "the last one that intersected" reads badly when scrolling upward.
  const observer = new IntersectionObserver(() => {
    const middle = window.innerHeight / 2;
    let best = null;
    let bestDistance = Infinity;
    steps.forEach(step => {
      const rect = step.getBoundingClientRect();
      if (rect.bottom < 0 || rect.top > window.innerHeight) return;
      const distance = Math.abs(rect.top + rect.height / 2 - middle);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = step;
      }
    });
    if (best) setActive(best);
  }, { threshold: [0, 0.25, 0.5, 0.75, 1], rootMargin: '-10% 0px -10% 0px' });

  steps.forEach(step => observer.observe(step));

  const onScroll = () => {
    const middle = window.innerHeight / 2;
    let best = null;
    let bestDistance = Infinity;
    steps.forEach(step => {
      const rect = step.getBoundingClientRect();
      if (rect.bottom < 0 || rect.top > window.innerHeight) return;
      const distance = Math.abs(rect.top + rect.height / 2 - middle);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = step;
      }
    });
    if (best) setActive(best);
  };
  window.addEventListener('scroll', onScroll, { passive: true });

  setActive(steps[0]);
  return () => {
    observer.disconnect();
    window.removeEventListener('scroll', onScroll);
  };
}

/** Run `callback` once, the first time `element` comes near the viewport. */
export function whenVisible(element, callback, rootMargin = '200px') {
  const observer = new IntersectionObserver(entries => {
    if (entries.some(entry => entry.isIntersecting)) {
      observer.disconnect();
      callback();
    }
  }, { rootMargin });
  observer.observe(element);
  return () => observer.disconnect();
}
