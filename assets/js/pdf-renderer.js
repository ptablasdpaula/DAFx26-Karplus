import * as pdfjsLib from 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.mjs';

pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.worker.mjs';

export async function renderPdfCanvases() {
  const canvases = Array.from(document.querySelectorAll('[data-pdf-src]'));
  await Promise.all(canvases.map(renderPdfCanvas));
}

async function renderPdfCanvas(canvas) {
  const source = canvas.dataset.pdfSrc;
  const fallback = canvas.nextElementSibling;
  try {
    const pdf = await pdfjsLib.getDocument(source).promise;
    const page = await pdf.getPage(1);
    const containerWidth = canvas.parentElement.clientWidth || 720;
    const natural = page.getViewport({ scale: 1 });
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const scale = containerWidth / natural.width;
    const viewport = page.getViewport({ scale });
    canvas.width = Math.floor(viewport.width * dpr);
    canvas.height = Math.floor(viewport.height * dpr);
    canvas.style.width = `${viewport.width}px`;
    canvas.style.height = `${viewport.height}px`;
    const context = canvas.getContext('2d');
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    await page.render({ canvasContext: context, viewport }).promise;
    if (fallback) fallback.hidden = true;
  } catch (error) {
    console.warn(`Could not render ${source}`, error);
    if (fallback) fallback.hidden = false;
  }
}
