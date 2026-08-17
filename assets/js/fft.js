// Minimal radix-2 FFT and STFT helpers. Self-contained on purpose: the page
// already pulls WaveSurfer from a CDN, but the DSP that backs a quantitative
// claim should not depend on a third party staying online.

/** In-place iterative radix-2 Cooley-Tukey. `re`/`im` must have length 2^k. */
export function fftInPlace(re, im) {
  const n = re.length;
  if (n <= 1) return;
  if ((n & (n - 1)) !== 0) throw new Error(`fft: length ${n} is not a power of two`);

  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const step = -2 * Math.PI / len;
    const wRe = Math.cos(step);
    const wIm = Math.sin(step);
    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let k = 0; k < len / 2; k += 1) {
        const aRe = re[i + k];
        const aIm = im[i + k];
        const bRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
        const bIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
        re[i + k] = aRe + bRe;
        im[i + k] = aIm + bIm;
        re[i + k + len / 2] = aRe - bRe;
        im[i + k + len / 2] = aIm - bIm;
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }
}

/** Inverse FFT, in place. */
export function ifftInPlace(re, im) {
  const n = re.length;
  for (let i = 0; i < n; i += 1) im[i] = -im[i];
  fftInPlace(re, im);
  for (let i = 0; i < n; i += 1) {
    re[i] /= n;
    im[i] = -im[i] / n;
  }
}


const isPowerOfTwo = n => n > 0 && (n & (n - 1)) === 0;

/**
 * Bluestein's algorithm: a DFT of any length, by turning it into a convolution
 * that a radix-2 FFT can do.
 *
 * Needed because the frame sizes the Section 2.2 demo offers are round numbers
 * of seconds — 2 s, 3 s and 4 s at 16 kHz are 32000, 48000 and 64000 samples —
 * and none of those is a power of two. The browser has to render the same frame
 * the loss was sampled with, so rounding to a nearby power of two would mean
 * showing one thing and measuring another.
 *
 * Uses jk = (j^2 + k^2 - (k - j)^2) / 2 to rewrite the transform as a linear
 * convolution of length >= 2n - 1, which is then padded up to a power of two.
 */
function bluestein(re, im, inverse) {
  const n = re.length;
  let m = 1;
  while (m < 2 * n + 1) m <<= 1;

  const sign = inverse ? 1 : -1;
  const cos = new Float64Array(n);
  const sin = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    // (i * i) % (2n) keeps the angle exact for large i, where i * i would not
    // survive double precision.
    const angle = (Math.PI * ((i * i) % (2 * n))) / n;
    cos[i] = Math.cos(angle);
    sin[i] = Math.sin(angle) * sign;
  }

  // a[j] = x[j] * w[j], with w[i] = cos + i*sin as built above.
  const aRe = new Float64Array(m);
  const aIm = new Float64Array(m);
  for (let i = 0; i < n; i += 1) {
    aRe[i] = re[i] * cos[i] - im[i] * sin[i];
    aIm[i] = re[i] * sin[i] + im[i] * cos[i];
  }

  // b[m] = conj(w[m]), mirrored so the cyclic convolution is a linear one.
  const bRe = new Float64Array(m);
  const bIm = new Float64Array(m);
  bRe[0] = cos[0];
  bIm[0] = -sin[0];
  for (let i = 1; i < n; i += 1) {
    bRe[i] = bRe[m - i] = cos[i];
    bIm[i] = bIm[m - i] = -sin[i];
  }

  fftInPlace(aRe, aIm);
  fftInPlace(bRe, bIm);
  for (let i = 0; i < m; i += 1) {
    const tr = aRe[i] * bRe[i] - aIm[i] * bIm[i];
    aIm[i] = aRe[i] * bIm[i] + aIm[i] * bRe[i];
    aRe[i] = tr;
  }
  // Inverse of the length-m transform, by conjugation.
  for (let i = 0; i < m; i += 1) aIm[i] = -aIm[i];
  fftInPlace(aRe, aIm);
  for (let i = 0; i < m; i += 1) {
    aRe[i] /= m;
    aIm[i] = -aIm[i] / m;
  }

  // X[k] = w[k] * conv[k]
  for (let i = 0; i < n; i += 1) {
    re[i] = aRe[i] * cos[i] - aIm[i] * sin[i];
    im[i] = aRe[i] * sin[i] + aIm[i] * cos[i];
  }
}

/** Forward transform of any length, in place. */
export function transform(re, im) {
  if (isPowerOfTwo(re.length)) fftInPlace(re, im);
  else bluestein(re, im, false);
}

/** Inverse transform of any length, in place. */
export function inverseTransform(re, im) {
  const n = re.length;
  if (isPowerOfTwo(n)) {
    ifftInPlace(re, im);
    return;
  }
  bluestein(re, im, true);
  for (let i = 0; i < n; i += 1) {
    re[i] /= n;
    im[i] /= n;
  }
}

/**
 * Inverse real FFT from a half spectrum of length n/2 + 1, mirroring it back to
 * full Hermitian length before transforming. Returns the real signal. `n` may be
 * any even length, not only a power of two.
 */
export function irfft(halfRe, halfIm, n) {
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  const half = n / 2;
  for (let k = 0; k <= half; k += 1) {
    re[k] = halfRe[k];
    im[k] = halfIm[k];
  }
  for (let k = 1; k < half; k += 1) {
    re[n - k] = halfRe[k];
    im[n - k] = -halfIm[k];
  }
  im[0] = 0;
  if (half < n) im[half] = 0;
  inverseTransform(re, im);
  return re;
}

/** Magnitude spectrum of a real signal, zero-padded to `n`. Length n/2 + 1. */
export function magnitudeSpectrum(signal, n) {
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  re.set(signal.subarray(0, Math.min(signal.length, n)));
  fftInPlace(re, im);
  const out = new Float64Array(n / 2 + 1);
  for (let k = 0; k < out.length; k += 1) out[k] = Math.hypot(re[k], im[k]);
  return out;
}

export function hannWindow(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i += 1) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n);
  return w;
}

/**
 * Log-magnitude STFT. Returns { frames, bins, data } with `data` laid out
 * frame-major, already in dB and clamped to `floorDb` below its own peak —
 * which is what the spectrogram canvases want.
 */
export function stft(signal, { fftSize = 512, hop = 128, floorDb = -80 } = {}) {
  const window = hannWindow(fftSize);
  const bins = fftSize / 2 + 1;
  const frames = Math.max(1, Math.floor((signal.length - fftSize) / hop) + 1);
  const data = new Float32Array(frames * bins);
  const re = new Float64Array(fftSize);
  const im = new Float64Array(fftSize);
  let peak = 1e-12;

  for (let f = 0; f < frames; f += 1) {
    const start = f * hop;
    im.fill(0);
    for (let i = 0; i < fftSize; i += 1) {
      const s = start + i;
      re[i] = s < signal.length ? signal[s] * window[i] : 0;
    }
    fftInPlace(re, im);
    for (let k = 0; k < bins; k += 1) {
      const mag = Math.hypot(re[k], im[k]);
      data[f * bins + k] = mag;
      if (mag > peak) peak = mag;
    }
  }

  for (let i = 0; i < data.length; i += 1) {
    const db = 20 * Math.log10(Math.max(data[i], 1e-12) / peak);
    data[i] = Math.max(db, floorDb) / -floorDb + 1; // 0 at the floor, 1 at the peak
  }

  return { frames, bins, data };
}
