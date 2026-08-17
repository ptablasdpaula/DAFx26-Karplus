// Closed-form magnitude responses for the two algorithms in Section 1.
//
// These are evaluated straight from the z-domain transfer functions, so the
// curve the reader sees is the filter itself rather than a measurement of it.
// The measured spectrum is drawn underneath by section-ksa.js; the two should
// agree, and tests/browser-check.html asserts that they do.

import { computeDynamicsR, lagrangeFractionalDelay, onePolePhaseDelay } from './interactive-ks.js';

const TWO_PI = Math.PI * 2;

/** Complex helpers. Values are plain {re, im} — these run over ~1k points. */
const mul = (a, b) => ({ re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re });
const div = (a, b) => {
  const d = b.re * b.re + b.im * b.im;
  return { re: (a.re * b.re + a.im * b.im) / d, im: (a.im * b.re - a.re * b.im) / d };
};
const sub = (a, b) => ({ re: a.re - b.re, im: a.im - b.im });
/** z^-k on the unit circle at angle omega. */
const zPow = (omega, k) => ({ re: Math.cos(omega * k), im: -Math.sin(omega * k) });

/**
 * Original KSA:  H(z) = 1 / (1 - 1/2 * z^-N (1 + z^-1)).
 *
 * The delay length N is an integer, so the comb teeth sit at multiples of
 * fs / (N + 1/2) — the extra half sample being the averager's phase delay.
 */
export function originalResponse(omega, { delayLength, decayGain = 0.996 }) {
  const zN = zPow(omega, delayLength);
  const z1 = zPow(omega, 1);
  const loop = mul(zN, { re: decayGain * 0.5 * (1 + z1.re), im: decayGain * 0.5 * z1.im });
  return div({ re: 1, im: 0 }, sub({ re: 1, im: 0 }, loop));
}

/**
 * Extended KSA, paper Eq. 10:
 *
 *     H(z) = H_D(z) H_P(z) / (1 - H_L(z) H_I(z) z^-L_int)
 *
 * with the loop filter of Eq. 2, the Lagrange interpolator of Eqs. 3-4, the
 * pluck-position comb of Eq. 6 and the dynamics filter of Eq. 7.
 */
export function extendedResponse(omega, { f0, fs, decay, damping, pluck, dynamic }) {
  const z1 = zPow(omega, 1);

  // Loop filter, exactly as ExtendedKsaProcessor runs it:
  //     y[n] = g(1 - a) x[n] + a y[n-1]   ->   H_L(z) = g(1-a) / (1 - a z^-1)
  // a one-pole lowpass whose DC gain is the decay g, per Eq. 2's decoupling of
  // overall decay from frequency-dependent damping. (The paper writes the pole
  // with the opposite sign convention for a1; this mirrors the code.)
  const hL = div(
    { re: decay * (1 - damping), im: 0 },
    sub({ re: 1, im: 0 }, { re: damping * z1.re, im: damping * z1.im })
  );

  // Round-trip delay, absorbing the loop filter's phase delay  (Eqs. 1, 5).
  const L = fs / f0;
  const { integerDelay, coefficients } = lagrangeFractionalDelay(
    L + onePolePhaseDelay(f0, damping, fs)
  );

  // Lagrange interpolator H_I(z) = sum h[n] z^-n   (Eq. 4)
  let hI = { re: 0, im: 0 };
  for (let n = 0; n < coefficients.length; n += 1) {
    const zn = zPow(omega, n);
    hI = { re: hI.re + coefficients[n] * zn.re, im: hI.im + coefficients[n] * zn.im };
  }

  const feedback = mul(mul(hL, hI), zPow(omega, integerDelay));

  // Pluck position, H_P(z) = 1 - z^-p, linearly interpolated  (Eq. 6)
  const p = L * pluck;
  const pi = Math.floor(p);
  const pf = p - pi;
  const zp0 = zPow(omega, pi);
  const zp1 = zPow(omega, pi + 1);
  const hP = sub(
    { re: 1, im: 0 },
    { re: (1 - pf) * zp0.re + pf * zp1.re, im: (1 - pf) * zp0.im + pf * zp1.im }
  );

  // Dynamics, H_D(z) = (1 - R) / (1 - R z^-1)   (Eq. 7)
  const R = computeDynamicsR(f0, dynamic, fs);
  const hD = div({ re: 1 - R, im: 0 }, sub({ re: 1, im: 0 }, { re: R * z1.re, im: R * z1.im }));

  return div(mul(hD, hP), sub({ re: 1, im: 0 }, feedback));
}

/**
 * Sample a magnitude response over `points` frequencies from 0 to Nyquist.
 * Returns dB relative to the curve's own peak, which is what gets plotted.
 */
export function sampleResponse(config, { fs, points = 1024, mode = 'extended', floorDb = -60 }) {
  const out = new Float64Array(points);
  let peak = 1e-12;
  for (let i = 0; i < points; i += 1) {
    const omega = (i / (points - 1)) * Math.PI;
    const h = mode === 'original'
      ? originalResponse(omega, config)
      : extendedResponse(omega, { ...config, fs });
    const mag = Math.hypot(h.re, h.im);
    out[i] = mag;
    if (mag > peak) peak = mag;
  }
  for (let i = 0; i < points; i += 1) {
    out[i] = Math.max(20 * Math.log10(Math.max(out[i], 1e-12) / peak), floorDb);
  }
  return out;
}

/** Frequency in Hz of bin `i` of a `points`-long response running to Nyquist. */
export const binToHz = (i, points, fs) => (i / (points - 1)) * (fs / 2);

/** Cents between a requested and an achieved frequency. */
export const centsError = (requested, achieved) => 1200 * Math.log2(achieved / requested);

export { TWO_PI };
