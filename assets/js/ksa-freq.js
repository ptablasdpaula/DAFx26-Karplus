// Frequency-sampling Karplus-Strong, paper Eq. 11:
//
//     H_fKSA(z_M) = H_D(z_M) H_P(z_M) / (1 - H_L(z_M) H_I(z_M) z_M^-L)
//
// evaluated on M = N_FFT/2 + 1 points of the unit circle and inverse
// transformed. Nothing here simulates time-aliasing: the Hermitian constraint
// forces a real impulse response of length N_FFT, so a decay longer than the
// frame wraps around on its own. That is exactly the artefact Fig. 4 of the
// paper shows, and it is why the fKSA marker in the Section 2 demo walks to the
// wrong optimum.

import { computeDynamicsR, lagrangeFractionalDelay, onePolePhaseDelay } from './interactive-ks.js';
import { irfft } from './fft.js';

/**
 * Impulse response of the frequency-sampling KSA for one set of parameters.
 * Returns a Float64Array of length `nFft`.
 */
export function frequencySamplingImpulseResponse({
  f0, fs, decay, damping, pluck, dynamic, nFft = 2048,
}) {
  const bins = nFft / 2 + 1;
  const outRe = new Float64Array(bins);
  const outIm = new Float64Array(bins);

  const L = fs / f0;
  const { integerDelay, coefficients } = lagrangeFractionalDelay(
    L + onePolePhaseDelay(f0, damping, fs)
  );
  const R = computeDynamicsR(f0, dynamic, fs);

  const p = L * pluck;
  const pInt = Math.floor(p);
  const pFrac = p - pInt;

  for (let k = 0; k < bins; k += 1) {
    const omega = (k / (nFft / 2)) * Math.PI;
    const c = Math.cos(omega);
    const s = -Math.sin(omega); // z^-1

    // H_L(z) = g(1-a) / (1 - a z^-1)
    const lNumRe = decay * (1 - damping);
    const lDenRe = 1 - damping * c;
    const lDenIm = -damping * s;
    const lDen2 = lDenRe * lDenRe + lDenIm * lDenIm;
    const hLRe = lNumRe * lDenRe / lDen2;
    const hLIm = -lNumRe * lDenIm / lDen2;

    // H_I(z) = sum h[n] z^-n
    let hIRe = 0;
    let hIIm = 0;
    for (let n = 0; n < coefficients.length; n += 1) {
      hIRe += coefficients[n] * Math.cos(omega * n);
      hIIm -= coefficients[n] * Math.sin(omega * n);
    }

    // z^-integerDelay
    const dRe = Math.cos(omega * integerDelay);
    const dIm = -Math.sin(omega * integerDelay);

    // feedback = H_L * H_I * z^-D
    const liRe = hLRe * hIRe - hLIm * hIIm;
    const liIm = hLRe * hIIm + hLIm * hIRe;
    const fbRe = liRe * dRe - liIm * dIm;
    const fbIm = liRe * dIm + liIm * dRe;

    // H_P(z) = 1 - ((1-f) z^-pi + f z^-(pi+1))
    const p0Re = Math.cos(omega * pInt);
    const p0Im = -Math.sin(omega * pInt);
    const p1Re = Math.cos(omega * (pInt + 1));
    const p1Im = -Math.sin(omega * (pInt + 1));
    const hPRe = 1 - ((1 - pFrac) * p0Re + pFrac * p1Re);
    const hPIm = -((1 - pFrac) * p0Im + pFrac * p1Im);

    // H_D(z) = (1-R) / (1 - R z^-1)
    const dDenRe = 1 - R * c;
    const dDenIm = -R * s;
    const dDen2 = dDenRe * dDenRe + dDenIm * dDenIm;
    const hDRe = (1 - R) * dDenRe / dDen2;
    const hDIm = -(1 - R) * dDenIm / dDen2;

    // numerator = H_D * H_P
    const numRe = hDRe * hPRe - hDIm * hPIm;
    const numIm = hDRe * hPIm + hDIm * hPRe;

    // denominator = 1 - feedback
    const denRe = 1 - fbRe;
    const denIm = -fbIm;
    const den2 = denRe * denRe + denIm * denIm;

    outRe[k] = (numRe * denRe + numIm * denIm) / den2;
    outIm[k] = (numIm * denRe - numRe * denIm) / den2;
  }

  return irfft(outRe, outIm, nFft);
}

/**
 * Render a single pluck through the frequency-sampling KSA: build the impulse
 * response, then place it at the onset. `length` is in samples.
 */
export function renderFrequencySampling({ onsetSamples = 0, length, ...params }) {
  const ir = frequencySamplingImpulseResponse(params);
  const out = new Float64Array(length);
  const start = Math.max(0, Math.round(onsetSamples));
  for (let i = 0; i < ir.length && start + i < length; i += 1) out[start + i] = ir[i];
  return out;
}
