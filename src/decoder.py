from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.synths.synth import Synth, SynthOutput
from src.synths.ddsp import lin_resample
from src.synths.param_registry import (
    PARAM_NAMES as KS_PARAM_NAMES,
    F0_MIN_HZ, F0_MAX_HZ,
    PLUCK_POSITION_MIN, PLUCK_POSITION_MAX,
    DYNAMIC_LEVEL_MIN, DYNAMIC_LEVEL_MAX,
    DAMPING_MIN, DAMPING_MAX,
    DECAY_MIN, DECAY_MAX,
    BURST_GAIN_MAX,
)


class Decoder(ABC, nn.Module):
    """Abstract decoder interface.

    Subclasses must define
    ---------------------
    num_params      : int
    param_names     : tuple[str, ...]
    activate(raw, detected) → dict
    synthesise(params) → SynthOutput
    oracle_synth(params) → SynthOutput
    """

    num_params: int
    param_names: tuple[str, ...]

    @abstractmethod
    def activate(
        self,
        raw: Tensor,
        detected: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Map raw encoder logits [B, P, T] → {name: [B, T]}.

        ``detected`` carries external-detector signals.  Each subclass
        decides which keys it consumes (see module docstring).
        """
        ...

    @abstractmethod
    def synthesise(self, params: dict[str, Tensor]) -> SynthOutput:
        """Differentiable synthesis.  Returns (audio [B, N], params)."""
        ...

    @abstractmethod
    def oracle_synth(self, params: dict[str, Tensor]) -> SynthOutput:
        """Non-differentiable high-fidelity synthesis."""
        ...

    def forward(self, params: dict[str, Tensor]) -> SynthOutput:
        return self.synthesise(params)

def _sigmoid_range(x: Tensor, lo: float, hi: float) -> Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


def _modified_sigmoid(x: Tensor) -> Tensor:
    """Modified sigmoid from Engel et al. (2020), eq. 5.

    Scaled output, steeper slope via exponentiation, and a floor at 1e-7
    for training stability:  ``2.0 · sigmoid(x)^log10 + 1e-7``
    """
    return 2.0 * torch.sigmoid(x) ** math.log(10) + 1e-7

# ═════════════════════════════════════════════════════════════════════════════
# Karplus-Strong Decoder
# ═════════════════════════════════════════════════════════════════════════════
class KSDecoder(Decoder):
    def __init__(
            self,
            synth: Synth,
            use_external_detectors: bool = False
    ):
        super().__init__()
        self.synth = synth
        self.use_external_detectors = use_external_detectors

        self.param_names = KS_PARAM_NAMES
        self.num_params = len(self.param_names)
        self._idx = {name: i for i, name in enumerate(self.param_names)}

    def activate(
        self,
        raw: Tensor,
        detected: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        detected = detected or {}
        ext = self.use_external_detectors

        # ── f0 ──
        det_f0 = detected.get("f0")
        if ext and det_f0 is not None:
            f0 = det_f0
        else:
            f0 = _sigmoid_range(raw[:, self._idx["f0"], :],
                                F0_MIN_HZ, F0_MAX_HZ)

        # ── burst_gain ──
        det_onsets = detected.get("onsets")
        if ext and det_onsets is not None:
            burst_gain = (
                BURST_GAIN_MAX
                * torch.sigmoid(raw[:, self._idx["burst_gain"], :])
                * det_onsets
            )
        else:
            burst_gain = F.relu(raw[:, self._idx["burst_gain"], :])

        # ── remaining params ──
        pluck_pos = _sigmoid_range(raw[:, self._idx["pluck_position"], :],
                                   PLUCK_POSITION_MIN, PLUCK_POSITION_MAX)
        dyn_level = _sigmoid_range(raw[:, self._idx["dynamic_level"], :],
                                   DYNAMIC_LEVEL_MIN + 1e-3, DYNAMIC_LEVEL_MAX)
        a1 = _sigmoid_range(raw[:, self._idx["a1"], :],
                            DAMPING_MIN, DAMPING_MAX)
        decay = _sigmoid_range(raw[:, self._idx["decay"], :],
                               DECAY_MIN, DECAY_MAX)

        return {
            "f0":             f0,
            "burst_gain":     burst_gain,
            "pluck_position": pluck_pos,
            "dynamic_level":  dyn_level,
            "a1":             a1,
            "decay":          decay,
        }

    def synthesise(self, params: dict[str, Tensor]) -> SynthOutput:
        return self.synth(params)

    @torch.no_grad()
    def oracle_synth(self, params: dict[str, Tensor]) -> SynthOutput:
        return self.synth.oracle_synth(params)


# ═════════════════════════════════════════════════════════════════════════════
# Harmonics + Noise Decoder
# ═════════════════════════════════════════════════════════════════════════════
class HarmonicsNoiseDecoder(Decoder):
    """Harmonics + Noise decoder (DDSP-style).
    Parameter layout (for default 100 harmonics, 65 noise bands)::

        Encoder ch   Name                    Activation
        ----------   --------------------    -----------------------
        0..99        harmonic_distribution   modified_sigmoid (per-harmonic)
        100..164     noise_magnitudes        modified_sigmoid

    Total encoder outputs: 165.  Amplitude = detected loudness directly.
    f0 always from external CREPE detector.

    External detectors (``detected`` dict) — **always required** for H+N::

        "f0"       : [B, T] Hz       → fundamental frequency
        "loudness" : [B, T] [0, 1]   → normalised loudness envelope

    Synthesis pipeline::

        1. Upsample frame-rate params → sample-rate signals
        2. Anti-alias harmonics (zero above Nyquist)
        3. Normalise harmonic distribution, scale by amplitude: A_k = A · c_k
        4. Additive sinusoidal synthesis via cumulative phase
        5. Filtered noise via frequency-domain IR → FFT convolution
        6. Sum harmonic + noise components

    Args:
        fs:                    Sample rate.
        num_samples:           Audio length in samples.
        n_harmonics:           Number of harmonic overtones.
        n_noise_bands:         Number of noise filter bands.
        use_external_detectors: Must be ``True`` for H+N (enforced).
    """

    def __init__(
        self,
        fs: int = 16000,
        num_samples: int = 64000,
        n_harmonics: int = 100,
        n_noise_bands: int = 65,
        use_external_detectors: bool = True,
    ):
        super().__init__()
        assert use_external_detectors, (
            "HarmonicsNoiseDecoder requires external detectors "
            "(f0 from CREPE, loudness from A-weighted analysis)."
        )
        self.fs = fs
        self.num_samples = num_samples
        self.n_harmonics = n_harmonics
        self.n_noise_bands = n_noise_bands
        self.use_external_detectors = True

        # harmonics + noise only (f0 and amplitude from detectors)
        self.num_params = n_harmonics + n_noise_bands
        self.param_names = ("amplitude", "harmonic_distribution",
                            "noise_magnitudes", "f0")

    def activate(
        self,
        raw: Tensor,
        detected: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Map encoder logits → synthesis parameters.

        Args:
            raw:      [B, 165, T] from encoder (100 harmonic + 65 noise).
            detected: Must contain ``"f0"`` [B, T] and ``"loudness"`` [B, T].
        """
        detected = detected or {}
        nh = self.n_harmonics
        nn_ = self.n_noise_bands

        amplitude = detected["loudness"]                                    # [B, T]
        f0 = detected["f0"]                                                 # [B, T]
        harm_dist = _modified_sigmoid(raw[:, :nh, :])                       # [B, nh, T]
        noise_mag = _modified_sigmoid(raw[:, nh:nh + nn_, :])               # [B, nn, T]

        return {
            "amplitude":             amplitude,       # [B, T]
            "harmonic_distribution": harm_dist,        # [B, nh, T]
            "noise_magnitudes":      noise_mag,        # [B, nn, T]
            "f0":                    f0,               # [B, T]
        }


    def synthesise(self, params: dict[str, Tensor]) -> SynthOutput: # Returns: (audio [B, N], params)
        f0 = params["f0"]                              # [B, T]
        amplitude = params["amplitude"]                 # [B, T]
        harm_dist = params["harmonic_distribution"]     # [B, nh, T]
        noise_mag = params["noise_magnitudes"]          # [B, nn, T]

        B, T = f0.shape
        N = self.num_samples

        f0_up = lin_resample(f0, N)               # [B, N]
        amp_up = lin_resample(amplitude, N)        # [B, N]
        harm_dist_up = F.interpolate(
            harm_dist, size=N, mode='linear', align_corners=False
        ).permute(0, 2, 1)  # [B, nh, N] → [B, N, nh]

        harm_dist_up = self._remove_above_nyquist(harm_dist_up, f0_up)

        # ── Normalise distribution and scale by amplitude ──
        harm_dist_up = harm_dist_up / (harm_dist_up.sum(dim=-1, keepdim=True) + 1e-8)
        amplitudes = harm_dist_up * amp_up.unsqueeze(-1)  # [B, N, nh]

        # ── Harmonic synthesis (additive, cumulative phase) ──
        harmonic = self._harmonic_synth(f0_up, amplitudes)  # [B, N]

        # ── Noise synthesis (filtered white noise) ──
        block_size = N // T
        noise_mag_t = noise_mag.permute(0, 2, 1)       # [B, T, nn]
        noise = self._noise_synth(noise_mag_t, block_size)  # [B, N]

        audio = harmonic + noise                        # [B, N]
        return audio, params

    @torch.no_grad()
    def oracle_synth(self, params: dict[str, Tensor]) -> SynthOutput:
        """Same as synthesise, without gradient tracking."""
        return self.synthesise(params)

    def _remove_above_nyquist(
        self, amplitudes: Tensor, f0: Tensor
    ) -> Tensor:
        """Zero out harmonic amplitudes whose frequency exceeds Nyquist.

        Args:
            amplitudes: [B, N, n_harmonics]
            f0:         [B, N]
        """
        n_harm = amplitudes.shape[-1]
        # f0 * k for k = 1, 2, …, n_harm
        harmonics_hz = f0.unsqueeze(-1) * torch.arange(
            1, n_harm + 1, device=f0.device, dtype=f0.dtype
        )                                                  # [B, N, n_harm]
        mask = (harmonics_hz < self.fs / 2).float() + 1e-4
        return amplitudes * mask

    def _harmonic_synth(self, f0: Tensor, amplitudes: Tensor) -> Tensor:
        """Additive synthesis via cumulative-phase oscillator bank.

        Args:
            f0:         [B, N] instantaneous frequency in Hz.
            amplitudes: [B, N, n_harmonics] per-harmonic amplitudes.

        Returns:
            [B, N] audio signal.
        """
        n_harm = amplitudes.shape[-1]

        # Instantaneous phase increment per sample
        omega = torch.cumsum(
            2 * math.pi * f0 / self.fs, dim=1
        )                                                   # [B, N]

        # Phase for each harmonic: ω(t) × k
        omegas = omega.unsqueeze(-1) * torch.arange(
            1, n_harm + 1, device=f0.device, dtype=f0.dtype
        )                                                   # [B, N, n_harm]

        return (torch.sin(omegas) * amplitudes).sum(dim=-1)  # [B, N]

    def _noise_synth(
        self, noise_mag: Tensor, block_size: int
    ) -> Tensor:
        """Filtered-noise synthesis.

        For each frame, the frequency-domain magnitudes are converted to a
        time-domain impulse response, then convolved with white noise.

        Args:
            noise_mag:  [B, T, n_noise_bands] filter magnitudes per frame.
            block_size: Samples per frame (= num_samples // num_frames).

        Returns:
            [B, N] noise signal.
        """
        # Frequency-domain magnitudes → causal impulse response
        ir = self._amp_to_impulse_response(noise_mag, block_size)  # [B, T, block_size]

        # White noise, one block per frame
        noise = torch.rand_like(ir) * 2 - 1

        # Per-frame FFT convolution
        filtered = self._fft_convolve(noise, ir)                   # [B, T, block_size]

        return filtered.reshape(filtered.shape[0], -1)             # [B, N]

    @staticmethod
    def _amp_to_impulse_response(amp: Tensor, target_size: int) -> Tensor:
        """Convert frequency-domain magnitudes to a windowed impulse response.

        Follows the IRCAM DDSP approach: treat magnitudes as a half-spectrum,
        IRFFT to time domain, apply Hann window, zero-pad to block size.

        Args:
            amp:         [B, T, n_bands]
            target_size: desired IR length (= block_size).

        Returns:
            [B, T, target_size]
        """
        # Build one-sided spectrum (real magnitudes, zero imaginary)
        amp_complex = torch.stack([amp, torch.zeros_like(amp)], dim=-1)
        amp_complex = torch.view_as_complex(amp_complex)
        ir = torch.fft.irfft(amp_complex)                  # [B, T, filter_size]

        filter_size = ir.shape[-1]

        # Centre the filter and window
        ir = torch.roll(ir, filter_size // 2, dims=-1)
        win = torch.hann_window(filter_size, dtype=ir.dtype, device=ir.device)
        ir = ir * win

        # Zero-pad to block_size and shift back
        ir = F.pad(ir, (0, target_size - filter_size))
        ir = torch.roll(ir, -(filter_size // 2), dims=-1)

        return ir

    @staticmethod
    def _fft_convolve(signal: Tensor, kernel: Tensor) -> Tensor:
        """FFT-based linear convolution (last dimension).

        Both tensors are zero-padded to avoid circular artefacts.

        Args:
            signal: [B, T, L]
            kernel: [B, T, L]

        Returns:
            [B, T, L] (truncated to original length).
        """
        signal = F.pad(signal, (0, signal.shape[-1]))
        kernel = F.pad(kernel, (kernel.shape[-1], 0))

        output = torch.fft.irfft(
            torch.fft.rfft(signal) * torch.fft.rfft(kernel)
        )
        output = output[..., output.shape[-1] // 2:]
        return output