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
    """Abstract decoder interface."""

    num_params: int
    param_names: tuple[str, ...]

    @abstractmethod
    def activate(
        self,
        raw,
        detected: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]: ...

    @abstractmethod
    def synthesise(self, params: dict[str, Tensor]) -> SynthOutput: ...

    @abstractmethod
    def oracle_synth(self, params: dict[str, Tensor]) -> SynthOutput: ...

    def forward(self, params: dict[str, Tensor]) -> SynthOutput:
        return self.synthesise(params)


# ═════════════════════════════════════════════════════════════════════════════
# Activation helpers
# ═════════════════════════════════════════════════════════════════════════════

def _sigmoid_range(x: Tensor, lo: float, hi: float) -> Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


def _modified_sigmoid(x: Tensor) -> Tensor:
    """Modified sigmoid from Engel et al. (2020), eq. 5."""
    return 2.0 * torch.sigmoid(x) ** math.log(10) + 1e-7


class STEReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return F.relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# ═════════════════════════════════════════════════════════════════════════════
# Karplus-Strong Decoder
# ═════════════════════════════════════════════════════════════════════════════

class KSDecoder(Decoder):
    """KS decoder for the dual-pipeline (CQT + LearnableFrontend) encoder.

    Expected input layout (from ``KSEncoder``)::

        raw = {
            "resonator":  [B, 3, T]
                ch 0 → f0 in Hz      ** pre-activated by soft-argmax **
                ch 1 → decay         (raw logit)
                ch 2 → a1 / damping  (raw logit)
            "excitation": [B, 3, T]
                ch 0 → burst_gain        (raw logit)
                ch 1 → dynamic_level     (raw logit)
                ch 2 → pluck_position    (raw logit)
            "f0_probs":   [B, T, n_f0_bins]   (for optional loss)
        }

    f0 arrives already in Hz from the encoder's soft-argmax, so
    ``activate()`` uses it directly (clamped to synth range).

    Args:
        synth:                  The KS ``Synth`` instance.
        use_external_detectors: If True, ``detected["onsets"]`` gates
                                burst_gain and ``detected["f0"]``
                                overrides the soft-argmax f0.
    """

    _RES_IDX = {"f0": 0, "decay": 1, "a1": 2}
    _EXC_IDX = {"burst_gain": 0, "dynamic_level": 1, "pluck_position": 2}

    def __init__(
        self,
        synth: Synth,
        use_external_detectors: bool = False,
    ):
        super().__init__()
        self.synth = synth
        self.use_external_detectors = use_external_detectors

        self.param_names = KS_PARAM_NAMES
        self.num_params = len(self.param_names)

    def activate(
        self,
        raw: dict[str, Tensor],
        detected: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        detected = detected or {}
        ext = self.use_external_detectors
        res = raw["resonator"]
        exc = raw["excitation"]

        ri, ei = self._RES_IDX, self._EXC_IDX

        # ── f0 (pre-activated in Hz by soft-argmax) ─────────────────────
        det_f0 = detected.get("f0")
        if ext and det_f0 is not None:
            f0 = det_f0
        else:
            f0 = res[:, ri["f0"], :].clamp(F0_MIN_HZ, F0_MAX_HZ)

        # ── decay, a1 (raw logits → sigmoid_range) ─────────────────────
        decay = _sigmoid_range(res[:, ri["decay"], :],
                               DECAY_MIN, DECAY_MAX)
        a1 = _sigmoid_range(res[:, ri["a1"], :],
                            DAMPING_MIN, DAMPING_MAX)

        # ── Excitation params ───────────────────────────────────────────
        det_onsets = detected.get("onsets")
        if ext and det_onsets is not None:
            burst_gain = (
                BURST_GAIN_MAX
                * torch.sigmoid(exc[:, ei["burst_gain"], :])
                * det_onsets
            )
        else:
            burst_gain = STEReLU.apply(exc[:, ei["burst_gain"], :])

        dynamic_level = _sigmoid_range(
            exc[:, ei["dynamic_level"], :],
            DYNAMIC_LEVEL_MIN + 1e-3, DYNAMIC_LEVEL_MAX,
        )
        pluck_position = _sigmoid_range(
            exc[:, ei["pluck_position"], :],
            PLUCK_POSITION_MIN, PLUCK_POSITION_MAX,
        ).clamp(min=PLUCK_POSITION_MIN, max=PLUCK_POSITION_MAX)

        return {
            "f0":             f0,
            "burst_gain":     burst_gain,
            "pluck_position": pluck_position,
            "dynamic_level":  dynamic_level,
            "a1":             a1,
            "decay":          decay,
        }

    def synthesise(self, params: dict[str, Tensor]) -> SynthOutput:
        return self.synth(params)

    @torch.no_grad()
    def oracle_synth(self, params: dict[str, Tensor]) -> SynthOutput:
        return self.synth.oracle_synth(params)


# ═════════════════════════════════════════════════════════════════════════════
# Harmonics + Noise Decoder  (unchanged — still uses flat [B, P, T] input)
# ═════════════════════════════════════════════════════════════════════════════

class HarmonicsNoiseDecoder(Decoder):
    """Harmonics + Noise decoder (DDSP-style)."""

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

        self.num_params = n_harmonics + n_noise_bands
        self.param_names = ("amplitude", "harmonic_distribution",
                            "noise_magnitudes", "f0")

    def activate(
        self,
        raw: Tensor,
        detected: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        detected = detected or {}
        nh = self.n_harmonics
        nn_ = self.n_noise_bands

        amplitude = detected["loudness"]
        f0 = detected["f0"]
        harm_dist = _modified_sigmoid(raw[:, :nh, :])
        noise_mag = _modified_sigmoid(raw[:, nh:nh + nn_, :])

        return {
            "amplitude":             amplitude,
            "harmonic_distribution": harm_dist,
            "noise_magnitudes":      noise_mag,
            "f0":                    f0,
        }

    def synthesise(self, params: dict[str, Tensor]) -> SynthOutput:
        f0 = params["f0"]
        amplitude = params["amplitude"]
        harm_dist = params["harmonic_distribution"]
        noise_mag = params["noise_magnitudes"]

        B, T = f0.shape
        N = self.num_samples

        f0_up = lin_resample(f0, N)
        amp_up = lin_resample(amplitude, N)
        harm_dist_up = F.interpolate(
            harm_dist, size=N, mode='linear', align_corners=False
        ).permute(0, 2, 1)

        harm_dist_up = self._remove_above_nyquist(harm_dist_up, f0_up)
        harm_dist_up = harm_dist_up / (harm_dist_up.sum(dim=-1, keepdim=True) + 1e-8)
        amplitudes = harm_dist_up * amp_up.unsqueeze(-1)

        harmonic = self._harmonic_synth(f0_up, amplitudes)
        block_size = N // T
        noise_mag_t = noise_mag.permute(0, 2, 1)
        noise = self._noise_synth(noise_mag_t, block_size)

        audio = harmonic + noise
        return audio, params

    @torch.no_grad()
    def oracle_synth(self, params: dict[str, Tensor]) -> SynthOutput:
        return self.synthesise(params)

    def _remove_above_nyquist(self, amplitudes, f0):
        n_harm = amplitudes.shape[-1]
        harmonics_hz = f0.unsqueeze(-1) * torch.arange(
            1, n_harm + 1, device=f0.device, dtype=f0.dtype
        )
        mask = (harmonics_hz < self.fs / 2).float() + 1e-4
        return amplitudes * mask

    def _harmonic_synth(self, f0, amplitudes):
        n_harm = amplitudes.shape[-1]
        omega = torch.cumsum(2 * math.pi * f0 / self.fs, dim=1)
        omegas = omega.unsqueeze(-1) * torch.arange(
            1, n_harm + 1, device=f0.device, dtype=f0.dtype
        )
        return (torch.sin(omegas) * amplitudes).sum(dim=-1)

    def _noise_synth(self, noise_mag, block_size):
        ir = self._amp_to_impulse_response(noise_mag, block_size)
        noise = torch.rand_like(ir) * 2 - 1
        filtered = self._fft_convolve(noise, ir)
        return filtered.reshape(filtered.shape[0], -1)

    @staticmethod
    def _amp_to_impulse_response(amp, target_size):
        amp_complex = torch.stack([amp, torch.zeros_like(amp)], dim=-1)
        amp_complex = torch.view_as_complex(amp_complex)
        ir = torch.fft.irfft(amp_complex)
        filter_size = ir.shape[-1]
        ir = torch.roll(ir, filter_size // 2, dims=-1)
        win = torch.hann_window(filter_size, dtype=ir.dtype, device=ir.device)
        ir = ir * win
        ir = F.pad(ir, (0, target_size - filter_size))
        ir = torch.roll(ir, -(filter_size // 2), dims=-1)
        return ir

    @staticmethod
    def _fft_convolve(signal, kernel):
        signal = F.pad(signal, (0, signal.shape[-1]))
        kernel = F.pad(kernel, (kernel.shape[-1], 0))
        output = torch.fft.irfft(
            torch.fft.rfft(signal) * torch.fft.rfft(kernel)
        )
        output = output[..., output.shape[-1] // 2:]
        return output