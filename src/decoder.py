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
    """KS decoder for the DETR-style Event Encoder.

    Expected input layout (from ``KSEventEncoder``)::

        raw = {
            "exists":   [B, max_events, 1]  (raw logits)
            "time":     [B, max_events, 1]  (raw logits)
            "f0_hz":    [B, max_events, 1]  (pre-activated by soft-argmax)
            "params":   [B, max_events, 5]  (raw logits for physical params)
        }

    Args:
        synth:                  The KS ``Synth`` instance.
        num_samples:            The dense sample resolution to expand to (e.g., 64000).
        exist_threshold:        Probability threshold to render an event.
        use_external_detectors: If True, overrides predicted exists/f0 with external signals.
    """

    _PARAM_IDX = {"burst_gain": 0, "decay": 1, "a1": 2, "pluck_position": 3, "dynamic_level": 4}

    def __init__(
            self,
            synth: Synth,
            num_samples: int = 64000,
            exist_threshold: float = 0.5,
            use_external_detectors: bool = False,
    ):
        super().__init__()
        self.synth = synth
        self.num_samples = num_samples
        self.exist_threshold = exist_threshold
        self.use_external_detectors = use_external_detectors

        self.param_names = KS_PARAM_NAMES
        self.num_params = len(self.param_names)

    def activate(
            self,
            raw: dict[str, Tensor],
            detected: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """
        Takes raw event logits, applies sigmoid boundaries, maps external detectors,
        and expands them into dense sample arrays for the synthesizer.
        """
        B, max_events, _ = raw["exists"].shape
        pi = self._PARAM_IDX

        # ── 1. Standard Activations ──────────────────────────────────────────
        exists_prob = torch.sigmoid(raw["exists"]).squeeze(-1)
        time_val = torch.sigmoid(raw["time"]).squeeze(-1)
        f0_val = raw["f0_hz"].squeeze(-1).clamp(F0_MIN_HZ, F0_MAX_HZ)

        # ── 2. External Detector Overrides ──────────────────────────────────
        if self.use_external_detectors and detected is not None:
            det_onsets = detected.get("onsets")  # Expected: [B, num_frames]
            det_f0 = detected.get("f0")  # Expected: [B, num_frames]

            if det_onsets is not None and det_f0 is not None:
                num_frames = det_f0.shape[1]

                # Convert continuous predicted time [0, 1] to a frame index [0, 249]
                frame_indices = (time_val * num_frames).long().clamp(0, num_frames - 1)

                # Gather the external detector values exactly where the queries are looking
                det_onsets_gathered = torch.gather(det_onsets, 1, frame_indices)
                det_f0_gathered = torch.gather(det_f0, 1, frame_indices)

                # Substitute existence and f0
                exists_prob = det_onsets_gathered
                f0_val = det_f0_gathered.clamp(F0_MIN_HZ, F0_MAX_HZ)

        # ── 3. Package Events ───────────────────────────────────────────────
        return {
            "exists": exists_prob,
            "time": time_val,
            "f0": f0_val,
            "burst_gain": _sigmoid_range(raw["params"][..., pi["burst_gain"]], 0.0, BURST_GAIN_MAX),
            "decay": _sigmoid_range(raw["params"][..., pi["decay"]], DECAY_MIN, DECAY_MAX),
            "a1": _sigmoid_range(raw["params"][..., pi["a1"]], DAMPING_MIN, DAMPING_MAX),
            "pluck_position": _sigmoid_range(raw["params"][..., pi["pluck_position"]], PLUCK_POSITION_MIN,
                                             PLUCK_POSITION_MAX),
            "dynamic_level": _sigmoid_range(raw["params"][..., pi["dynamic_level"]], DYNAMIC_LEVEL_MIN,
                                            DYNAMIC_LEVEL_MAX),
        }

    def _events_to_samples(self, events: dict[str, Tensor], B: int, num_samples: int) -> dict[str, Tensor]:
        device = events["exists"].device

        dense = {
            "f0": torch.full((B, num_samples), 440.0, device=device),
            "burst_gain": torch.zeros((B, num_samples), device=device),  # Gain stays 0
            "decay": torch.full((B, num_samples), DECAY_MIN, device=device),
            "a1": torch.full((B, num_samples), 0.5, device=device),
            "pluck_position": torch.full((B, num_samples), 0.5, device=device),
            "dynamic_level": torch.full((B, num_samples), 0.5, device=device),
        }

        for b in range(B):
            valid_mask = events["exists"][b] > 0.5

            if not valid_mask.any():
                continue

            valid_times = events["time"][b, valid_mask]
            sorted_idx = torch.argsort(valid_times)

            v_time = valid_times[sorted_idx]
            v_f0 = events["f0"][b, valid_mask][sorted_idx]
            v_bg = events["burst_gain"][b, valid_mask][sorted_idx]
            v_decay = events["decay"][b, valid_mask][sorted_idx]
            v_a1 = events["a1"][b, valid_mask][sorted_idx]
            v_pluck = events["pluck_position"][b, valid_mask][sorted_idx]
            v_dyn = events["dynamic_level"][b, valid_mask][sorted_idx]

            for i in range(len(v_time)):
                start_sample = int(v_time[i].item() * num_samples)
                start_sample = max(0, min(start_sample, num_samples - 1))

                end_sample = num_samples
                if i < len(v_time) - 1:
                    end_sample = int(v_time[i + 1].item() * num_samples)
                    end_sample = max(start_sample, min(end_sample, num_samples))

                fill_start = 0 if i == 0 else start_sample

                dense["f0"][b, fill_start:end_sample] = v_f0[i]
                dense["decay"][b, fill_start:end_sample] = v_decay[i]
                dense["a1"][b, fill_start:end_sample] = v_a1[i]
                dense["pluck_position"][b, fill_start:end_sample] = v_pluck[i]
                dense["dynamic_level"][b, fill_start:end_sample] = v_dyn[i]

                # 3. Apply the Soft-Gated impulse!
                dense["burst_gain"][b, start_sample] = v_bg[i]

        return dense

    def synthesise(self, params: dict[str, torch.Tensor]) -> SynthOutput:
        return self.synth(params)

    @torch.no_grad()
    def oracle_synth(self, params: dict[str, torch.Tensor]) -> SynthOutput:
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
            raw: dict[str, Tensor],
            detected: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        B, max_events, _ = raw["exists"].shape
        pi = self._PARAM_IDX

        exists_prob = torch.sigmoid(raw["exists"]).squeeze(-1)
        time_val = torch.sigmoid(raw["time"]).squeeze(-1)
        f0_val = raw["f0_hz"].squeeze(-1).clamp(F0_MIN_HZ, F0_MAX_HZ)

        if self.use_external_detectors and detected is not None:
            det_onsets = detected.get("onsets")
            det_f0 = detected.get("f0")
            if det_onsets is not None and det_f0 is not None:
                num_frames = det_f0.shape[1]
                frame_indices = (time_val * num_frames).long().clamp(0, num_frames - 1)
                exists_prob = torch.gather(det_onsets, 1, frame_indices)
                f0_val = torch.gather(det_f0, 1, frame_indices).clamp(F0_MIN_HZ, F0_MAX_HZ)

        return {
            "exists": exists_prob,
            "time": time_val,
            "f0": f0_val,
            "burst_gain": _sigmoid_range(raw["params"][..., pi["burst_gain"]], 0.0, BURST_GAIN_MAX),
            "decay": _sigmoid_range(raw["params"][..., pi["decay"]], DECAY_MIN, DECAY_MAX),
            "a1": _sigmoid_range(raw["params"][..., pi["a1"]], DAMPING_MIN, DAMPING_MAX),
            "pluck_position": _sigmoid_range(raw["params"][..., pi["pluck_position"]], PLUCK_POSITION_MIN,
                                             PLUCK_POSITION_MAX),
            "dynamic_level": _sigmoid_range(raw["params"][..., pi["dynamic_level"]], DYNAMIC_LEVEL_MIN,
                                            DYNAMIC_LEVEL_MAX),
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