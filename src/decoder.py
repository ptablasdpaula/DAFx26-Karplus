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
        if self.use_external_detectors:
            if detected is None or "onsets" not in detected or "f0" not in detected:
                raise ValueError(
                    "Decoder requested external detectors, but the batch is missing "
                    "the 'detected' dictionary with 'onsets' and 'f0'. "
                )

            det_onsets = detected.get("onsets")  # Expected: [B, num_frames] binary mask
            det_f0 = detected.get("f0")  # Expected: [B, num_frames] contour

            if det_onsets is not None and det_f0 is not None:
                num_frames = det_f0.shape[1]

                new_exists = torch.zeros_like(exists_prob)
                new_time = torch.zeros_like(time_val)
                new_f0 = torch.zeros_like(f0_val)

                for b in range(B):
                    onset_frames = torch.nonzero(det_onsets[b] > 0.5).squeeze(-1)

                    num_onsets = min(len(onset_frames), max_events)
                    onset_frames = onset_frames[:num_onsets]

                    if num_onsets == 0:
                        continue

                    new_exists[b, :num_onsets] = 1.0
                    new_time[b, :num_onsets] = onset_frames.float() / num_frames

                    for i in range(num_onsets):
                        start_frame = onset_frames[i]
                        end_frame = onset_frames[i + 1] if i < num_onsets - 1 else num_frames

                        segment_f0 = det_f0[b, start_frame:end_frame]
                        if len(segment_f0) > 0:
                            new_f0[b, i] = segment_f0.median()
                        else:
                            new_f0[b, i] = det_f0[b, start_frame]

                exists_prob = new_exists
                time_val = new_time
                f0_val = new_f0.clamp(F0_MIN_HZ, F0_MAX_HZ)

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


class DDSPMLP(nn.Module):
    """DDSP's MLP block replica: Dense -> LayerNorm -> LeakyReLU"""

    def __init__(self, in_features: int, hidden_features: int = 512, num_layers: int = 3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            d_in = in_features if i == 0 else hidden_features
            layers.extend([
                nn.Linear(d_in, hidden_features),
                nn.LayerNorm(hidden_features),
                nn.LeakyReLU(0.01)  # Standard DDSP LeakyReLU slope
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═════════════════════════════════════════════════════════════════════════════
# Harmonics + Noise Decoder
# ═════════════════════════════════════════════════════════════════════════════

class HarmonicsNoiseDecoder(Decoder):
    """DDSP Harmonics + Noise decoder from Engel et al. (2020)."""

    def __init__(
            self,
            fs: int = 16000,
            num_samples: int = 64000,
            n_harmonics: int = 100,
            n_noise_bands: int = 65,
            use_external_detectors: bool = True,
            z_dim: int = 16,
            hidden_dim: int = 512,
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
        self.z_dim = z_dim

        self.num_params = n_harmonics + n_noise_bands
        self.param_names = ("amplitude", "harmonic_distribution", "noise_magnitudes", "f0")

        self.f0_mlp = DDSPMLP(1, hidden_dim, num_layers=3)
        self.loudness_mlp = DDSPMLP(1, hidden_dim, num_layers=3)
        self.z_mlp = DDSPMLP(z_dim, hidden_dim, num_layers=3)

        self.rnn = nn.GRU(hidden_dim * 3, hidden_dim, batch_first=True)
        self.out_mlp = DDSPMLP(hidden_dim * 3, hidden_dim, num_layers=3)

        self.proj_amp = nn.Linear(hidden_dim, 1)
        self.proj_harm = nn.Linear(hidden_dim, n_harmonics)
        self.proj_noise = nn.Linear(hidden_dim, n_noise_bands)

    def activate(
            self,
            raw: torch.Tensor, # [B, z_dim, T]
            detected: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if detected is None or "f0" not in detected or "loudness" not in detected:
            raise ValueError("DDSP Decoder requires 'f0' and 'loudness' in the detected dict.")

        z = raw.permute(0, 2, 1) # [B, T, z_dim]
        f0 = detected["f0"].unsqueeze(-1)  # [B, T, 1]
        loudness = detected["loudness"].unsqueeze(-1)  # [B, T, 1]

        # Log scale the F0 to keep values small for the MLPs
        f0_scaled = torch.log2(f0 / 440.0 + 1e-5)

        f0_emb = self.f0_mlp(f0_scaled)
        loud_emb = self.loudness_mlp(loudness)
        z_emb = self.z_mlp(z)

        gru_in = torch.cat([f0_emb, loud_emb, z_emb], dim=-1)
        gru_out, _ = self.rnn(gru_in)

        x_out = torch.cat([f0_emb, loud_emb, gru_out], dim=-1)
        features = self.out_mlp(x_out)

        amp = _modified_sigmoid(self.proj_amp(features)).squeeze(-1)
        harm_dist = _modified_sigmoid(self.proj_harm(features))
        noise_mag = _modified_sigmoid(self.proj_noise(features))

        return {
            "f0": detected["f0"],
            "amplitude": amp,
            "harmonic_distribution": harm_dist,
            "noise_magnitudes": noise_mag
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
        noise = self._noise_synth(noise_mag, block_size)

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