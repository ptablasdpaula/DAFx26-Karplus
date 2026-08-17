"""Canonical and differentiable joint time--frequency sliced Wasserstein-2.

The NumPy functions implement the Fabiani--Schlecht--Elvander evaluation
metric using exact one-dimensional discrete W2.  ``DifferentiableTFOT`` uses
the same spectrogram, normalization, coordinates, and ten projections, with a
fixed 512-point differentiable inverse-CDF quadrature for optimization.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

TIME_SCALE_HZ_PER_SECOND = 1_000.0
PROJECTION_COUNT = 10
QUANTILE_COUNT = 512


@dataclass(frozen=True)
class TFOTConfig:
    sample_rate: int = 16_000
    n_fft: int = 1024
    hop_length: int = 256
    projections: int = PROJECTION_COUNT
    quantiles: int = QUANTILE_COUNT
    time_scale_hz_per_second: float = TIME_SCALE_HZ_PER_SECOND
    silence_floor: float = 1e-12


@lru_cache(maxsize=32)
def _numpy_projection_geometry(
    frequency_bins: int,
    time_bins: int,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    projections: int,
    time_scale: float,
) -> tuple[tuple[tuple[np.ndarray, np.ndarray], ...], float]:
    frequency = (
        np.arange(frequency_bins, dtype=np.float64) * sample_rate / n_fft
    )
    scaled_time = (
        np.arange(time_bins, dtype=np.float64)
        * hop_length
        / sample_rate
        * time_scale
    )
    frequency_coord = np.repeat(frequency, time_bins)
    time_coord = np.tile(scaled_time, frequency_bins)
    angles = 2.0 * np.pi * np.arange(projections) / projections
    geometry = []
    for angle in angles:
        positions = time_coord * np.cos(angle) + frequency_coord * np.sin(angle)
        order = np.argsort(positions, kind="stable")
        geometry.append((order, positions[order]))
    maximum = float(
        np.hypot(
            scaled_time[-1] if scaled_time.size else 0.0,
            frequency[-1] if frequency.size else 0.0,
        )
    )
    return tuple(geometry), maximum


def _numpy_wasserstein_2(
    weights_a: np.ndarray,
    weights_b: np.ndarray,
    positions: np.ndarray,
) -> float:
    cumulative_a = np.cumsum(weights_a, dtype=np.float64)
    cumulative_b = np.cumsum(weights_b, dtype=np.float64)
    cumulative_a[-1] = cumulative_b[-1] = 1.0
    quantiles = np.sort(np.concatenate((cumulative_a, cumulative_b)))
    index_a = np.minimum(
        np.searchsorted(cumulative_a, quantiles, side="left"), len(positions) - 1,
    )
    index_b = np.minimum(
        np.searchsorted(cumulative_b, quantiles, side="left"), len(positions) - 1,
    )
    widths = np.diff(np.concatenate(([0.0], quantiles)))
    squared = np.sum(widths * np.square(positions[index_a] - positions[index_b]))
    return float(np.sqrt(max(float(squared), 0.0)))


def canonical_tf_ot_spectrogram(
    magnitude_a: np.ndarray,
    magnitude_b: np.ndarray,
    *,
    config: TFOTConfig | None = None,
) -> float:
    """Exact canonical metric on matching nonnegative ``[frequency, time]`` arrays."""
    cfg = config or TFOTConfig()
    a = np.maximum(np.asarray(magnitude_a, np.float64), 0.0)
    b = np.maximum(np.asarray(magnitude_b, np.float64), 0.0)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError(f"expected matching [frequency, time], got {a.shape}, {b.shape}")
    mass_a, mass_b = float(a.sum()), float(b.sum())
    geometry, maximum = _numpy_projection_geometry(
        a.shape[0],
        a.shape[1],
        cfg.sample_rate,
        cfg.n_fft,
        cfg.hop_length,
        cfg.projections,
        cfg.time_scale_hz_per_second,
    )
    silent_a = mass_a <= cfg.silence_floor
    silent_b = mass_b <= cfg.silence_floor
    if silent_a and silent_b:
        return 0.0
    if silent_a != silent_b:
        return maximum
    weights_a = a.reshape(-1) / mass_a
    weights_b = b.reshape(-1) / mass_b
    if np.allclose(weights_a, weights_b, rtol=1e-6, atol=1e-14):
        return 0.0
    distance = float(
        np.mean(
            [
                _numpy_wasserstein_2(
                    weights_a[order], weights_b[order], positions,
                )
                for order, positions in geometry
            ]
        )
    )
    return 0.0 if distance <= 1e-5 * maximum else distance


def canonical_tf_ot_audio(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    *,
    config: TFOTConfig | None = None,
) -> float:
    """Exact canonical metric on two mono waveforms."""
    cfg = config or TFOTConfig()
    import librosa

    def magnitude(audio: np.ndarray) -> np.ndarray:
        return np.abs(
            librosa.stft(
                np.asarray(audio, np.float32).reshape(-1),
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                win_length=cfg.n_fft,
                window="hann",
                center=True,
                pad_mode="constant",
            )
        )

    return canonical_tf_ot_spectrogram(
        magnitude(audio_a), magnitude(audio_b), config=cfg,
    )


class DifferentiableTFOT(nn.Module):
    """Differentiable 512-quantile approximation of the canonical metric."""

    def __init__(self, config: TFOTConfig | None = None):
        super().__init__()
        self.config = config or TFOTConfig()
        if self.config.projections <= 0 or self.config.quantiles <= 1:
            raise ValueError("projections must be positive and quantiles must exceed one")
        self.register_buffer(
            "window", torch.hann_window(self.config.n_fft, periodic=True),
        )
        # Midpoint quadrature avoids evaluating the unstable inverse CDF at 0 or 1.
        self.register_buffer(
            "quantile_levels",
            (torch.arange(self.config.quantiles, dtype=torch.float64) + 0.5)
            / self.config.quantiles,
        )
        self._geometry_key: tuple[int, torch.device, torch.dtype] | None = None
        self._orders: Tensor | None = None
        self._positions: Tensor | None = None
        self._maximum: Tensor | None = None

    def magnitude_spectrogram(self, audio: Tensor) -> Tensor:
        if audio.ndim != 2:
            raise ValueError(f"expected [batch, samples] mono audio, got {audio.shape}")
        spectrum = torch.stft(
            audio,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=self.window.to(device=audio.device, dtype=audio.dtype),
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        return spectrum.abs()

    def _geometry(
        self, time_bins: int, device: torch.device, dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor]:
        key = (time_bins, device, dtype)
        if self._geometry_key == key:
            assert self._orders is not None
            assert self._positions is not None
            assert self._maximum is not None
            return self._orders, self._positions, self._maximum

        cfg = self.config
        frequency = (
            torch.arange(cfg.n_fft // 2 + 1, device=device, dtype=dtype)
            * cfg.sample_rate
            / cfg.n_fft
        )
        scaled_time = (
            torch.arange(time_bins, device=device, dtype=dtype)
            * cfg.hop_length
            / cfg.sample_rate
            * cfg.time_scale_hz_per_second
        )
        frequency_coord = frequency.repeat_interleave(time_bins)
        time_coord = scaled_time.repeat(frequency.numel())
        angles = (
            2.0
            * torch.pi
            * torch.arange(cfg.projections, device=device, dtype=dtype)
            / cfg.projections
        )
        positions = (
            time_coord.unsqueeze(0) * torch.cos(angles).unsqueeze(1)
            + frequency_coord.unsqueeze(0) * torch.sin(angles).unsqueeze(1)
        )
        orders = torch.argsort(positions, dim=-1, stable=True)
        sorted_positions = torch.gather(positions, 1, orders)
        maximum = torch.hypot(scaled_time[-1], frequency[-1])
        self._geometry_key = key
        self._orders = orders
        self._positions = sorted_positions
        self._maximum = maximum
        return orders, sorted_positions, maximum

    @staticmethod
    def _weighted_quantiles(weights: Tensor, positions: Tensor, levels: Tensor) -> Tensor:
        """Piecewise-linear inverse of the weighted mid-CDF.

        Search indices are piecewise constant, while interpolation through the
        cumulative masses supplies useful gradients away from those measure-zero
        boundaries.
        """
        midpoint_cdf = torch.cumsum(weights, dim=-1) - 0.5 * weights
        levels = levels.to(device=weights.device, dtype=weights.dtype)
        indices_hi = torch.searchsorted(
            midpoint_cdf.contiguous(), levels.contiguous(), right=False,
        )
        last = weights.shape[-1] - 1
        indices_hi = indices_hi.clamp(0, last)
        indices_lo = (indices_hi - 1).clamp(0, last)

        cdf_lo = torch.gather(midpoint_cdf, -1, indices_lo)
        cdf_hi = torch.gather(midpoint_cdf, -1, indices_hi)
        pos_lo = torch.gather(positions, -1, indices_lo)
        pos_hi = torch.gather(positions, -1, indices_hi)
        alpha = (levels - cdf_lo) / (cdf_hi - cdf_lo).clamp_min(1e-12)
        interpolated = pos_lo + alpha.clamp(0.0, 1.0) * (pos_hi - pos_lo)
        interpolated = torch.where(indices_hi == 0, positions[..., :1], interpolated)
        interpolated = torch.where(
            levels > midpoint_cdf[..., -1:], positions[..., -1:], interpolated,
        )
        return interpolated

    def spectrogram_distance(self, magnitude_a: Tensor, magnitude_b: Tensor) -> Tensor:
        if magnitude_a.ndim != 3 or magnitude_a.shape != magnitude_b.shape:
            raise ValueError(
                "expected matching [batch, frequency, time] magnitude spectrograms"
            )
        if torch.equal(magnitude_a, magnitude_b):
            # Explicitly select the zero subgradient at exact self-comparison.
            return (magnitude_a.sum() + magnitude_b.sum()) * 0.0

        batch, _, time_bins = magnitude_a.shape
        orders, positions, maximum = self._geometry(
            time_bins, magnitude_a.device, magnitude_a.dtype,
        )
        flat_a = magnitude_a.clamp_min(0.0).reshape(batch, -1)
        flat_b = magnitude_b.clamp_min(0.0).reshape(batch, -1)
        mass_a = flat_a.sum(dim=-1, keepdim=True)
        mass_b = flat_b.sum(dim=-1, keepdim=True)
        distances = []
        for batch_index in range(batch):
            silent_a = bool(mass_a[batch_index] <= self.config.silence_floor)
            silent_b = bool(mass_b[batch_index] <= self.config.silence_floor)
            zero_link = (flat_a[batch_index].sum() + flat_b[batch_index].sum()) * 0.0
            if silent_a and silent_b:
                distances.append(zero_link)
                continue
            if silent_a != silent_b:
                distances.append(maximum + zero_link)
                continue

            weights_a = flat_a[batch_index] / mass_a[batch_index]
            weights_b = flat_b[batch_index] / mass_b[batch_index]
            # Scalar-gain copies can retain a few float32 ulps after global
            # normalization.  Collapse that residue just as the canonical
            # evaluation implementation does, selecting the zero subgradient.
            if torch.allclose(weights_a, weights_b, rtol=1e-6, atol=1e-12):
                distances.append(zero_link)
                continue
            projection_distances = []
            for projection in range(self.config.projections):
                order = orders[projection]
                projected_positions = positions[projection]
                sorted_a = weights_a[order]
                sorted_b = weights_b[order]
                position_grid = projected_positions.unsqueeze(0)
                levels = self.quantile_levels.unsqueeze(0)
                quantiles_a = self._weighted_quantiles(
                    sorted_a.unsqueeze(0), position_grid, levels,
                )
                quantiles_b = self._weighted_quantiles(
                    sorted_b.unsqueeze(0), position_grid, levels,
                )
                squared_distance = (quantiles_a - quantiles_b).square().mean()
                # Several fixed slices can be exactly orthogonal to a real
                # displacement.  sqrt has an infinite derivative at zero, so
                # explicitly choose the zero subgradient for those slices.
                safe_root = torch.sqrt(
                    squared_distance.clamp_min(torch.finfo(squared_distance.dtype).tiny)
                )
                projection_distances.append(
                    torch.where(
                        squared_distance > 0.0,
                        safe_root,
                        squared_distance * 0.0,
                    )
                )
            distances.append(torch.stack(projection_distances).mean())
        return torch.stack(distances).mean()

    def forward(self, audio_a: Tensor, audio_b: Tensor) -> Tensor:
        if audio_a.shape != audio_b.shape or audio_a.ndim != 2:
            raise ValueError(f"expected matching [batch, samples], got {audio_a.shape}, {audio_b.shape}")
        if torch.equal(audio_a, audio_b):
            return (audio_a.sum() + audio_b.sum()) * 0.0
        return self.spectrogram_distance(
            self.magnitude_spectrogram(audio_a),
            self.magnitude_spectrogram(audio_b),
        )
