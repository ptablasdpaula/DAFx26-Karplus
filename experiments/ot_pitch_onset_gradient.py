#!/usr/bin/env python3
"""Plot the 2D-OT gradient field for static KS pitch/onset pairs.

Reference and prediction sounds lie on the diagonal joining an early, low
pluck to a late, high pluck.  For every reference/prediction pair, this script
records the differentiable 2D-OT distance and its gradients with respect to
the predicted onset (seconds) and pitch (Hz).

The excitation is a fixed 256-sample noise burst (one OT hop), placed with a
zero-padded Fourier fractional delay.  This retains the event renderer's
smooth onset derivative while avoiding its pitch-dependent burst length and
circular time shift, either of which would confound the gradient measurement.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from src.synths.ddsp import (
    dynamics_filter,
    karplus_strong_from_delay,
    pluck_position_filter,
)
from src.tf_ot import DifferentiableTFOT, TFOTConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "outputs" / "ot-pitch-onset-gradient"


@dataclass(frozen=True)
class ExperimentConfig:
    sample_rate: int = 8_000
    duration_seconds: float = 2.0
    points: int = 13
    minimum_pitch_hz: float = 80.0
    maximum_pitch_hz: float = 320.0
    decay: float = 0.999
    a1: float = 0.001
    pluck_position: float = 0.2
    dynamics: float = 1.0
    burst_samples: int = 256
    noise_seed: int = 42

    @property
    def num_samples(self) -> int:
        return round(self.sample_rate * self.duration_seconds)

    @property
    def latest_onset_seconds(self) -> float:
        # A burst beginning at duration_seconds would be silent after cropping.
        # Stop one complete burst earlier while retaining the requested [0, 2 s)
        # onset range.
        return self.duration_seconds - self.burst_samples / self.sample_rate


class StaticKSRenderer(torch.nn.Module):
    """Static-control time-domain KS renderer with differentiable onset."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.noise_seed)
        burst = torch.randn(config.burst_samples, generator=generator)
        burst = burst - burst.mean()
        burst = burst / burst.square().mean().sqrt().clamp_min(1e-12)
        fft_length = 1 << math.ceil(
            math.log2(config.num_samples + config.burst_samples)
        )
        padded = torch.nn.functional.pad(
            burst, (0, fft_length - config.burst_samples),
        )
        self.fft_length = fft_length
        self.register_buffer("burst_spectrum", torch.fft.rfft(padded))
        self.register_buffer(
            "angular_frequency",
            2.0
            * torch.pi
            * torch.arange(fft_length // 2 + 1, dtype=torch.float32)
            / fft_length,
        )

    def _fractionally_shifted_burst(self, onset_seconds: Tensor) -> Tensor:
        """Causally place the burst using a zero-padded Fourier delay."""
        onset_samples = onset_seconds.reshape(-1, 1) * self.config.sample_rate
        phase = torch.exp(
            -1j * self.angular_frequency.unsqueeze(0) * onset_samples,
        )
        shifted = torch.fft.irfft(
            self.burst_spectrum.unsqueeze(0) * phase, n=self.fft_length,
        )
        return shifted[:, : self.config.num_samples]

    def forward(self, onset_seconds: Tensor, pitch_hz: Tensor) -> Tensor:
        cfg = self.config
        excitation = self._fractionally_shifted_burst(onset_seconds)
        pitch_hz = pitch_hz.reshape(-1)
        if pitch_hz.shape[0] != excitation.shape[0]:
            raise ValueError("onset and pitch batch sizes must match")
        f0 = pitch_hz.reshape(-1, 1).expand(-1, cfg.num_samples)
        delay = cfg.sample_rate / f0
        decay = torch.full_like(f0, cfg.decay)
        a1 = torch.full_like(f0, cfg.a1)
        position = torch.full_like(f0, cfg.pluck_position)
        dynamics = torch.full_like(f0, cfg.dynamics)
        excitation = dynamics_filter(
            excitation, f0=f0, dynamic_level=dynamics, fs=cfg.sample_rate,
        )
        excitation = pluck_position_filter(
            excitation, f0=f0, position=position, fs=cfg.sample_rate,
        )
        return karplus_strong_from_delay(
            excitation, delay=delay, a1=a1, g=decay, fs=cfg.sample_rate,
        )


def require_cuda_extension() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    import torchlpc

    if getattr(torchlpc, "EXTENSION_LOADED", False) is not True:
        raise RuntimeError("torchlpc CUDA extension is not loaded")


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
    ).strip()


def symmetric_limit(values: np.ndarray) -> float:
    finite = np.abs(values[np.isfinite(values)])
    nonzero = finite[finite > 0.0]
    if nonzero.size == 0:
        return 1.0
    return max(float(np.quantile(nonzero, 0.98)), np.finfo(np.float32).eps)


def draw_matrix(
    axis: plt.Axes,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    labels: list[str],
    *,
    diverging: bool,
) -> None:
    if diverging:
        limit = symmetric_limit(values)
        displayed = np.clip(values, -limit, limit)
        image = axis.imshow(
            displayed, origin="lower", aspect="auto", cmap="coolwarm",
            vmin=-limit, vmax=limit,
        )
    else:
        positive = values[values > 0.0]
        floor = float(positive.min()) * 0.5 if positive.size else 1e-12
        displayed = np.log10(values + floor)
        image = axis.imshow(displayed, origin="lower", aspect="auto", cmap="viridis")
    axis.plot([-0.5, len(labels) - 0.5], [-0.5, len(labels) - 0.5], "w--", lw=1.2)
    step = max(1, math.ceil(len(labels) / 7))
    ticks = np.arange(0, len(labels), step)
    axis.set_xticks(ticks, [labels[index] for index in ticks], rotation=45, ha="right")
    axis.set_yticks(ticks, [labels[index] for index in ticks])
    axis.set_xlabel("predicted onset / pitch")
    axis.set_ylabel("target onset / pitch")
    axis.set_title(title)
    plt.colorbar(image, ax=axis, label=colorbar_label, fraction=0.046, pad=0.04)


def draw_descent_panel(
    axis: plt.Axes,
    target_index: int,
    onsets: np.ndarray,
    pitches: np.ndarray,
    onset_gradient: np.ndarray,
    pitch_gradient: np.ndarray,
) -> None:
    onset_span = float(onsets[-1] - onsets[0])
    pitch_span = float(pitches[-1] - pitches[0])
    # Convert both derivatives to normalized-coordinate derivatives before
    # normalizing each arrow.  Arrow direction is therefore meaningful even
    # though seconds and Hz have different numerical scales.
    normalized_onset = onset_gradient[target_index] * onset_span
    normalized_pitch = pitch_gradient[target_index] * pitch_span
    norm = np.hypot(normalized_onset, normalized_pitch)
    safe = np.where(norm > 0.0, norm, 1.0)
    arrow_onset = -normalized_onset / safe * 0.10 * onset_span
    arrow_pitch = -normalized_pitch / safe * 0.10 * pitch_span
    arrow_onset[norm == 0.0] = 0.0
    arrow_pitch[norm == 0.0] = 0.0

    axis.plot(onsets, pitches, color="0.75", lw=1.5, zorder=1)
    axis.scatter(onsets, pitches, c="0.25", s=18, zorder=2)
    axis.quiver(
        onsets, pitches, arrow_onset, arrow_pitch,
        angles="xy", scale_units="xy", scale=1.0, color="#ca3b37",
        width=0.006, headwidth=4.0, zorder=3,
    )
    axis.scatter(
        [onsets[target_index]], [pitches[target_index]], marker="*", s=180,
        c="#f2bd2e", edgecolors="black", linewidths=0.8, zorder=4,
    )
    axis.set_xlim(-0.04 * onset_span, onsets[-1] + 0.04 * onset_span)
    axis.set_ylim(pitches[0] - 0.07 * pitch_span, pitches[-1] + 0.07 * pitch_span)
    axis.set_xlabel("predicted onset (s)")
    axis.set_ylabel("predicted pitch (Hz)")
    axis.set_title(
        f"local normalized descent direction\n"
        f"target: {onsets[target_index]:.3f} s, {pitches[target_index]:.0f} Hz"
    )
    axis.grid(alpha=0.2)


def save_figure(
    output: Path,
    config: ExperimentConfig,
    onsets: np.ndarray,
    pitches: np.ndarray,
    distances: np.ndarray,
    onset_gradient: np.ndarray,
    pitch_gradient: np.ndarray,
) -> None:
    labels = [f"{time:.2f}s\n{pitch:.0f}Hz" for time, pitch in zip(onsets, pitches)]
    figure, axes = plt.subplots(2, 3, figsize=(17, 10.5), constrained_layout=True)
    draw_matrix(
        axes[0, 0], distances, "2D-OT along the pitch/onset diagonal",
        "log10 2D-OT", labels, diverging=False,
    )
    draw_matrix(
        axes[0, 1], onset_gradient, "onset gradient",
        "d(2D-OT) / d onset (s)", labels, diverging=True,
    )
    draw_matrix(
        axes[0, 2], pitch_gradient, "pitch gradient",
        "d(2D-OT) / d pitch (Hz)", labels, diverging=True,
    )
    anchors = [0, len(onsets) // 2, len(onsets) - 1]
    for axis, target_index in zip(axes[1], anchors):
        draw_descent_panel(
            axis, target_index, onsets, pitches, onset_gradient, pitch_gradient,
        )
    figure.suptitle(
        "Differentiable 2D time-frequency OT: static Karplus--Strong onset/pitch gradients\n"
        f"{config.sample_rate / 1000:g} kHz, g={config.decay}, a1={config.a1}, "
        f"pluck={config.pluck_position}, dynamics={config.dynamics}, "
        f"fixed {config.burst_samples}-sample noise burst\n"
        "Stars mark exact self-comparisons (zero subgradient); arrows show normalized -gradient",
        fontsize=15,
    )
    figure.savefig(output, dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def run(config: ExperimentConfig, output_dir: Path, device: torch.device) -> None:
    if config.points < 3:
        raise ValueError("at least three diagonal points are required")
    if config.latest_onset_seconds <= 0.0:
        raise ValueError("the burst must be shorter than the audio")
    if device.type == "cuda":
        require_cuda_extension()

    output_dir.mkdir(parents=True, exist_ok=True)
    renderer = StaticKSRenderer(config).to(device)
    ot_config = TFOTConfig(
        sample_rate=config.sample_rate, n_fft=1024, hop_length=256,
        projections=10, quantiles=512,
    )
    metric = DifferentiableTFOT(ot_config).to(device)
    onsets = np.linspace(0.0, config.latest_onset_seconds, config.points)
    pitches = np.linspace(config.minimum_pitch_hz, config.maximum_pitch_hz, config.points)

    target_magnitudes = []
    with torch.no_grad():
        for onset, pitch in zip(onsets, pitches):
            audio = renderer(
                torch.tensor(onset, device=device, dtype=torch.float32),
                torch.tensor(pitch, device=device, dtype=torch.float32),
            )
            target_magnitudes.append(metric.magnitude_spectrogram(audio))
    target_magnitudes_tensor = torch.cat(target_magnitudes, dim=0)

    shape = (config.points, config.points)
    distances = np.empty(shape, dtype=np.float64)
    onset_gradient = np.empty(shape, dtype=np.float64)
    pitch_gradient = np.empty(shape, dtype=np.float64)

    for prediction_index, (onset, pitch) in enumerate(zip(onsets, pitches)):
        predicted_onset = torch.tensor(
            onset, device=device, dtype=torch.float32, requires_grad=True,
        )
        predicted_pitch = torch.tensor(
            pitch, device=device, dtype=torch.float32, requires_grad=True,
        )
        predicted_audio = renderer(predicted_onset, predicted_pitch)
        predicted_magnitude = metric.magnitude_spectrogram(predicted_audio)
        for target_index in range(config.points):
            loss = metric.spectrogram_distance(
                predicted_magnitude,
                target_magnitudes_tensor[target_index : target_index + 1],
            )
            gradient = torch.autograd.grad(
                loss, (predicted_onset, predicted_pitch),
                retain_graph=target_index < config.points - 1,
            )
            distances[target_index, prediction_index] = float(loss.detach().cpu())
            onset_gradient[target_index, prediction_index] = float(gradient[0].detach().cpu())
            pitch_gradient[target_index, prediction_index] = float(gradient[1].detach().cpu())
        print(f"prediction {prediction_index + 1}/{config.points}", flush=True)

    for name, values in {
        "distance": distances,
        "onset gradient": onset_gradient,
        "pitch gradient": pitch_gradient,
    }.items():
        if not np.isfinite(values).all():
            raise FloatingPointError(f"non-finite {name}")
    diagonal = np.arange(config.points)
    if not np.allclose(distances[diagonal, diagonal], 0.0, atol=1e-5):
        raise AssertionError("self-comparison distances are not zero")
    if not np.allclose(onset_gradient[diagonal, diagonal], 0.0, atol=1e-7):
        raise AssertionError("self-comparison onset gradients are not zero")
    if not np.allclose(pitch_gradient[diagonal, diagonal], 0.0, atol=1e-7):
        raise AssertionError("self-comparison pitch gradients are not zero")

    np.savez(
        output_dir / "gradient_data.npz",
        onset_seconds=onsets,
        pitch_hz=pitches,
        distance=distances,
        onset_gradient=onset_gradient,
        pitch_gradient=pitch_gradient,
    )
    with (output_dir / "gradient_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_index", "prediction_index", "target_onset_seconds",
                "target_pitch_hz", "predicted_onset_seconds", "predicted_pitch_hz",
                "distance", "gradient_onset_seconds", "gradient_pitch_hz",
            ],
        )
        writer.writeheader()
        for target_index in range(config.points):
            for prediction_index in range(config.points):
                writer.writerow(
                    {
                        "target_index": target_index,
                        "prediction_index": prediction_index,
                        "target_onset_seconds": onsets[target_index],
                        "target_pitch_hz": pitches[target_index],
                        "predicted_onset_seconds": onsets[prediction_index],
                        "predicted_pitch_hz": pitches[prediction_index],
                        "distance": distances[target_index, prediction_index],
                        "gradient_onset_seconds": onset_gradient[target_index, prediction_index],
                        "gradient_pitch_hz": pitch_gradient[target_index, prediction_index],
                    }
                )
    metadata = {
        "experiment": asdict(config),
        "ot": asdict(ot_config),
        "source_commit": source_commit(),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "note": "The latest onset is one complete burst before 2 s; an onset at exactly 2 s is silent after cropping.",
    }
    (output_dir / "configuration.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    save_figure(
        output_dir / "ot_pitch_onset_gradient.png",
        config, onsets, pitches, distances, onset_gradient, pitch_gradient,
    )
    print(output_dir / "ot_pitch_onset_gradient.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--points", type=int, default=13)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=8_000)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        ExperimentConfig(
            sample_rate=arguments.sample_rate,
            points=arguments.points,
            duration_seconds=arguments.duration,
        ),
        arguments.output,
        torch.device(arguments.device),
    )
