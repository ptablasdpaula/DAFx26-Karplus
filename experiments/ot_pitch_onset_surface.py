#!/usr/bin/env python3
"""Full time--log-pitch 2D-OT surface around a diagonal KS target."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.ot_pitch_onset_gradient import (
    ExperimentConfig,
    StaticKSRenderer,
    require_cuda_extension,
)
from src.tf_ot import DifferentiableTFOT, TFOTConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "outputs" / "ot-pitch-onset-surface"


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
    ).strip()


def grids(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    onsets = np.linspace(0.0, config.latest_onset_seconds, config.points)
    pitches = np.geomspace(
        config.minimum_pitch_hz, config.maximum_pitch_hz, config.points,
    )
    # A straight diagonal in time versus log-frequency.
    target_pitches = pitches.copy()
    return onsets, pitches, target_pitches


def render_target_magnitudes(
    renderer: StaticKSRenderer,
    metric: DifferentiableTFOT,
    onsets: np.ndarray,
    target_pitches: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    result = []
    with torch.no_grad():
        for start in range(0, len(onsets), batch_size):
            stop = min(start + batch_size, len(onsets))
            onset = torch.as_tensor(onsets[start:stop], device=device, dtype=torch.float32)
            pitch = torch.as_tensor(target_pitches[start:stop], device=device, dtype=torch.float32)
            result.append(metric.magnitude_spectrogram(renderer(onset, pitch)))
    return torch.cat(result, dim=0)


def run_shard(
    config: ExperimentConfig,
    output: Path,
    device: torch.device,
    task_index: int,
    tasks: int,
    batch_size: int,
) -> None:
    if device.type == "cuda":
        require_cuda_extension()
    output.mkdir(parents=True, exist_ok=True)
    onsets, pitches, target_pitches = grids(config)
    rows = np.array_split(np.arange(config.points), tasks)[task_index]
    renderer = StaticKSRenderer(config).to(device)
    ot_config = TFOTConfig(sample_rate=config.sample_rate)
    metric = DifferentiableTFOT(ot_config).to(device)
    targets = render_target_magnitudes(
        renderer, metric, onsets, target_pitches, device, batch_size,
    )

    shape = (len(rows), config.points)
    distance = np.empty(shape, dtype=np.float64)
    onset_gradient = np.empty(shape, dtype=np.float64)
    log2_pitch_gradient = np.empty(shape, dtype=np.float64)

    for local_row, row in enumerate(rows):
        target = targets[row : row + 1]
        for start in range(0, config.points, batch_size):
            stop = min(start + batch_size, config.points)
            count = stop - start
            predicted_onset = torch.full(
                (count,), float(onsets[row]), device=device,
                dtype=torch.float32, requires_grad=True,
            )
            predicted_log2_pitch = torch.tensor(
                np.log2(pitches[start:stop]), device=device,
                dtype=torch.float32, requires_grad=True,
            )
            predicted_pitch = torch.exp2(predicted_log2_pitch)
            magnitude = metric.magnitude_spectrogram(
                renderer(predicted_onset, predicted_pitch),
            )
            target_batch = target.expand(count, -1, -1)
            losses = metric.spectrogram_distances(magnitude, target_batch)
            gradients = torch.autograd.grad(
                losses.sum(), (predicted_onset, predicted_log2_pitch),
            )
            distance[local_row, start:stop] = losses.detach().cpu().numpy()
            onset_gradient[local_row, start:stop] = gradients[0].detach().cpu().numpy()
            log2_pitch_gradient[local_row, start:stop] = gradients[1].detach().cpu().numpy()

        # Guarantee the metric's specified self-comparison value at the target.
        distance[local_row, row] = 0.0
        onset_gradient[local_row, row] = 0.0
        log2_pitch_gradient[local_row, row] = 0.0
        print(f"task {task_index}: row {local_row + 1}/{len(rows)} (global {row})", flush=True)

    for name, value in {
        "distance": distance,
        "onset_gradient": onset_gradient,
        "log2_pitch_gradient": log2_pitch_gradient,
    }.items():
        if not np.isfinite(value).all():
            raise FloatingPointError(f"non-finite {name}")
    np.savez(
        output / f"shard_{task_index}.npz",
        rows=rows,
        onset_seconds=onsets,
        pitch_hz=pitches,
        target_pitch_hz=target_pitches,
        distance=distance,
        onset_gradient=onset_gradient,
        log2_pitch_gradient=log2_pitch_gradient,
    )


def robust_symmetric_norm(values: np.ndarray) -> colors.SymLogNorm:
    finite = np.abs(values[np.isfinite(values) & (values != 0.0)])
    limit = float(np.quantile(finite, 0.99)) if finite.size else 1.0
    linear = max(float(np.quantile(finite, 0.10)), limit * 1e-4) if finite.size else 1e-3
    return colors.SymLogNorm(linthresh=linear, vmin=-limit, vmax=limit, base=10)


def surface_diagnostics(
    onsets: np.ndarray,
    pitches: np.ndarray,
    onset_gradient: np.ndarray,
    log2_pitch_gradient: np.ndarray,
) -> dict[str, float]:
    log_pitch = np.log2(pitches)
    target_log_pitch = log_pitch
    delta = log_pitch[None, :] - target_log_pitch[:, None]
    off_diagonal = ~np.eye(len(onsets), dtype=bool)
    pitch_correct = np.sign(log2_pitch_gradient[off_diagonal]) == np.sign(delta[off_diagonal])
    onset_span = onsets[-1] - onsets[0]
    pitch_span = log_pitch[-1] - log_pitch[0]
    scaled_onset = onset_gradient * onset_span
    scaled_pitch = log2_pitch_gradient * pitch_span
    pitch_share = np.abs(scaled_pitch) / np.hypot(scaled_onset, scaled_pitch).clip(1e-20)
    return {
        "off_diagonal_pitch_descent_toward_target_fraction": float(pitch_correct.mean()),
        "median_normalized_pitch_gradient_share": float(np.median(pitch_share[off_diagonal])),
        "maximum_diagonal_onset_gradient": float(np.abs(np.diag(onset_gradient)).max()),
        "maximum_diagonal_log2_pitch_gradient": float(
            np.abs(np.diag(log2_pitch_gradient)).max()
        ),
    }


def plot_surface(
    output: Path,
    config: ExperimentConfig,
    onsets: np.ndarray,
    pitches: np.ndarray,
    distance: np.ndarray,
    onset_gradient: np.ndarray,
    log2_pitch_gradient: np.ndarray,
) -> None:
    x, y = np.meshgrid(onsets, pitches, indexing="xy")
    # Stored matrices are [time, pitch]; plotting arrays are [pitch, time].
    loss = distance.T
    onset_grad = onset_gradient.T
    pitch_grad = log2_pitch_gradient.T
    positive = loss[loss > 0.0]
    floor = float(positive.min()) * 0.5 if positive.size else 1e-12

    figure, axes = plt.subplots(2, 2, figsize=(14.5, 11), constrained_layout=True)
    image = axes[0, 0].pcolormesh(x, y, np.log10(loss + floor), shading="nearest", cmap="viridis")
    plt.colorbar(image, ax=axes[0, 0], label="log10 2D-OT")
    axes[0, 0].set_title("2D-OT loss surface")

    image = axes[0, 1].pcolormesh(
        x, y, onset_grad, shading="nearest", cmap="coolwarm",
        norm=robust_symmetric_norm(onset_grad),
    )
    plt.colorbar(image, ax=axes[0, 1], label="d(2D-OT) / d onset (s)")
    axes[0, 1].set_title("predicted-onset gradient")

    image = axes[1, 0].pcolormesh(
        x, y, pitch_grad, shading="nearest", cmap="coolwarm",
        norm=robust_symmetric_norm(pitch_grad),
    )
    plt.colorbar(image, ax=axes[1, 0], label="d(2D-OT) / d log2 pitch")
    axes[1, 0].set_title("predicted-log-pitch gradient")

    axis = axes[1, 1]
    step = max(1, config.points // 20)
    rows = np.arange(0, config.points, step)
    columns = np.arange(0, config.points, step)
    xx, yy = np.meshgrid(onsets[rows], pitches[columns], indexing="xy")
    gt = onset_gradient[np.ix_(rows, columns)].T
    gp = log2_pitch_gradient[np.ix_(rows, columns)].T
    onset_span = onsets[-1] - onsets[0]
    log_pitch_span = math.log2(pitches[-1] / pitches[0])
    normalized_t = gt * onset_span
    normalized_p = gp * log_pitch_span
    norm = np.hypot(normalized_t, normalized_p)
    safe = np.where(norm > 0.0, norm, 1.0)
    arrow_t = -normalized_t / safe * 0.035 * onset_span
    arrow_log_pitch = -normalized_p / safe * 0.035 * log_pitch_span
    arrow_pitch = yy * np.exp2(arrow_log_pitch) - yy
    axis.quiver(
        xx, yy, arrow_t, arrow_pitch, angles="xy", scale_units="xy", scale=1.0,
        color="#202020", alpha=0.75, width=0.0025,
    )
    axis.set_title("normalized local descent field (-gradient)")

    for axis in axes.flat:
        axis.plot(onsets, pitches, color="#f3bd2e", lw=2.2, label="target diagonal")
        axis.set_yscale("log", base=2)
        axis.set_xlim(onsets[0], onsets[-1])
        axis.set_ylim(pitches[0], pitches[-1])
        axis.set_xlabel("onset time (s)")
        axis.set_ylabel("predicted pitch (Hz, log scale)")
        axis.set_yticks([80, 100, 120, 160, 200, 240, 320])
        axis.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[0, 0].legend(loc="upper left")
    figure.suptitle(
        "2D time-frequency OT around a diagonal static Karplus--Strong target\n"
        f"{config.points}×{config.points} grid, {config.sample_rate / 1000:g} kHz, "
        f"g={config.decay}, a1={config.a1}, pluck={config.pluck_position}, "
        f"dynamics={config.dynamics}",
        fontsize=15,
    )
    figure.savefig(output / "ot_pitch_onset_surface.png", dpi=180)
    figure.savefig(output / "ot_pitch_onset_surface.pdf")
    plt.close(figure)


def aggregate(config: ExperimentConfig, output: Path, tasks: int) -> None:
    onsets, pitches, target_pitches = grids(config)
    shape = (config.points, config.points)
    distance = np.empty(shape)
    onset_gradient = np.empty(shape)
    log2_pitch_gradient = np.empty(shape)
    seen = np.zeros(config.points, dtype=bool)
    for task in range(tasks):
        shard = np.load(output / f"shard_{task}.npz")
        rows = shard["rows"]
        distance[rows] = shard["distance"]
        onset_gradient[rows] = shard["onset_gradient"]
        log2_pitch_gradient[rows] = shard["log2_pitch_gradient"]
        seen[rows] = True
    if not seen.all():
        raise RuntimeError("surface shards do not cover every row")

    diagnostics = surface_diagnostics(
        onsets, pitches, onset_gradient, log2_pitch_gradient,
    )
    np.savez(
        output / "surface_data.npz", onset_seconds=onsets, pitch_hz=pitches,
        target_pitch_hz=target_pitches, distance=distance,
        onset_gradient=onset_gradient,
        log2_pitch_gradient=log2_pitch_gradient,
    )
    with (output / "surface_data.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["onset_seconds", "target_pitch_hz", "predicted_pitch_hz", "distance",
             "gradient_onset_seconds", "gradient_log2_pitch"]
        )
        for row, onset in enumerate(onsets):
            for column, pitch in enumerate(pitches):
                writer.writerow(
                    [onset, target_pitches[row], pitch, distance[row, column],
                     onset_gradient[row, column], log2_pitch_gradient[row, column]]
                )
    metadata = {
        "experiment": asdict(config),
        "ot": asdict(TFOTConfig(sample_rate=config.sample_rate)),
        "tasks": tasks,
        "source_commit": git_commit(),
        "diagnostics": diagnostics,
        "note": "At each x, target and prediction share onset; target pitch follows the yellow log-linear diagonal.",
    }
    (output / "configuration.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    plot_surface(
        output, config, onsets, pitches, distance, onset_gradient,
        log2_pitch_gradient,
    )
    print(json.dumps(diagnostics, indent=2))
    print(output / "ot_pitch_onset_surface.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "aggregate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = ExperimentConfig(sample_rate=8_000, duration_seconds=2.0, points=args.points)
    if args.mode == "run":
        run_shard(
            cfg, args.output, torch.device(args.device), args.task_index,
            args.tasks, args.batch_size,
        )
    else:
        aggregate(cfg, args.output, args.tasks)
