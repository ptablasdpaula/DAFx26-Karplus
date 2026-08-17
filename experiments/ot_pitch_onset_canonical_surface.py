#!/usr/bin/env python3
"""Exact canonical TF sliced-OT on the 100x100 KS time--pitch grid."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit, prange, set_num_threads
import numpy as np
import torch

from experiments.ot_pitch_onset_gradient import (
    ExperimentConfig,
    StaticKSRenderer,
    require_cuda_extension,
)
from experiments.ot_pitch_onset_surface import grids, render_target_magnitudes
from src.tf_ot import (
    DifferentiableTFOT,
    TFOTConfig,
    _numpy_projection_geometry,
    canonical_tf_ot_spectrogram,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "outputs" / "ot-pitch-onset-canonical"
DIFFERENTIABLE_OUTPUT = (
    PROJECT_ROOT / "experiments" / "outputs" / "ot-pitch-onset-surface" / "surface_data.npz"
)


@njit(cache=True, parallel=True)
def exact_sliced_w2_bank(
    prediction_weights: np.ndarray,
    target_weights: np.ndarray,
    orders: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Exact discrete 1D W2 for every projection and prediction.

    This is the linear-time two-pointer form of the same quantile coupling used
    by ``_numpy_wasserstein_2``.  It avoids repeatedly sorting the union of two
    already ordered CDFs, without introducing a quadrature approximation.
    """
    batch, support = prediction_weights.shape
    projections = orders.shape[0]
    result = np.empty(batch, dtype=np.float64)
    for batch_index in prange(batch):
        projection_sum = 0.0
        for projection in range(projections):
            order = orders[projection]
            coordinate = positions[projection]
            prediction_index = 0
            target_index = 0
            prediction_mass = prediction_weights[batch_index, order[0]]
            target_mass = target_weights[order[0]]
            squared_cost = 0.0
            while prediction_index < support and target_index < support:
                if prediction_mass <= 0.0:
                    prediction_index += 1
                    if prediction_index < support:
                        prediction_mass = prediction_weights[
                            batch_index, order[prediction_index]
                        ]
                    continue
                if target_mass <= 0.0:
                    target_index += 1
                    if target_index < support:
                        target_mass = target_weights[order[target_index]]
                    continue
                transported = min(prediction_mass, target_mass)
                displacement = coordinate[prediction_index] - coordinate[target_index]
                squared_cost += transported * displacement * displacement
                prediction_mass -= transported
                target_mass -= transported
            projection_sum += np.sqrt(max(squared_cost, 0.0))
        result[batch_index] = projection_sum / projections
    return result


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
    ).strip()


def normalized_weights(magnitude: np.ndarray) -> np.ndarray:
    flat = np.maximum(np.asarray(magnitude, dtype=np.float64), 0.0).reshape(
        magnitude.shape[0], -1,
    )
    mass = flat.sum(axis=1, keepdims=True)
    if np.any(mass <= 1e-12):
        raise ValueError("the canonical surface expects non-silent renders")
    return flat / mass


def geometry_arrays(config: TFOTConfig, time_bins: int) -> tuple[np.ndarray, np.ndarray]:
    geometry, _ = _numpy_projection_geometry(
        config.n_fft // 2 + 1,
        time_bins,
        config.sample_rate,
        config.n_fft,
        config.hop_length,
        config.projections,
        config.time_scale_hz_per_second,
    )
    orders = np.stack([item[0] for item in geometry]).astype(np.int64)
    positions = np.stack([item[1] for item in geometry]).astype(np.float64)
    return orders, positions


def run_shard(
    config: ExperimentConfig,
    output: Path,
    device: torch.device,
    task_index: int,
    tasks: int,
    render_batch_size: int,
    cpu_threads: int,
) -> None:
    if device.type == "cuda":
        require_cuda_extension()
    set_num_threads(cpu_threads)
    output.mkdir(parents=True, exist_ok=True)
    onsets, pitches, target_pitches = grids(config)
    rows = np.array_split(np.arange(config.points), tasks)[task_index]
    renderer = StaticKSRenderer(config).to(device)
    ot_config = TFOTConfig(sample_rate=config.sample_rate)
    spectrogram = DifferentiableTFOT(ot_config).to(device)
    target_magnitude = render_target_magnitudes(
        renderer, spectrogram, onsets, target_pitches, device, render_batch_size,
    ).cpu().numpy()
    target_weights = normalized_weights(target_magnitude)
    orders, positions = geometry_arrays(ot_config, target_magnitude.shape[-1])

    distance = np.empty((len(rows), config.points), dtype=np.float64)
    validated = False
    for local_row, row in enumerate(rows):
        prediction_magnitudes = []
        with torch.no_grad():
            for start in range(0, config.points, render_batch_size):
                stop = min(start + render_batch_size, config.points)
                onset = torch.full(
                    (stop - start,), float(onsets[row]), device=device,
                    dtype=torch.float32,
                )
                pitch = torch.as_tensor(
                    pitches[start:stop], device=device, dtype=torch.float32,
                )
                prediction_magnitudes.append(
                    spectrogram.magnitude_spectrogram(renderer(onset, pitch)).cpu().numpy()
                )
        prediction_magnitude = np.concatenate(prediction_magnitudes, axis=0)
        prediction_weights = normalized_weights(prediction_magnitude)
        distance[local_row] = exact_sliced_w2_bank(
            prediction_weights, target_weights[row], orders, positions,
        )
        distance[local_row, row] = 0.0

        if not validated:
            # Verify the accelerated coupling against the repository's literal
            # canonical evaluator on one non-self pair in every shard.
            column = 0 if row != 0 else config.points - 1
            reference = canonical_tf_ot_spectrogram(
                prediction_magnitude[column], target_magnitude[row], config=ot_config,
            )
            np.testing.assert_allclose(
                distance[local_row, column], reference, rtol=2e-11, atol=1e-9,
            )
            validated = True
        print(f"task {task_index}: row {local_row + 1}/{len(rows)} (global {row})", flush=True)

    if not np.isfinite(distance).all():
        raise FloatingPointError("non-finite canonical distance")
    np.savez(
        output / f"shard_{task_index}.npz",
        rows=rows,
        onset_seconds=onsets,
        pitch_hz=pitches,
        target_pitch_hz=target_pitches,
        canonical_distance=distance,
    )


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    result = np.empty_like(order, dtype=np.float64)
    result[order] = np.arange(len(values))
    return result


def comparison_metrics(
    pitches: np.ndarray,
    canonical: np.ndarray,
    differentiable: np.ndarray,
) -> dict[str, float]:
    mask = ~np.eye(len(pitches), dtype=bool)
    canonical_flat = canonical[mask]
    differentiable_flat = differentiable[mask]
    absolute_pitch_error = np.abs(
        np.log2(pitches)[None, :] - np.log2(pitches)[:, None]
    )
    row_correlations = []
    differentiable_row_correlations = []
    for row in range(len(pitches)):
        error_rank = rank(absolute_pitch_error[row])
        row_correlations.append(np.corrcoef(error_rank, rank(canonical[row]))[0, 1])
        differentiable_row_correlations.append(
            np.corrcoef(error_rank, rank(differentiable[row]))[0, 1]
        )
    return {
        "pearson_canonical_vs_differentiable": float(
            np.corrcoef(canonical_flat, differentiable_flat)[0, 1]
        ),
        "spearman_canonical_vs_differentiable": float(
            np.corrcoef(rank(canonical_flat), rank(differentiable_flat))[0, 1]
        ),
        "median_row_spearman_canonical_vs_abs_log_pitch_error": float(
            np.median(row_correlations)
        ),
        "median_row_spearman_differentiable_vs_abs_log_pitch_error": float(
            np.median(differentiable_row_correlations)
        ),
        "median_absolute_difference": float(
            np.median(np.abs(canonical_flat - differentiable_flat))
        ),
        "median_relative_difference": float(
            np.median(
                np.abs(canonical_flat - differentiable_flat)
                / np.maximum(canonical_flat, 1e-12)
            )
        ),
    }


def plot_comparison(
    output: Path,
    config: ExperimentConfig,
    onsets: np.ndarray,
    pitches: np.ndarray,
    canonical: np.ndarray,
    differentiable: np.ndarray,
    metrics: dict[str, float],
) -> None:
    x, y = np.meshgrid(onsets, pitches, indexing="xy")
    canonical_plot = canonical.T
    differentiable_plot = differentiable.T
    positive = np.concatenate(
        [canonical_plot[canonical_plot > 0], differentiable_plot[differentiable_plot > 0]]
    )
    floor = float(positive.min()) * 0.5
    log_canonical = np.log10(canonical_plot + floor)
    log_differentiable = np.log10(differentiable_plot + floor)
    vmin = min(float(log_canonical.min()), float(log_differentiable.min()))
    vmax = max(float(log_canonical.max()), float(log_differentiable.max()))

    figure, axes = plt.subplots(2, 2, figsize=(14.5, 11), constrained_layout=True)
    for axis, values, title in (
        (axes[0, 0], log_canonical, "exact canonical sliced OT"),
        (axes[0, 1], log_differentiable, "512-quantile differentiable approximation"),
    ):
        image = axis.pcolormesh(
            x, y, values, shading="nearest", cmap="viridis", vmin=vmin, vmax=vmax,
        )
        plt.colorbar(image, ax=axis, label="log10 2D-OT")
        axis.set_title(title)

    difference = log_differentiable - log_canonical
    limit = float(np.quantile(np.abs(difference[np.isfinite(difference)]), 0.99))
    image = axes[1, 0].pcolormesh(
        x, y, difference, shading="nearest", cmap="coolwarm", vmin=-limit, vmax=limit,
    )
    plt.colorbar(image, ax=axes[1, 0], label="log10 approximate − log10 canonical")
    axes[1, 0].set_title("log-distance discrepancy")

    mask = ~np.eye(config.points, dtype=bool)
    axes[1, 1].hexbin(
        canonical[mask], differentiable[mask], gridsize=55, bins="log", cmap="magma",
        mincnt=1,
    )
    low = min(float(canonical[mask].min()), float(differentiable[mask].min()))
    high = max(float(canonical[mask].max()), float(differentiable[mask].max()))
    axes[1, 1].plot([low, high], [low, high], "k--", lw=1.2)
    axes[1, 1].set_xlabel("exact canonical distance")
    axes[1, 1].set_ylabel("differentiable approximate distance")
    axes[1, 1].set_title(
        "off-diagonal agreement\n"
        f"Spearman={metrics['spearman_canonical_vs_differentiable']:.3f}, "
        f"median relative error={metrics['median_relative_difference']:.1%}"
    )

    for axis in axes.flat[:3]:
        axis.plot(onsets, pitches, color="#f3bd2e", lw=2.1, label="target diagonal")
        axis.set_yscale("log", base=2)
        axis.set_xlabel("onset time (s)")
        axis.set_ylabel("predicted pitch (Hz, log scale)")
        axis.set_yticks([80, 100, 120, 160, 200, 240, 320])
        axis.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[0, 0].legend(loc="upper left")
    figure.suptitle(
        "Exact Fabiani--Schlecht--Elvander sliced OT versus our differentiable surrogate\n"
        f"same {config.points}×{config.points} static KS grid at "
        f"{config.sample_rate / 1000:g} kHz",
        fontsize=15,
    )
    figure.savefig(output / "canonical_vs_differentiable_surface.png", dpi=180)
    figure.savefig(output / "canonical_vs_differentiable_surface.pdf")
    plt.close(figure)


def aggregate(
    config: ExperimentConfig,
    output: Path,
    tasks: int,
    differentiable_path: Path,
) -> None:
    onsets, pitches, target_pitches = grids(config)
    canonical = np.empty((config.points, config.points), dtype=np.float64)
    seen = np.zeros(config.points, dtype=bool)
    for task_index in range(tasks):
        shard = np.load(output / f"shard_{task_index}.npz")
        rows = shard["rows"]
        canonical[rows] = shard["canonical_distance"]
        seen[rows] = True
    if not seen.all():
        raise RuntimeError("canonical shards do not cover every row")
    differentiable_data = np.load(differentiable_path)
    np.testing.assert_allclose(onsets, differentiable_data["onset_seconds"])
    np.testing.assert_allclose(pitches, differentiable_data["pitch_hz"])
    differentiable = differentiable_data["distance"]
    metrics = comparison_metrics(pitches, canonical, differentiable)

    np.savez(
        output / "canonical_surface_data.npz",
        onset_seconds=onsets,
        pitch_hz=pitches,
        target_pitch_hz=target_pitches,
        canonical_distance=canonical,
        differentiable_distance=differentiable,
    )
    with (output / "canonical_surface_data.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["onset_seconds", "target_pitch_hz", "predicted_pitch_hz",
             "canonical_distance", "differentiable_distance"]
        )
        for row, onset in enumerate(onsets):
            for column, pitch in enumerate(pitches):
                writer.writerow(
                    [onset, target_pitches[row], pitch, canonical[row, column],
                     differentiable[row, column]]
                )
    metadata = {
        "experiment": asdict(config),
        "ot": asdict(TFOTConfig(sample_rate=config.sample_rate)),
        "source_commit": source_commit(),
        "tasks": tasks,
        "comparison": metrics,
        "algorithm": "Exact discrete 1D W2 via two-pointer monotone coupling; averaged over ten fixed TF projections.",
    }
    (output / "configuration.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    plot_comparison(
        output, config, onsets, pitches, canonical, differentiable, metrics,
    )
    print(json.dumps(metrics, indent=2))
    print(output / "canonical_vs_differentiable_surface.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "aggregate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--differentiable-data", type=Path, default=DIFFERENTIABLE_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--render-batch-size", type=int, default=10)
    parser.add_argument("--cpu-threads", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = ExperimentConfig(sample_rate=8_000, duration_seconds=2.0, points=args.points)
    if args.mode == "run":
        run_shard(
            cfg, args.output, torch.device(args.device), args.task_index, args.tasks,
            args.render_batch_size, args.cpu_threads,
        )
    else:
        aggregate(cfg, args.output, args.tasks, args.differentiable_data)
