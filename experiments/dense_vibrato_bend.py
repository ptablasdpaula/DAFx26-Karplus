#!/usr/bin/env python3
"""Dense Karplus--Strong vibrato/bend inverse experiment.

Examples
--------
Short CPU closure check::

    python experiments/dense_vibrato_bend.py smoke

One production fit::

    python experiments/dense_vibrato_bend.py run \
        --arm hybrid --lambda-sparse 0.01 --seed 2027 --device cuda

One Slurm-array task (three seeds)::

    python experiments/dense_vibrato_bend.py sweep --task-index 0 --device cuda
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import random
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.losses import MultiScaleSpectralLoss
from src.audio_objectives import (
    effective_active_frames,
    rms,
    rms_match,
    sparse_gate_penalty,
)
from src.synths.dense import DENSE_CONTROL_NAMES, DenseKSConfig, DenseKSSynth
from src.synths.param_registry import (
    DAMPING_MAX,
    DECAY_MAX,
    DECAY_MIN,
    DYNAMIC_LEVEL_MAX,
    DYNAMIC_LEVEL_MIN,
    PLUCK_POSITION_MAX,
    PLUCK_POSITION_MIN,
)
from src.tf_ot import DifferentiableTFOT, TFOTConfig, canonical_tf_ot_audio

ARMS = ("mss", "ot", "hybrid")
LAMBDAS = (0.001, 0.01, 0.1)
SEEDS = (2027, 2028, 2029)
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "outputs" / "dense-vibrato-bend-ot"
BASE_FREQUENCY_HZ = 110.0
_TERMINATION_REQUESTED = False


def _termination_handler(_signum: int, _frame: Any) -> None:
    global _TERMINATION_REQUESTED
    _TERMINATION_REQUESTED = True


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
        + "\n"
    )
    os.replace(temporary, path)


def source_provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    packages = {}
    for name in ("torch", "torchlpc", "philtorch", "numpy", "librosa", "soundfile"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "source_commit": git("rev-parse", "HEAD"),
        "source_branch": git("branch", "--show-current"),
        # Local editor settings are intentionally untracked in this checkout;
        # only tracked changes make a result's source unreproducible.
        "source_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def require_cuda_extension() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for production fits")
    import torchlpc

    if getattr(torchlpc, "EXTENSION_LOADED", False) is not True:
        raise RuntimeError(
            "torchlpc.EXTENSION_LOADED is false; refusing the Numba/CPU fallback"
        )
    capability = torch.cuda.get_device_capability()
    if capability != (8, 0):
        raise RuntimeError(f"the production preflight requires an sm_80 A100, got sm_{capability[0]}{capability[1]}")


def minimum_jerk(progress: Tensor) -> Tensor:
    progress = progress.clamp(0.0, 1.0)
    return 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5


def target_controls(config: DenseKSConfig, device: torch.device) -> dict[str, Tensor]:
    """Create the matched-renderer four-second A2 gesture at frame rate."""
    duration = config.num_samples / config.fs
    frame_times = (
        torch.arange(config.num_frames, device=device, dtype=torch.float32)
        * duration
        / config.num_frames
    )
    gate = torch.zeros(config.num_frames, device=device)
    onset_index = int(torch.argmin(torch.abs(frame_times - 0.1)).item())
    gate[onset_index] = 1.0

    cents = torch.zeros_like(frame_times)
    vibrato_region = (frame_times >= 0.4) & (frame_times <= 2.0)
    vibrato_phase = 2.0 * torch.pi * 5.0 * (frame_times - 0.4)
    # Raised-cosine 200 ms attack/release makes the vibrato join smoothly.
    attack = 0.5 - 0.5 * torch.cos(
        torch.pi * ((frame_times - 0.4) / 0.2).clamp(0.0, 1.0)
    )
    release = 0.5 - 0.5 * torch.cos(
        torch.pi * ((2.0 - frame_times) / 0.2).clamp(0.0, 1.0)
    )
    envelope = torch.minimum(attack, release)
    cents = cents + vibrato_region * (30.0 * envelope * torch.sin(vibrato_phase))
    cents = cents + 200.0 * minimum_jerk((frame_times - 2.0) / 1.2)

    f0 = BASE_FREQUENCY_HZ * torch.pow(2.0, cents / 1200.0)
    delay = config.fs / f0
    constant = lambda value: torch.full_like(frame_times, value)
    return {
        "noise_gate": gate.unsqueeze(0),
        "delay": delay.unsqueeze(0),
        "decay": constant(0.99).unsqueeze(0),
        "a1": constant(0.2).unsqueeze(0),
        "pluck_position": constant(0.25).unsqueeze(0),
        "dynamics": constant(0.9).unsqueeze(0),
    }


def _inverse_sigmoid(value: Tensor) -> Tensor:
    value = value.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(value) - torch.log1p(-value)


def _inverse_softplus(value: Tensor) -> Tensor:
    return value + torch.log(-torch.expm1(-value))


class OptimizableDenseControls(nn.Module):
    """Unconstrained logits mapped to the experiment's physical bounds."""

    def __init__(self, config: DenseKSConfig, seed: int, device: torch.device):
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        frames = config.num_frames
        t = torch.linspace(0.0, 1.0, frames)
        center = 0.08 + 0.10 * torch.rand((), generator=generator)
        width = 0.12 + 0.08 * torch.rand((), generator=generator)
        gate = 0.03 + torch.exp(-0.5 * ((t - center) / width) ** 2)
        gate = gate * (0.9 + 0.2 * torch.rand(frames, generator=generator))
        self.gate_raw = nn.Parameter(_inverse_softplus(gate).unsqueeze(0).to(device))

        def constant_logit(value: float, low: float, high: float) -> nn.Parameter:
            normalized = torch.full((1, frames), (value - low) / (high - low))
            return nn.Parameter(_inverse_sigmoid(normalized).to(device))

        self.cents_raw = constant_logit(0.0, config.min_pitch_cents, config.max_pitch_cents)
        self.decay_raw = constant_logit(0.97, DECAY_MIN, DECAY_MAX)
        self.a1_raw = constant_logit(0.4, 0.0, DAMPING_MAX)
        self.pluck_raw = constant_logit(0.35, PLUCK_POSITION_MIN, PLUCK_POSITION_MAX)
        self.dynamics_raw = constant_logit(0.5, DYNAMIC_LEVEL_MIN, DYNAMIC_LEVEL_MAX)
        self.config = config

    @staticmethod
    def bounded(raw: Tensor, low: float, high: float) -> Tensor:
        return low + (high - low) * torch.sigmoid(raw)

    def forward(self) -> dict[str, Tensor]:
        cents = self.bounded(
            self.cents_raw,
            self.config.min_pitch_cents,
            self.config.max_pitch_cents,
        )
        f0 = BASE_FREQUENCY_HZ * torch.pow(2.0, cents / 1200.0)
        return {
            "noise_gate": F.softplus(self.gate_raw),
            "delay": self.config.fs / f0,
            "decay": self.bounded(self.decay_raw, DECAY_MIN, DECAY_MAX),
            "a1": self.bounded(self.a1_raw, 0.0, DAMPING_MAX),
            "pluck_position": self.bounded(
                self.pluck_raw, PLUCK_POSITION_MIN, PLUCK_POSITION_MAX,
            ),
            "dynamics": self.bounded(
                self.dynamics_raw, DYNAMIC_LEVEL_MIN, DYNAMIC_LEVEL_MAX,
            ),
        }


def _gradient_rms(loss: Tensor, audio: Tensor) -> float:
    gradient = torch.autograd.grad(loss, audio, retain_graph=False)[0]
    value = float(gradient.square().mean().sqrt().detach().cpu())
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"initial audio-gradient RMS is invalid: {value}")
    return value


def frozen_loss_normalizers(
    raw_prediction: Tensor,
    target: Tensor,
    mss: MultiScaleSpectralLoss,
    ot: DifferentiableTFOT,
) -> dict[str, float | None]:
    """Measure each objective's initial gradient RMS w.r.t. rendered audio."""
    mss_audio = raw_prediction.detach().requires_grad_(True)
    mss_value = mss(rms_match(mss_audio, target), target)
    mss_gradient_rms = _gradient_rms(mss_value, mss_audio)

    ot_audio = raw_prediction.detach().requires_grad_(True)
    ot_value = ot(rms_match(ot_audio, target), target)
    ot_gradient_rms = _gradient_rms(ot_value, ot_audio)
    return {"mss": mss_gradient_rms, "ot": ot_gradient_rms}


def cosine_learning_rate(step: int, steps: int, high: float = 0.03, low: float = 0.003) -> float:
    if steps <= 1:
        return low
    progress = step / (steps - 1)
    return low + 0.5 * (high - low) * (1.0 + math.cos(math.pi * progress))


def run_name(arm: str, lambda_sparse: float, seed: int) -> str:
    return f"arm={arm}__lambda={lambda_sparse:g}__seed={seed}"


def save_checkpoint(
    path: Path,
    *,
    controls: OptimizableDenseControls,
    optimizer: torch.optim.Optimizer,
    next_step: int,
    steps: int,
    history: list[dict[str, float]],
    normalizers: dict[str, float],
    configuration: dict[str, Any],
) -> None:
    state = {
        "controls": controls.state_dict(),
        "optimizer": optimizer.state_dict(),
        "next_step": next_step,
        "steps": steps,
        "history": history,
        "normalizers": normalizers,
        "configuration": configuration,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    controls: OptimizableDenseControls,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    state = torch.load(path, map_location=device, weights_only=False)
    controls.load_state_dict(state["controls"])
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"].cpu())
    if torch.cuda.is_available() and state.get("cuda_random_state") is not None:
        torch.cuda.set_rng_state_all(state["cuda_random_state"])
    return state


def save_target_artifacts(
    output_root: Path,
    config: DenseKSConfig,
    controls: dict[str, Tensor],
    audio: Tensor,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    target_wav = output_root / "target.wav"
    target_npz = output_root / "target_controls.npz"
    if not target_wav.exists():
        temporary_wav = output_root / f".target.{os.getpid()}.wav"
        sf.write(temporary_wav, audio[0].detach().cpu().numpy(), config.fs)
        os.replace(temporary_wav, target_wav)
    if not target_npz.exists():
        temporary_npz = output_root / f".target_controls.{os.getpid()}.npz"
        np.savez(
            temporary_npz,
            **{name: value[0].detach().cpu().numpy() for name, value in controls.items()},
        )
        os.replace(temporary_npz, target_npz)
    write_json(
        output_root / "target.json",
        {
            "renderer": asdict(config),
            "gesture": {
                "base_frequency_hz": BASE_FREQUENCY_HZ,
                "onset_seconds": 0.1,
                "vibrato": {"start": 0.4, "end": 2.0, "rate_hz": 5.0, "depth_cents": 30.0},
                "bend": {"start": 2.0, "end": 3.2, "end_cents": 200.0, "shape": "minimum_jerk"},
            },
            "provenance": source_provenance(),
        },
    )


def control_metrics(
    predicted: dict[str, Tensor], target: dict[str, Tensor], config: DenseKSConfig,
) -> dict[str, float]:
    pred_delay = predicted["delay"][0]
    target_delay = target["delay"][0]
    cent_error = torch.abs(1200.0 * torch.log2(pred_delay / target_delay))
    duration = config.num_samples / config.fs
    frame_times = torch.arange(config.num_frames, device=pred_delay.device) * duration / config.num_frames
    vibrato = (frame_times >= 0.4) & (frame_times <= 2.0)
    bend = (frame_times >= 2.0) & (frame_times <= 3.2)
    gate = predicted["noise_gate"][0]
    gate_weights = gate / gate.sum().clamp_min(1e-12)
    pred_onset = torch.argmax(gate).float() * duration / config.num_frames
    target_onset = torch.argmax(target["noise_gate"][0]).float() * duration / config.num_frames

    def masked_mean(values: Tensor, mask: Tensor) -> float | None:
        return float(values[mask].mean().detach().cpu()) if bool(mask.any()) else None

    return {
        "delay_cent_error_mean": float(cent_error.mean().detach().cpu()),
        "delay_cent_error_median": float(cent_error.median().detach().cpu()),
        "delay_cent_error_vibrato_mean": masked_mean(cent_error, vibrato),
        "delay_cent_error_bend_mean": masked_mean(cent_error, bend),
        "gate_onset_error_seconds": float(torch.abs(pred_onset - target_onset).detach().cpu()),
        "effective_active_frames": float(effective_active_frames(gate.unsqueeze(0))[0].detach().cpu()),
        "effective_active_fraction": float(sparse_gate_penalty(gate.unsqueeze(0)).detach().cpu()),
        "decay_mae": float(torch.mean(torch.abs(predicted["decay"] - target["decay"])).detach().cpu()),
        "a1_mae": float(torch.mean(torch.abs(predicted["a1"] - target["a1"])).detach().cpu()),
        "gate_weighted_pluck_mae": float(
            torch.sum(gate_weights * torch.abs(predicted["pluck_position"][0] - target["pluck_position"][0])).detach().cpu()
        ),
        "gate_weighted_dynamics_mae": float(
            torch.sum(gate_weights * torch.abs(predicted["dynamics"][0] - target["dynamics"][0])).detach().cpu()
        ),
    }


def waveform_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    rmse_value = float(np.sqrt(np.mean(np.square(error))))
    correlation = float(np.corrcoef(prediction, target)[0, 1])
    target_energy = float(np.dot(target, target))
    projection = float(np.dot(prediction, target)) / max(target_energy, 1e-20) * target
    residual = prediction - projection
    si_sdr = 10.0 * math.log10(
        max(float(np.dot(projection, projection)), 1e-20)
        / max(float(np.dot(residual, residual)), 1e-20)
    )
    return {
        "waveform_rmse": rmse_value,
        "waveform_correlation": correlation,
        "waveform_si_sdr_db": si_sdr,
    }


@torch.no_grad()
def evaluate_result(
    raw_audio: Tensor,
    matched_audio: Tensor,
    target_audio: Tensor,
    predicted_controls: dict[str, Tensor],
    true_controls: dict[str, Tensor],
    config: DenseKSConfig,
    mss: MultiScaleSpectralLoss,
) -> dict[str, float]:
    matched_np = matched_audio[0].detach().cpu().numpy()
    target_np = target_audio[0].detach().cpu().numpy()
    metrics = control_metrics(predicted_controls, true_controls, config)
    metrics.update(waveform_metrics(matched_np, target_np))
    metrics.update(
        {
            "raw_mss": float(mss(matched_audio, target_audio).detach().cpu()),
            "canonical_2d_ot": canonical_tf_ot_audio(
                matched_np,
                target_np,
                config=TFOTConfig(
                    sample_rate=config.fs, n_fft=1024, hop_length=256,
                ),
            ),
            "raw_audio_rms": float(rms(raw_audio)[0, 0].detach().cpu()),
            "matched_audio_rms": float(rms(matched_audio)[0, 0].detach().cpu()),
            "target_audio_rms": float(rms(target_audio)[0, 0].detach().cpu()),
        }
    )
    return metrics


def save_figures(
    run_dir: Path,
    raw_audio: Tensor,
    matched_audio: Tensor,
    target_audio: Tensor,
    predicted: dict[str, Tensor],
    target: dict[str, Tensor],
    config: DenseKSConfig,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display

    target_np = target_audio[0].detach().cpu().numpy()
    matched_np = matched_audio[0].detach().cpu().numpy()
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for axis, audio, title in zip(axes, (target_np, matched_np), ("Target", "RMS-matched prediction")):
        spectrum = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio, n_fft=1024, hop_length=256, window="hann")),
            ref=np.max,
        )
        librosa.display.specshow(
            spectrum, sr=config.fs, hop_length=256, x_axis="time", y_axis="log", ax=axis,
        )
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(run_dir / "spectrogram.png", dpi=150)
    plt.close(figure)

    frame_time = np.arange(config.num_frames) * config.num_samples / config.fs / config.num_frames
    figure, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    labels = {
        "noise_gate": "relative noise gate",
        "delay": "delay (samples)",
        "decay": "decay g",
        "a1": "loop damping a1",
        "pluck_position": "pluck position",
        "dynamics": "dynamics",
    }
    for axis, name in zip(axes, DENSE_CONTROL_NAMES):
        axis.plot(frame_time, target[name][0].detach().cpu(), label="target", linewidth=2)
        axis.plot(frame_time, predicted[name][0].detach().cpu(), label="prediction", alpha=0.85)
        axis.set_ylabel(labels[name])
        axis.grid(alpha=0.2)
    axes[0].legend()
    axes[-1].set_xlabel("time (s)")
    figure.tight_layout()
    figure.savefig(run_dir / "controls.png", dpi=150)
    plt.close(figure)


def fit(
    *,
    arm: str,
    lambda_sparse: float,
    seed: int,
    output_root: Path,
    device: torch.device,
    steps: int = 1000,
    checkpoint_every: int = 25,
    renderer_config: DenseKSConfig | None = None,
    resume: bool = False,
    save_plots: bool = True,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if lambda_sparse not in LAMBDAS and steps == 1000:
        raise ValueError(f"production lambda must be one of {LAMBDAS}")
    if device.type == "cuda":
        require_cuda_extension()
    if steps <= 0:
        raise ValueError("steps must be positive")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = renderer_config or DenseKSConfig()
    run_dir = output_root / run_name(arm, lambda_sparse, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    provenance = source_provenance()
    configuration = {
        "arm": arm,
        "lambda_sparse": lambda_sparse,
        "seed": seed,
        "steps": steps,
        "learning_rate_start": 0.03,
        "learning_rate_end": 0.003,
        "gradient_clip_norm": 10.0,
        "renderer": asdict(config),
        "ot": asdict(TFOTConfig(sample_rate=config.fs)),
        "provenance": provenance,
    }
    write_json(run_dir / "configuration.json", configuration)

    synth = DenseKSSynth(config).to(device)
    truth = target_controls(config, device)
    with torch.no_grad():
        target_audio = synth(truth)
    save_target_artifacts(output_root, config, truth, target_audio)

    controls = OptimizableDenseControls(config, seed, device).to(device)
    optimizer = torch.optim.Adam(controls.parameters(), lr=0.03)
    mss = MultiScaleSpectralLoss().to(device)
    ot = DifferentiableTFOT(TFOTConfig(sample_rate=config.fs)).to(device)

    start_step = 0
    history: list[dict[str, float]] = []
    normalizers: dict[str, float]
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"cannot resume: {checkpoint_path} does not exist")
        state = load_checkpoint(checkpoint_path, controls, optimizer, device)
        if state["steps"] != steps:
            raise ValueError(
                f"checkpoint planned {state['steps']} steps, requested {steps}"
            )
        saved_config = state["configuration"]
        for key in ("arm", "lambda_sparse", "seed"):
            if saved_config[key] != configuration[key]:
                raise ValueError(f"checkpoint {key} mismatch")
        start_step = int(state["next_step"])
        history = list(state["history"])
        normalizers = dict(state["normalizers"])
    else:
        with torch.enable_grad():
            initial_raw = synth(controls())
            normalizers = frozen_loss_normalizers(initial_raw, target_audio, mss, ot)
        write_json(run_dir / "loss_normalizers.json", normalizers)

    started = time.perf_counter()
    old_term_handler = signal.signal(signal.SIGTERM, _termination_handler)
    old_int_handler = signal.signal(signal.SIGINT, _termination_handler)
    completed = False
    try:
        for step in range(start_step, steps):
            learning_rate = cosine_learning_rate(step, steps)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            physical = controls()
            raw_audio = synth(physical)
            matched_audio = rms_match(raw_audio, target_audio)
            sparse_value = sparse_gate_penalty(physical["noise_gate"])
            if arm == "mss":
                mss_value = mss(matched_audio, target_audio)
                ot_value = torch.full_like(mss_value, torch.nan)
                audio_objective = mss_value / normalizers["mss"]
            elif arm == "ot":
                ot_value = ot(matched_audio, target_audio)
                mss_value = torch.full_like(ot_value, torch.nan)
                audio_objective = ot_value / normalizers["ot"]
            else:
                mss_value = mss(matched_audio, target_audio)
                ot_value = ot(matched_audio, target_audio)
                audio_objective = (
                    ot_value / normalizers["ot"]
                    + 0.01 * mss_value / normalizers["mss"]
                )
            total = audio_objective + lambda_sparse * sparse_value
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite objective at step {step}")
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(controls.parameters(), 10.0)
            optimizer.step()
            history.append(
                {
                    "step": float(step),
                    "total": float(total.detach().cpu()),
                    "audio_objective": float(audio_objective.detach().cpu()),
                    "mss": (
                        float(mss_value.detach().cpu())
                        if torch.isfinite(mss_value)
                        else None
                    ),
                    "ot": (
                        float(ot_value.detach().cpu())
                        if torch.isfinite(ot_value)
                        else None
                    ),
                    "sparse_gate": float(sparse_value.detach().cpu()),
                    "effective_frames": float(effective_active_frames(physical["noise_gate"])[0].detach().cpu()),
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    "learning_rate": learning_rate,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            next_step = step + 1
            if next_step % checkpoint_every == 0 or next_step == steps or _TERMINATION_REQUESTED:
                save_checkpoint(
                    checkpoint_path,
                    controls=controls,
                    optimizer=optimizer,
                    next_step=next_step,
                    steps=steps,
                    history=history,
                    normalizers=normalizers,
                    configuration=configuration,
                )
            if _TERMINATION_REQUESTED:
                write_json(
                    run_dir / "status.json",
                    {"status": "interrupted", "next_step": next_step, "steps": steps},
                )
                return {"status": "interrupted", "run_dir": str(run_dir), "next_step": next_step}
        completed = True
    finally:
        signal.signal(signal.SIGTERM, old_term_handler)
        signal.signal(signal.SIGINT, old_int_handler)

    if not completed:
        raise RuntimeError("fit stopped without a checkpoint")

    with torch.no_grad():
        final_controls = controls()
        final_raw = synth(final_controls)
        final_matched = rms_match(final_raw, target_audio)
        metrics = evaluate_result(
            final_raw, final_matched, target_audio, final_controls, truth, config, mss,
        )
    runtime = time.perf_counter() - started
    metrics.update(
        {
            "arm": arm,
            "lambda_sparse": lambda_sparse,
            "seed": seed,
            "steps": steps,
            "runtime_seconds": runtime,
            "source_commit": provenance["source_commit"],
            "source_dirty": provenance["source_dirty"],
            "provenance": provenance,
            "status": "complete",
        }
    )

    sf.write(run_dir / "audio_raw.wav", final_raw[0].detach().cpu().numpy(), config.fs)
    sf.write(run_dir / "audio_rms_matched.wav", final_matched[0].detach().cpu().numpy(), config.fs)
    np.savez(
        run_dir / "controls.npz",
        **{name: value[0].detach().cpu().numpy() for name, value in final_controls.items()},
    )
    np.savez(
        run_dir / "history.npz",
        **{
            key: np.asarray([entry[key] for entry in history], dtype=np.float64)
            for key in history[0]
        },
    )
    write_json(run_dir / "history.json", {"history": history})
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "status.json", {"status": "complete", "steps": steps})
    if save_plots:
        save_figures(
            run_dir, final_raw, final_matched, target_audio, final_controls, truth, config,
        )
    return metrics


def sweep_config(task_index: int) -> tuple[str, float]:
    configurations = [(arm, value) for arm in ARMS for value in LAMBDAS]
    if not 0 <= task_index < len(configurations):
        raise ValueError(f"task index must be in [0, {len(configurations) - 1}]")
    return configurations[task_index]


def run_sweep(args: argparse.Namespace) -> None:
    tasks: Iterable[int] = range(9) if args.task_index is None else (args.task_index,)
    for task_index in tasks:
        arm, lambda_sparse = sweep_config(task_index)
        for seed in SEEDS:
            directory = args.output / run_name(arm, lambda_sparse, seed)
            complete = directory / "metrics.json"
            if complete.exists() and not args.overwrite:
                print(f"skip complete {directory.name}", flush=True)
                continue
            resume = (directory / "checkpoint.pt").exists()
            result = fit(
                arm=arm,
                lambda_sparse=lambda_sparse,
                seed=seed,
                output_root=args.output,
                device=torch.device(args.device),
                steps=args.steps,
                checkpoint_every=args.checkpoint_every,
                resume=resume,
                save_plots=not args.no_plots,
            )
            print(json.dumps(result, default=_json_default), flush=True)
            if result.get("status") == "interrupted":
                return


def aggregate(output_root: Path) -> dict[str, Any]:
    metric_paths = sorted(output_root.glob("arm=*__lambda=*__seed=*/metrics.json"))
    if not metric_paths:
        raise FileNotFoundError(f"no completed metrics under {output_root}")
    rows = [json.loads(path.read_text()) for path in metric_paths]
    columns = sorted({key for row in rows for key in row})
    with (output_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    groups: list[dict[str, Any]] = []
    numeric_keys = [
        "delay_cent_error_median",
        "delay_cent_error_mean",
        "delay_cent_error_vibrato_mean",
        "delay_cent_error_bend_mean",
        "effective_active_frames",
        "raw_mss",
        "canonical_2d_ot",
        "waveform_rmse",
        "runtime_seconds",
    ]
    for arm in ARMS:
        for lambda_sparse in LAMBDAS:
            selected = [
                row for row in rows
                if row["arm"] == arm and float(row["lambda_sparse"]) == lambda_sparse
            ]
            if not selected:
                continue
            group: dict[str, Any] = {
                "arm": arm,
                "lambda_sparse": lambda_sparse,
                "runs": len(selected),
            }
            for key in numeric_keys:
                values = [float(row[key]) for row in selected if key in row]
                group[f"median_{key}"] = float(np.nanmedian(values))
                group[f"iqr_{key}"] = float(
                    np.nanpercentile(values, 75) - np.nanpercentile(values, 25)
                )
            groups.append(group)
    groups.sort(
        key=lambda row: (
            row["median_delay_cent_error_median"],
            row["median_effective_active_frames"],
            row["median_raw_mss"],
        )
    )
    for rank, group in enumerate(groups, start=1):
        group["rank"] = rank
    group_columns = sorted({key for group in groups for key in group})
    with (output_root / "group_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=group_columns)
        writer.writeheader()
        writer.writerows(groups)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"mss": "tab:blue", "ot": "tab:orange", "hybrid": "tab:green"}
    for metric, label, filename in (
        ("raw_mss", "raw MSS", "pareto_mss.png"),
        ("canonical_2d_ot", "canonical 2D-OT", "pareto_ot.png"),
    ):
        figure, axis = plt.subplots(figsize=(7, 5))
        for row in rows:
            axis.scatter(
                row["effective_active_frames"], row[metric],
                color=colors[row["arm"]],
                marker={0.001: "o", 0.01: "s", 0.1: "^"}[float(row["lambda_sparse"])],
                alpha=0.8,
            )
        for arm, color in colors.items():
            axis.scatter([], [], color=color, label=arm)
        axis.set_xlabel("effective active gate frames (lower is sparser)")
        axis.set_ylabel(label + " (lower is better)")
        axis.grid(alpha=0.2)
        axis.legend(title="loss family")
        figure.tight_layout()
        figure.savefig(output_root / filename, dpi=160)
        plt.close(figure)

    summary = {
        "completed_runs": len(rows),
        "expected_runs": 27,
        "best": groups[0],
        "source_commits": sorted({str(row.get("source_commit")) for row in rows}),
    }
    write_json(output_root / "aggregate.json", summary)
    return summary


def preflight(output: Path) -> None:
    require_cuda_extension()
    device = torch.device("cuda")
    config = DenseKSConfig(num_samples=4096, num_frames=16)
    synth = DenseKSSynth(config).to(device)
    controls = target_controls(config, device)
    controls["delay"] = controls["delay"].detach().requires_grad_(True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    audio = synth(controls)
    loss = audio.square().mean()
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    gradient = controls["delay"].grad
    if not torch.isfinite(audio).all() or gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError("CUDA renderer forward/backward produced a non-finite value")
    report = {
        "status": "passed",
        "elapsed_seconds": elapsed,
        "audio_shape": list(audio.shape),
        "gradient_rms": float(gradient.square().mean().sqrt().cpu()),
        "provenance": source_provenance(),
    }
    write_json(output / "cuda_preflight.json", report)
    print(json.dumps(report, indent=2), flush=True)


def add_fit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--no-plots", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_parser = subparsers.add_parser("smoke", help="short CPU forward/backward fit")
    smoke_parser.add_argument("--steps", type=int, default=1)
    smoke_parser.add_argument("--samples", type=int, default=4096)
    smoke_parser.add_argument("--frames", type=int, default=16)

    run_parser = subparsers.add_parser("run", help="run one arm/lambda/seed fit")
    run_parser.add_argument("--arm", choices=ARMS, required=True)
    run_parser.add_argument("--lambda-sparse", type=float, choices=LAMBDAS, required=True)
    run_parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    run_parser.add_argument("--device", default="cuda")
    add_fit_arguments(run_parser)

    sweep_parser = subparsers.add_parser("sweep", help="run all fits or one 3-seed array task")
    sweep_parser.add_argument("--task-index", type=int)
    sweep_parser.add_argument("--device", default="cuda")
    sweep_parser.add_argument("--overwrite", action="store_true")
    add_fit_arguments(sweep_parser)

    resume_parser = subparsers.add_parser("resume", help="exactly resume one interrupted run")
    resume_parser.add_argument("--run-dir", type=Path, required=True)
    resume_parser.add_argument("--device", default="cuda")
    resume_parser.add_argument("--no-plots", action="store_true")

    subparsers.add_parser("aggregate", help="aggregate all completed run metrics")
    subparsers.add_parser("preflight", help="require sm_80 CUDA and torchlpc extension")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output = args.output.resolve()
    if args.command == "smoke":
        result = fit(
            arm="hybrid",
            lambda_sparse=0.01,
            seed=2027,
            output_root=args.output / "smoke",
            device=torch.device("cpu"),
            steps=args.steps,
            checkpoint_every=1,
            renderer_config=DenseKSConfig(
                num_samples=args.samples,
                num_frames=args.frames,
            ),
            save_plots=False,
        )
        print(json.dumps(result, indent=2, default=_json_default))
    elif args.command == "run":
        result = fit(
            arm=args.arm,
            lambda_sparse=args.lambda_sparse,
            seed=args.seed,
            output_root=args.output,
            device=torch.device(args.device),
            steps=args.steps,
            checkpoint_every=args.checkpoint_every,
            resume=False,
            save_plots=not args.no_plots,
        )
        print(json.dumps(result, indent=2, default=_json_default))
    elif args.command == "sweep":
        run_sweep(args)
    elif args.command == "resume":
        checkpoint = torch.load(args.run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
        configuration = checkpoint["configuration"]
        output_root = args.run_dir.parent.resolve()
        result = fit(
            arm=configuration["arm"],
            lambda_sparse=float(configuration["lambda_sparse"]),
            seed=int(configuration["seed"]),
            output_root=output_root,
            device=torch.device(args.device),
            steps=int(checkpoint["steps"]),
            checkpoint_every=25,
            renderer_config=DenseKSConfig(**configuration["renderer"]),
            resume=True,
            save_plots=not args.no_plots,
        )
        print(json.dumps(result, indent=2, default=_json_default))
    elif args.command == "aggregate":
        print(json.dumps(aggregate(args.output), indent=2, default=_json_default))
    elif args.command == "preflight":
        preflight(args.output)


if __name__ == "__main__":
    main()
