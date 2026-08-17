"""Dense-control time-domain Karplus--Strong renderer.

This module is deliberately separate from :class:`src.synths.synth.Synth`.
The event renderer remains the model used by the paper; this renderer is a
small inverse-problem instrument whose six controls all live on one fixed
frame grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.synths.constants import DEFAULT_FS, DEFAULT_LAGRANGE_ORDER
from src.synths.ddsp import (
    dynamics_filter,
    karplus_strong_from_delay,
    pluck_position_filter,
)
from src.synths.param_registry import (
    DAMPING_MAX,
    DECAY_MAX,
    DECAY_MIN,
    DYNAMIC_LEVEL_MAX,
    DYNAMIC_LEVEL_MIN,
    PLUCK_POSITION_MAX,
    PLUCK_POSITION_MIN,
)

DENSE_CONTROL_NAMES: tuple[str, ...] = (
    "noise_gate",
    "delay",
    "decay",
    "a1",
    "pluck_position",
    "dynamics",
)


@dataclass(frozen=True)
class DenseKSConfig:
    num_samples: int = 64_000
    num_frames: int = 250
    fs: int = DEFAULT_FS
    noise_seed: int = 42
    lagrange_order: int = DEFAULT_LAGRANGE_ORDER
    base_frequency_hz: float = 110.0
    min_pitch_cents: float = -100.0
    max_pitch_cents: float = 300.0
    silence_epsilon: float = 1e-12

    @property
    def min_delay(self) -> float:
        highest_f0 = self.base_frequency_hz * 2.0 ** (
            self.max_pitch_cents / 1200.0
        )
        return self.fs / highest_f0

    @property
    def max_delay(self) -> float:
        lowest_f0 = self.base_frequency_hz * 2.0 ** (
            self.min_pitch_cents / 1200.0
        )
        return self.fs / lowest_f0


class DenseKSSynth(nn.Module):
    """Render six ``[batch, frames]`` controls into mono audio.

    The excitation is a deterministic white-noise carrier multiplied by a
    zero-order-held relative gate.  The gate is normalized independently in
    every batch item, making global excitation gain intentionally absent from
    the parameterization.  All remaining controls are linearly interpolated
    to audio rate.
    """

    def __init__(self, config: DenseKSConfig | None = None):
        super().__init__()
        self.config = config or DenseKSConfig()
        if self.config.num_samples <= 0 or self.config.num_frames <= 0:
            raise ValueError("num_samples and num_frames must be positive")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.noise_seed)
        carrier = torch.randn(
            self.config.num_samples, generator=generator, dtype=torch.float32,
        )
        carrier = carrier - carrier.mean()
        carrier = carrier / carrier.square().mean().sqrt().clamp_min(1e-12)
        self.register_buffer("noise_carrier", carrier)

    def _validate(self, controls: dict[str, Tensor]) -> None:
        missing = set(DENSE_CONTROL_NAMES) - set(controls)
        extra = set(controls) - set(DENSE_CONTROL_NAMES)
        if missing or extra:
            raise KeyError(
                f"dense controls mismatch; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        shapes = {name: tuple(controls[name].shape) for name in DENSE_CONTROL_NAMES}
        expected_tail = self.config.num_frames
        if any(len(shape) != 2 or shape[1] != expected_tail for shape in shapes.values()):
            raise ValueError(
                f"all dense controls must be [batch, {expected_tail}], got {shapes}"
            )
        if len({shape[0] for shape in shapes.values()}) != 1:
            raise ValueError(f"dense control batch sizes differ: {shapes}")
        reference = controls[DENSE_CONTROL_NAMES[0]]
        if any(
            controls[name].device != reference.device
            or controls[name].dtype != reference.dtype
            for name in DENSE_CONTROL_NAMES
        ):
            raise ValueError("all dense controls must share device and dtype")
        if not reference.is_floating_point():
            raise TypeError("dense controls must be floating-point tensors")
        if any(not torch.isfinite(controls[name]).all() for name in DENSE_CONTROL_NAMES):
            raise ValueError("dense controls must be finite")

        bounds = {
            "noise_gate": (0.0, float("inf")),
            "delay": (self.config.min_delay, self.config.max_delay),
            "decay": (DECAY_MIN, DECAY_MAX),
            "a1": (0.0, DAMPING_MAX),
            "pluck_position": (PLUCK_POSITION_MIN, PLUCK_POSITION_MAX),
            "dynamics": (DYNAMIC_LEVEL_MIN, DYNAMIC_LEVEL_MAX),
        }
        for name, (low, high) in bounds.items():
            value = controls[name]
            if torch.any(value < low) or torch.any(value > high):
                raise ValueError(f"{name} must lie in [{low}, {high}]")

    def _linear(self, value: Tensor) -> Tensor:
        return F.interpolate(
            value.unsqueeze(1),
            size=self.config.num_samples,
            mode="linear",
            align_corners=False,
        ).squeeze(1)

    def _zoh(self, value: Tensor) -> Tensor:
        return F.interpolate(
            value.unsqueeze(1),
            size=self.config.num_samples,
            mode="nearest-exact",
        ).squeeze(1)

    def forward(self, controls: dict[str, Tensor]) -> Tensor:
        self._validate(controls)
        gate = controls["noise_gate"]
        gate_max = gate.amax(dim=-1, keepdim=True)
        if torch.any(gate_max <= self.config.silence_epsilon):
            raise ValueError("noise_gate is silent; at least one frame must be positive")
        relative_gate = gate / gate_max

        dense_gate = self._zoh(relative_gate)
        delay = self._linear(controls["delay"])
        decay = self._linear(controls["decay"])
        a1 = self._linear(controls["a1"])
        pluck_position = self._linear(controls["pluck_position"])
        dynamics = self._linear(controls["dynamics"])

        carrier = self.noise_carrier.to(device=gate.device, dtype=gate.dtype)
        excitation = dense_gate * carrier.unsqueeze(0)
        f0 = self.config.fs / delay
        excitation = dynamics_filter(
            excitation, f0=f0, dynamic_level=dynamics, fs=self.config.fs,
        )
        excitation = pluck_position_filter(
            excitation,
            f0=f0,
            position=pluck_position,
            fs=self.config.fs,
        )
        return karplus_strong_from_delay(
            excitation,
            delay=delay,
            a1=a1,
            g=decay,
            fs=self.config.fs,
            lagrange_order=self.config.lagrange_order,
        )
