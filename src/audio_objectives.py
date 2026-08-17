"""Small scale-free audio and excitation objective utilities."""
from __future__ import annotations

from torch import Tensor


def rms(audio: Tensor) -> Tensor:
    return audio.square().mean(dim=-1, keepdim=True).sqrt()


def rms_match(prediction: Tensor, target: Tensor, epsilon: float = 1e-12) -> Tensor:
    """Match prediction RMS to target RMS without exposing a gain control."""
    scale = rms(target) / rms(prediction).clamp_min(epsilon)
    return prediction * scale


def effective_active_frames(gate: Tensor, epsilon: float = 1e-12) -> Tensor:
    """Scale-invariant effective support, ``L1(gate) / Linf(gate)``."""
    return gate.abs().sum(dim=-1) / gate.abs().amax(dim=-1).clamp_min(epsilon)


def sparse_gate_penalty(gate: Tensor) -> Tensor:
    """Frame-normalized effective active count."""
    return (effective_active_frames(gate) / gate.shape[-1]).mean()
