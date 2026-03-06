from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from torch import Tensor

PARAM_NAMES: tuple[str, ...] = (
    "f0",
    "burst_gain",
    "pluck_position",
    "dynamic_level",
    "a1",
    "decay",
)


MIDI_D1   = 26
MIDI_D6   = 86

F0_MIN_HZ = 36.71    # D1
F0_MAX_HZ = 1174.66   # D6
PLUCK_POSITION_MIN = 0.01
PLUCK_POSITION_MAX = 0.5
DYNAMIC_LEVEL_MIN = 0.1
DYNAMIC_LEVEL_MAX = 1.0
DECAY_MIN = 0.9
DECAY_MAX = 1.0
DAMPING_MIN = 0.0
DAMPING_LOG_MIN = 1e-4              # floor for a1 when using LOG_MAE (avoids log 0)
DAMPING_MAX = 0.75
BURST_GAIN_MIN = 0.0
BURST_GAIN_MAX = 1.0

def midi_to_hz(midi):
    """Convert MIDI note number → frequency in Hz.  Works with float, np, torch."""
    if isinstance(midi, Tensor):
        return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def hz_to_midi(hz):
    """Convert frequency in Hz → MIDI note number.  Works with float, np, torch."""
    if isinstance(hz, Tensor):
        return 69.0 + 12.0 * torch.log2(hz / 440.0)
    return 69.0 + 12.0 * np.log2(hz / 440.0)


class LossType(Enum):
    MAE       = "mae"
    LOG_MAE   = "log_mae"        # MAE in log(x) space — for params where ratio matters
    LOG1M_MAE = "log1m_mae"      # MAE in log(1−x) space — for params near 1 (e.g. decay)
    HUNGARIAN = "hungarian"


@dataclass(frozen=True)
class ParamSpec:
    """Metadata for a single synthesis parameter."""

    name: str
    low: float
    high: float
    loss_type: LossType = LossType.MAE
    description: str = ""

    def to_logit(self, x: Tensor, eps: float = 1e-7) -> Tensor:
        """Map raw value → normalised logit ∈ [0, 1] in log domain.

        logit = (ln x − ln low) / (ln high − ln low)
        When low == 0 the log-floor is clamped to *eps*.
        """
        safe_low = max(self.low, eps)
        ln_low  = math.log(safe_low)
        ln_high = math.log(self.high)
        return (torch.log(x.clamp(min=safe_low)) - ln_low) / (ln_high - ln_low)

    def from_logit(self, logit: Tensor, eps: float = 1e-7) -> Tensor:
        """Map logit ∈ [0, 1] → raw value via exp(ln_low + (ln_high − ln_low) · logit).

        When low == 0 the log-floor is clamped to *eps*.
        """
        safe_low = max(self.low, eps)
        ln_low  = math.log(safe_low)
        ln_high = math.log(self.high)
        return torch.exp(ln_low + (ln_high - ln_low) * logit)

    def normalise(self, x: Tensor) -> Tensor:
        """Map raw value → [0, 1] linearly."""
        return (x - self.low) / (self.high - self.low)

    def denormalise(self, x_norm: Tensor) -> Tensor:
        """Map [0, 1] → raw value linearly."""
        return self.low + x_norm * (self.high - self.low)

    # ── log(1−x) helpers (for params near 1, e.g. decay) ────────────────
    def to_log1m(self, x: Tensor, eps: float = 1e-7) -> Tensor:
        """Map x → log(1 − x).  Larger magnitude ↔ closer to 1."""
        return torch.log((1.0 - x).clamp(min=eps))

    def log1m_mae(self, pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
        """MAE in log(1−x) space."""
        return (self.to_log1m(pred, eps) - self.to_log1m(target, eps)).abs().mean()


def make_default_registry(fs: int = 16000) -> dict[str, ParamSpec]:
    """Build the canonical parameter registry.
    Call once and share the resulting dict across Dataset, Synth, and Loss.
    """
    return {
        "f0": ParamSpec(
            name="f0",
            low=F0_MIN_HZ,
            high=F0_MAX_HZ,
            loss_type=LossType.LOG_MAE,
            description="Fundamental frequency (Hz)",
        ),
        "burst_gain": ParamSpec(
            name="burst_gain",
            low=BURST_GAIN_MIN,
            high=BURST_GAIN_MAX,
            loss_type=LossType.HUNGARIAN,
            description="Excitation gain (sparse onsets)",
        ),
        "pluck_position": ParamSpec(
            name="pluck_position",
            low=PLUCK_POSITION_MIN,
            high=PLUCK_POSITION_MAX,
            loss_type=LossType.MAE,
            description="Pluck position as fraction of string length",
        ),
        "dynamic_level": ParamSpec(
            name="dynamic_level",
            low=DYNAMIC_LEVEL_MIN,
            high=DYNAMIC_LEVEL_MAX,
            loss_type=LossType.LOG_MAE,
            description="Dynamics filter bandwidth (Hz-equivalent)",
        ),
        "a1": ParamSpec(
            name="a1",
            low=DAMPING_MIN,
            high=DAMPING_MAX,
            loss_type=LossType.MAE,
            description="Loop-filter pole coefficient",
        ),
        "decay": ParamSpec(
            name="decay",
            low=DECAY_MIN,
            high=DECAY_MAX,
            loss_type=LossType.LOG1M_MAE,
            description="Loop gain / decay coefficient",
        ),
    }


def validate_param_dict(params: dict, context: str = "") -> None:
    """Check that a param dict contains exactly the canonical keys.

    Raises KeyError with a helpful message on mismatch.
    """
    expected = set(PARAM_NAMES)
    actual   = set(params.keys())
    missing  = expected - actual
    extra    = actual - expected
    if missing or extra:
        msg = f"Parameter dict mismatch"
        if context:
            msg += f" in {context}"
        if missing:
            msg += f" — missing: {missing}"
        if extra:
            msg += f" — unexpected: {extra}"
        raise KeyError(msg)