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

EVENT_PARAM_NAMES: tuple[str, ...] = (
    "exists",           # 1.0 = real event, 0.0 = padding (no-object)
    "time",             # normalised onset time ∈ [0, 1]  (maps to 0–duration_s)
    "f0",               # fundamental frequency in Hz
    "burst_gain",       # pluck intensity ∈ [0, 1]
    "decay",            # loop gain ∈ [DECAY_MIN, DECAY_MAX]
    "a1",               # loop-filter pole ∈ [DAMPING_MIN, DAMPING_MAX]
    "pluck_position",   # string fraction ∈ [PLUCK_POSITION_MIN, PLUCK_POSITION_MAX]
    "dynamic_level",    # dynamics filter ∈ [DYNAMIC_LEVEL_MIN, DYNAMIC_LEVEL_MAX]
)

EVENT_SYNTH_PARAMS: tuple[str, ...] = (
    "f0", "burst_gain", "decay", "a1", "pluck_position", "dynamic_level",
)

MAX_EVENTS: int = 40

MIDI_D1   = 26
MIDI_D6   = 86

F0_MIN_HZ = 36.71      # D1
F0_MAX_HZ = 1174.66    # D6
PLUCK_POSITION_MIN = 0.01
PLUCK_POSITION_MAX = 0.5
DYNAMIC_LEVEL_MIN = 0.1
DYNAMIC_LEVEL_MAX = 1.0
DECAY_MIN = 0.9
DECAY_MAX = 1.0
DAMPING_MIN = 0.0
DAMPING_LOG_MIN = 1e-4
DAMPING_MAX = 0.75
BURST_GAIN_MIN = 0.0
BURST_GAIN_MAX = 1.0

def midi_to_hz(midi):
    if isinstance(midi, Tensor):
        return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def hz_to_midi(hz):
    if isinstance(hz, Tensor):
        return 69.0 + 12.0 * torch.log2(hz / 440.0)
    return 69.0 + 12.0 * np.log2(hz / 440.0)


class LossType(Enum):
    MAE       = "mae"
    LOG_MAE   = "log_mae"
    LOG1M_MAE = "log1m_mae"
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
        safe_low = max(self.low, eps)
        ln_low  = math.log(safe_low)
        ln_high = math.log(self.high)
        return (torch.log(x.clamp(min=safe_low)) - ln_low) / (ln_high - ln_low)

    def from_logit(self, logit: Tensor, eps: float = 1e-7) -> Tensor:
        safe_low = max(self.low, eps)
        ln_low  = math.log(safe_low)
        ln_high = math.log(self.high)
        return torch.exp(ln_low + (ln_high - ln_low) * logit)

    def normalise(self, x: Tensor) -> Tensor:
        return (x - self.low) / (self.high - self.low)

    def denormalise(self, x_norm: Tensor) -> Tensor:
        return self.low + x_norm * (self.high - self.low)

    def to_log1m(self, x: Tensor, eps: float = 1e-7) -> Tensor:
        return torch.log((1.0 - x).clamp(min=eps))


def make_default_registry(fs: int = 16000) -> dict[str, ParamSpec]:
    return {
        "f0": ParamSpec(
            name="f0", low=F0_MIN_HZ, high=F0_MAX_HZ,
            loss_type=LossType.LOG_MAE,
            description="Fundamental frequency (Hz)",
        ),
        "burst_gain": ParamSpec(
            name="burst_gain", low=BURST_GAIN_MIN, high=BURST_GAIN_MAX,
            loss_type=LossType.MAE,
            description="Excitation gain",
        ),
        "pluck_position": ParamSpec(
            name="pluck_position", low=PLUCK_POSITION_MIN, high=PLUCK_POSITION_MAX,
            loss_type=LossType.MAE,
            description="Pluck position as fraction of string length",
        ),
        "dynamic_level": ParamSpec(
            name="dynamic_level", low=DYNAMIC_LEVEL_MIN, high=DYNAMIC_LEVEL_MAX,
            loss_type=LossType.LOG_MAE,
            description="Dynamics filter bandwidth",
        ),
        "a1": ParamSpec(
            name="a1", low=DAMPING_MIN, high=DAMPING_MAX,
            loss_type=LossType.MAE,
            description="Loop-filter pole coefficient",
        ),
        "decay": ParamSpec(
            name="decay", low=DECAY_MIN, high=DECAY_MAX,
            loss_type=LossType.LOG1M_MAE,
            description="Loop gain / decay coefficient",
        ),
    }

def validate_param_dict(params: dict, context: str = "") -> None:
    """Check that a param dict contains exactly the frame-level keys."""
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


def validate_event_dict(events: dict, context: str = "") -> None:
    """Check that an event dict contains exactly the event-level keys."""
    expected = set(EVENT_PARAM_NAMES)
    actual   = set(events.keys())
    missing  = expected - actual
    extra    = actual - expected
    if missing or extra:
        msg = f"Event dict mismatch"
        if context:
            msg += f" in {context}"
        if missing:
            msg += f" — missing: {missing}"
        if extra:
            msg += f" — unexpected: {extra}"
        raise KeyError(msg)