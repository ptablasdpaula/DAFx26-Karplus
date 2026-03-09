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


# ═════════════════════════════════════════════════════════════════════════════
# Event → Frame expansion
# ═════════════════════════════════════════════════════════════════════════════

import numpy as np


def events_to_frames_np(
        events: dict[str, np.ndarray],
        n_events: int,
        num_samples: int,  # We pass in num_samples (64000) now, NOT num_frames!
        duration_s: float,
) -> dict[str, np.ndarray]:
    """Expands sparse events into dense sample-rate arrays.

    Initializes arrays with safe physical values to prevent ZeroDivisionError
    in the Karplus-Strong DSP engine, and backfills the first event to t=0.
    """

    # 1. Initialize with SAFE physical priors (not zeros!)
    dense = {
        "f0": np.full(num_samples, 440.0, dtype=np.float32),
        "burst_gain": np.zeros(num_samples, dtype=np.float32),  # Gain stays 0
        "decay": np.full(num_samples, DECAY_MIN, dtype=np.float32),
        "a1": np.full(num_samples, 0.5, dtype=np.float32),
        "pluck_position": np.full(num_samples, 0.5, dtype=np.float32),
        "dynamic_level": np.full(num_samples, 0.5, dtype=np.float32),
    }

    if n_events == 0:
        return dense

    for i in range(n_events):
        # Calculate exact sample indices
        start_sample = int(events["time"][i] * num_samples)
        start_sample = max(0, min(start_sample, num_samples - 1))

        end_sample = num_samples
        if i < n_events - 1:
            end_sample = int(events["time"][i + 1] * num_samples)
            end_sample = max(start_sample, min(end_sample, num_samples))

        # Backfill: If it's the first event, stretch its parameters to t=0
        # so the string is "ready" to be plucked.
        fill_start = 0 if i == 0 else start_sample

        # Fill continuous parameters
        dense["f0"][fill_start:end_sample] = events["f0"][i]
        dense["decay"][fill_start:end_sample] = events["decay"][i]
        dense["a1"][fill_start:end_sample] = events["a1"][i]
        dense["pluck_position"][fill_start:end_sample] = events["pluck_position"][i]
        dense["dynamic_level"][fill_start:end_sample] = events["dynamic_level"][i]

        # Burst gain is a sharp impulse!
        dense["burst_gain"][start_sample] = events["burst_gain"][i]

    return dense


def events_to_frames(
    events: dict[str, Tensor],
    n_events: Tensor,
    num_frames: int,
    duration_s: float,
) -> dict[str, Tensor]:
    """Batched, differentiable event → frame expansion (torch version).

    Same step-function logic as ``events_to_frames_np`` but operates on
    torch tensors.  Used by the decoder to convert DETR predictions into
    the dense frame-rate params the KS synth expects.

    Args:
        events:     ``{name: [B, max_events]}`` tensors.
        n_events:   ``[B]`` int tensor — number of real events per batch item.
        num_frames: Number of control-rate frames.
        duration_s: Audio duration in seconds.

    Returns:
        ``{name: [B, num_frames]}`` tensors.
    """
    B = events["time"].shape[0]
    device = events["time"].device
    dtype = events["time"].dtype
    fps = num_frames / duration_s

    frames = {pname: torch.zeros(B, num_frames, device=device, dtype=dtype)
              for pname in PARAM_NAMES}

    for b in range(B):
        n = int(n_events[b].item())
        if n == 0:
            continue

        # Sort by time
        times_b = events["time"][b, :n]
        order = torch.argsort(times_b)

        onset_frames_b = (times_b[order] * duration_s * fps).long().clamp(0, num_frames - 1)

        for i in range(n):
            idx = order[i]
            frame = onset_frames_b[i]
            next_frame = onset_frames_b[i + 1] if i + 1 < n else num_frames

            for pname in EVENT_SYNTH_PARAMS:
                if pname == "burst_gain":
                    frames["burst_gain"][b, frame] = events["burst_gain"][b, idx]
                else:
                    frames[pname][b, frame:next_frame] = events[pname][b, idx]

        # Fill pre-first-onset
        first_idx = order[0]
        first_frame = onset_frames_b[0]
        if first_frame > 0:
            for pname in EVENT_SYNTH_PARAMS:
                if pname != "burst_gain":
                    frames[pname][b, :first_frame] = events[pname][b, first_idx]

    return frames