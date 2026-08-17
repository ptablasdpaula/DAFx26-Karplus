from __future__ import annotations

import copy

import pytest
import torch

from src.synths.ddsp import Implementation
from src.synths.dense import DenseKSConfig, DenseKSSynth
from src.synths.synth import Synth, SynthConfig


def valid_controls(config: DenseKSConfig, *, requires_grad: bool = False):
    frames = config.num_frames
    make = lambda value: torch.full((1, frames), value, dtype=torch.float32)
    controls = {
        "noise_gate": make(0.0),
        "delay": make(config.fs / 110.0),
        "decay": make(0.97),
        "a1": make(0.2),
        "pluck_position": make(0.25),
        "dynamics": make(0.7),
    }
    controls["noise_gate"][0, 1] = 1.0
    if requires_grad:
        controls = {key: value.requires_grad_(True) for key, value in controls.items()}
    return controls


def test_dense_renderer_shape_bounds_finite_and_deterministic():
    config = DenseKSConfig(num_samples=512, num_frames=8)
    first = DenseKSSynth(config)
    second = DenseKSSynth(config)
    controls = valid_controls(config)
    audio_a = first(controls)
    audio_b = second(controls)
    assert audio_a.shape == (1, 512)
    assert torch.isfinite(audio_a).all()
    torch.testing.assert_close(first.noise_carrier, second.noise_carrier, rtol=0, atol=0)
    torch.testing.assert_close(audio_a, audio_b, rtol=0, atol=0)

    invalid = valid_controls(config)
    invalid["decay"] = torch.full_like(invalid["decay"], 1.0)
    with pytest.raises(ValueError, match="decay"):
        first(invalid)
    silent = valid_controls(config)
    silent["noise_gate"].zero_()
    with pytest.raises(ValueError, match="silent"):
        first(silent)


def test_gate_scale_invariance():
    config = DenseKSConfig(num_samples=512, num_frames=8)
    synth = DenseKSSynth(config)
    controls = valid_controls(config)
    scaled = {key: value.clone() for key, value in controls.items()}
    scaled["noise_gate"] *= 17.0
    torch.testing.assert_close(synth(controls), synth(scaled), rtol=2e-6, atol=2e-6)


def test_delay_gradient_matches_finite_difference_away_from_integer_boundary():
    config = DenseKSConfig(num_samples=384, num_frames=6)
    synth = DenseKSSynth(config)
    controls = valid_controls(config, requires_grad=True)
    probe = torch.linspace(-1.0, 1.0, config.num_samples)
    objective = (synth(controls)[0] * probe).sum()
    objective.backward()
    analytic = float(controls["delay"].grad[0, 3])

    epsilon = 1e-3
    plus = {key: value.detach().clone() for key, value in controls.items()}
    minus = {key: value.detach().clone() for key, value in controls.items()}
    plus["delay"][0, 3] += epsilon
    minus["delay"][0, 3] -= epsilon
    numerical = float(
        (((synth(plus)[0] - synth(minus)[0]) * probe).sum() / (2.0 * epsilon))
    )
    assert analytic == pytest.approx(numerical, rel=0.08, abs=0.02)


def test_dense_renderer_does_not_change_event_renderer_behavior():
    config = SynthConfig(
        num_samples=512,
        fs=16_000,
        random_seed=42,
        implementation=Implementation.TIME_DOMAIN,
    )
    params = {
        "exists": torch.tensor([[1.0]]),
        "time": torch.tensor([[0.05]]),
        "f0": torch.tensor([[110.0]]),
        "burst_gain": torch.tensor([[0.8]]),
        "decay": torch.tensor([[0.97]]),
        "a1": torch.tensor([[0.2]]),
        "pluck_position": torch.tensor([[0.25]]),
        "dynamic_level": torch.tensor([[0.7]]),
    }
    before, _ = Synth(config)(copy.deepcopy(params))
    dense_config = DenseKSConfig(num_samples=512, num_frames=8)
    DenseKSSynth(dense_config)(valid_controls(dense_config))
    after, _ = Synth(config)(copy.deepcopy(params))
    torch.testing.assert_close(before, after, rtol=0, atol=0)
