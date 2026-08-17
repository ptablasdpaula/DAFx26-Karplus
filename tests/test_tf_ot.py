from __future__ import annotations

import numpy as np
import pytest
import torch

from src.audio_objectives import rms_match
from src.tf_ot import (
    DifferentiableTFOT,
    TFOTConfig,
    canonical_tf_ot_spectrogram,
)


@pytest.fixture
def small_metric():
    config = TFOTConfig(
        sample_rate=16_000,
        n_fft=32,
        hop_length=8,
        projections=10,
        quantiles=512,
    )
    return config, DifferentiableTFOT(config)


def test_self_comparison_is_zero_with_zero_subgradient(small_metric):
    config, metric = small_metric
    magnitude = torch.rand(1, config.n_fft // 2 + 1, 9, requires_grad=True)
    loss = metric.spectrogram_distance(magnitude, magnitude)
    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(magnitude.grad, torch.zeros_like(magnitude))


@pytest.mark.parametrize("axis", ["time", "frequency"])
def test_displacements_track_canonical_metric(small_metric, axis):
    config, metric = small_metric
    first = torch.zeros(1, config.n_fft // 2 + 1, 9)
    second = torch.zeros_like(first)
    first[0, 4, 3] = 1.0
    if axis == "time":
        second[0, 4, 5] = 1.0
    else:
        second[0, 7, 3] = 1.0
    differentiable = float(metric.spectrogram_distance(first, second))
    canonical = canonical_tf_ot_spectrogram(
        first[0].numpy(), second[0].numpy(), config=config,
    )
    assert differentiable > 0.0
    assert differentiable == pytest.approx(canonical, rel=0.03, abs=1e-5)
    differentiable_first = first.clone().requires_grad_(True)
    metric.spectrogram_distance(differentiable_first, second).backward()
    assert torch.isfinite(differentiable_first.grad).all()


def test_global_gain_invariance_and_silence_are_finite(small_metric):
    config, metric = small_metric
    magnitude = torch.rand(1, config.n_fft // 2 + 1, 9)
    gain_distance = metric.spectrogram_distance(magnitude, 7.0 * magnitude)
    assert float(gain_distance) == pytest.approx(0.0, abs=2e-4)
    canonical_gain = canonical_tf_ot_spectrogram(
        magnitude[0].numpy(), (7.0 * magnitude[0]).numpy(), config=config,
    )
    assert canonical_gain == 0.0
    silence = torch.zeros_like(magnitude)
    assert metric.spectrogram_distance(silence, silence).item() == 0.0
    one_silent = metric.spectrogram_distance(silence, magnitude)
    assert torch.isfinite(one_silent)
    assert one_silent.item() > 0.0


def test_audio_gradient_is_finite(small_metric):
    config, metric = small_metric
    torch.manual_seed(4)
    prediction = torch.randn(1, 256, requires_grad=True)
    target = torch.randn_like(prediction)
    loss = metric(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0.0


def test_rms_matching_is_prediction_scale_invariant():
    torch.manual_seed(9)
    prediction = torch.randn(2, 128)
    target = torch.randn(2, 128)
    torch.testing.assert_close(
        rms_match(prediction, target),
        rms_match(13.0 * prediction, target),
        rtol=2e-6,
        atol=2e-6,
    )
