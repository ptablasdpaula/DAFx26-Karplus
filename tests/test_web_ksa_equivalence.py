"""Numerical regression test for the extended KSA used by the project page.

Run from the repository root with:
    cd main && pixi run python ../tests/test_web_ksa_equivalence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main"))
from src.synths import dsp


FS = 16_000
F0 = 233.08188075904496
N = 4096


def web_all_zero_comb(x: np.ndarray, delay: np.ndarray) -> np.ndarray:
    buffer = np.zeros(int(FS / dsp.F0_MIN))
    output = np.zeros_like(x)
    write = 0
    for n, sample in enumerate(x):
        integer = int(np.floor(delay[n]))
        fraction = delay[n] - integer
        index0 = (write - integer) % len(buffer)
        index1 = (write - integer - 1) % len(buffer)
        delayed = (1 - fraction) * buffer[index0] + fraction * buffer[index1]
        output[n] = sample - delayed
        buffer[write] = sample
        write = (write + 1) % len(buffer)
    return output


def web_pluck_position(x: np.ndarray, f0: np.ndarray, position: np.ndarray) -> np.ndarray:
    return web_all_zero_comb(x, (FS / f0) * np.clip(position, 0.01, 0.5))


def web_dynamics(x: np.ndarray, f0: np.ndarray, level: np.ndarray) -> np.ndarray:
    output = np.zeros_like(x)
    state = 0.0
    for n, sample in enumerate(x):
        coefficient = dsp.compute_dynamics_R(f0[n], level[n], FS)
        state = (1 - coefficient) * sample + coefficient * state
        output[n] = state
    return output


def web_karplus_strong(x: np.ndarray, f0: np.ndarray, a1: np.ndarray, decay: np.ndarray) -> np.ndarray:
    buffer = np.zeros(int(FS / dsp.F0_MIN))
    output = np.zeros_like(x)
    filter_state = 0.0
    write = 0
    for n, excitation in enumerate(x):
        corrected = FS / f0[n] + dsp.one_pole_phase_delay(f0[n], a1[n], FS)
        integer, coefficients = dsp.lagrange_fractional_delay(corrected, dsp.LAGRANGE_ORDER)
        delayed = sum(
            coefficient * buffer[(write - integer - k) % len(buffer)]
            for k, coefficient in enumerate(coefficients)
        )
        filter_state = decay[n] * (1 - a1[n]) * delayed + a1[n] * filter_state
        output[n] = excitation + filter_state
        buffer[write] = output[n]
        write = (write + 1) % len(buffer)
    return output


def error(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    return float(np.max(np.abs(reference - candidate))), float(np.sqrt(np.mean((reference - candidate) ** 2)))


def assert_equivalent(label: str, reference: np.ndarray, candidate: np.ndarray) -> None:
    maximum, rms = error(reference, candidate)
    print(f"{label:34s} max={maximum:.3e}  rms={rms:.3e}")
    np.testing.assert_allclose(candidate, reference, rtol=1e-11, atol=1e-12)


def main() -> None:
    rng = np.random.default_rng(7)
    excitation = rng.uniform(-1, 1, N)
    excitation[256:] = 0
    f0 = np.full(N, F0)

    for position_value in (0.01, 0.22, 0.333, 0.5):
        position = np.full(N, position_value)
        assert_equivalent(
            f"pluck position={position_value}",
            dsp.pluck_position_filter(excitation, f0, position, FS),
            web_pluck_position(excitation, f0, position),
        )

    for dynamics_value in (0.0, 0.2, 0.62, 1.0):
        dynamics = np.full(N, dynamics_value)
        assert_equivalent(
            f"dynamics={dynamics_value}",
            dsp.dynamics_filter(excitation, f0, dynamics, FS),
            web_dynamics(excitation, f0, dynamics),
        )

    impulse = np.zeros(N)
    impulse[0] = 1
    for damping_value in (0.00001, 0.1, 0.34, 0.9):
        damping = np.full(N, damping_value)
        decay = np.full(N, 0.994)
        assert_equivalent(
            f"KSA damping={damping_value}",
            dsp.karplus_strong(impulse, f0, damping, decay, FS),
            web_karplus_strong(impulse, f0, damping, decay),
        )

    for decay_value in (0.9, 0.97, 0.994, 0.99999):
        damping = np.full(N, 0.34)
        decay = np.full(N, decay_value)
        assert_equivalent(
            f"KSA decay={decay_value}",
            dsp.karplus_strong(impulse, f0, damping, decay, FS),
            web_karplus_strong(impulse, f0, damping, decay),
        )

    position = np.full(N, 0.22)
    dynamics = np.full(N, 0.62)
    damping = np.full(N, 0.34)
    decay = np.full(N, 0.994)
    reference_excitation = dsp.pluck_position_filter(excitation, f0, position, FS)
    reference_excitation = dsp.dynamics_filter(reference_excitation, f0, dynamics, FS)
    reference = dsp.karplus_strong(reference_excitation, f0, damping, decay, FS)
    web_excitation = web_pluck_position(excitation, f0, position)
    web_excitation = web_dynamics(web_excitation, f0, dynamics)
    candidate = web_karplus_strong(web_excitation, f0, damping, decay)
    assert_equivalent("full extended KSA", reference, candidate)

    print("All browser/reference equivalence checks passed.")


if __name__ == "__main__":
    main()
