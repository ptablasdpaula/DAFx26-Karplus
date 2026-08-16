"""Precompute the loss landscapes the project page's optimisation demos descend.

The browser cannot run the paper's synthesiser or its losses, so we sample them
here on a grid and ship the surface. The demos then do gradient descent on the
interpolated surface, which means the tKSA/fKSA divergence a reader sees is the
genuine one: it comes from the different landscapes time-aliasing produces, not
from anything faked in JavaScript.

Everything is forward-only, so this needs no autograd and runs fine on CPU.

Usage
-----
    KARPLUS_REPO=../DAFx26-Karplus-main \
      ../DAFx26-Karplus-main/.pixi/envs/default/bin/python tools/precompute_landscapes.py

Outputs ``assets/data/landscape_*.bin`` (uint16-quantised) plus a manifest
``assets/data/landscapes.json`` describing the axes and per-slice loss range.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

WEB_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = WEB_ROOT / "assets" / "data"

_default_repo = WEB_ROOT.parent / "DAFx26-Karplus-main"
KARPLUS_REPO = Path(os.environ.get("KARPLUS_REPO", _default_repo)).expanduser().resolve()

if not (KARPLUS_REPO / "src" / "synths" / "synth.py").is_file():
    sys.exit(
        f"Could not find src/synths/synth.py under {KARPLUS_REPO}.\n"
        "Set KARPLUS_REPO to a checkout of the DAFx26-Karplus code repository:\n"
        "  git clone -b main https://github.com/ptablasdpaula/DAFx26-Karplus.git"
    )

sys.path.insert(0, str(KARPLUS_REPO))

from src.losses import MultiScaleSpectralLoss, SOT2048Loss  # noqa: E402
from src.synths.ddsp import Implementation  # noqa: E402
from src.synths.synth import Synth, SynthConfig  # noqa: E402

# ── Signal configuration ────────────────────────────────────────────────────
# The paper's rate. It matters: at 4 kHz a 110 Hz string has only 18 harmonics
# below Nyquist against 72 here, and a bin-wise loss needs those harmonics to
# locate pitch. Sampled at 4 kHz the f0 gradient measures near chance for both
# losses, which contradicts Table 1; at 16 kHz it reproduces it.
FS = 16000
DURATION_S = 4.0
NUM_SAMPLES = int(FS * DURATION_S)

# Paper weights, Eq. 12.
W_MSS = 0.05
W_SOT = 1.0

# The paper's single-event gradient-analysis target (Sec. 6, "Gradient Analysis"):
# {f0, t, a1, g, p, dyn} = {110 Hz, 2 s, 0.2, 0.99, 0.25, 0.9}.
TARGET = dict(f0=110.0, decay=0.99, a1=0.2, pluck_position=0.25, dynamic_level=0.9)

# fKSA frame sizes. At 16 kHz these span 128 ms to 1.02 s, straddling the point
# where the frame stops containing the decay. The largest matches the N_FFT the
# paper uses.
FFT_SIZES = [2048, 4096, 8192, 16384]
# For landscape B we want fKSA to be effectively alias-free, mirroring the
# N_FFT = 16384 the paper uses for its multi-event onset experiments.
FFT_LONG = 16384

DEVICE = torch.device("cpu")


def make_params(batch: int, *, f0, decay, a1, onset, pluck=None, dynamic=None):
    """Build a single-event parameter dict for a batch of grid points.

    Scalars broadcast; arrays must be length ``batch``. ``onset`` is in seconds
    and gets normalised to the [0, 1] the synth expects.
    """

    def col(value):
        return torch.as_tensor(
            np.broadcast_to(np.asarray(value, dtype=np.float32), (batch,)).copy(),
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(1)

    return {
        "exists": torch.ones(batch, 1, device=DEVICE),
        "time": col(np.asarray(onset, dtype=np.float32) / DURATION_S),
        "f0": col(f0),
        "decay": col(decay),
        "a1": col(a1),
        "pluck_position": col(pluck if pluck is not None else TARGET["pluck_position"]),
        "dynamic_level": col(dynamic if dynamic is not None else TARGET["dynamic_level"]),
        "burst_gain": torch.ones(batch, 1, device=DEVICE),
    }


def build_synth(implementation: Implementation, n_fft: int) -> Synth:
    return Synth(
        SynthConfig(
            num_samples=NUM_SAMPLES,
            fs=FS,
            device=str(DEVICE),
            n_fft=n_fft,
            hop_length=n_fft // 8,
            implementation=implementation,
        )
    ).to(DEVICE)


class Objective:
    """L = w_MSS * L_MSS + w_SOT * L_SOT, evaluated per batch item."""

    def __init__(self):
        self.mss = MultiScaleSpectralLoss().to(DEVICE)
        self.sot = SOT2048Loss(sample_rate=FS, reduce=False).to(DEVICE)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        """Combined loss, at the paper's weights."""
        mss, sot = self.split(pred, target)
        return W_MSS * mss + W_SOT * sot

    def split(self, pred: torch.Tensor, target: torch.Tensor):
        """The two terms separately, so the page can reweight them live.

        This matters: the training objective is dominated by SOT, and Table 1
        shows SOT is close to chance on f0 while MSS is not. Storing only the
        sum would hide which term is responsible for which behaviour.
        """
        mss_out = np.empty(pred.shape[0], dtype=np.float64)
        with torch.no_grad():
            sot = self.sot(pred, target.expand_as(pred))
            sot_out = sot.reshape(pred.shape[0], -1).mean(dim=1).cpu().numpy().astype(np.float64)
            for i in range(pred.shape[0]):
                # MSS reduces over the batch, so feed it one pair at a time.
                mss_out[i] = float(self.mss(pred[i : i + 1], target[:1]))
        return mss_out, sot_out


def render(synth: Synth, **kwargs) -> torch.Tensor:
    n = len(np.atleast_1d(kwargs.get("decay", [0])))
    for key in ("decay", "a1", "f0", "onset"):
        value = kwargs.get(key)
        if value is not None:
            n = max(n, len(np.atleast_1d(value)))
    with torch.no_grad():
        audio, _ = synth(make_params(n, **kwargs))
    return audio


def quantise(grid: np.ndarray) -> tuple[np.ndarray, float, float]:
    lo, hi = float(np.nanmin(grid)), float(np.nanmax(grid))
    span = hi - lo if hi > lo else 1.0
    q = np.clip(np.rint((grid - lo) / span * 65535.0), 0, 65535).astype("<u2")
    return q, lo, hi


# ── Landscape A: decay x damping ────────────────────────────────────────────
def landscape_a(objective: Objective, manifest: dict, payload: list) -> None:
    n = 96
    # Geometric in (1 - g) so resolution concentrates where the decay is long,
    # which is where frequency sampling gets into trouble.
    one_minus_g = np.geomspace(1e-4, 1e-1, n)
    decay_axis = 1.0 - one_minus_g[::-1]
    a1_axis = np.geomspace(1e-3, 6e-1, n)

    onsets = {"0s": 0.0, "3s": 3.0}
    variants = [("tksa", Implementation.TIME_DOMAIN, FFT_LONG)]
    variants += [(f"fksa{f}", Implementation.FREQUENCY_SAMPLING, f) for f in FFT_SIZES]

    reference = build_synth(Implementation.TIME_DOMAIN, FFT_LONG)
    slices = {}

    for onset_name, onset_s in onsets.items():
        target = render(reference, f0=TARGET["f0"], decay=[TARGET["decay"]],
                        a1=TARGET["a1"], onset=onset_s)
        for var_name, impl, n_fft in variants:
            synth = build_synth(impl, n_fft)
            grid = np.empty((n, n), dtype=np.float64)
            t0 = time.time()
            for row, a1 in enumerate(a1_axis):
                pred = render(synth, f0=TARGET["f0"], decay=decay_axis,
                              a1=a1, onset=onset_s)
                grid[row] = objective(pred, target)
            key = f"A_{onset_name}_{var_name}"
            q, lo, hi = quantise(grid)
            slices[key] = {"offset": sum(p.nbytes for p in payload),
                           "min": lo, "max": hi}
            payload.append(q)
            print(f"  {key}: [{lo:.4f}, {hi:.4f}] in {time.time() - t0:.0f}s", flush=True)

    manifest["A"] = {
        "axes": {
            "x": {"name": "decay", "label": "decay $g$", "values": decay_axis.tolist()},
            "y": {"name": "a1", "label": "damping $a_1$", "values": a1_axis.tolist()},
        },
        "shape": [n, n],
        "target": {"decay": TARGET["decay"], "a1": TARGET["a1"]},
        "onsets": onsets,
        "fftSizes": FFT_SIZES,
        "slices": slices,
    }


# ── Landscape B: onset time x f0 ────────────────────────────────────────────
def landscape_b(objective: Objective, manifest: dict, payload: list) -> None:
    n = 96
    onset_axis = np.linspace(0.0, 3.5, n)
    f0_axis = np.geomspace(55.0, 880.0, n)

    # The target itself snaps to this 3x3 lattice: the landscape is defined
    # relative to a target, so an arbitrary draggable target is not precomputable.
    target_onsets = [0.5, 1.75, 3.0]
    target_f0s = [110.0, 220.0, 440.0]

    reference = build_synth(Implementation.TIME_DOMAIN, FFT_LONG)
    variants = [("tksa", Implementation.TIME_DOMAIN), ("fksa", Implementation.FREQUENCY_SAMPLING)]
    slices = {}

    for ti, t_onset in enumerate(target_onsets):
        for fi, t_f0 in enumerate(target_f0s):
            target = render(reference, f0=t_f0, decay=[TARGET["decay"]],
                            a1=TARGET["a1"], onset=t_onset)
            for var_name, impl in variants:
                synth = build_synth(impl, FFT_LONG)
                grid_mss = np.empty((n, n), dtype=np.float64)
                grid_sot = np.empty((n, n), dtype=np.float64)
                t0 = time.time()
                for row, f0 in enumerate(f0_axis):
                    pred = render(synth, f0=f0, decay=[TARGET["decay"]] * n,
                                  a1=TARGET["a1"], onset=onset_axis)
                    grid_mss[row], grid_sot[row] = objective.split(pred, target)
                for term, grid in (("mss", grid_mss), ("sot", grid_sot)):
                    key = f"B_t{ti}_f{fi}_{var_name}_{term}"
                    q, lo, hi = quantise(grid)
                    slices[key] = {"offset": sum(p.nbytes for p in payload),
                                   "min": lo, "max": hi}
                    payload.append(q)
                print(f"  B_t{ti}_f{fi}_{var_name}: mss [{grid_mss.min():.3f}, {grid_mss.max():.3f}] "
                      f"sot [{grid_sot.min():.3f}, {grid_sot.max():.3f}] in {time.time() - t0:.0f}s", flush=True)

    manifest["B"] = {
        "axes": {
            "x": {"name": "onset", "label": "onset time (s)", "values": onset_axis.tolist()},
            "y": {"name": "f0", "label": "$f_0$ (Hz)", "values": f0_axis.tolist()},
        },
        "shape": [n, n],
        "targets": {"onsets": target_onsets, "f0s": target_f0s},
        "fftLong": FFT_LONG,
        "terms": ["mss", "sot"],
        "weights": {"mss": W_MSS, "sot": W_SOT},
        "slices": slices,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    objective = Objective()
    manifest: dict = {
        "fs": FS,
        "durationS": DURATION_S,
        "numSamples": NUM_SAMPLES,
        "weights": {"mss": W_MSS, "sot": W_SOT},
        "fixed": TARGET,
        "binary": "landscapes.bin",
        "dtype": "uint16",
    }
    payload: list[np.ndarray] = []

    print("Landscape A (decay x damping)...", flush=True)
    landscape_a(objective, manifest, payload)
    print("Landscape B (onset x f0)...", flush=True)
    landscape_b(objective, manifest, payload)

    blob = b"".join(p.tobytes() for p in payload)
    (DATA_DIR / "landscapes.bin").write_bytes(blob)
    (DATA_DIR / "landscapes.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"\nWrote {len(blob) / 1024:.0f} KB to {DATA_DIR / 'landscapes.bin'}")
    print(f"Wrote manifest to {DATA_DIR / 'landscapes.json'}")


if __name__ == "__main__":
    main()
