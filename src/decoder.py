from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.synths.synth import Synth, SynthOutput
from src.synths.param_registry import (
    PARAM_NAMES as KS_PARAM_NAMES,
    F0_MIN_HZ, F0_MAX_HZ,
    PLUCK_POSITION_MIN, PLUCK_POSITION_MAX,
    DYNAMIC_LEVEL_MIN, DYNAMIC_LEVEL_MAX,
    DAMPING_MIN, DAMPING_MAX,
    DECAY_MIN, DECAY_MAX,
    BURST_GAIN_MAX,
)


class Decoder(ABC, nn.Module):
    """Abstract decoder interface.

    Subclasses must define
    ---------------------
    num_params      : int
    param_names     : tuple[str, ...]
    activate(raw, detected) → dict
    synthesise(params) → SynthOutput
    oracle_synth(params) → SynthOutput
    """

    num_params: int
    param_names: tuple[str, ...]

    @abstractmethod
    def activate(
        self,
        raw: Tensor,
        detected: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Map raw encoder logits [B, P, T] → {name: [B, T]}.

        ``detected`` carries external-detector signals.  Each subclass
        decides which keys it consumes (see module docstring).
        """
        ...

    @abstractmethod
    def synthesise(self, params: dict[str, Tensor]) -> SynthOutput:
        """Differentiable synthesis.  Returns (audio [B, N], params)."""
        ...

    @abstractmethod
    def oracle_synth(self, params: dict[str, Tensor]) -> SynthOutput:
        """Non-differentiable high-fidelity synthesis."""
        ...

    def forward(self, params: dict[str, Tensor]) -> SynthOutput:
        return self.synthesise(params)

def _sigmoid_range(x: Tensor, lo: float, hi: float) -> Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)

# ═════════════════════════════════════════════════════════════════════════════
# Karplus-Strong Decoder
# ═════════════════════════════════════════════════════════════════════════════
class KSDecoder(Decoder):
    def __init__(
            self,
            synth: Synth,
            use_external_detectors: bool = False
    ):
        super().__init__()
        self.synth = synth
        self.use_external_detectors = use_external_detectors

        self.param_names = KS_PARAM_NAMES
        self.num_params = len(self.param_names)
        self._idx = {name: i for i, name in enumerate(self.param_names)}

    def activate(
        self,
        raw: Tensor,
        detected: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        detected = detected or {}
        ext = self.use_external_detectors

        # ── f0 ──
        det_f0 = detected.get("f0")
        if ext and det_f0 is not None:
            f0 = det_f0
        else:
            f0 = _sigmoid_range(raw[:, self._idx["f0"], :],
                                F0_MIN_HZ, F0_MAX_HZ)

        # ── burst_gain ──
        det_onsets = detected.get("onsets")
        if ext and det_onsets is not None:
            burst_gain = (
                BURST_GAIN_MAX
                * torch.sigmoid(raw[:, self._idx["burst_gain"], :])
                * det_onsets
            )
        else:
            burst_gain = F.relu(raw[:, self._idx["burst_gain"], :])

        # ── remaining params ──
        pluck_pos = _sigmoid_range(raw[:, self._idx["pluck_position"], :],
                                   PLUCK_POSITION_MIN, PLUCK_POSITION_MAX)
        dyn_level = _sigmoid_range(raw[:, self._idx["dynamic_level"], :],
                                   DYNAMIC_LEVEL_MIN + 1e-3, DYNAMIC_LEVEL_MAX)
        a1 = _sigmoid_range(raw[:, self._idx["a1"], :],
                            DAMPING_MIN, DAMPING_MAX)
        decay = _sigmoid_range(raw[:, self._idx["decay"], :],
                               DECAY_MIN, DECAY_MAX)

        return {
            "f0":             f0,
            "burst_gain":     burst_gain,
            "pluck_position": pluck_pos,
            "dynamic_level":  dyn_level,
            "a1":             a1,
            "decay":          decay,
        }

    def synthesise(self, params: dict[str, Tensor]) -> SynthOutput:
        return self.synth(params)

    @torch.no_grad()
    def oracle_synth(self, params: dict[str, Tensor]) -> SynthOutput:
        return self.synth.oracle_synth(params)


# ═════════════════════════════════════════════════════════════════════════════
# Harmonics + Noise Decoder (stub)
# ═════════════════════════════════════════════════════════════════════════════
'''
class HarmonicsNoiseDecoder(Decoder):
    """Harmonics + Noise decoder (DDSP-style).

    Parameter layout (for default 100 harmonics, 65 noise bands)::

        Channel     Name                    Activation
        --------    --------------------    -------------------
        0           amplitude               modified_sigmoid
        1..100      harmonic_distribution   softmax (100 ch)
        101..165    noise_magnitudes        sigmoid → [0, 1]
        166         f0                      sigmoid_range

    Total encoder outputs: 167.

    External detectors (``detected`` dict)::

        "f0"       : [B, T] Hz       → replaces encoder f0
        "loudness" : [B, T] dB       → conditions amplitude

    Stub — implement ``synthesise()`` and ``oracle_synth()`` when the
    H+N DSP backend is ready.
    """

    def __init__(
        self,
        fs: int = 16000,
        num_samples: int = 64000,
        n_harmonics: int = 100,
        n_noise_bands: int = 65,
        use_external_detectors: bool = False,
    ):
        super().__init__()
        self.fs = fs
        self.num_samples = num_samples
        self.n_harmonics = n_harmonics
        self.n_noise_bands = n_noise_bands
        self.use_external_detectors = use_external_detectors

        # amp + harmonics + noise + f0
        self.num_params = 1 + n_harmonics + n_noise_bands + 1
        self.param_names = ("amplitude", "harmonic_distribution",
                            "noise_magnitudes", "f0")

    def activate(
        self,
        raw: Tensor,
        detected: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        detected = detected or {}
        ext = self.use_external_detectors
        nh = self.n_harmonics
        nn_ = self.n_noise_bands

        # ── amplitude — optionally conditioned on detected loudness ──
        det_loudness = detected.get("loudness")
        if ext and det_loudness is not None:
            # Scale raw amplitude logit by detected loudness (dB → linear)
            loudness_linear = 10.0 ** (det_loudness / 20.0)
            amplitude = torch.sigmoid(raw[:, 0, :]) * loudness_linear
        else:
            amplitude = torch.sigmoid(raw[:, 0, :])                          # [B, T]

        # ── harmonic distribution ──
        harm_dist = F.softmax(raw[:, 1:1 + nh, :], dim=1)                    # [B, nh, T]

        # ── noise magnitudes ──
        noise_mag = torch.sigmoid(raw[:, 1 + nh:1 + nh + nn_, :])            # [B, nn, T]

        # ── f0 ──
        det_f0 = detected.get("f0")
        if ext and det_f0 is not None:
            f0 = det_f0
        else:
            f0 = _sigmoid_range(raw[:, -1, :], F0_MIN_HZ, F0_MAX_HZ)         # [B, T]

        return {
            "amplitude":             amplitude,
            "harmonic_distribution": harm_dist,
            "noise_magnitudes":      noise_mag,
            "f0":                    f0,
        }

    def synthesise(self, params: dict[str, Tensor]) -> SynthOutput:
        raise NotImplementedError(
            "H+N synthesis not yet implemented. "
            "Plug in your DDSP harmonic + noise synthesiser here."
        )

    def oracle_synth(self, params: dict[str, Tensor]) -> SynthOutput:
        raise NotImplementedError("H+N oracle synth not yet implemented.")
'''