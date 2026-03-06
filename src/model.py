from __future__ import annotations

import torch
import torch.nn as nn

from src.encoder import Encoder
from src.decoder import Decoder
from src.synths.synth import SynthOutput


class SoundMatchingModel(nn.Module):
    """
    Encoder → decoder (activation + synthesis).

    Args:
        decoder:         Any ``Decoder`` subclass.
        encoder_kwargs:  Forwarded to ``Encoder.__init__()``.  ``num_outputs``
                         is set automatically from ``decoder.num_params``.
    """

    def __init__(
        self,
        decoder: Decoder,
        encoder_kwargs: dict | None = None,
    ):
        super().__init__()
        enc_kw = dict(encoder_kwargs or {})
        enc_kw["num_outputs"] = decoder.num_params

        self.encoder = Encoder(**enc_kw)
        self.decoder = decoder

    def forward(
        self,
        wav: torch.Tensor,
        detected: dict[str, torch.Tensor] | None = None,
    ) -> SynthOutput:
        """
        Args:
            wav:       [B, num_samples] input audio.
            detected:  External-detector signals.  Keys are decoder-specific:
                       KS expects ``{"onsets": …, "f0": …}``,
                       H+N expects ``{"loudness": …, "f0": …}``.
                       Pass ``None`` or ``{}`` when not using external detectors.

        Returns:
            (audio [B, num_samples], params {name: [B, T] or [B, C, T]})
        """
        raw = self.encoder(wav)                        # [B, P, T]
        params = self.decoder.activate(raw, detected)  # {name: tensor}

        if self.training:
            audio, params = self.decoder.synthesise(params)
        else:
            audio, params = self.decoder.oracle_synth(params)

        return audio, params