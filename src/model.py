from __future__ import annotations

import torch
import torch.nn as nn

from src.encoder import KSEventEncoder, HpNEncoder
from src.decoder import Decoder, KSDecoder, HarmonicsNoiseDecoder
from src.synths.synth import SynthOutput


class SoundMatchingModel(nn.Module):
    """
    Encoder → decoder (activation + synthesis).

    Auto-selects the encoder based on the decoder type:

    * ``KSDecoder`` → ``KSEventEncoder`` (Unified DETR-style event detector)
    * ``HarmonicsNoiseDecoder`` → ``HpNEncoder`` (single-pipeline frontend + TCN)

    Args:
        decoder:         Any ``Decoder`` subclass.
        encoder_kwargs:  Forwarded to the auto-selected encoder.
                         For KS this should contain kwargs like ``max_events``,
                         ``cross_attn_layers``, ``d_model``, etc.
                         For H+N this should contain ``tcn_channels``,
                         ``num_blocks``, etc.
    """

    def __init__(
            self,
            decoder: Decoder,
            encoder_kwargs: dict | None = None,
    ):
        super().__init__()
        enc_kw = dict(encoder_kwargs or {})

        if isinstance(decoder, KSDecoder):
            self.encoder = KSEventEncoder(**enc_kw)
        elif isinstance(decoder, HarmonicsNoiseDecoder):
            enc_kw["num_outputs"] = getattr(decoder, "z_dim", 16)
            self.encoder = HpNEncoder(**enc_kw)
        else:
            raise ValueError(f"Unknown decoder type: {type(decoder)}")

        self.decoder = decoder

    @property
    def last_f0_probs(self) -> torch.Tensor | None:
        """Access the soft-argmax f0 distribution from the last forward pass.

        Only available for ``KSEventEncoder``.  Returns [B, max_events, n_f0_bins].
        """
        if isinstance(self.encoder, KSEventEncoder):
            return getattr(self.encoder, 'last_f0_probs', None)
        return None

    def forward(
            self,
            wav: torch.Tensor,
            detected: dict[str, torch.Tensor] | None = None,
    ) -> SynthOutput:
        """
        Args:
            wav:       [B, num_samples] input audio.
            detected:  External-detector signals (decoder-specific).

        Returns:
            (audio [B, num_samples], params {name: [B, T] or [B, max_events]})
        """
        raw = self.encoder(wav)

        if isinstance(self.encoder, KSEventEncoder) and isinstance(raw, dict) and "f0_probs" in raw:
            self.encoder.last_f0_probs = raw["f0_probs"]

        params = self.decoder.activate(raw, detected)

        if self.training:
            audio, params = self.decoder.synthesise(params)
        else:
            audio, params = self.decoder.oracle_synth(params)

        return audio, params