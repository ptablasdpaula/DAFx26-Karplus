from __future__ import annotations

import torch
import torch.nn as nn

from src.encoder import KSEncoder, HpNEncoder
from src.decoder import Decoder, KSDecoder, HarmonicsNoiseDecoder
from src.synths.synth import SynthOutput


class SoundMatchingModel(nn.Module):
    """
    Encoder → decoder (activation + synthesis).

    Auto-selects the encoder based on the decoder type:

    *   ``KSDecoder`` → ``KSEncoder`` (CQT resonator + learnable excitation)
    *   ``HarmonicsNoiseDecoder`` → ``HpNEncoder`` (single-pipeline frontend + TCN)

    Args:
        decoder:         Any ``Decoder`` subclass.
        encoder_kwargs:  Forwarded to the auto-selected encoder.
                         For KS this should contain ``resonator_kwargs``
                         and ``excitation_kwargs`` dicts.
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
            self.encoder = KSEncoder(**enc_kw)
        elif isinstance(decoder, HarmonicsNoiseDecoder):
            enc_kw["num_outputs"] = decoder.num_params
            self.encoder = HpNEncoder(**enc_kw)
        else:
            raise ValueError(f"Unknown decoder type: {type(decoder)}")

        self.decoder = decoder

    @property
    def last_f0_probs(self) -> torch.Tensor | None:
        """Access the soft-argmax f0 distribution from the last forward pass.

        Only available for ``KSEncoder``.  Returns [B, T, n_f0_bins].
        """
        if isinstance(self.encoder, KSEncoder):
            return self.encoder.last_f0_probs
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
            (audio [B, num_samples], params {name: [B, T] or [B, C, T]})
        """
        raw = self.encoder(wav)
        params = self.decoder.activate(raw, detected)

        if self.training:
            audio, params = self.decoder.synthesise(params)
        else:
            audio, params = self.decoder.oracle_synth(params)

        return audio, params