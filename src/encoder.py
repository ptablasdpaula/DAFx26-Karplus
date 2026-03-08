from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class CausalConv1d(nn.Conv1d):
    """Causal 1-D convolution (left-pad only)."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True):
        super().__init__(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=0, dilation=dilation,
            groups=groups, bias=bias,
        )
        self._causal_padding = dilation * (kernel_size - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self._causal_padding, 0)))


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int,
                 kernel_size: int, dilation: int = 1,
                 dropout: float = 0.1, last_block: bool = False):
        super().__init__()
        block = [
            weight_norm(CausalConv1d(in_ch, hidden_ch, kernel_size, dilation=dilation)),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(CausalConv1d(hidden_ch, out_ch, kernel_size, dilation=dilation)),
        ]
        if not last_block:
            block.extend([nn.ReLU(), nn.Dropout(dropout)])
        self.block = nn.Sequential(*block)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.residual(x)


def _build_tcn(in_ch: int, tcn_channels: int, num_blocks: int,
               kernel_size: int, dilation_base: int,
               dropout: float) -> nn.Sequential:
    blocks = []
    ch = in_ch
    for i in range(num_blocks):
        dilation = dilation_base ** i
        blocks.append(TCNBlock(ch, tcn_channels, tcn_channels,
                               kernel_size, dilation, dropout,
                               last_block=(i == num_blocks - 1)))
        ch = tcn_channels
    return nn.Sequential(*blocks)


# ═════════════════════════════════════════════════════════════════════════════
# Frontends
# ═════════════════════════════════════════════════════════════════════════════

class LearnableFrontend(nn.Module):
    """Strided conv stack: raw audio [B, 1, N] → features [B, C, T_frames].

    Good for temporal/energy features (onsets, dynamics).
    Total default stride = 256 samples.
    """

    DEFAULT_CHANNELS = [32, 64, 64, 64]
    DEFAULT_STRIDES  = [4, 4, 4, 4]
    DEFAULT_KERNELS  = [16, 16, 8, 8]

    def __init__(self, channels=None, strides=None, kernels=None):
        super().__init__()
        channels = channels or self.DEFAULT_CHANNELS
        strides  = strides  or self.DEFAULT_STRIDES
        kernels  = kernels  or self.DEFAULT_KERNELS

        layers = []
        in_ch = 1
        for out_ch, stride, kernel in zip(channels, strides, kernels):
            pad = (kernel - stride) // 2
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                          stride=stride, padding=pad),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ])
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: [B, N] → [B, C, T_frames]"""
        return self.net(wav.unsqueeze(1))


class CQTFrontend(nn.Module):
    """Constant-Q Transform frontend for pitch/resonance features.

    Computes a pseudo-CQT via a bank of log-spaced Gabor-like filters
    realised as a single (non-trainable) Conv1d, followed by magnitude
    and log-compression.

    Args:
        fs:              Sample rate (Hz).
        f_min:           Lowest CQT bin centre (Hz).
        n_octaves:       Number of octaves above ``f_min``.
        bins_per_octave: Frequency resolution (12 = semitone, 24 = quarter-tone).
        hop_length:      Frame hop in samples.
        q_factor:        Quality factor multiplier.
    """

    def __init__(
        self,
        fs: int = 16000,
        f_min: float = 32.0,
        n_octaves: int = 7,
        bins_per_octave: int = 24,
        hop_length: int = 256,
        q_factor: float = 1.0,
    ):
        super().__init__()
        self.fs = fs
        self.hop_length = hop_length
        self.n_bins = n_octaves * bins_per_octave

        freqs = f_min * 2.0 ** (torch.arange(self.n_bins).float()
                                 / bins_per_octave)
        Q = q_factor * (2 ** (1.0 / bins_per_octave) - 1) ** -1

        max_len = int(math.ceil(Q * fs / freqs[0].item()))
        if max_len % 2 == 0:
            max_len += 1

        filters_real = torch.zeros(self.n_bins, 1, max_len)
        filters_imag = torch.zeros(self.n_bins, 1, max_len)

        for k, f_k in enumerate(freqs):
            N_k = int(math.ceil(Q * fs / f_k.item()))
            if N_k % 2 == 0:
                N_k += 1
            t = torch.arange(N_k).float() - (N_k - 1) / 2
            window = torch.hann_window(N_k)
            kernel_real = window * torch.cos(2 * math.pi * f_k * t / fs)
            kernel_imag = window * torch.sin(2 * math.pi * f_k * t / fs)
            norm = kernel_real.norm() + 1e-8
            kernel_real = kernel_real / norm
            kernel_imag = kernel_imag / norm
            pad_left = (max_len - N_k) // 2
            filters_real[k, 0, pad_left:pad_left + N_k] = kernel_real
            filters_imag[k, 0, pad_left:pad_left + N_k] = kernel_imag

        self.register_buffer("filters_real", filters_real)
        self.register_buffer("filters_imag", filters_imag)
        self._pad = max_len // 2
        self.out_channels = self.n_bins

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: [B, N] → [B, n_bins, T_frames]  (log-magnitude CQT)."""
        x = wav.unsqueeze(1)
        x = F.pad(x, (self._pad, self._pad))

        re = F.conv1d(x, self.filters_real, stride=self.hop_length)
        im = F.conv1d(x, self.filters_imag, stride=self.hop_length)
        mag = torch.sqrt(re ** 2 + im ** 2 + 1e-8)

        return torch.log1p(mag)

class ResonatorEncoder(nn.Module):
    """CQT → TCN → GRU → soft-argmax f0 + raw logits for (decay, a1).

    Following Engel et al. (2020) "Self-supervised Pitch Detection by
    Inverse Audio Synthesis":

    **Soft-argmax f0 head:**  Logits over log-spaced frequency bins →
    softmax → frequency-bin-weighted sum.  Fully differentiable, no
    hand-off between classification and regression.

    **GRU temporal smoothing:**  Single-layer GRU between TCN and all
    output heads.  Integrates evidence across frames, eliminating
    frame-to-frame f0/decay/damping jitter.

    Output channel layout::

        0 : f0 in Hz   (pre-activated via soft-argmax)
        1 : decay       (raw logit)
        2 : a1          (raw logit)
    """

    def __init__(
        self,
        num_outputs: int = 3,
        cqt_kwargs: dict | None = None,
        tcn_channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.1,
        n_f0_bins: int = 128,
        f0_min_hz: float = 32.0,
        f0_max_hz: float = 2000.0,
        gru_hidden: int = 256,
        gru_layers: int = 1,
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.n_f0_bins = n_f0_bins

        self.frontend = CQTFrontend(**(cqt_kwargs or {}))

        in_ch = self.frontend.out_channels
        self.tcn = _build_tcn(in_ch, tcn_channels, num_blocks,
                              kernel_size, dilation_base, dropout)

        self.gru = nn.GRU(
            input_size=tcn_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.gru_norm = nn.LayerNorm(gru_hidden)

        self.f0_head = nn.Linear(gru_hidden, n_f0_bins)

        bin_centres = f0_min_hz * (
            (f0_max_hz / f0_min_hz)
            ** (torch.arange(n_f0_bins, dtype=torch.float32)
                / (n_f0_bins - 1))
        )
        self.register_buffer("f0_bin_centres", bin_centres)

        self.decay_a1_head = nn.Linear(gru_hidden, 2)

        self.last_f0_probs: torch.Tensor | None = None

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [B, N] → [B, 3, T]."""
        x = self.frontend(wav)
        x = self.tcn(x)

        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = self.gru_norm(x)

        f0_logits = self.f0_head(x)
        f0_probs  = F.softmax(f0_logits, dim=-1)
        f0_hz = (f0_probs * self.f0_bin_centres).sum(dim=-1)

        self.last_f0_probs = f0_probs

        decay_a1 = self.decay_a1_head(x)

        out = torch.stack([
            f0_hz,
            decay_a1[..., 0],
            decay_a1[..., 1],
        ], dim=1)

        return out

class ExcitationEncoder(nn.Module):
    """LearnableFrontend → TCN → raw logits for excitation parameters
    (burst_gain, dynamic_level, pluck_position).
    """

    def __init__(
        self,
        num_outputs: int = 3,
        frontend_kwargs: dict | None = None,
        tcn_channels: int = 32,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.frontend = LearnableFrontend(**(frontend_kwargs or {}))

        in_ch = self.frontend.out_channels
        self.tcn = _build_tcn(in_ch, tcn_channels, num_blocks,
                              kernel_size, dilation_base, dropout)
        self.head = nn.Conv1d(tcn_channels, num_outputs, kernel_size=1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [B, N] → [B, num_outputs, T]."""
        x = self.frontend(wav)
        x = self.tcn(x)
        return self.head(x)

class KSEncoder(nn.Module):
    """Two-headed encoder for Karplus-Strong parameter estimation.

    Resonator pipeline  (CQT + GRU):  f0, decay, a1
    Excitation pipeline (learned frontend):  burst_gain, dynamic_level, pluck_position

    Output: ``{"resonator": [B,3,T], "excitation": [B,3,T], "f0_probs": [B,T,bins]}``

    **Important:** resonator channel 0 (f0) is pre-activated in Hz via
    soft-argmax.  The decoder must skip ``_sigmoid_range`` for it.

    Args:
        resonator_kwargs:  Forwarded to :class:`ResonatorEncoder`.
        excitation_kwargs: Forwarded to :class:`ExcitationEncoder`.
    """

    RESONATOR_PARAMS  = ("f0", "decay", "a1")
    EXCITATION_PARAMS = ("burst_gain", "dynamic_level", "pluck_position")

    def __init__(
        self,
        resonator_kwargs: dict | None = None,
        excitation_kwargs: dict | None = None,
    ):
        super().__init__()
        res_kw = dict(resonator_kwargs or {})
        exc_kw = dict(excitation_kwargs or {})

        res_kw.setdefault("num_outputs", len(self.RESONATOR_PARAMS))
        exc_kw.setdefault("num_outputs", len(self.EXCITATION_PARAMS))

        self.resonator  = ResonatorEncoder(**res_kw)
        self.excitation = ExcitationEncoder(**exc_kw)

        self.num_params = (self.resonator.num_outputs
                           + self.excitation.num_outputs)

    @property
    def last_f0_probs(self) -> torch.Tensor | None:
        """[B, T, n_f0_bins] — soft-argmax distribution from last forward."""
        return self.resonator.last_f0_probs

    def forward(
        self, wav: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        res_raw = self.resonator(wav)
        exc_raw = self.excitation(wav)

        T = min(res_raw.shape[-1], exc_raw.shape[-1])
        res_raw = res_raw[..., :T]
        exc_raw = exc_raw[..., :T]

        return {
            "resonator":  res_raw,
            "excitation": exc_raw,
            "f0_probs":   self.resonator.last_f0_probs[:, :T, :],
        }


# ═════════════════════════════════════════════════════════════════════════════
# H+N Encoder  (kept for HarmonicsNoiseDecoder compatibility)
# ═════════════════════════════════════════════════════════════════════════════

class HpNEncoder(nn.Module):
    """LearnableFrontend → TCN → [B, num_outputs, T] raw logits.

    Used only by ``HarmonicsNoiseDecoder``.
    """

    def __init__(
        self,
        num_outputs: int = 165,
        tcn_channels: int = 64,
        num_blocks: int = 5,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.frontend = LearnableFrontend()

        in_ch = self.frontend.out_channels
        self.tcn = _build_tcn(in_ch, tcn_channels, num_blocks,
                              kernel_size, dilation_base, dropout)
        self.head = nn.Conv1d(tcn_channels, num_outputs, kernel_size=1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.frontend(wav)
        x = self.tcn(x)
        return self.head(x)