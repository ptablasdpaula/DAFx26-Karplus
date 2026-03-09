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


class NonCausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True):
        padding = (dilation * (kernel_size - 1)) // 2
        super().__init__(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias,
        )


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int,
                 kernel_size: int, dilation: int = 1,
                 dropout: float = 0.1, last_block: bool = False,
                 causal: bool = False):
        super().__init__()
        Conv = CausalConv1d if causal else NonCausalConv1d
        block = [
            weight_norm(Conv(in_ch, hidden_ch, kernel_size, dilation=dilation)),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(Conv(hidden_ch, out_ch, kernel_size, dilation=dilation)),
        ]
        if not last_block:
            block.extend([nn.ReLU(), nn.Dropout(dropout)])
        self.block = nn.Sequential(*block)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.residual(x)


def _build_tcn(in_ch: int, tcn_channels: int, num_blocks: int,
               kernel_size: int, dilation_base: int,
               dropout: float, causal: bool = False) -> nn.Sequential:
    blocks = []
    ch = in_ch
    for i in range(num_blocks):
        dilation = dilation_base ** i
        blocks.append(TCNBlock(ch, tcn_channels, tcn_channels,
                               kernel_size, dilation, dropout,
                               last_block=(i == num_blocks - 1),
                               causal=causal))
        ch = tcn_channels
    return nn.Sequential(*blocks)

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

class CrossAttentionDecoderLayer(nn.Module):
    """A lean, standard softmax cross-attention layer for event detection.

    The learnable event queries look at the dense frame memory produced by
    the TCN backbone to lock onto specific, localized audio features.
    No self-attention is used to save parameters.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        ff_hidden = d_model * ff_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        queries: [B, max_events, d_model]
        memory:  [B, T_frames, d_model]
        """
        attn_out, _ = self.cross_attn(query=queries, key=memory, value=memory)
        queries = self.norm1(queries + attn_out)
        queries = self.norm2(queries + self.ffn(queries))
        return queries


class KSEventEncoder(nn.Module):
    """Unified DETR-style Encoder for Karplus-Strong Event Packets.

    Dual Frontend (CQT + Learnable) → Non-Causal TCN → Cross-Attention Decoder → MLP Heads.
    """

    def __init__(
            self,
            max_events: int = 40,
            cqt_kwargs: dict | None = None,
            frontend_kwargs: dict | None = None,
            d_model: int = 64,
            num_blocks: int = 6,
            kernel_size: int = 3,
            dilation_base: int = 2,
            dropout: float = 0.1,
            cross_attn_heads: int = 4,
            cross_attn_layers: int = 2,
            n_f0_bins: int = 128,
            f0_min_hz: float = 32.0,
            f0_max_hz: float = 2000.0,
    ):
        super().__init__()
        self.max_events = max_events
        self.n_f0_bins = n_f0_bins

        self.cqt = CQTFrontend(**(cqt_kwargs or {}))
        self.learnable = LearnableFrontend(**(frontend_kwargs or {}))

        in_ch = self.cqt.out_channels + self.learnable.out_channels

        self.proj_in = nn.Conv1d(in_ch, d_model, kernel_size=1)
        self.tcn = _build_tcn(d_model, d_model, num_blocks,
                              kernel_size, dilation_base, dropout,
                              causal=False)

        self.query_embed = nn.Parameter(torch.randn(1, max_events, d_model))

        self.decoder_blocks = nn.ModuleList([
            CrossAttentionDecoderLayer(d_model, cross_attn_heads, ff_mult=2, dropout=dropout)
            for _ in range(cross_attn_layers)
        ])

        # FOCAL LOSS
        prior_prob = 0.01
        bias_value = -math.log((1.0 - prior_prob) / prior_prob)
        self.head_exists = nn.Linear(d_model, 1)
        self.head_exists.bias.data.fill_(bias_value)

        self.head_time = nn.Linear(d_model, 1)
        self.head_f0 = nn.Linear(d_model, n_f0_bins)
        bin_centres = f0_min_hz * (
                (f0_max_hz / f0_min_hz)
                ** (torch.arange(n_f0_bins, dtype=torch.float32) / (n_f0_bins - 1))
        )
        self.register_buffer("f0_bin_centres", bin_centres)
        self.head_params = nn.Linear(d_model, 5)

    def forward(self, wav: torch.Tensor) -> dict[str, torch.Tensor]:
        """wav: [B, N] → Dict of event parameters."""
        B = wav.shape[0]

        # ── Frontends ──
        cqt_feat = self.cqt(wav)  # [B, C_cqt, T]
        lrn_feat = self.learnable(wav)  # [B, C_lrn, T]

        T = min(cqt_feat.shape[-1], lrn_feat.shape[-1])
        x = torch.cat([cqt_feat[..., :T], lrn_feat[..., :T]], dim=1)  # [B, C, T]

        # ── Backbone ──
        x = self.proj_in(x)  # [B, d_model, T]
        x = self.tcn(x)  # [B, d_model, T]

        # Prepare memory for transformer (needs [B, T, d_model])
        memory = x.permute(0, 2, 1)

        # ── Cross-Attention Decoder ──
        queries = self.query_embed.expand(B, -1, -1)  # [B, max_events, d_model]

        for block in self.decoder_blocks:
            queries = block(queries, memory)  # [B, max_events, d_model]

        # ── Heads ──
        exists_logits = self.head_exists(queries)  # [B, 40, 1]
        time_logits = self.head_time(queries)  # [B, 40, 1]
        param_logits = self.head_params(queries)  # [B, 40, 5]

        f0_logits = self.head_f0(queries)  # [B, 40, 128]
        f0_probs = F.softmax(f0_logits, dim=-1)
        f0_hz = (f0_probs * self.f0_bin_centres).sum(dim=-1, keepdim=True)  # [B, 40, 1]

        return {
            "exists": exists_logits,
            "time": time_logits,
            "f0_probs": f0_probs,
            "f0_hz": f0_hz,
            "params": param_logits
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