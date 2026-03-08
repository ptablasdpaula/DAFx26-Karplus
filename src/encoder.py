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


# ═════════════════════════════════════════════════════════════════════════════
# Linear Attention
# ═════════════════════════════════════════════════════════════════════════════

def _elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """Feature map φ(x) = ELU(x) + 1.

    Ensures non-negative outputs (required for the linear attention
    kernel trick to approximate softmax attention).  ELU+1 is the
    standard choice from Katharopoulos et al. (2020) "Transformers
    are RNNs: Fast Autoregressive Transformers with Linear Attention".
    """
    return F.elu(x) + 1.0


class LinearAttention(nn.Module):
    """Multi-head linear attention layer.

    Replaces the softmax in standard attention with a feature map φ,
    enabling O(T·d²) computation instead of O(T²·d)::

        Standard:  Softmax(Q K^T / √d) V        ← T×T matrix
        Linear:    φ(Q) · (φ(K)^T · V)           ← d×d matrix

    For the resonator encoder (~250 frames, d=64), both are cheap.
    The advantage here is *not* speed but *smoothness*: linear attention
    produces globally-averaged representations that act as a natural
    low-pass filter on parameter trajectories — ideal for f0, decay,
    and damping which are physically smooth quantities.

    Args:
        d_model:   Input/output feature dimension.
        n_heads:   Number of attention heads (must divide d_model).
        dropout:   Dropout on attention output.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d_model] → [B, T, d_model].

        The linear attention kernel trick:

        1. Project to Q, K, V and reshape into heads
        2. Apply feature map φ to Q and K  (ensures non-negativity)
        3. Compute  KV = φ(K)^T · V       → [B, H, d, d]  (the "summary")
        4. Compute  Z  = φ(K)^T · 1       → [B, H, d, 1]  (normaliser)
        5. Output   O  = φ(Q) · KV / (φ(Q) · Z + ε)

        No T×T matrix is ever formed.
        """
        B, T, _ = x.shape
        H, d = self.n_heads, self.d_head

        # ── Project and split into heads ────────────────────────────────
        Q = self.W_q(x).view(B, T, H, d).permute(0, 2, 1, 3)  # [B, H, T, d]
        K = self.W_k(x).view(B, T, H, d).permute(0, 2, 1, 3)
        V = self.W_v(x).view(B, T, H, d).permute(0, 2, 1, 3)

        # ── Apply feature map φ ─────────────────────────────────────────
        Q = _elu_feature_map(Q)  # [B, H, T, d]
        K = _elu_feature_map(K)  # [B, H, T, d]

        # ── Linear attention via associative trick ──────────────────────
        # KV = φ(K)^T · V  →  [B, H, d, d]
        KV = torch.einsum("bhsd,bhsv->bhdv", K, V)

        # Z = φ(K)^T · 1   →  [B, H, d, 1]  (per-head normaliser)
        Z = K.sum(dim=2)  # [B, H, d]

        # Numerator:  φ(Q) · KV  →  [B, H, T, d]
        numerator = torch.einsum("bhtd,bhdv->bhtv", Q, KV)

        # Denominator:  φ(Q) · Z  →  [B, H, T, 1]
        denominator = torch.einsum("bhtd,bhd->bht", Q, Z).unsqueeze(-1)

        # Normalised output
        out = numerator / (denominator + 1e-6)  # [B, H, T, d]

        # ── Merge heads and project ─────────────────────────────────────
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, self.d_model)
        out = self.W_out(out)
        out = self.dropout(out)

        return out


class LinearAttentionBlock(nn.Module):
    """Pre-norm linear attention block with residual connection.

    ::

        x → LayerNorm → LinearAttention → + → LayerNorm → FFN → + → out
              ↑___________________________↑      ↑_____________↑

    The feed-forward network (FFN) is a standard expand-contract MLP
    that gives the model per-position nonlinear capacity after the
    global mixing done by attention.

    Args:
        d_model:    Feature dimension.
        n_heads:    Number of attention heads.
        ff_mult:    FFN expansion factor (hidden = d_model × ff_mult).
        dropout:    Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        ff_mult: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = LinearAttention(d_model, n_heads, dropout)

        self.norm2 = nn.LayerNorm(d_model)
        ff_hidden = d_model * ff_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d_model] → [B, T, d_model]."""
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ═════════════════════════════════════════════════════════════════════════════
# Resonator Encoder  (CQT → TCN → Linear Attention → soft-argmax f0)
# ═════════════════════════════════════════════════════════════════════════════

class ResonatorEncoder(nn.Module):
    """CQT → TCN → Linear Attention → soft-argmax f0 + raw logits (decay, a1).

    The TCN extracts local spectral features from the CQT.  Linear
    attention then integrates these features across the full sequence,
    producing globally-informed representations at every frame.

    This is physically motivated: once a string is plucked, f0, decay,
    and damping are global properties of the entire vibration — not
    local features of a single frame.  Linear attention acts as a smooth
    global summary, which is more stable than a GRU's sequential state
    (especially during quiet tail sections where the GRU hidden state
    can drift).

    **Soft-argmax f0 head** (Engel et al. 2020):  logits over log-spaced
    frequency bins → softmax → bin-weighted sum.  This is a separate
    softmax from the attention mechanism — it lives in the output head,
    not in the attention layer.

    Output channel layout::

        0 : f0 in Hz   (pre-activated via soft-argmax)
        1 : decay       (raw logit)
        2 : a1          (raw logit)

    Args:
        num_outputs:      Always 3 (f0, decay, a1).
        cqt_kwargs:       Forwarded to :class:`CQTFrontend`.
        tcn_channels:     Hidden width of the TCN.
        num_blocks:       Number of TCN blocks.
        kernel_size:      TCN kernel size.
        dilation_base:    Exponential dilation base.
        dropout:          Dropout rate.
        n_f0_bins:        Frequency bins for soft-argmax.
        f0_min_hz:        Lower bound of soft-argmax range.
        f0_max_hz:        Upper bound of soft-argmax range.
        attn_heads:       Number of linear attention heads.
        attn_layers:      Number of stacked linear attention blocks.
        attn_ff_mult:     FFN expansion factor in attention blocks.
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
        # ── Soft-argmax f0 ──
        n_f0_bins: int = 128,
        f0_min_hz: float = 32.0,
        f0_max_hz: float = 2000.0,
        # ── Linear Attention ──
        attn_heads: int = 4,
        attn_layers: int = 2,
        attn_ff_mult: int = 2,
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.n_f0_bins = n_f0_bins

        # ── Frontend ────────────────────────────────────────────────────
        self.frontend = CQTFrontend(**(cqt_kwargs or {}))

        # ── TCN (local feature extraction, non-causal) ─────────────────
        in_ch = self.frontend.out_channels
        self.tcn = _build_tcn(in_ch, tcn_channels, num_blocks,
                              kernel_size, dilation_base, dropout,
                              causal=False)

        # ── Linear Attention (global temporal integration) ──────────────
        # d_model = tcn_channels — no projection needed, TCN output feeds
        # directly into attention.
        self.attn_blocks = nn.Sequential(*[
            LinearAttentionBlock(
                d_model=tcn_channels,
                n_heads=attn_heads,
                ff_mult=attn_ff_mult,
                dropout=dropout,
            )
            for _ in range(attn_layers)
        ])
        self.attn_norm = nn.LayerNorm(tcn_channels)

        # ── f0 soft-argmax head ─────────────────────────────────────────
        self.f0_head = nn.Linear(tcn_channels, n_f0_bins)

        bin_centres = f0_min_hz * (
            (f0_max_hz / f0_min_hz)
            ** (torch.arange(n_f0_bins, dtype=torch.float32)
                / (n_f0_bins - 1))
        )
        self.register_buffer("f0_bin_centres", bin_centres)

        # ── Decay + a1 head ─────────────────────────────────────────────
        self.decay_a1_head = nn.Linear(tcn_channels, 2)

        # Stash for external access (loss / viz)
        self.last_f0_probs: torch.Tensor | None = None

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [B, N] → [B, 3, T].

        Channel layout::

            0 : f0 in Hz   (pre-activated, soft-argmax)
            1 : decay       (raw logit)
            2 : a1          (raw logit)
        """
        x = self.frontend(wav)                  # [B, n_bins, T]
        x = self.tcn(x)                         # [B, tcn_ch, T]

        # ── Linear Attention ────────────────────────────────────────────
        x = x.permute(0, 2, 1)                  # [B, T, tcn_ch]
        x = self.attn_blocks(x)                  # [B, T, tcn_ch]
        x = self.attn_norm(x)                    # [B, T, tcn_ch]

        # ── f0 via soft-argmax (this softmax is NOT the attention one) ──
        f0_logits = self.f0_head(x)             # [B, T, n_f0_bins]
        f0_probs  = F.softmax(f0_logits, dim=-1)
        f0_hz = (f0_probs * self.f0_bin_centres).sum(dim=-1)  # [B, T]

        self.last_f0_probs = f0_probs

        # ── Decay & a1 ─────────────────────────────────────────────────
        decay_a1 = self.decay_a1_head(x)        # [B, T, 2]

        # ── Pack as [B, 3, T] ──────────────────────────────────────────
        out = torch.stack([
            f0_hz,
            decay_a1[..., 0],
            decay_a1[..., 1],
        ], dim=1)

        return out


# ═════════════════════════════════════════════════════════════════════════════
# Excitation Encoder
# ═════════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════════
# KS Encoder
# ═════════════════════════════════════════════════════════════════════════════

class KSEncoder(nn.Module):
    """Two-headed encoder for Karplus-Strong parameter estimation.

    Resonator pipeline  (CQT + Linear Attention):  f0, decay, a1
    Excitation pipeline (learned frontend + TCN):   burst_gain, dynamic_level, pluck_position

    Output: ``{"resonator": [B,3,T], "excitation": [B,3,T], "f0_probs": [B,T,bins]}``

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