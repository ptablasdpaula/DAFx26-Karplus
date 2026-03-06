from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

def sigmoid_range(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Sigmoid squashed into [lo, hi]."""
    return lo + (hi - lo) * torch.sigmoid(x)

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

class LearnableFrontend(nn.Module):
    """Strided conv stack: raw audio [B, 1, N] → features [B, C, T_frames]."""

    DEFAULT_CHANNELS = [32, 64, 64, 64]
    DEFAULT_STRIDES  = [4, 4, 4, 4]       # total stride = 256 = SYNTH_HOP
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
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ])
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: [B, N] → [B, C, T_frames]"""
        return self.net(wav.unsqueeze(1))

class Encoder(nn.Module):
    """
    Raw audio → [B, num_outputs, T] logits.

    Args:
        num_outputs:    Number of output channels (= number of parameters).
        tcn_channels:   Hidden width of the TCN.
        num_blocks:     Number of TCN blocks.
        kernel_size:    TCN kernel size.
        dilation_base:  Exponential dilation base.
        dropout:        Dropout rate in TCN blocks.
    """

    def __init__(
        self,
        num_outputs: int = 6,
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
        blocks = []
        for i in range(num_blocks):
            dilation = dilation_base ** i
            blocks.append(TCNBlock(in_ch, tcn_channels, tcn_channels,
                                   kernel_size, dilation, dropout))
            in_ch = tcn_channels
        self.tcn = nn.Sequential(*blocks)

        self.head = nn.Conv1d(tcn_channels, num_outputs, kernel_size=1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: [B, num_samples]
        returns: [B, num_outputs, T] raw logits (no activations).
        """
        x = self.frontend(wav)      # [B, C, T]
        x = self.tcn(x)             # [B, tcn_channels, T]
        return self.head(x)         # [B, num_outputs, T]