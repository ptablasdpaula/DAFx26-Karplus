from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torchaudio

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
    """Strided conv stack: raw audio [B, 1, N] → features [B, C, T_frames]."""
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
        return self.net(wav.unsqueeze(1))

# ── NEW: Mel Spectrogram Frontend ──
class MelSpectrogramFrontend(nn.Module):
    """Mel Spectrogram frontend for pitch/resonance features.
    
    Provides strict temporal precision across all frequency bands, preventing
    low-frequency smearing found in CQTs.
    """
    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        f_min: float = 32.0,
        f_max: float = 8000.0,
    ):
        super().__init__()
        self.fs = fs
        self.hop_length = hop_length
        self.out_channels = n_mels

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=fs,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            center=True,
            power=1.0,  # Use magnitude
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: [B, N] → [B, n_mels, T_frames]  (log-magnitude Mel)."""
        mel_spec = self.mel_transform(wav)
        # Apply logarithmic compression to prevent high-amplitude dominance
        return torch.log1p(mel_spec)

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class DETRDecoderLayer(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        ff_hidden = d_model * ff_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q_norm1 = self.norm1(queries)
        q2, _ = self.self_attn(query=q_norm1, key=q_norm1, value=q_norm1)
        queries = queries + q2

        q_norm2 = self.norm2(queries)
        q2, _ = self.cross_attn(query=q_norm2, key=memory, value=memory)
        queries = queries + q2

        q_norm3 = self.norm3(queries)
        queries = queries + self.ffn(q_norm3)
        return queries

class KSEventEncoder(nn.Module):
    def __init__(
            self,
            max_events: int = 40,
            mel_kwargs: dict | None = None,  # <-- Renamed
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

        self.mel = MelSpectrogramFrontend(**(mel_kwargs or {}))
        self.learnable = LearnableFrontend(**(frontend_kwargs or {}))

        in_ch = self.mel.out_channels + self.learnable.out_channels

        self.proj_in = nn.Conv1d(in_ch, d_model, kernel_size=1)
        self.tcn = _build_tcn(d_model, d_model, num_blocks,
                              kernel_size, dilation_base, dropout,
                              causal=False)

        self.pos_encoder = PositionalEncoding1D(d_model)
        self.memory_norm = nn.LayerNorm(d_model)

        self.query_embed = nn.Parameter(
            torch.randn(1, max_events, d_model) / math.sqrt(d_model)
        )

        self.decoder_blocks = nn.ModuleList([
            DETRDecoderLayer(d_model, cross_attn_heads, ff_mult=2, dropout=dropout)
            for _ in range(cross_attn_layers)
        ])

        prior_prob = 0.55
        bias_value = -math.log((1.0 - prior_prob) / prior_prob)
        self.head_exists = nn.Linear(d_model, 1)
        self.head_exists.bias.data.fill_(bias_value)
        nn.init.constant_(self.head_exists.weight, 0.0)

        self.head_time = nn.Linear(d_model, 1)
        self.head_f0 = nn.Linear(d_model, n_f0_bins)
        bin_centres = f0_min_hz * (
                (f0_max_hz / f0_min_hz)
                ** (torch.arange(n_f0_bins, dtype=torch.float32) / (n_f0_bins - 1))
        )
        self.register_buffer("f0_bin_centres", bin_centres)
        self.head_params = nn.Linear(d_model, 4)  # decay, a1, pluck, dyn
        self.head_global_gain = nn.Linear(d_model, 1)  # gain

    def forward(self, wav: torch.Tensor) -> dict[str, torch.Tensor]:
        B = wav.shape[0]

        # ── FIXED: Compute and concat Mel Features ──
        mel_feat = self.mel(wav)  # [B, C_mel, T]
        lrn_feat = self.learnable(wav)  # [B, C_lrn, T]

        T = min(mel_feat.shape[-1], lrn_feat.shape[-1])
        x = torch.cat([mel_feat[..., :T], lrn_feat[..., :T]], dim=1)  # [B, C, T]

        x = self.proj_in(x) 
        x = self.tcn(x)

        memory = x.permute(0, 2, 1)  
        memory = self.pos_encoder(memory)
        memory = self.memory_norm(memory)

        global_gain_logits = self.head_global_gain(memory.mean(dim=1))

        queries = self.query_embed.expand(B, -1, -1)

        for block in self.decoder_blocks:
            queries = block(queries, memory) 

        exists_logits = self.head_exists(queries)
        time_logits = self.head_time(queries)
        param_logits = self.head_params(queries)

        f0_logits = self.head_f0(queries)
        f0_probs = F.softmax(f0_logits, dim=-1)

        log_centres = torch.log(self.f0_bin_centres)
        expected_log_f0 = (f0_probs * log_centres).sum(dim=-1, keepdim=True)
        f0_hz = torch.exp(expected_log_f0)

        return {
            "exists": exists_logits,
            "time": time_logits,
            "f0_probs": f0_probs,
            "f0_hz": f0_hz,
            "params": param_logits,
            "global_gain": global_gain_logits,
        }

class HpNEncoder(nn.Module):
    """Replica of DDSP z-encoder: MFCC -> Normalization -> GRU -> Dense."""

    def __init__(
            self,
            num_outputs: int = 16,
            fs: int = 16000,
    ):
        super().__init__()
        self.num_outputs = num_outputs

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=fs,
            n_mfcc=30,
            melkwargs={
                "n_fft": 1024,
                "hop_length": 256,
                "n_mels": 128,
                "f_min": 20.0,
                "f_max": 8000.0,
                "center": True,
            }
        )

        self.norm = nn.InstanceNorm1d(30, affine=True)
        self.gru = nn.GRU(input_size=30, hidden_size=512, batch_first=True)
        self.head = nn.Linear(512, num_outputs)

    def forward(
            self,
            wav: torch.Tensor # [B, T_audio]
    ) -> torch.Tensor:
        mfccs = self.mfcc_transform(wav) # [B, 30, T_frames + 1]
        mfccs = mfccs[..., :-1] # [B, 30, T_frames]
        x = self.norm(mfccs)
        x = x.permute(0, 2, 1) # [B, T_frames, 30]
        x, _ = self.gru(x)
        z = self.head(x) # [B, T_frames, num_outputs]
        return z.permute(0, 2, 1) # [B, num_outputs, T_frames]

class HpN_Enhanced_Encoder(nn.Module):
    """
    Enhanced DDSP z-encoder using the same frontends as the KSEventEncoder.
    Matches the TCN processing power while maintaining frame-based H+N outputs.
    """
    def __init__(
            self,
            num_outputs: int = 16, # z_dim for H+N decoder
            fs: int = 16000,
            mel_kwargs: dict | None = None,
            frontend_kwargs: dict | None = None,
            d_model: int = 64,
            num_blocks: int = 7,
            kernel_size: int = 3,
            dilation_base: int = 2,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.num_outputs = num_outputs

        self.mel = MelSpectrogramFrontend(**(mel_kwargs or {}))
        self.learnable = LearnableFrontend(**(frontend_kwargs or {}))

        in_ch = self.mel.out_channels + self.learnable.out_channels

        self.proj_in = nn.Conv1d(in_ch, d_model, kernel_size=1)
        self.tcn = _build_tcn(
            in_ch=d_model,
            tcn_channels=d_model,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dilation_base=dilation_base,
            dropout=dropout,
            causal=False
        )

        self.head = nn.Linear(d_model, num_outputs)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        mel_feat = self.mel(wav)  # [B, C_mel, T]
        lrn_feat = self.learnable(wav)  # [B, C_lrn, T]

        T = min(mel_feat.shape[-1], lrn_feat.shape[-1])
        x = torch.cat([mel_feat[..., :T], lrn_feat[..., :T]], dim=1)

        x = self.proj_in(x)
        x = self.tcn(x)

        z = self.head(x.permute(0, 2, 1))
        return z.permute(0, 2, 1) # [B, num_outputs, T]