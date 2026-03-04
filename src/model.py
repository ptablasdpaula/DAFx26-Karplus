from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

import pytorch_lightning as pl

from src.synths.param_registry import (
    PARAM_NAMES,
    F0_MIN_HZ,
    F0_MAX_HZ,
    PLUCK_POSITION_MIN,
    PLUCK_POSITION_MAX,
    DYNAMIC_LEVEL_MIN,
    DYNAMIC_LEVEL_MAX,
    DAMPING_MIN,
    DAMPING_MAX,
    DECAY_MIN,
    DECAY_MAX,
)
from src.synths.synth import Synth, SynthConfig, SynthOutput
from src.losses import PLoss, MultiScaleSpectralLoss, SOT2048Loss


def sigmoid_range(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
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
    """Strided conv stack: raw audio [B, 1, 64000] → features [B, C, 250]."""

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

    No activations applied — the downstream model (DiffKSModel, etc.)
    is responsible for mapping logits to constrained parameter ranges.

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

#TODO: ok, we have this model - but what do we do when we want different training strategies etc...
class DiffKSModel(pl.LightningModule):
    """
    Behaviour:
        training:   synthesises via Synth.forward()       (differentiable)
        inference:  synthesises via Synth.oracle_synth()   (numpy, more accurate)
    returns ``(audio, params)``.

    Args:
        synth_config:     SynthConfig for the KS synthesiser.
        encoder_kwargs:   Forwarded to Encoder.__init__().
        lr:               Learning rate for Adam.
        w_mss:          Weight for MSS.
        w_sot:          Weight for SOT.
        w_param:          Weight for parameter loss (PLoss).
        param_weights:    Per-parameter weights forwarded to PLoss.
    """

    def __init__(
        self,
        synth_config: SynthConfig,
        encoder_kwargs: dict | None = None,
        lr: float = 1e-3,
        w_mss: float = 1.0,
        w_sot: float = 1.0,
        w_param: float = 1.0,
        param_weights: dict[str, float] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["synth_config"])

        self.lr = lr
        self.w_mss = w_mss
        self.w_sot = w_sot
        self.w_param = w_param

        # ── Encoder (learnable) ──
        self.encoder = Encoder(
            num_outputs=len(PARAM_NAMES),
            **(encoder_kwargs or {}),
        )

        # ── Synth (not learnable, but needs to live on the right device) ──
        self.synth = Synth(synth_config)

        # ── Losses ──
        self.ploss = PLoss(
            fs=synth_config.fs,
            weights=param_weights or {},
        )
        self.mss = MultiScaleSpectralLoss()
        self.sot = SOT2048Loss()

        # Channel index → param name (follows PARAM_NAMES order)
        self._idx = {name: i for i, name in enumerate(PARAM_NAMES)}

    def activate(self, raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply KS-specific activations to raw logits.

        raw: [B, P, T] → dict of [B, T] constrained parameters.
        """
        f0         = sigmoid_range(raw[:, self._idx["f0"], :],
                                   F0_MIN_HZ, F0_MAX_HZ)
        burst_gain = F.relu(raw[:, self._idx["burst_gain"], :])
        pluck_pos  = sigmoid_range(raw[:, self._idx["pluck_position"], :],
                                   PLUCK_POSITION_MIN, PLUCK_POSITION_MAX)
        dyn_level  = sigmoid_range(raw[:, self._idx["dynamic_level"], :],
                                   DYNAMIC_LEVEL_MIN + 1e-3, DYNAMIC_LEVEL_MAX)
        a1         = sigmoid_range(raw[:, self._idx["a1"], :],
                                   DAMPING_MIN, DAMPING_MAX)
        decay      = sigmoid_range(raw[:, self._idx["decay"], :],
                                   DECAY_MIN, DECAY_MAX)

        return {
            "f0":             f0,
            "burst_gain":     burst_gain,
            "pluck_position": pluck_pos,
            "dynamic_level":  dyn_level,
            "a1":             a1,
            "decay":          decay,
        }

    def forward(self, wav: torch.Tensor) -> SynthOutput:
        """
        wav: [B, num_samples]
        returns: (audio [B, num_samples], params {name: [B, T]})
        """
        raw = self.encoder(wav)           # [B, P, T]
        params = self.activate(raw)       # dict of [B, T]
        if self.training:
            audio, params = self.synth(params)
        else:
            audio, params = self.synth.oracle_synth(params)
        return audio, params

    def _shared_step(
        self,
        batch: Dict[str, Any],
        stage: str,
    ) -> torch.Tensor:
        """Shared logic for training and validation steps."""
        target_audio  = batch["audio"]        # [B, num_samples]
        target_params = batch["params"]       # {name: [B, T]}

        pred_audio, pred_params = self(target_audio)

        # Losses
        p_total, p_breakdown = self.ploss(pred_params, target_params)
        mss_loss = self.mss(pred_audio, target_audio)
        sot_loss = self.sot(pred_audio, target_audio)

        total = self.w_param * p_total + self.w_mss * mss_loss + self.w_sot * sot_loss

        # Logging
        self.log(f"{stage}/loss", total, prog_bar=True)
        self.log(f"{stage}/mss_loss", mss_loss, prog_bar=True)
        self.log(f"{stage}/sot_loss", sot_loss, prog_bar=True)
        self.log(f"{stage}/param_loss", p_total, prog_bar=True)
        for name, val in p_breakdown.items():
            self.log(f"{stage}/p/{name}", val)

        return total

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.encoder.parameters(), lr=self.lr)


if __name__ == "__main__":
    from src.data.synthetic_dataset import SyntheticDataset

    FS = 16000
    NUM_AUDIO_SAMPLES = 64000
    NUM_FRAMES = 250
    BATCH_SIZE = 4
    NUM_TRAIN = 16
    NUM_EPOCHS = 10
    LR = 1e-3

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. Generate dataset ──────────────────────────────────────────────
    print("Generating training data from SyntheticDataset...")
    ds = SyntheticDataset(
        num_samples_per_epoch=NUM_TRAIN,
        num_audio_samples=NUM_AUDIO_SAMPLES,
        num_frames=NUM_FRAMES,
        fs=FS,
        lti=False,
        random_seed=123,
    )
    samples = [s for s in ds]
    all_audio  = torch.stack([s['audio']  for s in samples]).to(DEVICE)
    all_params = {k: torch.stack([s['params'][k] for s in samples]).to(DEVICE)
                  for k in PARAM_NAMES}

    print(f"  audio:  {all_audio.shape}")
    print(f"  params: { {k: v.shape for k, v in all_params.items()} }")

    # ── 2. Build model ───────────────────────────────────────────────────
    synth_config = SynthConfig(num_samples=NUM_AUDIO_SAMPLES, fs=FS, device=DEVICE)

    model = DiffKSModel(
        synth_config=synth_config,
        encoder_kwargs=dict(dropout=0.0),
        lr=LR,
        w_mss=1.0,
        w_sot=1.0,
        w_param=1.0,
        param_weights={"f0": 2.0, "burst_gain": 5.0},
    ).to(DEVICE)

    n_total = sum(p.numel() for p in model.parameters())
    n_enc   = sum(p.numel() for p in model.encoder.parameters())
    print(f"\nDiffKSModel: {n_total:,} params (encoder: {n_enc:,})")

    # ── 3. Shape checks ─────────────────────────────────────────────────
    model.train()
    with torch.no_grad():
        train_audio, train_params = model(all_audio[:2])
    print(f"  train mode:  audio {train_audio.shape}, "
          f"params { {k: v.shape for k, v in train_params.items()} }")
    assert train_audio.shape == (2, NUM_AUDIO_SAMPLES)
    assert all(v.shape == (2, NUM_FRAMES) for v in train_params.values())

    model.eval()
    with torch.no_grad():
        eval_audio, eval_params = model(all_audio[:2])
    print(f"  eval mode:   audio {eval_audio.shape}, "
          f"params { {k: v.shape for k, v in eval_params.items()} }")
    assert eval_audio.shape == (2, NUM_AUDIO_SAMPLES)

    # ── 4. Manual training loop (no Trainer, to test on CPU quickly) ─────
    optimiser = model.configure_optimizers()
    n_batches = (NUM_TRAIN + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\n{'─' * 60}")
    print(f"Manual training: {NUM_EPOCHS} epochs × {n_batches} batches (B={BATCH_SIZE})")
    print(f"{'─' * 60}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        indices = torch.randperm(NUM_TRAIN)
        epoch_loss = 0.0

        for batch_start in range(0, NUM_TRAIN, BATCH_SIZE):
            idx = indices[batch_start:batch_start + BATCH_SIZE]

            batch = {
                "audio": all_audio[idx],
                "params": {k: v[idx] for k, v in all_params.items()},
            }

            loss = model.training_step(batch, batch_idx=0)

            optimiser.zero_grad()
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    print(f"Bad grad in {name}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            epoch_loss += loss.item()

        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}  loss={epoch_loss / n_batches:.4f}")

    # ── 5. Eval step ─────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Validation step (oracle synth)")
    print(f"{'─' * 60}")
    model.eval()
    with torch.no_grad():
        val_batch = {
            "audio": all_audio[:BATCH_SIZE],
            "params": {k: v[:BATCH_SIZE] for k, v in all_params.items()},
        }
        val_loss = model.validation_step(val_batch, batch_idx=0)
    print(f"  val loss: {val_loss.item():.4f}")

    # ── 6. Gradient flow check ───────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Gradient flow check")
    print(f"{'─' * 60}")
    model.train()
    batch = {
        "audio": all_audio[:BATCH_SIZE],
        "params": {k: v[:BATCH_SIZE] for k, v in all_params.items()},
    }
    loss = model.training_step(batch, batch_idx=0)
    optimiser.zero_grad()
    loss.backward()

    for name, param in model.named_parameters():
        gn = param.grad.norm().item() if param.grad is not None else 0.0
        if any(tag in name for tag in ["frontend.net.0.weight", "tcn.0.", "head."]):
            if "weight" in name:
                print(f"  {name:50s}  grad_norm={gn:.6f}")

    # ── 7. Lightning Trainer (if available) ──────────────────────────────
    print(f"\n{'─' * 60}")
    print("Lightning Trainer quick check")
    print(f"{'─' * 60}")

    from torch.utils.data import DataLoader, TensorDataset

    # Wrap into a DataLoader that yields the batch format DiffKSModel expects
    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, audio, params):
            self.audio = audio
            self.params = params
        def __len__(self):
            return self.audio.shape[0]
        def __getitem__(self, idx):
            return {
                "audio": self.audio[idx],
                "params": {k: v[idx] for k, v in self.params.items()},
            }

    train_dl = DataLoader(DictDataset(all_audio, all_params),
                          batch_size=BATCH_SIZE, shuffle=True)

    # Fresh model for the Trainer test
    model2 = DiffKSModel(
        synth_config=SynthConfig(num_samples=NUM_AUDIO_SAMPLES, fs=FS, device=DEVICE),
        encoder_kwargs=dict(dropout=0.0),
        lr=LR,
        param_weights={"f0": 2.0, "burst_gain": 5.0},
    )

    trainer = pl.Trainer(
        max_epochs=2,
        accelerator=DEVICE,
        enable_checkpointing=False,
        enable_progress_bar=True,
        logger=False,
        gradient_clip_val=1.0,
    )
    trainer.fit(model2, train_dataloaders=train_dl)

    print(f"\n✓ All smoke tests passed.")