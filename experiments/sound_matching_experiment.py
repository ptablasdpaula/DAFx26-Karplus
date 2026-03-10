from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import pytorch_lightning as pl
import wandb

from src.model import SoundMatchingModel
from src.losses import EventSetLoss, MultiScaleSpectralLoss, SOT2048Loss
from src.metrics import compute_rms


class SoundMatchingExperiment(pl.LightningModule):
    """
    Args:
        model:               ``SoundMatchingModel`` instance.
        objective:           ``"param_only"`` | ``"spectral_only"`` | ``"combined"``.
        param_only_epochs:   (combined) param-only warm-up epochs.
        fadein_epochs:       (combined) spectral fade-in epochs.
        w_mss / w_sot / w_param: Loss weights.
        event_loss_weights:  Weights passed directly to EventSetLoss.
        eval_synthetic_metrics: Compute synthetic metrics.
        eval_ood_metrics:    Compute OOD metrics.
        lr / fs / duration_s: Optimiser / audio config.
    """

    def __init__(
            self,
            model: SoundMatchingModel,
            objective: str = "combined",
            param_only_epochs: int = 0,
            fadein_epochs: int = 0,
            w_mss: float = 1.0,
            w_sot: float = 1.0,
            w_param: float = 1.0,
            min_param_ratio: float = 0.5,
            event_loss_weights: dict[str, float] | None = None,
            eval_synthetic_metrics: bool = True,
            eval_ood_metrics: bool = True,
            lr: float = 1e-3,
            fs: int = 16000,
            duration_s: float = 4.0,
            log_val_audio: bool = True,
            num_val_audio_examples: int = 4
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.event_loss = EventSetLoss(**(event_loss_weights or {}))

        self.mss = MultiScaleSpectralLoss()
        self.sot = SOT2048Loss(sample_rate=fs)

        self._val_audio_examples = {"val_synth": [], "val_ood": []}

    # ── Curriculum ───────────────────────────────────────────────────────

    @property
    def spectral_weight(self) -> float:
        obj = self.hparams.objective
        if obj == "param_only":
            return 0.0
        if obj == "spectral_only":
            return 1.0
        epoch = self.current_epoch
        if epoch < self.hparams.param_only_epochs:
            return 0.0
        fade = epoch - self.hparams.param_only_epochs
        if self.hparams.fadein_epochs > 0 and fade < self.hparams.fadein_epochs:
            return fade / self.hparams.fadein_epochs
        return 1.0

    # ── Loss ─────────────────────────────────────────────────────────────

    def _compute_losses(
            self,
            pred_audio: torch.Tensor,
            pred_raw: dict[str, torch.Tensor],
            target_audio: torch.Tensor,
            target_params: dict[str, torch.Tensor] | None,
            is_synthetic: bool,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        device = pred_audio.device
        info: dict[str, Any] = {}
        total = torch.tensor(0.0, device=device)

        sw = self.spectral_weight
        if sw > 0:
            mss_loss = self.mss(pred_audio, target_audio)
            sot_loss = self.sot(pred_audio, target_audio)
            total = total + self.hparams.w_mss * sw * mss_loss
            total = total + self.hparams.w_sot * sw * sot_loss
            info["mss"] = mss_loss.detach()
            info["sot"] = sot_loss.detach()

        if is_synthetic and self.hparams.objective != "spectral_only":
            max_drop = 1.0 - self.hparams.min_param_ratio
            param_multiplier = 1.0 - (sw * max_drop) 
            
            p_total, p_breakdown = self.event_loss(pred_raw, target_params)
            
            total = total + (self.hparams.w_param * param_multiplier) * p_total
            
            info["param"] = p_total.detach()
            info["param_breakdown"] = p_breakdown
            info["param_multiplier"] = torch.tensor(param_multiplier, device=device)

        return total, info

    # ── Single sub-batch step ────────────────────────────────────────────

    def _step_on_batch(
            self,
            batch: dict[str, Any],
            stage: str,
            tag: str,
    ) -> torch.Tensor:
        target_audio = batch["audio"]
        target_params = batch.get("events") or batch.get("params")
        detected = batch.get("detected")
        is_synthetic = target_params is not None

        pred_raw = self.model.encoder(target_audio)
        pred_params = self.model.decoder.activate(pred_raw, detected)

        if self.training:
            pred_audio, _ = self.model.decoder.synthesise(pred_params)
        else:
            pred_audio, _ = self.model.decoder.oracle_synth(pred_params)

        total, info = self._compute_losses(
            pred_audio=pred_audio,
            pred_raw=pred_raw,
            target_audio=target_audio,
            target_params=target_params,
            is_synthetic=is_synthetic,
        )

        # logging
        self.log(f"{tag}/loss", total, prog_bar=True, add_dataloader_idx=False)
        for k, v in info.items():
            if k == "param_breakdown":
                for pname, pval in v.items():
                    self.log(f"{tag}/p/{pname}", pval, add_dataloader_idx=False)
            elif isinstance(v, torch.Tensor):
                self.log(f"{tag}/{k}", v, add_dataloader_idx=False)

        # val metrics
        if stage == "val":
            self._log_audio_metrics(pred_audio, target_audio, tag)
            self._collect_val_audio_examples(pred_audio, target_audio, tag)

        return total

    # ── Training ─────────────────────────────────────────────────────────

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        sw = self.spectral_weight
        self.log("train/spectral_weight", sw)
        
        # Log the dynamic multiplier
        max_drop = 1.0 - self.hparams.min_param_ratio
        self.log("train/param_multiplier", 1.0 - (sw * max_drop))

        if isinstance(batch, dict) and "audio" not in batch:
            total = torch.tensor(0.0, device=self.device)
            for key, sub_batch in batch.items():
                total = total + self._step_on_batch(sub_batch, "train", f"train/{key}")
            self.log("train/loss_total", total, prog_bar=True)
            return total

        return self._step_on_batch(batch, "train", "train")

    # ── Validation ───────────────────────────────────────────────────────

    def on_validation_epoch_start(self) -> None:
        self._val_audio_examples = {"val_synth": [], "val_ood": []}

    def validation_step(
            self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0,
    ) -> None:
        is_synthetic = "events" in batch or "params" in batch
        tag = "val_synth" if is_synthetic else "val_ood"
        self._step_on_batch(batch, "val", tag)

    def on_validation_epoch_end(self) -> None:
        if not self.hparams.log_val_audio or not self.logger:
            return

        for tag, examples in self._val_audio_examples.items():
            audios = []
            for i, arr in enumerate(examples):
                audios.append(wandb.Audio(
                    arr, sample_rate=self.hparams.fs, caption=f"{tag}_{i}"
                ))
            if audios:
                self.logger.experiment.log(
                    {f"{tag}/audio": audios, "global_step": self.global_step}
                )

    @torch.no_grad()
    def _collect_val_audio_examples(self, pred_audio: torch.Tensor, target_audio: torch.Tensor, tag: str) -> None:
        if not self.hparams.log_val_audio:
            return

        store = self._val_audio_examples[tag]
        remaining = self.hparams.num_val_audio_examples - len(store)
        if remaining <= 0:
            return

        n_to_take = min(remaining, pred_audio.shape[0])
        gap = int(0.1 * self.hparams.fs)

        for b in range(n_to_take):
            target = target_audio[b].detach().float().cpu()
            pred = pred_audio[b].detach().float().cpu()

            separator = torch.zeros(gap, dtype=target.dtype)
            concat = torch.cat([target, separator, pred], dim=0)

            peak = concat.abs().max()
            if peak > 1.0:
                concat = concat / peak

            store.append(concat.numpy())

    # ── Metrics ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _log_audio_metrics(self, pred_audio, target_audio, tag):
        B = pred_audio.shape[0]
        fs = self.hparams.fs
        msss, sots, rmss = [], [], []

        for b in range(B):
            p_batch = pred_audio[b:b + 1].detach()
            t_batch = target_audio[b:b + 1].detach()

            msss.append(self.mss(p_batch, t_batch).item())
            sots.append(self.sot(p_batch, t_batch).item())

            p = pred_audio[b].detach().cpu().numpy()
            t = target_audio[b].detach().cpu().numpy()

            rmss.append(compute_rms(t[np.newaxis, :], p[np.newaxis, :], sample_rate=fs))

        self.log(f"{tag}/mss_metric", np.mean(msss), add_dataloader_idx=False)
        self.log(f"{tag}/sot_metric", np.mean(sots), add_dataloader_idx=False)
        self.log(f"{tag}/rms_cos", np.mean(rmss), add_dataloader_idx=False)

    # ── Optimiser ────────────────────────────────────────────────────────

    def configure_optimizers(self):
            return torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.hparams.lr, 
                weight_decay=1e-4
            )