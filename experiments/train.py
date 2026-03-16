"""Hydra entry point for sound-matching experiments.

Usage
-----
# Synthetic-only, param losses, time-domain KS, no detectors
python experiments/train.py model=ks_timedomain detector=none \\
    training=param_only data=synthetic_only experiment=synth_eval

# Synthetic + Nsynth, combined, freq-sampling, external detectors
python experiments/train.py model=ks_freqsampling detector=external \\
    training=combined data=synthetic_and_nsynth experiment=ood_eval

# Multirun sweep
python experiments/train.py --multirun \\
    model=ks_timedomain,ks_freqsampling \\
    training=param_only,spectral_only,combined \\
    data=synthetic_only experiment=synth_eval
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import WandbLogger

from src.synths.synth import Synth, SynthConfig
from src.synths.ddsp import Implementation
from src.decoder import KSDecoder
from src.model import SoundMatchingModel
from src.data.data_module import SoundMatchingDataModule
from experiments.sound_matching_experiment import SoundMatchingExperiment

from paths import EXPERIMENTS_DIR

import torch


IMPL_MAP = {
    "time_domain": Implementation.TIME_DOMAIN,
    "frequency_sampling": Implementation.FREQUENCY_SAMPLING,
}


def _build_decoder(cfg: DictConfig):
    if cfg.model.decoder == "ks":
        synth = Synth(SynthConfig(
            num_samples=cfg.num_audio_samples,
            fs=cfg.fs,
            implementation=IMPL_MAP[cfg.model.implementation],
            lagrange_order=cfg.model.synth.lagrange_order,
            n_fft=cfg.model.synth.n_fft,
        ))
        return KSDecoder(
            synth=synth,
            use_external_detectors=cfg.detector.use_external_detectors,
        )

    if cfg.model.decoder == "harmonics_noise":
        from src.decoder import HarmonicsNoiseDecoder
        return HarmonicsNoiseDecoder(
            fs=cfg.fs,
            num_samples=cfg.num_audio_samples,
            n_harmonics=cfg.model.get("n_harmonics", 100),
            n_noise_bands=cfg.model.get("n_noise_bands", 65),
            use_external_detectors=cfg.detector.use_external_detectors,
            z_dim=cfg.model.get("z_dim", 16),
            hidden_dim=cfg.model.get("hidden_dim", 512),
        )

    raise ValueError(f"Unknown decoder: {cfg.model.decoder}")


def _build_model(cfg: DictConfig) -> SoundMatchingModel:
    return SoundMatchingModel(
        decoder=_build_decoder(cfg),
        encoder_kwargs=OmegaConf.to_container(cfg.model.encoder, resolve=True),
    )


def _build_datamodule(cfg: DictConfig) -> SoundMatchingDataModule:
    nsynth_root = None
    nsynth_split_train = "training"
    nsynth_split_val = "test"
    if cfg.data.has_ood:
        nsynth_root = cfg.data.nsynth.root
        nsynth_split_train = cfg.data.nsynth.split_train
        nsynth_split_val = cfg.data.nsynth.split_val

    syn_cfg = None
    if cfg.data.has_synthetic:
        syn_cfg = OmegaConf.to_container(cfg.data.synthetic, resolve=True)

    return SoundMatchingDataModule(
        has_synthetic=cfg.data.has_synthetic,
        has_ood=cfg.data.has_ood,
        fs=cfg.fs,
        num_audio_samples=cfg.num_audio_samples,
        duration_s=cfg.duration_s,
        batch_size=cfg.experiment.batch_size,
        num_workers=cfg.experiment.num_workers,
        synthetic_cfg=syn_cfg,
        nsynth_root=nsynth_root,
        nsynth_split_train=nsynth_split_train,
        nsynth_split_val=nsynth_split_val,
        num_frames=cfg.num_frames,
    )


def _build_experiment(cfg: DictConfig, model: SoundMatchingModel) -> SoundMatchingExperiment:
    return SoundMatchingExperiment(
        model=model,
        objective=cfg.training.objective,
        w_mss=cfg.training.w_mss,
        w_sot=cfg.training.w_sot,
        w_param=cfg.training.w_param,
        event_loss_weights=OmegaConf.to_container(cfg.training.event_loss_weights, resolve=True),
        eval_synthetic_metrics=cfg.experiment.eval_synthetic_metrics,
        eval_ood_metrics=cfg.experiment.eval_ood_metrics,
        lr=cfg.lr,
        scheduler_patience=cfg.experiment.scheduler_patience,
        scheduler_factor=cfg.experiment.scheduler_factor,
        min_lr=cfg.experiment.min_lr,
        fs=cfg.fs,
        duration_s=cfg.duration_s,
        log_val_audio=cfg.experiment.log_val_audio,
        num_val_audio_examples=cfg.experiment.num_val_audio_examples,
    )

def _checkpoint_tag(cfg: DictConfig) -> str:
    data_tag = "Nsynth" if cfg.data.has_ood else "Synth"
    det_tag = "Det" if cfg.detector.use_external_detectors else "Free"

    impl = cfg.model.get("implementation", None)
    if impl == "time_domain":
        impl_tag = "Time"
    elif impl == "frequency_sampling":
        impl_tag = "Freq"
    else:
        enc_type = cfg.model.get("encoder", {}).get("type", "baseline")
        impl_tag = "HpNEnh" if enc_type == "enhanced" else "HpN"

    obj = cfg.training.objective
    obj_tag = {"param_only": "Super", "spectral_only": "Spec", "combined": "Comb"}[obj]

    return f"{data_tag}_{det_tag}_{impl_tag}_{obj_tag}"


class SyntheticEpochCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module) -> None:
        dm = trainer.datamodule
        if dm is not None and getattr(dm.hparams, "has_synthetic", False):
            dm.set_train_epoch(trainer.current_epoch)

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # ── Enforce Baseline Constraints ──
    errors = []

    # 1. Harmonics + Noise Constraints
    if cfg.model.decoder == "harmonics_noise":
        if not cfg.detector.use_external_detectors:
            errors.append("detector=external (H+N requires f0 + loudness detectors)")
        if cfg.training.objective != "spectral_only":
            errors.append("training=spectral_only (H+N has no ground-truth params)")
        if cfg.data.has_synthetic:
            errors.append("data=nsynth_only (H+N cannot use synthetic KS data)")
        if not cfg.data.has_ood:
            errors.append("data=nsynth_only (H+N requires NSynth OOD data)")
        if cfg.experiment.eval_synthetic_metrics:
            errors.append("experiment=ood_eval (H+N has no param-level metrics)")

    # 2. Karplus-Strong Constraints
    elif cfg.model.decoder == "ks":
        if cfg.detector.use_external_detectors:
            if cfg.training.objective != "spectral_only":
                errors.append(
                    "training=spectral_only (When using external detectors, "
                    "timing/f0 are overridden, making parameter loss disconnected/redundant)"
                )
        else:
            if cfg.training.objective == "spectral_only":
                errors.append(
                    "training=combined OR training=param_only (End-to-End detection "
                    "requires EventSetLoss to learn discrete event existence/timing)"
                )

    if errors:
        raise ValueError(
            "Invalid configuration combination detected:\n  - "
            + "\n  - ".join(errors)
            + "\n\nPlease adjust your Hydra overrides."
        )

    pl.seed_everything(cfg.seed, workers=True)

    model = _build_model(cfg)
    experiment = _build_experiment(cfg, model)
    datamodule = _build_datamodule(cfg)

    n_total = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    print(f"SoundMatchingModel: {n_total:,} params (encoder: {n_enc:,})")
    print(f"Decoder: {model.decoder.__class__.__name__} "
          f"({model.decoder.num_params} outputs)")
    if hasattr(model.encoder, 'resonator'):
        n_res = sum(p.numel() for p in model.encoder.resonator.parameters())
        n_exc = sum(p.numel() for p in model.encoder.excitation.parameters())
        print(f"  Resonator encoder: {n_res:,} params")
        print(f"  Excitation encoder: {n_exc:,} params")

    tag = _checkpoint_tag(cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{tag}_{timestamp}"

    logger = WandbLogger(
        project="DAFx26-Karplus",
        name=run_name,
        save_dir=str(EXPERIMENTS_DIR),
        log_model=False,
    )
    logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    monitor = "val_synth/loss" if cfg.experiment.eval_synthetic_metrics else "val_ood/loss"

    ckpt_dir = EXPERIMENTS_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_best = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"{run_name}_best",
        save_top_k=1,
        monitor=monitor,
        mode="min",
    )
    ckpt_last = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"{run_name}_last",
        save_top_k=1,
        save_last=False,      # we control the name ourselves
        every_n_epochs=1,
        monitor=None,         # always save (most recent)
    )

    OmegaConf.save(cfg, str(ckpt_dir / f"{run_name}_config.yaml"))
    print(f"Run name: {run_name}")

    early_stop = EarlyStopping(
        monitor=monitor,
        patience=cfg.experiment.patience,
        min_delta=cfg.experiment.min_delta,
        mode="min",
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.experiment.max_epochs,
        gradient_clip_val=cfg.experiment.gradient_clip_val,
        logger=logger,
        accelerator=accelerator,
        val_check_interval=cfg.experiment.val_check_interval,
        callbacks=[ckpt_best, ckpt_last, early_stop, SyntheticEpochCallback()],
        enable_progress_bar=True,
    )
    trainer.fit(experiment, datamodule=datamodule)

    for suffix in ["best.ckpt", "last.ckpt", "config.yaml"]:
        src = ckpt_dir / f"{run_name}_{suffix}"
        dst = ckpt_dir / f"{tag}_{suffix}"
        if src.exists():
            dst.unlink(missing_ok=True)
            dst.symlink_to(src.name)
            print(f"  {dst.name} → {src.name}")


if __name__ == "__main__":
    main()