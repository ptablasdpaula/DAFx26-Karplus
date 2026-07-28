"""Hydra entry point for sound-matching experiments.

Usage
-----
# Synthetic P-Only baseline
python experiments/train.py model=ksa_oracle detector=end_to_end \\
    training=param_only data=synth

# Real P+Audio frequency-sampling KS model
python experiments/train.py model=ksa_freq detector=end_to_end \\
    training=combined data=mix

# Multirun sweep
python experiments/train.py --multirun \\
    model=ksa_time,ksa_freq \\
    training=combined,spectral_only \\
    data=synth,mix,real
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
from data.data_module import SoundMatchingDataModule
from experiments.sound_matching_experiment import SoundMatchingExperiment

from paths import EXPERIMENTS_DIR

import torch


IMPL_MAP = {
    "time_domain": Implementation.TIME_DOMAIN,
    "frequency_sampling": Implementation.FREQUENCY_SAMPLING,
    "oracle": Implementation.TIME_DOMAIN,
}


def _build_decoder(cfg: DictConfig):
    if cfg.model.decoder == "ks":
        synth = Synth(SynthConfig(
            num_samples=cfg.num_audio_samples,
            fs=cfg.fs,
            implementation=IMPL_MAP[cfg.model.implementation],
            lagrange_order=cfg.model.synth.lagrange_order,
            n_fft=cfg.model.synth.n_fft,
            hop_length=cfg.model.synth.hop_length,
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
        batch_size=cfg.batch_size,          # Flattended!
        num_workers=cfg.num_workers,        # Flattended!
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
        eval_synthetic_metrics=cfg.data.eval_synthetic_metrics,  # Moved to cfg.data
        eval_ood_metrics=cfg.data.eval_ood_metrics,              # Moved to cfg.data
        lr=cfg.lr,
        scheduler_patience=cfg.scheduler_patience,               # Flattended!
        scheduler_factor=cfg.scheduler_factor,                   # Flattended!
        min_lr=cfg.min_lr,                                       # Flattended!
        fs=cfg.fs,
        duration_s=cfg.duration_s,
        log_val_audio=cfg.log_val_audio,                         # Flattended!
        num_val_audio_examples=cfg.num_val_audio_examples,       # Flattended!
        stopgrad=cfg.get("stopgrad", ""),
    )


def _checkpoint_tag(cfg: DictConfig) -> str:
    # 1. Engine
    decoder = cfg.model.decoder
    impl = cfg.model.get("implementation", None)
    enc_type = cfg.model.get("encoder", {}).get("type", "baseline")

    if decoder == "ks":
        if impl == "oracle": engine = "oKSA"
        elif impl == "time_domain": engine = "tKSA"
        elif impl == "frequency_sampling": engine = "fKSA"
    elif decoder == "harmonics_noise":
        engine = "hpn_p" if enc_type == "enhanced" else "hpn"

    # 2. Training regime
    objective = cfg.training.objective
    if objective == "param_only":
        regime = "p_only"
    elif objective == "combined":
        regime = "p_audio"
    elif objective == "spectral_only":
        regime = "audio_only"
    else:
        regime = objective

    experiment = "real" if cfg.data.has_ood else "synth"
    tag = f"{experiment}_{engine}_{regime}"

    # 4. Ablation suffix: keep these checkpoints distinct from the joint pilots so
    # evaluate.py's per-tag latest-checkpoint selection does not shadow them.
    sg = str(cfg.get("stopgrad", "") or "").lower()
    has_onset = ("onset" in sg) or ("time" in sg) or ("both" in sg)
    has_f0 = ("f0" in sg) or ("both" in sg)
    if has_onset and has_f0:
        tag = f"{tag}_detach_both"
    elif has_onset:
        tag = f"{tag}_detach_onset"
    elif has_f0:
        tag = f"{tag}_detach_f0"

    return tag

class SyntheticEpochCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module) -> None:
        dm = trainer.datamodule
        if dm is not None and getattr(dm.hparams, "has_synthetic", False):
            dm.set_train_epoch(trainer.current_epoch)

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # torchlpc's numba CUDA scan path (used by the time-domain KS synth) logs an
    # INFO line per GPU dealloc; under hydra/lightning's INFO root logger this
    # floods the run log and adds per-step overhead. Quiet it to WARNING.
    import logging as _logging
    _logging.getLogger("numba").setLevel(_logging.WARNING)

    # ── Enforce Baseline Constraints ──
    errors = []

    # 1. Harmonics + Noise Constraints
    if cfg.model.decoder == "harmonics_noise":
        if not cfg.detector.use_external_detectors:
            errors.append("detector=external (H+N requires f0 + loudness detectors)")
        if cfg.training.objective != "spectral_only":
            errors.append("training=spectral_only (H+N has no ground-truth params)")
        if cfg.data.has_synthetic:
            errors.append("data=real (H+N cannot use synthetic KS data)")
        if not cfg.data.has_ood:
            errors.append("data=real (H+N requires NSynth OOD data)")
        if cfg.data.eval_synthetic_metrics:
            errors.append("Evaluate OOD metrics only (H+N has no param-level metrics)")

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

            if cfg.model.implementation == "oracle" and cfg.training.objective != "param_only":
                errors.append("model=ksa_oracle (Oracle implementation strictly requires training=param_only)")

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

    monitor = "val_synth/loss" if cfg.data.eval_synthetic_metrics else "val_ood/loss"

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
        patience=cfg.patience,
        min_delta=cfg.min_delta,
        mode="min",
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        gradient_clip_val=cfg.gradient_clip_val,
        logger=logger,
        accelerator=accelerator,
        val_check_interval=cfg.val_check_interval,
        callbacks=[ckpt_best, ckpt_last, early_stop, SyntheticEpochCallback()],
        enable_progress_bar=True,
    )
    trainer.fit(experiment, datamodule=datamodule)

if __name__ == "__main__":
    main()
