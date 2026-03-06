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

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src.synths.synth import Synth, SynthConfig
from src.synths.ddsp import Implementation
from src.decoder import KSDecoder
from src.model import SoundMatchingModel
from src.data.data_module import SoundMatchingDataModule
from experiments.sound_matching_experiment import SoundMatchingExperiment

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

    return SoundMatchingDataModule(
        has_synthetic=cfg.data.has_synthetic,
        has_ood=cfg.data.has_ood,
        fs=cfg.fs,
        num_audio_samples=cfg.num_audio_samples,
        num_frames=cfg.num_frames,
        duration_s=cfg.duration_s,
        batch_size=cfg.experiment.batch_size,
        num_workers=cfg.experiment.num_workers,
        synthetic_cfg=OmegaConf.to_container(cfg.data.synthetic, resolve=True)
        if cfg.data.has_synthetic else None,
        nsynth_root=nsynth_root,
        nsynth_split_train=nsynth_split_train,
        nsynth_split_val=nsynth_split_val,
    )


def _build_experiment(cfg: DictConfig, model: SoundMatchingModel) -> SoundMatchingExperiment:
    return SoundMatchingExperiment(
        model=model,
        objective=cfg.training.objective,
        param_only_epochs=cfg.training.param_only_epochs,
        fadein_epochs=cfg.training.fadein_epochs,
        w_mss=cfg.training.w_mss,
        w_sot=cfg.training.w_sot,
        w_param=cfg.training.w_param,
        param_weights=OmegaConf.to_container(cfg.training.param_weights, resolve=True),
        eval_synthetic_metrics=cfg.experiment.eval_synthetic_metrics,
        eval_ood_metrics=cfg.experiment.eval_ood_metrics,
        lr=cfg.lr,
        fs=cfg.fs,
        duration_s=cfg.duration_s,
        log_val_audio=cfg.experiment.log_val_audio,
        num_val_audio_examples=cfg.experiment.num_val_audio_examples,
    )


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed, workers=True)

    model = _build_model(cfg)
    experiment = _build_experiment(cfg, model)
    datamodule = _build_datamodule(cfg)

    n_total = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    print(f"SoundMatchingModel: {n_total:,} params (encoder: {n_enc:,})")
    print(f"Decoder: {model.decoder.__class__.__name__} "
          f"({model.decoder.num_params} outputs)")

    logger = WandbLogger(project="DAFx26-Karplus", log_model=False)
    logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        max_epochs=cfg.experiment.max_epochs,
        gradient_clip_val=cfg.experiment.gradient_clip_val,
        logger=logger,
        accelerator=accelerator,
        val_check_interval=cfg.experiment.val_check_interval,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    trainer.fit(experiment, datamodule=datamodule)


if __name__ == "__main__":
    main()