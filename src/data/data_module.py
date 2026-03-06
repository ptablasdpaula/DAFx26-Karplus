
from __future__ import annotations

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.synthetic_dataset import SyntheticDataset

class SoundMatchingDataModule(pl.LightningDataModule):
    """
    Args:
        has_synthetic / has_ood:  Which data sources to include.
        fs / num_audio_samples / num_frames / duration_s: Audio params.
        batch_size / num_workers: DataLoader config.
        synthetic_cfg:  Dict of kwargs for ``SyntheticDataset``.
        nsynth_root:    Path to nsynth data dir.
        nsynth_split_train / nsynth_split_val: NSynth split names.
        val_synthetic_size / val_synthetic_seed: Fixed val set config.
    """

    def __init__(
        self,
        has_synthetic: bool = True,
        has_ood: bool = False,
        fs: int = 16000,
        num_audio_samples: int = 64000,
        num_frames: int = 250,
        duration_s: float = 4.0,
        batch_size: int = 16,
        num_workers: int = 4,
        synthetic_cfg: dict | None = None,
        nsynth_root: str | None = None,
        nsynth_split_train: str = "training",
        nsynth_split_val: str = "test",
        val_synthetic_size: int = 250,
        val_synthetic_seed: int = 99999,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        hp = self.hparams
        syn_cfg = hp.synthetic_cfg or {}

        if hp.has_synthetic:
            self.train_synthetic = SyntheticDataset(
                num_samples_per_epoch=syn_cfg.get("num_samples_per_epoch", 4096),
                num_audio_samples=hp.num_audio_samples,
                num_frames=hp.num_frames,
                fs=hp.fs,
                lti=syn_cfg.get("lti", False),
                blend_lti=syn_cfg.get("blend_lti", True),
                random_seed=syn_cfg.get("random_seed", 42),
            )
            self.val_synthetic = SyntheticDataset(
                num_samples_per_epoch=hp.val_synthetic_size,
                num_audio_samples=hp.num_audio_samples,
                num_frames=hp.num_frames,
                fs=hp.fs,
                lti=syn_cfg.get("lti", False),
                blend_lti=syn_cfg.get("blend_lti", True),
                random_seed=hp.val_synthetic_seed,
            )

        if hp.has_ood:
            from src.data.nsynth.nsynth_guitar_dataset import NsynthGuitarDataset

            assert hp.nsynth_root is not None, (
                "has_ood=True but nsynth_root not set."
            )
            self.train_nsynth = NsynthGuitarDataset(
                nsynth_root=hp.nsynth_root,
                split=hp.nsynth_split_train,
                num_frames=hp.num_frames,
                num_audio_samples=hp.num_audio_samples,
                duration_s=hp.duration_s,
            )
            self.val_nsynth = NsynthGuitarDataset(
                nsynth_root=hp.nsynth_root,
                split=hp.nsynth_split_val,
                num_frames=hp.num_frames,
                num_audio_samples=hp.num_audio_samples,
                duration_s=hp.duration_s,
            )

    def train_dataloader(self):
        hp = self.hparams
        loaders = {}

        if hp.has_synthetic:
            loaders["synthetic"] = DataLoader(
                self.train_synthetic,
                batch_size=hp.batch_size,
                num_workers=hp.num_workers,
            )
        if hp.has_ood:
            loaders["nsynth"] = DataLoader(
                self.train_nsynth,
                batch_size=hp.batch_size,
                shuffle=True,
                num_workers=hp.num_workers,
            )

        if len(loaders) == 1:
            return next(iter(loaders.values()))

        from pytorch_lightning.utilities import CombinedLoader
        return CombinedLoader(loaders, mode="max_size_cycle")

    def val_dataloader(self):
        hp = self.hparams
        loaders = []

        if hp.has_synthetic:
            loaders.append(DataLoader(
                self.val_synthetic,
                batch_size=hp.batch_size,
                num_workers=hp.num_workers,
            ))
        if hp.has_ood:
            loaders.append(DataLoader(
                self.val_nsynth,
                batch_size=hp.batch_size,
                num_workers=hp.num_workers,
            ))

        return loaders if len(loaders) > 1 else loaders[0]