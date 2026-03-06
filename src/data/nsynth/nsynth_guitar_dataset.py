from __future__ import annotations

import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class NsynthGuitarDataset(Dataset):
    """
    Args:
        nsynth_root:       Path to the nsynth data directory.
        split:             ``"training"`` | ``"validation"`` | ``"test"``.
        num_frames:        Number of control-rate frames the encoder produces.
        num_audio_samples: Expected audio length in samples.
        duration_s:        Audio duration in seconds.
    """

    def __init__(
        self,
        nsynth_root: str | Path,
        split: str = "test",
        num_frames: int = 250,
        num_audio_samples: int = 64000,
        duration_s: float = 4.0,
    ):
        super().__init__()
        self.nsynth_root = Path(nsynth_root).resolve()
        self.split = split
        self.num_frames = num_frames
        self.num_audio_samples = num_audio_samples
        self.duration_s = duration_s
        self._frame_hop_s = duration_s / num_frames

        meta_path = self.nsynth_root / split / "preprocessed" / "metadata.json"
        assert meta_path.exists(), (
            f"Missing {meta_path} — run preprocess_subset.py first."
        )
        self._meta: dict = json.loads(meta_path.read_text())
        self._keys: list[str] = list(self._meta.keys())

    def __len__(self) -> int:
        return len(self._keys)

    # ── Conversion helpers ───────────────────────────────────────────────

    def _onset_times_to_mask(self, onset_times: torch.Tensor) -> torch.Tensor:
        """Convert onset times (seconds) → binary mask [num_frames]."""
        mask = torch.zeros(self.num_frames, dtype=torch.float32)
        if onset_times.numel() == 0:
            return mask
        frame_idx = (onset_times / self._frame_hop_s).round().long()
        frame_idx = frame_idx.clamp(0, self.num_frames - 1)
        mask[frame_idx] = 1.0
        return mask

    def _resample_to_frames(self, x: torch.Tensor, fallback: float = 0.0) -> torch.Tensor:
        """Resample variable-length 1-D tensor → [num_frames] via linear interp."""
        if x.numel() == 0:
            return torch.full((self.num_frames,), fallback)
        if x.numel() == 1:
            return x.expand(self.num_frames)
        # F.interpolate expects [B, C, L]
        return F.interpolate(
            x.unsqueeze(0).unsqueeze(0),
            size=self.num_frames,
            mode="linear",
            align_corners=True,
        ).squeeze(0).squeeze(0)

    # ── __getitem__ ──────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> dict:
        key = self._keys[idx]
        item_path = self.nsynth_root / self._meta[key]["path"]
        pt = torch.load(item_path, weights_only=True)

        audio = pt["audio"].float()
        assert audio.numel() == self.num_audio_samples, (
            f"{key}: expected {self.num_audio_samples} samples, "
            f"got {audio.numel()}"
        )

        detected = {
            "onsets":    self._onset_times_to_mask(pt["onset_times"]),
            "f0":        self._resample_to_frames(pt["f0_hz"], fallback=220.0),
            "loudness":  self._resample_to_frames(pt["loudness"], fallback=-60.0),
        }

        return {
            "audio":    audio,
            "detected": detected,
        }

    def get_key(self, idx: int) -> str:
        return self._keys[idx]