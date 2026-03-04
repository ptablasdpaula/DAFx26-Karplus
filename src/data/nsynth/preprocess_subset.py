"""Preprocess NSynth guitar-acoustic subset with onset, f0, and loudness detection."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from paths import NSYNTH_DIR
from src.detectors import detect_onsets, detect_f0, detect_loudness
from src.synths.param_registry import MIDI_D1, MIDI_D6
from src.synths.constants import DEFAULT_FS

SAMPLE_RATE = DEFAULT_FS
SEGMENT_LENGTH = SAMPLE_RATE * 4  # 64 000 samples (NSynth = 4 s)


@torch.no_grad()
def preprocess_nsynth_guitar_acoustic(
    splits: List[str] = ("test",),
) -> None:
    nsynth_root = Path(NSYNTH_DIR).resolve()

    for split in splits:
        print(f"\n▶ Processing split: {split}")
        split_in = nsynth_root / split
        examples_json = split_in / "examples.json"
        assert examples_json.exists(), f"Missing {examples_json}"

        meta_json: Dict[str, Any] = json.loads(examples_json.read_text())

        # ---- Filter keys --------------------------------------------------------
        keys = []
        for k, m in meta_json.items():
            fam = m.get("instrument_family_str", "")
            src = m.get("instrument_source_str", "")
            midi = m.get("pitch", None)

            if fam != "guitar":
                continue
            if src != "acoustic":
                continue
            if midi is None or not (MIDI_D1 <= midi <= MIDI_D6):
                continue

            # Keep tempo-synced (idx 8), remove reverb (idx 9)
            q = m.get("qualities", [0] * 10)
            if len(q) > 9 and q[9] == 1:
                continue

            keys.append(k)

        print(f"Kept {len(keys)} items after filtering "
              f"(guitar/acoustic/D1..D6/no-reverb).")

        # ---- Prepare output dir -------------------------------------------------
        split_out = nsynth_root / split / "preprocessed"
        items_dir = split_out / "items"
        items_dir.mkdir(parents=True, exist_ok=True)
        metadata: Dict[str, Any] = {}

        # ---- Iterate over items -------------------------------------------------
        for k in tqdm(keys, ncols=100):
            wav_path = split_in / "audio" / f"{k}.wav"
            x, sr = torchaudio.load(str(wav_path))  # (1, T)
            assert sr == SAMPLE_RATE, f"{k}: expected {SAMPLE_RATE}, got {sr}"
            x = x.squeeze(0)  # (T,)
            assert x.numel() == SEGMENT_LENGTH, (
                f"{k}: expected {SEGMENT_LENGTH} samples, got {x.numel()}"
            )

            x_np = x.detach().cpu().numpy()

            # ---- Onsets (times in seconds) --------------------------------------
            onset_times = detect_onsets(x_np, sr=SAMPLE_RATE)

            # ---- F0 (CREPE) ----------------------------------------------------
            x_2d = x.unsqueeze(0)  # (1, T) – batch dim for torchcrepe
            f0_hz, _, _ = detect_f0(x_2d, sr=SAMPLE_RATE)
            f0_hz = f0_hz[:-1] # Drop last

            # ---- Loudness (A-weighted) ------------------------------------------
            loudness = detect_loudness(x_np, sampling_rate=SAMPLE_RATE)

            # ---- Save tensors ---------------------------------------------------
            item_pt = items_dir / f"{k}.pt"
            torch.save({
                "audio":          x.cpu(),                                      # (T,)
                "onset_times":    torch.from_numpy(onset_times.astype(np.float32)),  # (N_onsets,)
                "f0_hz":          torch.from_numpy(f0_hz.astype(np.float32)),        # (F,)
                "loudness":       torch.from_numpy(loudness.astype(np.float32)),     # (L,)
            }, item_pt)

            # ---- Compact metadata entry -----------------------------------------
            m = meta_json[k]
            metadata[k] = {
                "path":                  f"{split}/preprocessed/items/{k}.pt",
                "num_samples":           int(x.numel()),
                "instrument_family_str": m.get("instrument_family_str", ""),
                "instrument_source_str": m.get("instrument_source_str", ""),
                "midi_pitch":            int(m.get("pitch", -1)),
                "midi_velocity":         int(m.get("velocity", -1)),
                "onset_times":           onset_times.tolist(),
            }

        # ---- Save split metadata ------------------------------------------------
        meta_path = split_out / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        print(f"Wrote {meta_path} with {len(metadata)} items.")


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class GuitarAcousticDataset(torch.utils.data.Dataset):
    """
    Yields per item:
        audio          (T,)
        f0_hz          (F,)
        f0_confidence  (F,)
        f0_times       (F,)
        onset_times    (N_onsets,)
        loudness       (L,)
    """
    def __init__(self, split: str = "test"):
        self.base = Path(NSYNTH_DIR).resolve() / split / "preprocessed"
        self.meta = json.loads((self.base / "metadata.json").read_text())
        self.keys = list(self.meta.keys())

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        k = self.keys[idx]
        item_path = Path(NSYNTH_DIR).resolve() / self.meta[k]["path"]
        pt = torch.load(item_path, weights_only=True)

        return {
            "audio":         pt["audio"].float(),
            "f0_hz":         pt["f0_hz"].float(),
            "onset_times":   pt["onset_times"].float(),
            "loudness":      pt["loudness"].float(),
        }

    def get_filename(self, idx: int) -> str:
        return self.keys[idx]


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    preprocess_nsynth_guitar_acoustic(splits=("test",))

    ds = GuitarAcousticDataset(split="test")
    print(f"\nLoaded {len(ds)} examples.")
    item = ds[0]
    for name, t in item.items():
        print(f"  {name:16s} {tuple(t.shape)}")