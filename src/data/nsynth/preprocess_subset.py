"""Preprocess NSynth guitar-acoustic subset with onset, f0, and loudness detection."""
from __future__ import annotations
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torchaudio
import torchcrepe
from tqdm import tqdm

from paths import NSYNTH_DIR
from src.detectors import detect_onsets, detect_loudness
from src.synths.param_registry import MIDI_D1, MIDI_D6, F0_MIN_HZ, F0_MAX_HZ
from src.synths.constants import DEFAULT_FS, DEFAULT_CREPE_HOP_LENGTH

SAMPLE_RATE = DEFAULT_FS
SEGMENT_LENGTH = SAMPLE_RATE * 4  # 64 000 samples (NSynth = 4 s)

# --- Tunable knobs -----------------------------------------------------------
CREPE_MODEL = "tiny"  # "tiny" is ~10x faster than "full"; accurate enough for
                       # clean monophonic guitar.  Switch to "full" if needed.
AUDIO_BATCH_SIZE = 64  # audio files stacked per CREPE call
CREPE_FRAME_BATCH = 2048  # frames through CNN per forward pass
NUM_WORKERS = 4  # CPU workers for onset + loudness (match --cpus-per-task)


def _process_cpu(args):
    """Onset + loudness for one item (runs in a worker process)."""
    k, x_np, sr = args
    onset_times = detect_onsets(x_np, sr=sr)
    loudness = detect_loudness(x_np, sampling_rate=sr)
    return k, onset_times, loudness


@torch.no_grad()
def preprocess_nsynth_guitar_acoustic(
    splits: List[str] = ("test",),
) -> None:
    nsynth_root = Path(NSYNTH_DIR).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Pre-load CREPE weights once (avoids reload every predict call)
    torchcrepe.load.model(device, CREPE_MODEL)
    print(f"CREPE '{CREPE_MODEL}' model loaded on {device}")

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

            # Remove reverb (idx 9)
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

        split_t0 = time.time()

        # ---- Iterate in batches -------------------------------------------------
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
            num_batches = (len(keys) + AUDIO_BATCH_SIZE - 1) // AUDIO_BATCH_SIZE
            for batch_start in tqdm(range(0, len(keys), AUDIO_BATCH_SIZE),
                                    total=num_batches, desc=split, ncols=100):
                batch_keys = keys[batch_start : batch_start + AUDIO_BATCH_SIZE]

                # ---- Load audio -------------------------------------------------
                t0 = time.time()
                audios_np = []
                audios_t = []
                for k in batch_keys:
                    wav_path = split_in / "audio" / f"{k}.wav"
                    x, sr = torchaudio.load(str(wav_path))
                    assert sr == SAMPLE_RATE
                    x = x.squeeze(0)
                    assert x.numel() == SEGMENT_LENGTH
                    audios_np.append(x.numpy())
                    audios_t.append(x)
                t1 = time.time()

                # ---- Batched F0 (CREPE on GPU) ----------------------------------
                batch_tensor = torch.stack(audios_t).to(device)
                f0_batch, conf_batch = torchcrepe.predict(
                    batch_tensor.float(),
                    SAMPLE_RATE,
                    hop_length=DEFAULT_CREPE_HOP_LENGTH,
                    fmin=F0_MIN_HZ,
                    fmax=F0_MAX_HZ,
                    model=CREPE_MODEL,
                    batch_size=CREPE_FRAME_BATCH,
                    device=device,
                    return_periodicity=True,
                )
                t2 = time.time()

                # ---- CPU: parallel onset + loudness -----------------------------
                futures = {
                    pool.submit(_process_cpu, (k, audios_np[i], SAMPLE_RATE)): (i, k)
                    for i, k in enumerate(batch_keys)
                }
                cpu_results = {}
                for fut in as_completed(futures):
                    k_res, onset_times, loudness = fut.result()
                    cpu_results[k_res] = (onset_times, loudness)
                t3 = time.time()

                # ---- Save -------------------------------------------------------
                for i, k in enumerate(batch_keys):
                    onset_times, loudness = cpu_results[k]
                    f0_hz = f0_batch[i, :-1].cpu().numpy()

                    torch.save({
                        "audio":       audios_t[i].cpu(),
                        "onset_times": torch.from_numpy(onset_times.astype(np.float32)),
                        "f0_hz":       torch.from_numpy(f0_hz.astype(np.float32)),
                        "loudness":    torch.from_numpy(loudness.astype(np.float32)),
                    }, items_dir / f"{k}.pt")

                    m = meta_json[k]
                    metadata[k] = {
                        "path":                  f"{split}/preprocessed/items/{k}.pt",
                        "num_samples":           int(audios_t[i].numel()),
                        "instrument_family_str": m.get("instrument_family_str", ""),
                        "instrument_source_str": m.get("instrument_source_str", ""),
                        "midi_pitch":            int(m.get("pitch", -1)),
                        "midi_velocity":         int(m.get("velocity", -1)),
                        "onset_times":           onset_times.tolist(),
                    }
                t4 = time.time()

                print(f"  batch {batch_start//AUDIO_BATCH_SIZE + 1}/{num_batches}: "
                      f"load={t1-t0:.1f}s  crepe={t2-t1:.1f}s  "
                      f"cpu={t3-t2:.1f}s  save={t4-t3:.1f}s")

        split_elapsed = time.time() - split_t0
        print(f"Split '{split}' done in {split_elapsed:.1f}s "
              f"({split_elapsed/60:.1f} min)")

        # ---- Save split metadata ------------------------------------------------
        meta_path = split_out / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        print(f"Wrote {meta_path} with {len(metadata)} items.")

def compute_loudness_stats(splits=("test", "validation", "training")):
    """Compute global loudness min/max across all splits and save."""
    nsynth_root = Path(NSYNTH_DIR).resolve()
    global_min = float("inf")
    global_max = float("-inf")

    for split in splits:
        meta_path = nsynth_root / split / "preprocessed" / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        items_dir = nsynth_root / split / "preprocessed" / "items"

        for k in tqdm(meta.keys(), desc=f"stats/{split}", ncols=100):
            pt = torch.load(items_dir / f"{k}.pt", weights_only=True)
            loud = pt["loudness"]
            global_min = min(global_min, loud.min().item())
            global_max = max(global_max, loud.max().item())

    stats = {"loudness_min": global_min, "loudness_max": global_max}
    stats_path = nsynth_root / "loudness_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Loudness stats: min={global_min:.4f}, max={global_max:.4f}")
    print(f"Wrote {stats_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data.nsynth.nsynth_guitar_dataset import NsynthGuitarDataset

    preprocess_nsynth_guitar_acoustic(splits=("test", "validation", "training"))
    compute_loudness_stats()

    print(f"\nCUDA available: {torch.cuda.is_available()}")

    ds = NsynthGuitarDataset(nsynth_root=NSYNTH_DIR, split="test")
    print(f"Loaded {len(ds)} examples.")
    item = ds[0]
    for name, t in item.items():
        if isinstance(t, torch.Tensor):
            print(f"  {name:16s} {tuple(t.shape)}")
        else:
            print(f"  {name:16s} {type(t).__name__}: {t}")