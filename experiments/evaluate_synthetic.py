from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.synths.synth import Synth, SynthConfig
from src.synths.ddsp import Implementation
from src.decoder import KSDecoder
from src.encoder import Encoder
from src.model import SoundMatchingModel
from src.data.synthetic_dataset import SyntheticDataset
from src.losses import MultiScaleSpectralLoss, SOT2048Loss, PLoss
from src.metrics import (
    from_gain_frames_to_onsets_seconds,
    compute_onset_precision_recall,
    compute_mean_cents_distance,
    compute_wmfcc,
    compute_rms,
)

# ── Constants ────────────────────────────────────────────────────────────────

EVAL_SEED = 77777          # distinct from train (42) and val (99999)
NUM_EVAL_SAMPLES = 250
FS = 16000
NUM_AUDIO_SAMPLES = 64000
NUM_FRAMES = 250
DURATION_S = 4.0

IMPL_MAP = {
    "time_domain": Implementation.TIME_DOMAIN,
    "frequency_sampling": Implementation.FREQUENCY_SAMPLING,
}

TAGS = [
    "Synth_Free_Time_Super",
    "Synth_Free_Time_Spec",
    "Synth_Free_Time_Comb",
    "Synth_Free_Freq_Super",
    "Synth_Free_Freq_Spec",
    "Synth_Free_Freq_Comb",
    "Synth_Det_Time_Super",
    "Synth_Det_Time_Spec",
    "Synth_Det_Time_Comb",
    "Synth_Det_Freq_Super",
    "Synth_Det_Freq_Spec",
    "Synth_Det_Freq_Comb",
]


# ── Model construction from saved config ─────────────────────────────────────

def _build_model_from_cfg(cfg) -> SoundMatchingModel:
    """Reconstruct a SoundMatchingModel from a resolved OmegaConf config."""
    if cfg.model.decoder == "ks":
        synth = Synth(SynthConfig(
            num_samples=cfg.num_audio_samples,
            fs=cfg.fs,
            implementation=IMPL_MAP[cfg.model.implementation],
            lagrange_order=cfg.model.synth.lagrange_order,
            n_fft=cfg.model.synth.n_fft,
        ))
        decoder = KSDecoder(
            synth=synth,
            use_external_detectors=cfg.detector.use_external_detectors,
        )
    else:
        raise ValueError(f"Unsupported decoder for synthetic eval: {cfg.model.decoder}")

    return SoundMatchingModel(
        decoder=decoder,
        encoder_kwargs=OmegaConf.to_container(cfg.model.encoder, resolve=True),
    )


def _load_model(tag: str, ckpt_dir: Path, device: torch.device) -> SoundMatchingModel:
    """Load config + best checkpoint for a given tag."""
    cfg_path = ckpt_dir / f"{tag}_config.yaml"
    assert cfg_path.exists(), f"Missing config: {cfg_path}"

    ckpt_path = ckpt_dir / f"{tag}_best.ckpt"
    if not ckpt_path.exists():
        # fall back to last if best not found
        ckpt_path = ckpt_dir / f"{tag}_last.ckpt"
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"

    cfg = OmegaConf.load(cfg_path)
    model = _build_model_from_cfg(cfg)

    # Lightning wraps model state under "state_dict" with "model." prefix
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"]
    model_state = {
        k.replace("model.", "", 1): v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    print(f"  Loaded {tag} from {ckpt_path.name}")
    return model


# ── Dataset generation ───────────────────────────────────────────────────────

def _generate_eval_set() -> list[dict]:
    """Generate NUM_EVAL_SAMPLES synthetic examples with a fixed seed."""
    ds = SyntheticDataset(
        num_samples_per_epoch=NUM_EVAL_SAMPLES,
        num_audio_samples=NUM_AUDIO_SAMPLES,
        num_frames=NUM_FRAMES,
        fs=FS,
        lti=False,
        blend_lti=True,
        random_seed=EVAL_SEED,
    )
    samples = []
    for sample in ds:
        samples.append(sample)
    return samples


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _run_inference(
    model: SoundMatchingModel,
    samples: list[dict],
    batch_size: int,
    device: torch.device,
) -> tuple[list[np.ndarray], list[dict]]:
    """Run encoder → activate → oracle_synth on all samples.

    Returns:
        pred_audios:  list of [num_audio_samples] numpy arrays
        pred_params_list: list of {name: [num_frames] numpy} dicts
    """
    pred_audios = []
    pred_params_list = []

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        audio_batch = torch.stack([s["audio"] for s in batch_samples]).to(device)

        # Encoder
        raw = model.encoder(audio_batch)                      # [B, P, T]
        params = model.decoder.activate(raw, detected=None)   # {name: [B, T]}

        # Oracle synthesis (high-fidelity, non-differentiable)
        with torch.no_grad():
            pred_audio, params = model.decoder.oracle_synth(params)

        # Collect per-example
        B = pred_audio.shape[0]
        for b in range(B):
            pred_audios.append(pred_audio[b].cpu().numpy())
            pred_params_list.append(
                {k: v[b].cpu().numpy() for k, v in params.items()}
            )

    return pred_audios, pred_params_list


# ── Metrics ──────────────────────────────────────────────────────────────────

def _compute_all_metrics(
    pred_audios: list[np.ndarray],
    pred_params_list: list[dict],
    samples: list[dict],
    ploss_fn: PLoss,
    mss_fn: MultiScaleSpectralLoss,
    sot_fn: SOT2048Loss,
    device: torch.device,
) -> dict[str, float]:
    """Compute all metrics averaged over the eval set."""
    precs, recs, f1s, cents_list = [], [], [], []
    mss_list, sot_list, wmfcc_list, rms_list = [], [], [], []
    ploss_totals = []

    for i in range(len(samples)):
        target_audio_np = samples[i]["audio"].numpy()
        target_params_np = {k: v.numpy() for k, v in samples[i]["params"].items()}
        pred_audio_np = pred_audios[i]
        pred_params_np = pred_params_list[i]

        # ── Onset P / R / F1 ──
        pred_onsets_s = from_gain_frames_to_onsets_seconds(
            pred_params_np["burst_gain"], DURATION_S
        )
        target_onsets_s = from_gain_frames_to_onsets_seconds(
            target_params_np["burst_gain"], DURATION_S
        )
        if len(target_onsets_s) > 0:
            p, r, f = compute_onset_precision_recall(pred_onsets_s, target_onsets_s)
            precs.append(p)
            recs.append(r)
            f1s.append(f)

        # ── Cents ──
        cents_list.append(
            compute_mean_cents_distance(pred_params_np["f0"], target_params_np["f0"])
        )

        # ── Spectral losses (as metrics) ──
        pred_t = torch.from_numpy(pred_audio_np).unsqueeze(0).to(device)
        tgt_t = torch.from_numpy(target_audio_np).unsqueeze(0).to(device)
        mss_list.append(mss_fn(pred_t, tgt_t).item())
        sot_list.append(sot_fn(pred_t, tgt_t).item())

        # ── PLoss ──
        pred_params_t = {k: torch.from_numpy(v).unsqueeze(0).to(device)
                         for k, v in pred_params_np.items()}
        target_params_t = {k: torch.from_numpy(v).unsqueeze(0).to(device)
                           for k, v in target_params_np.items()}
        pl_total, _ = ploss_fn(pred_params_t, target_params_t)
        ploss_totals.append(pl_total.item())

        # ── wMFCC ──
        wmfcc_list.append(compute_wmfcc(target_audio_np, pred_audio_np, sample_rate=FS))

        # ── RMS cosine similarity ──
        rms_list.append(
            compute_rms(
                target_audio_np[np.newaxis, :],
                pred_audio_np[np.newaxis, :],
                sample_rate=FS,
            )
        )

    return {
        "Precision": np.mean(precs) if precs else float("nan"),
        "Recall": np.mean(recs) if recs else float("nan"),
        "F1": np.mean(f1s) if f1s else float("nan"),
        "Cents": np.mean(cents_list),
        "PLoss": np.mean(ploss_totals),
        "MSS": np.mean(mss_list),
        "SOT": np.mean(sot_list),
        "wMFCC": np.mean(wmfcc_list),
        "RMS": np.mean(rms_list),
    }


# ── Audio saving ─────────────────────────────────────────────────────────────

def _short_tag(tag: str) -> str:
    """Synth_Free_Time_Super → free_time_super."""
    parts = tag.split("_")
    return "_".join(p.lower() for p in parts[1:])  # drop "Synth"


def _save_targets(samples: list[dict], audio_dir: Path) -> None:
    """Save target audio once (shared across all checkpoints)."""
    for i in range(len(samples)):
        target_np = samples[i]["audio"].numpy()
        sf.write(audio_dir / f"target_{i+1:03d}.wav", target_np, FS)


def _save_predictions(
    pred_audios: list[np.ndarray],
    audio_dir: Path,
    tag: str,
) -> None:
    """Save predicted audio with short tag prefix."""
    prefix = _short_tag(tag)
    for i, pred_np in enumerate(pred_audios):
        sf.write(audio_dir / f"{prefix}_{i+1:03d}.wav", pred_np, FS)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Synthetic evaluation")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints",
                        help="Directory containing checkpoint and config files")
    parser.add_argument("--out_dir", type=str, default="synthetic_eval",
                        help="Output directory for audio and results")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not set)")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Generate eval set (once, shared across all checkpoints) ──
    print(f"Generating {NUM_EVAL_SAMPLES} synthetic eval examples (seed={EVAL_SEED})...")
    samples = _generate_eval_set()
    print(f"  Done. {len(samples)} samples generated.")

    # ── Shared loss/metric functions ──
    ploss_fn = PLoss(fs=FS, weights={"f0": 2.0, "burst_gain": 5.0}).to(device)
    mss_fn = MultiScaleSpectralLoss().to(device)
    sot_fn = SOT2048Loss(sample_rate=FS, device=device).to(device)

    # ── Save targets once ──
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    print("Saving target audio...")
    _save_targets(samples, audio_dir)

    # ── Evaluate each checkpoint ──
    all_results = {}

    for tag in TAGS:
        cfg_path = ckpt_dir / f"{tag}_config.yaml"
        if not cfg_path.exists():
            print(f"\n⚠ Skipping {tag} — config not found at {cfg_path}")
            continue

        print(f"\n{'═' * 60}")
        print(f"Evaluating: {tag}")
        print(f"{'═' * 60}")

        model = _load_model(tag, ckpt_dir, device)

        print("  Running inference...")
        pred_audios, pred_params_list = _run_inference(
            model, samples, args.batch_size, device
        )

        print("  Computing metrics...")
        metrics = _compute_all_metrics(
            pred_audios, pred_params_list, samples,
            ploss_fn, mss_fn, sot_fn, device,
        )
        all_results[tag] = metrics

        # Print per-model results
        for k, v in metrics.items():
            print(f"    {k:12s}: {v:.4f}")

        print("  Saving predictions...")
        _save_predictions(pred_audios, audio_dir, tag)

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary table ────────────────────────────────────────────────────
    if not all_results:
        print("\nNo checkpoints found. Nothing to summarise.")
        return

    df = pd.DataFrame(all_results).T
    df.index.name = "Model"

    print(f"\n{'═' * 80}")
    print("SYNTHETIC EVALUATION — SUMMARY")
    print(f"{'═' * 80}")
    print(df.to_string(float_format="{:.4f}".format))

    csv_path = out_dir / "synthetic_eval_results.csv"
    df.to_csv(csv_path)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()