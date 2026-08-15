import argparse
import os
import sys
from pathlib import Path

# Run directly from the repository root: `python experiments/evaluate.py ...`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Perceptual Models via Hugging Face
from transformers import EncodecModel, ClapModel, ClapProcessor

# Local imports
from src.synths.synth import Synth, SynthConfig
from src.synths.ddsp import Implementation
from src.decoder import KSDecoder
from src.model import SoundMatchingModel
from data.nsynth.nsynth_guitar_dataset import NsynthGuitarDataset
from paths import NSYNTH_DIR
from data.synthetic_dataset import SyntheticDataset
from src.detectors import run_detectors_on_batch

from src.losses import MultiScaleSpectralLoss, SOT2048Loss, EventSetLoss
from src.metrics import (
    compute_wmfcc,
    compute_rms,
    compute_mean_cents_distance,
    compute_onset_precision_recall
)


def compute_kad(real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
    with tqdm(total=4, desc="  ↳ Computing KAD", position=1, leave=False,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
        x = real_features.to(torch.float64)
        y = fake_features.to(torch.float64)
        n, m = x.size(0), y.size(0)
        pbar.update(1)

        def sq_dist(mat1, mat2):
            norm1 = mat1.pow(2).sum(dim=1, keepdim=True)
            norm2 = mat2.pow(2).sum(dim=1)
            return torch.clamp(norm1 + norm2 - 2.0 * mat1 @ mat2.T, min=0.0)

        dist_sq_xx = sq_dist(x, x)
        nonzero_ref_dists_sq = dist_sq_xx[dist_sq_xx > 0]
        sigma_sq = torch.median(nonzero_ref_dists_sq)
        gamma = 1.0 / (2.0 * sigma_sq + 1e-8)
        pbar.update(1)

        def rbf_kernel(mat1, mat2):
            return torch.exp(-gamma * sq_dist(mat1, mat2))

        K_xx = rbf_kernel(x, x)
        K_yy = rbf_kernel(y, y)
        K_xy = rbf_kernel(x, y)
        pbar.update(1)

        mmd_sq = (
                (K_xx.sum() - torch.trace(K_xx)) / (n * (n - 1)) +
                (K_yy.sum() - torch.trace(K_yy)) / (m * (m - 1)) -
                2.0 * K_xy.mean()
        )
        pbar.update(1)

    return float(mmd_sq.item() * 1000.0)


# ── MODEL LOADING HELPERS ──────────────────────────────────────────────────
IMPL_MAP = {
    "time_domain": Implementation.TIME_DOMAIN,
    "frequency_sampling": Implementation.FREQUENCY_SAMPLING,
    "oracle": Implementation.TIME_DOMAIN
}

def _build_model_from_cfg(cfg) -> SoundMatchingModel:
    if cfg.model.decoder == "ks":
        synth = Synth(SynthConfig(
            num_samples=cfg.num_audio_samples, fs=cfg.fs,
            implementation=IMPL_MAP[cfg.model.implementation],
            lagrange_order=cfg.model.synth.lagrange_order,
            n_fft=cfg.model.synth.n_fft,
            hop_length=cfg.model.synth.hop_length
        ))
        decoder = KSDecoder(synth=synth, use_external_detectors=cfg.detector.use_external_detectors)
    elif cfg.model.decoder == "harmonics_noise":
        from src.decoder import HarmonicsNoiseDecoder
        decoder = HarmonicsNoiseDecoder(
            fs=cfg.fs, num_samples=cfg.num_audio_samples,
            n_harmonics=cfg.model.get("n_harmonics", 100),
            n_noise_bands=cfg.model.get("n_noise_bands", 65),
            use_external_detectors=cfg.detector.use_external_detectors,
            z_dim=cfg.model.get("z_dim", 16), hidden_dim=cfg.model.get("hidden_dim", 512)
        )
    return SoundMatchingModel(decoder=decoder, encoder_kwargs=OmegaConf.to_container(cfg.model.encoder, resolve=True))


def _get_latest_run(tag: str, ckpt_dir: Path, expected: dict | None = None):
    """Finds the newest checkpoint and config file for a given tag based on the timestamp."""
    import re
    valid_runs = []
    # The tag must be followed *only* by the run timestamp, so a base tag like
    # "tKSA_p_audio" does NOT also match sub-variants such as
    # "tKSA_p_audio_detach_onset_..." (whose suffix starts with "_detach", not "_<digits>").
    ts_suffix = re.compile(r"_\d{8}_\d{6}(?:_\d{6})?_best\.ckpt$")

    # Grab all checkpoints that start with our tag
    for ckpt_path in ckpt_dir.glob(f"{tag}*best.ckpt"):
        if not ts_suffix.match(ckpt_path.name[len(tag):]):
            continue
        # Predict what the matching config file should be named
        config_name = ckpt_path.name.replace("_best.ckpt", "_config.yaml")
        config_path = ckpt_dir / config_name

        # Only consider it valid if BOTH the checkpoint and config exist
        if config_path.exists():
            if expected is not None and not _config_matches(config_path, expected):
                continue
            valid_runs.append((ckpt_path, config_path))

    if not valid_runs:
        return None, None

    # YYYYMMDD_HHMMSS, an alphabetical sort is identical to a chronological sort!
    valid_runs.sort(key=lambda x: x[0].name)

    # The last item in the list is the newest one
    latest_ckpt, latest_config = valid_runs[-1]
    return latest_config, latest_ckpt


def _config_matches(config_path: Path, expected: dict) -> bool:
    cfg = OmegaConf.load(config_path)
    for dotted_key, value in expected.items():
        node = cfg
        for part in dotted_key.split("."):
            node = node.get(part)
        if node != value:
            return False
    return True


def _load_model(config_path: Path, ckpt_path: Path, device: torch.device):
    cfg = OmegaConf.load(config_path)
    model = _build_model_from_cfg(cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def _split_tag(tag: str) -> tuple[str, str, str, str]:
    experiment, remainder = tag.split("_", 1)
    for regime in ("audio_only", "p_audio", "p_only"):
        marker = f"_{regime}"
        idx = remainder.find(marker)
        if idx < 0:
            continue
        suffix = remainder[idx + len(marker):].lstrip("_")
        if suffix and not suffix.startswith("detach_"):
            continue
        return experiment, remainder[:idx], regime, suffix
    raise ValueError(f"Could not parse evaluation tag: {tag}")


def _tag_to_rel_path(tag: str) -> str:
    _, engine, regime, suffix = _split_tag(tag)
    rel_path = f"{engine.lower()}/{regime}"
    if suffix:
        rel_path = f"{rel_path}/{suffix.lower()}"
    return rel_path


def _expected_config(mode: str, tag: str) -> dict:
    experiment, _, regime, _ = _split_tag(tag)
    if regime == "p_only":
        return {
            "training.objective": "param_only",
            "data.has_synthetic": True,
            "data.has_ood": False,
        }
    if regime == "p_audio":
        return {
            "training.objective": "combined",
            "data.has_synthetic": True,
            "data.has_ood": experiment == "real",
        }
    if regime == "audio_only":
        return {
            "training.objective": "spectral_only",
            "data.has_synthetic": experiment == "synth",
            "data.has_ood": experiment == "real",
        }
    return {}




# ── MAIN EVALUATION LOOP ────────────────────────────────────────────────────
def run_evaluation(mode, args):
    device = torch.device(args.device)
    ckpt_dir = Path(args.ckpt_dir)
    audio_mode = "real" if mode == "nsynth" else mode
    audio_root = Path(args.audio_out_dir) / audio_mode

    # 1. Conditionally Load Perceptual Models
    if mode == "nsynth" and not getattr(args, "render_only", False):
        print(f"\n--- Loading Perceptual Models to {device.type.upper()} ---")
        encodec = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device).eval()
        clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device).eval()

    # 2. Setup Data and Domain-Specific Metric Functions
    if mode == "nsynth":
        ds = NsynthGuitarDataset(nsynth_root=NSYNTH_DIR, split="test")
        tags = [
            "real_fKSA_p_audio",
            "real_tKSA_p_audio",
            "real_fKSA_audio_only",
            "real_tKSA_audio_only",
            "real_hpn_audio_only",
            "real_hpn_p_audio_only",
            "synth_oKSA_p_only",
            # P+Audio with audio-loss stop-gradient on onset/f0 (mix regime)
            "real_tKSA_p_audio_detach_onset", "real_tKSA_p_audio_detach_f0", "real_tKSA_p_audio_detach_both",
            "real_fKSA_p_audio_detach_onset", "real_fKSA_p_audio_detach_f0", "real_fKSA_p_audio_detach_both",
        ]

    else:
        ds = SyntheticDataset(num_samples_per_epoch=290, fs=16000, random_seed=77777)
        tags = [
            "synth_oKSA_p_only",
            "synth_fKSA_p_audio",
            "synth_tKSA_p_audio",
            "synth_fKSA_audio_only",
            "synth_tKSA_audio_only",
            # P+Audio with audio-loss stop-gradient on onset/f0 (synth regime)
            "synth_tKSA_p_audio_detach_onset", "synth_tKSA_p_audio_detach_f0", "synth_tKSA_p_audio_detach_both",
            "synth_fKSA_p_audio_detach_onset", "synth_fKSA_p_audio_detach_f0", "synth_fKSA_p_audio_detach_both",
        ]
        event_loss_fn = EventSetLoss(fs=16000).to(device)

    if getattr(args, "tags", None):
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    mss_fn = MultiScaleSpectralLoss().to(device)
    sot_fn = SOT2048Loss(sample_rate=16000).to(device)

    target_path = audio_root / "target"
    target_path.mkdir(parents=True, exist_ok=True)
    targets_saved = False

    print(f"\n--- Starting {mode.upper()} Evaluation ---")
    all_results = {}

    tag_pbar = tqdm(tags, desc="Overall Progress", position=0)
    for tag in tag_pbar:
        tag_pbar.set_postfix({"Current Model": tag})

        # Look for the latest timestamped files dynamically
        config_path, ckpt_path = _get_latest_run(tag, ckpt_dir, _expected_config(mode, tag))

        if config_path is None or ckpt_path is None:
            print(f"\n⚠️ Skipping {tag}: No valid checkpoint/config pair found.")
            continue

        print(f"\n  ↳ Loading weights from: {ckpt_path.name}")
        model = _load_model(config_path, ckpt_path, device)
        pred_path = audio_root / "pred" / _tag_to_rel_path(tag)
        pred_path.mkdir(parents=True, exist_ok=True)

        metrics = {"mss": [], "sot": [], "wmfcc": [], "rms": []}

        if mode == "nsynth":
            all_clap_tgt, all_clap_pred = [], []
            all_encodec_tgt, all_encodec_pred = [], []
        else:
            metrics.update({"precision": [], "recall": [], "f1": [], "cents_dist": [], "param_loss": []})

        idx = 1
        with torch.inference_mode():
            batch_pbar = tqdm(loader, desc="  ↳ Processing Batches", position=1, leave=False)
            for batch in batch_pbar:
                tgt_audio = batch["audio"].to(device)

                # Check if dataset has pre-computed detections
                detected = {k: v.to(device) for k, v in batch["detected"].items()} if "detected" in batch else None

                # ON-THE-FLY DETECTION: Mirroring your training loop!
                if model.decoder.use_external_detectors and detected is None:
                    num_frames = model.encoder.num_frames if hasattr(model.encoder, 'num_frames') else 250
                    detected = run_detectors_on_batch(
                        tgt_audio,
                        sr=16000,
                        num_frames=num_frames
                    )
                    # run_detectors_on_batch keeps things on the device, but let's be 100% safe
                    detected = {k: v.to(device) for k, v in detected.items()}

                raw = model.encoder(tgt_audio)
                params = model.decoder.activate(raw, detected=detected)
                pred_audio, _ = model.decoder.oracle_synth(params)

                metrics["mss"].append(mss_fn(pred_audio, tgt_audio).item())
                metrics["sot"].append(sot_fn(pred_audio, tgt_audio).item())

                if mode == "nsynth" and not getattr(args, "render_only", False):
                    t24 = torchaudio.functional.resample(tgt_audio, 16000, 24000).unsqueeze(1)
                    p24 = torchaudio.functional.resample(pred_audio, 16000, 24000).unsqueeze(1)

                    # Continuous encoder embeddings for EnCodec KAD
                    with torch.no_grad():
                        enc_tgt = encodec.encoder(t24).flatten(1)  # (B, C*T)
                        enc_pred = encodec.encoder(p24).flatten(1)
                    all_encodec_tgt.append(enc_tgt)
                    all_encodec_pred.append(enc_pred)

                    t48 = torchaudio.functional.resample(tgt_audio, 16000, 48000).cpu().numpy()
                    p48 = torchaudio.functional.resample(pred_audio, 16000, 48000).cpu().numpy()

                    out_tgt = clap_model.get_audio_features(
                        **clap_processor(audio=[a for a in t48], return_tensors="pt", sampling_rate=48000).to(device)
                    )
                    out_pred = clap_model.get_audio_features(
                        **clap_processor(audio=[a for a in p48], return_tensors="pt", sampling_rate=48000).to(device)
                    )

                    c_tgt = out_tgt if isinstance(out_tgt, torch.Tensor) else out_tgt.pooler_output
                    c_pred = out_pred if isinstance(out_pred, torch.Tensor) else out_pred.pooler_output

                    all_clap_tgt.append(c_tgt)
                    all_clap_pred.append(c_pred)

                elif mode == "synthetic":
                    batch_events_dev = {k: v.to(device) for k, v in batch["events"].items()}
                    param_loss, _ = event_loss_fn(raw, batch_events_dev)
                    metrics["param_loss"].append(param_loss.item())

                for b in range(tgt_audio.shape[0]):
                    p_wav = pred_audio[b].cpu().numpy()
                    t_wav = tgt_audio[b].cpu().numpy()

                    metrics["wmfcc"].append(compute_wmfcc(t_wav, p_wav, sample_rate=16000))
                    metrics["rms"].append(compute_rms(t_wav, p_wav, sample_rate=16000))

                    if mode == "synthetic" and "events" in batch:
                        tgt_exists = batch["events"]["exists"][b].detach().cpu().numpy()
                        duration_s = tgt_audio.shape[-1] / 16000.0

                        if model.decoder.use_external_detectors:
                            # Det models: exists/time/f0 come from activated params (detector-derived)
                            pred_exists = params["exists"][b].detach().cpu().numpy()
                            pred_times_s = params["time"][b].detach().cpu().numpy() * duration_s
                            pred_f0 = params["f0"][b].detach().cpu().numpy()
                        else:
                            # Free models: exists/time from raw logits, f0 from encoder head
                            pred_exists = torch.sigmoid(raw["exists"][b]).squeeze(-1).detach().cpu().numpy()
                            pred_times_s = torch.sigmoid(raw["time"][b]).squeeze(-1).detach().cpu().numpy() * duration_s
                            pred_f0 = raw["f0_hz"][b].squeeze(-1).detach().cpu().numpy()

                        tgt_times_s = batch["events"]["time"][b].detach().cpu().numpy() * duration_s
                        tgt_f0 = batch["events"]["f0"][b].detach().cpu().numpy()

                        pred_mask = pred_exists > 0.5
                        tgt_mask = tgt_exists > 0.5

                        pred_onsets_s = np.sort(pred_times_s[pred_mask])
                        tgt_onsets_s = np.sort(tgt_times_s[tgt_mask])

                        try:
                            p, r, f1 = compute_onset_precision_recall(pred_onsets_s, tgt_onsets_s)
                            metrics["precision"].append(p)
                            metrics["recall"].append(r)
                            metrics["f1"].append(f1)
                        except Exception as e:
                            print(f"mir_eval error: {e}") 
                            metrics["precision"].append(0.0)
                            metrics["recall"].append(0.0)
                            metrics["f1"].append(0.0)

                        # Cents distance (f0)
                        valid_p_f0 = pred_f0[pred_mask]
                        valid_t_f0 = tgt_f0[tgt_mask]
                        if len(valid_p_f0) > 0 and len(valid_t_f0) > 0:
                            metrics["cents_dist"].append(compute_mean_cents_distance(valid_p_f0, valid_t_f0[0]))

                    if not targets_saved:
                        sf.write(target_path / f"{idx:03d}.wav", t_wav, 16000)
                    sf.write(pred_path / f"{idx:03d}.wav", p_wav, 16000)
                    idx += 1

        targets_saved = True

        all_results[tag] = {k: np.mean(v) for k, v in metrics.items() if len(v) > 0}

        if mode == "nsynth" and not getattr(args, "render_only", False):
            clap_kad = compute_kad(torch.cat(all_clap_tgt, dim=0), torch.cat(all_clap_pred, dim=0))
            all_results[tag]["CLAP_KAD"] = clap_kad
            encodec_kad = compute_kad(torch.cat(all_encodec_tgt, dim=0), torch.cat(all_encodec_pred, dim=0))
            all_results[tag]["EnCodec_KAD"] = encodec_kad

    result_mode = "real" if mode == "nsynth" else mode
    out_csv = getattr(args, "out_csv", None) or (Path(args.out_dir) / f"{result_mode}_results.csv")
    pd.DataFrame(all_results).T.to_csv(out_csv)
    print(f"\n✅ Done! Results saved to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="synthetic", choices=["nsynth", "synthetic", "all"])
    parser.add_argument("--ckpt_dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--out_dir", type=str, default="experiments/evaluation")
    parser.add_argument("--audio_out_dir", type=str, default="experiments/evaluation/audio")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tags", type=str, default=None, help="comma-separated subset of tags to evaluate (for chunked eval)")
    parser.add_argument("--out_csv", type=str, default=None, help="explicit output CSV path (so chunks don't clobber each other)")
    parser.add_argument("--render_only", action="store_true", help="render prediction audio only; skip the perceptual KAD models (CPU-friendly)")
    args = parser.parse_args()

    if args.mode == "all":
        run_evaluation("nsynth", args)
        run_evaluation("synthetic", args)
    else:
        run_evaluation(args.mode, args)
