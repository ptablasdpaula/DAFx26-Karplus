import argparse
import os
from pathlib import Path

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
from src.data.nsynth.nsynth_guitar_dataset import NsynthGuitarDataset
from src.data.synthetic_dataset import SyntheticDataset
from src.losses import MultiScaleSpectralLoss, SOT2048Loss
from src.metrics import compute_wmfcc, compute_rms, compute_mean_cents_distance

# ── FAITHFUL KERNEL AUDIO DISTANCE (KAD) IMPLEMENTATION ──────────────────────
# This replicates exactly the math from kadtk/mmd.py to avoid dependency hell.

def compute_kad_faithful(real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
    """
    Computes Kernel Audio Distance (unbiased MMD^2) with RBF kernel 
    and Median Heuristic for bandwidth selection.
    """
    x = real_features
    y = fake_features
    n = x.size(0)
    m = y.size(0)

    # 1. Compute pairwise distances for the RBF kernel
    # Combined for median heuristic
    z = torch.cat([x, y], dim=0)
    dist_z = torch.cdist(z, z, p=2)
    
    # Median Heuristic for sigma (bandwidth)
    sigma = torch.median(dist_z[dist_z > 0])
    gamma = 1.0 / (2.0 * sigma**2 + 1e-8)

    # 2. Compute Kernels
    def rbf_kernel(mat1, mat2):
        dists = torch.cdist(mat1, mat2, p=2)
        return torch.exp(-gamma * dists**2)

    K_xx = rbf_kernel(x, x)
    K_yy = rbf_kernel(y, y)
    K_xy = rbf_kernel(x, y)

    # 3. Unbiased MMD^2 Estimator
    mmd_sq = (
        (K_xx.sum() - torch.trace(K_xx)) / (n * (n - 1)) +
        (K_yy.sum() - torch.trace(K_yy)) / (m * (m - 1)) -
        2 * K_xy.mean()
    )
    
    # Standard KAD scale factor
    return mmd_sq.item() * 1000

# ── MODEL LOADING HELPERS ──────────────────────────────────────────────────

IMPL_MAP = {"time_domain": Implementation.TIME_DOMAIN, "frequency_sampling": Implementation.FREQUENCY_SAMPLING}

def _build_model_from_cfg(cfg) -> SoundMatchingModel:
    if cfg.model.decoder == "ks":
        synth = Synth(SynthConfig(
            num_samples=cfg.num_audio_samples, fs=cfg.fs,
            implementation=IMPL_MAP[cfg.model.implementation],
            lagrange_order=cfg.model.synth.lagrange_order, n_fft=cfg.model.synth.n_fft
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

def _load_model(tag, ckpt_dir, device):
    cfg = OmegaConf.load(ckpt_dir / f"{tag}_config.yaml")
    model = _build_model_from_cfg(cfg)
    ckpt = torch.load(ckpt_dir / f"{tag}_best.ckpt", map_location=device, weights_only=False)
    state_dict = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state_dict)
    return model.to(device).eval()

def _tag_to_rel_path(tag: str) -> str:
    # Converts 'Nsynth_Free_Freq_Super' to 'free/freq/super'
    return "/".join(tag.lower().split("_")[1:])

# ── MAIN EVALUATION LOOP ────────────────────────────────────────────────────

def run_evaluation(mode="nsynth"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--out_dir", type=str, default="experiments/evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    ckpt_dir = Path(args.ckpt_dir)
    audio_root = Path(args.out_dir) / "audio" / mode
    
    # 1. Load Perceptual Models (HF Versions)
    encodec = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)
    clap_processor = ClapProcessor.from_pretrained("laion/clap-htat-unfused")
    clap_model = ClapModel.from_pretrained("laion/clap-htat-unfused").to(device)
    
    # 2. Setup Data
    if mode == "nsynth":
        ds = NsynthGuitarDataset(nsynth_root="src/data/nsynth", split="test")
        tags = ["Nsynth_Free_Freq_Super", "Nsynth_Free_Freq_Comb", "Nsynth_Free_Time_Super", 
                "Nsynth_Free_Time_Comb", "Nsynth_Det_Freq_Spec", "Nsynth_Det_Time_Spec", "Nsynth_Det_HpN_Spec"]
    else:
        ds = SyntheticDataset(num_samples_per_epoch=250, fs=16000, random_seed=77777)
        tags = ["Synth_Free_Freq_Super", "Synth_Free_Freq_Comb", "Synth_Free_Time_Super", 
                "Synth_Free_Time_Comb", "Synth_Det_Freq_Spec", "Synth_Det_Time_Spec"]

    loader = DataLoader(ds, batch_size=16, shuffle=False)
    mss_fn = MultiScaleSpectralLoss().to(device)
    sot_fn = SOT2048Loss(sample_rate=16000).to(device)

    # 3. Save Target Audio Once
    target_path = audio_root / "target"
    target_path.mkdir(parents=True, exist_ok=True)
    targets_collected = []
    
    print(f"--- Starting {mode.upper()} Evaluation ---")
    
    all_results = {}
    for tag in tags:
        if not (ckpt_dir / f"{tag}_config.yaml").exists(): continue
        
        print(f"\nProcessing {tag}...")
        model = _load_model(tag, ckpt_dir, device)
        pred_path = audio_root / "pred" / _tag_to_rel_path(tag)
        pred_path.mkdir(parents=True, exist_ok=True)
        
        metrics = {"mss": [], "sot": [], "wmfcc": [], "rms": [], "encodec_mse": []}
        all_clap_tgt, all_clap_pred = [], []
        
        idx = 1
        for batch in tqdm(loader):
            tgt_audio = batch["audio"].to(device)
            detected = {k: v.to(device) for k, v in batch["detected"].items()} if "detected" in batch else None
            
            with torch.no_grad():
                raw = model.encoder(tgt_audio)
                params = model.decoder.activate(raw, detected=detected)
                pred_audio, _ = model.decoder.oracle_synth(params)
                
                # Perceptual Features
                # Encodec (24kHz)
                t24 = torchaudio.functional.resample(tgt_audio, 16000, 24000).unsqueeze(1)
                p24 = torchaudio.functional.resample(pred_audio, 16000, 24000).unsqueeze(1)
                e_tgt = encodec.encode(t24).audio_codes.float()
                e_pred = encodec.encode(p24).audio_codes.float()
                metrics["encodec_mse"].append(torch.nn.functional.mse_loss(e_pred, e_tgt).item())
                
                # CLAP Embeddings (48kHz)
                t48 = torchaudio.functional.resample(tgt_audio, 16000, 48000).cpu().numpy()
                p48 = torchaudio.functional.resample(pred_audio, 16000, 48000).cpu().numpy()
                c_tgt = clap_model.get_audio_features(**clap_processor(audios=list(t48), return_tensors="pt", sampling_rate=48000).to(device))
                c_pred = clap_model.get_audio_features(**clap_processor(audios=list(p48), return_tensors="pt", sampling_rate=48000).to(device))
                all_clap_tgt.append(c_tgt); all_clap_pred.append(c_pred)
                
                # Audio Metrics
                metrics["mss"].append(mss_fn(pred_audio, tgt_audio).item())
                metrics["sot"].append(sot_fn(pred_audio, tgt_audio).item())

            # Save Audio
            for b in range(tgt_audio.shape[0]):
                if tag == tags[0]: # Only save targets on the first model pass
                    sf.write(target_path / f"{idx:03d}.wav", tgt_audio[b].cpu().numpy(), 16000)
                sf.write(pred_path / f"{idx:03d}.wav", pred_audio[b].cpu().numpy(), 16000)
                idx += 1

        # Global Dataset-Level Metrics (KAD)
        kad_score = compute_kad_faithful(torch.cat(all_clap_tgt), torch.cat(all_clap_pred))
        
        all_results[tag] = {k: np.mean(v) for k, v in metrics.items()}
        all_results[tag]["CLAP_KAD"] = kad_score

    pd.DataFrame(all_results).T.to_csv(Path(args.out_dir) / f"{mode}_results.csv")
    print(f"\nDone! Results saved to {args.out_dir}/{mode}_results.csv")

if __name__ == "__main__":
    run_evaluation(mode="nsynth")
    run_evaluation(mode="synthetic")