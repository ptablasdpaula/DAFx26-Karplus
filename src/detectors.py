import librosa, torchcrepe, numpy as np, torch
from src.synths.constants import (
    DEFAULT_FS,
    DEFAULT_CREPE_HOP_LENGTH,
    DEFAULT_ONSET_HOP_LENGTH,
    DEFAULT_ONSET_PAD_DURATION, DEFAULT_LOUDNESS_STFT
)

from src.synths.param_registry import F0_MIN_HZ, F0_MAX_HZ

import torch.nn.functional as F

def run_detectors_on_batch(
        audio_batch: torch.Tensor,
        sr: int = DEFAULT_FS,
        num_frames: int = 250
) -> dict[str, torch.Tensor]:
    """Safely runs CREPE (on GPU if available) and Onsets (CPU) on a batch."""
    device = audio_batch.device
    B = audio_batch.shape[0]

    # 1. GPU-accelerated CREPE (Fast and Memory Safe!)
    f0_batch, _ = torchcrepe.predict(
        audio_batch.float(),
        sr,
        hop_length=DEFAULT_CREPE_HOP_LENGTH,
        fmin=F0_MIN_HZ, fmax=F0_MAX_HZ,
        model='tiny',
        batch_size=512,  # Safe limit to prevent VRAM spikes
        device=device,
        return_periodicity=True
    )

    # 2. Sequential CPU Onsets (Safe on the main thread)
    # Move audio to CPU numpy just for librosa
    audio_np = audio_batch.detach().cpu().numpy()
    onset_mask = torch.zeros((B, num_frames), device=device, dtype=torch.float32)
    duration_s = audio_batch.shape[1] / sr
    frame_hop_s = duration_s / num_frames

    for b in range(B):
        onsets = detect_onsets(audio_np[b], sr=sr)
        if len(onsets) > 0:
            idx = np.round(onsets / frame_hop_s).astype(int)
            idx = np.clip(idx, 0, num_frames - 1)
            onset_mask[b, idx] = 1.0

    # 3. Resample F0 to exactly `num_frames`
    f0_t = f0_batch[:, :-1]  # Slice to exactly match NSynth preprocessing
    if f0_t.shape[1] > 1:
        f0_res = F.interpolate(
            f0_t.unsqueeze(1),
            size=num_frames, mode='linear', align_corners=True
        ).squeeze(1)
    else:
        f0_res = torch.full((B, num_frames), 220.0, device=device)

    return {"onsets": onset_mask, "f0": f0_res}

def detect_onsets(audio_np, sr=DEFAULT_FS, hop_length=DEFAULT_ONSET_HOP_LENGTH, pad_duration=DEFAULT_ONSET_PAD_DURATION):
    pad_samples = int(pad_duration * sr)
    audio_padded = np.pad(audio_np, (pad_samples, 0), mode='constant', constant_values=0)

    onset_frames = librosa.onset.onset_detect(
        y=audio_padded,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,
        units='frames',
        onset_envelope=None,  # Spectral flux
    )

    duration = len(audio_np) / sr
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length) - pad_duration
    onset_times = onset_times[onset_times >= -0.05]
    onset_times = np.clip(onset_times, 0, duration)

    return onset_times

def detect_f0(audio_tensor, sr=DEFAULT_FS, hop_length=DEFAULT_CREPE_HOP_LENGTH):
    f0, confidence = torchcrepe.predict(
        audio_tensor.float(),
        sr,
        hop_length=hop_length,
        fmin=F0_MIN_HZ,
        fmax=F0_MAX_HZ,
        model='full',
        batch_size=256,
        device='cpu',
        return_periodicity=True,
    )

    f0_np = f0.squeeze().cpu().numpy()
    conf_np = confidence.squeeze().cpu().numpy()
    times = torch.arange(f0.shape[1]) * hop_length / sr
    times_np = times.cpu().numpy()

    return f0_np, conf_np, times_np

def detect_loudness(signal, sampling_rate, n_fft=2048):
    S = librosa.stft(
        signal,
        n_fft=n_fft,
        hop_length=DEFAULT_LOUDNESS_STFT,
        win_length=n_fft,
        center=True,
    )
    S = np.log(abs(S) + 1e-7)
    f = librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
    a_weight = librosa.A_weighting(f)

    S = S + a_weight.reshape(-1, 1)
    S = np.mean(S, 0)[..., :-1]
    return S