import librosa, torchcrepe, numpy as np, torch
from src.synths.constants import DEFAULT_FS, DEFAULT_CREPE_HOP_LENGTH, DEFAULT_ONSET_HOP_LENGTH, DEFAULT_ONSET_PAD_DURATION

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
        audio_tensor.float(), sr, hop_length=hop_length,
        fmin=40, fmax=2000, model='full', batch_size=256,
        device='cpu', return_periodicity=True,
    )

    f0_np = f0.squeeze().cpu().numpy()
    conf_np = confidence.squeeze().cpu().numpy()
    times = torch.arange(f0.shape[1]) * hop_length / sr
    times_np = times.cpu().numpy()

    return f0_np, conf_np, times_np