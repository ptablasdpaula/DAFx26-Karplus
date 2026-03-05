import numpy as np
import librosa

from src.synths.constants import DEFAULT_FS
from dtw import dtw

# @TODO: Precision, Recall, F1 on Onsets (Synthetic Only)

def compute_mean_cents_distance(pred_f0: np.ndarray, target_f0: np.ndarray) -> float:
    high_f0 = np.maximum(pred_f0, target_f0)
    low_f0 = np.minimum(pred_f0, target_f0)
    cents_distance = 1200 * np.log2(high_f0 / low_f0)
    return np.mean(cents_distance)

def compute_mfcc(x: np.ndarray, sample_rate: int = DEFAULT_FS) -> np.ndarray:
    window_length = int(0.05 * sample_rate)
    hop_length = int(0.01 * sample_rate)

    mfcc = librosa.feature.mfcc(
        y=x,
        sr=sample_rate,
        n_mfcc=20,
        n_fft=window_length,
        hop_length=hop_length,
        n_mels=128,
    )

    return mfcc

def compute_wmfcc(target: np.ndarray, pred: np.ndarray, sample_rate: int = DEFAULT_FS) -> float:
    target_mfcc = compute_mfcc(target, sample_rate=sample_rate)
    pred_mfcc = compute_mfcc(pred, sample_rate=sample_rate)

    target_mfcc = target_mfcc.reshape(-1, target_mfcc.shape[-1])
    pred_mfcc = pred_mfcc.reshape(-1, pred_mfcc.shape[-1])

    def l1(a, b):
        return np.mean(np.abs(a - b))

    dist = dtw(target_mfcc.T, pred_mfcc.T, dist_method=l1, distance_only=True)
    return dist.normalizedDistance

def compute_rms(target: np.ndarray, pred: np.ndarray, sample_rate: int = DEFAULT_FS) -> float:
    win_length = int(0.05 * sample_rate)
    hop_length = int(0.025 * sample_rate)

    target_rms = librosa.feature.rms(
        y=target.mean(axis=0), frame_length=win_length, hop_length=hop_length
    )
    pred_rms = librosa.feature.rms(
        y=pred.mean(axis=0), frame_length=win_length, hop_length=hop_length
    )

    target_norm = np.linalg.vector_norm(target_rms, axis=-1, ord=2)
    pred_norm = np.linalg.vector_norm(pred_rms, axis=-1, ord=2)

    cosine_sim = np.dot(target_rms[0], pred_rms[0]) / (target_norm * pred_norm)

    return cosine_sim.mean()