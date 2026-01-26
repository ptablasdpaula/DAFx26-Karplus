import torch
from torch import Tensor as T

def no_dc_burst(
        burst_length: int,
        seed: int = 42,
        device: torch.device = None
) -> torch.Tensor:
    """
    Generates a DC-removed noise burst

    :param burst_length: length of burst in samples
    :param seed: random seed
    :param device: torch device
    :return burst: [burst_length] ε [-1, 1]
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    burst = torch.rand(burst_length, generator=generator, device=device)
    burst = burst / burst.max()
    burst = (burst - 0.5) * 2
    return burst

def diff_excitation_onset(
        onset_probs: T,
        signal_length: int,
        f0: T,
        fs: int = 16000,
        noise_seed: int = 42,
        training: bool = True,
        threshold: float = 0.5,
) -> tuple[T, T]:
    """
    Create excitation with per-frame onset decisions

    :param onset_probs: [batch, num_frames] - logit per frame
    :param signal_length: total signal length in samples
    :param f0: [batch, num_frames] - fundamental frequency in Hz per frame
    :param fs: sample rate in Hz
    :param noise_seed: seed for deterministic noise generation
    :param training: if True, continuous gates [0,1]; if False, binary {0,1}
    :param threshold: threshold for binary gating at inference
    :return excitation: [batch, signal_length]; noise burst excitation signal
    :return onset_gates: [batch, num_frames]; onset gates (continuous if training=True)
    """
    assert onset_probs.dim() == 2 == f0.dim()
    assert onset_probs.shape == f0.shape
    assert torch.all((onset_probs >= 0.0) & (onset_probs <= 1.0))
    batch_size, num_frames = onset_probs.shape
    hop_length = signal_length // num_frames
    device = onset_probs.device

    excitation = torch.zeros(batch_size, signal_length, device=device)
    onset_gates = (onset_probs >= threshold).float() if not training else onset_probs

    for b in range(batch_size):
        for i in range(num_frames):
            if not training and onset_gates[b, i].item() == 0.0:
                continue
            f0_hz = f0[b, i].item()
            assert 0 < f0_hz <= fs / 2
            burst_len = int(fs / f0_hz)
            noise_burst = no_dc_burst(burst_len, seed=noise_seed, device=device)
            frame_start = i * hop_length
            frame_end = min(frame_start + burst_len, signal_length)
            actual_len = frame_end - frame_start
            excitation[b, frame_start:frame_end] += onset_gates[b, i] * noise_burst[:actual_len]

    return excitation, onset_gates