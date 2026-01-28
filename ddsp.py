import torch
from torch import Tensor as T
from philtorch.lpv import fir
from dsp import FS_MIN, LAGRANGE_ORDER, F0_MIN

def no_dc_burst(
        burst_length: int,
        seed: int = 42,
        device: torch.device = None
) -> T:
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
        fs: int = FS_MIN,
        noise_seed: int = 42,
        training: bool = True,
        threshold: float = 0.5,
) -> tuple[T, T]:
    """
    Create excitation with per-frame onset decisions

    :param onset_probs: [batch, num_frames] - logit per frame, expects ε [0, 1]
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


#================================================================================================
#                   DIFFERENTIABLE TIME-DOMAIN
#================================================================================================

def lagrange_coefficients(D: T, N: int = LAGRANGE_ORDER) -> T:
    """
    Compute Lagrange interpolation coefficients for fractional delay.

    :param D: Desired fractional delay [B, T]
    :param N: Order of interpolation
    :return: Coefficients h[..., 0], h[..., 1], ..., h[..., N] with shape [B, T, N+1]
    """
    device = D.device
    dtype = D.dtype

    # Indices for Lagrange formula: h_n = ∏_{k≠n} (D - k) / (n - k)
    n_idx = torch.arange(N + 1, device=device, dtype=dtype)  # [N+1]
    k_idx = torch.arange(N + 1, device=device, dtype=dtype)  # [N+1]

    # Expand dimensions for broadcasting
    D_exp = D.unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
    n_exp = n_idx.view(1, 1, N + 1, 1)     # [1, 1, N+1, 1]
    k_exp = k_idx.view(1, 1, 1, N + 1)     # [1, 1, 1, N+1]

    # Compute (D - k) for numerator and (n - k) for denominator
    numerator = D_exp - k_exp              # [B, T, N+1, N+1]
    denominator = n_exp - k_exp            # [1, 1, N+1, N+1]

    # Mask diagonal (where n == k)
    mask = ~torch.eye(N + 1, dtype=torch.bool, device=device)  # [N+1, N+1]

    # Compute ratio, setting diagonal to 1 (excluded from product)
    ratio = torch.where(mask, numerator / (denominator + 1e-10), torch.ones_like(numerator))

    # Product over k dimension
    h = ratio.prod(dim=-1)  # [B, T, N+1]

    return h

def lagrange_fractional_delay(
        L: T,
        N: int = LAGRANGE_ORDER
) -> tuple[T, T]:
    """
    Compute Lagrange interpolation coefficients for a given delay.
    Centers the fractional delay around N/2 for optimal frequency response.

    :param L: Total desired delay in samples [B, T]
    :param N: Order of Lagrange filter
    :return: (L_int, h) where L_int is integer delay [B, T] and h are coefficients [B, T, N+1]
    """
    offset = N // 2
    L_adjusted = L - offset
    L_int = torch.floor(L_adjusted).to(torch.long)
    D = L_adjusted - L_int
    D_centered = D + offset
    h = lagrange_coefficients(D_centered, N)
    return L_int, h

def td_all_zero_comb(
        x: T,
        L: T,
        fs: int = FS_MIN,
        lagrange_order: int = LAGRANGE_ORDER
) -> T:
    """
    All-zero comb filter with time-varying fractional delay: y(n) = x(n) - x(n - L)
    Uses Lagrange interpolation for fractional delays.

    :param x: Input signal [B, N]
    :param L: Delay in samples [B, N], can be fractional
    :param fs: Sample rate (used to determine max buffer size)
    :param lagrange_order: Order of Lagrange interpolator
    :return: Filtered signal [B, N]
    """
    assert x.shape == L.shape
    B, N = x.shape

    # Get integer and centered fractional delays
    L_int, h = lagrange_fractional_delay(L=L, N=lagrange_order)

    # Determine filter length needed
    max_delay = int(L_int.max().item()) + lagrange_order + 1
    M = max_delay

    # Build FIR coefficient matrix [B, N, M + 1]
    b = torch.zeros(B, N, M + 1, device=x.device, dtype=x.dtype)

    # Set direct path: b[:, :, 0] = 1
    b[:, :, 0] = 1.0

    # Scatter Lagrange weights at appropriate delays
    # For each batch and time sample, place -h at positions [L_int, L_int+1, ..., L_int+lagrange_order]
    batch_idx = torch.arange(B, device=x.device)[:, None].expand(B, N)
    time_idx = torch.arange(N, device=x.device)[None, :].expand(B, N)

    for k in range(lagrange_order + 1):
        delay_idx = (L_int + k)
        b[batch_idx, time_idx, delay_idx] -= h[:, :, k]

    # Apply time-varying FIR filter
    return fir(b, x)


def td_pluck_position_filter(
        x: T,
        f0: T,
        position: T,
        fs: int = FS_MIN,
        lagrange_order: int = LAGRANGE_ORDER
) -> T:
    """
    Simulate pluck position using an all-zero comb filter as per section
    "Simulation of a Moving Pick" in "Extensions of the Karplus-Strong Plucked-String Algorithm"
    by David A. Jaffe and Julius O. Smith

    Creates spectral notches at harmonics corresponding to pluck position.
    - position = 0.5 (midpoint): removes even harmonics
    - position = 0.1: removes every 10th harmonic
    - position = 1/N: approximates sul ponticello (bright, near bridge)

    :param x: Input signal (excitation) [B, N]
    :param f0: Fundamental frequency in Hz [B, N]
    :param position: Pluck position as fraction of string length, range [0, 1] [B, N]
    :param fs: Sample rate in Hz
    :param lagrange_order: Order of Lagrange interpolator
    :return: Filtered excitation signal [B, N]
    """
    assert x.shape == f0.shape == position.shape
    assert torch.all((position >= 0.0) & (position <= 1.0))

    L = fs / f0
    comb_L = L * position
    return td_all_zero_comb(x, comb_L, fs, lagrange_order)