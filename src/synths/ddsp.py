import torch
from torch import Tensor as T
import torch.nn.functional as F
from philtorch.lpv import fir, allpole, lfilter
from enum import Enum
import numpy as np

from src.synths.constants import (
    F0_MIN,
    DEFAULT_LAGRANGE_ORDER as LAGRANGE_ORDER,
    DEFAULT_FS as FS_MIN,
    DEFAULT_RND_SEED as RND_SEED,
    DEFAULT_N_FFT as N_FFT,
)

class Implementation(Enum):
    TIME_DOMAIN = "time_domain"
    FREQUENCY_SAMPLING = "frequency_sampling"


# =============================================================================
#                           SHARED UTILITIES
# =============================================================================

def lin_resample(x: T, signal_length: int) -> T:
    return F.interpolate(x.unsqueeze(1), size=signal_length, mode='linear', align_corners=False).squeeze(1)

def lin_resample_many(
        signal_length: int,
        **frame_params: T
) -> dict[str, T]:
    """
    Upsample multiple frame-rate parameters to sample-rate.

    :param signal_length: Target signal length in samples
    :param frame_params: Arbitrary number of [B, num_frames] tensors to upsample
    :return: Dictionary of upsampled [B, num_samples] tensors with same keys
    """
    upsampled = {}
    for key, tensor in frame_params.items():
        upsampled[key] = lin_resample(tensor, signal_length)
    return upsampled

# =============================================================================
#                           EXCITATION
# =============================================================================

def no_dc_burst(
        burst_length: int,
        seed: int = RND_SEED,
        device: torch.device = None
) -> T:
    """
    Generates a DC-removed noise burst

    :param burst_length: length of burst in samples
    :param seed: random seed
    :param device: torch device
    :return burst: [burst_length] ε [-1, 1]
    """
    np.random.seed(seed)
    burst = np.random.rand(burst_length)
    burst_torch = torch.from_numpy(burst).to(device)
    return ((burst_torch / burst_torch.max()) - 0.5) * 2

def excitation(
        times: T,
        exists: T,
        f0: T,
        signal_length: int,
        fs: int = FS_MIN,
        noise_seed: int = RND_SEED,
) -> T:
    """Generates excitation using frequency-domain phase rotation."""
    batch_size, max_events = times.shape
    device = times.device

    n_fft = signal_length
    n_bins = n_fft // 2 + 1

    bins = torch.arange(n_bins, device=device).float()
    omega = 2.0 * torch.pi * bins / n_fft

    total_X = torch.zeros((batch_size, n_bins), device=device, dtype=torch.complex64)

    for i in range(max_events):
        f0_hz = f0[:, i].mean().item()
        burst_len = int(fs / f0_hz)

        noise = no_dc_burst(burst_len, seed=noise_seed + i, device=device)
        noise_padded = F.pad(noise, (0, n_fft - burst_len))
        X_noise = torch.fft.rfft(noise_padded)  # [n_bins]

        t_samples = times[:, i:i + 1] * (signal_length - 1)
        phase_shift = torch.exp(-1j * omega.unsqueeze(0) * t_samples)

        exists_gate = (exists[:, i:i + 1] > 0.5).float().detach()

        event_X = X_noise.unsqueeze(0) * phase_shift * exists_gate
        total_X += event_X

    return torch.fft.irfft(total_X, n=n_fft)


# =============================================================================
#                           LAGRANGE INTERPOLATION
# =============================================================================

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

# =============================================================================
#                           PLUCK POSITION FILTER
# =============================================================================

def _time_all_zero_comb(x: T, L: T) -> T:
    """
    All-zero comb filter H(z) = 1 - z^(-L) with linear interpolation.
    
    :param x: [B, N] input signal
    :param L: [B, N] fractional delay in samples (≥ 0)
    :return: [B, N] filtered signal
    """
    B, N = x.shape

    L_int = torch.floor(L).to(torch.long)
    frac = L - L_int.to(L.dtype)

    max_delay = int(L.max().item()) + 2  # +1 for the extra tap, +1 for indexing

    b = torch.zeros(B, N, max_delay, device=x.device, dtype=x.dtype)
    b[:, :, 0] = 1.0

    batch_idx = torch.arange(B, device=x.device)[:, None].expand(B, N)
    time_idx = torch.arange(N, device=x.device)[None, :].expand(B, N)

    b[batch_idx, time_idx, L_int] -= (1.0 - frac)
    b[batch_idx, time_idx, L_int + 1] -= frac

    return fir(b, x)


def _freq_all_zero_comb(
        X: T,
        comb_L: T,  # [B, num_frames]
        n_fft: int = N_FFT,
) -> T:
    bins = torch.arange(0, n_fft // 2 + 1, device=X.device).view(1, 1, -1)
    angle = -torch.pi * comb_L.unsqueeze(-1) / (n_fft / 2)
    mod_sig = torch.polar(torch.ones_like(angle), angle)
    z_L = mod_sig ** bins
    H = 1.0 - z_L
    return X * H


def pluck_position_filter(
        x: T,
        f0: T,
        position: T,
        implementation: Implementation = Implementation.TIME_DOMAIN,
        fs: int = FS_MIN,
        n_fft: int = N_FFT,
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
    :param implementation: Implementation.TIME_DOMAIN, Implementation.FREQUENCY_SAMPLING
    :param fs: Sample rate in Hz
    :param lagrange_order: Order of Lagrange interpolator
    :param n_fft: length of the FFT
    :return: Filtered excitation signal [B, N]
    """
    if implementation == Implementation.FREQUENCY_SAMPLING:
        assert x.ndim == 3 and f0.ndim == 2 and position.ndim == 2
        assert x.shape[:2] == f0.shape == position.shape
    else:
        assert x.shape == f0.shape == position.shape

    assert torch.all((position >= 0.0) & (position <= 1.0))

    L = fs / f0
    comb_L = (L * position)

    if implementation == Implementation.TIME_DOMAIN:
        return _time_all_zero_comb(x, comb_L)
    elif implementation == Implementation.FREQUENCY_SAMPLING:
        return _freq_all_zero_comb(x, comb_L, n_fft)
    else:
        raise NotImplementedError

# =============================================================================
#                           DYNAMICS FILTER
# =============================================================================

def compute_dynamics_R(
        f0: T,
        dynamic_level: T,
        fs: int = FS_MIN
) -> T:
    """
    Compute dynamics filter coefficient R for a given pitch and dynamic level.

    :param f0: Fundamental frequency in Hz [B, N]
    :param dynamic_level: Bandwidth [B, N]
    :param fs: Sample rate in Hz
    :return: Filter coefficient R [B, N]
    """
    min_bw = F0_MIN
    max_bw = fs / 2.0

    bw_hz = min_bw * (max_bw / min_bw) ** dynamic_level
    fm = torch.sqrt(torch.tensor(min_bw * max_bw, device=f0.device, dtype=f0.dtype))
    Ts = 1.0 / fs

    R_L = torch.exp(-bw_hz * torch.pi * Ts)

    # Compute G_L = (1 - R_L) / |1 - R_L * exp(-j * 2π * fm * Ts)|
    exp_term = torch.exp(-1j * 2 * torch.pi * fm * Ts)
    denominator = 1 - R_L * exp_term
    G_L = (1 - R_L) / torch.abs(denominator)

    # Left side: (1 - G_L² * cos(2π * f0 * Ts)) / (1 - G_L²)
    cos_term = torch.cos(2 * torch.pi * f0 * Ts)
    left_side_num = 1 - G_L ** 2 * cos_term
    left_side_den = 1 - G_L ** 2
    left_side = left_side_num / left_side_den

    # Right side: 2 * G_L * sin(π * f0 * Ts) * sqrt(1 - G_L² * cos²(π * f0 * Ts)) / (1 - G_L²)
    sin_term = torch.sin(torch.pi * f0 * Ts)
    cos_half_term = torch.cos(torch.pi * f0 * Ts)
    right_side_outside = 2 * G_L * sin_term
    right_side_num = torch.sqrt(1 - G_L ** 2 * cos_half_term ** 2)
    right_side_den = 1 - G_L ** 2
    right_side = right_side_outside * (right_side_num / right_side_den)

    # Choose R_plus or R_minus based on which has smaller absolute value
    R_plus = left_side + right_side
    R_minus = left_side - right_side

    R = torch.where(torch.abs(R_plus) < 1.0, R_plus, R_minus)

    return R


def _freq_dynamics_filter(X: T, R: T, n_fft: int) -> T:
    device = X.device

    # Feedforward: B = (1-R) * X
    b0 = (1.0 - R).unsqueeze(-1)  # [B, num_frames, 1]
    B = b0 * X  # [B, num_frames, n_bins]

    # Feedback: H_fb = R * z^(-1)
    # z^(-1) in frequency domain is exp(-j*omega) where omega = 2*pi*k/(n_fft)
    bins = torch.arange(0, n_fft // 2 + 1, device=device)
    omega = 2.0 * torch.pi * bins / n_fft  # [n_bins]
    z_inv = torch.exp(-1j * omega)  # [n_bins] - one sample delay

    # Identity matrix [B, num_frames, n_bins]
    I = torch.ones_like(X)

    # A = I - R * z^(-1)
    a1 = R.unsqueeze(-1)  # [B, num_frames, 1]
    A = I - a1 * z_inv.unsqueeze(0).unsqueeze(0)  # [B, num_frames, n_bins]

    # Solve: Y = A^(-1) * B
    Y = B / A

    return Y


def _time_dynamics_filter(x: T, R: T) -> T:
    x_eff = (1.0 - R) * x
    a = -R.unsqueeze(-1)
    return allpole(a, x_eff)


def dynamics_filter(
        x: T,
        f0: T,
        dynamic_level: T,
        implementation: Implementation = Implementation.TIME_DOMAIN,
        n_fft: int = N_FFT,
        fs: int = FS_MIN,
) -> T:
    """
    Apply dynamics filter to excitation with time-varying dynamic level.
    :param x: Input signal (excitation) [B, N]
    :param f0: Fundamental frequency in Hz [B, N]
    :param dynamic_level: 0.0 = soft/dark, 1.0 = loud/bright
    :param implementation: Implementation.TIME_DOMAIN or Implementation.FREQUENCY_SAMPLING
    :param n_fft: length of the FFT used in frequency sampling
    :param fs: Sample rate in Hz
    """
    if implementation == Implementation.FREQUENCY_SAMPLING:
        assert x.ndim == 3 and f0.ndim == 2 and dynamic_level.ndim == 2
        assert x.shape[:2] == f0.shape == dynamic_level.shape
    else:
        assert x.shape == f0.shape == dynamic_level.shape

    assert torch.all((dynamic_level >= 0.0) & (dynamic_level <= 1.0))

    if implementation is Implementation.FREQUENCY_SAMPLING:
        dynamic_level = dynamic_level.clamp(min=0.01)

    R = compute_dynamics_R(f0, dynamic_level, fs)

    if implementation == Implementation.FREQUENCY_SAMPLING:
        return _freq_dynamics_filter(x, R, n_fft)
    else:
        return _time_dynamics_filter(x, R)


# =============================================================================
#                           KARPLUS-STRONG
# =============================================================================

def one_pole_phase_delay(f0: T, a1: T, fs: int) -> T:
    """
    Compute phase delay of one-pole loop filter at fundamental frequency.
    :param f0: Fundamental frequency in Hz [B, N]
    :param a1: Loop filter pole coefficient [B, N]
    :param fs: Sample rate in Hz
    :return: Phase delay in samples [B, N]
    """
    omega0 = 2.0 * torch.pi * f0 / fs
    denom_real = 1.0 - a1 * torch.cos(omega0)
    denom_imag = a1 * torch.sin(omega0)
    phase = torch.atan2(denom_imag, denom_real)
    phase_delay = -phase / omega0
    return phase_delay

def _time_karplus_strong(
        x: T, L: T, a1: T, g: T, lagrange_order: int,
) -> T:
    B, N = x.shape
    b0 = g * (1.0 - a1)  # [B, N]

    L_int, h = lagrange_fractional_delay(L, lagrange_order)

    max_L = int(L_int.max().item())
    max_order = max_L + lagrange_order

    # --- Denominator: a[k] coefficients [B, N, max_order] ---
    a_coeffs = torch.zeros(B, N, max_order, device=x.device, dtype=x.dtype)
    a_coeffs[:, :, 0] = -a1  # tap at delay 1

    batch_idx = torch.arange(B, device=x.device)[:, None].expand(B, N)
    time_idx  = torch.arange(N, device=x.device)[None, :].expand(B, N)

    for n in range(lagrange_order + 1):
        idx = (L_int + n - 1).clamp(0, max_order - 1)
        a_coeffs[batch_idx, time_idx, idx] += -b0 * h[..., n]

    # --- Numerator: [1, -a1] ---
    b_coeffs = torch.stack([torch.ones_like(a1), -a1], dim=-1)  # [B, N, 2]

    return lfilter(b_coeffs, a_coeffs, x, form='df2', backend='torchlpc')


def _freq_karplus_strong(
        X: T,  # [B, num_frames, n_bins]
        L: T,  # [B, num_frames] - delay in samples
        a1: T,  # [B, num_frames]
        g: T,  # [B, num_frames]
        n_fft: int,
) -> T:
    """Apply Karplus-Strong using closed-loop transfer function."""
    device = X.device
    n_bins = n_fft // 2 + 1

    # Frequency bins [0, 1, 2, ..., n_fft//2]
    bins = torch.arange(0, n_bins, device=device).float()

    # z^(-1) at each bin: exp(-j * 2π * k / n_fft)
    # This is the unit delay phasor
    omega = 2.0 * torch.pi * bins / n_fft  # [n_bins]
    z_inv = torch.exp(-1j * omega)  # [n_bins]

    # z^(-L) - fractional delay of L samples
    # angle = -π * L / (n_fft/2) per bin, then raised to bin power
    angle = -omega.unsqueeze(0).unsqueeze(0) * L.unsqueeze(-1)  # [B, num_frames, n_bins]
    z_minus_L = torch.exp(1j * angle)  # [B, num_frames, n_bins] - this is z^(-L)

    # Loop filter: H_loop(z) = g * (1 - a1) / (1 - a1 * z^(-1))
    b0 = (g * (1.0 - a1)).unsqueeze(-1)  # [B, num_frames, 1]
    a1_exp = a1.unsqueeze(-1)  # [B, num_frames, 1]

    # Denominator of loop filter: 1 - a1 * z^(-1)
    H_loop_denom = 1.0 - a1_exp * z_inv.unsqueeze(0).unsqueeze(0)  # [B, num_frames, n_bins]
    H_loop = b0 / H_loop_denom  # [B, num_frames, n_bins]

    # Closed-loop transfer function for KS:
    # Y(z) = X(z) / (1 - H_loop(z) * z^(-L))
    denominator = 1.0 - H_loop * z_minus_L  # [B, num_frames, n_bins]
    Y = X / denominator
    return Y

def karplus_strong(
        x: T,
        f0: T,
        a1: T,
        g: T,
        implementation: Implementation = Implementation.TIME_DOMAIN,
        fs: int = FS_MIN,
        lagrange_order: int = LAGRANGE_ORDER,
        n_fft: int = N_FFT,
) -> T:
    """
    Karplus-Strong string synthesis with fractional delay and loop filtering.

    :param x: Input signal
    :param f0: Fundamental Frequency (Hz)
    :param a1: Coefficient Value of DC normalised, first-order IIR loop-filter
    :param g: Coefficient of DC normalised
    :param implementation: Implementation.TIME_DOMAIN, Implementation.FREQUENCY_SAMPLING
    :param fs: Sampling frequency
    :param lagrange_order: Lagrange order
    :param iir_truncation: IIR truncation of loop filter (TIME_DOMAIN only)
    :param n_fft: FFT size (in samples)
    """
    if implementation == Implementation.FREQUENCY_SAMPLING:
        assert x.ndim == 3 and f0.ndim == 2 and a1.ndim == 2 and g.ndim == 2
        assert x.shape[:2] == f0.shape == a1.shape == g.shape
    else:
        assert x.shape == f0.shape == a1.shape == g.shape

    assert torch.all((a1 >= 0.0) & (a1 <= 1.0))
    assert torch.all((g >= 0.0) & (g <= 1.0))

    L = fs / f0
    phase_delay = one_pole_phase_delay(f0, a1, fs)
    L_corrected = L + phase_delay

    if implementation == Implementation.TIME_DOMAIN:
        return _time_karplus_strong(x, L_corrected, a1, g, lagrange_order)
    elif implementation == Implementation.FREQUENCY_SAMPLING:
        eps = 1e-7
        a1_stable = torch.clamp(a1, 0.0, 1.0 - eps)
        g_stable = torch.clamp(g, 0.0, 1.0 - eps)
        return _freq_karplus_strong(x, L_corrected, a1_stable, g_stable, n_fft)
    else:
        raise NotImplementedError
