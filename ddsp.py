import torch
from torch import Tensor as T
import torch.nn.functional as F
from philtorch.lpv import fir, allpole
from torchlpc import sample_wise_lpc
from enum import Enum

LAGRANGE_ORDER = 5
F0_MIN = 20
FS_MIN = 16000
RND_SEED = 42
IIR_TRUNCATION = 20
ONSET_THRESHOLD = 0.5
N_FFT = 2048

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
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    burst = torch.rand(burst_length, generator=generator, device=device)
    burst = burst / burst.max()
    burst = (burst - 0.5) * 2
    return burst

def excitation_onset(
        onset_probs: T,
        signal_length: int,
        f0: T,
        fs: int = FS_MIN,
        noise_seed: int = RND_SEED,
        training: bool = True,
        threshold: float = ONSET_THRESHOLD,
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
    :return excitation: [batch, num_samples]; noise burst excitation signal
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

def _time_all_zero_comb(x: T, L: T, lagrange_order: int) -> T:
    B, N = x.shape

    L_int, h = lagrange_fractional_delay(L=L, N=lagrange_order)
    max_delay = int(L.max().item()) + lagrange_order + 1

    b = torch.zeros(B, N, max_delay + 1, device=x.device, dtype=x.dtype)
    b[:, :, 0] = 1.0

    batch_idx = torch.arange(B, device=x.device)[:, None].expand(B, N)
    time_idx = torch.arange(N, device=x.device)[None, :].expand(B, N)

    for k in range(lagrange_order + 1):
        delay_idx = L_int + k
        b[batch_idx, time_idx, delay_idx] -= h[:, :, k]

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
        lagrange_order: int = LAGRANGE_ORDER,
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
    comb_L = 1.0 + position * (L - 1)

    if implementation == Implementation.TIME_DOMAIN:
        return _time_all_zero_comb(x, comb_L, lagrange_order)
    elif implementation == Implementation.FREQUENCY_SAMPLING:
        return _freq_all_zero_comb(x, comb_L, n_fft)
    else:
        raise NotImplementedError

# =============================================================================
#                           DYNAMICS FILTER
# =============================================================================

def compute_dynamics_R(
        f0: T,
        bw: T,
        fs: int = FS_MIN
) -> T:
    """
    Compute dynamics filter coefficient R for a given pitch and dynamic level.

    :param f0: Fundamental frequency in Hz [B, N]
    :param bw: Bandwidth (dynamic level) [B, N]
    :param fs: Sample rate in Hz
    :return: Filter coefficient R [B, N]
    """
    bw_scaled = bw * (fs / 2.0)
    fm = torch.sqrt(torch.tensor(F0_MIN * (fs / 2.0), device=f0.device, dtype=f0.dtype))
    Ts = 1.0 / fs

    R_L = torch.exp(-bw_scaled * torch.pi * Ts)

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

def dynamics_filter(
        x: T,
        f0: T,
        dynamic_level: T,
        fs: int = FS_MIN,
) -> T:
    """
    Apply dynamics filter to excitation with time-varying dynamic level.
    :param x: Input signal (excitation) [B, N]
    :param f0: Fundamental frequency in Hz [B, N]
    :param dynamic_level: 0.0 = soft/dark, 1.0 = loud/bright
    :param fs: Sample rate in Hz
    """
    assert x.shape == f0.shape == dynamic_level.shape, f"{x.shape}, {f0.shape}, {dynamic_level.shape}"
    assert torch.all((dynamic_level >= 0.0) & (dynamic_level <= 1.0))
    R = compute_dynamics_R(f0, dynamic_level, fs)
    x_eff = (1.0 - R) * x
    a = -R.unsqueeze(-1)
    return allpole(a, x_eff)

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
        x: T, L: T, a1: T, g: T, lagrange_order: int, iir_truncation: int
) -> T:
    B, N = x.shape
    K = iir_truncation

    b0 = g * (1.0 - a1)

    L_int, weights = lagrange_fractional_delay(L, lagrange_order)

    a1_padded = F.pad(a1, (K - 1, 0), value=1.0)
    b0_padded = F.pad(b0, (K - 1, 0), value=0.0)

    a1_windows = a1_padded.unfold(1, K, 1).flip(-1)
    b0_windows = b0_padded.unfold(1, K, 1).flip(-1)

    cumprods = torch.cumprod(a1_windows, dim=-1)
    cumprods_shifted = torch.cat([
        torch.ones(B, N, 1, device=x.device, dtype=x.dtype),
        cumprods[:, :, :-1]
    ], dim=-1)

    iir_coeffs = b0_windows * cumprods_shifted

    L_len = lagrange_order + 1
    total_len = L_len + K - 1

    weights_expanded = weights.unsqueeze(-1)
    iir_coeffs_expanded = iir_coeffs.unsqueeze(2)
    conv_product = weights_expanded * iir_coeffs_expanded

    iir_expanded = torch.zeros(B, N, total_len, device=x.device, dtype=x.dtype)
    for i in range(L_len):
        iir_expanded[:, :, i:i + K] += conv_product[:, :, i, :]

    max_delay = int(L_int.max().item()) + total_len
    A = torch.zeros(B, N, max_delay, device=x.device, dtype=x.dtype)

    batch_idx = torch.arange(B, device=x.device)[:, None].expand(B, N)
    time_idx = torch.arange(N, device=x.device)[None, :].expand(B, N)

    for k in range(total_len):
        delay_idx = (L_int + k - 1).clamp(0, max_delay - 1)
        A[batch_idx, time_idx, delay_idx] -= iir_expanded[..., k]

    return sample_wise_lpc(x, A)


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
        iir_truncation: int = IIR_TRUNCATION,
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
        return _time_karplus_strong(x, L_corrected, a1, g, lagrange_order, iir_truncation)
    elif implementation == Implementation.FREQUENCY_SAMPLING:
        eps = 1e-7
        a1_stable = torch.clamp(a1, 0.0, 1.0 - eps)
        g_stable = torch.clamp(g, 0.0, 1.0 - eps)
        return _freq_karplus_strong(x, L_corrected, a1_stable, g_stable, n_fft)
    else:
        raise NotImplementedError

# =============================================================================
#                           PHYSICAL MODEL
# =============================================================================

def physical_model(
        onset_probs: T,     # [B, num_frames]
        f0: T,              # [B, num_frames]
        pluck_position: T,  # [B, num_frames]
        burst_gain: T,      # [B, num_frames]
        dynamic_level: T,   # [B, num_frames]
        a1: T,              # [B, num_frames]
        decay: T,           # [B, num_frames]
        num_samples: int,
        use_freq_pluck: bool = False,
        use_freq_ksa: bool = False,
        fs: int = FS_MIN,
        n_fft: int = N_FFT,
        window = None,
        hop_length: int = None,
        lagrange_order: int = LAGRANGE_ORDER,
        iir_truncation: int = IIR_TRUNCATION,
        random_seed: int = RND_SEED,
        training: bool = True,
        onset_threshold: float = ONSET_THRESHOLD,
) -> T:
    x, _ = excitation_onset(
        onset_probs=onset_probs,
        signal_length=num_samples,
        f0=f0,
        fs=fs,
        noise_seed=random_seed,
        training=training,
        threshold=onset_threshold
    )

    p = {
        'f0': f0,
        'pluck_position': pluck_position,
        'burst_gain': burst_gain,
        'dynamic_level': dynamic_level,
        'a1': a1,
        'decay': decay,
    }

    p_time = lin_resample_many(signal_length=num_samples, **p)

    if use_freq_pluck or use_freq_ksa:
        device = x.device
        # Convert to freq
        if hop_length is None:
            hop_length = n_fft // 4  # 75% overlap TODO: find out best-performing overlap
        if window is None:
            window = torch.ones(n_fft, device=device)

        X = torch.stft(x, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        num_stft_frames = X.shape[-1]
        X = X.permute(0, 2, 1)  # [B, num_stft_frames, n_bins]
        stft_p = lin_resample_many(signal_length=num_stft_frames, **p)

    x = x * p_time['burst_gain']
    x = dynamics_filter(
        x=x,
        f0= p_time['f0'],
        dynamic_level= p_time['dynamic_level'],
        fs=fs,
    )

    if use_freq_pluck:
        # do freq_pluck
        X = pluck_position_filter(
            x=X,
            f0=stft_p['f0'],
            position=stft_p['pluck_position'],
            implementation=Implementation.FREQUENCY_SAMPLING,
            fs=fs,
            n_fft=n_fft,
            lagrange_order=lagrange_order
        )

        if not use_freq_ksa:
            # convert back to time
            X = X.permute(0, 2, 1)  # [B, n_bins, num_stft_frames]
            x = torch.istft(X, n_fft=n_fft, hop_length=hop_length, window=window, length=num_samples)
    else:
        # do time-domain pluck
        x = pluck_position_filter(
            x=x,
            f0=p_time['f0'],
            position=p_time['pluck_position'],
            implementation=Implementation.TIME_DOMAIN,
            fs=fs,
            n_fft=n_fft,
            lagrange_order=lagrange_order
        )

    if use_freq_ksa:
        # do freq-domain ksa
        X = karplus_strong(
            x=X,
            f0=stft_p['f0'],
            a1=stft_p['a1'],
            g=stft_p['decay'],
            implementation=Implementation.FREQUENCY_SAMPLING,
            fs=fs,
            n_fft=n_fft,
            lagrange_order=lagrange_order,
            iir_truncation=iir_truncation
        )

        # convert to time
        X = X.permute(0, 2, 1)  # [B, n_bins, num_stft_frames]
        x = torch.istft(X, n_fft=n_fft, hop_length=hop_length, window=window, length=num_samples)
    else:
        # do time-domain pluck
        x = karplus_strong(
            x=x,
            f0=p_time['f0'],
            a1=p_time['a1'],
            g=p_time['decay'],
            implementation=Implementation.TIME_DOMAIN,
            fs=fs,
            n_fft=n_fft,
            lagrange_order=lagrange_order,
            iir_truncation=iir_truncation
        )

    return x



# =============================================================================
#                           TESTS
# =============================================================================

if __name__ == "__main__":
    duration = 1
    sample_rates = [16000, 32000, 44100]
    num_frames = 100

    defaults = {
        'f0': 220.0,
        'pluck_position': 0.5,
        'burst_gain': 0.5,
        'dynamic_level': 0.5,
        'a1': 0.5,
        'decay': 0.995,
    }

    sweeps = {
        'f0': (55.0, 3520.0),
        'pluck_position': (0.0, 1.0),
        'burst_gain': (0.0, 1.0),
        'dynamic_level': (0.0, 1.0),
        'a1': (0.0, 1.0),
        'decay': (0.0, 1.0),
    }

    all_passed = True
    test_count = 0

    # All four combinations of frequency/time-domain implementations
    impl_combinations = [
        (False, False, "TD pluck + TD KS"),
        (False, True, "TD pluck + FD KS"),
        (True, False, "FD pluck + TD KS"),
        (True, True, "FD pluck + FD KS"),
    ]

    for use_freq_pluck, use_freq_ksa, impl_name in impl_combinations:
        for fs in sample_rates:
            num_samples = int(fs * duration)

            print(f"\n{'=' * 60}")
            print(f"Testing [{impl_name}] at fs={fs}Hz ({num_samples} samples)")
            print(f"{'=' * 60}")

            onset_probs = torch.zeros(1, num_frames)
            onset_probs[0, [0, 25, 50, 75]] = 1.0

            for param_name, (min_val, max_val) in sweeps.items():
                params = {k: torch.full((1, num_frames), v) for k, v in defaults.items()}
                params[param_name] = torch.linspace(min_val, max_val, num_frames).unsqueeze(0)

                y = physical_model(
                    onset_probs=onset_probs,
                    num_samples=num_samples,
                    use_freq_pluck=use_freq_pluck,
                    use_freq_ksa=use_freq_ksa,
                    fs=fs,
                    **params
                )

                if torch.isnan(y).any() or torch.isinf(y).any():
                    print(f"  FAIL: {param_name} sweep")
                    all_passed = False
                else:
                    print(f"  PASS: {param_name} sweep")
                test_count += 1

            # All params sweeping
            params = {k: torch.linspace(*v, num_frames).unsqueeze(0) for k, v in sweeps.items()}
            y = physical_model(
                onset_probs=onset_probs,
                num_samples=num_samples,
                use_freq_pluck=use_freq_pluck,
                use_freq_ksa=use_freq_ksa,
                fs=fs,
                **params
            )

            if torch.isnan(y).any() or torch.isinf(y).any():
                print(f"  FAIL: all parameters sweeping")
                all_passed = False
            else:
                print(f"  PASS: all parameters sweeping")
            test_count += 1

    print(f"\n{'=' * 60}")
    print(f"✓ All {test_count} tests passed!" if all_passed else "✗ Some tests failed")
    print(f"{'=' * 60}")