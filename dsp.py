from numba import jit
import numpy as np
import numpy.typing as npt

LAGRANGE_ORDER = 5
F0_MIN = 20
FS_MIN = 16000
RND_SEED = 42

@jit(nopython=True)
def linear_upsample(x: npt.NDArray, num_samples: int) -> npt.NDArray:
    return np.interp(np.linspace(0, 1, num_samples),  # new x
                     np.linspace(0, 1, len(x)),  # old x
                     x)  # old y

@jit(nopython=True)
def no_dc_burst(
        burst_length: int,
        seed: int=RND_SEED
) -> npt.NDArray:
    np.random.seed(seed)  # Use old-style seed instead of default_rng
    burst = np.random.random(burst_length)
    burst = burst / np.max(burst)
    return (burst - 0.5) * 2

@jit(nopython=True)
def noise_burst_excitation(
        num_samples: int,
        trigger_samples: npt.NDArray,
        f0: npt.NDArray,
        fs: int,
        seed: int=RND_SEED
) -> npt.NDArray:
    """
    Each trigger creates a noise burst with length equal to one period
    of the fundamental frequency at that trigger point.

    :param num_samples: Total length of output signal in samples
    :param trigger_samples: Array of trigger sample indices
    :param f0: Array of fundamental frequencies in Hz
    :param fs: Sample rate in Hz
    :return: Excitation signal with noise bursts at trigger times
    """
    assert len(f0) == num_samples
    excitation = np.zeros(num_samples)

    for i in range(len(trigger_samples)):
        trigger_sample = int(trigger_samples[i])
        assert trigger_sample <= num_samples
        f0_at_trigger = f0[trigger_sample]
        burst_length = int(fs / f0_at_trigger)
        end_sample = min(trigger_sample + burst_length, num_samples)
        burst_length = end_sample - trigger_sample
        excitation[trigger_sample:end_sample] = no_dc_burst(burst_length, seed=seed)
    return excitation

@jit(nopython=True)
def all_zero_comb(
        x: npt.NDArray,
        L: npt.NDArray,
        fs: int = FS_MIN,
        lagrange_order: int = LAGRANGE_ORDER
) -> npt.NDArray:
    """
    All-zero comb filter with time-varying fractional delay: y(n) = x(n) - x(n - L)
    Uses Lagrange interpolation for fractional delays.

    :param x: Input signal [num_samples]
    :param L: Delay in samples [num_samples], can be fractional
    :param fs: Sample rate (used to determine max buffer size)
    :param lagrange_order: Order of Lagrange interpolator
    :return: Filtered signal [num_samples]
    """
    assert len(x) == len(L)
    num_samples = len(x)
    y = np.zeros_like(x)

    max_delay = int(fs / F0_MIN)
    delay_buffer = np.zeros(max_delay)
    write_idx = 0

    for n in range(num_samples):
        L_int, h = lagrange_fractional_delay(L[n], lagrange_order)
        delayed_sample = 0.0
        for k in range(lagrange_order + 1):
            read_idx = (write_idx - L_int - k) % len(delay_buffer)
            delayed_sample += h[k] * delay_buffer[read_idx]
        y[n] = x[n] - delayed_sample
        delay_buffer[write_idx] = x[n]
        write_idx = (write_idx + 1) % len(delay_buffer)
    return y

@jit(nopython=True)
def pluck_position_filter(
        x: npt.NDArray,
        f0: npt.NDArray,
        position: npt.NDArray,
        fs: int = FS_MIN,
        lagrange_order: int = LAGRANGE_ORDER
) -> npt.NDArray:
    """
    Simulate pluck position using an all-zero comb filter as per section
    "Simulation of a Moving Pick" in "Extensions of the Karplus-Strong Plucked-String Algorithm"
    by David A. Jaffe and Julius O. Smith

    Creates spectral notches at harmonics corresponding to pluck position.
    - position = 0.5 (midpoint): removes even harmonics
    - position = 0.1: removes every 10th harmonic
    - position = 1/N: approximates sul ponticello (bright, near bridge)

    :param x: Input signal (excitation) [num_samples]
    :param f0: Fundamental frequency in Hz [num_samples]
    :param position: Pluck position as fraction of string length, range [0, 1] [num_samples]
    :param fs: Sample rate in Hz
    :param lagrange_order: Order of Lagrange interpolator
    :return: Filtered excitation signal [num_samples]
    """
    assert len(x) == len(f0) == len(position)
    assert np.all((position >= 0.0) & (position <= 1.0))

    L = fs / f0
    comb_L = L * position
    return all_zero_comb(x, comb_L, fs, lagrange_order)

@jit(nopython=True)
def compute_dynamics_R(
        f0: float,
        bw: float,
        fs: int = FS_MIN
) -> float:
    """
    Compute dynamics filter coefficient R for a given pitch and dynamic level.
    """
    bw *= (fs / 2.0)
    fm = np.sqrt(F0_MIN * (fs / 2.0))
    Ts = 1.0 / fs
    R_L = np.exp(-bw * np.pi * Ts)
    G_L = (1 - R_L) / np.abs(1 - R_L * np.exp(-1j * 2 * np.pi * fm * Ts))
    left_side_num = 1 - G_L ** 2 * np.cos(2 * np.pi * f0 * Ts)
    left_side_den = 1 - G_L ** 2
    left_side = left_side_num / left_side_den

    right_side_outside = 2 * G_L * np.sin(np.pi * f0 * Ts)
    right_side_num = np.sqrt(1 - G_L ** 2 * np.cos(np.pi * f0 * Ts) ** 2)
    right_side_den = 1 - G_L ** 2
    right_side = right_side_outside * (right_side_num / right_side_den)

    R_plus = left_side + right_side
    R_minus = left_side - right_side
    return R_plus if np.abs(R_plus) < 1 else R_minus


@jit(nopython=True)
def dynamics_filter(
        x: npt.NDArray,
        f0: npt.NDArray,
        dynamic_level: npt.NDArray,
        fs: int = FS_MIN
) -> npt.NDArray:
    """
    Apply dynamics filter to excitation with time-varying dynamic level.

    From Jaffe & Smith "Extensions of the Karplus-Strong Plucked-String Algorithm":
    Controls spectral bandwidth to simulate pluck dynamics. Harder plucks have
    more high-frequency content (wide bandwidth), softer plucks are darker
    (narrow bandwidth).

    :param x: Input signal (excitation) [num_samples]
    :param f0: Fundamental frequency in Hz [num_samples]
    :param dynamic_level: Dynamic level parameter [num_samples]
                          0.0 = soft/dark (narrow bandwidth)
                          1.0 = loud/bright (wide bandwidth)
    :param fs: Sample rate in Hz
    :return: Filtered excitation signal [num_samples]
    """
    assert len(x) == len(f0) == len(dynamic_level)
    assert np.all((dynamic_level >= 0.0) & (dynamic_level <= 1.0))

    num_samples = len(x)
    y = np.zeros_like(x)
    y_prev = 0.0

    for n in range(num_samples):
        # Compute R for this pitch and dynamic level
        R = compute_dynamics_R(f0[n], dynamic_level[n], fs)

        # Apply one-pole filter: y[n] = (1-R)*x[n] + R*y[n-1]
        y[n] = (1.0 - R) * x[n] + R * y_prev
        y_prev = y[n]

    return y

@jit(nopython=True)
def one_pole_phase_delay(f0: float, a1: float, fs: int) -> float:
    """
    Compute phase delay of one-pole loop filter at fundamental frequency.
    Note: only valid for positive b0.

    :param f0: Fundamental frequency in Hz
    :param a1: Loop filter pole coefficient
    :param fs: Sample rate in Hz
    :return: Phase delay in samples
    """
    omega0 = 2.0 * np.pi * f0 / fs

    # Transfer function: H(e^jω) = b0 / (1 - a1·e^(-jω))
    # Denominator: 1 - a1·e^(-jω) = (1 - a1·cos(ω)) + j·a1·sin(ω)
    denom_real = 1.0 - a1 * np.cos(omega0)
    denom_imag = a1 * np.sin(omega0)
    phase = np.arctan2(denom_imag, denom_real)

    # Phase delay: τ = -φ(ω) / ω
    phase_delay = -phase / omega0
    return phase_delay


@jit(nopython=True)
def lagrange_coefficients(D: float, N: int = LAGRANGE_ORDER) -> npt.NDArray:
    """
    Compute Lagrange interpolation coefficients for fractional delay.

    :param D: Desired fractional delay
    :param N: Order of interpolation
    :return: Coefficients h[0], h[1], ..., h[N]
    """
    h = np.zeros(N + 1)
    for n in range(N + 1):
        h_n = 1.0
        for k in range(N + 1):
            if k != n:
                h_n *= (D - k) / (n - k)
        h[n] = h_n
    return h


@jit(nopython=True)
def lagrange_fractional_delay(L: float, N: int = LAGRANGE_ORDER) -> tuple[int, npt.NDArray]:
    """
    Compute Lagrange interpolation coefficients for a given delay.
    Centers the fractional delay around N/2 for optimal frequency response.

    :param L: Total desired delay in samples
    :param N: Order of Lagrange filter
    :return: (L_int, h) where L_int is integer delay and h are coefficients
    """
    offset = N // 2
    L_adjusted = L - offset
    L_int = int(np.floor(L_adjusted))
    D = L_adjusted - L_int
    D_centered = D + offset
    h = lagrange_coefficients(D_centered, N)
    return L_int, h


@jit(nopython=True)
def karplus_strong(
        x: npt.NDArray,
        f0: npt.NDArray,
        a1: npt.NDArray,
        g: npt.NDArray,
        fs: int = FS_MIN,
        lagrange_order: int = LAGRANGE_ORDER,
) -> npt.NDArray:
    """
    Karplus-Strong string synthesis with fractional delay, loop filtering and phase delay tuning.
    See "Physical Modeling of Plucked String Instruments with Application to Real-Time Sound Synthesis"
    by Vesa Välimäki, J. Huopaniemi, M. Karjalainen

    :param x: Excitation signal [num_samples]
    :param f0: Fundamental frequency in Hz [num_samples]
    :param a1: Loop filter pole coefficient, range [0, 1] [num_samples]
    :param g: Loop gain, range (0, 1) [num_samples]
    :param fs: Sample rate in Hz
    :param lagrange_order: Order of interpolator
    :return: Synthesized audio signal [num_samples]
    """
    assert len(x) == len(f0) == len(a1) == len(g)
    assert np.all((a1 >= 0.0) & (a1 <= 1.0))
    assert np.all((g >= 0.0) & (g <= 1.0))

    num_samples = len(x)
    y = np.zeros_like(x)
    L = fs / f0
    delay_buffer = np.zeros(int(fs / F0_MIN))
    filter_state = 0.0
    write_idx = 0

    for n in range(num_samples):
        phase_delay = one_pole_phase_delay(f0[n], a1[n], fs)
        L_corrected = L[n] + phase_delay

        L_int, h = lagrange_fractional_delay(L_corrected, lagrange_order)
        delayed_sample = 0.0
        for k in range(lagrange_order + 1):
            read_idx = (write_idx - L_int - k) % len(delay_buffer)
            delayed_sample += h[k] * delay_buffer[read_idx]

        b0 = g[n] * (1.0 - a1[n])
        filtered_sample = b0 * delayed_sample + a1[n] * filter_state
        filter_state = filtered_sample

        output_sample = x[n] + filtered_sample
        delay_buffer[write_idx] = output_sample
        write_idx = (write_idx + 1) % len(delay_buffer)
        y[n] = output_sample
    return y

@jit(nopython=True)
def oracle_physical_model(
        num_samples: int,
        trigger_samples: npt.NDArray,
        f0: npt.NDArray,
        pluck_position: npt.NDArray,
        burst_gain: npt.NDArray,
        dynamic_level: npt.NDArray,
        a1: npt.NDArray,
        decay: npt.NDArray,
        fs: int = FS_MIN,
        lagrange_order: int = LAGRANGE_ORDER,
        random_seed: int = RND_SEED,
) -> npt.NDArray:
    x = noise_burst_excitation(num_samples=num_samples, trigger_samples=trigger_samples, f0=f0, fs=fs, seed=random_seed)
    x *= burst_gain
    x = pluck_position_filter(x=x, f0=f0, position=pluck_position, fs=fs, lagrange_order=lagrange_order)
    x = dynamics_filter(x=x, f0=f0, dynamic_level=dynamic_level, fs=fs)
    return karplus_strong(x=x, f0=f0, a1=a1, g=decay, fs=fs, lagrange_order=lagrange_order)


if __name__ == "__main__":
    fs = FS_MIN
    duration = 0.1
    num_samples = int(fs * duration)

    # Test different f0 values
    f0_values = [55.0, 110.0, 220.0, 440.0, 880.0, 1760.0, 3520.0]  # A1 to A7

    # Edge case parameters (valid ranges only)
    edge_cases = [
        (0.0, 1.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (1.0, 0.0),
    ]

    all_passed = True

    # Test 1: Edge cases with different f0 values
    for target_f0 in f0_values:
        for a1_val, g_val in edge_cases:
            f0 = np.full(num_samples, target_f0)
            a1 = np.full(num_samples, a1_val)
            g = np.full(num_samples, g_val)

            x = np.zeros(num_samples)
            x[0] = 1.0

            y = karplus_strong(x, f0, a1, g, fs)

            if np.isnan(y).any() or np.isinf(y).any():
                print(f"FAIL: a1={a1_val}, g={g_val}, f0={target_f0:.1f}Hz")
                all_passed = False

    # Test 2: Parameter sweeps
    f0_min, f0_max = min(f0_values), max(f0_values)
    a1_min, a1_max = min(a1 for a1, _ in edge_cases), max(a1 for a1, _ in edge_cases)
    g_min, g_max = min(g for _, g in edge_cases), max(g for _, g in edge_cases)

    sweep_tests = [
        (np.linspace(f0_min, f0_max, num_samples),
         np.linspace(a1_min, a1_max, num_samples),
         np.linspace(g_min, g_max, num_samples),
         f"f0 sweep ({f0_min:.0f}-{f0_max:.0f}Hz) with a1 sweep ({a1_min} to {a1_max}) and g sweep ({g_min} to {g_max})"),
        (np.full(num_samples, f0_values[len(f0_values) // 2]),
         np.linspace(a1_min, a1_max, num_samples),
         np.full(num_samples, g_max),
         f"a1 sweep ({a1_min} to {a1_max}) with f0={f0_values[len(f0_values) // 2]:.0f}Hz and g={g_max}"),
        (np.linspace(f0_min, f0_max, num_samples),
         np.full(num_samples, (a1_min + a1_max) / 2),
         np.full(num_samples, (g_min + g_max) / 2),
         f"f0 sweep ({f0_min:.0f}-{f0_max:.0f}Hz) with a1={(a1_min + a1_max) / 2} and g={(g_min + g_max) / 2}"),
    ]

    for f0_sweep, a1_sweep, g_sweep, description in sweep_tests:
        x = np.zeros(num_samples)
        x[0] = 1.0

        y = karplus_strong(x, f0_sweep, a1_sweep, g_sweep, fs)

        if np.isnan(y).any() or np.isinf(y).any():
            print(f"FAIL: {description}")
            all_passed = False

    if all_passed:
        print("All tests passed")