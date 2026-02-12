import torch
import torch.nn as nn
from torch import Tensor as T
from dataclasses import dataclass
from omegaconf import DictConfig

from synths.constants import (
    DEFAULT_FS,
    DEFAULT_N_FFT,
    DEFAULT_LAGRANGE_ORDER,
    DEFAULT_IIR_TRUNCATION,
    DEFAULT_RND_SEED,
    DEFAULT_ONSET_THRESHOLD,
)
from synths.dsp import oracle_physical_model
from synths.ddsp import (
    lin_resample_many,
    excitation_onset,
    dynamics_filter,
    Implementation,
    pluck_position_filter,
    karplus_strong,
)


@dataclass
class PhysicalModelConfig:
    """Configuration for PhysicalModel structure and hyperparameters."""
    num_samples: int
    fs: int = DEFAULT_FS
    device: str = 'cpu'
    n_fft: int = DEFAULT_N_FFT
    hop_length: int | None = None
    lagrange_order: int = DEFAULT_LAGRANGE_ORDER
    iir_truncation: int = DEFAULT_IIR_TRUNCATION
    random_seed: int = DEFAULT_RND_SEED
    use_freq_pluck: bool = False
    use_freq_ksa: bool = False
    training: bool = True
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD


class PhysicalModel(nn.Module):
    def __init__(self, config: PhysicalModelConfig | DictConfig):
        super().__init__()
        if isinstance(config, DictConfig):
            config = PhysicalModelConfig(**config)

        self.num_samples = config.num_samples
        self.fs = config.fs
        self.device = torch.device(config.device)
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length if config.hop_length is not None else config.n_fft // 4
        self.lagrange_order = config.lagrange_order
        self.iir_truncation = config.iir_truncation
        self.random_seed = config.random_seed
        self.use_freq_pluck = config.use_freq_pluck
        self.use_freq_ksa = config.use_freq_ksa
        self._training = config.training
        self.onset_threshold = config.onset_threshold

        window_tensor = torch.ones(self.n_fft)
        self.register_buffer('window', window_tensor.to(self.device))

    def forward(self, params: dict[str, T]) -> T:
        """
        Synthesize plucked string audio.

        Args:
            params: Dictionary containing:
                - onset_probs: [B, num_frames] - onset probability per frame [0, 1]
                - f0: [B, num_frames] - fundamental frequency in Hz
                - pluck_position: [B, num_frames] - pluck position [0, 1]
                - burst_gain: [B, num_frames] - excitation gain [0, 1]
                - dynamic_level: [B, num_frames] - dynamic level (0=soft, 1=bright)
                - a1: [B, num_frames] - loop filter coefficient [0, 1]
                - decay: [B, num_frames] - decay/damping coefficient [0, 1]

        Returns:
            Synthesized audio [B, num_samples]
        """
        self.resample_parameters(params)

        x = self._generate_excitation(params['onset_probs'], params['f0'])
        x = x * self.p_time['burst_gain']
        x = self._apply_dynamics_filter(x)
        x = self._apply_pluck_filter(x)
        x = self._apply_karplus_strong(x)

        return x

    @torch.no_grad()
    def oracle_synth(self, params: dict[str, T]) -> T:
        """
        Synthesize using the NumPy oracle physical model with the same
        params dict interface as forward().

        Args:
            params: Same dictionary as forward()

        Returns:
            Synthesized audio [B, num_samples]
        """
        batch_size = next(iter(params.values())).shape[0]
        outputs = []

        for b in range(batch_size):
            trigger_frames = (params['onset_probs'][b] >= self.onset_threshold).cpu().numpy().astype(float)

            y = oracle_physical_model(
                trigger_frames=trigger_frames,
                f0=params['f0'][b].cpu().numpy(),
                pluck_position=params['pluck_position'][b].cpu().numpy(),
                burst_gain=params['burst_gain'][b].cpu().numpy(),
                dynamic_level=params['dynamic_level'][b].cpu().numpy(),
                a1=params['a1'][b].cpu().numpy(),
                decay=params['decay'][b].cpu().numpy(),
                num_samples=self.num_samples,
                fs=self.fs,
                lagrange_order=self.lagrange_order,
                random_seed=self.random_seed,
            )
            outputs.append(torch.from_numpy(y).to(params['f0'].device, params['f0'].dtype))

        return torch.stack(outputs, dim=0)

    def resample_parameters(self, params: dict[str, T]) -> None:
        self._params = params
        self.p_time = lin_resample_many(signal_length=self.num_samples, **params)
        self.p_stft = None

    def _get_stft_params(self, num_stft_frames: int) -> dict[str, T]:
        """Get STFT-rate parameters, computing them lazily."""
        if self.p_stft is None:
            self.p_stft = lin_resample_many(signal_length=num_stft_frames, **self._params)
        return self.p_stft

    def _generate_excitation(self, onset_probs: T, f0: T) -> T:
        """Generate noise burst excitation with onset gating."""
        x, _ = excitation_onset(
            onset_probs=onset_probs,
            signal_length=self.num_samples,
            f0=f0,
            fs=self.fs,
            noise_seed=self.random_seed,
            training=self._training,
            threshold=self.onset_threshold
        )
        return x

    def _apply_dynamics_filter(self, x: T) -> T:
        """Apply dynamics filter (always in time domain)."""
        return dynamics_filter(
            x=x,
            f0=self.p_time['f0'],
            dynamic_level=self.p_time['dynamic_level'],
            fs=self.fs,
        )

    def _apply_pluck_filter(self, x: T) -> T:
        """Apply pluck position filter."""
        if self.use_freq_pluck:
            x, num_frames = self._to_freq_domain(x)
            p = self._get_stft_params(num_frames)
            implementation = Implementation.FREQUENCY_SAMPLING
        else:
            p = self.p_time
            implementation = Implementation.TIME_DOMAIN

        return pluck_position_filter(
            x=x,
            f0=p['f0'],
            position=p['pluck_position'],
            implementation=implementation,
            fs=self.fs,
            lagrange_order=self.lagrange_order,
            n_fft=self.n_fft,
        )

    def _apply_karplus_strong(self, x: T) -> T:
        """Apply Karplus-Strong algorithm."""
        if self.use_freq_ksa:
            if not self.use_freq_pluck:
                x, num_frames = self._to_freq_domain(x)
            else:
                num_frames = x.shape[1]  # Already in freq domain
            p = self._get_stft_params(num_frames)
            implementation = Implementation.FREQUENCY_SAMPLING
        else:
            if self.use_freq_pluck:
                x = self._to_time_domain(x)
            p = self.p_time
            implementation = Implementation.TIME_DOMAIN

        x = karplus_strong(
            x=x,
            f0=p['f0'],
            a1=p['a1'],
            g=p['decay'],
            implementation=implementation,
            fs=self.fs,
            lagrange_order=self.lagrange_order,
            iir_truncation=self.iir_truncation,
            n_fft=self.n_fft,
        )

        return self._to_time_domain(x) if self.use_freq_ksa else x

    def _to_freq_domain(self, x: T) -> tuple[T, int]:
        """Convert time-domain signal to frequency domain and return frame count."""
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                       window=self.window, return_complex=True)
        X = X.permute(0, 2, 1)  # [B, num_stft_frames, n_bins]
        return X, X.shape[1]

    def _to_time_domain(self, X: T) -> T:
        """Convert frequency-domain signal to time domain."""
        X_perm = X.permute(0, 2, 1)
        return torch.istft(X_perm, n_fft=self.n_fft, hop_length=self.hop_length,
                           window=self.window, length=self.num_samples)


# =============================================================================
#                           TESTS
# =============================================================================
if __name__ == "__main__":
    duration = 4
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

    def check_output(y, label):
        if torch.isnan(y).any() or torch.isinf(y).any():
            print(f"  FAIL: {label}")
            return False
        print(f"  PASS: {label}")
        return True

    all_passed = True
    test_count = 0

    # =========================================================================
    # Test differentiable forward (all implementation combinations)
    # =========================================================================
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

            config = PhysicalModelConfig(
                num_samples=num_samples,
                fs=fs,
                device='cpu',
                use_freq_pluck=use_freq_pluck,
                use_freq_ksa=use_freq_ksa,
            )
            model = PhysicalModel(config)

            onset_probs = torch.zeros(1, num_frames)
            onset_probs[0, [0, 25, 50, 75]] = 1.0

            for param_name, (min_val, max_val) in sweeps.items():
                params = {k: torch.full((1, num_frames), v) for k, v in defaults.items()}
                params[param_name] = torch.linspace(min_val, max_val, num_frames).unsqueeze(0)
                params['onset_probs'] = onset_probs

                y = model(params)
                if not check_output(y, f"{param_name} sweep"):
                    all_passed = False
                test_count += 1

            params = {k: torch.linspace(*v, num_frames).unsqueeze(0) for k, v in sweeps.items()}
            params['onset_probs'] = onset_probs
            y = model(params)
            if not check_output(y, "all parameters sweeping"):
                all_passed = False
            test_count += 1

    # =========================================================================
    # Test oracle_synth
    # =========================================================================
    for fs in sample_rates:
        num_samples = int(fs * duration)

        print(f"\n{'=' * 60}")
        print(f"Testing [oracle_synth] at fs={fs}Hz ({num_samples} samples)")
        print(f"{'=' * 60}")

        config = PhysicalModelConfig(
            num_samples=num_samples,
            fs=fs,
            device='cpu',
            training=False,
        )
        model = PhysicalModel(config)

        onset_probs = torch.zeros(1, num_frames)
        onset_probs[0, [0, 25, 50, 75]] = 1.0

        for param_name, (min_val, max_val) in sweeps.items():
            params = {k: torch.full((1, num_frames), v) for k, v in defaults.items()}
            params[param_name] = torch.linspace(min_val, max_val, num_frames).unsqueeze(0)
            params['onset_probs'] = onset_probs

            y = model.oracle_synth(params)
            if not check_output(y, f"oracle {param_name} sweep"):
                all_passed = False
            test_count += 1

        params = {k: torch.linspace(*v, num_frames).unsqueeze(0) for k, v in sweeps.items()}
        params['onset_probs'] = onset_probs
        y = model.oracle_synth(params)
        if not check_output(y, "oracle all parameters sweeping"):
            all_passed = False
        test_count += 1

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"✓ All {test_count} tests passed!" if all_passed else "✗ Some tests failed")
    print(f"{'=' * 60}")