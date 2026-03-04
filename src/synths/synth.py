import torch
import torch.nn as nn
from torch import Tensor as T
from dataclasses import dataclass
from omegaconf import DictConfig

from src.synths.constants import (
    DEFAULT_FS,
    DEFAULT_N_FFT,
    DEFAULT_LAGRANGE_ORDER,
    DEFAULT_IIR_TRUNCATION,
    DEFAULT_RND_SEED,
)
from src.synths.param_registry import PARAM_NAMES, validate_param_dict
from src.synths.dsp import oracle_physical_model
from src.synths.ddsp import (
    lin_resample_many,
    excitation,
    dynamics_filter,
    Implementation,
    pluck_position_filter,
    karplus_strong,
)

# Canonical return type: (audio, params_dict).
SynthOutput = tuple[T, dict[str, T]]


@dataclass
class SynthConfig:
    """Configuration for Synth structure and hyperparameters."""
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
    use_lti: bool = False


class Synth(nn.Module):
    def __init__(self, config: SynthConfig | DictConfig):
        super().__init__()
        if isinstance(config, DictConfig):
            config = SynthConfig(**config)

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
        self.use_lti = config.use_lti

        window_tensor = torch.ones(self.n_fft)
        self.register_buffer('window', window_tensor.to(self.device))

    def forward(self, params: dict[str, T]) -> SynthOutput:
        """
        Differentiable Karplus-Strong synthesis.

        Args:
            params: Dictionary with keys matching PARAM_NAMES.

        Returns:
            (audio [B, num_samples], params passed through)
        """
        validate_param_dict(params, context="Synth.forward")
        self.resample_parameters(params)

        x = self._generate_excitation(params['burst_gain'], params['f0'])
        x = self._apply_dynamics_filter(x)
        x = self._apply_pluck_filter(x)
        x = self._apply_karplus_strong(x)

        return x, params

    @torch.no_grad()
    def oracle_synth(self, params: dict[str, T]) -> SynthOutput:
        """
        Synthesize using the NumPy oracle physical model.

        Returns:
            (audio [B, num_samples], params passed through)
        """
        validate_param_dict(params, context="Synth.oracle_synth")
        batch_size = next(iter(params.values())).shape[0]
        outputs = []

        for b in range(batch_size):
            y = oracle_physical_model(
                **{name: params[name][b].cpu().numpy() for name in PARAM_NAMES},
                num_samples=self.num_samples,
                fs=self.fs,
                lagrange_order=self.lagrange_order,
                random_seed=self.random_seed,
            )
            outputs.append(torch.from_numpy(y).to(params['f0'].device, params['f0'].dtype))

        return torch.stack(outputs, dim=0), params

    def resample_parameters(self, params: dict[str, T]) -> None:
        if self.use_lti:
            num_frames = next(iter(params.values())).shape[1]
            assert num_frames == 1, (
                f"use_lti=True requires num_frames=1, got {num_frames}"
            )

        self._params = params

        if self.use_lti:
            self.p_time = {k: v.expand(-1, self.num_samples)
                          for k, v in params.items()}
        else:
            self.p_time = lin_resample_many(signal_length=self.num_samples, **params)

        self.p_stft = None

    def _get_stft_params(self, num_stft_frames: int) -> dict[str, T]:
        if self.p_stft is None:
            self.p_stft = lin_resample_many(signal_length=num_stft_frames, **self._params)
        return self.p_stft

    def _generate_excitation(self, burst_gain: T, f0: T) -> T:
        return excitation(
            burst_gain=burst_gain,
            signal_length=self.num_samples,
            f0=f0,
            fs=self.fs,
            noise_seed=self.random_seed,
        )

    def _apply_dynamics_filter(self, x: T) -> T:
        return dynamics_filter(
            x=x,
            f0=self.p_time['f0'],
            dynamic_level=self.p_time['dynamic_level'],
            fs=self.fs,
        )

    def _apply_pluck_filter(self, x: T) -> T:
        if self.use_freq_pluck:
            if self.use_lti:
                x = self._to_lti_freq_domain(x)
                p = self._params
                n_fft = self.num_samples
            else:
                x, num_frames = self._to_stft_domain(x)
                p = self._get_stft_params(num_frames)
                n_fft = self.n_fft
            implementation = Implementation.FREQUENCY_SAMPLING
        else:
            p = self.p_time
            implementation = Implementation.TIME_DOMAIN
            n_fft = self.n_fft

        return pluck_position_filter(
            x=x,
            f0=p['f0'],
            position=p['pluck_position'],
            implementation=implementation,
            fs=self.fs,
            n_fft=n_fft,
        )

    def _apply_karplus_strong(self, x: T) -> T:
        if self.use_freq_ksa:
            if self.use_lti:
                if not self.use_freq_pluck:
                    x = self._to_lti_freq_domain(x)
                p = self._params
                n_fft = self.num_samples
            else:
                if not self.use_freq_pluck:
                    x, num_frames = self._to_stft_domain(x)
                else:
                    num_frames = x.shape[1]
                p = self._get_stft_params(num_frames)
                n_fft = self.n_fft
            implementation = Implementation.FREQUENCY_SAMPLING
        else:
            if self.use_freq_pluck:
                x = self._from_lti_freq_domain(x) if self.use_lti else self._from_stft_domain(x)
            p = self.p_time
            implementation = Implementation.TIME_DOMAIN
            n_fft = self.n_fft

        x = karplus_strong(
            x=x,
            f0=p['f0'],
            a1=p['a1'],
            g=p['decay'],
            implementation=implementation,
            fs=self.fs,
            lagrange_order=self.lagrange_order,
            iir_truncation=self.iir_truncation,
            n_fft=n_fft,
        )

        if self.use_freq_ksa:
            return self._from_lti_freq_domain(x) if self.use_lti else self._from_stft_domain(x)
        return x

    def _to_lti_freq_domain(self, x: T) -> T:
        return torch.fft.rfft(x, n=self.num_samples).unsqueeze(1)

    def _from_lti_freq_domain(self, X: T) -> T:
        return torch.fft.irfft(X.squeeze(1), n=self.num_samples)

    def _to_stft_domain(self, x: T) -> tuple[T, int]:
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                       window=self.window, return_complex=True)
        X = X.permute(0, 2, 1)
        return X, X.shape[1]

    def _from_stft_domain(self, X: T) -> T:
        return torch.istft(X.permute(0, 2, 1), n_fft=self.n_fft, hop_length=self.hop_length,
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
        'pluck_position': (0.1, 0.5),
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

            config = SynthConfig(
                num_samples=num_samples,
                fs=fs,
                device='cpu',
                use_freq_pluck=use_freq_pluck,
                use_freq_ksa=use_freq_ksa,
            )
            model = Synth(config)

            for param_name, (min_val, max_val) in sweeps.items():
                params = {k: torch.full((1, num_frames), v) for k, v in defaults.items()}
                params[param_name] = torch.linspace(min_val, max_val, num_frames).unsqueeze(0)

                params['burst_gain'] = torch.zeros(1, num_frames)
                params['burst_gain'][0, [0, 25, 50, 75]] = defaults['burst_gain']
                if param_name == 'burst_gain':
                    params['burst_gain'][0, [0, 25, 50, 75]] = torch.linspace(min_val + 0.01, max_val, 4)

                y, _ = model(params)
                if not check_output(y, f"{param_name} sweep"):
                    all_passed = False
                test_count += 1

            params = {k: torch.linspace(*v, num_frames).unsqueeze(0) for k, v in sweeps.items()}
            params['burst_gain'] = torch.zeros(1, num_frames)
            params['burst_gain'][0, [0, 25, 50, 75]] = 0.5
            y, _ = model(params)
            if not check_output(y, "all parameters sweeping"):
                all_passed = False
            test_count += 1

    # =========================================================================
    # Test LTI mode — all implementation combos, including pure TD
    # =========================================================================
    lti_combinations = [
        (False, False, "LTI TD pluck + TD KS"),
        (False, True,  "LTI TD pluck + FD KS"),
        (True,  False, "LTI FD pluck + TD KS"),
        (True,  True,  "LTI FD pluck + FD KS"),
    ]

    for use_freq_pluck, use_freq_ksa, impl_name in lti_combinations:
        for fs in sample_rates:
            num_samples = int(fs * duration)

            print(f"\n{'=' * 60}")
            print(f"Testing [{impl_name}] at fs={fs}Hz ({num_samples} samples)")
            print(f"{'=' * 60}")

            config = SynthConfig(
                num_samples=num_samples,
                fs=fs,
                device='cpu',
                use_freq_pluck=use_freq_pluck,
                use_freq_ksa=use_freq_ksa,
                use_lti=True,
            )
            model = Synth(config)

            # LTI mode: single frame params
            for param_name in defaults:
                params = {k: torch.full((1, 1), v) for k, v in defaults.items()}
                params['burst_gain'] = torch.tensor([[defaults['burst_gain']]])

                y, _ = model(params)
                if not check_output(y, f"LTI {param_name} constant"):
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

        config = SynthConfig(
            num_samples=num_samples,
            fs=fs,
            device='cpu',
        )
        model = Synth(config)

        for param_name, (min_val, max_val) in sweeps.items():
            params = {k: torch.full((1, num_frames), v) for k, v in defaults.items()}
            params[param_name] = torch.linspace(min_val, max_val, num_frames).unsqueeze(0)

            params['burst_gain'] = torch.zeros(1, num_frames)
            params['burst_gain'][0, [0, 25, 50, 75]] = defaults['burst_gain']
            if param_name == 'burst_gain':
                params['burst_gain'][0, [0, 25, 50, 75]] = torch.linspace(min_val + 0.01, max_val, 4)

            y, _ = model.oracle_synth(params)
            if not check_output(y, f"oracle {param_name} sweep"):
                all_passed = False
            test_count += 1

        params = {k: torch.linspace(*v, num_frames).unsqueeze(0) for k, v in sweeps.items()}
        params['burst_gain'] = torch.zeros(1, num_frames)
        params['burst_gain'][0, [0, 25, 50, 75]] = 0.5
        y, _ = model.oracle_synth(params)
        if not check_output(y, "oracle all parameters sweeping"):
            all_passed = False
        test_count += 1

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"✓ All {test_count} tests passed!" if all_passed else "✗ Some tests failed")
    print(f"{'=' * 60}")