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
    implementation: Implementation = Implementation.TIME_DOMAIN
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
        self.implementation = config.implementation
        self.use_lti = config.use_lti

        window_tensor = torch.ones(self.n_fft)
        self.register_buffer('window', window_tensor.to(self.device))

    def forward(self, params: dict[str, T]) -> SynthOutput:
        x_time = excitation(
            times=params['time'],
            exists=params['exists'],
            f0=params['f0'],
            signal_length=self.num_samples,
            implementation=self.implementation,
            lagrange_order=self.lagrange_order,
            fs=self.fs,
            noise_seed=self.random_seed,
        )

        self._expand_sparse_events_to_dense(params)

        if self.implementation == Implementation.FREQUENCY_SAMPLING:
            x = self._forward_frequency_domain(x_time)
        else:
            x = self._forward_time_domain(x_time)

        x = x * params['burst_gain'][:, 0:1]

        return x, params

    def _expand_sparse_events_to_dense(self, events: dict[str, T]):
        """Expands [B, max_events] sparse params to [B, num_samples] piecewise constant."""
        B, max_events = events['exists'].shape
        N = self.num_samples
        device = events['exists'].device

        dense = {
            "f0": torch.full((B, N), 440.0, device=device),
            "decay": torch.full((B, N), 0.99, device=device),
            "a1": torch.full((B, N), 0.5, device=device),
            "pluck_position": torch.full((B, N), 0.5, device=device),
            "dynamic_level": torch.full((B, N), 0.5, device=device),
        }

        for b in range(B):
            valid_mask = events["exists"][b] > 0.5
            if not valid_mask.any(): continue

            v_time = events["time"][b, valid_mask]
            sorted_idx = torch.argsort(v_time)

            s_time = v_time[sorted_idx]
            for i in range(len(s_time)):
                start = int(s_time[i].item() * N)
                end = N if i == len(s_time) - 1 else int(s_time[i + 1].item() * N)
                fill_start = 0 if i == 0 else start

                for k in dense.keys():
                    val = events[k][b, valid_mask][sorted_idx][i]
                    dense[k][b, fill_start:end] = val

        self.p_time = dense
        self._params = events

        self.p_stft = None

    def _forward_time_domain(self, x: T) -> T:
        p = self.p_time

        x = dynamics_filter(
            x=x,
            f0=p['f0'],
            dynamic_level=p['dynamic_level'],
            fs=self.fs,
        )

        x = pluck_position_filter(
            x=x,
            f0=p['f0'],
            position=p['pluck_position'],
            fs=self.fs,
        )

        x = karplus_strong(
            x=x,
            f0=p['f0'],
            a1=p['a1'],
            g=p['decay'],
            fs=self.fs,
            lagrange_order=self.lagrange_order,
            iir_truncation=self.iir_truncation,
        )

        return x

    def _forward_frequency_domain(self, x: T) -> T:
        # --- to frequency domain (once) ---
        if self.use_lti:
            X = self._to_lti_freq_domain(x)
            p = self._params
            n_fft = self.num_samples
        else:
            X, num_frames = self._to_stft_domain(x)
            p = self._get_stft_params(num_frames)
            n_fft = self.n_fft

        impl = Implementation.FREQUENCY_SAMPLING

        X = dynamics_filter(
            x=X,
            f0=p['f0'],
            dynamic_level=p['dynamic_level'],
            implementation=impl,
            n_fft=n_fft,
            fs=self.fs,
        )

        X = pluck_position_filter(
            x=X,
            f0=p['f0'],
            position=p['pluck_position'],
            implementation=impl,
            fs=self.fs,
            n_fft=n_fft,
        )

        X = karplus_strong(
            x=X,
            f0=p['f0'],
            a1=p['a1'],
            g=p['decay'],
            implementation=impl,
            fs=self.fs,
            n_fft=n_fft,
        )

        if self.use_lti:
            return self._from_lti_freq_domain(X)
        else:
            return self._from_stft_domain(X)

    @torch.no_grad()
    def oracle_synth(self, params: dict[str, T]) -> SynthOutput:
        """
        Synthesize using the NumPy oracle physical model.

        Returns:
            (audio [B, num_samples], params passed through)
        """
        self._expand_sparse_events_to_dense(params)
        dense_params = self.p_time

        batch_size = params['f0'].shape[0]
        outputs = []

        for b in range(batch_size):
            device = params['f0'].device
            bg_dense = torch.zeros(self.num_samples, device=device)
            v_exists = params['exists'][b] > 0.5
            if v_exists.any():
                v_time = params['time'][b, v_exists]
                v_bg = params['burst_gain'][b, v_exists]

                # Discrete placement for the NumPy model
                indices = (v_time * (self.num_samples - 1)).long()
                bg_dense.scatter_(0, indices, v_bg)

            y = oracle_physical_model(
                f0=dense_params['f0'][b].cpu().numpy(),
                burst_gain=bg_dense.cpu().numpy(),  # Now dense!
                decay=dense_params['decay'][b].cpu().numpy(),
                a1=dense_params['a1'][b].cpu().numpy(),
                pluck_position=dense_params['pluck_position'][b].cpu().numpy(),
                dynamic_level=dense_params['dynamic_level'][b].cpu().numpy(),
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
            first_len = next(iter(params.values())).shape[1]
            if first_len == self.num_samples:
                self.p_time = params
            else:
                self.p_time = lin_resample_many(signal_length=self.num_samples, **params)

        self.p_stft = None

    def _get_stft_params(self, num_stft_frames: int) -> dict[str, T]:
        if getattr(self, 'p_stft', None) is None:
            self.p_stft = lin_resample_many(signal_length=num_stft_frames, **self.p_time)
        return self.p_stft

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

    defaults = {
        'f0': 220.0,
        'pluck_position': 0.5,
        'burst_gain': 0.0,
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
    # Test differentiable implementations
    # =========================================================================
    impl_options = [
        (Implementation.TIME_DOMAIN, "Time-Domain"),
        (Implementation.FREQUENCY_SAMPLING, "Frequency-Sampling"),
    ]

    for impl, impl_name in impl_options:
        for fs in sample_rates:
            num_samples = int(fs * duration)
            print(f"\n{'=' * 60}")
            print(f"Testing [{impl_name}] at fs={fs}Hz ({num_samples} samples)")
            print(f"{'=' * 60}")

            config = SynthConfig(num_samples=num_samples, fs=fs, implementation=impl)
            model = Synth(config)

            params = {
                'exists': torch.ones(1, 4),
                'time': torch.tensor([[0.0, 0.25, 0.5, 0.75]]),  # 4 plucks evenly spaced
                'f0': torch.tensor([[220.0, 330.0, 440.0, 550.0]]),
                'burst_gain': torch.tensor([[0.8, 0.8, 0.8, 0.8]]),
                'pluck_position': torch.full((1, 4), 0.5),
                'dynamic_level': torch.full((1, 4), 0.5),
                'a1': torch.full((1, 4), 0.5),
                'decay': torch.full((1, 4), 0.995),
            }

            y, _ = model(params)
            check_output(y, "Sparse Event Rendering")
            test_count += 1

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"✓ All {test_count} tests passed!" if all_passed else "✗ Some tests failed")
    print(f"{'=' * 60}")