from __future__ import annotations
import torch
import torch.nn as nn
import scipy.signal.windows
from torch import Tensor
from sot import Wasserstein1DLoss

EPS = 1e-8

class MultiScaleSpectralLoss(nn.Module):
    """Multi-scale log-magnitude STFT distance.
    Args:
        win_lengths:  Analysis window sizes (also used as FFT sizes unless
                      *fft_sizes* is given).  Defaults to the Schwär & Müller
                      prime-length set [67, 127, 257, 509, 1021, 2053].
        fft_sizes:    FFT sizes per scale (default: same as *win_lengths*).
        hop_sizes:    Hop sizes per scale (default: ``win_length // 2``).
        window:       ``"flat_top"`` | ``"hann"`` | ``"rect"``.
        gamma:        Compression coefficient for ``log(1 + γ·|S|)``.
        p:            Norm order (default 2 = Euclidean).
    """

    def __init__(
        self,
        win_lengths: list[int] | None = None,
        fft_sizes: list[int] | None = None,
        hop_sizes: list[int] | None = None,
        window: str = "flat_top",
        gamma: float = 1.0,
        p: int = 2,
    ):
        super().__init__()
        if win_lengths is None:
            win_lengths = [67, 127, 257, 509, 1021, 2053]
        if fft_sizes is None:
            fft_sizes = win_lengths
        if hop_sizes is None:
            hop_sizes = [w // 2 for w in win_lengths]

        assert len(fft_sizes) == len(win_lengths) == len(hop_sizes)

        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.gamma = gamma
        self.p = p

        for wl in win_lengths:
            self.register_buffer(f"win_{wl}", self._make_window(window, wl))

    def forward(
            self,
            x: Tensor,          # [B, num_samples]
            x_target: Tensor    # [B, num_samples]
    ) -> Tensor:                # Scalar (Batch Mean then Sum)
        assert x.dim() == 2 == x_target.dim(), "only mono audio is supported. Got {x.shape} and {x_target.shape}"

        dists = []
        for fft_size, hop_size, wl in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            win = getattr(self, f"win_{wl}")

            Sx = torch.stft(
                x,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=wl,
                window=win,
                return_complex=True,
            ).abs()

            Sx_target = torch.stft(
                x_target,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=wl,
                window=win,
                return_complex=True,
            ).abs()

            log_Sx = torch.log1p(self.gamma * Sx + EPS)
            log_Sx_target = torch.log1p(self.gamma * Sx_target + EPS)

            dist = torch.linalg.vector_norm(
                log_Sx_target - log_Sx, ord=self.p, dim=(-2, -1),
            )
            dists.append(dist)

        return torch.stack(dists, dim=1).sum(dim=1).mean()

    @staticmethod
    def _make_window(window: str, n: int) -> Tensor:
        if window == "flat_top":
            w = scipy.signal.windows.flattop(n, sym=False)
            return torch.from_numpy(w).float()
        if window == "hann":
            return torch.hann_window(n, periodic=True)
        if window == "rect":
            return torch.ones(n)
        raise ValueError(f"Unknown window type: {window}")


class SOT2048Loss(nn.Module):
    """
    SOT-2048-style wrapper using the bernardo-torres/spectral-optimal-transport repo.

    Defaults chosen to match the paper's SOT-2048 SOT term as closely as the repo exposes:
      - transform: STFT
      - fft_size: 2048
      - hop_length: 256
      - window: flattop
      - square_magnitude: True  (power spectrum)
      - p: 2                 (quadratic cost; returns W2^2 by default because apply_root=False)
      - quantile_lowpass: True (repo's frequency-cutoff-like behaviour)
      - balanced: False        (recommended when using quantile_lowpass, per repo docstring)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        device: str | torch.device = 'cpu',
        reduce: bool = True,
    ):
        super().__init__()

        self.loss_fn = Wasserstein1DLoss(
            transform="stft",
            fft_size=2048,
            hop_length=256,
            sample_rate=sample_rate,
            window="flattop",
            square_magnitude=True,
            p=2,
            apply_root=False,
            normalize=True,
            balanced=False,
            quantile_lowpass=True,
            reduce=reduce,
            device=device,
        )

    def forward(self, x: Tensor, x_target: Tensor) -> Tensor:
        assert x.dim() == 2 == x_target.dim(), f"only mono audio is supported. Got {x.shape} and {x_target.shape}"
        return self.loss_fn(x, x_target)

if __name__ == "__main__":
    audio = torch.randn(3, 64000)
    audio_target = torch.randn(3, 64000)

    mss = MultiScaleSpectralLoss()
    print(mss(audio, audio_target))

    sot2048 = SOT2048Loss()
    print(sot2048(audio, audio_target))
