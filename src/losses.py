from __future__ import annotations
import torch
import torch.nn as nn
import scipy.signal.windows
from torch import Tensor
from sot import Wasserstein1DLoss

from src.synths.param_registry import ParamSpec, PARAM_NAMES, make_default_registry, LossType
from scipy.optimize import linear_sum_assignment

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

class HungarianOnsetLoss(nn.Module):
    """Bipartite-matching loss for sparse onset (burst_gain) prediction.

    Matches predicted active frames to ground-truth onset frames via the
    Hungarian algorithm, then penalises time error, gain error, unmatched
    predictions, and missed ground-truth onsets.

    Args:
        active_threshold: minimum predicted gain to count as "active".
        w_time:           weight for normalised frame-distance of matched pairs.
        w_gain:           weight for |pred_gain − gt_gain| of matched pairs.
        w_unmatched_pred: weight penalty per unmatched predicted onset.
        w_missed_gt:      weight penalty per missed ground-truth onset.
    """

    def __init__(
        self,
        active_threshold: float = 1e-4,
        w_time: float = 0.1,
        w_gain: float = 1.0,
        w_unmatched_pred: float = 1.0,
        w_missed_gt: float = 2.0,
    ):
        super().__init__()
        self.active_threshold = active_threshold
        self.w_time = w_time
        self.w_gain = w_gain
        self.w_unmatched_pred = w_unmatched_pred
        self.w_missed_gt = w_missed_gt

    def forward(
        self,
        pred_gains: Tensor,   # [B, T]
        gt_gains: Tensor,     # [B, T]  (ground-truth burst_gain; >0 = onset)
    ) -> tuple[Tensor, dict]:
        """
        Returns:
            loss   – scalar (batch-mean).
            info   – dict with diagnostics (n_matched, n_unmatched, …).
        """
        B, T = pred_gains.shape
        device = pred_gains.device

        # Keep graph-connected zero
        total_loss = (pred_gains * 0).sum()
        n_matched = n_unmatched = n_missed = 0
        sum_time_err = sum_gain_err = 0.0

        for b in range(B):
            # --- predicted onsets ---
            pred_mask = pred_gains[b] > self.active_threshold
            pred_idx = torch.where(pred_mask)[0]
            pred_g = pred_gains[b, pred_idx]
            N = len(pred_idx)

            # --- ground-truth onsets ---
            gt_mask = gt_gains[b] > 0.0
            gt_idx = torch.where(gt_mask)[0]
            gt_g = gt_gains[b, gt_idx]
            M = len(gt_idx)

            if M == 0 and N == 0:
                continue
            if M == 0:
                total_loss = total_loss + self.w_unmatched_pred * pred_g.sum()
                n_unmatched += N
                continue
            if N == 0:
                total_loss = total_loss + self.w_missed_gt * M
                n_missed += M
                continue

            # --- cost matrix & assignment ---
            cost = torch.cdist(
                pred_idx.float().unsqueeze(1),
                gt_idx.float().unsqueeze(1),
            ) / T
            row, col = linear_sum_assignment(cost.detach().cpu().numpy())
            row_t = torch.tensor(row, device=device, dtype=torch.long)
            col_t = torch.tensor(col, device=device, dtype=torch.long)

            # --- matched pairs ---
            time_err = (pred_idx[row_t].float() - gt_idx[col_t].float()).abs() / T
            gain_err = (pred_g[row_t] - gt_g[col_t]).abs()
            loss_matched = (self.w_time * time_err + self.w_gain * gain_err).sum()

            # --- unmatched / missed ---
            matched_set = set(row.tolist())
            unmatched_idx = [i for i in range(N) if i not in matched_set]
            loss_unmatched = self.w_unmatched_pred * pred_g[unmatched_idx].sum() if unmatched_idx else 0.0
            n_missed_b = M - len(row)
            loss_missed = self.w_missed_gt * n_missed_b

            total_loss = total_loss + loss_matched + loss_unmatched + loss_missed
            n_matched += len(row)
            n_unmatched += len(unmatched_idx)
            n_missed += n_missed_b
            sum_time_err += time_err.sum().item()
            sum_gain_err += gain_err.sum().item()

        loss = total_loss / B

        info = dict(
            n_matched=n_matched / B,
            n_unmatched=n_unmatched / B,
            n_missed=n_missed / B,
            time_err_frames=sum_time_err / max(n_matched, 1) * T,
            gain_err=sum_gain_err / max(n_matched, 1),
        )
        return loss, info

class PLoss(nn.Module):
    """Per-parameter loss between predicted and target synthesis parameters.

    For each parameter the loss type is determined by the ParamSpec registry:

    * **MAE** – frame-wise L1.
    * **LOG_MAE** – L1 on log-normalised logits:
      ``logit = (ln(x) − ln(low)) / (ln(high) − ln(low))``
      This gives perceptually uniform weighting for f0 and dynamic_level.
    * **HUNGARIAN** – bipartite matching loss for sparse burst_gain.

    Args:
        fs:              Sample rate (used to build default registry bounds).
        registry:        Optional pre-built ``{name: ParamSpec}`` dict.
                         If *None*, ``make_default_registry(fs)`` is used.
        weights:         Optional ``{param_name: float}`` loss weights.
                         Missing entries default to 1.0.
        hungarian_kwargs: Extra kwargs forwarded to ``HungarianOnsetLoss``.
    """

    def __init__(
        self,
        fs: int = 16000,
        registry: dict[str, ParamSpec] | None = None,
        weights: dict[str, float] | None = None,
        hungarian_kwargs: dict | None = None,
    ):
        super().__init__()
        self.registry = registry or make_default_registry(fs)
        self.fs = fs
        self.weights = weights or {}
        self.hungarian = HungarianOnsetLoss(**(hungarian_kwargs or {}))

    def _dynamic_level_to_hz(self, dl: Tensor) -> Tensor:
        """Convert dynamic_level ∈ [0, 1] → bandwidth in Hz."""
        return dl * (self.fs / 2.0)

    def _log_mae(self, pred: Tensor, target: Tensor, spec: ParamSpec) -> Tensor:
        """MAE in log-normalised logit space."""
        # For dynamic_level the raw param is in [0, 1]; map to Hz first.
        if spec.name == "dynamic_level":
            pred = self._dynamic_level_to_hz(pred)
            target = self._dynamic_level_to_hz(target)

        pred_logit = spec.to_logit(pred)
        target_logit = spec.to_logit(target)
        return (pred_logit - target_logit).abs().mean()

    def forward(
        self,
        pred_params: dict[str, Tensor],    # {name: [B, num_frames]}
        target_params: dict[str, Tensor],  # {name: [B, num_frames]}
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute weighted sum of per-parameter losses.

        Returns:
            total_loss – scalar.
            breakdown  – ``{param_name: scalar_loss}`` before weighting.
        """
        breakdown: dict[str, Tensor] = {}
        total = torch.tensor(0.0, device=next(iter(pred_params.values())).device)

        for name in PARAM_NAMES:
            if name not in pred_params or name not in target_params:
                continue

            spec = self.registry[name]
            pred = pred_params[name]
            target = target_params[name]
            w = self.weights.get(name, 1.0)

            if spec.loss_type == LossType.MAE:
                loss_i = (pred - target).abs().mean()

            elif spec.loss_type == LossType.LOG_MAE:
                loss_i = self._log_mae(pred, target, spec)

            elif spec.loss_type == LossType.HUNGARIAN:
                loss_i, _info = self.hungarian(pred, target)

            else:
                raise ValueError(f"Unknown loss type: {spec.loss_type}")

            breakdown[name] = loss_i.detach()
            total = total + w * loss_i

        return total, breakdown

if __name__ == "__main__":
    from src.data.synthetic_dataset import SyntheticDataset

    FS = 16000
    NUM_AUDIO_SAMPLES = 64000
    NUM_FRAMES = 250

    print("Generating 4 samples from SyntheticDataset...")
    ds = SyntheticDataset(
        num_samples_per_epoch=4,
        num_audio_samples=NUM_AUDIO_SAMPLES,
        num_frames=NUM_FRAMES,
        fs=FS,
        lti=False,
        random_seed=123,
    )
    samples = [s for s in ds]
    target_audio  = torch.stack([s['audio']  for s in samples])          # [4, 64000]
    target_params = {k: torch.stack([s['params'][k] for s in samples])   # {name: [4, 250]}
                     for k in PARAM_NAMES}

    print(f"  target_audio:  {target_audio.shape}")
    print(f"  target_params: { {k: v.shape for k, v in target_params.items()} }")

    pred_params = {}
    for k, v in target_params.items():
        if k == "burst_gain":
            # Shift onsets by a couple of frames and scale gain slightly
            pred_params[k] = torch.roll(v, shifts=12, dims=1) * 0.9
        elif k == "f0":
            pred_params[k] = v * (1.0 + 0.02 * torch.randn_like(v))
        else:
            pred_params[k] = (v + 0.05 * torch.randn_like(v)).clamp(min=0.0)

    print("\n─── PLoss ───")
    ploss = PLoss(
        fs=FS,
        weights={"f0": 2.0, "burst_gain": 5.0},
    )
    total, breakdown = ploss(pred_params, target_params)
    print(f"  total:  {total.item():.4f}")
    for k, v in breakdown.items():
        print(f"  {k:20s}: {v.item():.4f}")

    print("\n─── Gradient flow check ───")
    for p in pred_params.values():
        p.requires_grad_(True)
    total_g, _ = ploss(pred_params, target_params)
    total_g.backward()
    for k, v in pred_params.items():
        grad_norm = v.grad.norm().item() if v.grad is not None else 0.0
        print(f"  {k:20s} grad norm: {grad_norm:.6f}")

    print("\n─── MultiScaleSpectralLoss ───")
    mss = MultiScaleSpectralLoss()
    mss_val = mss(target_audio, target_audio + 0.01 * torch.randn_like(target_audio))
    print(f"  mss_val:  {mss_val.item():.4f}")

    print("\n─── MultiScaleSpectralLoss ───")
    sot = SOT2048Loss()
    sot_val = sot(target_audio, target_audio + 0.01 * torch.randn_like(target_audio))
    print(f"  sot_val:  {sot_val.item():.4f}")

    print("\n✓ All smoke tests passed.")