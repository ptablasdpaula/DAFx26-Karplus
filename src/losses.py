from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal.windows
from torch import Tensor
from scipy.optimize import linear_sum_assignment
from sot import Wasserstein1DLoss

from src.synths.param_registry import (
    ParamSpec, PARAM_NAMES, make_default_registry, LossType,
    EVENT_PARAM_NAMES,
    F0_MIN_HZ, F0_MAX_HZ,
    PLUCK_POSITION_MIN, PLUCK_POSITION_MAX,
    DYNAMIC_LEVEL_MIN, DYNAMIC_LEVEL_MAX,
    DAMPING_MIN, DAMPING_MAX,
    DECAY_MIN, DECAY_MAX,
    BURST_GAIN_MAX,
)

EPS = 1e-8

def _sigmoid_range(x: Tensor, lo: float, hi: float) -> Tensor:
    """Maps unbounded logits to the physical range [lo, hi]."""
    return lo + (hi - lo) * torch.sigmoid(x)

def mu_law(x: Tensor, mu: float = 255.0) -> Tensor:
    """μ-law compression:  F(x) = ln(1 + μx) / ln(1 + μ)."""
    return torch.log1p(mu * x) / math.log1p(mu)

def inverse_mu_law(x: Tensor, mu: float = 255.0) -> Tensor:
    """Inverse μ-law compression:  G(x) = μ_law(1 − x)."""
    return mu_law((1.0 - x).clamp(min=0.0, max=1.0), mu)

def compute_harmonic_rt60(
        f0: torch.Tensor,
        a1: torch.Tensor,
        g: torch.Tensor,
        fs: float = 16000.0,
        n_harmonics: int = 10
) -> torch.Tensor:
    """
    Differentiable RT60 calculation for the first N harmonics of a KS string.
    Returns: [B, num_events, n_harmonics] tensor of RT60 times in seconds.
    """
    # Create harmonic indices: [1, 2, 3, ..., n_harmonics]
    k = torch.arange(1, n_harmonics + 1, device=f0.device).view(1, 1, -1)

    # Frequencies of the harmonics: f_k = k * f0
    freqs = f0.unsqueeze(-1) * k  # [B, N, H]

    # Mask out harmonics that exceed Nyquist (fs / 2)
    valid_mask = (freqs < fs / 2).float()

    # Convert to radians/sample
    omega = 2.0 * math.pi * freqs / fs
    z_inv = torch.exp(-1j * omega)

    # Loop filter Transfer Function: H_loop(z) = g * (1 - a1) / (1 - a1 * z^-1)
    num = g.unsqueeze(-1) * (1.0 - a1.unsqueeze(-1))
    den = 1.0 - a1.unsqueeze(-1) * z_inv

    # Magnitude response of the loop filter at these exact frequencies
    H_mag = torch.abs(num / den)

    # Clamp to prevent log(0) or infinite RT60 if H_mag >= 1.0
    H_mag = torch.clamp(H_mag, min=1e-6, max=1.0 - 1e-6)

    # RT60 formula derived from loop attenuation
    rt60_seconds = -3.0 / (f0.unsqueeze(-1) * torch.log10(H_mag))

    return rt60_seconds * valid_mask

def sigmoid_focal_loss(
        inputs: Tensor,
        targets: Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
) -> Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example (raw logits).
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 or 1).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25.
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor (scalar).
    """
    # Compute standard Cross Entropy Loss (numerically stable using logits)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    # Compute the predicted probabilities
    p = torch.sigmoid(inputs)

    # p_t is the probability of the true class
    p_t = p * targets + (1 - p) * (1 - targets)

    # Modulating factor: (1 - p_t)^gamma
    loss = ce_loss * ((1 - p_t) ** gamma)

    # Alpha weighting
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()

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
        assert x.dim() == 2 == x_target.dim(), (
            f"only mono audio is supported. Got {x.shape} and {x_target.shape}"
        )

        dists = []
        for fft_size, hop_size, wl in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            win = getattr(self, f"win_{wl}")

            S = torch.stft(
                x,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=wl,
                window=win,
                return_complex=True,
            )

            S_target = torch.stft(
                x_target,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=wl,
                window=win,
                return_complex=True,
            )

            Sx = (S.real**2 + S.imag**2 + EPS).sqrt()
            Sx_target = (S_target.real**2 + S_target.imag**2 + EPS).sqrt()

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
    def __init__(self, sample_rate: int = 16000, reduce: bool = True):
        super().__init__()
        self.sample_rate = sample_rate
        self.reduce = reduce
        self.loss_fn = None
        self._device = None

    def _build(self, device: torch.device):
        self.loss_fn = Wasserstein1DLoss(
            transform="stft",
            fft_size=2048,
            hop_length=256,
            sample_rate=self.sample_rate,
            window="flattop",
            square_magnitude=True,
            p=2,
            apply_root=False,
            normalize=True,
            balanced=False,
            quantile_lowpass=True,
            reduce=self.reduce,
            device=device,
        )
        self._device = device

    def forward(self, x: Tensor, x_target: Tensor) -> Tensor:
        assert x.dim() == 2 == x_target.dim(), (
            f"only mono audio is supported. Got {x.shape} and {x_target.shape}"
        )

        if self.loss_fn is None or self._device != x.device:
            self._build(x.device)

        return self.loss_fn(x, x_target)


class EventSetLoss(nn.Module):
    """
    Bipartite matching loss for DETR-style Event Packets.

    Matches a set of N predicted events to N ground-truth events (padded with
    exists=0). Matching is based on Time, Pitch, and Gain.
    """

    def __init__(
            self,
            # ── Matching Weights (used to build the cost matrix) ──
            cost_class: float = 1.0,
            cost_time: float = 1.0,
            cost_f0: float = 1.0,
            cost_gain: float = 0.5,
            # ── Loss Weights (used for backprop after matching) ──
            w_exists: float = 2.0,
            w_time: float = 5.0,
            w_f0: float = 2.0,
            w_gain: float = 1.0,
            w_rt60: float = 1.0,  # <-- REPLACES w_decay AND w_damping
            w_pluck: float = 0.5,
            w_dyn: float = 0.5,
            mu: float = 255.0,
            f0_min_hz: float = 32.0,
            f0_max_hz: float = 2000.0,
            n_f0_bins: int = 128,
            fs: float = 16000.0,  # <-- ADDED sample rate
    ):
        super().__init__()
        self.cost_time = cost_time
        self.cost_f0 = cost_f0
        self.cost_gain = cost_gain
        self.cost_class = cost_class

        self.w_exists = w_exists
        self.w_time = w_time
        self.w_f0 = w_f0
        self.w_gain = w_gain
        self.w_rt60 = w_rt60
        self.w_pluck = w_pluck
        self.w_dyn = w_dyn
        self.mu = mu
        self.fs = fs

        bin_centres = f0_min_hz * (
                (f0_max_hz / f0_min_hz)
                ** (torch.arange(n_f0_bins, dtype=torch.float32) / (n_f0_bins - 1))
        )
        self.register_buffer("f0_bin_centres", bin_centres)

    def _hz_to_bin_idx(self, f0_hz: Tensor) -> Tensor:
        """Converts ground-truth Hz to the nearest bin index for CE loss."""
        diffs = torch.abs(f0_hz.unsqueeze(-1) - self.f0_bin_centres)
        return torch.argmin(diffs, dim=-1)

    @torch.no_grad()
    def _match(self, pred: dict[str, Tensor], tgt: dict[str, Tensor]) -> list[tuple[Tensor, Tensor]]:
        B, num_queries = pred["exists"].shape[:2]

        p_exists = torch.sigmoid(pred["exists"]).squeeze(-1)  # [B, N]
        p_time = torch.sigmoid(pred["time"]).squeeze(-1)  # [B, N]
        p_gain = torch.sigmoid(pred["params"][..., 0])  # [B, N]
        p_f0 = pred["f0_hz"].squeeze(-1)  # [B, N]

        t_time = tgt["time"]
        t_gain = tgt["burst_gain"]
        t_f0 = tgt["f0"]
        t_exists = tgt["exists"]

        indices = []
        for b in range(B):
            tgt_idx = torch.nonzero(t_exists[b]).squeeze(-1)

            if len(tgt_idx) == 0:
                indices.append((torch.arange(num_queries), torch.arange(num_queries)))
                continue

            tgt_t = t_time[b, tgt_idx]
            tgt_g = t_gain[b, tgt_idx]
            tgt_f = t_f0[b, tgt_idx]

            out_exists = p_exists[b]  # [num_queries]
            out_t = p_time[b]
            out_g = p_gain[b]
            out_f = p_f0[b]

            cost_time = torch.cdist(out_t.unsqueeze(-1), tgt_t.unsqueeze(-1), p=1)
            cost_f0 = torch.cdist(out_f.unsqueeze(-1), tgt_f.unsqueeze(-1), p=1) / 1000.0

            out_g_mu = mu_law(out_g, self.mu)
            tgt_g_mu = mu_law(tgt_g, self.mu)
            cost_gain = torch.cdist(out_g_mu.unsqueeze(-1), tgt_g_mu.unsqueeze(-1), p=1)

            # ── THE FIX: Subtract the predicted probability from the cost ──
            # If a query is highly confident (prob -> 1.0), its cost drops,
            # strongly encouraging the matcher to pair it with a real note
            cost_class = -out_exists.unsqueeze(-1)  # Broadcasts to [num_queries, num_real_targets]

            # Total cost matrix
            C = (self.cost_time * cost_time +
                 self.cost_f0 * cost_f0 +
                 self.cost_gain * cost_gain +
                 self.cost_class * cost_class)  # ADDED COST CLASS HERE

            C = C.cpu().numpy()

            row_ind, col_ind = linear_sum_assignment(C)

            # [Keep the rest of the matching logic exactly as it was]
            actual_tgt_ind = tgt_idx[col_ind]
            # ...

            # We must assign the unmatched queries to the dummy padded targets.
            unmatched_queries = set(range(num_queries)) - set(row_ind)
            unmatched_tgts = set(range(num_queries)) - set(actual_tgt_ind.tolist())

            full_row = torch.cat([torch.tensor(row_ind), torch.tensor(list(unmatched_queries))])
            full_col = torch.cat([actual_tgt_ind, torch.tensor(list(unmatched_tgts))])

            # Sort by row (query index) so everything stays aligned
            sort_idx = torch.argsort(full_row)
            indices.append((full_row[sort_idx], full_col[sort_idx]))

        return indices

    def forward(self, pred_raw: dict[str, Tensor], tgt: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """
        Computes the matching and the final loss.
        """
        device = pred_raw["exists"].device

        # ── THE FIX 1: Extract num_queries here! ──
        B, num_queries = pred_raw["exists"].shape[:2]

        # 1. Bipartite Matching
        indices = self._match(pred_raw, tgt)

        # 2. Reorder targets to match predictions
        tgt_reordered = {
            k: torch.zeros(B, num_queries, device=device, dtype=v.dtype)
            for k, v in tgt.items()
        }

        for b in range(B):
            _, tgt_idx = indices[b]

            # ── THE FIX: Force the indices to be integers! ──
            tgt_idx = tgt_idx.long()

            for k in tgt.keys():
                tgt_reordered[k][b] = tgt[k][b][tgt_idx]

        # 3. Compute Existence Loss (FOCAL LOSS)
        # Replaces BCE + pos_weight. Gamma=2.0 dynamically penalizes hard examples
        # (missing notes) while zeroing out the loss for obvious silence.
        exists_loss = sigmoid_focal_loss(
            pred_raw["exists"].squeeze(-1),
            tgt_reordered["exists"],
            alpha=0.5,
            gamma=2.0
        )

        # 4. Compute Physical Parameter Losses
        # We only compute parameter losses for queries that matched a REAL event (exists=1)
        real_mask = tgt_reordered["exists"] > 0

        if not real_mask.any():
            return self.w_exists * exists_loss, {"exists_loss": exists_loss.item(), "total": exists_loss.item()}

        # Extract predictions for real events
        p_time = torch.sigmoid(pred_raw["time"]).squeeze(-1)[real_mask]
        p_f0_probs = pred_raw["f0_probs"][real_mask]
        p_params = pred_raw["params"][real_mask]

        # Apply strict physical boundaries to match the decoder's reality
        p_bg = _sigmoid_range(p_params[..., 0], 0.0, BURST_GAIN_MAX)
        p_decay = _sigmoid_range(p_params[..., 1], DECAY_MIN, DECAY_MAX)
        p_a1 = _sigmoid_range(p_params[..., 2], DAMPING_MIN, DAMPING_MAX)
        p_pluck = _sigmoid_range(p_params[..., 3], PLUCK_POSITION_MIN, PLUCK_POSITION_MAX)
        p_dyn = _sigmoid_range(p_params[..., 4], DYNAMIC_LEVEL_MIN, DYNAMIC_LEVEL_MAX)

        # Extract targets for real events
        t_time = tgt_reordered["time"][real_mask]
        t_f0 = tgt_reordered["f0"][real_mask]
        t_bg = tgt_reordered["burst_gain"][real_mask]
        t_decay = tgt_reordered["decay"][real_mask]
        t_a1 = tgt_reordered["a1"][real_mask]
        t_pluck = tgt_reordered["pluck_position"][real_mask]
        t_dyn = tgt_reordered["dynamic_level"][real_mask]

        # Losses
        time_loss = F.l1_loss(p_time, t_time)

        # f0 Loss: CrossEntropy on the bins
        t_f0_bins = self._hz_to_bin_idx(t_f0)
        f0_loss = F.nll_loss(torch.log(p_f0_probs + 1e-8), t_f0_bins)

        # ── PERCEPTUAL TIMBRE LOSS (Unified RT60) ──
        # Calculate the RT60 across 10 harmonics for predictions and targets.
        # Notice we feed the TARGET pitch (t_f0) to both, so the loss gradient
        # is strictly isolated to correcting the a1 and decay (g) parameters!
        p_rt60 = compute_harmonic_rt60(t_f0, p_a1, p_decay, fs=self.fs, n_harmonics=10)
        t_rt60 = compute_harmonic_rt60(t_f0, t_a1, t_decay, fs=self.fs, n_harmonics=10)

        # Log-space L1 loss (because differences in short decays are perceptually
        # more obvious than identical differences in long decays).
        rt60_loss = F.l1_loss(torch.log1p(p_rt60), torch.log1p(t_rt60))

        # Physical param losses (Gain, Pluck, Dynamics)
        gain_loss = F.l1_loss(mu_law(p_bg, self.mu), mu_law(t_bg, self.mu))
        pluck_loss = F.l1_loss(p_pluck, t_pluck)
        dyn_loss = F.l1_loss(p_dyn, t_dyn)

        # Total Weighted Loss
        total_loss = (
                self.w_exists * exists_loss +
                self.w_time * time_loss +
                self.w_f0 * f0_loss +
                self.w_gain * gain_loss +
                self.w_rt60 * rt60_loss +  # <-- REPLACES w_decay and w_damping
                self.w_pluck * pluck_loss +
                self.w_dyn * dyn_loss
        )

        breakdown = {
            "total": total_loss.item(),
            "exists": exists_loss.item(),
            "time": time_loss.item(),
            "f0_ce": f0_loss.item(),
            "gain_mu": gain_loss.item(),
            "rt60": rt60_loss.item(),
            "pluck": pluck_loss.item(),
            "dyn": dyn_loss.item(),
        }

        return total_loss, breakdown

# =============================================================================
#                           TESTS
# =============================================================================
if __name__ == "__main__":
    from src.synths.param_registry import EVENT_PARAM_NAMES

    B, num_queries = 2, 40
    d_model = 64

    pred_raw = {
        "exists": torch.randn(B, num_queries, 1),
        "time": torch.randn(B, num_queries, 1),
        "f0_probs": F.softmax(torch.randn(B, num_queries, 128), dim=-1),
        "f0_hz": torch.rand(B, num_queries, 1) * 1000 + 100,
        "params": torch.randn(B, num_queries, 5)
    }

    tgt = {k: torch.zeros(B, num_queries) for k in EVENT_PARAM_NAMES}

    tgt["exists"][0, 0] = 1.0
    tgt["time"][0, 0] = 0.5
    tgt["f0"][0, 0] = 440.0
    tgt["burst_gain"][0, 0] = 0.8
    tgt["decay"][0, 0] = 0.99

    tgt["exists"][1, 0:2] = 1.0
    tgt["time"][1, 0] = 0.2
    tgt["f0"][1, 0] = 220.0
    tgt["burst_gain"][1, 0] = 0.5

    tgt["time"][1, 1] = 0.8
    tgt["f0"][1, 1] = 880.0
    tgt["burst_gain"][1, 1] = 0.9

    print("Testing EventSetLoss...")
    loss_fn = EventSetLoss()
    total_loss, breakdown = loss_fn(pred_raw, tgt)

    print(f"\nTotal Loss: {total_loss.item():.4f}")
    for k, v in breakdown.items():
        if k != "total":
            print(f"  {k:15s}: {v:.4f}")

    for tensor in pred_raw.values():
        tensor.requires_grad_(True)

    total_loss, _ = loss_fn(pred_raw, tgt)
    total_loss.backward()

    print("\nGradient Check:")
    for k, v in pred_raw.items():
        grad_norm = v.grad.norm().item() if v.grad is not None else 0.0
        print(f"  {k:15s} grad_norm: {grad_norm:.6f}")

    print("\n✓ Smoke test passed.")