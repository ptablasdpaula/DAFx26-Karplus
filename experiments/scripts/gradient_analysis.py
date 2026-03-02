#!/usr/bin/env python3
"""
Triangular Kernel: Autograd Gradient Accuracy Analysis

Optimised version: uses forward-mode AD (torch.func.jvp) to compute the
Jacobian dY/dp once per init point, then derives ALL (target × kernel)
gradient signs analytically via matrix multiplies on GPU.

Complexity reduction per CGA method:
  Old:  N_init_batches × N_targets × K  backward passes (with retain_graph)
  New:  N_init_batches × 1  jvp pass  +  cheap matmuls

For N=2000, K=4: ~8000× fewer autograd passes.

Falls back to reverse-mode autograd if jvp is incompatible with the synth.

Usage:
  python gradient_analysis_fast.py                                 # defaults
  python gradient_analysis_fast.py compute.device=cuda             # override device
  python gradient_analysis_fast.py compute.batch_size=200          # larger batches now!
  python gradient_analysis_fast.py grid.n_points=2000              # full grid
"""

import os
import sys
import time

# ── Ensure the project root is on sys.path ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
import pandas as pd

from synths.synth import Synth, SynthConfig


# ═══════════════════════════════════════════════════════════
# Helpers  (unchanged)
# ═══════════════════════════════════════════════════════════

def resolve_device(requested: str) -> str:
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU")
        return 'cpu'
    return requested


def make_params(cfg: DictConfig, device: str, **overrides):
    t = cfg.target
    nf = cfg.audio.num_frames
    defaults = {
        'f0': t.f0, 'pluck_position': t.pluck_position,
        'a1': t.a1, 'decay': t.decay, 'dynamic_level': t.dynamic_level,
    }
    params = {}
    for k, v in defaults.items():
        val = overrides.get(k, v)
        if isinstance(val, (int, float)):
            params[k] = torch.full((1, nf), float(val), device=device)
        else:
            params[k] = val.to(device) if isinstance(val, torch.Tensor) else val
    if 'burst_gain' in overrides:
        params['burst_gain'] = overrides['burst_gain']
        if isinstance(params['burst_gain'], torch.Tensor):
            params['burst_gain'] = params['burst_gain'].to(device)
    else:
        params['burst_gain'] = torch.zeros(1, nf, device=device)
        params['burst_gain'][0, 0] = t.burst_gain
    return params


def make_params_batch(cfg: DictConfig, n_batch: int, param_name: str,
                      param_values, device: str):
    t = cfg.target
    nf = cfg.audio.num_frames
    defaults = {
        'f0': t.f0, 'pluck_position': t.pluck_position,
        'a1': t.a1, 'decay': t.decay, 'dynamic_level': t.dynamic_level,
    }
    params = {}
    for k, v in defaults.items():
        if k == param_name:
            params[k] = param_values
        else:
            params[k] = torch.full((n_batch, nf), float(v), device=device)
    params['burst_gain'] = torch.zeros(n_batch, nf, device=device)
    params['burst_gain'][:, 0] = t.burst_gain
    return params

def debug_complex_error(cfg, synth, grid_np, param_name, device):
    """Triggers a loud traceback to find Real->Complex boundary issues."""
    print("\n" + "!" * 60)
    print("!!! HERE IS THE ERROR !!!")
    print("Attempting to trace the exact line causing the jvp failure...")
    print("!" * 60 + "\n")
    
    import traceback
    from torch.autograd.forward_ad import dual_level, make_dual
    
    try:
        with dual_level():
            # Test with a single point from the grid
            test_val = torch.tensor([grid_np[0]], dtype=torch.float32, 
                                    device=device, requires_grad=True)
            test_tangent = torch.ones_like(test_val)
            
            # Create the dual tensor (primal + tangent)
            dual_p = make_dual(test_val, test_tangent)
            
            # Wrap in batch format for the synth
            params = make_params_batch(cfg, 1, param_name, dual_p.unsqueeze(1), device=device)
            
            # This should trigger the complex-type error if it exists
            _ = synth(params)
            
    except Exception:
        traceback.print_exc()
        print("\n" + "!" * 60)
        print("END OF ERROR TRACE")
        print("Look for the last line in 'synths/ddsp.py' or 'synths/synth.py'")
        print("!" * 60 + "\n")


def triangular_kernel_spectrum(length: int, n_fft: int, device: str):
    if length == 1:
        return torch.ones(n_fft // 2 + 1, device=device)
    n = torch.arange(length, dtype=torch.float32, device=device)
    center = (length - 1) / 2
    kernel = 1.0 - torch.abs(n - center) / center
    padded = torch.zeros(n_fft, device=device)
    padded[:length] = kernel
    return torch.fft.rfft(padded).abs()


# ── Grid builders ──

def make_f0_grid(cfg: DictConfig):
    g = cfg.grid
    fs = cfg.audio.fs
    if g.snap_to_integer_samples:
        D_min = int(np.ceil(fs / g.f0_max))
        D_max = int(np.floor(fs / g.f0_min))
        D_vals = np.arange(D_min, D_max + 1)
        grid = np.sort(fs / D_vals.astype(float))
        print(f"  f0 coarse grid (integer delay): {len(grid)} pts, "
              f"D in [{D_min}, {D_max}]")
    else:
        grid = np.geomspace(g.f0_min, g.f0_max, num=g.n_points)
    return grid


def make_pluck_grid(cfg: DictConfig):
    g = cfg.grid
    if g.snap_to_integer_samples:
        L = cfg.audio.fs / cfg.target.f0
        k_min, k_max = 2, int(np.floor(L))
        k_vals = np.arange(k_min, k_max + 1)
        grid = (k_vals - 1.0) / (L - 1.0)
        grid = grid[(grid >= g.pluck_min) & (grid <= g.pluck_max)]
        print(f"  pluck coarse grid (integer comb_L): {len(grid)} pts, "
              f"k in [{k_min}, {k_max}]")
    else:
        grid = np.linspace(g.pluck_min, g.pluck_max, g.n_points)
    return grid


def make_f0_offset_grid(cfg: DictConfig):
    half = cfg.fga.f0_half_cents
    return np.linspace(-half, half, num=cfg.grid.n_points)


def make_pluck_offset_grid(cfg: DictConfig):
    half = cfg.fga.pluck_half
    return np.linspace(-half, half, num=cfg.grid.n_points)


def f0_offset_to_absolute(target_f0, offset_cents):
    return target_f0 * 2.0 ** (offset_cents / 1200.0)


def pluck_offset_to_absolute(target_pluck, offset_pluck):
    return target_pluck + offset_pluck


# ── Metrics ──

def hz_to_cents(f0_hz, ref_hz):
    return 1200.0 * np.log2(f0_hz / ref_hz)


def off_diag_mask(n):
    return np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) > 1


def fga_off_centre_mask(n_targets, n_offsets):
    centre = n_offsets // 2
    offset_indices = np.arange(n_offsets)
    near_centre = np.abs(offset_indices - centre) <= 1
    mask = np.ones((n_targets, n_offsets), dtype=bool)
    mask[:, near_centre] = False
    return mask


def cga_from_map(binary_map):
    n = binary_map.shape[0]
    mask = off_diag_mask(n)
    return binary_map[mask].mean()


def fga_from_map(binary_map, valid_mask=None):
    n_targets, n_offsets = binary_map.shape
    mask = fga_off_centre_mask(n_targets, n_offsets)
    if valid_mask is not None:
        mask = mask & valid_mask
    if mask.sum() == 0:
        return 0.0
    return binary_map[mask].mean()


def pluck_to_delay_samples(pluck_pos, f0, fs):
    return 1.0 + pluck_pos * (fs / f0 - 1)


def log_normalise(arr):
    sign = np.sign(arr)
    log_abs = np.log1p(np.abs(arr))
    mx = np.nanmax(log_abs)
    return sign * log_abs / mx if mx > 1e-12 else np.zeros_like(arr)


# ═══════════════════════════════════════════════════════════
# Plotting  (unchanged)
# ═══════════════════════════════════════════════════════════

def plot_cga_2x2(axes, data_maps, method_keys, grids, titles, xlabels, ylabels,
                 cmap, vmin, vmax, log_axes_flags, metric_name=None):
    for ax, key, grid, title, xlabel, ylabel, use_log in zip(
        axes.flat, method_keys, grids, titles, xlabels, ylabels, log_axes_flags
    ):
        display_data = np.nan_to_num(data_maps[key].T, nan=0.0)
        im = ax.pcolormesh(grid, grid, display_data,
                           cmap=cmap, vmin=vmin, vmax=vmax)
        ax.plot([grid[0], grid[-1]], [grid[0], grid[-1]],
                color='cyan', ls='--', linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if use_log:
            ax.set_xscale('log')
            ax.set_yscale('log')
        if metric_name and cmap == 'gray':
            n = len(grid)
            mask = off_diag_mask(n)
            pct = data_maps[key][mask].mean() * 100
            ax.text(0.02, 0.98, f"{metric_name}: {pct:.1f}%",
                    transform=ax.transAxes, fontsize=11, va='top', ha='left',
                    color='cyan',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='black', alpha=0.7))
        elif cmap != 'gray':
            plt.colorbar(im, ax=ax, label='Log gradient (+ = correct)')


def plot_fga_2x2(axes, data_maps, method_keys, target_grids, offset_grids,
                 titles, xlabels, ylabels, cmap, vmin, vmax,
                 log_x_flags, metric_name=None, valid_masks=None):
    for ax, key, tgrid, ogrid, title, xlabel, ylabel, log_x in zip(
        axes.flat, method_keys, target_grids, offset_grids,
        titles, xlabels, ylabels, log_x_flags
    ):
        data = data_maps[key]
        display_data = np.nan_to_num(data.T, nan=0.0)
        im = ax.pcolormesh(tgrid, ogrid, display_data,
                           cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axhline(0, color='cyan', ls='--', linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if log_x:
            ax.set_xscale('log')
        if metric_name and cmap == 'gray':
            vm = valid_masks.get(key) if valid_masks else None
            pct = fga_from_map(data, valid_mask=vm) * 100
            ax.text(0.02, 0.98, f"{metric_name}: {pct:.1f}%",
                    transform=ax.transAxes, fontsize=11, va='top', ha='left',
                    color='cyan',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='black', alpha=0.7))
        elif cmap != 'gray':
            plt.colorbar(im, ax=ax, label='Log gradient (+ = correct)')


# ═══════════════════════════════════════════════════════════
# Batched target-FFT helper (shared by both CGA and FGA)
# ═══════════════════════════════════════════════════════════

def compute_target_ffts(cfg, synth, grid_np, param_name, device,
                        batch_size, loss_n_fft, label=''):
    """Synthesise all target signals and return their FFTs [N, F]."""
    n = len(grid_np)
    effective_batch = n if (batch_size <= 0 or batch_size >= n) else batch_size
    ffts = []
    with torch.no_grad():
        for i in tqdm(range(0, n, effective_batch),
                      desc=f"  [{label}] target FFTs",
                      total=(n + effective_batch - 1) // effective_batch):
            end = min(i + effective_batch, n)
            vals = torch.tensor(
                grid_np[i:end], dtype=torch.float32, device=device
            ).unsqueeze(1)
            params = make_params_batch(cfg, end - i, param_name, vals,
                                       device=device)
            y = synth(params)
            ffts.append(torch.fft.rfft(y, n=loss_n_fft))
    return torch.cat(ffts, dim=0)


# ═══════════════════════════════════════════════════════════
# JVP-based CGA  (the fast path)
# ═══════════════════════════════════════════════════════════

def _cga_jvp_core(cfg, synth, grid_np, param_name, target_ffts,
                   kernel_sq, kernel_list, device, batch_size,
                   loss_n_fft, method_name):
    """
    Core CGA computation using forward-mode AD.

    For each batch of init points:
      1. jvp gives Y[B,F] and dY/dp[B,F] in one forward+tangent pass
      2. Gradient for (init i, target j, kernel k) is:
           g = Re(Σ_f  w_k²[f] · conj(Y[i,f] − T[j,f]) · dY/dp[i,f])
         Factor into target-independent + target-dependent terms:
           A[i,k]   = Re(Σ_f  w_k²·conj(Y)·dY/dp)      — no j
           B[i,k,j] = Re(Σ_f  w_k²·conj(T[j])·dY/dp)    — matmul
           g[i,k,j] = A[i,k] − B[i,k,j]
      3. Correctness: init > target ⟹ gradient should be positive (for
         increasing-loss direction), so sign(g) · sign(i−j) > 0.

    Also computes per-target min-loss errors and losses for the full grid.
    """
    from torch.func import jvp

    n = len(grid_np)
    K = len(kernel_list)
    F = target_ffts.shape[1]
    effective_batch = n if (batch_size <= 0 or batch_size >= n) else batch_size

    # Output arrays
    correct = {kl: np.zeros((n, n), dtype=bool) for kl, _ in kernel_list}
    intens = {kl: np.full((n, n), np.nan) for kl, _ in kernel_list}
    min_loss_errors = {kl: np.full(n, np.inf) for kl, _ in kernel_list}

    # Mark diagonal band as correct
    for kl, _ in kernel_list:
        for i in range(n):
            for j in range(max(0, i - 1), min(n, i + 2)):
                correct[kl][i, j] = True

    # Precompute target terms for min-loss:
    # |w·T|²  per target per kernel → [K, N]
    with torch.no_grad():
        T_sq_weighted = torch.einsum(
            'nf,kf->kn', target_ffts.abs().pow(2), kernel_sq
        ) / F  # [K, N]

    # Pre-transpose target_ffts for matmul: [F, N]
    target_ffts_conj_T = target_ffts.conj().T.contiguous()  # [F, N]

    pbar = tqdm(total=(n + effective_batch - 1) // effective_batch,
                desc=f"  [{method_name}] jvp batches")

    for b_start in range(0, n, effective_batch):
        b_end = min(b_start + effective_batch, n)
        B = b_end - b_start

        grid_param = torch.tensor(
            grid_np[b_start:b_end], dtype=torch.float32, device=device
        ).unsqueeze(1)  # [B, 1]

        # ── jvp: forward + tangent in one pass ──
        def fwd(p):
            params = make_params_batch(cfg, B, param_name, p, device=device)
            return synth(params)

        tangent = torch.ones_like(grid_param)
        y, dy_dp = jvp(fwd, (grid_param,), (tangent,))  # both [B, T] (real)
        Y = torch.fft.rfft(y, n=loss_n_fft)             # [B, F] (complex)
        dY_dp = torch.fft.rfft(dy_dp, n=loss_n_fft)     # [B, F] (complex)

        with torch.no_grad():
            # ── Term A: [B, K]  (target-independent) ──
            # A[i,k] = Re(Σ_f w²[k,f] · conj(Y[i,f]) · dY/dp[i,f])
            YdY = Y.conj() * dY_dp                        # [B, F]
            A = torch.einsum('bf,kf->bk', YdY, kernel_sq).real  # [B, K]

            # ── Term B: [B, K, N]  (target-dependent, via matmul) ──
            # P[i,k,f] = w²[k,f] · dY/dp[i,f]
            P = kernel_sq.unsqueeze(0) * dY_dp.unsqueeze(1)  # [B, K, F]
            # B[i,k,j] = Re(Σ_f P[i,k,f] · conj(T[j,f]))
            #           = Re(P @ conj(T)^T)
            P_flat = P.reshape(B * K, F)                      # [B·K, F]
            B_term = (P_flat @ target_ffts_conj_T).real       # [B·K, N]
            B_term = B_term.reshape(B, K, n)                  # [B, K, N]

            # ── Gradient values: g[i,k,j] = A[i,k] − B[i,k,j] ──
            grad_vals = A.unsqueeze(-1) - B_term              # [B, K, N]

            # ── Min-loss tracking ──
            # loss[i,k,j] = mean |w_k (Y_i − T_j)|²
            #             = mean(w²|Y|²) + mean(w²|T|²) − 2·Re(mean(w²·Y·conj(T)))
            Y_sq_weighted = torch.einsum(
                'bf,kf->bk', Y.abs().pow(2), kernel_sq
            ) / F  # [B, K]

            # Cross: [B, K, N]
            P2 = kernel_sq.unsqueeze(0) * Y.unsqueeze(1)     # [B, K, F]
            P2_flat = P2.reshape(B * K, F)
            cross = (P2_flat @ target_ffts_conj_T).real / F   # [B·K, N]
            cross = cross.reshape(B, K, n)

            losses = (Y_sq_weighted.unsqueeze(-1)
                      + T_sq_weighted.unsqueeze(0)
                      - 2.0 * cross)                          # [B, K, N]

            # ── Write results (on CPU) ──
            grad_np = grad_vals.cpu().numpy()     # [B, K, N]
            loss_np = losses.cpu().numpy()        # [B, K, N]

        indices = np.arange(b_start, b_end)  # [B]

        for ki, (kl, _) in enumerate(kernel_list):
            gv = grad_np[:, ki, :]           # [B, N]
            lv = loss_np[:, ki, :]           # [B, N]

            # Correctness: for each target j, check init points far from j
            # grad_vals convention: positive g means d(loss)/dp > 0
            # If init > target (indices[i] > j), we expect positive gradient
            # If init < target (indices[i] < j), we expect negative gradient
            # So: correct ⟺ sign(g) == sign(i − j)
            for j in range(n):
                far = np.abs(indices - j) > 1
                if not far.any():
                    continue
                dirs = np.where(indices > j, 1.0, -1.0)
                signed = gv[:, j] * dirs
                intens[kl][j, b_start:b_end][far] = signed[far]
                correct[kl][j, b_start:b_end][far] = signed[far] > 0

            # Min-loss errors: for each target j, find the init with lowest loss
            min_idx = lv.argmin(axis=0)               # [N]
            min_losses = lv[min_idx, np.arange(n)]    # [N]
            for j in range(n):
                bi = min_idx[j]
                gi = b_start + bi
                err = abs(grid_np[gi] - grid_np[j])
                if err < min_loss_errors[kl][j]:
                    min_loss_errors[kl][j] = err

        del Y, dY_dp, P, P_flat, B_term, grad_vals, losses
        torch.cuda.empty_cache()
        pbar.update(1)

    pbar.close()

    mean_errors = {kl: min_loss_errors[kl].mean() for kl, _ in kernel_list}
    return {kl: (correct[kl], intens[kl], mean_errors[kl])
            for kl, _ in kernel_list}


# ═══════════════════════════════════════════════════════════
# Reverse-mode CGA fallback
# ═══════════════════════════════════════════════════════════

def _cga_reverse_core(cfg, synth, grid_np, param_name, target_ffts,
                      kernel_sq, kernel_list, device, batch_size,
                      loss_n_fft, method_name):
    """
    Fallback CGA using reverse-mode autograd.
    Restructured vs original: recompute forward per target-chunk to avoid
    massive retain_graph accumulation.
    """
    n = len(grid_np)
    K = len(kernel_list)
    F = target_ffts.shape[1]
    effective_batch = n if (batch_size <= 0 or batch_size >= n) else batch_size

    correct = {kl: np.zeros((n, n), dtype=bool) for kl, _ in kernel_list}
    intens = {kl: np.full((n, n), np.nan) for kl, _ in kernel_list}
    min_loss_errors = {kl: np.full(n, np.inf) for kl, _ in kernel_list}

    for kl, _ in kernel_list:
        for i in range(n):
            for j in range(max(0, i - 1), min(n, i + 2)):
                correct[kl][i, j] = True

    n_total = n * K
    pbar = tqdm(total=n_total, desc=f"  [{method_name}] reverse-mode sweeps")

    for b_start in range(0, n, effective_batch):
        b_end = min(b_start + effective_batch, n)
        B = b_end - b_start

        grid_param = torch.tensor(
            grid_np[b_start:b_end], dtype=torch.float32, device=device
        ).unsqueeze(1).requires_grad_(True)

        params = make_params_batch(cfg, B, param_name, grid_param, device=device)
        y = synth(params)
        Y = torch.fft.rfft(y, n=loss_n_fft)

        for t_idx in range(n):
            target_fft = target_ffts[t_idx]
            diff = Y - target_fft.unsqueeze(0)

            for ki, (kl, ks) in enumerate(kernel_list):
                weighted = diff * ks.unsqueeze(0)
                losses = (weighted.abs() ** 2).mean(dim=-1)

                with torch.no_grad():
                    batch_min_idx = losses.argmin().item()
                    global_idx = b_start + batch_min_idx
                    err = abs(grid_np[global_idx] - grid_np[t_idx])
                    if err < min_loss_errors[kl][t_idx]:
                        min_loss_errors[kl][t_idx] = err

                total_loss = losses.sum()
                is_last_kernel = (ki == K - 1)
                is_last_target = (t_idx == n - 1)
                retain = not (is_last_kernel and is_last_target)
                grads = torch.autograd.grad(
                    total_loss, grid_param, retain_graph=retain
                )[0]
                grad_np = grads.squeeze(1).detach().cpu().numpy()

                indices = np.arange(b_start, b_end)
                far_mask = np.abs(indices - t_idx) > 1
                directions = np.where(indices > t_idx, 1.0, -1.0)
                signed = grad_np * directions

                intens[kl][t_idx, b_start:b_end][far_mask] = signed[far_mask]
                correct[kl][t_idx, b_start:b_end][far_mask] = signed[far_mask] > 0

                if b_start == 0:
                    pbar.update(1)

        del y, Y, params
        torch.cuda.empty_cache()

    pbar.close()
    mean_errors = {kl: min_loss_errors[kl].mean() for kl, _ in kernel_list}
    return {kl: (correct[kl], intens[kl], mean_errors[kl])
            for kl, _ in kernel_list}


# ═══════════════════════════════════════════════════════════
# CGA dispatcher (try jvp, fall back to reverse)
# ═══════════════════════════════════════════════════════════

def compute_cga_maps(cfg: DictConfig, synth, grid, param_name,
                     kernel_spectra, device: str, batch_size: int,
                     method_name: str = ''):
    """
    CGA: both target and init use the same grid.
    Returns dict: kl → (correct[N,N], intensity[N,N], mean_min_loss_error)
    """
    num_samples = cfg.audio.fs * cfg.audio.duration
    loss_n_fft = num_samples
    n = len(grid)
    grid_np = np.asarray(grid, dtype=np.float64)

    kernel_list = list(kernel_spectra.items())
    K = len(kernel_list)
    kernel_sq = torch.stack([ks ** 2 for _, ks in kernel_list]).to(device)

    # Phase 1: target FFTs
    target_ffts = compute_target_ffts(
        cfg, synth, grid_np, param_name, device, batch_size,
        loss_n_fft, label=f"{method_name}/targets")

    # Phase 2: try jvp, fall back to reverse
    try:
        print(f"  [{method_name}] Trying forward-mode AD (jvp)…")
        result = _cga_jvp_core(
            cfg, synth, grid_np, param_name, target_ffts,
            kernel_sq, kernel_list, device, batch_size,
            loss_n_fft, method_name)
        print(f"  [{method_name}] jvp succeeded ✓")
        return result
    except Exception as e:
        print(f"  [{method_name}] jvp failed: {e}")
        print(f"  [{method_name}] Falling back to reverse-mode autograd…")
        debug_complex_error(cfg, synth, grid_np, param_name, device)
        return _cga_reverse_core(
            cfg, synth, grid_np, param_name, target_ffts,
            kernel_sq, kernel_list, device, batch_size,
            loss_n_fft, method_name)


# ═══════════════════════════════════════════════════════════
# JVP-based FGA
# ═══════════════════════════════════════════════════════════

def _fga_jvp_core(cfg, synth, target_np, offset_np, param_name,
                   offset_to_abs_fn, target_ffts_all, kernel_sq, kernel_list,
                   device, batch_size, loss_n_fft, method_name, valid_range):
    """
    Fast FGA using forward-mode AD.

    For each target:
      1. Compute absolute init values for all offsets
      2. jvp over valid init batch → Y, dY/dp
      3. Gradient sign for kernel k:
           g = −sign(offset) · Re(Σ_f w²·conj(Y_init−Y_target)·dY/dp)
         Positive g ⟹ correct direction.
    """
    from torch.func import jvp

    n_targets = len(target_np)
    n_offsets = len(offset_np)
    K = len(kernel_list)
    centre = n_offsets // 2
    effective_batch = n_offsets if (batch_size <= 0 or batch_size >= n_offsets) else batch_size

    correct = {kl: np.zeros((n_targets, n_offsets), dtype=bool) for kl, _ in kernel_list}
    intens = {kl: np.full((n_targets, n_offsets), np.nan) for kl, _ in kernel_list}
    valid_mask = np.ones((n_targets, n_offsets), dtype=bool)

    for kl, _ in kernel_list:
        for o in range(max(0, centre - 1), min(n_offsets, centre + 2)):
            correct[kl][:, o] = True

    offset_signs = np.sign(offset_np)
    pbar = tqdm(total=n_targets, desc=f"  [{method_name}] FGA-jvp targets")

    for t_idx in range(n_targets):
        target_val = target_np[t_idx]

        # Absolute init values
        abs_inits = np.array([offset_to_abs_fn(target_val, o)
                              for o in offset_np], dtype=np.float64)

        # Validity
        if valid_range is not None:
            lo, hi = valid_range
            offset_valid = (abs_inits >= lo) & (abs_inits <= hi)
            valid_mask[t_idx] = offset_valid
        else:
            offset_valid = np.ones(n_offsets, dtype=bool)

        if not offset_valid.any():
            pbar.update(1)
            continue

        # Target FFT  (precomputed)
        target_fft = target_ffts_all[t_idx]  # [F]

        # Process valid offsets in batches
        valid_idx = np.where(offset_valid)[0]
        valid_abs = abs_inits[valid_idx]

        for vb_start in range(0, len(valid_idx), effective_batch):
            vb_end = min(vb_start + effective_batch, len(valid_idx))
            vB = vb_end - vb_start

            chunk_abs = valid_abs[vb_start:vb_end]
            chunk_global_idx = valid_idx[vb_start:vb_end]

            init_param = torch.tensor(
                chunk_abs, dtype=torch.float32, device=device
            ).unsqueeze(1)  # [vB, 1]

            def fwd(p):
                params = make_params_batch(cfg, vB, param_name, p, device=device)
                return synth(params)

            tangent = torch.ones_like(init_param)
            y, dy_dp = jvp(fwd, (init_param,), (tangent,))
            Y = torch.fft.rfft(y, n=loss_n_fft)
            dY_dp = torch.fft.rfft(dy_dp, n=loss_n_fft)

            with torch.no_grad():
                # Gradient for each kernel:
                # g[o, k] = Re(Σ_f w²[k,f] · conj(Y[o,f]−T[f]) · dY/dp[o,f])
                diff_conj_dY = (Y - target_fft.unsqueeze(0)).conj() * dY_dp  # [vB, F]
                # [vB, K]
                grad_vals = torch.einsum('bf,kf->bk', diff_conj_dY, kernel_sq).real

                grad_np = grad_vals.cpu().numpy()  # [vB, K]

            chunk_signs = offset_signs[chunk_global_idx]  # [vB]

            for ki, (kl, _) in enumerate(kernel_list):
                gv = grad_np[:, ki]                        # [vB]
                signed = -gv * chunk_signs                 # positive = correct

                for vi in range(vB):
                    fi = chunk_global_idx[vi]
                    if abs(fi - centre) > 1:
                        intens[kl][t_idx, fi] = signed[vi]
                        correct[kl][t_idx, fi] = signed[vi] > 0

            del Y, dY_dp
            torch.cuda.empty_cache()

        pbar.update(1)

    pbar.close()

    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"  [{method_name}] {n_invalid}/{valid_mask.size} offset cells "
              f"outside valid_range {valid_range}, excluded")

    results = {kl: (correct[kl], intens[kl]) for kl, _ in kernel_list}
    return results, valid_mask


# ═══════════════════════════════════════════════════════════
# Reverse-mode FGA fallback
# ═══════════════════════════════════════════════════════════

def _fga_reverse_core(cfg, synth, target_np, offset_np, param_name,
                      offset_to_abs_fn, target_ffts_all, kernel_sq, kernel_list,
                      device, batch_size, loss_n_fft, method_name, valid_range):
    """Fallback FGA using reverse-mode autograd."""
    n_targets = len(target_np)
    n_offsets = len(offset_np)
    K = len(kernel_list)
    centre = n_offsets // 2
    effective_batch = n_offsets if (batch_size <= 0 or batch_size >= n_offsets) else batch_size

    correct = {kl: np.zeros((n_targets, n_offsets), dtype=bool) for kl, _ in kernel_list}
    intens = {kl: np.full((n_targets, n_offsets), np.nan) for kl, _ in kernel_list}
    valid_mask = np.ones((n_targets, n_offsets), dtype=bool)

    for kl, _ in kernel_list:
        for o in range(max(0, centre - 1), min(n_offsets, centre + 2)):
            correct[kl][:, o] = True

    offset_signs = np.sign(offset_np)
    pbar = tqdm(total=n_targets * K, desc=f"  [{method_name}] FGA reverse sweeps")

    for t_idx in range(n_targets):
        target_val = target_np[t_idx]
        abs_inits = np.array([offset_to_abs_fn(target_val, o)
                              for o in offset_np], dtype=np.float64)

        if valid_range is not None:
            lo, hi = valid_range
            offset_valid = (abs_inits >= lo) & (abs_inits <= hi)
            valid_mask[t_idx] = offset_valid
        else:
            offset_valid = np.ones(n_offsets, dtype=bool)

        if not offset_valid.any():
            pbar.update(K)
            continue

        target_fft = target_ffts_all[t_idx]

        valid_idx = np.where(offset_valid)[0]
        valid_abs = abs_inits[valid_idx]
        n_valid = len(valid_idx)

        for vb_start in range(0, n_valid, effective_batch):
            vb_end = min(vb_start + effective_batch, n_valid)
            vB = vb_end - vb_start

            chunk_abs = valid_abs[vb_start:vb_end]
            chunk_global_idx = valid_idx[vb_start:vb_end]

            init_param = torch.tensor(
                chunk_abs, dtype=torch.float32, device=device
            ).unsqueeze(1).requires_grad_(True)

            params = make_params_batch(cfg, vB, param_name, init_param, device=device)
            y = synth(params)
            Y = torch.fft.rfft(y, n=loss_n_fft)
            diff = Y - target_fft.unsqueeze(0)

            for ki, (kl, ks) in enumerate(kernel_list):
                weighted = diff * ks.unsqueeze(0)
                losses = (weighted.abs() ** 2).mean(dim=-1)
                total_loss = losses.sum()
                retain = (ki < K - 1)
                grads = torch.autograd.grad(
                    total_loss, init_param, retain_graph=retain
                )[0]
                grad_np = grads.squeeze(1).detach().cpu().numpy()

                batch_signs = offset_signs[chunk_global_idx]
                signed = -grad_np * batch_signs

                for vi in range(vB):
                    fi = chunk_global_idx[vi]
                    if abs(fi - centre) > 1:
                        intens[kl][t_idx, fi] = signed[vi]
                        correct[kl][t_idx, fi] = signed[vi] > 0

                if vb_start == 0:
                    pbar.update(1)

            del y, Y, params
            torch.cuda.empty_cache()

    pbar.close()

    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"  [{method_name}] {n_invalid}/{valid_mask.size} offset cells "
              f"outside valid_range {valid_range}, excluded")

    results = {kl: (correct[kl], intens[kl]) for kl, _ in kernel_list}
    return results, valid_mask


# ═══════════════════════════════════════════════════════════
# FGA dispatcher
# ═══════════════════════════════════════════════════════════

def compute_fga_maps(cfg: DictConfig, synth, target_grid, offset_grid,
                     param_name, offset_to_abs_fn,
                     kernel_spectra, device: str, batch_size: int,
                     method_name: str = '', valid_range=None):
    """
    FGA: x-axis = targets, y-axis = offsets.
    Returns (results_dict, valid_mask).
    """
    num_samples = cfg.audio.fs * cfg.audio.duration
    loss_n_fft = num_samples

    target_np = np.asarray(target_grid, dtype=np.float64)
    offset_np = np.asarray(offset_grid, dtype=np.float64)

    kernel_list = list(kernel_spectra.items())
    kernel_sq = torch.stack([ks ** 2 for _, ks in kernel_list]).to(device)

    # Precompute ALL target FFTs for this grid
    target_ffts = compute_target_ffts(
        cfg, synth, target_np, param_name, device, batch_size,
        loss_n_fft, label=f"{method_name}/targets")

    try:
        print(f"  [{method_name}] Trying forward-mode AD (jvp)…")
        result, vm = _fga_jvp_core(
            cfg, synth, target_np, offset_np, param_name,
            offset_to_abs_fn, target_ffts, kernel_sq, kernel_list,
            device, batch_size, loss_n_fft, method_name, valid_range)
        print(f"  [{method_name}] jvp succeeded ✓")
        return result, vm
    except Exception as e:
        print(f"  [{method_name}] jvp failed: {e}")
        print(f"  [{method_name}] Falling back to reverse-mode autograd…")
        debug_complex_error(cfg, synth, target_np, param_name, device)
        return _fga_reverse_core(
            cfg, synth, target_np, offset_np, param_name,
            offset_to_abs_fn, target_ffts, kernel_sq, kernel_list,
            device, batch_size, loss_n_fft, method_name, valid_range)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

@hydra.main(config_path="../../configs", config_name="gradient_analysis", version_base=None)
def main(cfg: DictConfig):
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(OmegaConf.to_yaml(cfg))

    # ── Derived constants ──
    fs = cfg.audio.fs
    duration = cfg.audio.duration
    num_samples = fs * duration
    loss_n_fft = num_samples
    kernel_lengths = list(cfg.kernel_lengths)

    device = resolve_device(cfg.compute.device)
    batch_size = cfg.compute.batch_size

    print(f"Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Batch size: {'all' if batch_size == 0 else batch_size}")
    print(f"\nSignal: {duration}s @ {fs}Hz = {num_samples} samples")
    print(f"f0 range: {cfg.grid.f0_min}\u2013{cfg.grid.f0_max} Hz")
    print(f"Pluck range: {cfg.grid.pluck_min}\u2013{cfg.grid.pluck_max}")
    print(f"snap_to_integer_samples = {cfg.grid.snap_to_integer_samples}")

    plt.rcParams['figure.dpi'] = 100

    # ── Kernel spectra ──
    kernel_spectra = {
        kl: triangular_kernel_spectrum(kl, loss_n_fft, device=device)
        for kl in kernel_lengths
    }
    print(f"Kernel spectra cached on {device}: {kernel_lengths}")

    # ── Grids ──
    pluck_grid = make_pluck_grid(cfg)
    f0_grid = make_f0_grid(cfg)
    f0_cents = hz_to_cents(f0_grid, cfg.target.f0)

    pluck_offset_grid = make_pluck_offset_grid(cfg)
    f0_offset_grid = make_f0_offset_grid(cfg)

    print(f"\nCoarse grids:")
    print(f"  Pluck: {len(pluck_grid)} pts "
          f"[{pluck_grid[0]:.4f}, {pluck_grid[-1]:.4f}]")
    print(f"  f0:    {len(f0_grid)} pts "
          f"[{f0_grid[0]:.2f}, {f0_grid[-1]:.2f}] Hz "
          f"([{f0_cents[0]:.0f}, {f0_cents[-1]:.0f}] cents)")

    print(f"\nFine offset grids:")
    print(f"  Pluck: {len(pluck_offset_grid)} pts "
          f"\u00b1{cfg.fga.pluck_half} around target")
    print(f"  f0:    {len(f0_offset_grid)} pts "
          f"\u00b1{cfg.fga.f0_half_cents} cents around target")

    # ── Synths ──
    synths = {
        'pluck_td': Synth(SynthConfig(
            num_samples=num_samples, fs=fs, device=device, n_fft=loss_n_fft,
            use_freq_pluck=False, use_freq_ksa=False)),
        'pluck_fd': Synth(SynthConfig(
            num_samples=num_samples, fs=fs, device=device, n_fft=loss_n_fft,
            use_freq_pluck=True, use_freq_ksa=False, use_lti=True)),
        'ks_td': Synth(SynthConfig(
            num_samples=num_samples, fs=fs, device=device, n_fft=loss_n_fft,
            use_freq_pluck=False, use_freq_ksa=False)),
        'ks_fd': Synth(SynthConfig(
            num_samples=num_samples, fs=fs, device=device, n_fft=loss_n_fft,
            use_freq_pluck=False, use_freq_ksa=True, use_lti=True)),
    }

    # ── Method definitions ──
    methods = ['pluck_td', 'pluck_fd', 'ks_td', 'ks_fd']
    method_params = {
        'pluck_td': 'pluck_position', 'pluck_fd': 'pluck_position',
        'ks_td': 'f0', 'ks_fd': 'f0',
    }
    coarse_grids = {
        'pluck_td': pluck_grid, 'pluck_fd': pluck_grid,
        'ks_td': f0_grid, 'ks_fd': f0_grid,
    }
    offset_grids_map = {
        'pluck_td': pluck_offset_grid, 'pluck_fd': pluck_offset_grid,
        'ks_td': f0_offset_grid, 'ks_fd': f0_offset_grid,
    }
    offset_to_abs_fns = {
        'pluck_td': pluck_offset_to_absolute,
        'pluck_fd': pluck_offset_to_absolute,
        'ks_td': f0_offset_to_absolute,
        'ks_fd': f0_offset_to_absolute,
    }

    fga_target_grids = {
        'pluck_td': pluck_grid, 'pluck_fd': pluck_grid,
        'ks_td': f0_grid, 'ks_fd': f0_grid,
    }

    pluck_valid_range = (cfg.grid.pluck_min, cfg.grid.pluck_max)
    fga_valid_ranges = {
        'pluck_td': pluck_valid_range, 'pluck_fd': pluck_valid_range,
        'ks_td': None, 'ks_fd': None,
    }

    # ── Storage ──
    cga_binary = {m: {} for m in methods}
    cga_intensity = {m: {} for m in methods}
    cga_min_err = {m: {} for m in methods}

    fga_binary = {m: {} for m in methods}
    fga_intensity = {m: {} for m in methods}
    fga_valid_masks = {}

    t_total_start = time.time()

    # ── CGA sweeps ──
    for mi, method in enumerate(methods):
        t0 = time.time()
        print(f"\n{'=' * 60}")
        print(f'[{mi + 1}/{len(methods) * 2}] CGA: {method}')
        print(f"{'=' * 60}")

        result = compute_cga_maps(
            cfg=cfg, synth=synths[method],
            grid=coarse_grids[method],
            param_name=method_params[method],
            kernel_spectra=kernel_spectra,
            device=device, batch_size=batch_size,
            method_name=f"{method}/CGA",
        )
        for kl in kernel_lengths:
            cga_binary[method][kl] = result[kl][0]
            cga_intensity[method][kl] = result[kl][1]
            cga_min_err[method][kl] = result[kl][2]

        print(f"  {method}/CGA done in {time.time() - t0:.1f}s")

    # ── FGA sweeps ──
    for mi, method in enumerate(methods):
        t0 = time.time()
        print(f"\n{'=' * 60}")
        print(f'[{len(methods) + mi + 1}/{len(methods) * 2}] FGA: {method}')
        print(f"{'=' * 60}")

        result, valid_mask = compute_fga_maps(
            cfg=cfg, synth=synths[method],
            target_grid=fga_target_grids[method],
            offset_grid=offset_grids_map[method],
            param_name=method_params[method],
            offset_to_abs_fn=offset_to_abs_fns[method],
            kernel_spectra=kernel_spectra,
            device=device, batch_size=batch_size,
            method_name=f"{method}/FGA",
            valid_range=fga_valid_ranges[method],
        )
        for kl in kernel_lengths:
            fga_binary[method][kl] = result[kl][0]
            fga_intensity[method][kl] = result[kl][1]
        fga_valid_masks[method] = valid_mask

        print(f"  {method}/FGA done in {time.time() - t0:.1f}s")

    total_elapsed = time.time() - t_total_start
    print(f'\nAll autograd maps computed in {total_elapsed:.1f}s')

    # ── Results table ──
    combos = list(product(kernel_lengths, [
        ('Time', 'pluck_td', 'ks_td'),
        ('Freq', 'pluck_fd', 'ks_fd'),
    ]))

    results = []
    for (kl, (prefix, pluck_m, ks_m)) in tqdm(combos, desc="Building results table"):
        row = {'name': f"{prefix}{kl}"}
        row['p_cga'] = cga_from_map(cga_binary[pluck_m][kl])
        row['f_cga'] = cga_from_map(cga_binary[ks_m][kl])
        row['p_fga'] = fga_from_map(fga_binary[pluck_m][kl],
                                    valid_mask=fga_valid_masks[pluck_m])
        row['f_fga'] = fga_from_map(fga_binary[ks_m][kl],
                                    valid_mask=fga_valid_masks[ks_m])

        p_err = cga_min_err[pluck_m][kl]
        row['p_L'] = abs(
            pluck_to_delay_samples(cfg.target.pluck_position + p_err,
                                   cfg.target.f0, fs)
            - pluck_to_delay_samples(cfg.target.pluck_position,
                                     cfg.target.f0, fs))

        f_err = cga_min_err[ks_m][kl]
        row['f_cents'] = abs(hz_to_cents(cfg.target.f0 + f_err, cfg.target.f0))

        results.append(row)

    df = pd.DataFrame(results).set_index('name')
    df = df.reindex([f"{p}{kl}" for kl in kernel_lengths
                     for p in ['Time', 'Freq']])
    df.columns = pd.MultiIndex.from_tuples([
        ('Pluck Position', 'CGA'), ('Fundamental Frequency (KS)', 'CGA'),
        ('Pluck Position', 'FGA'), ('Fundamental Frequency (KS)', 'FGA'),
        ('Pluck Position', 'L'),
        ('Fundamental Frequency (KS)', 'cents'),
    ])

    csv_path = os.path.join(output_dir, 'results_table.csv')
    df.to_csv(csv_path)
    print(f"Saved: {csv_path}")

    print(f"\n{'':>12} | {'Pluck Position':^30} | "
          f"{'Fundamental Frequency (KS)':^36}")
    print(f"{'':>12} | {'CGA':>8} {'FGA':>8} {'L':>8} | "
          f"{'CGA':>8} {'FGA':>8} {'cents':>10}")
    print("-" * 90)
    for idx, row in df.iterrows():
        p = 'Pluck Position'
        f = 'Fundamental Frequency (KS)'
        print(f"{idx:>12} | {row[(p, 'CGA')]:>7.1%} {row[(p, 'FGA')]:>7.1%} "
              f"{row[(p, 'L')]:>7.2f} | "
              f"{row[(f, 'CGA')]:>7.1%} {row[(f, 'FGA')]:>7.1%} "
              f"{row[(f, 'cents')]:>9.2f}")

    # ── Sanity check ──
    print("\n\u2500\u2500 Sanity check \u2500\u2500")
    check = {
        'pluck_td': ('pluck_position', 0.3, 0.5, synths['pluck_td']),
        'pluck_fd': ('pluck_position', 0.3, 0.5, synths['pluck_fd']),
        'ks_td':    ('f0', 200., 220., synths['ks_td']),
        'ks_fd':    ('f0', 200., 220., synths['ks_fd']),
    }
    for name, (pname, iv, tv, s) in check.items():
        with torch.no_grad():
            yt = s(make_params(cfg, device, **{pname: tv})).squeeze(0)
            Yt = torch.fft.rfft(yt, n=loss_n_fft)
        p = torch.tensor([[iv]], device=device, requires_grad=True)
        y = s(make_params_batch(cfg, 1, pname, p, device=device)).squeeze(0)
        Y_pred = torch.fft.rfft(y, n=loss_n_fft)
        loss = ((Y_pred - Yt).abs() ** 2).mean()
        loss.backward()
        print(f'{name:>10s}:  grad={p.grad.item():+.6e}  '
              f'loss={loss.item():.4e}')

    print()
    print('If ks_td and ks_fd grads match, torchlpc parallel-scan backward')
    print('is as accurate as closed-form FD at this signal length.')

    # ══════════════════════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════════════════════

    print("\n\u2500\u2500 Generating figures \u2500\u2500")

    def best_cga_kernel(method):
        grid = coarse_grids[method]
        mask = off_diag_mask(len(grid))
        return max(kernel_lengths,
                   key=lambda kl: cga_binary[method][kl][mask].mean())

    def best_fga_kernel(method):
        vm = fga_valid_masks[method]
        return max(kernel_lengths,
                   key=lambda kl: fga_from_map(fga_binary[method][kl],
                                               valid_mask=vm))

    best_c = {m: best_cga_kernel(m) for m in methods}
    best_f = {m: best_fga_kernel(m) for m in methods}
    print(f"Best kernel (CGA): {({m: best_c[m] for m in methods})}")
    print(f"Best kernel (FGA): {({m: best_f[m] for m in methods})}")

    # ── Fig 1: Binary CGA ──
    best_cga_bin = {m: cga_binary[m][best_c[m]].astype(float) for m in methods}
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_cga_2x2(
        axes, best_cga_bin, methods,
        grids=[pluck_grid, pluck_grid, f0_grid, f0_grid],
        titles=[f"Pluck \u2014 Time (N'={best_c['pluck_td']})",
                f"Pluck \u2014 Freq (N'={best_c['pluck_fd']})",
                f"f0 (KS) \u2014 Time (N'={best_c['ks_td']})",
                f"f0 (KS) \u2014 Freq (N'={best_c['ks_fd']})"],
        xlabels=['Target pluck', 'Target pluck',
                 'Target f0 (Hz)', 'Target f0 (Hz)'],
        ylabels=['Init pluck', 'Init pluck',
                 'Init f0 (Hz)', 'Init f0 (Hz)'],
        cmap='gray', vmin=0, vmax=1,
        log_axes_flags=[False, False, True, True],
        metric_name='CGA',
    )
    fig.suptitle('Binary CGA \u2014 White = correct, Black = wrong', fontsize=14)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(output_dir, f'fig1_binary_cga.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved: fig1_binary_cga.pdf/png")

    # ── Fig 2: Binary FGA ──
    best_fga_bin = {m: fga_binary[m][best_f[m]].astype(float) for m in methods}
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_fga_2x2(
        axes, best_fga_bin, methods,
        target_grids=[pluck_grid, pluck_grid, f0_grid, f0_grid],
        offset_grids=[pluck_offset_grid, pluck_offset_grid,
                      f0_offset_grid, f0_offset_grid],
        titles=[f"Pluck \u2014 Time (N'={best_f['pluck_td']})",
                f"Pluck \u2014 Freq (N'={best_f['pluck_fd']})",
                f"f0 (KS) \u2014 Time (N'={best_f['ks_td']})",
                f"f0 (KS) \u2014 Freq (N'={best_f['ks_fd']})"],
        xlabels=['Target pluck', 'Target pluck',
                 'Target f0 (Hz)', 'Target f0 (Hz)'],
        ylabels=['Init offset (pluck)', 'Init offset (pluck)',
                 'Init offset (cents)', 'Init offset (cents)'],
        cmap='gray', vmin=0, vmax=1,
        log_x_flags=[False, False, True, True],
        metric_name='FGA',
        valid_masks=fga_valid_masks,
    )
    fig.suptitle(f'Binary FGA (\u00b1{cfg.fga.f0_half_cents}\u00a2 / '
                 f'\u00b1{cfg.fga.pluck_half} pluck) \u2014 '
                 f'White = correct, Black = wrong', fontsize=14)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(output_dir, f'fig2_binary_fga.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved: fig2_binary_fga.pdf/png")

    # ── Fig 3: Intensity (coarse / CGA) ──
    cga_int_log = {
        m: log_normalise(
            np.nan_to_num(cga_intensity[m][best_c[m]], nan=0.0))
        for m in methods
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_cga_2x2(
        axes, cga_int_log, methods,
        grids=[pluck_grid, pluck_grid, f0_grid, f0_grid],
        titles=[f"Pluck \u2014 Time (N'={best_c['pluck_td']})",
                f"Pluck \u2014 Freq (N'={best_c['pluck_fd']})",
                f"f0 (KS) \u2014 Time (N'={best_c['ks_td']})",
                f"f0 (KS) \u2014 Freq (N'={best_c['ks_fd']})"],
        xlabels=['Target pluck', 'Target pluck',
                 'Target f0 (Hz)', 'Target f0 (Hz)'],
        ylabels=['Init pluck', 'Init pluck',
                 'Init f0 (Hz)', 'Init f0 (Hz)'],
        cmap='magma', vmin=-1, vmax=1,
        log_axes_flags=[False, False, True, True],
    )
    fig.suptitle('Gradient Intensity (log) \u2014 Coarse', fontsize=14)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(output_dir, f'fig3_intensity_coarse.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved: fig3_intensity_coarse.pdf/png")

    # ── Fig 4: Intensity (fine / FGA) ──
    fga_int_log = {
        m: log_normalise(
            np.nan_to_num(fga_intensity[m][best_f[m]], nan=0.0))
        for m in methods
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_fga_2x2(
        axes, fga_int_log, methods,
        target_grids=[pluck_grid, pluck_grid, f0_grid, f0_grid],
        offset_grids=[pluck_offset_grid, pluck_offset_grid,
                      f0_offset_grid, f0_offset_grid],
        titles=[f"Pluck \u2014 Time (N'={best_f['pluck_td']})",
                f"Pluck \u2014 Freq (N'={best_f['pluck_fd']})",
                f"f0 (KS) \u2014 Time (N'={best_f['ks_td']})",
                f"f0 (KS) \u2014 Freq (N'={best_f['ks_fd']})"],
        xlabels=['Target pluck', 'Target pluck',
                 'Target f0 (Hz)', 'Target f0 (Hz)'],
        ylabels=['Init offset (pluck)', 'Init offset (pluck)',
                 'Init offset (cents)', 'Init offset (cents)'],
        cmap='magma', vmin=-1, vmax=1,
        log_x_flags=[False, False, True, True],
    )
    fig.suptitle(f'Gradient Intensity (log) \u2014 Fine '
                 f'(\u00b1{cfg.fga.f0_half_cents}\u00a2 / '
                 f'\u00b1{cfg.fga.pluck_half} pluck)', fontsize=14)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(output_dir, f'fig4_intensity_fine.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved: fig4_intensity_fine.pdf/png")

    print(f"\n\u2713 All outputs saved to: {output_dir}")
    print(f"\u2713 Total wall time: {time.time() - t_total_start:.1f}s")


if __name__ == "__main__":
    main()