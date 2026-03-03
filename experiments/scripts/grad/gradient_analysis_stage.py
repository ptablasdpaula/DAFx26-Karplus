#!/usr/bin/env python3
"""
Gradient Analysis — Stage Runner

Runs ONE of four independent stages and saves results to .npz:
  cga_pluck  — CGA for pluck_td + pluck_fd
  cga_ks     — CGA for ks_td + ks_fd
  fga_pluck  — FGA for pluck_td + pluck_fd
  fga_ks     — FGA for ks_td + ks_fd

Usage (via Hydra):
  python gradient_analysis_stage.py +stage=cga_pluck +results_dir=/path/to/shared
  python gradient_analysis_stage.py +stage=fga_ks   +results_dir=/path/to/shared

All four stages can run in parallel on separate GPUs.  A separate
combine_gradient_results.py script reads the .npz files and produces
the final table and figures.
"""

import os
import sys
import time

import paths

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm

from src.synths.synth import Synth, SynthConfig


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def resolve_device(requested: str) -> str:
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU")
        return 'cpu'
    return requested


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
    else:
        grid = np.linspace(g.pluck_min, g.pluck_max, g.n_points)
    return grid


def make_f0_offset_grid(cfg: DictConfig):
    return np.linspace(-cfg.fga.f0_half_cents, cfg.fga.f0_half_cents,
                       num=cfg.grid.n_points)


def make_pluck_offset_grid(cfg: DictConfig):
    return np.linspace(-cfg.fga.pluck_half, cfg.fga.pluck_half,
                       num=cfg.grid.n_points)


def f0_offset_to_absolute(target_f0, offset_cents):
    return target_f0 * 2.0 ** (offset_cents / 1200.0)


def pluck_offset_to_absolute(target_pluck, offset_pluck):
    return target_pluck + offset_pluck


# ═══════════════════════════════════════════════════════════
# Target FFT helper
# ═══════════════════════════════════════════════════════════

def compute_target_ffts(cfg, synth, grid_np, param_name, device,
                        batch_size, loss_n_fft, label=''):
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
# Reverse-mode CGA
# ═══════════════════════════════════════════════════════════

def compute_cga_reverse(cfg, synth, grid_np, param_name, target_ffts,
                        kernel_sq, kernel_list, device, batch_size,
                        loss_n_fft, method_name):
    n = len(grid_np)
    K = len(kernel_list)
    effective_batch = n if (batch_size <= 0 or batch_size >= n) else batch_size

    correct = {kl: np.zeros((n, n), dtype=bool) for kl, _ in kernel_list}
    intens = {kl: np.full((n, n), np.nan) for kl, _ in kernel_list}
    min_loss_errors = {kl: np.full(n, np.inf) for kl, _ in kernel_list}

    # Mark the trivial "near-diagonal" cells as correct by definition
    for kl, _ in kernel_list:
        for i in range(n):
            for j in range(max(0, i - 1), min(n, i + 2)):
                correct[kl][i, j] = True

    n_batches = (n + effective_batch - 1) // effective_batch
    # CGA work is per-(batch,target), not per-batch — so track that
    pbar = tqdm(total=n_batches * n, desc=f"  [{method_name}] CGA targets×batches")

    for b_start in range(0, n, effective_batch):
        b_end = min(b_start + effective_batch, n)
        B = b_end - b_start

        grid_param = torch.tensor(
            grid_np[b_start:b_end], dtype=torch.float32, device=device
        ).unsqueeze(1).requires_grad_(True)

        params = make_params_batch(cfg, B, param_name, grid_param, device=device)
        y = synth(params)
        Y = torch.fft.rfft(y, n=loss_n_fft)

        indices = np.arange(b_start, b_end)
        directions = np.where(indices[:, None] > np.arange(n)[None, :], 1.0, -1.0)  # not used directly; kept for clarity

        for t_idx in range(n):
            target_fft = target_ffts[t_idx]
            diff = Y - target_fft.unsqueeze(0)

            far_mask = np.abs(indices - t_idx) > 1
            cols = indices[far_mask]
            dir_vec = np.where(indices > t_idx, 1.0, -1.0)

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

                # Only drop the graph on the *final* grad call of the *final* target of the *final* batch
                is_last = (b_end == n) and (ki == K - 1) and (t_idx == n - 1)

                grads = torch.autograd.grad(
                    total_loss, grid_param, retain_graph=not is_last
                )[0]
                grad_np = grads.squeeze(1).detach().cpu().numpy()

                signed = grad_np * dir_vec

                # IMPORTANT: avoid chained boolean indexing (can write into a copy)
                if cols.size > 0:
                    intens[kl][t_idx, cols] = signed[far_mask]
                    correct[kl][t_idx, cols] = signed[far_mask] > 0

            pbar.update(1)

        del y, Y, params
        torch.cuda.empty_cache()

    pbar.close()
    mean_errors = {kl: min_loss_errors[kl].mean() for kl, _ in kernel_list}
    return {kl: (correct[kl], intens[kl], mean_errors[kl])
            for kl, _ in kernel_list}


# ═══════════════════════════════════════════════════════════
# Reverse-mode FGA
# ═══════════════════════════════════════════════════════════

def compute_fga_reverse(cfg, synth, target_np, offset_np, param_name,
                        offset_to_abs_fn, target_ffts_all, kernel_sq,
                        kernel_list, device, batch_size, loss_n_fft,
                        method_name, valid_range):
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
    pbar = tqdm(total=n_targets, desc=f"  [{method_name}] FGA targets")

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
            pbar.update(1)
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

            params = make_params_batch(cfg, vB, param_name, init_param,
                                       device=device)
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
                signed = grad_np * batch_signs

                for vi in range(vB):
                    fi = chunk_global_idx[vi]
                    if abs(fi - centre) > 1:
                        intens[kl][t_idx, fi] = signed[vi]
                        correct[kl][t_idx, fi] = signed[vi] > 0

            del y, Y, params
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
# Stage runners
# ═══════════════════════════════════════════════════════════

STAGE_METHODS = {
    'cga_pluck': ['pluck_td', 'pluck_fd'],
    'cga_ks':    ['ks_td',    'ks_fd'],
    'fga_pluck': ['pluck_td', 'pluck_fd'],
    'fga_ks':    ['ks_td',    'ks_fd'],
}

METHOD_PARAM = {
    'pluck_td': 'pluck_position', 'pluck_fd': 'pluck_position',
    'ks_td': 'f0', 'ks_fd': 'f0',
}

SYNTH_CONFIGS = {
    'pluck_td': dict(use_freq_pluck=False, use_freq_ksa=False),
    'pluck_fd': dict(use_freq_pluck=True,  use_freq_ksa=False, use_lti=True),
    'ks_td':    dict(use_freq_pluck=False, use_freq_ksa=False),
    'ks_fd':    dict(use_freq_pluck=False, use_freq_ksa=True,  use_lti=True),
}


def run_cga_stage(cfg, methods, device, batch_size, results_dir):
    fs = cfg.audio.fs
    num_samples = fs * cfg.audio.duration
    loss_n_fft = num_samples
    kernel_lengths = list(cfg.kernel_lengths)

    kernel_spectra = {
        kl: triangular_kernel_spectrum(kl, loss_n_fft, device=device)
        for kl in kernel_lengths
    }

    param_name = METHOD_PARAM[methods[0]]
    if param_name == 'pluck_position':
        grid = make_pluck_grid(cfg)
        stage_name = 'cga_pluck'
    else:
        grid = make_f0_grid(cfg)
        stage_name = 'cga_ks'

    grid_np = np.asarray(grid, dtype=np.float64)
    print(f"\nGrid: {len(grid_np)} points, param={param_name}")

    save_dict = {
        'grid': grid_np,
        'kernel_lengths': np.array(kernel_lengths),
        'config_yaml': OmegaConf.to_yaml(cfg),
    }

    for method in methods:
        print(f"\n{'=' * 60}")
        print(f"  CGA: {method}")
        print(f"{'=' * 60}")
        t0 = time.time()

        synth = Synth(SynthConfig(
            num_samples=num_samples, fs=fs, device=device, n_fft=loss_n_fft,
            **SYNTH_CONFIGS[method]))

        kernel_list = list(kernel_spectra.items())
        kernel_sq = torch.stack([ks ** 2 for _, ks in kernel_list]).to(device)

        target_ffts = compute_target_ffts(
            cfg, synth, grid_np, param_name, device, batch_size,
            loss_n_fft, label=f"{method}/targets")

        result = compute_cga_reverse(
            cfg, synth, grid_np, param_name, target_ffts,
            kernel_sq, kernel_list, device, batch_size,
            loss_n_fft, method)

        for kl in kernel_lengths:
            correct, intensity, mean_err = result[kl]
            save_dict[f'{method}__correct__{kl}'] = correct
            save_dict[f'{method}__intensity__{kl}'] = intensity
            save_dict[f'{method}__min_err__{kl}'] = np.array(mean_err)

        print(f"  {method} done in {time.time() - t0:.1f}s")

        del synth, target_ffts, kernel_sq
        torch.cuda.empty_cache()

    out_path = os.path.join(results_dir, f'{stage_name}.npz')
    np.savez_compressed(out_path, **save_dict)
    print(f"\n✓ Saved: {out_path}")


def run_fga_stage(cfg, methods, device, batch_size, results_dir):
    fs = cfg.audio.fs
    num_samples = fs * cfg.audio.duration
    loss_n_fft = num_samples
    kernel_lengths = list(cfg.kernel_lengths)

    kernel_spectra = {
        kl: triangular_kernel_spectrum(kl, loss_n_fft, device=device)
        for kl in kernel_lengths
    }

    param_name = METHOD_PARAM[methods[0]]
    if param_name == 'pluck_position':
        target_grid = make_pluck_grid(cfg)
        offset_grid = make_pluck_offset_grid(cfg)
        offset_to_abs_fn = pluck_offset_to_absolute
        valid_range = (cfg.grid.pluck_min, cfg.grid.pluck_max)
        stage_name = 'fga_pluck'
    else:
        target_grid = make_f0_grid(cfg)
        offset_grid = make_f0_offset_grid(cfg)
        offset_to_abs_fn = f0_offset_to_absolute
        valid_range = None
        stage_name = 'fga_ks'

    target_np = np.asarray(target_grid, dtype=np.float64)
    offset_np = np.asarray(offset_grid, dtype=np.float64)

    print(f"\nTarget grid: {len(target_np)} pts, "
          f"Offset grid: {len(offset_np)} pts, param={param_name}")

    save_dict = {
        'target_grid': target_np,
        'offset_grid': offset_np,
        'kernel_lengths': np.array(kernel_lengths),
        'config_yaml': OmegaConf.to_yaml(cfg),
    }

    for method in methods:
        print(f"\n{'=' * 60}")
        print(f"  FGA: {method}")
        print(f"{'=' * 60}")
        t0 = time.time()

        synth = Synth(SynthConfig(
            num_samples=num_samples, fs=fs, device=device, n_fft=loss_n_fft,
            **SYNTH_CONFIGS[method]))

        kernel_list = list(kernel_spectra.items())
        kernel_sq = torch.stack([ks ** 2 for _, ks in kernel_list]).to(device)

        target_ffts = compute_target_ffts(
            cfg, synth, target_np, param_name, device, batch_size,
            loss_n_fft, label=f"{method}/targets")

        result, valid_mask = compute_fga_reverse(
            cfg, synth, target_np, offset_np, param_name,
            offset_to_abs_fn, target_ffts, kernel_sq, kernel_list,
            device, batch_size, loss_n_fft, method, valid_range)

        save_dict[f'{method}__valid_mask'] = valid_mask

        for kl in kernel_lengths:
            correct, intensity = result[kl]
            save_dict[f'{method}__correct__{kl}'] = correct
            save_dict[f'{method}__intensity__{kl}'] = intensity

        print(f"  {method} done in {time.time() - t0:.1f}s")

        del synth, target_ffts, kernel_sq
        torch.cuda.empty_cache()

    out_path = os.path.join(results_dir, f'{stage_name}.npz')
    np.savez_compressed(out_path, **save_dict)
    print(f"\n✓ Saved: {out_path}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

@hydra.main(config_path=str(paths.CONFIGS_DIR), config_name="gradient_analysis",
            version_base=None)
def main(cfg: DictConfig):
    stage = cfg.get('stage', None)
    if stage is None:
        print("ERROR: must specify +stage=cga_pluck|cga_ks|fga_pluck|fga_ks")
        sys.exit(1)
    if stage not in STAGE_METHODS:
        print(f"ERROR: unknown stage '{stage}'. "
              f"Choose from: {list(STAGE_METHODS.keys())}")
        sys.exit(1)

    results_dir = cfg.get('results_dir', None)
    if results_dir is None:
        results_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(results_dir, exist_ok=True)

    print(f"Stage:       {stage}")
    print(f"Results dir: {results_dir}")
    print(OmegaConf.to_yaml(cfg))

    device = resolve_device(cfg.compute.device)
    batch_size = cfg.compute.batch_size

    print(f"Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {vram:.1f} GB")
    print(f"Batch size: {'all' if batch_size == 0 else batch_size}")

    methods = STAGE_METHODS[stage]
    t0 = time.time()

    if stage.startswith('cga'):
        run_cga_stage(cfg, methods, device, batch_size, results_dir)
    else:
        run_fga_stage(cfg, methods, device, batch_size, results_dir)

    print(f"\n✓ Stage '{stage}' completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()