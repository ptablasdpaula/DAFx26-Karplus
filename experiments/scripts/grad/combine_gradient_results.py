#!/usr/bin/env python3
"""
Gradient Analysis — Combine Results

Reads the four .npz files produced by gradient_analysis_stage.py
and generates the final results table (CSV + stdout) and figures.

Also runs a lightweight sanity check (requires GPU).

Usage:
  python combine_gradient_results.py <results_dir>

Where <results_dir> contains:
  cga_pluck.npz, cga_ks.npz, fga_pluck.npz, fga_ks.npz
"""

import os
import sys
import argparse
import time
import paths

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
from omegaconf import OmegaConf

from synths.synth import Synth, SynthConfig


# ═══════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════

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

def hz_to_cents(f0_hz, ref_hz):
    return 1200.0 * np.log2(f0_hz / ref_hz)

def log_normalise(arr):
    sign = np.sign(arr)
    log_abs = np.log1p(np.abs(arr))
    mx = np.nanmax(log_abs)
    return sign * log_abs / mx if mx > 1e-12 else np.zeros_like(arr)


# ═══════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════

def plot_cga_2x2(
    axes, data_maps, method_keys, grids, titles, xlabels, ylabels,
    cmap, vmin, vmax, log_axes_flags, metric_name=None,
    show_colorbar=False, cbar_label='Log gradient (+ = correct)'
):
    """
    Matches the 'fast' script styling:
    - Right column: no y-label, no title, no y tick labels
    - Optional per-axis colorbar only if show_colorbar=True (usually False)
    - Returns last im for building a shared colorbar externally
    """
    last_im = None

    for idx, (ax, key, grid, title, xlabel, ylabel, use_log) in enumerate(
        zip(axes.flat, method_keys, grids, titles, xlabels, ylabels, log_axes_flags)
    ):
        col = idx % 2

        display_data = np.nan_to_num(data_maps[key].T, nan=0.0)
        im = ax.pcolormesh(grid, grid, display_data, cmap=cmap, vmin=vmin, vmax=vmax)
        last_im = im

        ax.plot([grid[0], grid[-1]], [grid[0], grid[-1]],
                color='cyan', ls='--', linewidth=1.5)
        ax.set_xlabel(xlabel)

        if col == 0:
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        else:
            ax.set_ylabel('')
            ax.set_title('')
            ax.tick_params(labelleft=False)

        if use_log:
            ax.set_xscale('log')
            ax.set_yscale('log')

        if metric_name and cmap == 'gray':
            n = len(grid)
            mask = off_diag_mask(n)
            pct = data_maps[key][mask].mean() * 100
            ax.text(
                0.02, 0.98, f"{metric_name}: {pct:.1f}%",
                transform=ax.transAxes, fontsize=11, va='top', ha='left',
                color='cyan',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
            )
        elif (cmap != 'gray') and show_colorbar and (col == 1):
            plt.colorbar(im, ax=ax, label=cbar_label)

    return last_im


def plot_fga_2x2(
    axes, data_maps, method_keys, target_grids, offset_grids,
    titles, xlabels, ylabels, cmap, vmin, vmax,
    log_x_flags, metric_name=None, valid_masks=None,
    show_colorbar=False, cbar_label='Log gradient (+ = correct)'
):
    """
    Matches the 'fast' script styling:
    - Right column cleaned (no y-label/title/ticks)
    - Optional per-axis colorbar only if show_colorbar=True (usually False)
    - Returns last im for building a shared colorbar externally
    """
    last_im = None

    for idx, (ax, key, tgrid, ogrid, title, xlabel, ylabel, log_x) in enumerate(
        zip(axes.flat, method_keys, target_grids, offset_grids,
            titles, xlabels, ylabels, log_x_flags)
    ):
        col = idx % 2

        data = data_maps[key]
        display_data = np.nan_to_num(data.T, nan=0.0)
        im = ax.pcolormesh(tgrid, ogrid, display_data, cmap=cmap, vmin=vmin, vmax=vmax)
        last_im = im

        ax.axhline(0, color='cyan', ls='--', linewidth=1.5)
        ax.set_xlabel(xlabel)

        if col == 0:
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        else:
            ax.set_ylabel('')
            ax.set_title('')
            ax.tick_params(labelleft=False)

        if log_x:
            ax.set_xscale('log')

        if metric_name and cmap == 'gray':
            vm = valid_masks.get(key) if valid_masks else None
            pct = fga_from_map(data, valid_mask=vm) * 100
            ax.text(
                0.02, 0.98, f"{metric_name}: {pct:.1f}%",
                transform=ax.transAxes, fontsize=11, va='top', ha='left',
                color='cyan',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
            )
        elif (cmap != 'gray') and show_colorbar and (col == 1):
            plt.colorbar(im, ax=ax, label=cbar_label)

    return last_im

# ═══════════════════════════════════════════════════════════
# Loader
# ═══════════════════════════════════════════════════════════

def load_stage(results_dir, filename):
    path = os.path.join(results_dir, filename)
    if not os.path.exists(path):
        print(f"ERROR: missing {path}")
        sys.exit(1)
    data = np.load(path, allow_pickle=True)
    print(f"  Loaded: {path}  ({len(data.files)} arrays)")
    return data


def extract_cga(data, methods, kernel_lengths):
    """Extract CGA results from an .npz file."""
    cga_binary = {}
    cga_intensity = {}
    cga_min_err = {}
    for method in methods:
        cga_binary[method] = {}
        cga_intensity[method] = {}
        cga_min_err[method] = {}
        for kl in kernel_lengths:
            cga_binary[method][kl] = data[f'{method}__correct__{kl}']
            cga_intensity[method][kl] = data[f'{method}__intensity__{kl}']
            cga_min_err[method][kl] = float(data[f'{method}__min_err__{kl}'])
    grid = data['grid']
    return cga_binary, cga_intensity, cga_min_err, grid


def extract_fga(data, methods, kernel_lengths):
    """Extract FGA results from an .npz file."""
    fga_binary = {}
    fga_intensity = {}
    fga_valid_masks = {}
    for method in methods:
        fga_binary[method] = {}
        fga_intensity[method] = {}
        fga_valid_masks[method] = data[f'{method}__valid_mask']
        for kl in kernel_lengths:
            fga_binary[method][kl] = data[f'{method}__correct__{kl}']
            fga_intensity[method][kl] = data[f'{method}__intensity__{kl}']
    target_grid = data['target_grid']
    offset_grid = data['offset_grid']
    return fga_binary, fga_intensity, fga_valid_masks, target_grid, offset_grid


# ═══════════════════════════════════════════════════════════
# Sanity check helpers
# ═══════════════════════════════════════════════════════════

def make_params(cfg, device, **overrides):
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
    params['burst_gain'] = torch.zeros(1, nf, device=device)
    params['burst_gain'][0, 0] = t.burst_gain
    return params


def make_params_batch(cfg, n_batch, param_name, param_values, device):
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


SYNTH_CONFIGS = {
    'pluck_td': dict(use_freq_pluck=False, use_freq_ksa=False),
    'pluck_fd': dict(use_freq_pluck=True,  use_freq_ksa=False, use_lti=True),
    'ks_td':    dict(use_freq_pluck=False, use_freq_ksa=False),
    'ks_fd':    dict(use_freq_pluck=False, use_freq_ksa=True,  use_lti=True),
}


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Combine gradient analysis stage results into table + figures")
    parser.add_argument('results_dir', type=str,
                        help="Directory containing the 4 .npz stage files")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Where to write figures/CSV (default: results_dir)")
    parser.add_argument('--skip-sanity', action='store_true',
                        help="Skip the sanity check (useful if no GPU)")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams['figure.dpi'] = 100
    t_start = time.time()

    # ── Load all stages ──
    print("Loading stage results…")
    d_cga_pluck = load_stage(results_dir, 'cga_pluck.npz')
    d_cga_ks    = load_stage(results_dir, 'cga_ks.npz')
    d_fga_pluck = load_stage(results_dir, 'fga_pluck.npz')
    d_fga_ks    = load_stage(results_dir, 'fga_ks.npz')

    # ── Recover config from any stage ──
    cfg = OmegaConf.create(str(d_cga_pluck['config_yaml']))
    kernel_lengths = list(d_cga_pluck['kernel_lengths'])
    fs = cfg.audio.fs

    # ── Extract data ──
    cga_binary_p, cga_intensity_p, cga_min_err_p, pluck_grid = \
        extract_cga(d_cga_pluck, ['pluck_td', 'pluck_fd'], kernel_lengths)
    cga_binary_k, cga_intensity_k, cga_min_err_k, f0_grid = \
        extract_cga(d_cga_ks, ['ks_td', 'ks_fd'], kernel_lengths)

    fga_binary_p, fga_intensity_p, fga_valid_masks_p, pluck_tgrid, pluck_ogrid = \
        extract_fga(d_fga_pluck, ['pluck_td', 'pluck_fd'], kernel_lengths)
    fga_binary_k, fga_intensity_k, fga_valid_masks_k, f0_tgrid, f0_ogrid = \
        extract_fga(d_fga_ks, ['ks_td', 'ks_fd'], kernel_lengths)

    # ── Merge into unified dicts ──
    methods = ['pluck_td', 'pluck_fd', 'ks_td', 'ks_fd']
    cga_binary = {**cga_binary_p, **cga_binary_k}
    cga_intensity = {**cga_intensity_p, **cga_intensity_k}
    cga_min_err = {**cga_min_err_p, **cga_min_err_k}
    fga_binary = {**fga_binary_p, **fga_binary_k}
    fga_intensity = {**fga_intensity_p, **fga_intensity_k}
    fga_valid_masks = {**fga_valid_masks_p, **fga_valid_masks_k}

    f0_cents = hz_to_cents(f0_grid, cfg.target.f0)
    pluck_offset_grid = pluck_ogrid
    f0_offset_grid = f0_ogrid

    coarse_grids = {
        'pluck_td': pluck_grid, 'pluck_fd': pluck_grid,
        'ks_td': f0_grid, 'ks_fd': f0_grid,
    }

    print(f"\nGrids loaded:")
    print(f"  Pluck coarse: {len(pluck_grid)} pts")
    print(f"  f0 coarse:    {len(f0_grid)} pts")
    print(f"  Pluck FGA:    {len(pluck_tgrid)}×{len(pluck_ogrid)}")
    print(f"  f0 FGA:       {len(f0_tgrid)}×{len(f0_ogrid)}")

    # ══════════════════════════════════════════════════════════
    # Results table
    # ══════════════════════════════════════════════════════════

    combos = list(product(kernel_lengths, [
        ('Time', 'pluck_td', 'ks_td'),
        ('Freq', 'pluck_fd', 'ks_fd'),
    ]))

    results = []
    for (kl, (prefix, pluck_m, ks_m)) in combos:
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
    print(f"\nSaved: {csv_path}")

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

    # ══════════════════════════════════════════════════════════
    # Sanity check
    # ══════════════════════════════════════════════════════════

    if not args.skip_sanity:
        print("\n── Sanity check ──")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_samples = fs * cfg.audio.duration
        loss_n_fft = num_samples

        check = {
            'pluck_td': ('pluck_position', 0.3, 0.5),
            'pluck_fd': ('pluck_position', 0.3, 0.5),
            'ks_td':    ('f0', 200., 220.),
            'ks_fd':    ('f0', 200., 220.),
        }
        for name, (pname, iv, tv) in check.items():
            synth = Synth(SynthConfig(
                num_samples=num_samples, fs=fs, device=device, n_fft=loss_n_fft,
                **SYNTH_CONFIGS[name]))
            with torch.no_grad():
                yt = synth(make_params(cfg, device, **{pname: tv})).squeeze(0)
                Yt = torch.fft.rfft(yt, n=loss_n_fft)
            p = torch.tensor([[iv]], device=device, requires_grad=True)
            y = synth(make_params_batch(cfg, 1, pname, p, device=device)).squeeze(0)
            Y_pred = torch.fft.rfft(y, n=loss_n_fft)
            loss = ((Y_pred - Yt).abs() ** 2).mean()
            loss.backward()
            print(f'{name:>10s}:  grad={p.grad.item():+.6e}  '
                  f'loss={loss.item():.4e}')
            del synth

        print()
        print('If ks_td and ks_fd grads match, torchlpc parallel-scan backward')
        print('is as accurate as closed-form FD at this signal length.')

    # ══════════════════════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════════════════════

    print("\n── Generating figures ──")

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
        m: log_normalise(np.nan_to_num(cga_intensity[m][best_c[m]], nan=0.0))
        for m in methods
    }

    # Manual layout controls (same as fast script)
    fig_wspace = 0.15
    fig_hspace = 0.20
    cbar_x, cbar_y, cbar_w, cbar_h = 0.88, 0.25, 0.02, 0.50
    fig_right = cbar_x - 0.02

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey='row')
    fig.subplots_adjust(wspace=fig_wspace, hspace=fig_hspace, right=fig_right)

    im = plot_cga_2x2(
        axes, cga_int_log, methods,
        grids=[pluck_grid, pluck_grid, f0_grid, f0_grid],
        titles=[f"Pluck — Time (N'={best_c['pluck_td']})",
                f"Pluck — Freq (N'={best_c['pluck_fd']})",
                f"f0 (KS) — Time (N'={best_c['ks_td']})",
                f"f0 (KS) — Freq (N'={best_c['ks_fd']})"],
        xlabels=['Target pluck', 'Target pluck',
                'Target f0 (Hz)', 'Target f0 (Hz)'],
        ylabels=['Init pluck', 'Init pluck',
                'Init f0 (Hz)', 'Init f0 (Hz)'],
        cmap='magma', vmin=-1, vmax=1,
        log_axes_flags=[False, False, True, True],
        show_colorbar=False,
    )

    cax = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])
    fig.colorbar(im, cax=cax, label='Log gradient (+ = correct)')

    fig.suptitle('Gradient Intensity (log) — Coarse', fontsize=14)

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(output_dir, f'fig3_intensity_coarse.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved: fig3_intensity_coarse.pdf/png")

    # ── Fig 4: Intensity (fine / FGA) ──
    fga_int_log = {
        m: log_normalise(np.nan_to_num(fga_intensity[m][best_f[m]], nan=0.0))
        for m in methods
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey='row')
    fig.subplots_adjust(wspace=fig_wspace, hspace=fig_hspace, right=fig_right)

    im = plot_fga_2x2(
        axes, fga_int_log, methods,
        target_grids=[pluck_grid, pluck_grid, f0_grid, f0_grid],
        offset_grids=[pluck_offset_grid, pluck_offset_grid,
                    f0_offset_grid, f0_offset_grid],
        titles=[f"Pluck — Time (N'={best_f['pluck_td']})",
                f"Pluck — Freq (N'={best_f['pluck_fd']})",
                f"f0 (KS) — Time (N'={best_f['ks_td']})",
                f"f0 (KS) — Freq (N'={best_f['ks_fd']})"],
        xlabels=['Target pluck', 'Target pluck',
                'Target f0 (Hz)', 'Target f0 (Hz)'],
        ylabels=['Init offset (pluck)', 'Init offset (pluck)',
                'Init offset (cents)', 'Init offset (cents)'],
        cmap='magma', vmin=-1, vmax=1,
        log_x_flags=[False, False, True, True],
        show_colorbar=False,
    )

    cax = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])
    fig.colorbar(im, cax=cax, label='Log gradient (+ = correct)')

    fig.suptitle(
        f"Gradient Intensity (log) — Fine (±{cfg.fga.f0_half_cents}¢ / ±{cfg.fga.pluck_half} pluck)",
        fontsize=14
    )

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(output_dir, f'fig4_intensity_fine.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved: fig4_intensity_fine.pdf/png")


if __name__ == "__main__":
    main()