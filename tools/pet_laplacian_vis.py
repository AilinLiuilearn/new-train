#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Laplacian pyramid frequency decoupling on CIPA-normalized PET.

Step-2 style decoupling:
  X_ll  : average-pooled coarse low-frequency map (smaller spatial size)
  X_hh  : high-frequency branch = sum of Laplacian residuals (merged at full res)

Single-level:
  X_ll      = AvgPool(X)
  X_ll_up   = Upsample(X_ll)
  residual  = X - X_ll_up
  X_hh      = residual

Multi-level:
  Gaussian pyramid G0=X, G_{k+1}=AvgPool(G_k)
  L_k = G_k - Upsample(G_{k+1})
  X_ll = G_K (coarsest)
  X_hh = sum_k Upsample^k(L_k) to full resolution
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

plt.rcParams['axes.unicode_minus'] = False


def imread_gray(path):
    return np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0


def resolve_paths(root, image_id):
    case_id, slice_id = image_id.split('_', 1)
    case_dir = os.path.join(root, case_id)
    base = f'{case_id}_{slice_id}'
    return {
        'image_id': image_id,
        'pet_path': os.path.join(case_dir, f'{base}_PET.png'),
        'mask_path': os.path.join(case_dir, f'{base}_mask.png'),
    }


def normalize_cipa(img):
    return img * 3.2 - 1.6


def normalize_display(arr, p_low=2, p_high=98, symmetric=False):
    x = arr.copy()
    if symmetric:
        v = np.percentile(np.abs(x), p_high)
        v = max(float(v), 1e-8)
        return np.clip(x / v, -1, 1)
    lo, hi = np.percentile(x, [p_low, p_high])
    if hi - lo < 1e-8:
        return np.clip(x, 0, 1)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def avg_pool2d(img, factor=2):
    h, w = img.shape
    h2, w2 = h // factor, w // factor
    if h2 == 0 or w2 == 0:
        raise ValueError(f'image too small for pool factor={factor}: {img.shape}')
    crop = img[: h2 * factor, : w2 * factor]
    return crop.reshape(h2, factor, w2, factor).mean(axis=(1, 3)).astype(np.float32)


def upsample_bilinear(img, out_hw):
    out_h, out_w = out_hw
    arr = np.array(
        Image.fromarray(img.astype(np.float32)).resize((out_w, out_h), Image.BILINEAR),
        dtype=np.float32,
    )
    return arr


def laplacian_decouple(x, levels=1, pool_factor=2):
    """Build Gaussian / Laplacian pyramid and return decoupled branches."""
    gaussian = [x.astype(np.float32)]
    for _ in range(levels):
        gaussian.append(avg_pool2d(gaussian[-1], factor=pool_factor))

    laplacian = []
    for i in range(levels):
        up = upsample_bilinear(gaussian[i + 1], gaussian[i].shape)
        lap = gaussian[i] - up
        laplacian.append(lap)

    x_ll = gaussian[-1]
    x_ll_up = upsample_bilinear(x_ll, x.shape)

    # Merge all Laplacian residuals to full resolution -> X_hh
    x_hh = np.zeros_like(x, dtype=np.float32)
    for i, lap in enumerate(laplacian):
        merged = lap
        for j in range(i):
            merged = upsample_bilinear(merged, gaussian[i - j - 1].shape)
        if merged.shape != x.shape:
            merged = upsample_bilinear(merged, x.shape)
        x_hh += merged

    recon = gaussian[-1]
    for i in reversed(range(levels)):
        recon = upsample_bilinear(recon, gaussian[i].shape) + laplacian[i]
    recon_err = np.abs(x - recon)

    return {
        'x': x,
        'x_ll': x_ll,
        'x_ll_up': x_ll_up,
        'x_hh': x_hh,
        'laplacian': laplacian,
        'gaussian': gaussian,
        'recon': recon,
        'recon_err': recon_err,
        'levels': levels,
        'pool_factor': pool_factor,
    }


def band_stats(name, arr, mask_bin=None, symmetric=False):
    stats = {
        'name': name,
        'shape': list(arr.shape),
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'energy': float(np.mean(arr ** 2)),
    }
    if mask_bin is not None:
        mask_r = np.array(
            Image.fromarray((mask_bin * 255).astype(np.uint8)).resize(
                (arr.shape[1], arr.shape[0]), Image.NEAREST
            )
        ) > 127
        use = np.abs(arr) if symmetric else arr
        fg = use[mask_r]
        bg = use[~mask_r]
        stats['fg_energy'] = float(np.mean(fg ** 2)) if fg.size else 0.0
        stats['bg_energy'] = float(np.mean(bg ** 2)) if bg.size else 0.0
        stats['fg_minus_bg_energy'] = stats['fg_energy'] - stats['bg_energy']
    return stats


def render_sample(paths, levels=1, pool_factor=2, out_path=None):
    pet = normalize_cipa(imread_gray(paths['pet_path']))

    mask_bin = None
    if os.path.isfile(paths['mask_path']):
        mask = imread_gray(paths['mask_path'])
        mask_bin = (mask >= 0.5).astype(np.float32)

    dec = laplacian_decouple(pet, levels=levels, pool_factor=pool_factor)

    stats = {
        'image_id': paths['image_id'],
        'norm_mode': 'cipa',
        'levels': levels,
        'pool_factor': pool_factor,
        'input_shape': list(pet.shape),
        'x_ll_shape': list(dec['x_ll'].shape),
        'recon_mae': float(dec['recon_err'].mean()),
        'recon_max_err': float(dec['recon_err'].max()),
        'bands': [
            band_stats('X_ll_up', dec['x_ll_up'], mask_bin, symmetric=False),
            band_stats('X_hh', dec['x_hh'], mask_bin, symmetric=True),
        ],
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f'Laplacian pyramid decouple | {paths["image_id"]} | '
        f'levels={levels}, pool={pool_factor}x, norm=cipa',
        fontsize=12,
    )

    axes[0, 0].imshow(pet, cmap='gray', vmin=-1.6, vmax=1.6)
    axes[0, 0].set_title('CIPA PET')
    if mask_bin is not None:
        axes[0, 0].contour(mask_bin, levels=[0.5], colors='lime', linewidths=0.8)

    axes[0, 1].imshow(dec['x_ll'], cmap='gray')
    axes[0, 1].set_title(f'X_ll (low-freq)\n{dec["x_ll"].shape[0]}x{dec["x_ll"].shape[1]}')

    axes[0, 2].imshow(dec['x_ll_up'], cmap='gray')
    axes[0, 2].set_title('Upsample(X_ll)\nlow-freq @ full res')
    if mask_bin is not None:
        axes[0, 2].contour(mask_bin, levels=[0.5], colors='lime', linewidths=0.8)

    im_hh = axes[0, 3].imshow(normalize_display(dec['x_hh'], symmetric=True), cmap='coolwarm')
    hh_stats = stats['bands'][1]
    title = 'X_hh (high-freq merged)'
    if 'fg_minus_bg_energy' in hh_stats:
        title += f"\nfg-bg energy={hh_stats['fg_minus_bg_energy']:.2e}"
    axes[0, 3].set_title(title)
    plt.colorbar(im_hh, ax=axes[0, 3], fraction=0.046)
    if mask_bin is not None:
        axes[0, 3].contour(mask_bin, levels=[0.5], colors='lime', linewidths=0.8)

    # per-level Laplacian residuals
    for i, lap in enumerate(dec['laplacian'][:2]):
        ax = axes[1, i]
        ax.imshow(normalize_display(lap, symmetric=True), cmap='coolwarm')
        ax.set_title(f'Laplacian L{i}\n(residual level {i})')
        if mask_bin is not None and lap.shape == mask_bin.shape:
            ax.contour(mask_bin, levels=[0.5], colors='lime', linewidths=0.8)
        ax.axis('off')

    axes[1, 2].imshow(dec['recon'], cmap='gray', vmin=-1.6, vmax=1.6)
    axes[1, 2].set_title(f'Reconstruction\nMAE={stats["recon_mae"]:.2e}')

    ax = axes[1, 3]
    energies = [float(np.mean(g ** 2)) for g in dec['gaussian']]
    lap_energies = [float(np.mean(l ** 2)) for l in dec['laplacian']]
    xs = np.arange(len(energies))
    ax.bar(xs - 0.15, energies, width=0.3, label='Gaussian G_k', color='steelblue')
    ax2_xs = np.arange(len(lap_energies))
    ax.bar(ax2_xs + 0.15, lap_energies, width=0.3, label='Laplacian L_k', color='coral')
    ax.set_xlabel('pyramid level')
    ax.set_ylabel('mean energy')
    ax.set_title('Pyramid energy by level')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    for ax in axes.ravel()[:7]:
        if ax is not axes[1, 3]:
            ax.axis('off')

    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    compare_path = out_path.replace('.png', '_compare.png') if out_path else None
    if compare_path:
        fig2, axes2 = plt.subplots(1, 5, figsize=(16, 4))
        fig2.suptitle(f'{paths["image_id"]} | Laplacian low/high decouple', fontsize=11)
        panels = [
            (pet, 'CIPA PET', 'gray', (-1.6, 1.6), False),
            (dec['x_ll_up'], 'X_ll_up (low)', 'gray', None, False),
            (dec['x_hh'], 'X_hh (high)', 'coolwarm', None, True),
            (normalize_display(dec['x_hh'], symmetric=True), 'X_hh (norm)', 'magma', None, False),
            (np.abs(dec['x_hh']), '|X_hh|', 'viridis', None, False),
        ]
        for ax, (arr, title, cmap, vr, sym) in zip(axes2, panels):
            if vr:
                ax.imshow(arr, cmap=cmap, vmin=vr[0], vmax=vr[1])
            elif sym:
                ax.imshow(normalize_display(arr, symmetric=True), cmap=cmap)
            else:
                ax.imshow(normalize_display(arr) if cmap == 'magma' or cmap == 'viridis' else arr, cmap=cmap)
            if mask_bin is not None and arr.shape == mask_bin.shape:
                ax.contour(mask_bin, levels=[0.5], colors='lime', linewidths=0.8)
            ax.set_title(title, fontsize=9)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(compare_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)

    return stats


def main():
    parser = argparse.ArgumentParser(description='Laplacian pyramid PET decouple vis')
    parser.add_argument('--root', type=str, default='/root/autodl-tmp/data/PCLT20K')
    parser.add_argument('--levels', type=int, default=1, help='Laplacian pyramid levels')
    parser.add_argument('--pool_factor', type=int, default=2)
    parser.add_argument('--out_dir', type=str,
                        default='/root/autodl-tmp/mkd-main/new-train/experiments/pet_laplacian_vis')
    parser.add_argument('--image_ids', type=str, nargs='+',
                        default=['0139_008', '0487_037', '0487_001', '0441_040'])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_stats = []

    for image_id in args.image_ids:
        paths = resolve_paths(args.root, image_id)
        if not os.path.isfile(paths['pet_path']):
            print(f'[skip] missing PET: {paths["pet_path"]}')
            continue
        tag = f'l{args.levels}_p{args.pool_factor}'
        out_path = os.path.join(args.out_dir, f'{image_id}_laplacian_{tag}_cipa.png')
        stats = render_sample(
            paths, levels=args.levels, pool_factor=args.pool_factor, out_path=out_path,
        )
        all_stats.append(stats)
        print(f'[saved] {out_path}')
        print(f'[saved] {out_path.replace(".png", "_compare.png")}')

    summary_path = os.path.join(args.out_dir, f'stats_{tag}_cipa.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'norm_mode': 'cipa',
            'levels': args.levels,
            'pool_factor': args.pool_factor,
            'samples': all_stats,
        }, f, indent=2)
    print(f'[saved] {summary_path}')


if __name__ == '__main__':
    main()
