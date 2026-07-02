#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize 2D DCT / IDCT on CIPA-normalized PET images.

DCT-II: DC coefficient at top-left (0,0), energy concentrates in low frequencies.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

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
    """CIPA training normalization: [0,1] PET -> [-1.6, 1.6]."""
    return img * 3.2 - 1.6


def normalize_display(arr, p_low=2, p_high=98, log=False, abs_val=False):
    x = arr.copy()
    if abs_val:
        x = np.abs(x)
    if log:
        x = np.log1p(x)
    lo, hi = np.percentile(x, [p_low, p_high])
    if hi - lo < 1e-8:
        return np.clip(x, 0, 1)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def dct2_forward(img):
    coeffs = dctn(img, type=2, norm='ortho')
    mag = np.abs(coeffs)
    return coeffs, mag


def dct2_inverse(coeffs):
    return idctn(coeffs, type=2, norm='ortho')


def dct_block_mask(shape, keep_frac, high_pass=False):
    """Rectangular mask on DCT coeffs. keep_frac: fraction of H/W to keep from top-left."""
    h, w = shape
    kh = max(1, int(round(h * keep_frac)))
    kw = max(1, int(round(w * keep_frac)))
    mask = np.zeros((h, w), dtype=np.float32)
    mask[:kh, :kw] = 1.0
    if high_pass:
        mask = 1.0 - mask
    return mask, kh, kw


def apply_dct_filter(coeffs, keep_frac, high_pass=False):
    mask, kh, kw = dct_block_mask(coeffs.shape, keep_frac, high_pass=high_pass)
    filtered = coeffs * mask
    spatial = dct2_inverse(filtered)
    return spatial, mask, kh, kw


def render_sample(paths, keep_frac=0.12, out_path=None):
    pet_raw = imread_gray(paths['pet_path'])
    pet = normalize_cipa(pet_raw)

    mask_bin = None
    if os.path.isfile(paths['mask_path']):
        mask = imread_gray(paths['mask_path'])
        mask_bin = (mask >= 0.5).astype(np.float32)

    coeffs, mag = dct2_forward(pet)
    recon = dct2_inverse(coeffs)
    recon_err = np.abs(pet - recon)

    low_spatial, low_mask, lkh, lkw = apply_dct_filter(coeffs, keep_frac, high_pass=False)
    high_spatial, high_mask, _, _ = apply_dct_filter(coeffs, keep_frac, high_pass=True)

    h, w = pet.shape
    stats = {
        'image_id': paths['image_id'],
        'norm_mode': 'cipa',
        'input_shape': [h, w],
        'input_range': [float(pet.min()), float(pet.max())],
        'recon_mae': float(recon_err.mean()),
        'recon_max_err': float(recon_err.max()),
        'dc_coeff': float(coeffs[0, 0]),
        'keep_frac': keep_frac,
        'low_keep_block': [lkh, lkw],
        'coeff_energy_low_block': float(np.mean(coeffs[:lkh, :lkw] ** 2)),
        'coeff_energy_high_rest': float(np.mean(coeffs[lkh:, :lkw] ** 2) + np.mean(coeffs[:lkh, lkw:] ** 2) + np.mean(coeffs[lkh:, lkw:] ** 2)) / 3.0,
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f'PET DCT / IDCT | {paths["image_id"]} | norm=cipa | keep={keep_frac:.0%}',
        fontsize=12,
    )

    axes[0, 0].imshow(pet, cmap='gray', vmin=-1.6, vmax=1.6)
    axes[0, 0].set_title(f'CIPA PET\n[{pet.min():.2f}, {pet.max():.2f}]')
    if mask_bin is not None:
        axes[0, 0].contour(mask_bin, levels=[0.5], colors='lime', linewidths=0.8)

    im_mag = axes[0, 1].imshow(normalize_display(mag, log=True), cmap='inferno')
    axes[0, 1].set_title('DCT |coeff|\nlog scale, DC at top-left')
    plt.colorbar(im_mag, ax=axes[0, 1], fraction=0.046)

    im_signed = axes[0, 2].imshow(
        normalize_display(coeffs, log=False, abs_val=True),
        cmap='magma',
    )
    axes[0, 2].set_title('DCT coefficients\n|C| display norm')
    plt.colorbar(im_signed, ax=axes[0, 2], fraction=0.046)

    axes[0, 3].imshow(recon, cmap='gray', vmin=-1.6, vmax=1.6)
    axes[0, 3].set_title(f'IDCT reconstruction\nMAE={stats["recon_mae"]:.2e}')

    axes[1, 0].imshow(low_mask, cmap='gray')
    axes[1, 0].set_title(f'Low-freq mask\n(top-left {lkh}x{lkw})')

    axes[1, 1].imshow(low_spatial, cmap='gray')
    axes[1, 1].set_title('Low-freq IDCT\n(smooth / hotspot)')
    if mask_bin is not None:
        axes[1, 1].contour(mask_bin, levels=[0.5], colors='lime', linewidths=0.8)

    axes[1, 2].imshow(high_spatial, cmap='gray')
    axes[1, 2].set_title('High-freq IDCT\n(edges / detail)')
    if mask_bin is not None:
        axes[1, 2].contour(mask_bin, levels=[0.5], colors='lime', linewidths=0.8)

    # energy decay along DCT index (average over rows/cols)
    row_energy = np.mean(coeffs ** 2, axis=1)
    col_energy = np.mean(coeffs ** 2, axis=0)
    axes[1, 3].semilogy(row_energy + 1e-12, label='row mean energy', color='steelblue')
    axes[1, 3].semilogy(col_energy + 1e-12, label='col mean energy', color='coral', alpha=0.8)
    axes[1, 3].set_xlabel('DCT index')
    axes[1, 3].set_ylabel('mean coeff²')
    axes[1, 3].set_title('Coefficient energy decay')
    axes[1, 3].legend(fontsize=8)
    axes[1, 3].grid(True, alpha=0.3)

    for ax in axes.ravel()[:7]:
        ax.axis('off')

    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    compare_path = out_path.replace('.png', '_compare.png') if out_path else None
    if compare_path:
        fig2, axes2 = plt.subplots(1, 4, figsize=(14, 4))
        fig2.suptitle(f'{paths["image_id"]} | CIPA PET DCT spatial comparison', fontsize=11)
        panels = [
            (pet, 'CIPA PET', 'gray', (-1.6, 1.6)),
            (low_spatial, 'Low-freq IDCT', 'gray', None),
            (high_spatial, 'High-freq IDCT', 'gray', None),
            (normalize_display(high_spatial), 'High-freq (display norm)', 'magma', None),
        ]
        for ax, (arr, title, cmap, vr) in zip(axes2, panels):
            if vr:
                ax.imshow(arr, cmap=cmap, vmin=vr[0], vmax=vr[1])
            else:
                ax.imshow(arr, cmap=cmap)
            if mask_bin is not None:
                ax.contour(mask_bin, levels=[0.5], colors='lime', linewidths=0.8)
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(compare_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)

    return stats


def main():
    parser = argparse.ArgumentParser(description='Visualize PET DCT / IDCT (CIPA norm)')
    parser.add_argument('--root', type=str, default='/root/autodl-tmp/data/PCLT20K')
    parser.add_argument('--keep_frac', type=float, default=0.12,
                        help='Fraction of DCT block to keep for low/high split')
    parser.add_argument('--out_dir', type=str,
                        default='/root/autodl-tmp/mkd-main/new-train/experiments/pet_dct_vis')
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
        out_path = os.path.join(args.out_dir, f'{image_id}_dct_cipa.png')
        stats = render_sample(paths, keep_frac=args.keep_frac, out_path=out_path)
        all_stats.append(stats)
        print(f'[saved] {out_path}')
        print(f'[saved] {out_path.replace(".png", "_compare.png")}')

    summary_path = os.path.join(args.out_dir, 'stats_cipa.json')
    with open(summary_path, 'w') as f:
        json.dump({'norm_mode': 'cipa', 'keep_frac': args.keep_frac, 'samples': all_stats}, f, indent=2)
    print(f'[saved] {summary_path}')


if __name__ == '__main__':
    main()
