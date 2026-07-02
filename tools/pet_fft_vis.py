#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize 2D FFT / IFFT on normalized PET images.

Shows magnitude spectrum, phase, low/high-pass filtering, and IFFT reconstruction.
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

IMAGENET_MEAN = 0.485
IMAGENET_STD = 0.229


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


def normalize_pet(img, mode='minmax'):
    """Normalize PET before FFT (match common training preprocess)."""
    if mode == 'minmax':
        # already [0, 1] from imread_gray
        return img.astype(np.float32)
    if mode == 'cipa':
        return img * 3.2 - 1.6
    if mode == 'imagenet':
        return (img - IMAGENET_MEAN) / IMAGENET_STD
    if mode == 'zscore':
        m, s = img.mean(), img.std()
        return (img - m) / max(s, 1e-8)
    raise ValueError(f'unknown norm mode: {mode}')


def normalize_display(arr, p_low=2, p_high=98, log=False):
    x = arr.copy()
    if log:
        x = np.log1p(np.abs(x))
    lo, hi = np.percentile(x, [p_low, p_high])
    if hi - lo < 1e-8:
        return np.clip(x, 0, 1)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def fft2_forward(img):
    F = np.fft.fft2(img)
    F_shift = np.fft.fftshift(F)
    mag = np.abs(F_shift)
    phase = np.angle(F_shift)
    return F, F_shift, mag, phase


def fft2_inverse(F):
    recon = np.fft.ifft2(F).real
    return recon


def radial_mask(shape, radius_frac, high_pass=False):
    """Circular mask in shifted frequency domain. radius_frac in (0, 1]."""
    h, w = shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = np.sqrt(cy ** 2 + cx ** 2)
    r = radius_frac * max_r
    mask = dist <= r
    if high_pass:
        mask = ~mask
    return mask.astype(np.float32)


def apply_freq_filter(F_shift, radius_frac, high_pass=False):
    mask = radial_mask(F_shift.shape, radius_frac, high_pass=high_pass)
    F_filt = F_shift * mask
    F_unshift = np.fft.ifftshift(F_filt)
    spatial = fft2_inverse(F_unshift)
    return spatial, mask


def render_sample(paths, norm_mode='minmax', out_path=None):
    pet_raw = imread_gray(paths['pet_path'])
    pet = normalize_pet(pet_raw, norm_mode)

    mask_bin = None
    if os.path.isfile(paths['mask_path']):
        mask = imread_gray(paths['mask_path'])
        mask_bin = (mask >= 0.5).astype(np.float32)

    F, F_shift, mag, phase = fft2_forward(pet)
    recon = fft2_inverse(F)
    recon_err = np.abs(pet - recon)

    low_spatial, low_mask = apply_freq_filter(F_shift, radius_frac=0.12, high_pass=False)
    high_spatial, high_mask = apply_freq_filter(F_shift, radius_frac=0.12, high_pass=True)

    stats = {
        'image_id': paths['image_id'],
        'norm_mode': norm_mode,
        'input_shape': list(pet.shape),
        'recon_mae': float(recon_err.mean()),
        'recon_max_err': float(recon_err.max()),
        'mag_dc': float(mag[pet.shape[0] // 2, pet.shape[1] // 2]),
        'mag_mean': float(mag.mean()),
        'mag_std': float(mag.std()),
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f'PET FFT / IFFT | {paths["image_id"]} | norm={norm_mode}',
        fontsize=12,
    )

    axes[0, 0].imshow(pet, cmap='gray')
    axes[0, 0].set_title(f'Normalized PET\n[{pet.min():.2f}, {pet.max():.2f}]')
    if mask_bin is not None:
        axes[0, 0].contour(mask_bin, levels=[0.5], colors='lime', linewidths=0.8)

    im_mag = axes[0, 1].imshow(normalize_display(mag, log=True), cmap='inferno')
    axes[0, 1].set_title('FFT magnitude\nlog(1+|F|), centered')
    plt.colorbar(im_mag, ax=axes[0, 1], fraction=0.046)

    im_phase = axes[0, 2].imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0, 2].set_title('FFT phase')
    plt.colorbar(im_phase, ax=axes[0, 2], fraction=0.046)

    axes[0, 3].imshow(recon, cmap='gray')
    axes[0, 3].set_title(f'IFFT reconstruction\nMAE={stats["recon_mae"]:.2e}')

    axes[1, 0].imshow(low_mask, cmap='gray')
    axes[1, 0].set_title('Low-pass mask\n(center 12% radius)')

    axes[1, 1].imshow(low_spatial, cmap='gray')
    axes[1, 1].set_title('Low-pass (smooth / hotspot)')
    if mask_bin is not None:
        axes[1, 1].contour(mask_bin, levels=[0.5], colors='lime', linewidths=0.8)

    axes[1, 2].imshow(high_spatial, cmap='gray')
    axes[1, 2].set_title('High-pass (edges / detail)')
    if mask_bin is not None:
        axes[1, 2].contour(mask_bin, levels=[0.5], colors='lime', linewidths=0.8)

    # spectrum radial profile
    h, w = pet.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    max_r = dist.max()
    radial_mean = np.zeros(max_r + 1)
    for r in range(max_r + 1):
        vals = mag[dist == r]
        radial_mean[r] = vals.mean() if vals.size else 0.0
    axes[1, 3].semilogy(radial_mean + 1e-12, color='steelblue')
    axes[1, 3].set_xlabel('radius from DC')
    axes[1, 3].set_ylabel('mean magnitude')
    axes[1, 3].set_title('Radial spectrum profile')
    axes[1, 3].grid(True, alpha=0.3)

    for ax in axes.ravel()[:7]:
        ax.axis('off')

    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Side-by-side: original vs low vs high
    compare_path = out_path.replace('.png', '_compare.png') if out_path else None
    if compare_path:
        fig2, axes2 = plt.subplots(1, 4, figsize=(14, 4))
        fig2.suptitle(f'{paths["image_id"]} | spatial domain comparison', fontsize=11)
        panels = [
            (pet, 'Normalized PET', 'gray'),
            (low_spatial, 'Low-pass', 'gray'),
            (high_spatial, 'High-pass', 'gray'),
            (normalize_display(high_spatial), 'High-pass (display norm)', 'magma'),
        ]
        for ax, (arr, title, cmap) in zip(axes2, panels):
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
    parser = argparse.ArgumentParser(description='Visualize PET FFT / IFFT')
    parser.add_argument('--root', type=str, default='/root/autodl-tmp/data/PCLT20K')
    parser.add_argument('--norm_mode', type=str, default='minmax',
                        choices=('minmax', 'cipa', 'imagenet', 'zscore'))
    parser.add_argument('--out_dir', type=str,
                        default='/root/autodl-tmp/mkd-main/new-train/experiments/pet_fft_vis')
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
        out_path = os.path.join(args.out_dir, f'{image_id}_fft_{args.norm_mode}.png')
        stats = render_sample(paths, norm_mode=args.norm_mode, out_path=out_path)
        all_stats.append(stats)
        print(f'[saved] {out_path}')
        print(f'[saved] {out_path.replace(".png", "_compare.png")}')

    summary_path = os.path.join(args.out_dir, f'stats_{args.norm_mode}.json')
    with open(summary_path, 'w') as f:
        json.dump({'norm_mode': args.norm_mode, 'samples': all_stats}, f, indent=2)
    print(f'[saved] {summary_path}')


if __name__ == '__main__':
    main()
