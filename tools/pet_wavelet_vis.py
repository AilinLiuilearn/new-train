#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize 2D discrete wavelet transform (DWT) on raw PET images.

One-level DWT produces four subbands:
  LL (approximation), LH (horizontal), HL (vertical), HH (diagonal).
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pywt
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


def normalize_display(arr, p_low=2, p_high=98, symmetric=False):
    if symmetric:
        v = np.percentile(np.abs(arr), p_high)
        v = max(v, 1e-8)
        return np.clip(arr / v, -1, 1)
    lo, hi = np.percentile(arr, [p_low, p_high])
    if hi - lo < 1e-8:
        return np.clip(arr, 0, 1)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def dwt2_decompose(img, wavelet='db2', level=1):
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level, mode='symmetric')
    cA, details = coeffs[0], coeffs[1:]
    cH, cV, cD = details[0]
    return {
        'LL': cA,
        'LH': cH,
        'HL': cV,
        'HH': cD,
        'coeffs': coeffs,
    }


def idwt2_reconstruct(coeffs, wavelet='db2'):
    return pywt.waverec2(coeffs, wavelet=wavelet, mode='symmetric')


def subband_stats(name, arr, mask_bin=None):
    stats = {
        'name': name,
        'shape': list(arr.shape),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'energy': float(np.mean(arr ** 2)),
    }
    if mask_bin is not None:
        mask_down = np.array(
            Image.fromarray((mask_bin * 255).astype(np.uint8)).resize(
                (arr.shape[1], arr.shape[0]), Image.NEAREST
            )
        ) > 127
        fg = arr[mask_down]
        bg = arr[~mask_down]
        stats['fg_energy'] = float(np.mean(fg ** 2)) if fg.size else 0.0
        stats['bg_energy'] = float(np.mean(bg ** 2)) if bg.size else 0.0
        stats['fg_minus_bg_energy'] = stats['fg_energy'] - stats['bg_energy']
    return stats


def render_sample(paths, wavelet='db2', level=1, out_path=None):
    pet = imread_gray(paths['pet_path'])
    mask = None
    mask_bin = None
    if os.path.isfile(paths['mask_path']):
        mask = imread_gray(paths['mask_path'])
        mask_bin = (mask >= 0.5).astype(np.float32)

    bands = dwt2_decompose(pet, wavelet=wavelet, level=level)
    recon = idwt2_reconstruct(bands['coeffs'], wavelet=wavelet)
    recon = recon[: pet.shape[0], : pet.shape[1]]
    recon_err = np.abs(pet - recon)

    stats = {
        'image_id': paths['image_id'],
        'wavelet': wavelet,
        'level': level,
        'input_shape': list(pet.shape),
        'recon_mae': float(recon_err.mean()),
        'recon_max_err': float(recon_err.max()),
        'subbands': [],
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        f'PET 2D DWT | {paths["image_id"]} | wavelet={wavelet}, level={level}',
        fontsize=12,
    )

    axes[0, 0].imshow(pet, cmap='gray')
    axes[0, 0].set_title('Original PET')
    if mask_bin is not None:
        axes[0, 0].contour(mask_bin, levels=[0.5], colors='r', linewidths=0.8)

    titles = {
        'LL': 'LL (approx, low-freq)',
        'LH': 'LH (horizontal detail)',
        'HL': 'HL (vertical detail)',
        'HH': 'HH (diagonal detail)',
    }
    positions = {'LL': (0, 1), 'LH': (0, 2), 'HL': (1, 0), 'HH': (1, 1)}

    for name in ['LL', 'LH', 'HL', 'HH']:
        arr = bands[name]
        sb_stats = subband_stats(name, arr, mask_bin)
        stats['subbands'].append(sb_stats)
        r, c = positions[name]
        symmetric = name != 'LL'
        disp = normalize_display(arr, symmetric=symmetric)
        cmap = 'gray' if name == 'LL' else 'coolwarm'
        im = axes[r, c].imshow(disp, cmap=cmap)
        title = titles[name]
        if mask_bin is not None and 'fg_minus_bg_energy' in sb_stats:
            title += f"\nfg-bg energy={sb_stats['fg_minus_bg_energy']:.2e}"
        axes[r, c].set_title(title, fontsize=10)
        plt.colorbar(im, ax=axes[r, c], fraction=0.046)

    axes[1, 2].imshow(recon, cmap='gray')
    axes[1, 2].set_title(f'IDWT reconstruction\nMAE={stats["recon_mae"]:.2e}')
    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Mosaic: original size layout of four subbands
    mosaic_path = out_path.replace('.png', '_mosaic.png') if out_path else None
    if mosaic_path:
        ll = normalize_display(bands['LL'])
        lh = normalize_display(bands['LH'], symmetric=True)
        hl = normalize_display(bands['HL'], symmetric=True)
        hh = normalize_display(bands['HH'], symmetric=True)
        top = np.concatenate([ll, lh], axis=1)
        bottom = np.concatenate([hl, hh], axis=1)
        mosaic = np.concatenate([top, bottom], axis=0)

        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
        ax2.imshow(mosaic, cmap='gray')
        ax2.set_title(f'{paths["image_id"]} | LL|LH / HL|HH mosaic')
        ax2.axis('off')
        plt.tight_layout()
        plt.savefig(mosaic_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)

    return stats


def main():
    parser = argparse.ArgumentParser(description='Visualize PET 2D wavelet decomposition')
    parser.add_argument('--root', type=str, default='/root/autodl-tmp/data/PCLT20K')
    parser.add_argument('--wavelet', type=str, default='db2',
                        help='PyWavelets name, e.g. haar, db2, sym4')
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--out_dir', type=str,
                        default='/root/autodl-tmp/mkd-main/new-train/experiments/pet_wavelet_vis')
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
        out_path = os.path.join(args.out_dir, f'{image_id}_wavelet_{args.wavelet}.png')
        stats = render_sample(paths, wavelet=args.wavelet, level=args.level, out_path=out_path)
        all_stats.append(stats)
        print(f'[saved] {out_path}')
        print(f'[saved] {out_path.replace(".png", "_mosaic.png")}')

    summary_path = os.path.join(args.out_dir, f'stats_{args.wavelet}.json')
    with open(summary_path, 'w') as f:
        json.dump({'wavelet': args.wavelet, 'samples': all_stats}, f, indent=2)
    print(f'[saved] {summary_path}')


if __name__ == '__main__':
    main()
