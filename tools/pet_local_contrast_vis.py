#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize local-contrast PET saliency (DoG-style) for CT attention prior exploration.

Pipeline per scale:
  P      = min-max(PET)
  B      = GaussianBlur(P, sigma)   # local baseline
  C      = ReLU(P - B)              # local contrast
  S_loc  = C / (max(C) + eps)       # normalized saliency

Compare against S_full (ReLU(P-mean)/max) and raw min-max P.
Metrics: tumor fg-bg contrast, bg mean (lower = fewer deep FP), top10% tumor coverage.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

plt.rcParams['axes.unicode_minus'] = False

STAGE_SCALES = (
    ('full_512', 512),
    ('c1_128', 128),
    ('c2_64', 64),
    ('c3_32', 32),
    ('c4_16', 16),
)


def imread_gray(path):
    return np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0


def normalize_cipa(img):
    return img * 3.2 - 1.6


def resize_bilinear(img, size):
    return np.array(
        Image.fromarray(img.astype(np.float32)).resize((size, size), Image.BILINEAR),
        dtype=np.float32,
    )


def resize_nearest(mask, size):
    arr = (mask * 255).astype(np.uint8)
    return (
        np.array(Image.fromarray(arr).resize((size, size), Image.NEAREST), dtype=np.float32) > 127
    ).astype(np.float32)


def minmax_per_sample(x, eps=1e-6):
    x_min = float(x.min())
    x_max = float(x.max())
    return (x - x_min) / (x_max - x_min + eps)


def metabolic_saliency(p_l, eps=1e-6, threshold='mean'):
    """S = ReLU(P - threshold) / max."""
    if threshold == 'mean':
        thr = float(p_l.mean())
    elif threshold == 'median':
        thr = float(np.median(p_l))
    elif threshold == 'p70':
        thr = float(np.percentile(p_l, 70))
    else:
        thr = float(threshold)
    s_l = np.maximum(p_l - thr, 0.0)
    s_l = s_l / (float(s_l.max()) + eps)
    return thr, s_l


def stage_sigma(hw, sigma_mode='adaptive', fixed_sigma=3.0):
    if sigma_mode == 'fixed':
        return fixed_sigma
    return max(0.8, hw / 64.0)


def local_contrast_saliency(p_l, sigma, eps=1e-6, blur_mode='gaussian'):
    if blur_mode == 'gaussian':
        baseline = ndimage.gaussian_filter(p_l, sigma=sigma, mode='nearest')
    else:
        k = max(3, int(round(sigma * 2)) | 1)
        baseline = ndimage.uniform_filter(p_l, size=k, mode='nearest')
    contrast = np.maximum(p_l - baseline, 0.0)
    s_loc = contrast / (float(contrast.max()) + eps)
    return baseline, contrast, s_loc


def ct_display(ct_raw, p_low=1, p_high=99):
    lo, hi = np.percentile(ct_raw, [p_low, p_high])
    x = np.clip(ct_raw, lo, hi)
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def guidance_stats(s_map, mask_bin):
    stats = {
        'saliency_nonzero_ratio': float((s_map > 1e-6).mean()),
        'saliency_max': float(s_map.max()),
        'fg_mean': None,
        'bg_mean': None,
        'fg_minus_bg': None,
        'fg_coverage_in_top10pct': None,
    }
    if mask_bin is not None and mask_bin.shape == s_map.shape:
        fg = mask_bin > 0.5
        bg = ~fg
        if fg.any():
            stats['fg_mean'] = float(s_map[fg].mean())
        if bg.any():
            stats['bg_mean'] = float(s_map[bg].mean())
        if stats['fg_mean'] is not None and stats['bg_mean'] is not None:
            stats['fg_minus_bg'] = stats['fg_mean'] - stats['bg_mean']
        thresh = np.percentile(s_map, 90)
        top = s_map >= thresh
        if top.any():
            stats['fg_coverage_in_top10pct'] = float((top & fg).sum() / max(fg.sum(), 1))
    return stats


def resolve_paths(root, image_id):
    case_id, slice_id = image_id.split('_', 1)
    case_dir = os.path.join(root, case_id)
    base = f'{case_id}_{slice_id}'
    return {
        'image_id': image_id,
        'pet_path': os.path.join(case_dir, f'{base}_PET.png'),
        'ct_path': os.path.join(case_dir, f'{base}_CT.png'),
        'mask_path': os.path.join(case_dir, f'{base}_mask.png'),
    }


def _load_mask(path):
    if not os.path.isfile(path):
        return None
    return (imread_gray(path) > 0.5).astype(np.float32)


def process_scale(pet_raw, ct_raw, mask, hw, sigma_mode, fixed_sigma, blur_mode, eps):
    pet_s = resize_bilinear(pet_raw, hw)
    ct_s = resize_bilinear(ct_raw, hw)
    ct_disp = ct_display(ct_s)
    mask_s = resize_nearest(mask, hw) if mask is not None else None

    p_l = minmax_per_sample(pet_s, eps=eps)
    sigma = stage_sigma(hw, sigma_mode=sigma_mode, fixed_sigma=fixed_sigma)
    baseline, contrast, s_loc = local_contrast_saliency(
        p_l, sigma=sigma, eps=eps, blur_mode=blur_mode,
    )
    _, s_full = metabolic_saliency(p_l, eps=eps, threshold='mean')

    stats = {
        'sigma': float(sigma),
        'local': guidance_stats(s_loc, mask_s),
        'full': guidance_stats(s_full, mask_s),
        'minmax': guidance_stats(p_l, mask_s),
    }
    return {
        'pet_s': pet_s,
        'ct_disp': ct_disp,
        'mask_s': mask_s,
        'p_l': p_l,
        'baseline': baseline,
        'contrast': contrast,
        's_loc': s_loc,
        's_full': s_full,
        'sigma': sigma,
        'stats': stats,
    }


def save_sample_figure(paths, out_path, sigma_mode='adaptive', fixed_sigma=3.0,
                       blur_mode='gaussian', eps=1e-6):
    pet_raw = imread_gray(paths['pet_path'])
    ct_raw = imread_gray(paths['ct_path'])
    mask = _load_mask(paths['mask_path'])

    n_scales = len(STAGE_SCALES)
    fig, axes = plt.subplots(n_scales, 8, figsize=(22, 3.0 * n_scales))
    if n_scales == 1:
        axes = axes[None, :]

    sample_stats = {}
    fig.suptitle(
        f'{paths["image_id"]} | Local-contrast PET saliency (DoG) vs full saliency',
        fontsize=13,
        y=0.995,
    )

    for row, (scale_name, hw) in enumerate(STAGE_SCALES):
        out = process_scale(
            pet_raw, ct_raw, mask, hw, sigma_mode, fixed_sigma, blur_mode, eps,
        )
        sample_stats[scale_name] = out['stats']
        mask_s = out['mask_s']
        loc_st = out['stats']['local']
        fg_bg = loc_st.get('fg_minus_bg')
        bg_mean = loc_st.get('bg_mean')
        suffix = ''
        if fg_bg is not None and bg_mean is not None:
            suffix = f'\nfg-bg={fg_bg:.3f}, bg={bg_mean:.3f}'

        panels = [
            (out['ct_disp'], 'CT', 'gray', (0, 1), False, None),
            (out['p_l'], 'P min-max', 'hot', (0, 1), False, None),
            (out['baseline'], f'B blur\nσ={out["sigma"]:.1f}', 'hot', (0, 1), False, None),
            (out['contrast'], 'ReLU(P-B)\nlocal contrast', 'inferno', None, False, None),
            (out['s_loc'], f'S_loc\n(local saliency){suffix}', 'magma', (0, 1), False, None),
            (out['s_full'], 'S_full\n(full saliency)', 'magma', (0, 1), False, None),
            (out['ct_disp'], 'CT + S_loc', 'gray', (0, 1), True, out['s_loc']),
            (out['s_loc'] - out['s_full'], 'S_loc - S_full', 'coolwarm', None, False, None),
        ]

        for col, (arr, title, cmap, vr, overlay, sal) in enumerate(panels):
            ax = axes[row, col]
            if overlay and sal is not None:
                ax.imshow(arr, cmap='gray', vmin=0, vmax=1)
                ax.imshow(sal, cmap='magma', alpha=0.55, vmin=0, vmax=1)
            elif col == 7:
                v = max(abs(float(arr.min())), abs(float(arr.max())), 1e-6)
                ax.imshow(arr, cmap=cmap, vmin=-v, vmax=v)
            elif vr is None:
                vmax = max(float(arr.max()), 1e-6)
                ax.imshow(arr, cmap=cmap, vmin=0, vmax=vmax)
            else:
                ax.imshow(arr, cmap=cmap, vmin=vr[0], vmax=vr[1])
            if mask_s is not None and arr.shape == mask_s.shape:
                ax.contour(mask_s, levels=[0.5], colors='lime', linewidths=0.7)
            ax.set_title(f'{scale_name} | {title}', fontsize=7)
            ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    compare_path = out_path.replace('.png', '_compare.png')
    hw = 512
    out512 = process_scale(
        pet_raw, ct_raw, mask, hw, sigma_mode, fixed_sigma, blur_mode, eps,
    )
    mask_s = out512['mask_s']

    fig2, axes2 = plt.subplots(2, 5, figsize=(18, 7))
    fig2.suptitle(
        f'{paths["image_id"]} | Local contrast @512 (σ={out512["sigma"]:.1f}) — '
        f'lesion highlight & background suppression',
        fontsize=11,
    )

    row0 = [
        (out512['ct_disp'], 'CT', 'gray', (0, 1), False, None),
        (out512['p_l'], 'P (min-max PET)', 'hot', (0, 1), False, None),
        (out512['baseline'], f'B = blur(P, σ={out512["sigma"]:.1f})', 'hot', (0, 1), False, None),
        (out512['s_loc'], 'S_loc (local contrast)', 'magma', (0, 1), False, None),
        (out512['ct_disp'], 'CT + S_loc', 'gray', (0, 1), True, out512['s_loc']),
    ]
    row1 = [
        (out512['s_full'], 'S_full (baseline)', 'magma', (0, 1), False, None),
        (out512['ct_disp'], 'CT + S_full', 'gray', (0, 1), True, out512['s_full']),
        (out512['s_loc'] - out512['s_full'], 'S_loc - S_full\n(+ = local wins)', 'coolwarm', None, False, None),
        (out512['s_loc'] * out512['ct_disp'], 'S_loc × CT (gate)', 'magma', None, False, None),
        (np.stack([out512['ct_disp'], out512['s_loc'], out512['s_full']], axis=-1),
         'RGB: CT+S_loc+S_full', None, None, False, None),
    ]

    for r, row_panels in enumerate([row0, row1]):
        for ax, (arr, title, cmap, vr, overlay, sal) in zip(axes2[r], row_panels):
            if title.startswith('RGB'):
                ax.imshow(np.clip(arr, 0, 1))
            elif overlay and sal is not None:
                ax.imshow(arr, cmap='gray', vmin=0, vmax=1)
                ax.imshow(sal, cmap='magma', alpha=0.55, vmin=0, vmax=1)
            elif title.startswith('S_loc -'):
                v = max(abs(float(arr.min())), abs(float(arr.max())), 1e-6)
                ax.imshow(arr, cmap=cmap, vmin=-v, vmax=v)
            elif vr is None:
                vmax = max(float(arr.max()), 1e-6)
                ax.imshow(arr, cmap=cmap, vmin=0, vmax=vmax)
            else:
                ax.imshow(arr, cmap=cmap, vmin=vr[0], vmax=vr[1])
            if mask_s is not None and (arr.ndim == 2 or title.startswith('RGB')):
                if arr.ndim == 2 and arr.shape == mask_s.shape:
                    ax.contour(mask_s, levels=[0.5], colors='lime', linewidths=0.8)
                elif title.startswith('RGB'):
                    ax.contour(mask_s, levels=[0.5], colors='lime', linewidth=0.8)
            ax.set_title(title, fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(compare_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)

    deep_path = out_path.replace('.png', '_deep.png')
    fig3, axes3 = plt.subplots(2, 3, figsize=(12, 7))
    fig3.suptitle(
        f'{paths["image_id"]} | Deep layers c3/c4: S_loc suppresses diffuse uptake (lower bg = less FP)',
        fontsize=11,
    )
    for idx, (scale_name, hw) in enumerate([('c3_32', 32), ('c4_16', 16)]):
        out = process_scale(
            pet_raw, ct_raw, mask, hw, sigma_mode, fixed_sigma, blur_mode, eps,
        )
        mask_s = out['mask_s']
        for j, (arr, title, sal_name) in enumerate([
            (out['s_loc'], f'{scale_name} S_loc\nbg={out["stats"]["local"].get("bg_mean")}', 'local'),
            (out['s_full'], f'{scale_name} S_full\nbg={out["stats"]["full"].get("bg_mean")}', 'full'),
            (out['s_loc'] - out['s_full'], f'{scale_name} S_loc-S_full', 'diff'),
        ]):
            ax = axes3[idx, j]
            if sal_name == 'diff':
                v = max(abs(float(arr.min())), abs(float(arr.max())), 1e-6)
                ax.imshow(arr, cmap='coolwarm', vmin=-v, vmax=v)
            else:
                ax.imshow(arr, cmap='magma', vmin=0, vmax=1)
            if mask_s is not None:
                ax.contour(mask_s, levels=[0.5], colors='lime', linewidths=0.8)
            ax.set_title(title, fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(deep_path, dpi=150, bbox_inches='tight')
    plt.close(fig3)

    return sample_stats


def main():
    parser = argparse.ArgumentParser(description='Local-contrast PET saliency visualization')
    parser.add_argument('--root', type=str, default='/root/autodl-tmp/data/PCLT20K')
    parser.add_argument('--out_dir', type=str,
                        default='/root/autodl-tmp/mkd-main/new-train/experiments/pet_local_contrast_vis')
    parser.add_argument('--image_ids', type=str, nargs='+',
                        default=['0487_001', '0487_037', '0441_040', '0139_008', '0540_041'])
    parser.add_argument('--sigma_mode', type=str, default='adaptive', choices=['adaptive', 'fixed'])
    parser.add_argument('--fixed_sigma', type=float, default=3.0)
    parser.add_argument('--blur_mode', type=str, default='gaussian', choices=['gaussian', 'box'])
    parser.add_argument('--eps', type=float, default=1e-6)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_stats = {}
    tag = f'{args.sigma_mode}_b{args.blur_mode}'

    for image_id in args.image_ids:
        paths = resolve_paths(args.root, image_id)
        if not os.path.isfile(paths['pet_path']):
            print(f'[skip] missing PET: {paths["pet_path"]}')
            continue
        out_path = os.path.join(args.out_dir, f'{image_id}_local_contrast_{tag}.png')
        stats = save_sample_figure(
            paths, out_path,
            sigma_mode=args.sigma_mode,
            fixed_sigma=args.fixed_sigma,
            blur_mode=args.blur_mode,
            eps=args.eps,
        )
        all_stats[image_id] = stats
        print(f'[saved] {out_path}')
        print(f'[saved] {out_path.replace(".png", "_compare.png")}')
        print(f'[saved] {out_path.replace(".png", "_deep.png")}')

    stats_path = os.path.join(args.out_dir, f'stats_{tag}.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            'sigma_mode': args.sigma_mode,
            'blur_mode': args.blur_mode,
            'pipeline': 'S_loc = ReLU(P - blur(P,sigma)) / max',
            'samples': all_stats,
        }, f, indent=2, ensure_ascii=False)
    print(f'[saved] {stats_path}')


if __name__ == '__main__':
    main()
