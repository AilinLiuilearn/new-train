#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Band-ratio gated PET saliency: S_full × min-max(R), R = |X_hh| / (|X_ll_up| + eps).

Laplacian-style decouple at each scale:
  X_ll     = AvgPool(P)
  X_ll_up  = Upsample(X_ll)
  X_hh     = P - X_ll_up
  R        = |X_hh| / (|X_ll_up| + eps)
  S_ratio  = S_full × min-max(R)
  S_ratio2 = S_full × ReLU(R - mean(R)) / max   # alternative gate
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

STAGE_SCALES = (
    ('full_512', 512),
    ('c1_128', 128),
    ('c2_64', 64),
    ('c3_32', 32),
    ('c4_16', 16),
)


def imread_gray(path):
    return np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0


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


def metabolic_saliency(p_l, eps=1e-6):
    mean_val = float(p_l.mean())
    s_l = np.maximum(p_l - mean_val, 0.0)
    s_l = s_l / (float(s_l.max()) + eps)
    return mean_val, s_l


def avg_pool2d(img, factor=2):
    h, w = img.shape
    h2, w2 = h // factor, w // factor
    if h2 == 0 or w2 == 0:
        raise ValueError(f'image too small for pool factor={factor}: {img.shape}')
    crop = img[: h2 * factor, : w2 * factor]
    return crop.reshape(h2, factor, w2, factor).mean(axis=(1, 3)).astype(np.float32)


def upsample_bilinear(img, out_hw):
    out_h, out_w = out_hw
    return np.array(
        Image.fromarray(img.astype(np.float32)).resize((out_w, out_h), Image.BILINEAR),
        dtype=np.float32,
    )


def band_ratio_decompose(p_l, eps=1e-6):
    x_ll = avg_pool2d(p_l, factor=2)
    x_ll_up = upsample_bilinear(x_ll, p_l.shape)
    x_hh = p_l - x_ll_up
    r = np.abs(x_hh) / (np.abs(x_ll_up) + eps)
    return x_ll_up, x_hh, r


def band_ratio_saliency(p_l, s_full, r, eps=1e-6):
    r_norm = minmax_per_sample(r, eps=eps)
    s_ratio = s_full * r_norm
    s_ratio = s_ratio / (float(s_ratio.max()) + eps)

    r_mean = float(r.mean())
    r_gate = np.maximum(r - r_mean, 0.0)
    r_gate = r_gate / (float(r_gate.max()) + eps)
    s_ratio_relu = s_full * r_gate
    s_ratio_relu = s_ratio_relu / (float(s_ratio_relu.max()) + eps)
    return r_norm, r_gate, s_ratio, s_ratio_relu, r_mean


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


def process_scale(pet_raw, ct_raw, mask, hw, eps):
    pet_s = resize_bilinear(pet_raw, hw)
    ct_s = resize_bilinear(ct_raw, hw)
    ct_disp = ct_display(ct_s)
    mask_s = resize_nearest(mask, hw) if mask is not None else None

    p_l = minmax_per_sample(pet_s, eps=eps)
    _, s_full = metabolic_saliency(p_l, eps=eps)
    x_ll_up, x_hh, r = band_ratio_decompose(p_l, eps=eps)
    r_norm, r_gate, s_ratio, s_ratio_relu, r_mean = band_ratio_saliency(
        p_l, s_full, r, eps=eps,
    )

    stats = {
        'r_mean': r_mean,
        'ratio': guidance_stats(s_ratio, mask_s),
        'ratio_relu': guidance_stats(s_ratio_relu, mask_s),
        'full': guidance_stats(s_full, mask_s),
        'r_norm': guidance_stats(r_norm, mask_s),
    }
    return {
        'pet_s': pet_s,
        'ct_disp': ct_disp,
        'mask_s': mask_s,
        'p_l': p_l,
        'x_ll_up': x_ll_up,
        'x_hh': x_hh,
        'r': r,
        'r_norm': r_norm,
        'r_gate': r_gate,
        's_full': s_full,
        's_ratio': s_ratio,
        's_ratio_relu': s_ratio_relu,
        'stats': stats,
    }


def save_sample_figure(paths, out_path, eps=1e-6):
    pet_raw = imread_gray(paths['pet_path'])
    ct_raw = imread_gray(paths['ct_path'])
    mask = _load_mask(paths['mask_path'])

    n_scales = len(STAGE_SCALES)
    fig, axes = plt.subplots(n_scales, 9, figsize=(24, 3.0 * n_scales))
    if n_scales == 1:
        axes = axes[None, :]

    sample_stats = {}
    fig.suptitle(
        f'{paths["image_id"]} | Band-ratio gated PET saliency: S_full × min-max(R)',
        fontsize=13,
        y=0.995,
    )

    for row, (scale_name, hw) in enumerate(STAGE_SCALES):
        out = process_scale(pet_raw, ct_raw, mask, hw, eps)
        sample_stats[scale_name] = out['stats']
        mask_s = out['mask_s']
        ratio_st = out['stats']['ratio']
        fg_bg = ratio_st.get('fg_minus_bg')
        bg_mean = ratio_st.get('bg_mean')
        suffix = ''
        if fg_bg is not None and bg_mean is not None:
            suffix = f'\nfg-bg={fg_bg:.3f}, bg={bg_mean:.3f}'

        panels = [
            (out['ct_disp'], 'CT', 'gray', (0, 1), False, None),
            (out['p_l'], 'P min-max', 'hot', (0, 1), False, None),
            (np.abs(out['x_ll_up']), '|X_ll_up|', 'hot', None, False, None),
            (np.abs(out['x_hh']), '|X_hh|', 'viridis', None, False, None),
            (out['r_norm'], 'min-max(R)\n(band ratio)', 'plasma', (0, 1), False, None),
            (out['s_ratio'], f'S_ratio\n(S_full×R){suffix}', 'magma', (0, 1), False, None),
            (out['s_full'], 'S_full', 'magma', (0, 1), False, None),
            (out['ct_disp'], 'CT + S_ratio', 'gray', (0, 1), True, out['s_ratio']),
            (out['s_ratio'] - out['s_full'], 'S_ratio - S_full', 'coolwarm', None, False, None),
        ]

        for col, (arr, title, cmap, vr, overlay, sal) in enumerate(panels):
            ax = axes[row, col]
            if overlay and sal is not None:
                ax.imshow(arr, cmap='gray', vmin=0, vmax=1)
                ax.imshow(sal, cmap='magma', alpha=0.55, vmin=0, vmax=1)
            elif col == 8:
                v = max(abs(float(arr.min())), abs(float(arr.max())), 1e-6)
                ax.imshow(arr, cmap=cmap, vmin=-v, vmax=v)
            elif vr is None:
                lo, hi = np.percentile(arr, [2, 98])
                ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi if hi > lo else lo + 1e-6)
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
    out512 = process_scale(pet_raw, ct_raw, mask, hw, eps)
    mask_s = out512['mask_s']

    fig2, axes2 = plt.subplots(2, 5, figsize=(18, 7))
    fig2.suptitle(
        f'{paths["image_id"]} | Band-ratio @512 — focal hotspot vs diffuse uptake',
        fontsize=11,
    )

    row0 = [
        (out512['ct_disp'], 'CT', 'gray', (0, 1), False, None),
        (out512['r_norm'], 'min-max(R) gate', 'plasma', (0, 1), False, None),
        (out512['s_ratio'], 'S_ratio = S_full×R', 'magma', (0, 1), False, None),
        (out512['ct_disp'], 'CT + S_ratio', 'gray', (0, 1), True, out512['s_ratio']),
        (out512['s_ratio'] * out512['ct_disp'], 'S_ratio × CT (gate)', 'magma', None, False, None),
    ]
    row1 = [
        (out512['s_full'], 'S_full (baseline)', 'magma', (0, 1), False, None),
        (out512['s_ratio_relu'], 'S_ratio_relu\n(S_full×ReLU(R-mean))', 'magma', (0, 1), False, None),
        (out512['ct_disp'], 'CT + S_full', 'gray', (0, 1), True, out512['s_full']),
        (out512['s_ratio'] - out512['s_full'], 'S_ratio - S_full', 'coolwarm', None, False, None),
        (np.stack([out512['ct_disp'], out512['s_ratio'], out512['s_full']], axis=-1),
         'RGB: CT+S_ratio+S_full', None, None, False, None),
    ]

    for r, row_panels in enumerate([row0, row1]):
        for ax, (arr, title, cmap, vr, overlay, sal) in zip(axes2[r], row_panels):
            if title.startswith('RGB'):
                ax.imshow(np.clip(arr, 0, 1))
            elif overlay and sal is not None:
                ax.imshow(arr, cmap='gray', vmin=0, vmax=1)
                ax.imshow(sal, cmap='magma', alpha=0.55, vmin=0, vmax=1)
            elif title.startswith('S_ratio -'):
                v = max(abs(float(arr.min())), abs(float(arr.max())), 1e-6)
                ax.imshow(arr, cmap=cmap, vmin=-v, vmax=v)
            elif vr is None:
                vmax = max(float(arr.max()), 1e-6)
                ax.imshow(arr, cmap=cmap, vmin=0, vmax=vmax)
            else:
                ax.imshow(arr, cmap=cmap, vmin=vr[0], vmax=vr[1])
            if mask_s is not None:
                ax.contour(mask_s, levels=[0.5], colors='lime', linewidths=0.8)
            ax.set_title(title, fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(compare_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)

    deep_path = out_path.replace('.png', '_deep.png')
    fig3, axes3 = plt.subplots(2, 4, figsize=(16, 7))
    fig3.suptitle(
        f'{paths["image_id"]} | Deep c3/c4: S_ratio vs S_full vs S_loc-proxy (lower bg = less FP)',
        fontsize=11,
    )

    for idx, (scale_name, hw) in enumerate([('c3_32', 32), ('c4_16', 16)]):
        out = process_scale(pet_raw, ct_raw, mask, hw, eps)
        mask_s = out['mask_s']
        panels = [
            (out['s_ratio'], f'{scale_name} S_ratio\nbg={out["stats"]["ratio"].get("bg_mean")}', 'ratio'),
            (out['s_full'], f'{scale_name} S_full\nbg={out["stats"]["full"].get("bg_mean")}', 'full'),
            (out['r_norm'], f'{scale_name} min-max(R)', 'r'),
            (out['s_ratio'] - out['s_full'], f'{scale_name} S_ratio-S_full', 'diff'),
        ]
        for j, (arr, title, kind) in enumerate(panels):
            ax = axes3[idx, j]
            if kind == 'diff':
                v = max(abs(float(arr.min())), abs(float(arr.max())), 1e-6)
                ax.imshow(arr, cmap='coolwarm', vmin=-v, vmax=v)
            elif kind == 'r':
                ax.imshow(arr, cmap='plasma', vmin=0, vmax=1)
            else:
                ax.imshow(arr, cmap='magma', vmin=0, vmax=1)
            if mask_s is not None:
                ax.contour(mask_s, levels=[0.5], colors='lime', linewidths=0.8)
            ax.set_title(title, fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(deep_path, dpi=150, bbox_inches='tight')
    plt.close(fig3)

    tri_path = out_path.replace('.png', '_tri_compare.png')
    fig4, axes4 = plt.subplots(1, 4, figsize=(14, 4))
    fig4.suptitle(f'{paths["image_id"]} | @512 prior comparison: S_full vs S_ratio vs S_ratio_relu', fontsize=11)
    tri_panels = [
        (out512['s_full'], 'S_full'),
        (out512['s_ratio'], 'S_ratio (×R)'),
        (out512['s_ratio_relu'], 'S_ratio_relu'),
        (out512['ct_disp'], 'CT + S_ratio'),
    ]
    for ax, (arr, title) in zip(axes4, tri_panels):
        if title.startswith('CT +'):
            ax.imshow(out512['ct_disp'], cmap='gray', vmin=0, vmax=1)
            ax.imshow(out512['s_ratio'], cmap='magma', alpha=0.55, vmin=0, vmax=1)
        else:
            ax.imshow(arr, cmap='magma', vmin=0, vmax=1)
        if mask_s is not None:
            ax.contour(mask_s, levels=[0.5], colors='lime', linewidths=0.8)
        st = out512['stats']['full' if title == 'S_full' else ('ratio' if '×R' in title else 'ratio_relu')]
        fg_bg = st.get('fg_minus_bg')
        bg = st.get('bg_mean')
        extra = f'\nfg-bg={fg_bg:.3f}, bg={bg:.3f}' if fg_bg is not None and bg is not None else ''
        ax.set_title(title + extra, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(tri_path, dpi=150, bbox_inches='tight')
    plt.close(fig4)

    return sample_stats


def main():
    parser = argparse.ArgumentParser(description='Band-ratio gated PET saliency visualization')
    parser.add_argument('--root', type=str, default='/root/autodl-tmp/data/PCLT20K')
    parser.add_argument('--out_dir', type=str,
                        default='/root/autodl-tmp/mkd-main/new-train/experiments/pet_band_ratio_vis')
    parser.add_argument('--image_ids', type=str, nargs='+',
                        default=['0487_001', '0487_037', '0441_040', '0139_008', '0540_041'])
    parser.add_argument('--eps', type=float, default=1e-6)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_stats = {}

    for image_id in args.image_ids:
        paths = resolve_paths(args.root, image_id)
        if not os.path.isfile(paths['pet_path']):
            print(f'[skip] missing PET: {paths["pet_path"]}')
            continue
        out_path = os.path.join(args.out_dir, f'{image_id}_band_ratio.png')
        stats = save_sample_figure(paths, out_path, eps=args.eps)
        all_stats[image_id] = stats
        print(f'[saved] {out_path}')
        print(f'[saved] {out_path.replace(".png", "_compare.png")}')
        print(f'[saved] {out_path.replace(".png", "_deep.png")}')
        print(f'[saved] {out_path.replace(".png", "_tri_compare.png")}')

    stats_path = os.path.join(args.out_dir, 'stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            'pipeline': 'R=|X_hh|/(|X_ll_up|+eps); S_ratio=S_full*min-max(R)',
            'samples': all_stats,
        }, f, indent=2, ensure_ascii=False)
    print(f'[saved] {stats_path}')


if __name__ == '__main__':
    main()
