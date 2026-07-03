#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-scale Laplacian high-frequency PET saliency for CT-guidance exploration.

At each encoder scale (512/128/64/32/16):
  1. Resize CIPA-normalized PET to stage resolution
  2. Laplacian pyramid decouple -> X_ll (low-freq) + X_hh (high-freq merged)
  3. Build high-freq saliency on |X_hh|:
       P_hh = min-max(|X_hh|)
       S_hh = ReLU(P_hh - mean(P_hh)) / (max + eps)
  4. Compare with full-PET saliency S_full (same pipeline on raw PET)
  5. Overlay S_hh on CT to inspect whether high-freq PET can guide CT attention
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


def metabolic_saliency(x, eps=1e-6, use_abs=False):
    """ReLU(min-max(x) - mean) / max, optionally on |x| for signed high-freq."""
    src = np.abs(x) if use_abs else x
    p_l = minmax_per_sample(src, eps=eps)
    mean_val = float(p_l.mean())
    s_l = np.maximum(p_l - mean_val, 0.0)
    s_l = s_l / (float(s_l.max()) + eps)
    return p_l, mean_val, s_l


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


def laplacian_decouple(x, levels=1, pool_factor=2):
    gaussian = [x.astype(np.float32)]
    for _ in range(levels):
        gaussian.append(avg_pool2d(gaussian[-1], factor=pool_factor))

    laplacian = []
    for i in range(levels):
        up = upsample_bilinear(gaussian[i + 1], gaussian[i].shape)
        laplacian.append(gaussian[i] - up)

    x_ll = gaussian[-1]
    x_ll_up = upsample_bilinear(x_ll, x.shape)

    x_hh = np.zeros_like(x, dtype=np.float32)
    for i, lap in enumerate(laplacian):
        merged = lap
        for _ in range(i):
            merged = upsample_bilinear(merged, (merged.shape[0] * 2, merged.shape[1] * 2))
        if merged.shape != x.shape:
            merged = upsample_bilinear(merged, x.shape)
        x_hh += merged

    return {
        'x_ll': x_ll,
        'x_ll_up': x_ll_up,
        'x_hh': x_hh,
        'laplacian': laplacian,
    }


def model_style_hh(pet):
    """Match pet_lap_hgl_prior: |Laplacian kernel| + |pet - up(pool(pet))|."""
    h, w = pet.shape
    x_ll = avg_pool2d(pet, factor=2)
    x_ll_up = upsample_bilinear(x_ll, (h, w))
    x_lap = pet - x_ll_up
    kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    x_hp = ndimage.convolve(pet, kernel, mode='nearest')
    hh_amp = 0.5 * (np.abs(x_hp) + np.abs(x_lap))
    return x_lap, x_hp, hh_amp


def ct_display(ct_raw, p_low=1, p_high=99):
    lo, hi = np.percentile(ct_raw, [p_low, p_high])
    x = np.clip(ct_raw, lo, hi)
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def guidance_stats(s_map, mask_bin, ct_grad=None):
    stats = {
        'saliency_nonzero_ratio': float((s_map > 1e-6).mean()),
        'saliency_max': float(s_map.max()),
        'fg_mean': None,
        'bg_mean': None,
        'fg_minus_bg': None,
        'fg_coverage_in_top10pct': None,
        'ct_grad_corr': None,
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
    if ct_grad is not None and ct_grad.shape == s_map.shape:
        a = s_map.ravel()
        b = ct_grad.ravel()
        if a.std() > 1e-8 and b.std() > 1e-8:
            stats['ct_grad_corr'] = float(np.corrcoef(a, b)[0, 1])
    return stats


def ct_gradient_mag(ct):
    gx = ndimage.sobel(ct, axis=1)
    gy = ndimage.sobel(ct, axis=0)
    g = np.sqrt(gx ** 2 + gy ** 2)
    return g / (g.max() + 1e-6)


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


def process_scale(pet_cipa, ct_raw, mask, hw, levels, pool_factor, eps):
    pet_s = resize_bilinear(pet_cipa, hw)
    ct_s = resize_bilinear(ct_raw, hw)
    ct_disp = ct_display(ct_s)
    mask_s = resize_nearest(mask, hw) if mask is not None else None

    dec = laplacian_decouple(pet_s, levels=levels, pool_factor=pool_factor)
    x_hh = dec['x_hh']
    hh_abs = np.abs(x_hh)

    x_lap, x_hp, hh_model = model_style_hh(pet_s)

    p_hh, mean_hh, s_hh = metabolic_saliency(x_hh, eps=eps, use_abs=True)
    _, _, s_hh_model = metabolic_saliency(hh_model, eps=eps, use_abs=False)
    p_full, mean_full, s_full = metabolic_saliency(pet_s, eps=eps, use_abs=False)

    ct_g = ct_gradient_mag(ct_disp)
    stats = {
        'pyramid_hh': guidance_stats(s_hh, mask_s, ct_g),
        'model_hh': guidance_stats(s_hh_model, mask_s, ct_g),
        'full_pet': guidance_stats(s_full, mask_s, ct_g),
        'mean_hh': mean_hh,
        'mean_full': mean_full,
        'hh_energy': float(np.mean(hh_abs ** 2)),
        'll_energy': float(np.mean(dec['x_ll_up'] ** 2)),
    }
    return {
        'pet_s': pet_s,
        'ct_disp': ct_disp,
        'mask_s': mask_s,
        'dec': dec,
        'hh_abs': hh_abs,
        'x_lap': x_lap,
        'x_hp': x_hp,
        'hh_model': hh_model,
        's_hh': s_hh,
        's_hh_model': s_hh_model,
        's_full': s_full,
        'p_hh': p_hh,
        'p_full': p_full,
        'ct_g': ct_g,
        'stats': stats,
    }


def save_sample_figure(paths, out_path, levels=1, pool_factor=2, eps=1e-6):
    pet_raw = imread_gray(paths['pet_path'])
    ct_raw = imread_gray(paths['ct_path'])
    mask = _load_mask(paths['mask_path'])
    pet_cipa = normalize_cipa(pet_raw)

    n_scales = len(STAGE_SCALES)
    fig, axes = plt.subplots(n_scales, 8, figsize=(22, 3.0 * n_scales))
    if n_scales == 1:
        axes = axes[None, :]

    sample_stats = {}
    fig.suptitle(
        f'{paths["image_id"]} | Multi-scale Laplacian high-freq PET saliency vs CT guidance',
        fontsize=13,
        y=0.995,
    )

    for row, (scale_name, hw) in enumerate(STAGE_SCALES):
        out = process_scale(pet_cipa, ct_raw, mask, hw, levels, pool_factor, eps)
        sample_stats[scale_name] = out['stats']
        mask_s = out['mask_s']
        st = out['stats']['pyramid_hh']
        fg_bg = st.get('fg_minus_bg')
        fg_top = st.get('fg_coverage_in_top10pct')
        title_suffix = ''
        if fg_bg is not None:
            title_suffix = f'\nfg-bg={fg_bg:.3f}, top10%={fg_top:.2f}' if fg_top is not None else f'\nfg-bg={fg_bg:.3f}'

        panels = [
            (out['ct_disp'], 'CT', 'gray', (0, 1), False),
            (out['pet_s'], 'CIPA PET', 'gray', (-1.6, 1.6), False),
            (out['dec']['x_ll_up'], 'X_ll_up\n(low-freq)', 'gray', None, False),
            (out['hh_abs'], '|X_hh|\n(pyramid high)', 'viridis', None, False),
            (out['s_hh'], f'S_hh\n(hf saliency){title_suffix}', 'magma', (0, 1), False),
            (out['s_full'], 'S_full\n(full PET saliency)', 'magma', (0, 1), False),
            (out['ct_disp'], 'CT + S_hh overlay', 'gray', (0, 1), True),
            (out['s_hh'] - out['s_full'], 'S_hh - S_full\n(hf vs full)', 'coolwarm', None, False),
        ]

        for col, (arr, title, cmap, vr, overlay) in enumerate(panels):
            ax = axes[row, col]
            if overlay:
                ax.imshow(arr, cmap='gray', vmin=0, vmax=1)
                ax.imshow(out['s_hh'], cmap='magma', alpha=0.55, vmin=0, vmax=1)
            elif col == 7:
                v = max(abs(float(arr.min())), abs(float(arr.max())), 1e-6)
                ax.imshow(arr, cmap=cmap, vmin=-v, vmax=v)
            elif vr is None:
                lo, hi = np.percentile(arr, [2, 98])
                ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi if hi > lo else lo + 1e-6)
            else:
                ax.imshow(arr, cmap=cmap, vmin=vr[0], vmax=vr[1])
            if mask_s is not None and arr.shape == mask_s.shape and col != 6:
                ax.contour(mask_s, levels=[0.5], colors='lime', linewidths=0.7)
            if mask_s is not None and col == 6:
                ax.contour(mask_s, levels=[0.5], colors='lime', linewidth=1.0)
            ax.set_title(f'{scale_name} | {title}', fontsize=7)
            ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    compare_path = out_path.replace('.png', '_compare.png')
    hw = 512
    out512 = process_scale(pet_cipa, ct_raw, mask, hw, levels, pool_factor, eps)
    fig2, axes2 = plt.subplots(2, 5, figsize=(18, 7))
    fig2.suptitle(
        f'{paths["image_id"]} | High-freq saliency CT-guidance @512 '
        f'(levels={levels}, pool={pool_factor}x)',
        fontsize=11,
    )
    mask_s = out512['mask_s']

    row0 = [
        (out512['ct_disp'], 'CT', 'gray', (0, 1), False, None),
        (out512['pet_s'], 'CIPA PET', 'gray', (-1.6, 1.6), False, None),
        (out512['hh_abs'], '|X_hh| pyramid', 'viridis', None, False, None),
        (out512['s_hh'], 'S_hh (hf saliency)', 'magma', (0, 1), False, None),
        (out512['ct_disp'], 'CT + S_hh', 'gray', (0, 1), True, out512['s_hh']),
    ]
    row1 = [
        (out512['hh_model'], 'hh_model |hp|+|lap|', 'viridis', None, False, None),
        (out512['s_hh_model'], 'S_hh_model', 'magma', (0, 1), False, None),
        (out512['s_full'], 'S_full (baseline)', 'magma', (0, 1), False, None),
        (out512['ct_g'], 'CT gradient', 'inferno', (0, 1), False, None),
        (out512['ct_disp'], 'CT + S_full', 'gray', (0, 1), True, out512['s_full']),
    ]

    for r, row_panels in enumerate([row0, row1]):
        for ax, (arr, title, cmap, vr, overlay, sal) in zip(axes2[r], row_panels):
            if overlay and sal is not None:
                ax.imshow(arr, cmap='gray', vmin=0, vmax=1)
                ax.imshow(sal, cmap='magma', alpha=0.55, vmin=0, vmax=1)
            elif vr is None:
                lo, hi = np.percentile(arr, [2, 98])
                ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi if hi > lo else lo + 1e-6)
            else:
                ax.imshow(arr, cmap=cmap, vmin=vr[0], vmax=vr[1])
            if mask_s is not None and arr.shape == mask_s.shape:
                ax.contour(mask_s, levels=[0.5], colors='lime', linewidth=0.8)
            ax.set_title(title, fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(compare_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)

    guide_path = out_path.replace('.png', '_guide.png')
    fig3, axes3 = plt.subplots(1, 4, figsize=(14, 4))
    fig3.suptitle(f'{paths["image_id"]} | Can high-freq PET guide CT? (fg=tumor, lime contour)', fontsize=11)
    s_hh, s_full, ct_d = out512['s_hh'], out512['s_full'], out512['ct_disp']
    guide_panels = [
        (np.stack([ct_d, ct_d, s_hh], axis=-1), 'RGB: CT+CT+S_hh', None),
        (np.stack([ct_d, s_hh, s_full], axis=-1), 'RGB: CT+S_hh+S_full', None),
        (s_hh * ct_d, 'S_hh * CT (gate proxy)', 'magma'),
        (np.abs(s_hh - s_full), '|S_hh - S_full|', 'hot'),
    ]
    for ax, (arr, title, cmap) in zip(axes3, guide_panels):
        if cmap is None:
            ax.imshow(np.clip(arr, 0, 1))
        else:
            ax.imshow(arr, cmap=cmap, vmin=0, vmax=max(float(arr.max()), 1e-6))
        if mask_s is not None:
            ax.contour(mask_s, levels=[0.5], colors='lime', linewidth=1.0)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(guide_path, dpi=150, bbox_inches='tight')
    plt.close(fig3)

    return sample_stats


def main():
    parser = argparse.ArgumentParser(
        description='Multi-scale Laplacian high-freq PET saliency for CT guidance vis',
    )
    parser.add_argument('--root', type=str, default='/root/autodl-tmp/data/PCLT20K')
    parser.add_argument('--out_dir', type=str,
                        default='/root/autodl-tmp/mkd-main/new-train/experiments/pet_lap_hf_saliency_vis')
    parser.add_argument('--image_ids', type=str, nargs='+',
                        default=['0487_001', '0487_037', '0441_040', '0139_008', '0540_041'])
    parser.add_argument('--levels', type=int, default=1, help='Laplacian pyramid levels per scale')
    parser.add_argument('--pool_factor', type=int, default=2)
    parser.add_argument('--eps', type=float, default=1e-6)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_stats = {}
    tag = f'l{args.levels}_p{args.pool_factor}'

    for image_id in args.image_ids:
        paths = resolve_paths(args.root, image_id)
        if not os.path.isfile(paths['pet_path']):
            print(f'[skip] missing PET: {paths["pet_path"]}')
            continue
        out_path = os.path.join(args.out_dir, f'{image_id}_lap_hf_saliency_{tag}.png')
        stats = save_sample_figure(
            paths, out_path, levels=args.levels, pool_factor=args.pool_factor, eps=args.eps,
        )
        all_stats[image_id] = stats
        print(f'[saved] {out_path}')
        print(f'[saved] {out_path.replace(".png", "_compare.png")}')
        print(f'[saved] {out_path.replace(".png", "_guide.png")}')

    stats_path = os.path.join(args.out_dir, f'stats_{tag}.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            'levels': args.levels,
            'pool_factor': args.pool_factor,
            'pipeline': 'Laplacian decouple -> S_hh on |X_hh| vs S_full on PET',
            'samples': all_stats,
        }, f, indent=2, ensure_ascii=False)
    print(f'[saved] {stats_path}')


if __name__ == '__main__':
    main()
