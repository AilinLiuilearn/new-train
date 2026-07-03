#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize PET metabolic saliency: ReLU(P_l - mean(P_l)) after per-sample min-max."""

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


def normalize_cipa(img):
    return img * 3.2 - 1.6


def resize_bilinear(img, size):
    return np.array(
        Image.fromarray(img.astype(np.float32)).resize((size, size), Image.BILINEAR),
        dtype=np.float32,
    )


def minmax_per_sample(x, eps=1e-6):
    x_min = float(x.min())
    x_max = float(x.max())
    return (x - x_min) / (x_max - x_min + eps)


def pet_metabolic_saliency(pet_resized, eps=1e-6):
    """
    pet_resized: [H,W] float, any range
    returns P_l [0,1], mean_val, S_l [0,1]
    """
    p_l = minmax_per_sample(pet_resized, eps=eps)
    mean_val = float(p_l.mean())
    s_l = np.maximum(p_l - mean_val, 0.0)
    s_l = s_l / (float(s_l.max()) + eps)
    return p_l, mean_val, s_l


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


def _panel_stats(p_l, s_l, mean_val, mask=None):
    stats = {
        'mean_p_l': mean_val,
        'saliency_nonzero_ratio': float((s_l > 1e-6).mean()),
        'saliency_max': float(s_l.max()),
        'p_l_fg_mean': None,
        's_l_fg_mean': None,
    }
    if mask is not None and mask.shape == p_l.shape:
        fg = mask > 0.5
        if fg.any():
            stats['p_l_fg_mean'] = float(p_l[fg].mean())
            stats['s_l_fg_mean'] = float(s_l[fg].mean())
    return stats


def save_sample_figure(paths, out_path, eps=1e-6):
    pet_raw = imread_gray(paths['pet_path'])
    mask = _load_mask(paths['mask_path'])
    pet_cipa = normalize_cipa(pet_raw)

    n_scales = len(STAGE_SCALES)
    fig, axes = plt.subplots(n_scales, 6, figsize=(18, 3.2 * n_scales))
    if n_scales == 1:
        axes = axes[None, :]

    sample_stats = {}
    fig.suptitle(
        f'{paths["image_id"]} | PET metabolic saliency pipeline',
        fontsize=13,
        y=0.995,
    )

    for row, (scale_name, hw) in enumerate(STAGE_SCALES):
        pet_resized = resize_bilinear(pet_raw, hw)
        pet_cipa_resized = resize_bilinear(pet_cipa, hw)
        p_l, mean_val, s_l = pet_metabolic_saliency(pet_resized, eps=eps)
        mask_s = resize_bilinear(mask, hw) if mask is not None else None
        if mask_s is not None:
            mask_s = (mask_s > 0.5).astype(np.float32)

        sample_stats[scale_name] = _panel_stats(p_l, s_l, mean_val, mask_s)

        panels = [
            (pet_resized, 'Raw PET\n[0,1] resize', 'gray', (0, 1)),
            (pet_cipa_resized, 'CIPA PET\n(x*3.2-1.6)', 'gray', (-1.6, 1.6)),
            (p_l, f'P_l min-max\nmean={mean_val:.3f}', 'hot', (0, 1)),
            (np.maximum(p_l - mean_val, 0), 'ReLU(P_l-mean)', 'inferno', None),
            (s_l, 'S_l saliency\n(focus map)', 'magma', (0, 1)),
            (s_l, 'S_l + mask', 'magma', (0, 1)),
        ]

        for col, (arr, title, cmap, vr) in enumerate(panels):
            ax = axes[row, col]
            if vr is None:
                vmax = max(float(arr.max()), 1e-6)
                ax.imshow(arr, cmap=cmap, vmin=0, vmax=vmax)
            else:
                ax.imshow(arr, cmap=cmap, vmin=vr[0], vmax=vr[1])
            if mask_s is not None and arr.shape == mask_s.shape and col != 5:
                ax.contour(mask_s, levels=[0.5], colors='lime', linewidths=0.7)
            if col == 5 and mask_s is not None:
                ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, alpha=0.85)
                ax.contour(mask_s, levels=[0.5], colors='lime', linewidths=1.0)
            ax.set_title(f'{scale_name} | {title}', fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    compare_path = out_path.replace('.png', '_compare.png')
    fig2, axes2 = plt.subplots(1, 5, figsize=(16, 4))
    hw = 512
    pet_resized = resize_bilinear(pet_raw, hw)
    p_l, mean_val, s_l = pet_metabolic_saliency(pet_resized, eps=eps)
    mask_s = resize_bilinear(mask, hw) if mask is not None else None
    if mask_s is not None:
        mask_s = (mask_s > 0.5).astype(np.float32)

    compare_panels = [
        (pet_resized, 'Raw PET', 'gray', (0, 1)),
        (p_l, f'P_l min-max (mean={mean_val:.3f})', 'hot', (0, 1)),
        (np.maximum(p_l - mean_val, 0), 'ReLU(P_l - mean)', 'inferno', None),
        (s_l, 'S_l saliency', 'magma', (0, 1)),
        (pet_resized, 'Raw + S_l overlay', 'gray', (0, 1)),
    ]
    for ax, (arr, title, cmap, vr) in zip(axes2, compare_panels):
        if title.startswith('Raw + S_l'):
            ax.imshow(arr, cmap='gray', vmin=0, vmax=1)
            ax.imshow(s_l, cmap='magma', alpha=0.55, vmin=0, vmax=1)
        elif vr is None:
            ax.imshow(arr, cmap=cmap, vmin=0, vmax=max(float(arr.max()), 1e-6))
        else:
            ax.imshow(arr, cmap=cmap, vmin=vr[0], vmax=vr[1])
        if mask_s is not None and arr.shape == mask_s.shape:
            ax.contour(mask_s, levels=[0.5], colors='lime', linewidths=0.8)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    fig2.suptitle(f'{paths["image_id"]} | PET saliency @512 (training input size)', fontsize=11)
    plt.tight_layout()
    plt.savefig(compare_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)

    return sample_stats


def main():
    parser = argparse.ArgumentParser(description='PET metabolic saliency visualization')
    parser.add_argument('--root', type=str, default='/root/autodl-tmp/data/PCLT20K')
    parser.add_argument('--out_dir', type=str,
                        default='/root/autodl-tmp/mkd-main/new-train/experiments/pet_saliency_vis')
    parser.add_argument('--image_ids', type=str, nargs='+',
                        default=['0487_001', '0487_037', '0441_040', '0139_008', '0126_058', '0540_041'])
    parser.add_argument('--eps', type=float, default=1e-6)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_stats = {}
    for image_id in args.image_ids:
        paths = resolve_paths(args.root, image_id)
        out_path = os.path.join(args.out_dir, f'{image_id}_pet_saliency.png')
        stats = save_sample_figure(paths, out_path, eps=args.eps)
        all_stats[image_id] = stats
        print(f'[saved] {out_path}')

    stats_path = os.path.join(args.out_dir, 'stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f'[saved] {stats_path}')


if __name__ == '__main__':
    main()
