#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize frozen DINOv3 PET dense features for A1 diagnosis.

Outputs per sample:
  - input PET / CT / mask
  - DINO feature mean, L2-norm, PCA-RGB maps (upsampled to image size)
  - lesion vs background feature statistics
  - optional comparison: CIPA norm vs ImageNet norm
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.pet_prompted_ct_decoder import FrozenDINOv3PETEncoder

plt.rcParams['axes.unicode_minus'] = False

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def imread_gray(path):
    return np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0


def resize_gray(img, size):
    return np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)) / 255.0


def normalize_cipa_rgb(ch):
    # ch: [1, H, W] -> [3, H, W]
    rgb = np.repeat(ch, 3, axis=0)
    return rgb * 3.2 - 1.6


def normalize_imagenet_rgb(ch):
    rgb = np.repeat(ch, 3, axis=0)
    return (rgb - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]


def resolve_paths(root, image_id):
    case_id, slice_id = image_id.split('_', 1)
    case_dir = os.path.join(root, case_id)
    base = f'{case_id}_{slice_id}'
    return {
        'image_id': image_id,
        'ct_path': os.path.join(case_dir, f'{base}_CT.png'),
        'pet_path': os.path.join(case_dir, f'{base}_PET.png'),
        'mask_path': os.path.join(case_dir, f'{base}_mask.png'),
    }


def select_samples(root, list_path, num_samples=6, seed=42):
    with open(list_path, 'r') as f:
        ids = [x.strip() for x in f if x.strip()]
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    selected = []
    used_cases = set()
    for image_id in ids:
        paths = resolve_paths(root, image_id)
        if not all(os.path.isfile(p) for p in [paths['ct_path'], paths['pet_path'], paths['mask_path']]):
            continue
        mask = imread_gray(paths['mask_path'])
        if mask.sum() < 50:
            continue
        case_id = image_id.split('_', 1)[0]
        if case_id in used_cases and len(selected) < num_samples - 1:
            continue
        selected.append(paths)
        used_cases.add(case_id)
        if len(selected) >= num_samples:
            break
    return selected


def normalize_display(arr, p_low=2, p_high=98):
    lo, hi = np.percentile(arr, [p_low, p_high])
    if hi - lo < 1e-8:
        return np.clip(arr, 0, 1)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def pca_rgb(feat_hwc):
    """feat_hwc: [H,W,C] -> RGB [H,W,3] via PCA."""
    h, w, c = feat_hwc.shape
    x = feat_hwc.reshape(-1, c).astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    pcs = x @ vt[:3].T
    pcs = pcs.reshape(h, w, 3)
    for i in range(3):
        pcs[..., i] = normalize_display(pcs[..., i])
    return pcs


@torch.no_grad()
def extract_dino_features(encoder, pet_rgb_tensor):
    feat = encoder(pet_rgb_tensor)
    return feat[0].detach().float().cpu().numpy()


def feature_maps(feat_chw):
    mean_map = feat_chw.mean(axis=0)
    l2_map = np.linalg.norm(feat_chw, axis=0)
    pca_map = pca_rgb(np.transpose(feat_chw, (1, 2, 0)))
    return mean_map, l2_map, pca_map


def lesion_stats(map2d, mask, up_size=512):
    mask_up = np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).resize((up_size, up_size), Image.NEAREST)
    ) > 127
    map_up = np.array(
        Image.fromarray(normalize_display(map2d)).resize((up_size, up_size), Image.BILINEAR)
    )
    fg = map_up[mask_up]
    bg = map_up[~mask_up]
    return {
        'fg_mean': float(fg.mean()) if fg.size else 0.0,
        'bg_mean': float(bg.mean()) if bg.size else 0.0,
        'fg_std': float(fg.std()) if fg.size else 0.0,
        'bg_std': float(bg.std()) if bg.size else 0.0,
        'fg_minus_bg': float(fg.mean() - bg.mean()) if fg.size and bg.size else 0.0,
    }


def upsample_map(map2d, size):
    t = torch.from_numpy(map2d[None, None].astype(np.float32))
    up = F.interpolate(t, size=(size, size), mode='bilinear', align_corners=False)[0, 0].numpy()
    return up


def render_sample(paths, encoder, image_size=512, out_path=None, norm_mode='cipa'):
    ct = resize_gray(imread_gray(paths['ct_path']), image_size)
    pet = resize_gray(imread_gray(paths['pet_path']), image_size)
    mask = resize_gray(imread_gray(paths['mask_path']), image_size)
    mask_bin = (mask >= 0.5).astype(np.float32)

    pet_ch = pet[None, ...]
    if norm_mode == 'cipa':
        pet_rgb = normalize_cipa_rgb(pet_ch)
    else:
        pet_rgb = normalize_imagenet_rgb(pet_ch)

    pet_tensor = torch.from_numpy(pet_rgb[None]).float()
    if next(encoder.parameters()).is_cuda:
        pet_tensor = pet_tensor.cuda()

    feat = extract_dino_features(encoder, pet_tensor)
    mean_map, l2_map, pca_map = feature_maps(feat)
    mean_up = upsample_map(mean_map, image_size)
    l2_up = upsample_map(l2_map, image_size)

    stats = {
        'image_id': paths['image_id'],
        'norm_mode': norm_mode,
        'feature_shape': list(feat.shape),
        'feature_global_mean': float(feat.mean()),
        'feature_global_std': float(feat.std()),
        'mean_map': lesion_stats(mean_map, mask_bin, image_size),
        'l2_map': lesion_stats(l2_map, mask_bin, image_size),
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"DINOv3 PET features | {paths['image_id']} | norm={norm_mode}", fontsize=12)

    axes[0, 0].imshow(pet, cmap='gray')
    axes[0, 0].set_title('PET input')
    axes[0, 1].imshow(ct, cmap='gray')
    axes[0, 1].set_title('CT')
    axes[0, 2].imshow(mask_bin, cmap='gray')
    axes[0, 2].set_title('Mask')

    overlay = np.stack([pet, pet, pet], axis=-1)
    overlay[..., 0] = np.clip(overlay[..., 0] + mask_bin * 0.5, 0, 1)
    axes[0, 3].imshow(overlay)
    axes[0, 3].set_title('PET + mask (red)')

    im1 = axes[1, 0].imshow(normalize_display(mean_up), cmap='magma')
    axes[1, 0].set_title(f'mean feat (fg-bg={stats["mean_map"]["fg_minus_bg"]:.3f})')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

    im2 = axes[1, 1].imshow(normalize_display(l2_up), cmap='viridis')
    axes[1, 1].set_title(f'L2 norm (fg-bg={stats["l2_map"]["fg_minus_bg"]:.3f})')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

    pca_up = np.array(Image.fromarray((pca_map * 255).astype(np.uint8)).resize((image_size, image_size), Image.BILINEAR)) / 255.0
    axes[1, 2].imshow(pca_up)
    axes[1, 2].set_title('PCA-RGB (upsampled)')

    contour = axes[1, 3].imshow(normalize_display(l2_up), cmap='gray')
    axes[1, 3].contour(mask_bin, levels=[0.5], colors='r', linewidths=1.0)
    axes[1, 3].set_title('L2 norm + mask contour')

    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return stats


def render_norm_compare(paths, encoder, image_size=512, out_path=None):
    ct = resize_gray(imread_gray(paths['ct_path']), image_size)
    pet = resize_gray(imread_gray(paths['pet_path']), image_size)
    mask = resize_gray(imread_gray(paths['mask_path']), image_size)
    mask_bin = (mask >= 0.5).astype(np.float32)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f'Norm comparison | {paths["image_id"]}', fontsize=12)

    for row, (norm_mode, title) in enumerate([('cipa', 'CIPA (training)'), ('imagenet', 'ImageNet (DINO default)')]):
        pet_ch = pet[None, ...]
        pet_rgb = normalize_cipa_rgb(pet_ch) if norm_mode == 'cipa' else normalize_imagenet_rgb(pet_ch)
        pet_tensor = torch.from_numpy(pet_rgb[None]).float()
        if next(encoder.parameters()).is_cuda:
            pet_tensor = pet_tensor.cuda()
        feat = extract_dino_features(encoder, pet_tensor)
        _, l2_map, _ = feature_maps(feat)
        l2_up = upsample_map(l2_map, image_size)
        stats = lesion_stats(l2_map, mask_bin, image_size)

        axes[row, 0].imshow(pet, cmap='gray')
        axes[row, 0].set_title('PET')
        im = axes[row, 1].imshow(normalize_display(l2_up), cmap='viridis')
        axes[row, 1].set_title(f'{title}\nL2 fg-bg={stats["fg_minus_bg"]:.3f}')
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046)
        axes[row, 2].imshow(normalize_display(l2_up), cmap='gray')
        axes[row, 2].contour(mask_bin, levels=[0.5], colors='r', linewidths=1.0)
        axes[row, 2].set_title('L2 + mask contour')

    for ax in axes.ravel():
        ax.axis('off')
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize DINOv3 PET dense features')
    parser.add_argument('--root', type=str, default='/root/autodl-tmp/data/PCLT20K')
    parser.add_argument('--list', type=str, default='test.txt')
    parser.add_argument('--num_samples', type=int, default=6)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dinov3_model_name', type=str, default='vit_small_patch16_dinov3')
    parser.add_argument('--dinov3_pretrained_path', type=str,
                        default='/root/autodl-tmp/mkd-main/new-train/pretrained/dinov3_small')
    parser.add_argument('--out_dir', type=str,
                        default='/root/autodl-tmp/mkd-main/new-train/experiments/dinov3_pet_feature_vis')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--norm_mode', type=str, default='cipa', choices=('cipa', 'imagenet'))
    parser.add_argument('--compare_norms', action='store_true', default=True)
    parser.add_argument('--image_ids', type=str, nargs='*', default=None,
                        help='Optional explicit image ids, e.g. 0487_037')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.image_ids:
        samples = []
        for image_id in args.image_ids:
            paths = resolve_paths(args.root, image_id)
            if all(os.path.isfile(p) for p in [paths['ct_path'], paths['pet_path'], paths['mask_path']]):
                samples.append(paths)
    else:
        list_path = args.list if os.path.isabs(args.list) else os.path.join(args.root, args.list)
        samples = select_samples(args.root, list_path, args.num_samples, args.seed)

    if not samples:
        raise RuntimeError('No valid samples found for visualization.')

    encoder = FrozenDINOv3PETEncoder(
        model_name=args.dinov3_model_name,
        pretrained_path=args.dinov3_pretrained_path,
    )
    if torch.cuda.is_available():
        encoder = encoder.cuda()
    encoder.eval()

    all_stats = []
    for idx, paths in enumerate(samples):
        out_path = os.path.join(args.out_dir, f'{idx:02d}_{paths["image_id"]}_dinov3_pet.png')
        stats = render_sample(paths, encoder, args.image_size, out_path, norm_mode=args.norm_mode)
        all_stats.append(stats)
        print(f'[saved] {out_path}')

        if args.compare_norms:
            cmp_path = os.path.join(args.out_dir, f'{idx:02d}_{paths["image_id"]}_norm_compare.png')
            render_norm_compare(paths, encoder, args.image_size, cmp_path)
            print(f'[saved] {cmp_path}')

    summary = {
        'num_samples': len(all_stats),
        'norm_mode': args.norm_mode,
        'dinov3_model_name': args.dinov3_model_name,
        'samples': all_stats,
        'avg_fg_minus_bg_mean': float(np.mean([s['mean_map']['fg_minus_bg'] for s in all_stats])),
        'avg_fg_minus_bg_l2': float(np.mean([s['l2_map']['fg_minus_bg'] for s in all_stats])),
    }
    summary_path = os.path.join(args.out_dir, 'feature_stats.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[saved] {summary_path}')
    print(f'avg lesion-background contrast (mean map): {summary["avg_fg_minus_bg_mean"]:.4f}')
    print(f'avg lesion-background contrast (L2 map):   {summary["avg_fg_minus_bg_l2"]:.4f}')


if __name__ == '__main__':
    main()
