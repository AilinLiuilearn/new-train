#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""频域解耦可视化：分别提取 CT/PET 的低频与高频成分，便于观察两模态在频域下的差异。"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

plt.rcParams['axes.unicode_minus'] = False

DEFAULT_ROOT = '/root/autodl-tmp/data/PCLT20K'
DEFAULT_OUT = '/root/autodl-tmp/mkd-main/new-train/experiments/freq_decouple_vis'


def imread_gray(path):
    img = np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0
    return img


def make_gaussian_masks(h, w, cutoff_ratio=0.15, sigma_scale=1.0):
    """以图像中心为原点的二维高斯低通掩膜；高通 = 1 - 低通。"""
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = cutoff_ratio * min(h, w) / 2.0
    sigma = max(radius * sigma_scale, 1e-6)
    low_mask = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
    high_mask = 1.0 - low_mask
    return low_mask.astype(np.float32), high_mask.astype(np.float32)


def freq_decouple(img, cutoff_ratio=0.15, sigma_scale=1.0):
    """2D FFT 频域解耦，返回低频/高频重建图及幅度谱。"""
    h, w = img.shape
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(f_shift))

    low_mask, high_mask = make_gaussian_masks(h, w, cutoff_ratio, sigma_scale)
    low_fft = f_shift * low_mask
    high_fft = f_shift * high_mask

    low_spatial = np.real(np.fft.ifft2(np.fft.ifftshift(low_fft)))
    high_spatial = np.real(np.fft.ifft2(np.fft.ifftshift(high_fft)))

    low_energy = float(np.sum(low_mask * np.abs(f_shift) ** 2))
    high_energy = float(np.sum(high_mask * np.abs(f_shift) ** 2))
    total = low_energy + high_energy + 1e-12
    return {
        'low': low_spatial,
        'high': high_spatial,
        'magnitude': magnitude,
        'low_mask': low_mask,
        'high_mask': high_mask,
        'low_energy_ratio': low_energy / total,
        'high_energy_ratio': high_energy / total,
    }


def radial_profile(magnitude, num_bins=64):
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = min(cy, cx)
    bins = np.linspace(0, max_r, num_bins + 1)
    profile = []
    for i in range(num_bins):
        mask = (dist >= bins[i]) & (dist < bins[i + 1])
        profile.append(float(magnitude[mask].mean()) if mask.any() else 0.0)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers / max_r, np.array(profile)


def normalize_display(arr, p_low=1, p_high=99):
    lo, hi = np.percentile(arr, [p_low, p_high])
    if hi - lo < 1e-8:
        return np.clip(arr, 0, 1)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def resolve_paths(root, image_id):
    case_id, slice_id = image_id.split('_', 1)
    case_dir = os.path.join(root, case_id)
    base = f'{case_id}_{slice_id}'
    return {
        'image_id': image_id,
        'case_id': case_id,
        'slice_id': slice_id,
        'ct_path': os.path.join(case_dir, f'{base}_CT.png'),
        'pet_path': os.path.join(case_dir, f'{base}_PET.png'),
        'mask_path': os.path.join(case_dir, f'{base}_mask.png'),
    }


def select_samples_with_lesion(root, list_path, num_samples=5, seed=42):
    """从划分文件中选取含病灶、且 CT/PET 均存在的样本，尽量覆盖不同 case。"""
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
        if paths['case_id'] in used_cases and len(selected) < num_samples - 1:
            continue
        selected.append({**paths, 'lesion_pixels': int((mask > 0.5).sum())})
        used_cases.add(paths['case_id'])
        if len(selected) >= num_samples:
            break
    return selected


def select_samples_manual(root, image_ids):
    samples = []
    for image_id in image_ids:
        paths = resolve_paths(root, image_id)
        if not all(os.path.isfile(p) for p in [paths['ct_path'], paths['pet_path']]):
            raise FileNotFoundError(f'样本缺失: {image_id}')
        samples.append(paths)
    return samples


def save_single_sample_figure(sample, ct_result, pet_result, mask, out_path, cutoff_ratio):
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(
        f"Freq Decouple | {sample['image_id']} | cutoff={cutoff_ratio:.2f}",
        fontsize=14, fontweight='bold',
    )

    ct_low_vis = normalize_display(ct_result['low'])
    ct_high_vis = normalize_display(ct_result['high'])
    pet_low_vis = normalize_display(pet_result['low'])
    pet_high_vis = normalize_display(pet_result['high'])

    row0 = [
        ('CT Original', sample['ct'], 'gray'),
        ('CT Low-freq', ct_low_vis, 'gray'),
        ('CT High-freq', ct_high_vis, 'gray'),
        ('CT Spectrum (log)', ct_result['magnitude'], 'magma'),
    ]
    row1 = [
        ('PET Original', sample['pet'], 'gray'),
        ('PET Low-freq', pet_low_vis, 'gray'),
        ('PET High-freq', pet_high_vis, 'gray'),
        ('PET Spectrum (log)', pet_result['magnitude'], 'magma'),
    ]

    for col, (title, img, cmap) in enumerate(row0):
        axes[0, col].imshow(img, cmap=cmap)
        axes[0, col].set_title(title)
        axes[0, col].axis('off')

    for col, (title, img, cmap) in enumerate(row1):
        axes[1, col].imshow(img, cmap=cmap)
        axes[1, col].set_title(title)
        axes[1, col].axis('off')

    # 第三行：掩膜 + 频域掩膜 + 径向谱对比 + 能量占比
    axes[2, 0].imshow(mask, cmap='gray')
    axes[2, 0].set_title('Lesion Mask')
    axes[2, 0].axis('off')

    overlay = np.zeros((*ct_result['low_mask'].shape, 3))
    overlay[..., 1] = ct_result['low_mask']
    overlay[..., 0] = ct_result['high_mask'] * 0.6
    axes[2, 1].imshow(overlay)
    axes[2, 1].set_title('Freq Mask (green=LP, red=HP)')
    axes[2, 1].axis('off')

    r_ct, p_ct = radial_profile(ct_result['magnitude'])
    r_pet, p_pet = radial_profile(pet_result['magnitude'])
    axes[2, 2].plot(r_ct, p_ct, label='CT', color='#1f77b4', lw=2)
    axes[2, 2].plot(r_pet, p_pet, label='PET', color='#ff7f0e', lw=2)
    axes[2, 2].axvline(cutoff_ratio, color='red', ls='--', alpha=0.7, label=f'cutoff={cutoff_ratio}')
    axes[2, 2].set_xlabel('Normalized Radius')
    axes[2, 2].set_ylabel('Mean log Magnitude')
    axes[2, 2].set_title('Radial Spectrum')
    axes[2, 2].legend(fontsize=8)
    axes[2, 2].grid(alpha=0.3)

    labels = ['CT-L', 'CT-H', 'PET-L', 'PET-H']
    ratios = [
        ct_result['low_energy_ratio'], ct_result['high_energy_ratio'],
        pet_result['low_energy_ratio'], pet_result['high_energy_ratio'],
    ]
    colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78']
    axes[2, 3].bar(labels, ratios, color=colors)
    axes[2, 3].set_ylim(0, 1)
    axes[2, 3].set_title('Energy Ratio')
    for i, v in enumerate(ratios):
        axes[2, 3].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_summary_figure(all_results, out_path, cutoff_ratio):
    n = len(all_results)
    fig, axes = plt.subplots(n, 6, figsize=(18, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f'Freq Decouple Overview (cutoff={cutoff_ratio})', fontsize=14, fontweight='bold')
    col_titles = ['CT', 'CT-L', 'CT-H', 'PET', 'PET-L', 'PET-H']

    for i, item in enumerate(all_results):
        ct, pet = item['ct'], item['pet']
        ct_r, pet_r = item['ct_result'], item['pet_result']
        row_imgs = [
            ct, normalize_display(ct_r['low']), normalize_display(ct_r['high']),
            pet, normalize_display(pet_r['low']), normalize_display(pet_r['high']),
        ]
        for j, img in enumerate(row_imgs):
            axes[i, j].imshow(img, cmap='gray')
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=10)
            axes[i, j].set_ylabel(item['image_id'], fontsize=8)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_spectrum_compare(all_results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for item in all_results:
        ct_r, pet_r = item['ct_result'], item['pet_result']
        r_ct, p_ct = radial_profile(ct_r['magnitude'])
        r_pet, p_pet = radial_profile(pet_r['magnitude'])
        axes[0].plot(r_ct, p_ct, alpha=0.8, label=item['image_id'])
        axes[1].plot(r_pet, p_pet, alpha=0.8, label=item['image_id'])

    axes[0].set_title('CT Radial Spectrum (5 samples)')
    axes[1].set_title('PET Radial Spectrum (5 samples)')
    for ax in axes:
        ax.set_xlabel('Normalized Radius')
        ax.set_ylabel('Mean log Magnitude')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='CT/PET 频域解耦可视化')
    parser.add_argument('--root', default=DEFAULT_ROOT)
    parser.add_argument('--out_dir', default=DEFAULT_OUT)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--cutoff_ratio', type=float, default=0.12,
                        help='低通截止半径占 min(H,W)/2 的比例')
    parser.add_argument('--sigma_scale', type=float, default=1.0)
    parser.add_argument('--split', default='val.txt', help='用于自动选样的划分文件')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image_ids', nargs='*', default=None,
                        help='手动指定 image_id，如 0201_015 0126_045')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.image_ids:
        samples = select_samples_manual(args.root, args.image_ids)
    else:
        list_path = os.path.join(args.root, args.split)
        samples = select_samples_with_lesion(
            args.root, list_path, num_samples=args.num_samples, seed=args.seed,
        )
        if len(samples) < args.num_samples:
            print(f'[warn] 仅找到 {len(samples)} 个有效样本', file=sys.stderr)

    if not samples:
        print('[error] 未找到可用样本，请检查数据路径', file=sys.stderr)
        sys.exit(1)

    all_results = []
    meta = []

    for idx, sample in enumerate(samples):
        ct = imread_gray(sample['ct_path'])
        pet = imread_gray(sample['pet_path'])
        mask = imread_gray(sample['mask_path']) if os.path.isfile(sample.get('mask_path', '')) else np.zeros_like(ct)

        ct_result = freq_decouple(ct, args.cutoff_ratio, args.sigma_scale)
        pet_result = freq_decouple(pet, args.cutoff_ratio, args.sigma_scale)

        sample_data = {
            **sample,
            'ct': ct,
            'pet': pet,
            'ct_result': ct_result,
            'pet_result': pet_result,
        }
        all_results.append(sample_data)

        out_name = f'{idx:02d}_{sample["image_id"]}_freq_decouple.png'
        out_path = os.path.join(args.out_dir, out_name)
        save_single_sample_figure(sample_data, ct_result, pet_result, mask, out_path, args.cutoff_ratio)
        print(f'[saved] {out_path}')

        meta.append({
            'index': idx,
            'image_id': sample['image_id'],
            'case_id': sample['case_id'],
            'slice_id': sample['slice_id'],
            'ct_low_energy': ct_result['low_energy_ratio'],
            'ct_high_energy': ct_result['high_energy_ratio'],
            'pet_low_energy': pet_result['low_energy_ratio'],
            'pet_high_energy': pet_result['high_energy_ratio'],
            'lesion_pixels': sample.get('lesion_pixels'),
            'figure': out_name,
        })

    summary_path = os.path.join(args.out_dir, 'summary_all_samples.png')
    save_summary_figure(all_results, summary_path, args.cutoff_ratio)
    print(f'[saved] {summary_path}')

    spectrum_path = os.path.join(args.out_dir, 'spectrum_radial_compare.png')
    save_spectrum_compare(all_results, spectrum_path)
    print(f'[saved] {spectrum_path}')

    meta_path = os.path.join(args.out_dir, 'selected_samples.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'cutoff_ratio': args.cutoff_ratio,
            'sigma_scale': args.sigma_scale,
            'samples': meta,
        }, f, indent=2, ensure_ascii=False)
    print(f'[saved] {meta_path}')
    print(f'\n完成：共处理 {len(samples)} 张图，输出目录 -> {args.out_dir}')


if __name__ == '__main__':
    main()
