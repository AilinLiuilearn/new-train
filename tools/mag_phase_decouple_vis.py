#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FFT 幅值-相位解耦可视化：分离 |F| 与 phase(F)，重建并对比 CT/PET 及跨模态交换效果。"""

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
DEFAULT_OUT = '/root/autodl-tmp/mkd-main/new-train/experiments/mag_phase_decouple_vis'


def imread_gray(path):
    return np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0


def normalize_display(arr, p_low=1, p_high=99):
    lo, hi = np.percentile(arr, [p_low, p_high])
    if hi - lo < 1e-8:
        return np.clip(arr, 0, 1)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def fft_decompose(img):
    """分解为幅值谱、相位谱，并重建仅幅值/仅相位分量。"""
    f_shift = np.fft.fftshift(np.fft.fft2(img))
    magnitude = np.abs(f_shift)
    phase = np.angle(f_shift)

    mag_only = np.real(np.fft.ifft2(np.fft.ifftshift(magnitude * np.exp(1j * 0.0))))
    phase_only = np.real(np.fft.ifft2(np.fft.ifftshift(np.exp(1j * phase))))

    return {
        'f_shift': f_shift,
        'magnitude': magnitude,
        'phase': phase,
        'mag_spectrum_log': np.log1p(magnitude),
        'mag_only': mag_only,
        'phase_only': phase_only,
    }


def reconstruct_from(magnitude, phase):
    """由给定幅值与相位重建空间域图像。"""
    f_shift = magnitude * np.exp(1j * phase)
    return np.real(np.fft.ifft2(np.fft.ifftshift(f_shift)))


def compare_to_original(recon, original):
    recon_n = normalize_display(recon)
    orig_n = normalize_display(original)
    mse = float(np.mean((recon_n - orig_n) ** 2))
    corr = float(np.corrcoef(recon_n.ravel(), orig_n.ravel())[0, 1])
    return {'mse': mse, 'corr': corr}


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


def select_samples_manual(root, image_ids):
    samples = []
    for image_id in image_ids:
        paths = resolve_paths(root, image_id)
        if not all(os.path.isfile(p) for p in [paths['ct_path'], paths['pet_path']]):
            raise FileNotFoundError(f'missing sample: {image_id}')
        samples.append(paths)
    return samples


def select_samples_with_lesion(root, list_path, num_samples=5, seed=42):
    with open(list_path, 'r') as f:
        ids = [x.strip() for x in f if x.strip()]
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    selected, used_cases = [], set()
    for image_id in ids:
        paths = resolve_paths(root, image_id)
        if not all(os.path.isfile(p) for p in [paths['ct_path'], paths['pet_path'], paths['mask_path']]):
            continue
        mask = imread_gray(paths['mask_path'])
        if mask.sum() < 50:
            continue
        if paths['case_id'] in used_cases and len(selected) < num_samples - 1:
            continue
        selected.append(paths)
        used_cases.add(paths['case_id'])
        if len(selected) >= num_samples:
            break
    return selected


def save_single_figure(sample, ct_d, pet_d, mask, out_path):
    ct_mag_pet_phase = reconstruct_from(pet_d['magnitude'], ct_d['phase'])
    pet_mag_ct_phase = reconstruct_from(ct_d['magnitude'], pet_d['phase'])

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle(
        f'Mag-Phase Decouple (FFT) | {sample["image_id"]}',
        fontsize=14, fontweight='bold',
    )

    col_titles = [
        'Original',
        'Magnitude |F| (log)',
        'Phase angle(F)',
        'Mag-only Recon\n(phase=0)',
        'Phase-only Recon\n(|F|=1)',
    ]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=10)

    # Row 1: CT
    axes[0, 0].imshow(sample['ct'], cmap='gray')
    axes[0, 1].imshow(ct_d['mag_spectrum_log'], cmap='magma')
    axes[0, 2].imshow(ct_d['phase'], cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0, 3].imshow(normalize_display(ct_d['mag_only']), cmap='gray')
    axes[0, 4].imshow(normalize_display(ct_d['phase_only']), cmap='gray')
    axes[0, 0].set_ylabel('CT', fontsize=12, fontweight='bold')

    # Row 2: PET
    axes[1, 0].imshow(sample['pet'], cmap='gray')
    axes[1, 1].imshow(pet_d['mag_spectrum_log'], cmap='magma')
    axes[1, 2].imshow(pet_d['phase'], cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, 3].imshow(normalize_display(pet_d['mag_only']), cmap='gray')
    axes[1, 4].imshow(normalize_display(pet_d['phase_only']), cmap='gray')
    axes[1, 0].set_ylabel('PET', fontsize=12, fontweight='bold')

    for ax in axes[:2].flat:
        ax.axis('off')

    # Row 3: cross-modal swap
    axes[2, 0].imshow(mask, cmap='gray')
    axes[2, 0].set_title('Lesion Mask')
    axes[2, 0].axis('off')

    axes[2, 1].imshow(normalize_display(ct_mag_pet_phase), cmap='gray')
    axes[2, 1].set_title('CT phase + PET magnitude\n(PET energy, CT structure)')
    axes[2, 1].axis('off')

    axes[2, 2].imshow(normalize_display(pet_mag_ct_phase), cmap='gray')
    axes[2, 2].set_title('PET phase + CT magnitude\n(CT energy, PET structure)')
    axes[2, 2].axis('off')

    # verify: original = mag * exp(j*phase)
    ct_verify = reconstruct_from(ct_d['magnitude'], ct_d['phase'])
    pet_verify = reconstruct_from(pet_d['magnitude'], pet_d['phase'])
    axes[2, 3].imshow(normalize_display(ct_verify), cmap='gray')
    axes[2, 3].set_title('CT verify\n|F|+phase -> original')
    axes[2, 3].axis('off')
    axes[2, 4].imshow(normalize_display(pet_verify), cmap='gray')
    axes[2, 4].set_title('PET verify\n|F|+phase -> original')
    axes[2, 4].axis('off')

    # Row 4: similarity bars
    ct_mag_cmp = compare_to_original(ct_d['mag_only'], sample['ct'])
    ct_pha_cmp = compare_to_original(ct_d['phase_only'], sample['ct'])
    pet_mag_cmp = compare_to_original(pet_d['mag_only'], sample['pet'])
    pet_pha_cmp = compare_to_original(pet_d['phase_only'], sample['pet'])

    labels = ['CT-Mag', 'CT-Phase', 'PET-Mag', 'PET-Phase']
    corrs = [ct_mag_cmp['corr'], ct_pha_cmp['corr'], pet_mag_cmp['corr'], pet_pha_cmp['corr']]
    colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78']
    axes[3, 0].bar(labels, corrs, color=colors)
    axes[3, 0].set_ylim(-0.2, 1.05)
    axes[3, 0].set_title('Corr vs Original\n(higher = closer to original)')
    axes[3, 0].tick_params(axis='x', rotation=20, labelsize=8)
    for i, v in enumerate(corrs):
        axes[3, 0].text(i, v + 0.03, f'{v:.2f}', ha='center', fontsize=8)

    axes[3, 1].text(
        0.05, 0.95,
        'How to read:\n'
        'F(u,v) = |F| * exp(j*phase)\n\n'
        'Mag-only: keep |F|, set phase=0\n'
        '  -> energy distribution, blurry\n\n'
        'Phase-only: keep phase, |F|=1\n'
        '  -> structure/edges, weak contrast\n\n'
        'Phase often carries structure;\n'
        'magnitude carries energy/contrast.',
        transform=axes[3, 1].transAxes,
        va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4),
    )
    axes[3, 1].axis('off')

    axes[3, 2].text(
        0.05, 0.95,
        'Cross-modal swap:\n\n'
        'CT phase + PET mag:\n'
        '  PET uptake pattern with\n'
        '  CT-like edge layout\n\n'
        'PET phase + CT mag:\n'
        '  CT contrast with\n'
        '  PET-like structure',
        transform=axes[3, 2].transAxes,
        va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.4),
    )
    axes[3, 2].axis('off')

    axes[3, 3].axis('off')
    axes[3, 4].axis('off')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        'ct_mag_corr': ct_mag_cmp['corr'],
        'ct_phase_corr': ct_pha_cmp['corr'],
        'pet_mag_corr': pet_mag_cmp['corr'],
        'pet_phase_corr': pet_pha_cmp['corr'],
    }


def save_summary(all_results, out_path):
    n = len(all_results)
    fig, axes = plt.subplots(n, 8, figsize=(22, 2.8 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    titles = ['CT', 'CT-MagR', 'CT-PhaR', 'PET', 'PET-MagR', 'PET-PhaR', 'CTph+PETmag', 'PETph+CTmag']
    fig.suptitle('Mag-Phase Decouple Overview', fontsize=14, fontweight='bold')

    for i, item in enumerate(all_results):
        ct_d, pet_d = item['ct_d'], item['pet_d']
        ct_ph_pet_mag = reconstruct_from(pet_d['magnitude'], ct_d['phase'])
        pet_ph_ct_mag = reconstruct_from(ct_d['magnitude'], pet_d['phase'])
        imgs = [
            item['ct'],
            normalize_display(ct_d['mag_only']),
            normalize_display(ct_d['phase_only']),
            item['pet'],
            normalize_display(pet_d['mag_only']),
            normalize_display(pet_d['phase_only']),
            normalize_display(ct_ph_pet_mag),
            normalize_display(pet_ph_ct_mag),
        ]
        for j, img in enumerate(imgs):
            axes[i, j].imshow(img, cmap='gray')
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=9)
            axes[i, j].set_ylabel(item['image_id'], fontsize=7)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='FFT magnitude-phase decoupling visualization')
    parser.add_argument('--root', default=DEFAULT_ROOT)
    parser.add_argument('--out_dir', default=DEFAULT_OUT)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--split', default='val.txt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image_ids', nargs='*', default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.image_ids:
        samples = select_samples_manual(args.root, args.image_ids)
    else:
        list_path = os.path.join(args.root, args.split)
        samples = select_samples_with_lesion(args.root, list_path, args.num_samples, args.seed)

    if not samples:
        print('[error] no valid samples', file=sys.stderr)
        sys.exit(1)

    all_results, meta = [], []
    for idx, sample in enumerate(samples):
        ct = imread_gray(sample['ct_path'])
        pet = imread_gray(sample['pet_path'])
        mask = imread_gray(sample['mask_path']) if os.path.isfile(sample.get('mask_path', '')) else np.zeros_like(ct)

        ct_d = fft_decompose(ct)
        pet_d = fft_decompose(pet)

        item = {**sample, 'ct': ct, 'pet': pet, 'ct_d': ct_d, 'pet_d': pet_d}
        all_results.append(item)

        out_name = f'{idx:02d}_{sample["image_id"]}_mag_phase.png'
        out_path = os.path.join(args.out_dir, out_name)
        stats = save_single_figure(item, ct_d, pet_d, mask, out_path)
        print(f'[saved] {out_path}')

        meta.append({
            'index': idx,
            'image_id': sample['image_id'],
            'figure': out_name,
            **stats,
        })

    summary_path = os.path.join(args.out_dir, 'summary_all_samples.png')
    save_summary(all_results, summary_path)
    print(f'[saved] {summary_path}')

    meta_path = os.path.join(args.out_dir, 'selected_samples.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({'samples': meta}, f, indent=2, ensure_ascii=False)
    print(f'[saved] {meta_path}')
    print(f'\nDone -> {args.out_dir}')


if __name__ == '__main__':
    main()
