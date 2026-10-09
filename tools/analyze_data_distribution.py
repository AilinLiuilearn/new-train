# -*- coding: utf-8 -*-
"""CT/PET lesion visibility and distribution analysis for PCLT20K.

Samples slices from train_original/test splits and reports, per modality:
  - global intensity histogram stats;
  - lesion-inside vs background intensity (mean/std, contrast-to-noise);
  - lesion size distribution;
  - CT-PET pixel correlation inside vs outside lesions;
  - fraction of lesions brighter than background in each modality.

Guides fusion design: which modality carries the lesion signal, how small
lesions are, and whether simple intensity cues suffice.
"""
import argparse
import json
import os
import random

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_triplet(root, case, idx):
    base = os.path.join(root, case, f'{case}_{idx:03d}')
    ct = np.array(Image.open(base + '_CT.png')).astype(np.float32)
    pet = np.array(Image.open(base + '_PET.png')).astype(np.float32)
    mask = np.array(Image.open(base + '_mask.png')) > 127
    return ct, pet, mask


def list_samples(root, split_file):
    samples = []
    with open(os.path.join(root, split_file)) as f:
        for line in f:
            line = line.strip()
            if line:
                case, idx = line.split('_')
                samples.append((case, int(idx)))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/root/autodl-tmp/data/PCLT20K')
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='data_analysis')
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    per_split = {}
    all_rows = []
    all_keys = []
    for split in ('train_original.txt', 'test.txt'):
        samples = list_samples(args.root, split)
        sel = rng.sample(samples, min(args.n, len(samples)))
        rows = []
        for case, idx in sel:
            ct, pet, mask = load_triplet(args.root, case, idx)
            bg = ~mask
            area = float(mask.mean())
            ct_in, ct_bg = float(ct[mask].mean()) if mask.any() else np.nan, float(ct[bg].mean())
            pet_in, pet_bg = float(pet[mask].mean()) if mask.any() else np.nan, float(pet[bg].mean())
            ct_std = float(ct[bg].std())
            pet_std = float(pet[bg].std())
            cnr_ct = (ct_in - ct_bg) / (ct_std + 1e-6)
            cnr_pet = (pet_in - pet_bg) / (pet_std + 1e-6)
            if mask.sum() > 10:
                cc = float(np.corrcoef(ct[mask].ravel(), pet[mask].ravel())[0, 1])
                cb = float(np.corrcoef(ct[bg].ravel()[::97], pet[bg].ravel()[::97])[0, 1])
            else:
                cc, cb = np.nan, np.nan
            rows.append(dict(area=area, ct_in=ct_in, ct_bg=ct_bg, pet_in=pet_in,
                             pet_bg=pet_bg, cnr_ct=cnr_ct, cnr_pet=cnr_pet,
                             corr_in=cc, corr_bg=cb,
                             pet_hotter=float(pet_in > pet_bg),
                             ct_hotter=float(ct_in > ct_bg)))
            all_keys.append((case, idx))
        per_split[split] = rows
        all_rows.extend(rows)

    def agg(key):
        v = np.array([r[key] for r in all_rows if np.isfinite(r[key])])
        return dict(n=int(v.size), mean=float(v.mean()), std=float(v.std()),
                    p5=float(np.percentile(v, 5)), p50=float(np.percentile(v, 50)),
                    p95=float(np.percentile(v, 95)))

    report = {
        'n_slices': len(all_rows),
        'lesion_area_ratio': agg('area'),
        'ct_inside': agg('ct_in'),
        'ct_bg': agg('ct_bg'),
        'pet_inside': agg('pet_in'),
        'pet_bg': agg('pet_bg'),
        'cnr_ct': agg('cnr_ct'),
        'cnr_pet': agg('cnr_pet'),
        'corr_inside': agg('corr_in'),
        'corr_bg': agg('corr_bg'),
        'frac_pet_hotter': float(np.mean([r['pet_hotter'] for r in all_rows])),
        'frac_ct_hotter': float(np.mean([r['ct_hotter'] for r in all_rows])),
        'frac_area_lt_01pct': float(np.mean([r['area'] < 0.001 for r in all_rows])),
    }
    with open(os.path.join(args.out, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].hist([r['ct_in'] for r in all_rows if np.isfinite(r['ct_in'])],
                     bins=60, alpha=0.6, label='lesion')
    axes[0, 0].hist([r['ct_bg'] for r in all_rows], bins=60, alpha=0.6, label='bg')
    axes[0, 0].set_title('CT intensity: lesion vs bg (per-slice mean)')
    axes[0, 0].legend()
    axes[0, 1].hist([r['pet_in'] for r in all_rows if np.isfinite(r['pet_in'])],
                     bins=60, alpha=0.6, label='lesion')
    axes[0, 1].hist([r['pet_bg'] for r in all_rows], bins=60, alpha=0.6, label='bg')
    axes[0, 1].set_title('PET intensity: lesion vs bg (per-slice mean)')
    axes[0, 1].legend()
    axes[0, 2].hist([r['cnr_ct'] for r in all_rows if np.isfinite(r['cnr_ct'])],
                     bins=60, alpha=0.6, label='CT CNR')
    axes[0, 2].hist([r['cnr_pet'] for r in all_rows if np.isfinite(r['cnr_pet'])],
                     bins=60, alpha=0.6, label='PET CNR')
    axes[0, 2].set_title('Contrast-to-noise ratio')
    axes[0, 2].legend()
    axes[1, 0].hist([r['area'] * 100 for r in all_rows], bins=60)
    axes[1, 0].set_title('Lesion area (% of image)')
    axes[1, 0].set_xlabel('%')
    axes[1, 1].hist([r['corr_in'] for r in all_rows if np.isfinite(r['corr_in'])],
                     bins=60, alpha=0.6, label='inside')
    axes[1, 1].hist([r['corr_bg'] for r in all_rows if np.isfinite(r['corr_bg'])],
                     bins=60, alpha=0.6, label='bg')
    axes[1, 1].set_title('CT-PET correlation')
    axes[1, 1].legend()
    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5, 'see example_triplets.png', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'distributions.png'), dpi=100)
    # Example triplets: top-2 and bottom-2 PET CNR slices.
    order = sorted(range(len(all_rows)),
                   key=lambda i: all_rows[i]['cnr_pet'] if np.isfinite(all_rows[i]['cnr_pet']) else -99)
    fig2, ax2 = plt.subplots(2, 4, figsize=(16, 8))
    for j, i in enumerate(list(order[-2:]) + list(order[:2])):
        case, idx = all_keys[i]
        ct, pet, mask = load_triplet(args.root, case, idx)
        ax2[0, j].imshow(ct, cmap='gray')
        ax2[0, j].set_title(f'{case}_{idx:03d} CT (PET-CNR={all_rows[i]["cnr_pet"]:.1f})')
        ax2[0, j].axis('off')
        ax2[1, j].imshow(pet, cmap='hot')
        ax2[1, j].contour(mask, colors='cyan', linewidths=0.8)
        ax2[1, j].set_title('PET + mask')
        ax2[1, j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'example_triplets.png'), dpi=100)
    print('saved', os.path.abspath(args.out))


if __name__ == '__main__':
    main()
