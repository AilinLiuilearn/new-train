# -*- coding: utf-8 -*-
"""Analyze bad samples for MDT segmentation checkpoints."""

import argparse
import csv
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from utils.metrics_seg import compute_hd95_pair


def _load_dataset_module():
    dataset_path = os.path.join(os.getcwd(), 'datasets', 'pclt20k_seg.py')
    spec = importlib.util.spec_from_file_location('local_pclt20k_seg', dataset_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_loader(config, split='val'):
    dataset_mod = _load_dataset_module()
    if getattr(config, 'cipa_aligned', False):
        loaders = dataset_mod.get_pclt20k_loaders_cipa_aligned(
            config.root,
            config.image_size_2d,
            config.batch_size,
            config.num_workers,
            config.random_state,
            pin_memory=getattr(config, 'pin_memory', True),
        )
    else:
        loaders = dataset_mod.get_pclt20k_loaders(
            config.root,
            config.image_size_2d,
            config.batch_size,
            config.num_workers,
            val_ratio=config.val_ratio,
            random_state=config.random_state,
            use_case_split=getattr(config, 'use_case_split', True),
            pin_memory=getattr(config, 'pin_memory', True),
        )
    mapping = {'train': 0, 'val': 1, 'test': 2}
    return loaders[mapping[split]]


def _load_config(path):
    with open(path, 'r') as f:
        data = json.load(f)
    data.pop('checkpoint_dir', None)
    return SegMDTConfig(args=data)


def _load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'[ckpt] loaded {ckpt_path}')
    print(f'[ckpt] missing={len(missing)} unexpected={len(unexpected)}')


def _norm01(arr):
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    return arr / (arr.max() + 1e-8)


def _to_display(img_3ch):
    arr = img_3ch.detach().float().cpu().numpy()
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = np.clip(arr * std + mean, 0, 1)
        return arr.mean(axis=2)
    return _norm01(arr.squeeze())


def _compute_single(prob, gt, threshold):
    pred = prob > threshold
    target = gt > 0.5
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, np.logical_not(target)).sum())
    fn = int(np.logical_and(np.logical_not(pred), target).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(target)).sum())
    dice = 1.0 if 2 * tp + fp + fn == 0 else (2 * tp / (2 * tp + fp + fn))
    iou = 1.0 if tp + fp + fn == 0 else (tp / (tp + fp + fn))
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    hd95 = compute_hd95_pair(pred, target)
    gt_area = int(target.sum())
    pred_area = int(pred.sum())
    fp_ratio = 0.0 if pred_area == 0 else fp / pred_area
    fn_ratio = 0.0 if gt_area == 0 else fn / gt_area
    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'hd95': hd95,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'gt_area': gt_area,
        'pred_area': pred_area,
        'fp_ratio': fp_ratio,
        'fn_ratio': fn_ratio,
    }


def _overlay_error(gray, gt, prob, threshold):
    pred = prob > threshold
    target = gt > 0.5
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    tp = np.logical_and(pred, target)
    fp = np.logical_and(pred, np.logical_not(target))
    fn = np.logical_and(np.logical_not(pred), target)
    rgb[tp] = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    rgb[fp] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    rgb[fn] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return np.clip(rgb, 0.0, 1.0)


def _save_panel(row, out_path, threshold):
    import matplotlib.pyplot as plt
    ct_img = row['_ct_img']
    pet_img = row['_pet_img']
    gt = row['_gt']
    prob = row['_prob']
    pred = (prob > threshold).astype(np.float32)
    error = _overlay_error(ct_img, gt, prob, threshold)
    panels = [
        ('CT', ct_img, 'gray'),
        ('PET', pet_img, 'inferno'),
        ('GT', gt, 'gray'),
        ('Prob', prob, 'jet'),
        ('Pred', pred, 'gray'),
        ('Error TP=Y FP=G FN=R', error, None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (title, img, cmap) in zip(axes.flat, panels):
        if img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle(
        f"{row['rank_tag']} | {row['image_id']} | Dice={row['dice']:.4f} HD95={row['hd95']:.2f} "
        f"FP={row['fp']} FN={row['fn']} GT={row['gt_area']} Pred={row['pred_area']}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def _record_from_dataset(dataset, idx):
    if hasattr(dataset, 'records'):
        return dataset.records[int(idx)]
    return {}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--split', default='val', choices=('train', 'val', 'test'))
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--topk', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.getcwd() != script_dir:
        os.chdir(script_dir)
    sys.path.insert(0, os.getcwd())
    sys.modules.pop('datasets', None)

    cfg = _load_config(args.config)
    cfg.gpus = [args.gpu]
    cfg.batch_size = args.batch_size
    cfg.num_workers = 0

    device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    model = build_mdt_seg_teacher(cfg)['model'].to(device)
    _load_checkpoint(model, args.ckpt)
    model.eval()

    loader = _build_loader(cfg, split=args.split)
    dataset = loader.dataset
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for batch_idx, batch in enumerate(loader):
        ct = batch['ct'].float().to(device)
        pet = batch['pet'].float().to(device)
        mask = batch['mask'].float().to(device)
        outputs = model(ct, pet, target_size=mask.shape[-2:])
        logit = outputs['preds'][0] if isinstance(outputs.get('preds'), list) else outputs['pred']
        prob = torch.sigmoid(logit).squeeze(1).cpu().numpy()
        gt = mask.squeeze(1).cpu().numpy()
        idxs = batch['idx'].cpu().numpy().tolist()

        for b in range(ct.shape[0]):
            metrics = _compute_single(prob[b], gt[b], args.threshold)
            record = _record_from_dataset(dataset, idxs[b])
            row = {
                **metrics,
                'dataset_idx': int(idxs[b]),
                'case_id': record.get('case_id', ''),
                'image_id': record.get('image_id', f'b{batch_idx}_{b}'),
                'ct_path': record.get('ct_path', ''),
                'pet_path': record.get('pet_path', ''),
                'mask_path': record.get('mask_path', ''),
                '_ct_img': _to_display(ct[b]),
                '_pet_img': _to_display(pet[b]),
                '_gt': gt[b].astype(np.float32),
                '_prob': prob[b].astype(np.float32),
            }
            rows.append(row)
        if (batch_idx + 1) % 20 == 0:
            print(f'[progress] batches={batch_idx + 1} samples={len(rows)}')

    csv_fields = [
        'rank_tag', 'dataset_idx', 'case_id', 'image_id', 'dice', 'iou', 'precision', 'recall', 'hd95',
        'tp', 'fp', 'fn', 'tn', 'gt_area', 'pred_area', 'fp_ratio', 'fn_ratio', 'ct_path', 'pet_path', 'mask_path',
    ]

    categories = [
        ('lowest_dice', sorted(rows, key=lambda x: x['dice'])[:args.topk]),
        ('highest_hd95', sorted(rows, key=lambda x: x['hd95'], reverse=True)[:args.topk]),
        ('most_fp', sorted(rows, key=lambda x: x['fp'], reverse=True)[:args.topk]),
        ('most_fn', sorted(rows, key=lambda x: x['fn'], reverse=True)[:args.topk]),
        ('largest_gt', sorted(rows, key=lambda x: x['gt_area'], reverse=True)[:args.topk]),
        ('smallest_gt', sorted([r for r in rows if r['gt_area'] > 0], key=lambda x: x['gt_area'])[:args.topk]),
    ]

    all_csv = os.path.join(args.out_dir, 'all_samples_metrics.csv')
    with open(all_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields[1:])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in csv_fields[1:]})

    for tag, selected in categories:
        tag_dir = os.path.join(args.out_dir, tag)
        os.makedirs(tag_dir, exist_ok=True)
        tag_csv = os.path.join(args.out_dir, f'{tag}.csv')
        with open(tag_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for rank, row in enumerate(selected):
                row['rank_tag'] = f'{tag}_{rank:03d}'
                writer.writerow({k: row.get(k, '') for k in csv_fields})
                name = f"{rank:03d}_{row['image_id']}_dice{row['dice']:.3f}_hd{row['hd95']:.1f}_fp{row['fp']}_fn{row['fn']}.png"
                _save_panel(row, os.path.join(tag_dir, name), args.threshold)

    mean_dice = float(np.mean([r['dice'] for r in rows])) if rows else 0.0
    mean_hd95 = float(np.mean([r['hd95'] for r in rows])) if rows else 0.0
    print(f'[summary] samples={len(rows)} mean_dice={mean_dice:.4f} mean_hd95={mean_hd95:.4f}')
    print(f'[summary] saved to {args.out_dir}')


if __name__ == '__main__':
    main()
