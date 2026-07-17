# -*- coding: utf-8 -*-
"""Per-sample analysis for PET contribution.

This script compares a CT-only checkpoint against a CT+PET checkpoint on the same
validation/test split and reports per-sample improvements, degradations, and
simple attribution-style signals.
"""

import argparse
import csv
import importlib.util
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher


def _load_dataset_module():
    root = os.getcwd()
    dataset_path = os.path.join(root, 'datasets', 'pclt20k_seg.py')
    spec = importlib.util.spec_from_file_location('local_pclt20k_seg', dataset_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_env(config):
    gpus = [int(g) for g in config.gpus] if config.gpus else [0]
    if torch.cuda.is_available():
        visible = torch.cuda.device_count()
        gpus = [g for g in gpus if 0 <= g < visible]
        if not gpus:
            gpus = [0]
    config.gpus = gpus
    if torch.cuda.is_available():
        torch.cuda.set_device(gpus[0])
    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)
    return gpus[0]


def _load_checkpoint(task, path):
    ckpt = torch.load(path, map_location='cpu')
    for k, v in task.networks.items():
        if k in ckpt:
            task.load_model_state_dict(v, ckpt[k], strict=False)


def _set_checkpoint_config(path, model_arch, ct_backbone, pet_backbone=None):
    cfg = SegMDTConfig.parse_arguments()
    cfg.model_arch = model_arch
    cfg.ct_backbone = ct_backbone
    if pet_backbone is not None:
        cfg.pet_backbone = pet_backbone
    cfg.checkpoint_dir = os.path.dirname(path)
    cfg.gpus = [0]
    cfg.mixed_precision = False
    cfg.eval_full_pet = True
    cfg.eval_fixed_missing_pet = False
    cfg.eval_random_missing_pet = False
    cfg.use_deep_supervision = False
    cfg.deep_supervision = False
    cfg.batch_size = 1
    cfg.num_workers = 0
    cfg.pin_memory = False
    cfg.norm_mode = 'cipa'
    cfg.aug_mode = 'none'
    return cfg


def _build_val_loader(args):
    dataset_mod = _load_dataset_module()
    return dataset_mod.get_pclt20k_loaders_textproxy_aligned(
        args.root,
        args.image_size_2d,
        args.batch_size,
        args.num_workers,
        args.random_state,
        pin_memory=False,
        aug_mode='none',
        norm_mode=args.norm_mode,
        train_list=args.train_list,
        val_list=args.val_list,
        test_list=args.test_list,
        pet_drop_prob=0.0,
    )[1]


def _dice_from_logits(logits, mask, threshold=0.5):
    pred = (torch.sigmoid(logits) >= threshold).float()
    inter = (pred * mask).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3))
    return ((2 * inter + 1e-6) / (denom + 1e-6))


def _hd95_proxy(pred, target):
    pred_edges = F.max_pool2d(pred, kernel_size=3, stride=1, padding=1) - F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    tgt_edges = F.max_pool2d(target, kernel_size=3, stride=1, padding=1) - F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    return (pred_edges - tgt_edges).abs().mean(dim=(1, 2, 3))


def _get_image_ids(batch):
    if 'image_id' in batch:
        ids = batch['image_id']
        if isinstance(ids, (list, tuple)):
            return [str(x) for x in ids]
        return [str(ids)]
    if 'case_id' in batch and 'slice_id' in batch:
        case_ids = batch['case_id']
        slice_ids = batch['slice_id']
        if not isinstance(case_ids, (list, tuple)):
            case_ids = [case_ids]
        if not isinstance(slice_ids, (list, tuple)):
            slice_ids = [slice_ids]
        return [f'{c}_{s}' for c, s in zip(case_ids, slice_ids)]
    idx = batch.get('idx', [])
    if not isinstance(idx, (list, tuple)):
        idx = [idx]
    return [str(x) for x in idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/root/autodl-tmp/data/PCLT20K')
    parser.add_argument('--image_size_2d', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--random_state', type=int, default=2023)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--norm_mode', type=str, default='cipa', choices=('imagenet', 'cipa'))
    parser.add_argument('--train_list', type=str, default='train_original.txt')
    parser.add_argument('--val_list', type=str, default='test.txt')
    parser.add_argument('--test_list', type=str, default='test.txt')
    parser.add_argument('--ct_ckpt', type=str, required=True)
    parser.add_argument('--full_ckpt', type=str, required=True)
    parser.add_argument('--ct_backbone', type=str, default='convnextv2_nano')
    parser.add_argument('--pet_backbone', type=str, default='mit_b1')
    parser.add_argument('--out_csv', type=str, default='pet_contribution_per_sample.csv')
    parser.add_argument('--top_k', type=int, default=20)
    args = parser.parse_args()

    _prepare_env(args)
    loader = _build_val_loader(args)

    ct_cfg = _set_checkpoint_config(args.ct_ckpt, 'pet_contribution_ct_only', args.ct_backbone)
    full_cfg = _set_checkpoint_config(args.full_ckpt, 'pet_contribution_full', args.ct_backbone, args.pet_backbone)

    ct_net = build_mdt_seg_teacher(ct_cfg)
    full_net = build_mdt_seg_teacher(full_cfg)
    ct_task = MDTSegTeacher(ct_net, ct_cfg)
    full_task = MDTSegTeacher(full_net, full_cfg)
    _load_checkpoint(ct_task, args.ct_ckpt)
    _load_checkpoint(full_task, args.full_ckpt)

    ct_task.networks['model'].eval()
    full_task.networks['model'].eval()

    rows = []
    threshold = 0.5

    with torch.no_grad():
        for batch in loader:
            ct = batch['ct'].float().to(ct_task.device)
            pet = batch['pet'].float().to(full_task.device)
            mask = batch['mask'].float().to(ct_task.device)
            image_ids = _get_image_ids(batch)

            ct_out = ct_task.networks['model'](ct, pet, target_size=mask.shape[-2:], forward_mode='auto')
            full_out = full_task.networks['model'](ct, pet, target_size=mask.shape[-2:], forward_mode='auto')

            ct_logits = ct_out['logits'].detach().float().cpu()
            full_logits = full_out['logits'].detach().float().cpu()
            mask_cpu = mask.detach().float().cpu()

            ct_dice = _dice_from_logits(ct_logits, mask_cpu, threshold=threshold)
            full_dice = _dice_from_logits(full_logits, mask_cpu, threshold=threshold)
            dice_gain = full_dice - ct_dice

            ct_pred = (torch.sigmoid(ct_logits) >= threshold).float()
            full_pred = (torch.sigmoid(full_logits) >= threshold).float()
            pred_delta = (full_pred - ct_pred).abs().mean(dim=(1, 2, 3))
            logit_delta = (full_logits - ct_logits).abs().mean(dim=(1, 2, 3))
            hd95_proxy_ct = _hd95_proxy(ct_pred, mask_cpu)
            hd95_proxy_full = _hd95_proxy(full_pred, mask_cpu)
            hd95_proxy_gain = hd95_proxy_ct - hd95_proxy_full

            for i in range(ct.shape[0]):
                rows.append({
                    'image_id': image_ids[i] if i < len(image_ids) else str(i),
                    'ct_dice': float(ct_dice[i]),
                    'full_dice': float(full_dice[i]),
                    'dice_gain': float(dice_gain[i]),
                    'ct_hd95_proxy': float(hd95_proxy_ct[i]),
                    'full_hd95_proxy': float(hd95_proxy_full[i]),
                    'hd95_proxy_gain': float(hd95_proxy_gain[i]),
                    'pred_delta_mean': float(pred_delta[i]),
                    'logit_delta_mean': float(logit_delta[i]),
                    'mask_pos_fraction': float(mask_cpu[i].mean()),
                })

    rows.sort(key=lambda x: x['dice_gain'], reverse=True)
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    top_positive = rows[: args.top_k]
    top_negative = sorted(rows, key=lambda x: x['dice_gain'])[: args.top_k]

    summary = {
        'num_samples': len(rows),
        'mean_ct_dice': float(np.mean([r['ct_dice'] for r in rows])) if rows else 0.0,
        'mean_full_dice': float(np.mean([r['full_dice'] for r in rows])) if rows else 0.0,
        'mean_dice_gain': float(np.mean([r['dice_gain'] for r in rows])) if rows else 0.0,
        'median_dice_gain': float(np.median([r['dice_gain'] for r in rows])) if rows else 0.0,
        'mean_hd95_proxy_gain': float(np.mean([r['hd95_proxy_gain'] for r in rows])) if rows else 0.0,
        'top_positive': top_positive[:5],
        'top_negative': top_negative[:5],
        'csv_path': os.path.abspath(args.out_csv),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('\nTop positive PET contribution samples:')
    for r in top_positive[: min(10, len(top_positive))]:
        print(f"  {r['image_id']}: dice_gain={r['dice_gain']:+.4f}, ct_dice={r['ct_dice']:.4f}, full_dice={r['full_dice']:.4f}, hd95_proxy_gain={r['hd95_proxy_gain']:+.4f}")
    print('\nTop negative PET contribution samples:')
    for r in top_negative[: min(10, len(top_negative))]:
        print(f"  {r['image_id']}: dice_gain={r['dice_gain']:+.4f}, ct_dice={r['ct_dice']:.4f}, full_dice={r['full_dice']:.4f}, hd95_proxy_gain={r['hd95_proxy_gain']:+.4f}")

    print('\nGuidance signals:')
    print('- Samples with large positive dice_gain / hd95_proxy_gain are cases where PET improves lesion completeness or boundary quality.')
    print('- Samples with near-zero gain suggest the lesion is already clear in CT, so PET adds little.')
    print('- Negative-gain samples may indicate PET noise, misregistration, or that the model over-relies on PET.')
    print('- If gains concentrate on small/ambiguous lesions, PET is acting as a boundary/ambiguity resolver rather than a global segmenter.')


if __name__ == '__main__':
    main()
