# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher
from utils.metrics_seg import SegmentationMetricsCIPA


def _group_by_case(records):
    grouped = defaultdict(list)
    for r in records:
        grouped[r['case_id']].append(r)
    return grouped


@torch.inference_mode()
def _run_full_test(task, loader, case_mask):
    metric = SegmentationMetricsCIPA()
    total_loss = []
    total_case_ids = set()
    missing_case_ids = set()
    slice_count = 0
    for batch in loader:
        ct = batch['ct'].to(task.device, non_blocking=True)
        pet = batch['pet'].to(task.device, non_blocking=True)
        mask = batch['mask'].to(task.device, non_blocking=True).float()
        case_ids = list(batch['case_id'])
        total_case_ids.update(case_ids)
        pet_available = torch.tensor([0 if case_mask.get(cid, 1) else 1 for cid in case_ids], device=task.device, dtype=torch.long)
        missing_case_ids.update([cid for cid in case_ids if case_mask.get(cid, 1) == 1])
        outputs = task.model(ct, pet, pet_available=pet_available, forward_mode='auto')
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        loss, _ = task.criterion(logits, mask)
        metric.update(logits, mask)
        total_loss.append(float(loss))
        slice_count += len(case_ids)
    out = metric.compute()
    out['loss'] = float(np.mean(total_loss)) if total_loss else 0.0
    out['missing_case_count'] = len(missing_case_ids)
    out['total_case_count'] = len(total_case_ids)
    out['slice_count'] = slice_count
    return out


def _build_case_mask(case_ids, missing_rate, seed):
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(len(case_ids)))
    cut = int(round(float(missing_rate) * len(case_ids)))
    missing = {case_ids[idx] for idx in perm[:cut]}
    return {cid: (1 if cid in missing else 0) for cid in case_ids}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', type=str, required=True)
    p.add_argument('--root', type=str, default='/root/autodl-tmp/data/PCLT20K')
    p.add_argument('--random_state', type=int, default=2023)
    args = p.parse_args()

    ckpt = torch.load(os.path.join(args.checkpoint_dir, 'ckpt.best_joint.pth.tar'), map_location='cpu')
    saved_config = dict(ckpt['config'])
    saved_config.pop('checkpoint_dir', None)
    saved_config['root'] = args.root
    saved_config['random_state'] = args.random_state
    saved_config['ct_pretrained_path'] = None
    saved_config['pet_pretrained_path'] = None
    cfg = SegMDTConfig(args=saved_config)

    task = MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)
    task.model.load_state_dict(ckpt['model'], strict=True)
    task.model.eval()

    from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned
    _, _, test_loader = get_pclt20k_loaders_cipa_aligned(
        cfg.root,
        cfg.image_size_2d,
        cfg.batch_size,
        cfg.num_workers,
        cfg.random_state,
        cfg.pin_memory,
        'none',
        cfg.norm_mode,
        cfg.train_split_file,
        cfg.val_split_file,
        cfg.test_split_file,
        checkpoint_dir=cfg.checkpoint_dir,
    )

    all_case_ids = []
    for batch in test_loader:
        all_case_ids.extend(list(batch['case_id']))
    all_case_ids = sorted(set(all_case_ids))

    rates = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []
    assignments = {}
    for rate in rates:
        case_mask = _build_case_mask(all_case_ids, rate, int(args.random_state))
        assignments[str(rate)] = case_mask
        out = _run_full_test(task, test_loader, case_mask)
        results.append({
            'missing_rate': rate,
            'dice': out['dice'],
            'iou': out['iou'],
            'acc': out['acc'],
            'acc_pixel': out['acc_pixel'],
            'hd95': out['hd95'],
            'loss': out['loss'],
            'missing_case_count': out['missing_case_count'],
            'total_case_count': out['total_case_count'],
            'slice_count': out['slice_count'],
        })

    csv_path = os.path.join(args.checkpoint_dir, 'final_test_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    with open(os.path.join(args.checkpoint_dir, 'final_test_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.checkpoint_dir, 'final_missing_case_assignments.json'), 'w') as f:
        json.dump(assignments, f, indent=2)
    print('done')


if __name__ == '__main__':
    main()
