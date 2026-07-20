# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch

from configs.seg_mdt import SegMDTConfig
from datasets.pclt20k_seg import _records_from_ids, _read_list
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher
from utils.metrics_seg import SegmentationMetricsCIPA


def _group_by_case(records):
    grouped = defaultdict(list)
    for r in records:
        grouped[r['case_id']].append(r)
    return grouped


def _run_eval(task, records, mode, pet_available_mask=None):
    metric = SegmentationMetricsCIPA()
    total_loss = []
    for i, rec in enumerate(records):
        batch = {
            'ct': rec['ct'],
            'pet': rec['pet'],
            'mask': rec['mask'],
        }
        if mode == 'full':
            outputs = task.model(batch['ct'].unsqueeze(0).to(task.device), batch['pet'].unsqueeze(0).to(task.device), forward_mode='full')
        elif mode == 'missing':
            outputs = task.model(batch['ct'].unsqueeze(0).to(task.device), None, forward_mode='missing')
        else:
            pet = batch['pet'].unsqueeze(0).to(task.device)
            ct = batch['ct'].unsqueeze(0).to(task.device)
            pet_available = torch.tensor([pet_available_mask[i]], device=task.device, dtype=torch.long)
            outputs = task.model(ct, pet, pet_available=pet_available, forward_mode='auto')
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        loss, _ = task.criterion(logits, batch['mask'].unsqueeze(0).to(task.device).float())
        metric.update(logits, batch['mask'].unsqueeze(0).to(task.device).float())
        total_loss.append(float(loss))
    out = metric.compute()
    out['total_loss'] = float(np.mean(total_loss)) if total_loss else 0.0
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', type=str, required=True)
    p.add_argument('--root', type=str, default='/root/autodl-tmp/data/PCLT20K')
    p.add_argument('--random_state', type=int, default=2023)
    args = p.parse_args()
    ckpt = torch.load(os.path.join(args.checkpoint_dir, 'ckpt.best_joint.pth.tar'), map_location='cpu')
    cfg = SegMDTConfig(args={**ckpt['config'], 'root': args.root, 'random_state': args.random_state, 'checkpoint_dir': args.checkpoint_dir})
    task = MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)
    task.model.load_state_dict(ckpt['model'])
    task.model.eval()

    from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned
    _, _, test_loader = get_pclt20k_loaders_cipa_aligned(cfg.root, cfg.image_size_2d, cfg.batch_size, cfg.num_workers, cfg.random_state, cfg.pin_memory, 'none', cfg.norm_mode, cfg.train_split_file, cfg.val_split_file, cfg.test_split_file, checkpoint_dir=cfg.checkpoint_dir)
    test_records = []
    for batch in test_loader:
        for i in range(batch['ct'].shape[0]):
            test_records.append({
                'ct': batch['ct'][i],
                'pet': batch['pet'][i],
                'mask': batch['mask'][i],
                'case_id': batch['case_id'][i],
                'slice_id': batch['slice_id'][i],
            })

    case_ids = sorted({r['case_id'] for r in test_records})
    rng = np.random.default_rng(int(args.random_state))
    perm = list(rng.permutation(len(case_ids)))
    case_to_rank = {case_ids[idx]: rank for rank, idx in enumerate(perm)}
    def build_mask(rate):
        cut = int(round(rate * len(case_ids)))
        return {cid: 1 if case_to_rank[cid] < cut else 0 for cid in case_ids}

    results = []
    assignments = {}
    for rate in [0.0, 0.25, 0.5, 0.75, 1.0]:
        mask = build_mask(rate)
        assignments[str(rate)] = mask
        full_cases = [r for r in test_records if mask[r['case_id']] == 0]
        miss_cases = [r for r in test_records if mask[r['case_id']] == 1]
        full_out = _run_eval(task, full_cases, 'full') if full_cases else {'dice': 0, 'iou': 0, 'acc': 0, 'acc_pixel': 0, 'hd95': 0, 'total_loss': 0}
        miss_out = _run_eval(task, miss_cases, 'missing') if miss_cases else {'dice': 0, 'iou': 0, 'acc': 0, 'acc_pixel': 0, 'hd95': 0, 'total_loss': 0}
        joint_dice = 0.5 * full_out['dice'] + 0.5 * miss_out['dice']
        results.append({
            'missing_rate': rate,
            'full_dice': full_out['dice'],
            'full_iou': full_out['iou'],
            'full_acc': full_out['acc'],
            'full_acc_pixel': full_out['acc_pixel'],
            'full_hd95': full_out['hd95'],
            'missing_dice': miss_out['dice'],
            'missing_iou': miss_out['iou'],
            'missing_acc': miss_out['acc'],
            'missing_acc_pixel': miss_out['acc_pixel'],
            'missing_hd95': miss_out['hd95'],
            'joint_dice': joint_dice,
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
