# -*- coding: utf-8 -*-
"""Full-modal baseline evaluation: single Full operating point, no missing rates."""
import argparse
import csv
import json
import os

import numpy as np
import torch

from configs.seg_mdt import SegMDTConfig
from configs.base import str2bool
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher
from utils.metrics_seg import SegmentationMetricsCIPA


@torch.inference_mode()
def _run_full_test(task, loader):
    metric = SegmentationMetricsCIPA()
    total_loss = []
    total_case_ids = set()
    slice_count = 0
    for batch in loader:
        ct = batch['ct'].to(task.device, non_blocking=True)
        pet = batch['pet'].to(task.device, non_blocking=True)
        mask = batch['mask'].to(task.device, non_blocking=True).float()
        case_ids = list(batch['case_id'])
        total_case_ids.update(case_ids)
        outputs = task.model(ct, pet)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        loss, _ = task.criterion(logits, mask)
        metric.update(logits, mask)
        total_loss.append(float(loss))
        slice_count += len(case_ids)
    out = metric.compute()
    out['loss'] = float(np.mean(total_loss)) if total_loss else 0.0
    out['total_case_count'] = len(total_case_ids)
    out['slice_count'] = slice_count
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', type=str, required=True)
    p.add_argument('--root', type=str, default='/root/autodl-tmp/data/PCLT20K')
    p.add_argument('--random_state', type=int, default=2023)
    p.add_argument('--use_ema', type=str2bool, default=False)
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
    state_dict = ckpt['model']
    if args.use_ema:
        if ckpt.get('model_ema') is None:
            raise SystemExit('--use_ema requested but the checkpoint has no model_ema')
        state_dict = ckpt['model_ema']
        print('[eval_joint_baseline] using EMA weights for evaluation')
    else:
        print('[eval_joint_baseline] using raw model weights for evaluation')
    task.model.load_state_dict(state_dict, strict=True)
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

    out = _run_full_test(task, test_loader)
    results = [{
        'missing_rate': 0.0,
        'dice': out['dice'],
        'iou': out['iou'],
        'acc': out['acc'],
        'acc_pixel': out['acc_pixel'],
        'hd95': out['hd95'],
        'loss': out['loss'],
        'total_case_count': out['total_case_count'],
        'slice_count': out['slice_count'],
    }]

    csv_path = os.path.join(args.checkpoint_dir, 'final_test_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    with open(os.path.join(args.checkpoint_dir, 'final_test_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('done')


if __name__ == '__main__':
    main()
