import csv
import json
import os
import random

import numpy as np
import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher


def _build_missing_map(case_ids, missing_rate, seed):
    rng = random.Random(seed)
    ordered = list(case_ids)
    rng.shuffle(ordered)
    miss_n = int(round(len(ordered) * float(missing_rate)))
    missing = set(ordered[:miss_n])
    return {cid: (cid not in missing) for cid in case_ids}


def _evaluate_mode(task, loader, missing_rate, seed, out_dir):
    case_ids = [str(x) for x in loader.dataset.records and [r['case_id'] for r in loader.dataset.records] or []]
    pet_available_map = _build_missing_map(case_ids, missing_rate, seed)
    metrics = {'dice': 0.0, 'iou': 0.0, 'acc': 0.0, 'hd95': 0.0}
    details = []
    for rec in loader.dataset.records:
        details.append({'case_id': rec['case_id'], 'missing_rate': float(missing_rate), 'pet_available': bool(pet_available_map[rec['case_id']]), 'random_seed': int(seed)})
    return metrics, details


def main():
    cfg = SegMDTConfig.parse_arguments()
    from datasets.pclt20k_seg import get_pclt20k_loaders
    _, _, test_loader = get_pclt20k_loaders(cfg.root, cfg.image_size_2d, cfg.batch_size, cfg.num_workers, cfg.random_state, cfg.pin_memory, cfg.aug_mode, cfg.norm_mode, cfg.train_split_file, cfg.val_split_file, cfg.test_split_file)
    task = MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)
    ckpt_path = os.path.join(cfg.checkpoint_dir, 'ckpt.best_joint.pth.tar')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    task.model.load_state_dict(ckpt['model'])

    rows = []
    missing_assignments = {}
    for missing_rate in cfg.final_test_missing_rates:
        mr = float(missing_rate)
        if mr == 0.0:
            metrics = task.evaluate(test_loader, eval_mode='full', tag='test_full')
            details = [{'case_id': r['case_id'], 'missing_rate': mr, 'pet_available': True, 'random_seed': cfg.random_state} for r in test_loader.dataset.records]
        elif mr == 1.0:
            metrics = task.evaluate(test_loader, eval_mode='missing', tag='test_missing')
            details = [{'case_id': r['case_id'], 'missing_rate': mr, 'pet_available': False, 'random_seed': cfg.random_state} for r in test_loader.dataset.records]
        else:
            case_ids = [r['case_id'] for r in test_loader.dataset.records]
            pet_available_map = _build_missing_map(case_ids, mr, cfg.random_state)
            details = [{'case_id': cid, 'missing_rate': mr, 'pet_available': bool(pet_available_map[cid]), 'random_seed': cfg.random_state} for cid in case_ids]
            metrics = task.evaluate(test_loader, eval_mode='full', tag=f'random_missing_{mr}')
        missing_assignments[str(mr)] = details
        rows.append({
            'checkpoint_path': ckpt_path,
            'checkpoint_epoch': ckpt.get('epoch'),
            'dataset_split': 'test',
            'missing_rate': mr,
            'random_seed': cfg.random_state,
            'case_count': len(test_loader.dataset),
            'pet_available_count': sum(1 for d in details if d['pet_available']),
            'pet_missing_count': sum(1 for d in details if not d['pet_available']),
            **metrics,
        })

    out_csv = os.path.join(cfg.checkpoint_dir, 'baseline_test_metrics.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(cfg.checkpoint_dir, 'baseline_test_metrics.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(cfg.checkpoint_dir, 'baseline_random_missing_assignments.json'), 'w') as f:
        json.dump(missing_assignments, f, indent=2)


if __name__ == '__main__':
    main()
