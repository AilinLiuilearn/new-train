# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher
from utils.reproducibility import configure_reproducibility, describe_reproducibility_env
from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned


def _tensor_sha256(tensor: torch.Tensor) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _state_digest(model: torch.nn.Module) -> dict:
    return {k: _tensor_sha256(v) for k, v in model.state_dict().items() if torch.is_tensor(v)}


def _compare_metrics(a: dict, b: dict) -> dict:
    keys = ['total_loss', 'dice', 'iou', 'acc', 'acc_pixel', 'hd95', 'sensitivity', 'specificity', 'precision', 'f1']
    out = {}
    for k in keys:
        av = float(a.get(k, 0.0))
        bv = float(b.get(k, 0.0))
        diff = abs(av - bv)
        rel = diff / (abs(av) + 1e-12)
        out[k] = {'a': av, 'b': bv, 'abs_diff': diff, 'rel_diff': rel}
    return out


def run_one(task, loader, eval_mode, tag):
    before = _state_digest(task.model)
    out = task.evaluate(loader, eval_mode=eval_mode, tag=tag)
    after = _state_digest(task.model)
    return out, before, after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg = SegMDTConfig(args=ckpt['config'])
    configure_reproducibility(int(cfg.random_state), getattr(cfg, 'deterministic_mode', 'strict'))

    train_loader, val_loader, test_loader = get_pclt20k_loaders_cipa_aligned(
        cfg.root, cfg.image_size_2d, cfg.batch_size, cfg.num_workers, cfg.random_state,
        cfg.pin_memory, cfg.aug_mode, cfg.norm_mode, cfg.train_split_file, cfg.val_split_file, cfg.test_split_file,
        checkpoint_dir=cfg.checkpoint_dir,
    )
    task = MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)
    task.model.load_state_dict(ckpt['model'], strict=True)
    if ckpt.get('optimizer'):
        task.optimizer.load_state_dict(ckpt['optimizer'])
    if ckpt.get('scheduler') and task.scheduler is not None:
        task.scheduler.load_state_dict(ckpt['scheduler'])

    reports = {}
    for mode, loader in [('full', val_loader), ('missing', val_loader)]:
        r1, b1, a1 = run_one(task, loader, mode if mode == 'full' else 'fixed_missing', f'{mode}_1')
        r2, b2, a2 = run_one(task, loader, mode if mode == 'full' else 'fixed_missing', f'{mode}_2')
        reports[mode] = {
            'metrics_diff': _compare_metrics(r1, r2),
            'state_before_equal': b1 == b2,
            'state_after_equal': a1 == a2,
        }

    manifest = {
        'checkpoint': os.path.abspath(args.checkpoint),
        'env': describe_reproducibility_env(),
        'reports': reports,
    }
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
