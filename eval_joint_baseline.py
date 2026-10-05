# -*- coding: utf-8 -*-
"""Evaluate a saved checkpoint on the test set at all missing rates.

  --mode dual     loads the dual model and routes per-sample state
                  (Full and Missing come from the SAME checkpoint).
  --mode ct_only  loads the CT-only model; results are PET-invariant.
  --use_ema       evaluates the EMA weights instead of raw.

Both experiments share the same seeded patient-level missing assignments,
so their operating points are directly comparable. Output:
  final_test_metrics.csv / .json + final_missing_case_assignments.json
"""
import argparse
import os

import torch

from configs.seg_mdt import SegMDTConfig
from configs.base import str2bool
from models.build_mdt_seg import build_ct_only_model, build_dual_model
from tasks.mdt_seg import MDTSegTeacher
from utils.run_common import eval_missing_rates


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--mode', type=str, required=True, choices=('dual', 'ct_only'))
    p.add_argument('--root', type=str, default='/root/autodl-tmp/data/PCLT20K')
    p.add_argument('--random_state', type=int, default=2023)
    p.add_argument('--use_ema', type=str2bool, default=False)
    p.add_argument('--output_dir', type=str, default=None)
    args = p.parse_args()

    ckpt = MDTSegTeacher.load_state_dicts(args.checkpoint)
    saved_config = dict(ckpt['config'])
    saved_config.pop('checkpoint_dir', None)
    saved_config['root'] = args.root
    saved_config['random_state'] = args.random_state
    saved_config['ct_pretrained_path'] = None
    saved_config['pet_pretrained_path'] = None
    saved_config['pretrained'] = False
    cfg = SegMDTConfig(args=saved_config)

    ct_only = args.mode == 'ct_only'
    builder = build_ct_only_model if ct_only else build_dual_model
    task = MDTSegTeacher(builder(cfg), cfg)
    state_dict = ckpt['model']
    if args.use_ema:
        if ckpt.get('model_ema') is None:
            raise SystemExit('--use_ema requested but the checkpoint has no model_ema')
        state_dict = ckpt['model_ema']
        print('[eval] using EMA weights for evaluation')
    else:
        print('[eval] using raw model weights for evaluation')
    task.model.load_state_dict(state_dict, strict=True)
    task.model.eval()

    from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned
    _, _, test_loader = get_pclt20k_loaders_cipa_aligned(
        cfg.root, cfg.image_size_2d, cfg.batch_size, cfg.num_workers,
        cfg.random_state, cfg.pin_memory, 'none', cfg.norm_mode,
        cfg.train_split_file, cfg.val_split_file, cfg.test_split_file,
        checkpoint_dir=cfg.checkpoint_dir,
        allow_val_equals_test=bool(getattr(cfg, 'allow_val_equals_test', False)), ct_only=ct_only,
    )
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    weights_tag = 'ema' if args.use_ema else 'raw'
    eval_missing_rates(task, task.model, test_loader, int(args.random_state),
                       output_dir, 'final_test', weights_tag=weights_tag,
                       ct_only=ct_only)
    print('done')


if __name__ == '__main__':
    main()