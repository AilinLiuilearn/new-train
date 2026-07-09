# -*- coding: utf-8 -*-
"""Evaluate a saved MDT segmentation checkpoint under multiple PET missing rates.

This script keeps the missing masks reproducible by fixing the random seed for each
run and writing the generated per-rate results into a dedicated output directory.
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from configs.base import str2bool
from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher


def _set_seed(seed):
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _load_checkpoint(task, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'epoch' in ckpt:
        print(f"[checkpoint] epoch={ckpt['epoch']}")
    for k, v in task.networks.items():
        if k in ckpt:
            task.load_model_state_dict(v, ckpt[k], strict=False)
        else:
            raise KeyError(f'Missing network key in checkpoint: {k}')


def _build_eval_loader(config):
    # Reuse the exact same loader construction as training.
    import importlib.util

    root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(root, 'datasets', 'pclt20k_seg.py')
    spec = importlib.util.spec_from_file_location('local_pclt20k_seg', dataset_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if getattr(config, 'use_aligned_loader', False):
        loaders = module.get_pclt20k_loaders_textproxy_aligned(
            config.root,
            config.image_size_2d,
            config.batch_size,
            config.num_workers,
            config.random_state,
            pin_memory=getattr(config, 'pin_memory', True),
            aug_mode=getattr(config, 'aug_mode', 'cipa'),
            norm_mode=getattr(config, 'norm_mode', 'cipa'),
            train_list=getattr(config, 'train_list', 'train_original.txt'),
            val_list=getattr(config, 'val_list', 'test.txt'),
            test_list=getattr(config, 'test_list', 'test.txt'),
            pet_drop_prob=0.0,
        )
    else:
        loaders = module.get_pclt20k_loaders(
            config.root,
            config.image_size_2d,
            config.batch_size,
            config.num_workers,
            val_ratio=config.val_ratio,
            random_state=config.random_state,
            use_case_split=getattr(config, 'use_case_split', True),
            pin_memory=getattr(config, 'pin_memory', True),
            aug_mode=getattr(config, 'aug_mode', 'cipa'),
            norm_mode=getattr(config, 'norm_mode', 'imagenet'),
        )
    return loaders[2], module


def main():
    parents = [
        SegMDTConfig.ddp_parser(),
        SegMDTConfig.data_parser(),
        SegMDTConfig.model_parser(),
        SegMDTConfig.train_parser(),
        SegMDTConfig.logging_parser(),
        SegMDTConfig.task_specific_parser(),
    ]
    parser = argparse.ArgumentParser(
        description='Evaluate MDT checkpoint with fixed PET missing rates.',
        parents=parents,
        fromfile_prefix_chars='@',
    )
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--missing_rates', type=float, nargs='+', default=[0.0, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument('--export_only_missing_masks', type=str2bool, default=False)
    config = parser.parse_args()
    os.makedirs(config.output_dir, exist_ok=True)
    _set_seed(config.eval_random_seed)

    networks = build_mdt_seg_teacher(config)
    task = MDTSegTeacher(networks, config)
    _load_checkpoint(task, config.ckpt_path)

    loader, module = _build_eval_loader(config)
    test_dataset = loader.dataset
    results = {}

    def _dataset_meta(i):
        record = test_dataset.records[i]
        image_id = record.get('image_id', str(i))
        parts = image_id.split('_')
        slice_id = parts[-1] if len(parts) > 1 else image_id
        return {
            'dataset_idx': int(i),
            'case_id': record.get('case_id', parts[0] if parts else ''),
            'slice_id': slice_id,
            'image_id': image_id,
            'has_pet': 1 if record.get('pet_path') is not None else 0,
        }

    for rate in config.missing_rates:
        rate = float(rate)
        seed = int(config.eval_random_seed)
        tag = f'missing_{int(round(rate * 100)):03d}pct_seed_{seed}'
        print(f'\n[evaluate] rate={rate:.3f} seed={seed} tag={tag}')
        rng = np.random.default_rng(seed)
        if rate <= 0.0:
            eval_pet_available = np.ones(len(test_dataset), dtype=np.int64)
        elif rate >= 1.0:
            eval_pet_available = np.zeros(len(test_dataset), dtype=np.int64)
        else:
            eval_pet_available = (rng.random(len(test_dataset)) >= rate).astype(np.int64)
        sample_records = []
        for i in range(len(test_dataset)):
            meta = _dataset_meta(i)
            pet_available = int(eval_pet_available[i])
            sample_records.append({
                **meta,
                'pet_available': pet_available,
                'missing': int(1 - pet_available),
                'missing_rate': rate,
                'seed': seed,
            })
        results[f'{rate:.3f}'] = {
            'seed': seed,
            'samples': sample_records,
        }
        rate_dir = os.path.join(config.output_dir, f'random_missing_{int(round(rate * 100)):03d}pct')
        os.makedirs(rate_dir, exist_ok=True)
        with open(os.path.join(rate_dir, 'missing_samples.json'), 'w') as f:
            json.dump(sample_records, f, indent=2)

    with open(os.path.join(config.output_dir, 'all_missing_rate_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print('\n[done] Saved missing-mask results to:', config.output_dir)


if __name__ == '__main__':
    main()
