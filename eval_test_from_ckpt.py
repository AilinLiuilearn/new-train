# -*- coding: utf-8 -*-
"""Evaluate a PET/CT segmentation checkpoint on val/test split (full PET-CT).

Uses config_args.json saved beside the checkpoint. Requires:
  - transformers==4.38.2 (Segformer MiT-B1 backbone)
  - medpy + scipy (HD95, consistent with training logs)
"""

import argparse
import importlib.util
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher


def _default_config_dict():
    parents = [
        SegMDTConfig.ddp_parser(),
        SegMDTConfig.data_parser(),
        SegMDTConfig.model_parser(),
        SegMDTConfig.train_parser(),
        SegMDTConfig.logging_parser(),
        SegMDTConfig.task_specific_parser(),
    ]
    parser = argparse.ArgumentParser(add_help=False, parents=parents)
    config = SegMDTConfig()
    parser.parse_args(args=[], namespace=config)
    return vars(config)


def _load_dataset_module(project_root):
    dataset_path = os.path.join(project_root, 'datasets', 'pclt20k_seg.py')
    spec = importlib.util.spec_from_file_location('local_pclt20k_seg', dataset_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_saved_config(ckpt_path):
    config_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), 'config_args.json')
    if not os.path.isfile(config_path):
        return None
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _make_config(args):
    defaults = _default_config_dict()
    defaults.update(_load_saved_config(args.ckpt) or {})
    overrides = {
        'root': args.root,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'gpus': args.gpus,
        'mixed_precision': False,
        'checkpoint_dir': os.path.dirname(os.path.abspath(args.ckpt)),
    }
    for key, value in overrides.items():
        if value is not None:
            defaults[key] = value
    if isinstance(defaults.get('gpus'), str):
        defaults['gpus'] = [int(x) for x in defaults['gpus'].split(',') if x.strip()]
    if not defaults.get('gpus'):
        defaults['gpus'] = [0]
    return SimpleNamespace(**defaults)


def _prepare_env(config):
    seed = int(getattr(config, 'random_state', 2023))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        visible = torch.cuda.device_count()
        gpus = [int(g) for g in getattr(config, 'gpus', [0]) if 0 <= int(g) < visible]
        config.gpus = gpus or [0]
        torch.cuda.set_device(int(config.gpus[0]))
        torch.cuda.manual_seed_all(seed)
    else:
        config.gpus = [0]


def _build_loaders(config, project_root):
    dataset_mod = _load_dataset_module(project_root)
    if getattr(config, 'use_aligned_loader', False):
        return dataset_mod.get_pclt20k_loaders_textproxy_aligned(
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
    raise ValueError('Only use_aligned_loader=True is supported for this eval script.')


def _load_checkpoint(task, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    loaded = False
    for key, model in task.networks.items():
        if key in ckpt:
            missing, unexpected = task.load_model_state_dict(model, ckpt[key], strict=False)
            print(f'[load] {key}: missing={len(missing)} unexpected={len(unexpected)}')
            if missing:
                print(f'[load] missing examples: {missing[:8]}')
            loaded = True
    if not loaded:
        state = ckpt.get('state_dict') or ckpt.get('model') or ckpt
        missing, unexpected = task.load_model_state_dict(task.networks['model'], state, strict=False)
        print(f'[load] model fallback: missing={len(missing)} unexpected={len(unexpected)}')
    print(f'[load] checkpoint epoch={ckpt.get("epoch", "?")} path={ckpt_path}')


def _check_hd95_backend():
    try:
        from medpy.metric.binary import hd95  # noqa: F401
        print('[deps] HD95 backend: medpy (recommended, matches training)')
        return
    except ImportError:
        pass
    try:
        from scipy.ndimage import distance_transform_edt  # noqa: F401
        print('[warn] medpy not found; HD95 will use scipy fallback (may differ from training logs)')
        return
    except ImportError:
        pass
    print('[warn] medpy/scipy not found; HD95 uses numpy fallback and can be MUCH larger than training logs!')


def _format_metrics(metrics):
    keys = ['dice', 'iou', 'acc', 'acc_pixel', 'sensitivity', 'specificity', 'hd95', 'total_loss']
    parts = []
    for key in keys:
        if key not in metrics:
            continue
        val = float(metrics[key])
        if key == 'hd95':
            parts.append(f'{key}={val:.2f}')
        else:
            parts.append(f'{key}={val:.4f}')
    return ' '.join(parts)


def main():
    parser = argparse.ArgumentParser(description='Evaluate checkpoint on val/test split (full PET-CT).')
    parser.add_argument(
        '--ckpt',
        type=str,
        default='/root/autodl-tmp/mkd-main/new-train/checkpoints_new/MDT/dmome_ds_no_tpe_v7/ckpt.best_dice.pth.tar',
    )
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--split', type=str, default='test', choices=('test', 'val', 'both'))
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--gpus', type=lambda s: [int(x) for x in s.split(',') if x.strip()], default=None)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--out-json', type=str, default=None)
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f'Checkpoint not found: {args.ckpt}')

    project_root = os.path.dirname(os.path.abspath(__file__))
    config = _make_config(args)
    _prepare_env(config)
    _check_hd95_backend()

    _, val_loader, test_loader = _build_loaders(config, project_root)
    task = MDTSegTeacher(build_mdt_seg_teacher(config), config)
    _load_checkpoint(task, args.ckpt)

    splits = []
    if args.split in ('test', 'both'):
        splits.append(('test', test_loader))
    if args.split in ('val', 'both'):
        splits.append(('val', val_loader))

    results = {}
    for name, loader in splits:
        metrics = task.evaluate(
            loader,
            threshold=args.threshold,
            eval_mode='full',
            tag=f'{name}_full_pet_ct',
        )
        results[name] = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        print(f'=== {name.upper()} (full PET-CT) ===')
        print(_format_metrics(metrics))

    out_json = args.out_json or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), 'test_best_dice_metrics.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'checkpoint': args.ckpt, 'results': results}, f, indent=2, ensure_ascii=False)
    print(f'[saved] {out_json}')


if __name__ == '__main__':
    main()
