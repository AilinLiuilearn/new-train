# -*- coding: utf-8 -*-
"""Evaluate a trained MDT checkpoint under different PET missing rates.

This script is deterministic: it fixes Python/NumPy/PyTorch seeds and uses the
same random generator seed for each missing-rate evaluation so results are
reproducible across runs on the same hardware/software stack.
"""

import argparse
import importlib.util
import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

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
    seed = int(config.random_state)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if torch.cuda.is_available():
        torch.cuda.set_device(gpus[0])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, 'enable_math_sdp'):
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return gpus[0]


def _build_test_loader(config):
    dataset_mod = _load_dataset_module()
    if getattr(config, 'use_aligned_loader', False):
        _, _, test_loader = dataset_mod.get_pclt20k_loaders_textproxy_aligned(
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
        return test_loader
    if getattr(config, 'cipa_aligned', False):
        _, _, test_loader = dataset_mod.get_pclt20k_loaders_cipa_aligned(
            config.root,
            config.image_size_2d,
            config.batch_size,
            config.num_workers,
            config.random_state,
            pin_memory=getattr(config, 'pin_memory', True),
            aug_mode=getattr(config, 'aug_mode', 'cipa'),
            norm_mode=getattr(config, 'norm_mode', 'imagenet'),
            train_split_file=getattr(config, 'train_split_file', 'train.txt'),
            val_split_file=getattr(config, 'val_split_file', 'val.txt'),
            test_split_file=getattr(config, 'test_split_file', 'test.txt'),
        )
        return test_loader
    _, _, test_loader = dataset_mod.get_pclt20k_loaders(
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
    return test_loader


def _unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _load_checkpoint(task, path):
    ckpt = torch.load(path, map_location='cpu')
    for k, v in task.networks.items():
        if k in ckpt:
            task.load_model_state_dict(v, ckpt[k], strict=False)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if os.getcwd() != project_root:
        os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    sys.modules.pop('datasets', None)

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--checkpoint_path', type=str, default=None)
    pre_parser.add_argument('--eval_missing_rates', type=str, default='0.0,0.25,0.5,0.75,1.0')
    pre_args, remaining_argv = pre_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv

    config = SegMDTConfig.parse_arguments()
    config.checkpoint_path = pre_args.checkpoint_path
    config.eval_missing_rates = pre_args.eval_missing_rates
    g0 = _prepare_env(config)
    print(f'GPU={g0}')

    checkpoint_path = getattr(config, 'checkpoint_path', None)
    if not checkpoint_path:
        checkpoint_path = os.path.join(
            config.checkpoint_dir,
            'ckpt.best_joint_dice.pth.tar',
        )
    missing_rates = getattr(config, 'eval_missing_rates', None)
    if not missing_rates:
        missing_rates = [0.0, 0.25, 0.5, 0.75, 1.0]
    else:
        if isinstance(missing_rates, str):
            missing_rates = [float(x.strip()) for x in missing_rates.split(',') if x.strip()]
        else:
            missing_rates = [float(x) for x in missing_rates]

    print(f'checkpoint_path={checkpoint_path}')
    print(f'eval_missing_rates={missing_rates}')
    print(f'random_state={config.random_state}')
    print(f'batch_size={config.batch_size}')

    test_loader = _build_test_loader(config)
    networks = build_mdt_seg_teacher(config)
    task = MDTSegTeacher(networks, config)
    _load_checkpoint(task, checkpoint_path)

    model = _unwrap(task.networks['model'])
    model.eval()

    for rate in missing_rates:
        metrics = task.evaluate(
            test_loader,
            eval_mode='random_missing',
            random_pet_drop_prob=float(rate),
            random_seed=int(config.random_state),
            tag=f'test_missing_rate_{rate:.2f}',
        )
        print(
            f'missing_rate={rate:.2f} '
            f'Dice={metrics["dice"]:.4f} '
            f'IoU={metrics["iou"]:.4f} '
            f'Acc={metrics["acc"]:.4f} '
            f'HD95={metrics["hd95"]:.2f} '
            f'Loss={metrics["total_loss"]:.6f}'
        )


if __name__ == '__main__':
    main()
