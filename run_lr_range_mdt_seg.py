# -*- coding: utf-8 -*-
"""LR range test for MDT segmentation baseline."""

import csv
import importlib.util
import json
import math
import os
import random
import sys

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
    g0 = int(config.gpus[0]) if config.gpus else 0
    config.gpus = [g0]
    seed = int(config.random_state)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(g0)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return g0


def _build_train_loader(config):
    dataset_mod = _load_dataset_module()
    if getattr(config, 'cipa_aligned', False):
        train_loader, _, _ = dataset_mod.get_pclt20k_loaders_cipa_aligned(
            config.root,
            config.image_size_2d,
            config.batch_size,
            config.num_workers,
            config.random_state,
            pin_memory=getattr(config, 'pin_memory', True),
            aug_mode=getattr(config, 'aug_mode', 'cipa'),
            norm_mode=getattr(config, 'norm_mode', 'imagenet'),
        )
    else:
        train_loader, _, _ = dataset_mod.get_pclt20k_loaders(
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
    return train_loader


def _grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        val = p.grad.detach().data.norm(2).item()
        total += val * val
    return math.sqrt(total)


def _save_plot(csv_path, png_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[lr_range] matplotlib not installed, skip plot')
        return
    lrs, losses = [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lrs.append(float(row['lr']))
            losses.append(float(row['smooth_loss']))
    if not lrs:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(lrs, losses, linewidth=2)
    plt.xscale('log')
    plt.xlabel('Learning rate')
    plt.ylabel('Smoothed training loss')
    plt.title('LR Range Test')
    plt.grid(True, which='both', linestyle='--', alpha=0.35)
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()


def main():
    if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.getcwd())
    sys.modules.pop('datasets', None)

    config = SegMDTConfig.parse_arguments()
    config.task = 'MDT_LR_Range'
    _prepare_env(config)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    with open(os.path.join(config.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(config), f, indent=4)

    print('[lr_range] backbone={} fusion={} decoder={} aug={} norm={}'.format(
        config.backbone, config.fusion_type, config.decoder_type, config.aug_mode, config.norm_mode))
    train_loader = _build_train_loader(config)
    networks = build_mdt_seg_teacher(config)
    task = MDTSegTeacher(networks, config)
    for group in task.optimizer.param_groups:
        group['lr'] = float(config.lr_find_start)

    start_lr = float(config.lr_find_start)
    end_lr = float(config.lr_find_end)
    num_iter = min(int(config.lr_find_num_iter), len(train_loader))
    mult = (end_lr / start_lr) ** (1.0 / max(1, num_iter - 1))
    beta = 0.98
    avg_loss = 0.0
    best_loss = float('inf')
    clip_params = [p for net in task.networks.values() for p in net.parameters() if p.requires_grad]

    csv_path = os.path.join(config.checkpoint_dir, 'lr_range.csv')
    png_path = os.path.join(config.checkpoint_dir, 'lr_range.png')
    task.networks['model'].train()
    task.optimizer.zero_grad(set_to_none=True)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iter', 'lr', 'loss', 'smooth_loss', 'grad_norm'])
        for step, batch in enumerate(train_loader, start=1):
            if step > num_iter:
                break
            lr = start_lr * (mult ** (step - 1))
            for group in task.optimizer.param_groups:
                group['lr'] = lr

            with torch.cuda.amp.autocast(enabled=config.mixed_precision):
                loss, _, _, _ = task.train_step(batch)
            if not torch.isfinite(loss):
                print(f'[lr_range] non-finite loss at iter={step}, lr={lr:.3e}; stop')
                break

            task.optimizer.zero_grad(set_to_none=True)
            if task.scaler:
                task.scaler.scale(loss).backward()
                task.scaler.unscale_(task.optimizer)
                gn = _grad_norm(clip_params)
                task.scaler.step(task.optimizer)
                task.scaler.update()
            else:
                loss.backward()
                gn = _grad_norm(clip_params)
                task.optimizer.step()

            loss_val = float(loss.detach().item())
            avg_loss = beta * avg_loss + (1 - beta) * loss_val
            smooth = avg_loss / (1 - beta ** step)
            best_loss = min(best_loss, smooth)
            writer.writerow([step, f'{lr:.10e}', f'{loss_val:.8f}', f'{smooth:.8f}', f'{gn:.8f}'])

            if step % 10 == 0:
                print(f'[lr_range] iter={step}/{num_iter} lr={lr:.3e} loss={loss_val:.4f} smooth={smooth:.4f} grad={gn:.3f}')
            if step > 20 and smooth > best_loss * float(config.lr_find_stop_factor):
                print(f'[lr_range] loss diverged at iter={step}, lr={lr:.3e}; stop')
                break

    _save_plot(csv_path, png_path)
    print(f'[lr_range] saved CSV: {csv_path}')
    print(f'[lr_range] saved plot: {png_path}')


if __name__ == '__main__':
    main()
