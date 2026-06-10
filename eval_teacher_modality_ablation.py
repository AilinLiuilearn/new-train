# -*- coding: utf-8 -*-
"""Evaluate teacher modality dependence by ablating CT/PET at inference time.

This script answers: does the trained PET-CT teacher actually use PET?

Example:
    python eval_teacher_modality_ablation.py \
        --teacher_dir ./checkpoints_new/MDT/2026-05-09_12-18-03 \
        --ckpt_name ckpt.best_dice.pth.tar
"""

import argparse
import importlib.util
import json
import os
import random
import sys

import numpy as np
import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from utils.metrics_seg import SegmentationMetricsCIPA
from utils.seg_losses import BCEDiceLoss

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


MODES = (
    'full',
    'pet_zero',
    'pet_shuffle',
    'pet_noise',
    'ct_zero',
    'ct_shuffle',
)


def _load_dataset_module():
    root = os.getcwd()
    dataset_path = os.path.join(root, 'datasets', 'pclt20k_seg.py')
    spec = importlib.util.spec_from_file_location('local_pclt20k_seg', dataset_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_env(config):
    g0 = int(config.gpus[0]) if getattr(config, 'gpus', None) else 0
    config.gpus = [g0]
    seed = int(getattr(config, 'random_state', 2023))
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
    return torch.device('cuda', g0) if torch.cuda.is_available() else torch.device('cpu')


def _build_loaders(config):
    dataset_mod = _load_dataset_module()
    if getattr(config, 'cipa_aligned', False):
        return dataset_mod.get_pclt20k_loaders_cipa_aligned(
            config.root,
            config.image_size_2d,
            config.batch_size,
            config.num_workers,
            config.random_state,
            pin_memory=getattr(config, 'pin_memory', True),
        )
    return dataset_mod.get_pclt20k_loaders(
        config.root,
        config.image_size_2d,
        config.batch_size,
        config.num_workers,
        val_ratio=config.val_ratio,
        random_state=config.random_state,
        use_case_split=getattr(config, 'use_case_split', True),
        pin_memory=getattr(config, 'pin_memory', True),
    )


def _select_main_pred(outputs):
    if isinstance(outputs, dict):
        if 'pred' in outputs:
            return outputs['pred']
        preds = outputs.get('preds')
        if isinstance(preds, (list, tuple)):
            return preds[0]
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


def _shuffle_batch(x):
    if x.size(0) <= 1:
        return torch.flip(x, dims=[-1])
    perm = torch.randperm(x.size(0), device=x.device)
    return x[perm]


def _apply_mode(ct, pet, mode):
    if mode == 'full':
        return ct, pet
    if mode == 'pet_zero':
        return ct, torch.zeros_like(pet)
    if mode == 'pet_shuffle':
        return ct, _shuffle_batch(pet)
    if mode == 'pet_noise':
        return ct, torch.randn_like(pet)
    if mode == 'ct_zero':
        return torch.zeros_like(ct), pet
    if mode == 'ct_shuffle':
        return _shuffle_batch(ct), pet
    raise ValueError(f'Unknown mode: {mode}')


@torch.no_grad()
def evaluate_mode(model, loader, loss_fn, device, mode, threshold=0.5):
    model.eval()
    metrics = SegmentationMetricsCIPA(threshold=threshold).to(device)
    metrics.reset()
    total_loss, n = 0.0, 0

    for batch in loader:
        ct = batch['ct'].float().to(device)
        pet = batch['pet'].float().to(device)
        mask = batch['mask'].float().to(device)
        ct_in, pet_in = _apply_mode(ct, pet, mode)
        outputs = model(ct_in, pet_in, target_size=mask.shape[-2:])
        pred = _select_main_pred(outputs)
        loss, _ = loss_fn(pred, mask)
        total_loss += loss.item() * ct.size(0)
        n += ct.size(0)
        metrics.update(pred, mask)

    out = metrics.compute()
    out['total_loss'] = total_loss / max(n, 1)
    return out


def _load_config(teacher_dir):
    cfg_path = os.path.join(teacher_dir, 'config_args.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f'Config not found: {cfg_path}')
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    cfg.pop('checkpoint_dir', None)
    config = SegMDTConfig(args=cfg)
    config.task = 'MDT_Teacher'
    return config


def _load_model(config, ckpt_path, device):
    networks = build_mdt_seg_teacher(config)
    model = networks['model'].to(device)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
        print(f'[+] Loaded model from {ckpt_path}')
    else:
        model.load_state_dict(ckpt, strict=False)
        print(f'[+] Loaded raw state_dict from {ckpt_path}')
    model.eval()
    return model


def _print_results(results):
    full = results.get('full')
    print('\n' + '=' * 92)
    print('Teacher modality ablation results')
    print('=' * 92)
    print(f'{"mode":14s} {"loss":>8s} {"dice":>8s} {"iou":>8s} {"acc":>8s} {"hd95":>10s} {"dice_drop":>11s}')
    print('-' * 92)
    for mode, m in results.items():
        drop = full['dice'] - m['dice'] if full is not None else 0.0
        print(
            f'{mode:14s} '
            f'{m["total_loss"]:8.4f} '
            f'{m["dice"]:8.4f} '
            f'{m["iou"]:8.4f} '
            f'{m["acc"]:8.4f} '
            f'{m["hd95"]:10.2f} '
            f'{drop:11.4f}'
        )
    print('=' * 92)


def _save_results(results, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('mode,total_loss,dice,iou,acc,acc_pixel,hd95,dice_drop\n')
        full_dice = results['full']['dice'] if 'full' in results else 0.0
        for mode, m in results.items():
            f.write(
                f'{mode},{m["total_loss"]:.6f},{m["dice"]:.6f},{m["iou"]:.6f},'
                f'{m["acc"]:.6f},{m["acc_pixel"]:.6f},{m["hd95"]:.6f},'
                f'{full_dice - m["dice"]:.6f}\n'
            )
    print(f'[+] Saved results to {out_path}')


def parse_args():
    p = argparse.ArgumentParser('Teacher modality ablation')
    p.add_argument('--teacher_dir', type=str, required=True)
    p.add_argument('--ckpt_name', type=str, default='ckpt.best_dice.pth.tar')
    p.add_argument('--split', type=str, default='test', choices=('val', 'test'))
    p.add_argument('--modes', type=str, nargs='+', default=list(MODES), choices=MODES)
    p.add_argument('--out_name', type=str, default='modality_ablation.csv')
    return p.parse_args()


def main():
    if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.getcwd())
    sys.modules.pop('datasets', None)

    args = parse_args()
    config = _load_config(args.teacher_dir)
    device = _prepare_env(config)
    ckpt_path = os.path.join(args.teacher_dir, args.ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    print(f'[+] Teacher dir: {args.teacher_dir}')
    print(f'[+] Checkpoint : {ckpt_path}')
    print(f'[+] Backbone   : {getattr(config, "backbone", None)}')
    print(f'[+] use_tcpm   : {getattr(config, "use_tcpm", None)}')
    print(f'[+] root       : {getattr(config, "root", None)}')

    _, val_loader, test_loader = _build_loaders(config)
    loader = val_loader if args.split == 'val' else test_loader
    model = _load_model(config, ckpt_path, device)
    loss_fn = BCEDiceLoss(
        bce_weight=getattr(config, 'bce_weight', 1.0),
        dice_weight=getattr(config, 'dice_weight', 1.0),
        smooth=getattr(config, 'loss_smooth', 1.0),
        pos_weight=getattr(config, 'pos_weight', None),
    ).to(device)

    results = {}
    for mode in args.modes:
        print(f'[*] Evaluating mode: {mode}')
        results[mode] = evaluate_mode(
            model, loader, loss_fn, device, mode,
            threshold=getattr(config, 'eval_threshold', 0.5),
        )

    _print_results(results)
    _save_results(results, os.path.join(args.teacher_dir, args.out_name))


if __name__ == '__main__':
    main()
