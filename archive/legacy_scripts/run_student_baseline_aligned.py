#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train baseline student (aligned to KD data usage):
- Use the same data loaders as KD student (`get_pclt20k_loaders_student`).
- use_full_paired=False: steps=min(paired,mri)，与 KD 学生原逻辑一致
- use_full_paired=True: steps=len(paired)，用满配对数据，mri 循环

Usage:
  python run_student_baseline_aligned.py --root /path/to/PCLT20K --epochs 50 --batch_size 16 --missing_rate 0.3
  python run_student_baseline_aligned.py --root /path/to/PCLT20K --use_full_paired True  # 用满配对
"""

import os
import sys
import itertools

if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch
import numpy as np

from configs.seg_mdt import SegMDTConfig
from utils.optimization import get_cosine_scheduler, get_optimizer
from utils.model_profile import print_model_profile
from utils.train_logger import init_train_log, append_epoch_log
from utils.loss_seg import DiceBCELoss
from utils.train_config import (
    save_full_config, build_model_summary, build_data_summary,
    _get_flops_params, update_config_with_test_results, print_test_results,
)
from datasets.pclt20k_seg import get_pclt20k_loaders_student
from models.build_mdt_seg import build_mdt_seg_student


def set_gpu(config):
    gpus = getattr(config, 'gpus', ['0'])
    if gpus is None or len(gpus) == 0:
        gpus = [0]
    g0 = gpus[0]
    if isinstance(g0, str):
        g0 = int(g0)
    config.gpus = [g0]
    if torch.cuda.is_available():
        torch.cuda.set_device(g0)
    print("使用 GPU:", config.gpus)


def student_forward(networks, ct, target_size, device, config):
    feats = networks['extractor'](ct, return_list=True)
    h = feats[-1]
    if networks.get('projector') is not None:
        h = networks['projector'](h)
    z_general = networks['encoder_general'](h)
    z_mri = networks['encoder_mri'](h)
    if getattr(config, 'use_specific', True):
        fusion_s = torch.cat([z_general, z_mri], dim=1)
    else:
        fusion_s = z_general
    fpn_input_list = [feats[0], feats[1], feats[2], fusion_s]
    logit_s = networks['segmentor'](fpn_input_list, target_size=target_size)
    return logit_s


def evaluate(networks, loader, device, config):
    for v in networks.values():
        v.eval()
    from utils.metrics_seg import SegmentationMetricsCIPA
    seg_metrics = SegmentationMetricsCIPA(threshold=0.5).to(device)
    loss_fn = DiceBCELoss(
        smooth=getattr(config, 'dice_smooth', 1.0),
        pos_weight=getattr(config, 'pos_weight', None),
        bce_weight=getattr(config, 'bce_weight', 1.0),
        dice_weight=getattr(config, 'dice_weight', 1.0),
    )
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            ct = batch['ct'].float().to(device)
            mask = batch['mask'].float().to(device)
            logit_s = student_forward(networks, ct, mask.shape[-2:], device, config)
            loss_seg = loss_fn(logit_s, mask)
            total_loss += loss_seg.item() * ct.size(0)
            n += ct.size(0)
            seg_metrics.update(logit_s, mask)
    metrics = seg_metrics.compute()
    metrics['total_loss'] = total_loss / max(n, 1)
    for v in networks.values():
        v.train()
    return metrics


def main():
    config = SegMDTConfig.parse_arguments()
    config.task = 'Student_Baseline_Aligned'
    if not getattr(config, 'root', None):
        config.root = '../data/PCLT20K'
    print("数据根目录:", config.root)
    set_gpu(config)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_state)

    missing_rate = getattr(config, 'missing_rate', 0.3)
    val_ratio = getattr(config, 'val_ratio', 0.1)
    use_case_split = getattr(config, 'use_case_split', True)
    train_paired_loader, train_mri_loader, val_loader, test_loader = get_pclt20k_loaders_student(
        config.root,
        image_size=config.image_size_2d,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        missing_rate=missing_rate,
        val_ratio=val_ratio,
        random_state=config.random_state,
        use_case_split=use_case_split,
    )
    use_full_paired = getattr(config, 'use_full_paired', False)
    if train_paired_loader is not None and train_mri_loader is not None:
        if use_full_paired:
            steps_per_epoch = len(train_paired_loader)
            print("Baseline [use_full_paired=True] steps_per_epoch:", steps_per_epoch, "(用满配对，mri 循环)")
        else:
            steps_per_epoch = min(len(train_paired_loader), len(train_mri_loader))
            print("Baseline [use_full_paired=False] steps_per_epoch:", steps_per_epoch, "(min 模式)")
    elif train_paired_loader is not None:
        steps_per_epoch = len(train_paired_loader)
    elif train_mri_loader is not None:
        steps_per_epoch = len(train_mri_loader)
    else:
        raise RuntimeError("配对与缺失 PET 样本数均为 0，请设置 0 < missing_rate < 1")

    student_networks = build_mdt_seg_student(config)
    device = torch.device('cuda', config.gpus[0] if isinstance(config.gpus[0], int) else int(config.gpus[0]))
    for v in student_networks.values():
        v.to(device)
    print_model_profile("学生(baseline)", student_networks, config, is_teacher=False)
    params_m, flops_g = _get_flops_params(student_networks, config, is_teacher=False)
    if flops_g is not None:
        print(f"[学生 baseline] 计算复杂度: Params={params_m:.2f}M, FLOPs={flops_g:.2f}G")

    loss_fn = DiceBCELoss(
        smooth=getattr(config, 'dice_smooth', 1.0),
        pos_weight=getattr(config, 'pos_weight', None),
        bce_weight=getattr(config, 'bce_weight', 1.0),
        dice_weight=getattr(config, 'dice_weight', 1.0),
    )

    # build optimizer similar to MDTSegStudent._build_optimizer
    params = []
    for name, net in student_networks.items():
        lr = config.learning_rate * 0.1 if name.startswith(('encoder_', 'segmentor')) else config.learning_rate
        params.append({'params': net.parameters(), 'lr': lr})
    optimizer = get_optimizer(params, config.optimizer, config.learning_rate, config.weight_decay)
    scheduler = get_cosine_scheduler(
        optimizer, config.epochs,
        warmup_steps=config.cosine_warmup * steps_per_epoch,
        min_lr=config.cosine_min_lr,
        steps_per_epoch=steps_per_epoch,
    )
    scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    extras = {
        'model_summary': build_model_summary('Student_Baseline_Aligned', config, is_teacher=False),
        'data_split': build_data_summary(None, val_loader, test_loader, 'Student_Baseline_Aligned',
                                         train_paired_loader, train_mri_loader, missing_rate),
        'params_m': round(params_m, 2),
        'flops_g': round(flops_g, 2) if flops_g is not None else None,
    }
    save_full_config(config, extras)
    log_path = os.path.join(config.checkpoint_dir, 'train_log_baseline_aligned.csv')
    init_train_log(log_path)

    best_dice = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    early_stop_patience = getattr(config, 'early_stop_patience', 15)
    grad_clip = getattr(config, 'grad_clip', 5.0)
    print(f"早停机制: patience={early_stop_patience} (0 表示不早停)")

    for epoch in range(1, config.epochs + 1):
        train_loss_sum = 0.0
        train_n = 0
        if train_paired_loader is not None and train_mri_loader is not None:
            paired_iter = iter(train_paired_loader)
            mri_iter = itertools.cycle(iter(train_mri_loader)) if use_full_paired else iter(train_mri_loader)
            for i in range(steps_per_epoch):
                batch = next(paired_iter)
                batch_mri = next(mri_iter)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=config.mixed_precision):
                    ct_p = batch['ct'].float().to(device)
                    mask_p = batch['mask'].float().to(device)
                    logit_p = student_forward(student_networks, ct_p, mask_p.shape[-2:], device, config)
                    loss_seg_p = loss_fn(logit_p, mask_p)
                    ct_m = batch_mri['ct'].float().to(device)
                    mask_m = batch_mri['mask'].float().to(device)
                    logit_m = student_forward(student_networks, ct_m, mask_m.shape[-2:], device, config)
                    loss_seg_m = loss_fn(logit_m, mask_m)
                    n_p, n_m = ct_p.size(0), ct_m.size(0)
                    n_total = n_p + n_m
                    total_loss = config.alpha_seg * (loss_seg_p * (n_p / n_total) + loss_seg_m * (n_m / n_total))
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        params_all = [p for n in student_networks.values() for p in n.parameters()]
                        torch.nn.utils.clip_grad_norm_(params_all, grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    if grad_clip > 0:
                        params_all = [p for n in student_networks.values() for p in n.parameters()]
                        torch.nn.utils.clip_grad_norm_(params_all, grad_clip)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                train_loss_sum += total_loss.item()
                train_n += 1
                if (i + 1) % 50 == 0:
                    print(f"Epoch {epoch} [{i+1}/{steps_per_epoch}] loss={total_loss.item():.4f}")
        elif train_paired_loader is not None:
            for i, batch in enumerate(train_paired_loader):
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=config.mixed_precision):
                    ct_p = batch['ct'].float().to(device)
                    mask_p = batch['mask'].float().to(device)
                    total_loss = config.alpha_seg * loss_fn(student_forward(student_networks, ct_p, mask_p.shape[-2:], device, config), mask_p)
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        params_all = [p for n in student_networks.values() for p in n.parameters()]
                        torch.nn.utils.clip_grad_norm_(params_all, grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    if grad_clip > 0:
                        params_all = [p for n in student_networks.values() for p in n.parameters()]
                        torch.nn.utils.clip_grad_norm_(params_all, grad_clip)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                train_loss_sum += total_loss.item()
                train_n += 1
                if (i + 1) % 50 == 0:
                    print(f"Epoch {epoch} [{i+1}/{len(train_paired_loader)}] loss={total_loss.item():.4f}")
        else:
            for i, batch_mri in enumerate(train_mri_loader):
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=config.mixed_precision):
                    ct_m = batch_mri['ct'].float().to(device)
                    mask_m = batch_mri['mask'].float().to(device)
                    total_loss = config.alpha_seg * loss_fn(student_forward(student_networks, ct_m, mask_m.shape[-2:], device, config), mask_m)
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        params_all = [p for n in student_networks.values() for p in n.parameters()]
                        torch.nn.utils.clip_grad_norm_(params_all, grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    if grad_clip > 0:
                        params_all = [p for n in student_networks.values() for p in n.parameters()]
                        torch.nn.utils.clip_grad_norm_(params_all, grad_clip)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                train_loss_sum += total_loss.item()
                train_n += 1
                if (i + 1) % 50 == 0:
                    print(f"Epoch {epoch} [{i+1}/{len(train_mri_loader)}] loss={total_loss.item():.4f}")

        train_loss_avg = train_loss_sum / max(train_n, 1)
        val_metrics = evaluate(student_networks, val_loader, device, config)
        append_epoch_log(log_path, epoch, train_loss_avg, val_metrics)
        print(f"Epoch {epoch} train_loss={train_loss_avg:.4f} val_loss={val_metrics['total_loss']:.4f} Dice={val_metrics['dice']:.4f} IoU={val_metrics['iou']:.4f}")
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_epoch = epoch
            epochs_no_improve = 0
            ckpt = {k: v.state_dict() for k, v in student_networks.items()}
            ckpt['epoch'] = epoch
            ckpt['optimizer'] = optimizer.state_dict()
            ckpt_path = os.path.join(config.checkpoint_dir, 'ckpt.best.pth.tar')
            torch.save(ckpt, ckpt_path)
            print(f"  >>> 保存最优权重 (Dice={best_dice:.4f}): {ckpt_path}")
        else:
            epochs_no_improve += 1
        if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
            print(f"早停：验证 Dice 连续 {early_stop_patience} 轮无提升，停止(epoch {epoch})")
            break

    ckpt = {k: v.state_dict() for k, v in student_networks.items()}
    ckpt['epoch'] = epoch
    ckpt['optimizer'] = optimizer.state_dict()
    torch.save(ckpt, os.path.join(config.checkpoint_dir, 'ckpt.last.pth.tar'))
    print("最佳 epoch:", best_epoch, "Val Dice:", best_dice)
    print("训练日志:", log_path)

    ckpt = torch.load(os.path.join(config.checkpoint_dir, 'ckpt.best.pth.tar'), map_location='cpu')
    for k, v in student_networks.items():
        if k in ckpt:
            v.load_state_dict(ckpt[k], strict=False)
    test_metrics = evaluate(student_networks, test_loader, device, config)
    update_config_with_test_results(config, test_metrics, best_epoch, best_dice)
    print_test_results(test_metrics, 'Student_Baseline', config.checkpoint_dir)


if __name__ == '__main__':
    main()
