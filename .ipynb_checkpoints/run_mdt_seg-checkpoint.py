# -*- coding: utf-8 -*-
"""
MDT 教师阶段入口：2D CT+PET 双模态分割，完整 PET。
用法（在 new-train 目录下）：
  python run_mdt_seg.py --root ../data/PCLT20K --epochs 50 --batch_size 8
"""

import os
import sys
import argparse

# 保证 new-train 为当前目录，便于 import
if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch
import numpy as np

from configs.seg_mdt import SegMDTConfig
from utils.optimization import get_optimizer, get_cosine_scheduler
from datasets.pclt20k_seg import get_pclt20k_loaders
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher


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


def main():
    config = SegMDTConfig.parse_arguments()
    config.task = 'MDT'
    if not getattr(config, 'root', None):
        config.root = '../data/PCLT20K'
    print("数据根目录:", config.root)
    set_gpu(config)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_state)

    # 教师阶段：missing_rate=0，仅用双模态；val_ratio 从 train 分出验证集
    train_loader, val_loader, test_loader = get_pclt20k_loaders(
        config.root,
        image_size=config.image_size_2d,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        missing_rate=0.0,
        val_ratio=getattr(config, 'val_ratio', 0.1),
        random_state=config.random_state,
    )
    networks = build_mdt_seg_teacher(config)
    task = MDTSegTeacher(networks, config)
    steps_per_epoch = len(train_loader)
    task.scheduler = get_cosine_scheduler(
        task.optimizer, config.epochs,
        warmup_steps=config.cosine_warmup * steps_per_epoch,
        min_lr=config.cosine_min_lr,
        steps_per_epoch=steps_per_epoch,
    )
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    config.save()

    best_dice = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    early_stop_patience = getattr(config, 'early_stop_patience', 20)
    for epoch in range(1, config.epochs + 1):
        for i, batch in enumerate(train_loader):
            task.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=config.mixed_precision):
                total_loss, seg_logit, mask, loss_dict = task.train_step(batch)
            if task.scaler is not None:
                task.scaler.scale(total_loss).backward()
                task.scaler.step(task.optimizer)
                task.scaler.update()
            else:
                total_loss.backward()
                task.optimizer.step()
            if task.scheduler is not None:
                task.scheduler.step()
            if (i + 1) % 50 == 0:
                print(f"Epoch {epoch} [{i+1}/{len(train_loader)}] loss={total_loss.item():.4f} seg={loss_dict['loss_seg'].item():.4f}")
        val_metrics = task.evaluate(val_loader)
        print(f"Epoch {epoch} Val Loss={val_metrics['total_loss']:.4f} Dice={val_metrics['dice']:.4f} IoU={val_metrics['iou']:.4f}")
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_epoch = epoch
            epochs_no_improve = 0
            task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.best.pth.tar'), epoch)
        else:
            epochs_no_improve += 1
        if epoch % config.save_every == 0:
            task.save_checkpoint(os.path.join(config.checkpoint_dir, f'ckpt.{epoch}.pth.tar'), epoch)
        if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
            print(f"早停：验证 Dice 连续 {early_stop_patience} 轮无提升，停止(epoch {epoch})")
            break
    task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.last.pth.tar'), epoch)
    print("最佳 epoch:", best_epoch, "Val Dice:", best_dice)
    ckpt = torch.load(os.path.join(config.checkpoint_dir, 'ckpt.best.pth.tar'), map_location='cpu')
    for k, v in task.networks.items():
        if k in ckpt:
            v.load_state_dict(ckpt[k])
    test_metrics = task.evaluate(test_loader)
    print("Test Dice:", test_metrics['dice'], "IoU:", test_metrics['iou'])


if __name__ == '__main__':
    main()
