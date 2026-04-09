# -*- coding: utf-8 -*-
"""
MDT+ 阶段入口：学生反哺教师，训练教师 CT 分支与学生一致。
用法（在 new-train 目录下）：
  python run_mdt_plus_seg.py --root ../data/PCLT20K --teacher_ckpt path/to/teacher/ckpt.best.pth.tar --student_ckpt path/to/student/ckpt.best.pth.tar --epochs 30
  教师/学生权重通常位于 ./checkpoints_new/MDT/<hash>/ 与 ./checkpoints_new/MDT-student/<hash>/
  python run_mdt_plus_seg.py --root ../../data/PCLT20K  --teacher_ckpt ./checkpoints_new/MDT/2026-03-02_21-39-15/ckpt.best.pth.tar --student_ckpt ./checkpoints_new/MDT-student/2026-03-03_09-47-57/ckpt.best.pth.tar --epochs 30
"""

import os
import sys

if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch
import numpy as np

from configs.seg_mdt import SegMDTConfig
from utils.optimization import get_cosine_scheduler
from datasets.pclt20k_seg import get_pclt20k_loaders
from models.build_mdt_seg import build_mdt_seg_teacher_plus, build_mdt_seg_student
from tasks.mdt_plus_seg import MDTSegPlus


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
    config.task = 'MDT_Plus'
    if not getattr(config, 'root', None):
        config.root = '../data/PCLT20K'
    teacher_ckpt = getattr(config, 'teacher_ckpt', None)
    student_ckpt = getattr(config, 'student_ckpt', None)
    if not teacher_ckpt or not os.path.isfile(teacher_ckpt):
        raise FileNotFoundError("请指定教师权重: --teacher_ckpt path/to/ckpt.best.pth.tar")
    if not student_ckpt or not os.path.isfile(student_ckpt):
        raise FileNotFoundError("请指定学生权重: --student_ckpt path/to/ckpt.best.pth.tar")
    print("数据根目录:", config.root, "教师:", teacher_ckpt, "学生:", student_ckpt)
    set_gpu(config)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_state)

    train_loader, val_loader, test_loader = get_pclt20k_loaders(
        config.root,
        image_size=config.image_size_2d,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        missing_rate=0.0,
        random_state=config.random_state,
    )
    teacher_networks = build_mdt_seg_teacher_plus(config)
    ckpt_t = torch.load(teacher_ckpt, map_location='cpu')
    for k, v in teacher_networks.items():
        if k in ckpt_t and k != 'segmentor_ct':
            v.load_state_dict(ckpt_t[k])
    student_networks = build_mdt_seg_student(config)
    ckpt_s = torch.load(student_ckpt, map_location='cpu')
    for k, v in student_networks.items():
        if k in ckpt_s:
            v.load_state_dict(ckpt_s[k])
    task = MDTSegPlus(teacher_networks, student_networks, config)
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
                total_loss, logit_t, mask, loss_dict = task.train_step(batch)
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
                print(f"Epoch {epoch} [{i+1}/{len(train_loader)}] loss={total_loss.item():.4f} seg={loss_dict['loss_seg'].item():.4f} cons={loss_dict['loss_cons'].item():.4f}")
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
            print(f"早停：验证 Dice 连续 {early_stop_patience} 轮无提升，停止训练 (epoch {epoch})")
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

    
    # 1) 教师
# python run_mdt_seg.py --root ../data/PCLT20K --epochs 50 --batch_size 8

# # 2) 学生（需先有教师 ckpt）
# python run_mdt_student_seg.py --root ../data/PCLT20K --teacher_ckpt ./checkpoints_new/MDT/<hash>/ckpt.best.pth.tar --epochs 50
# 学生 best: ./checkpoints_new/MDT-student/<hash>/ckpt.best.pth.tar

# # 3) MDT+
# python run_mdt_plus_seg.py --root ../data/PCLT20K --teacher_ckpt .../MDT/.../ckpt.best.pth.tar --student_ckpt .../MDT-student/.../ckpt.best.pth.tar --epochs 30