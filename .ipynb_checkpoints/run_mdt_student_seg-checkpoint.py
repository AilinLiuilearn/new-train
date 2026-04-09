# -*- coding: utf-8 -*-
"""
MDT 学生阶段入口（对齐 mkd：https://github.com/min9kwak/mkd）：
- 双 DataLoader：train_paired（有 PET）、train_mri（缺失 PET）；每步 zip(paired, mri) 做蒸馏+seg 与仅 seg。
- 缺失率：--missing_rate 为训练集上模拟 PET 缺失比例，建议 0 < missing_rate < 1。
用法：
  python run_mdt_student_seg.py --root ../data/PCLT20K --teacher_ckpt path/to/ckpt.best.pth.tar --epochs 50 [--missing_rate 0.3]
  教师权重通常位于 ./checkpoints_new/MDT/<hash>/ckpt.best.pth.tar，学生保存到 ./checkpoints_new/MDT-student/<hash>/
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
from datasets.pclt20k_seg import get_pclt20k_loaders_student
from models.build_mdt_seg import build_mdt_seg_teacher, build_mdt_seg_student
from tasks.mdt_student_seg import MDTSegStudent


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
    config.task = 'MDT_Student'
    if not getattr(config, 'root', None):
        config.root = '../data/PCLT20K'
    teacher_ckpt = getattr(config, 'teacher_ckpt', None)
    if not teacher_ckpt or not os.path.isfile(teacher_ckpt):
        raise FileNotFoundError("请指定已训练教师权重: --teacher_ckpt path/to/ckpt.best.pth.tar")
    print("数据根目录:", config.root, "教师权重:", teacher_ckpt)
    set_gpu(config)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_state)

    train_paired_loader, train_mri_loader, val_loader, test_loader = get_pclt20k_loaders_student(
        config.root,
        image_size=config.image_size_2d,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        missing_rate=getattr(config, 'missing_rate', 0.3),
        random_state=config.random_state,
    )
    # 步数对齐 mkd：取 min(paired, mri)；若一侧为空则用另一侧长度
    if train_paired_loader is not None and train_mri_loader is not None:
        steps_per_epoch = min(len(train_paired_loader), len(train_mri_loader))
    elif train_paired_loader is not None:
        steps_per_epoch = len(train_paired_loader)
    elif train_mri_loader is not None:
        steps_per_epoch = len(train_mri_loader)
    else:
        raise RuntimeError("配对与缺失 PET 样本数均为 0，请设置 0 < missing_rate < 1")
    print("学生阶段 steps_per_epoch:", steps_per_epoch)
    teacher_networks = build_mdt_seg_teacher(config)
    ckpt = torch.load(teacher_ckpt, map_location='cpu')
    for k, v in teacher_networks.items():
        if k in ckpt:
            v.load_state_dict(ckpt[k])
    student_networks = build_mdt_seg_student(config)
    task = MDTSegStudent(student_networks, teacher_networks, config)
    task.scheduler = get_cosine_scheduler(
        task.optimizer, config.epochs,
        warmup_steps=config.cosine_warmup * steps_per_epoch,
        min_lr=config.cosine_min_lr,
        steps_per_epoch=steps_per_epoch,
    )
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    config.save()

    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        if train_paired_loader is not None and train_mri_loader is not None:
            paired_iter, mri_iter = iter(train_paired_loader), iter(train_mri_loader)
            for i in range(steps_per_epoch):
                batch = next(paired_iter)
                batch_mri = next(mri_iter)
                task.optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=config.mixed_precision):
                    total_loss, logit_s, mask, loss_dict = task.train_step(batch, batch_mri)
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
                    ls = loss_dict.get('loss_seg', torch.tensor(0., device=task.device))
                    lf = loss_dict.get('loss_feat', torch.tensor(0., device=task.device))
                    ll = loss_dict.get('loss_logit', torch.tensor(0., device=task.device))
                    print(f"Epoch {epoch} [{i+1}/{steps_per_epoch}] loss={total_loss.item():.4f} seg={ls.item():.4f} feat={lf.item():.4f} logit={ll.item():.4f}")
        elif train_paired_loader is not None:
            for i, batch in enumerate(train_paired_loader):
                task.optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=config.mixed_precision):
                    total_loss, logit_s, mask, loss_dict = task.train_step(batch_paired=batch, batch_mri=None)
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
                    ls = loss_dict.get('loss_seg', torch.tensor(0., device=task.device))
                    lf = loss_dict.get('loss_feat', torch.tensor(0., device=task.device))
                    ll = loss_dict.get('loss_logit', torch.tensor(0., device=task.device))
                    print(f"Epoch {epoch} [{i+1}/{len(train_paired_loader)}] loss={total_loss.item():.4f} seg={ls.item():.4f} feat={lf.item():.4f} logit={ll.item():.4f}")
        else:
            for i, batch_mri in enumerate(train_mri_loader):
                task.optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=config.mixed_precision):
                    total_loss, logit_s, mask, loss_dict = task.train_step(batch_paired=None, batch_mri=batch_mri)
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
                    ls = loss_dict.get('loss_seg', torch.tensor(0., device=task.device))
                    print(f"Epoch {epoch} [{i+1}/{len(train_mri_loader)}] loss={total_loss.item():.4f} seg={ls.item():.4f}")
        val_metrics = task.evaluate(val_loader)
        print(f"Epoch {epoch} Val Loss={val_metrics['total_loss']:.4f} Dice={val_metrics['dice']:.4f} IoU={val_metrics['iou']:.4f}")
        if val_metrics['total_loss'] < best_loss:
            best_loss = val_metrics['total_loss']
            best_epoch = epoch
            task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.best.pth.tar'), epoch)
        if epoch % config.save_every == 0:
            task.save_checkpoint(os.path.join(config.checkpoint_dir, f'ckpt.{epoch}.pth.tar'), epoch)
    task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.last.pth.tar'), config.epochs)
    print("最佳 epoch:", best_epoch)
    ckpt = torch.load(os.path.join(config.checkpoint_dir, 'ckpt.best.pth.tar'), map_location='cpu')
    for k, v in task.networks.items():
        if k in ckpt:
            v.load_state_dict(ckpt[k])
    test_metrics = task.evaluate(test_loader)
    print("Test Dice:", test_metrics['dice'], "IoU:", test_metrics['iou'])


if __name__ == '__main__':
    main()
