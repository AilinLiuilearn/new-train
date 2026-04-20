# -*- coding: utf-8 -*-
"""
MDT+ 阶段入口：增强教师模型。学生向教师的 CT 分支传递知识。
- 训练对象：教师（学生冻结）
- 数据：配对双模态（complete），使用与教师/学生相同的 missing_rate，保持现实缺失率
用法：
  python run_mdt_plus_seg.py --root ../data/PCLT20K --teacher_ckpt .../ckpt.best.pth.tar --student_ckpt .../ckpt.best.pth.tar --epochs 30 [--missing_rate 0.3]
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
from utils.model_profile import print_model_profile
from utils.train_logger import init_train_log, append_epoch_log
from utils.train_config import (
    save_full_config, build_model_summary, build_data_summary,
    _get_flops_params, update_config_with_test_results, print_test_results,
)
from datasets.pclt20k_seg import get_pclt20k_loaders
from models.build_mdt_seg import build_mdt_seg_teacher, build_mdt_seg_student
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
    print("use_cmx (教师浅层 FSF):", getattr(config, 'use_cmx', True))
    set_gpu(config)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_state)

    missing_rate = getattr(config, 'missing_rate', 0.3)
    val_ratio = getattr(config, 'val_ratio', 0.1)
    use_case_split = getattr(config, 'use_case_split', True)
    train_loader, val_loader, test_loader = get_pclt20k_loaders(
        config.root,
        image_size=config.image_size_2d,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        missing_rate=missing_rate,
        val_ratio=val_ratio,
        random_state=config.random_state,
        use_case_split=use_case_split,
    )
    print(f"MDT+ 使用 complete（配对）数据，missing_rate={missing_rate} 与教师/学生一致")

    teacher_networks = build_mdt_seg_teacher(config)
    ckpt_t = torch.load(teacher_ckpt, map_location='cpu')
    for k, v in teacher_networks.items():
        if k in ckpt_t:
            v.load_state_dict(ckpt_t[k], strict=False)
    student_networks = build_mdt_seg_student(config)
    ckpt_s = torch.load(student_ckpt, map_location='cpu')
    for k, v in student_networks.items():
        if k in ckpt_s:
            v.load_state_dict(ckpt_s[k], strict=False)

    print_model_profile("MDT+ 教师", teacher_networks, config, is_teacher=True)
    params_m, flops_g = _get_flops_params(teacher_networks, config, is_teacher=True)
    task = MDTSegPlus(teacher_networks, student_networks, config)
    steps_per_epoch = len(train_loader)
    task.scheduler = get_cosine_scheduler(
        task.optimizer, config.epochs,
        warmup_steps=config.cosine_warmup * steps_per_epoch,
        min_lr=config.cosine_min_lr,
        steps_per_epoch=steps_per_epoch,
    )
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    extras = {
        'model_summary': build_model_summary('MDT_Plus', config, is_teacher=True),
        'data_split': build_data_summary(train_loader, val_loader, test_loader, 'MDT_Plus', missing_rate=missing_rate),
        'params_m': round(params_m, 2),
        'flops_g': round(flops_g, 2) if flops_g is not None else None,
        'teacher_ckpt': teacher_ckpt,
        'student_ckpt': student_ckpt,
    }
    save_full_config(config, extras)
    log_path = os.path.join(config.checkpoint_dir, 'train_log.csv')
    init_train_log(log_path)

    best_dice = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    early_stop_patience = getattr(config, 'early_stop_patience', 20)
    for epoch in range(1, config.epochs + 1):
        train_loss_sum = 0.0
        train_n = 0
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
            train_loss_sum += total_loss.item()
            train_n += 1
            if (i + 1) % 50 == 0:
                print(f"Epoch {epoch} [{i+1}/{len(train_loader)}] loss={total_loss.item():.4f} seg={loss_dict['loss_seg'].item():.4f} kd_repr={loss_dict['loss_kd_repr'].item():.4f}")
        train_loss_avg = train_loss_sum / max(train_n, 1)
        val_metrics = task.evaluate(val_loader)
        append_epoch_log(log_path, epoch, train_loss_avg, val_metrics)
        print(f"Epoch {epoch} train_loss={train_loss_avg:.4f} val_loss={val_metrics['total_loss']:.4f} Dice={val_metrics['dice']:.4f} IoU={val_metrics['iou']:.4f}")
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_epoch = epoch
            epochs_no_improve = 0
            task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.best.pth.tar'), epoch)
        else:
            epochs_no_improve += 1
        if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
            print(f"早停：验证 Dice 连续 {early_stop_patience} 轮无提升，停止训练 (epoch {epoch})")
            break
    task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.last.pth.tar'), epoch)
    print("最佳 epoch:", best_epoch, "Val Dice:", best_dice)
    print("训练日志:", log_path)
    ckpt = torch.load(os.path.join(config.checkpoint_dir, 'ckpt.best.pth.tar'), map_location='cpu')
    for k, v in task.networks.items():
        if k in ckpt:
            v.load_state_dict(ckpt[k], strict=False)
    test_metrics = task.evaluate(test_loader)
    update_config_with_test_results(config, test_metrics, best_epoch, best_dice)
    print_test_results(test_metrics, 'MDT_Plus', config.checkpoint_dir)


if __name__ == '__main__':
    main()
