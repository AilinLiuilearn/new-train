# -*- coding: utf-8 -*-
"""训练日志：写入 checkpoint_dir/train_log.csv，便于观察收敛"""

import os
import csv


def init_train_log(log_path):
    """初始化日志文件，写入表头"""
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            'epoch', 'train_loss', 'val_loss', 'val_dice', 'val_iou',
            'val_acc', 'val_acc_pixel', 'val_hd95',
        ])


def append_epoch_log(log_path, epoch, train_loss_avg, val_metrics):
    """追加一行 epoch 日志"""
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            epoch,
            f'{train_loss_avg:.4f}',
            f"{val_metrics['total_loss']:.4f}",
            f"{val_metrics['dice']:.4f}",
            f"{val_metrics['iou']:.4f}",
            f"{val_metrics.get('acc', 0):.4f}",
            f"{val_metrics.get('acc_pixel', 0):.4f}",
            f"{val_metrics.get('hd95', 0):.4f}",
        ])
