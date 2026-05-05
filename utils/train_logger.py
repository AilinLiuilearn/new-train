# -*- coding: utf-8 -*-
"""训练日志：同时写入 CSV 和易读文本格式"""

import csv
import os


CSV_HEADER = [
    'epoch', 'train_loss', 'val_loss', 'val_dice', 'val_iou',
    'val_acc', 'val_acc_pixel', 'val_hd95',
]


def init_train_log(log_path):
    readable_path = _readable_path(log_path)
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)

    with open(readable_path, 'w', encoding='utf-8') as f:
        f.write('Training Log\n')
        f.write('=' * 64 + '\n')


def append_epoch_log(log_path, epoch, train_loss_avg, val_metrics):
    row = {
        'epoch': int(epoch),
        'train_loss': float(train_loss_avg),
        'val_loss': float(val_metrics['total_loss']),
        'val_dice': float(val_metrics['dice']),
        'val_iou': float(val_metrics['iou']),
        'val_acc': float(val_metrics.get('acc', 0.0)),
        'val_acc_pixel': float(val_metrics.get('acc_pixel', 0.0)),
        'val_hd95': float(val_metrics.get('hd95', 0.0)),
    }

    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            row['epoch'],
            f"{row['train_loss']:.4f}",
            f"{row['val_loss']:.4f}",
            f"{row['val_dice']:.4f}",
            f"{row['val_iou']:.4f}",
            f"{row['val_acc']:.4f}",
            f"{row['val_acc_pixel']:.4f}",
            f"{row['val_hd95']:.4f}",
        ])

    with open(_readable_path(log_path), 'a', encoding='utf-8') as f:
        f.write(f"Epoch {row['epoch']}\n")
        f.write(f"  train_loss    : {row['train_loss']:.4f}\n")
        f.write(f"  val_loss      : {row['val_loss']:.4f}\n")
        f.write(f"  val_dice      : {row['val_dice']:.4f}\n")
        f.write(f"  val_iou       : {row['val_iou']:.4f}\n")
        f.write(f"  val_acc       : {row['val_acc']:.4f}\n")
        f.write(f"  val_acc_pixel : {row['val_acc_pixel']:.4f}\n")
        f.write(f"  val_hd95      : {row['val_hd95']:.4f}\n")
        f.write('-' * 64 + '\n')


def _readable_path(log_path):
    base, _ = os.path.splitext(log_path)
    return base + '_readable.txt'
