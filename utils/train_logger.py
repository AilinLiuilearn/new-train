# -*- coding: utf-8 -*-
"""训练日志：同时写入 CSV 和易读文本格式"""

import csv
import os


CSV_HEADER = [
    'epoch', 'train_loss', 'val_loss', 'val_dice', 'val_iou',
    'val_acc', 'val_acc_pixel', 'val_hd95', 'lr', 'grad_norm',
]


def init_train_log(log_path, extra_headers=None):
    readable_path = _readable_path(log_path)
    headers = list(CSV_HEADER) + list(extra_headers or [])
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(headers)

    with open(readable_path, 'w', encoding='utf-8') as f:
        f.write('Training Log\n')
        f.write('=' * 64 + '\n')


def append_epoch_log(log_path, epoch, train_loss_avg, val_metrics, lr=None, grad_norm=None, extra_metrics=None):
    extra_metrics = extra_metrics or {}
    row = {
        'epoch': int(epoch),
        'train_loss': float(train_loss_avg),
        'val_loss': float(val_metrics['total_loss']),
        'val_dice': float(val_metrics['dice']),
        'val_iou': float(val_metrics['iou']),
        'val_acc': float(val_metrics.get('acc', 0.0)),
        'val_acc_pixel': float(val_metrics.get('acc_pixel', 0.0)),
        'val_hd95': float(val_metrics.get('hd95', 0.0)),
        'lr': float(lr) if lr is not None else 0.0,
        'grad_norm': float(grad_norm) if grad_norm is not None else 0.0,
    }

    csv_values = [
        row['epoch'],
        f"{row['train_loss']:.4f}",
        f"{row['val_loss']:.4f}",
        f"{row['val_dice']:.4f}",
        f"{row['val_iou']:.4f}",
        f"{row['val_acc']:.4f}",
        f"{row['val_acc_pixel']:.4f}",
        f"{row['val_hd95']:.4f}",
        f"{row['lr']:.8f}",
        f"{row['grad_norm']:.6f}",
    ]
    csv_values.extend(f"{float(v):.6f}" for v in extra_metrics.values())

    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(csv_values)

    with open(_readable_path(log_path), 'a', encoding='utf-8') as f:
        f.write(f"Epoch {row['epoch']}\n")
        f.write(f"  train_loss    : {row['train_loss']:.4f}\n")
        f.write(f"  val_loss      : {row['val_loss']:.4f}\n")
        f.write(f"  val_dice      : {row['val_dice']:.4f}\n")
        f.write(f"  val_iou       : {row['val_iou']:.4f}\n")
        f.write(f"  val_acc       : {row['val_acc']:.4f}\n")
        f.write(f"  val_acc_pixel : {row['val_acc_pixel']:.4f}\n")
        f.write(f"  val_hd95      : {row['val_hd95']:.4f}\n")
        f.write(f"  lr            : {row['lr']:.8f}\n")
        f.write(f"  grad_norm     : {row['grad_norm']:.6f}\n")
        for key, value in extra_metrics.items():
            f.write(f"  {key:<13}: {float(value):.6f}\n")
        f.write('-' * 64 + '\n')


def _readable_path(log_path):
    base, _ = os.path.splitext(log_path)
    return base + '_readable.txt'
