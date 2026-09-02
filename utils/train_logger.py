# -*- coding: utf-8 -*-
"""训练日志：同时写入 CSV 和易读文本格式"""

import csv
import os


CSV_HEADER = [
    'epoch', 'train_loss', 'val_loss', 'val_dice', 'val_iou',
    'val_acc', 'val_acc_pixel', 'val_hd95', 'lr', 'grad_norm',
]

_LOG_HEADERS = {}


def init_train_log(log_path, extra_headers=None):
    readable_path = _readable_path(log_path)
    headers = list(CSV_HEADER) + list(extra_headers or [])
    _LOG_HEADERS[log_path] = headers
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(headers)

    with open(readable_path, 'w', encoding='utf-8') as f:
        f.write('Training Log\n')
        f.write('=' * 64 + '\n')


def _read_csv_headers(log_path):
    with open(log_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        return next(reader)


def append_epoch_log(log_path, epoch, train_loss_avg, val_metrics, lr=None, grad_norm=None, extra_metrics=None):
    extra_metrics = extra_metrics or {}
    headers = _LOG_HEADERS.get(log_path)
    if headers is None:
        headers = _read_csv_headers(log_path)
        _LOG_HEADERS[log_path] = headers

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
    row.update({k: float(v) for k, v in extra_metrics.items()})

    csv_values = []
    for header_name in headers:
        if header_name not in row:
            raise RuntimeError(
                f"CSV log missing value for header '{header_name}'. "
                f"Available keys: {sorted(row.keys())[:12]}..."
            )
        value = row[header_name]
        if header_name == 'epoch':
            csv_values.append(int(value))
        elif header_name in ('lr',):
            csv_values.append(f"{float(value):.8f}")
        elif header_name in ('grad_norm',):
            csv_values.append(f"{float(value):.6f}")
        else:
            csv_values.append(f"{float(value):.6f}")

    if len(csv_values) != len(headers):
        raise RuntimeError(
            f"CSV row/header length mismatch: row={len(csv_values)} header={len(headers)}"
        )

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
        for key in headers:
            if key in CSV_HEADER:
                continue
            f.write(f"  {key:<28}: {float(row[key]):.6f}\n")
        f.write('-' * 64 + '\n')


def _readable_path(log_path):
    base, _ = os.path.splitext(log_path)
    return base + '_readable.txt'
