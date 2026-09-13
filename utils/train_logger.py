# -*- coding: utf-8 -*-
"""训练日志：同时写入 CSV 和易读文本格式

CSV 列对齐合同：init_train_log() 将完整 header 注册到
_LOG_HEADERS[log_path]；append_epoch_log() 严格按该 header 逐 key 取值，
未知字段直接报错，缺失字段写空并警告。杜绝按 dict 插入顺序写值错列。
"""

import csv
import os
import warnings


CSV_HEADER = [
    'epoch', 'train_loss', 'val_loss', 'val_dice', 'val_iou',
    'val_acc', 'val_acc_pixel', 'val_hd95', 'lr', 'grad_norm',
]

_LOG_HEADERS = {}


def get_log_headers(log_path):
    return list(_LOG_HEADERS.get(os.path.abspath(log_path), []))


def init_train_log(log_path, extra_headers=None):
    readable_path = _readable_path(log_path)
    headers = list(CSV_HEADER) + list(extra_headers or [])
    if len(set(headers)) != len(headers):
        dupes = sorted({h for h in headers if headers.count(h) > 1})
        raise ValueError(f'Duplicate CSV headers: {dupeS}'.replace('dupeS', str(dupes)))
    _LOG_HEADERS[os.path.abspath(log_path)] = list(headers)
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(headers)

    with open(readable_path, 'w', encoding='utf-8') as f:
        f.write('Training Log\n')
        f.write('=' * 64 + '\n')
    return list(headers)


def append_epoch_log(log_path, epoch, train_loss_avg, val_metrics, lr=None, grad_norm=None, extra_metrics=None):
    extra_metrics = extra_metrics or {}
    headers = _LOG_HEADERS.get(os.path.abspath(log_path))
    if headers is None:
        raise RuntimeError(
            f'append_epoch_log called before init_train_log for {log_path}; '
            'header registry is required for column alignment'
        )
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
    row.update({k: v for k, v in extra_metrics.items()})

    known = set(headers)
    unknown = [k for k in row.keys() if k not in known]
    if unknown:
        raise ValueError(
            f'Unknown log fields not in header (refusing silent misalignment): {sorted(unknown)}'
        )
    missing = [h for h in headers if h not in row]
    if missing:
        warnings.warn(f'Missing log fields written as empty: {missing}')

    csv_values = []
    for h in headers:
        v = row.get(h, '')
        if v == '':
            csv_values.append('')
        elif h == 'epoch':
            csv_values.append(str(int(v)))
        elif h in ('lr',):
            csv_values.append(f"{float(v):.8f}")
        elif h in ('train_loss', 'val_loss', 'val_dice', 'val_iou', 'val_acc',
                   'val_acc_pixel', 'val_hd95', 'grad_norm'):
            csv_values.append(f"{float(v):.4f}" if h != 'grad_norm' else f"{float(v):.6f}")
        else:
            try:
                csv_values.append(f"{float(v):.6f}")
            except (TypeError, ValueError):
                csv_values.append(str(v))

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
            if key in CSV_HEADER or key == 'epoch':
                continue
            value = row.get(key, '')
            try:
                f.write(f"  {key:<13}: {float(value):.6f}\n")
            except (TypeError, ValueError):
                f.write(f"  {key:<13}: {value}\n")
        f.write('-' * 64 + '\n')


def _readable_path(log_path):
    base, _ = os.path.splitext(log_path)
    return base + '_readable.txt'
