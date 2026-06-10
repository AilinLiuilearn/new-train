# -*- coding: utf-8 -*-
"""Summarize per-sample segmentation metrics with lesion-size filtering."""

import argparse
import csv
import os
from collections import Counter, defaultdict

import numpy as np


NUMERIC_KEYS = [
    'dice', 'iou', 'precision', 'recall', 'hd95',
    'tp', 'fp', 'fn', 'tn', 'pred_area', 'gt_area', 'fp_ratio', 'fn_ratio',
]


def _load_rows(path):
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in NUMERIC_KEYS:
                if key in row and row[key] != '':
                    row[key] = float(row[key])
            rows.append(row)
    return rows


def _safe_mean(rows, key):
    vals = [float(r[key]) for r in rows if key in r]
    return float(np.mean(vals)) if vals else 0.0


def _safe_median(rows, key):
    vals = [float(r[key]) for r in rows if key in r]
    return float(np.median(vals)) if vals else 0.0


def _aggregate(rows):
    tp = sum(float(r.get('tp', 0.0)) for r in rows)
    fp = sum(float(r.get('fp', 0.0)) for r in rows)
    fn = sum(float(r.get('fn', 0.0)) for r in rows)
    tn = sum(float(r.get('tn', 0.0)) for r in rows)
    dice_global = 1.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    iou_global = 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)
    precision_global = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall_global = 0.0 if tp + fn == 0 else tp / (tp + fn)
    pred_empty = sum(1 for r in rows if float(r.get('pred_area', 0.0)) == 0)
    dice_zero = sum(1 for r in rows if float(r.get('dice', 0.0)) < 1e-6)
    hd95_999 = sum(1 for r in rows if float(r.get('hd95', 0.0)) >= 999.0)
    return {
        'n': len(rows),
        'dice_global': dice_global,
        'iou_global': iou_global,
        'precision_global': precision_global,
        'recall_global': recall_global,
        'dice_mean': _safe_mean(rows, 'dice'),
        'dice_median': _safe_median(rows, 'dice'),
        'hd95_mean': _safe_mean(rows, 'hd95'),
        'hd95_median': _safe_median(rows, 'hd95'),
        'fp_mean': _safe_mean(rows, 'fp'),
        'fn_mean': _safe_mean(rows, 'fn'),
        'pred_empty': pred_empty,
        'dice_zero': dice_zero,
        'hd95_999': hd95_999,
    }


def _write_rank(out_dir, name, rows, key, reverse=False, topk=100):
    if not rows:
        return
    path = os.path.join(out_dir, f'{name}.csv')
    selected = sorted(rows, key=lambda r: float(r.get(key, 0.0)), reverse=reverse)[:topk]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=selected[0].keys())
        writer.writeheader()
        writer.writerows(selected)


def _write_case_summary(out_dir, rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get('case_id', '')].append(row)
    case_rows = []
    for case_id, items in grouped.items():
        agg = _aggregate(items)
        case_rows.append({'case_id': case_id, **agg})
    case_rows = sorted(case_rows, key=lambda r: r['dice_mean'])
    path = os.path.join(out_dir, 'case_summary_valid.csv')
    if case_rows:
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=case_rows[0].keys())
            writer.writeheader()
            writer.writerows(case_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--tiny_thr', type=float, default=100.0)
    parser.add_argument('--normal_thr', type=float, default=500.0)
    parser.add_argument('--large_thr', type=float, default=2000.0)
    parser.add_argument('--topk', type=int, default=100)
    args = parser.parse_args()

    rows = _load_rows(args.csv)
    groups = {
        'all': rows,
        f'valid_ge{int(args.tiny_thr)}': [r for r in rows if float(r.get('gt_area', 0.0)) >= args.tiny_thr],
        f'tiny_lt{int(args.tiny_thr)}': [r for r in rows if 0 < float(r.get('gt_area', 0.0)) < args.tiny_thr],
        f'small_{int(args.tiny_thr)}_{int(args.normal_thr)}': [r for r in rows if args.tiny_thr <= float(r.get('gt_area', 0.0)) < args.normal_thr],
        f'normal_ge{int(args.normal_thr)}': [r for r in rows if float(r.get('gt_area', 0.0)) >= args.normal_thr],
        f'large_ge{int(args.large_thr)}': [r for r in rows if float(r.get('gt_area', 0.0)) >= args.large_thr],
    }

    os.makedirs(args.out_dir, exist_ok=True)
    summary_rows = []
    for name, items in groups.items():
        agg = _aggregate(items)
        summary_rows.append({'group': name, **agg})
        group_dir = os.path.join(args.out_dir, name)
        os.makedirs(group_dir, exist_ok=True)
        _write_rank(group_dir, 'lowest_dice', items, 'dice', reverse=False, topk=args.topk)
        _write_rank(group_dir, 'highest_hd95', items, 'hd95', reverse=True, topk=args.topk)
        _write_rank(group_dir, 'most_fp', items, 'fp', reverse=True, topk=args.topk)
        _write_rank(group_dir, 'most_fn', items, 'fn', reverse=True, topk=args.topk)
        if name.startswith('valid_ge'):
            _write_case_summary(group_dir, items)

    summary_path = os.path.join(args.out_dir, 'filtered_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    type_counter = Counter(r.get('error_type', '') for r in rows)
    with open(os.path.join(args.out_dir, 'error_type_counts.txt'), 'w') as f:
        for key, value in type_counter.most_common():
            f.write(f'{key},{value}\n')

    print(f'[filter_summary] saved {summary_path}')
    for row in summary_rows:
        print(
            f"{row['group']}: n={row['n']} dice_g={row['dice_global']:.4f} "
            f"dice_m={row['dice_mean']:.4f} hd95_m={row['hd95_mean']:.2f} "
            f"empty={row['pred_empty']} zero={row['dice_zero']} hd999={row['hd95_999']}"
        )


if __name__ == '__main__':
    main()
