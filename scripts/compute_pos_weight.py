#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描 PCLT20K 数据集的 mask，统计正负像素比，用于计算 BCE 的 pos_weight。
建议先运行此脚本，将输出的 pos_weight 作为 --pos_weight 传入训练命令。

用法:
  python scripts/compute_pos_weight.py --root /root/autodl-tmp/data/PCLT20K
  python scripts/compute_pos_weight.py --root /path/to/PCLT20K --train_only --clamp 20
"""

import os
import sys
import argparse
import random

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


def _collect_mask_paths(root, train_only=False, random_state=2023):
    """收集 mask 路径，若 train_only 则只返回训练集（按 70% 病例划分）"""
    records = []
    for name in sorted(os.listdir(root)):
        case_dir = os.path.join(root, name)
        if not os.path.isdir(case_dir):
            continue
        for fname in os.listdir(case_dir):
            if not fname.endswith('_mask.png'):
                continue
            mask_path = os.path.join(case_dir, fname)
            records.append({'case_id': name, 'mask_path': mask_path})

    if not records:
        return []

    if train_only:
        cases = list(set(r['case_id'] for r in records))
        random.Random(random_state).shuffle(cases)
        n = len(cases)
        t_end = int(n * 0.7)
        train_c = set(cases[:t_end])
        records = [r for r in records if r['case_id'] in train_c]

    return records


def main():
    parser = argparse.ArgumentParser(description='计算 PCLT20K 数据集的 pos_weight')
    parser.add_argument('--root', type=str, default='/root/autodl-tmp/data/PCLT20K',
                        help='PCLT20K 根目录')
    parser.add_argument('--train_only', action='store_true',
                        help='仅统计训练集（70% 病例），否则统计全部')
    parser.add_argument('--clamp', type=float, default=20.0,
                        help='pos_weight 上限，避免过大导致训练不稳定，0 表示不截断')
    parser.add_argument('--random_state', type=int, default=2023)
    args = parser.parse_args()

    if Image is None:
        print("需要 PIL: pip install Pillow")
        sys.exit(1)

    records = _collect_mask_paths(args.root, train_only=args.train_only, random_state=args.random_state)
    if not records:
        print(f"未找到 mask 文件，请检查 --root {args.root}")
        sys.exit(1)

    total_pos = 0
    total_neg = 0
    n = 0
    for r in records:
        m = np.array(Image.open(r['mask_path']))
        m = (m > 0).astype(np.float32)
        pos = float(m.sum())
        neg = float(m.size - m.sum())
        total_pos += pos
        total_neg += neg
        n += 1

    total = total_pos + total_neg
    r = total_pos / max(total, 1)
    pos_weight_raw = total_neg / max(total_pos, 1)

    if args.clamp > 0:
        pos_weight_clip = min(pos_weight_raw, args.clamp)
    else:
        pos_weight_clip = pos_weight_raw

    print(f"样本数: {n}")
    print(f"正像素(病灶): {total_pos:.0f}")
    print(f"负像素(背景): {total_neg:.0f}")
    print(f"病灶占比 r: {r:.6f}")
    print(f"pos_weight (理论): {pos_weight_raw:.2f}")
    print(f"pos_weight (建议, clamp={args.clamp}): {pos_weight_clip:.2f}")
    print()
    print(f"建议命令行: --pos_weight {int(pos_weight_clip)}")
    if pos_weight_clip != pos_weight_raw:
        print(f"  (理论值 {pos_weight_raw:.1f} 已截断为 {pos_weight_clip:.1f})")


if __name__ == '__main__':
    main()
