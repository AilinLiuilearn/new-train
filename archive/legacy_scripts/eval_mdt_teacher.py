# -*- coding: utf-8 -*-
"""
仅评估已训练的教师 checkpoint，指定固定阈值（如 0.5，与 CIPA 对齐）。
用法（在 new-train 目录下）：
  python eval_mdt_teacher.py --checkpoint_dir ./checkpoints_new/MDT/2026-03-21_09-01-01 --threshold 0.5
"""

import os
import sys
import json
import argparse

if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch

from configs.seg_mdt import SegMDTConfig
from datasets.pclt20k_seg import get_pclt20k_loaders, get_pclt20k_loaders_cipa_aligned
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher


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


def main():
    parser = argparse.ArgumentParser(description='MDT 教师：固定阈值评估测试集')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='含 configs.json 与 ckpt.best.pth.tar 的目录')
    parser.add_argument('--ckpt', type=str, default='ckpt.best.pth.tar', help='权重文件名')
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化阈值（与 CIPA 对齐用 0.5）')
    args = parser.parse_args()

    cfg_path = os.path.join(args.checkpoint_dir, 'configs.json')
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f'未找到 {cfg_path}')

    with open(cfg_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    # 与 ConfigBase.from_json 一致：去掉 checkpoint_dir，由 hash 计算
    raw.pop('checkpoint_dir', None)
    config = SegMDTConfig(args=raw)
    config.task = 'MDT'
    # 仅评估时无需 AMP；旧版 torch 无 torch.amp.GradScaler 时避免报错
    config.mixed_precision = False

    set_gpu(config)
    root = getattr(config, 'root', '../data/PCLT20K')
    print('数据根目录:', root)
    print('cipa_aligned:', getattr(config, 'cipa_aligned', False))
    print('加载 checkpoint:', os.path.join(args.checkpoint_dir, args.ckpt))
    print('测试集评估 threshold =', args.threshold)

    cipa_aligned = getattr(config, 'cipa_aligned', False)
    if cipa_aligned:
        _, _, test_loader = get_pclt20k_loaders_cipa_aligned(
            root,
            image_size=config.image_size_2d,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            random_state=config.random_state,
        )
    else:
        missing_rate = getattr(config, 'missing_rate', 0.3)
        val_ratio = getattr(config, 'val_ratio', 0.1)
        use_case_split = getattr(config, 'use_case_split', True)
        _, _, test_loader = get_pclt20k_loaders(
            root,
            image_size=config.image_size_2d,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            missing_rate=missing_rate,
            val_ratio=val_ratio,
            random_state=config.random_state,
            use_case_split=use_case_split,
        )

    networks = build_mdt_seg_teacher(config)
    task = MDTSegTeacher(networks, config)

    ckpt_path = os.path.join(args.checkpoint_dir, args.ckpt)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    for k, v in task.networks.items():
        if k in ckpt:
            v.load_state_dict(ckpt[k], strict=False)

    test_metrics = task.evaluate(test_loader, threshold=args.threshold)

    print('\n========== 测试集结果 (threshold=%.2f) ==========' % args.threshold)
    for k in (
        'dice', 'iou', 'hd95', 'acc', 'acc_pixel', 'sensitivity', 'specificity',
        'precision', 'f1', 'total_loss',
    ):
        if k in test_metrics:
            v = test_metrics[k]
            if isinstance(v, float):
                print(f'  {k}: {v:.6f}')
            else:
                print(f'  {k}: {v}')

    # 写回 configs.json（追加字段，不覆盖原 test_results）
    out_key = 'test_results_threshold_%.2f' % args.threshold
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg_full = json.load(f)

    def _scalar(x):
        if hasattr(x, 'item'):
            return float(x.item())
        return float(x)

    cfg_full[out_key] = {k: _scalar(v) for k, v in test_metrics.items()}
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(cfg_full, f, indent=2, ensure_ascii=False)
    print(f'\n已写入 configs.json 字段: {out_key}')


if __name__ == '__main__':
    main()
