# -*- coding: utf-8 -*-
"""训练运行记录：命令行、结构说明、损失曲线图"""

import os
import sys
import csv


def save_run_command(checkpoint_dir, argv=None):
    argv = argv if argv is not None else sys.argv
    path = os.path.join(checkpoint_dir, 'run_command.txt')
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(' '.join(argv))
        f.write('\n')


def print_run_banner(argv=None):
    argv = argv if argv is not None else sys.argv
    print('\n' + '=' * 72)
    print('【本次运行命令】')
    print(' '.join(argv))
    print('=' * 72 + '\n')


def print_architecture_doc():
    """教师：双编码器 + 浅层 FSF 或相加 + 深层 FDMF + UNet/FPN。"""
    doc = """
【教师模型结构 — 仅频域解耦路径】
1) 双编码器 extractor_mri / extractor_pet（独立权重），多尺度 stage0~3。
2) 可选 Projector 1×1：最深层对齐到 hidden（默认 256）。
3) 浅层：use_cmx=True → FreqSpatialFusionBlock×3；False → CT+PET 逐尺度相加。
4) 深层：MedicalFDMFPETCTFusion（ASD→ME→LFGF），输出 fusion3；辅助损失见 alpha_fdmf_*。
5) 解码：teacher_decoder=unet（默认）或 fpn；fpn_dropout 为末端 Dropout2d。
6) 训练损失：alpha_seg * DiceBCE + FDMF 模态内/低频 MI 项（模块内缓存）。
"""
    print(doc)


def plot_train_log_csv(log_csv_path, out_png_path):
    """根据 train_log.csv 绘制训练/验证损失与验证 Dice、HD95。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[train_run_utils] 未安装 matplotlib，跳过损失曲线:', out_png_path)
        return
    if not os.path.isfile(log_csv_path):
        return
    rows = []
    with open(log_csv_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        return
    epochs = [int(row['epoch']) for row in rows]

    def col(name, default=0.0):
        out = []
        for row in rows:
            try:
                out.append(float(row.get(name, default)))
            except (TypeError, ValueError):
                out.append(default)
        return out

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, col('train_loss'), label='train_loss', marker='o', markersize=3)
    axes[0].plot(epochs, col('val_loss'), label='val_loss', marker='s', markersize=3)
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].legend()
    axes[0].set_title('Training / validation loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, col('val_dice'), label='val_dice', color='C2', marker='o', markersize=3)
    if rows[0].get('val_hd95') is not None:
        axes[1].plot(epochs, col('val_hd95'), label='val_hd95', color='C3', marker='^', markersize=3)
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('metric')
    axes[1].legend()
    axes[1].set_title('Validation Dice / HD95')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png_path) or '.', exist_ok=True)
    plt.savefig(out_png_path, dpi=150)
    plt.close()
    print(f'[train_run_utils] 损失曲线已保存: {out_png_path}')
