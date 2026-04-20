# -*- coding: utf-8 -*-
"""
解耦特征 t-SNE 可视化脚本
-------------------------------------------------
验证目标（论文图）：
  图1：z_ct_g 与 z_pet_g 分布高度重叠 → 通用特征已对齐
  图2：z_ct   与 z_ct_g 分布明显分离 → 专属与通用特征已区分
  图3：四组综合对比图（z_ct_g, z_pet_g, z_ct, z_pet）

用法（在 new-train 目录下）：
  # 使用训练好的 best checkpoint 做可视化
  python scripts/visualize_decouple.py \
      --ckpt_dir checkpoints_new/MDT/2026-xx-xx_xx-xx-xx \
      --root ../data/PCLT20K \
      --split val \
      --max_samples 800 \
      --out_dir vis_tsne

  # 指定自定义 checkpoint 文件（默认找 ckpt.best.pth.tar）
  python scripts/visualize_decouple.py \
      --ckpt_dir checkpoints_new/MDT/your_run \
      --ckpt_file ckpt.last.pth.tar
"""

import os
import sys
import json
import argparse

# 保证从 new-train 根目录解析所有 import
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.getcwd() != _ROOT:
    os.chdir(_ROOT)

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import _teacher_forward


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def load_config_from_dir(ckpt_dir):
    """从 configs.json 恢复训练时的 config"""
    cfg_path = os.path.join(ckpt_dir, 'configs.json')
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"未找到 configs.json: {cfg_path}")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg_dict = json.load(f)
    config = SegMDTConfig()
    for k, v in cfg_dict.items():
        setattr(config, k, v)
    return config


def load_checkpoint(networks, ckpt_path, device):
    """将 checkpoint 中的权重加载进 networks"""
    ckpt = torch.load(ckpt_path, map_location=device)
    loaded, skipped = [], []
    for k, net in networks.items():
        if k in ckpt:
            net.load_state_dict(ckpt[k], strict=False)
            loaded.append(k)
        else:
            skipped.append(k)
    if loaded:
        print(f"  已加载: {loaded}")
    if skipped:
        print(f"  跳过（checkpoint 中无对应 key）: {skipped}")


def get_loader(config, split='val'):
    """构建数据 loader"""
    from datasets.pclt20k_seg import get_pclt20k_loaders, get_pclt20k_loaders_cipa_aligned
    cipa_aligned = getattr(config, 'cipa_aligned', False)
    if cipa_aligned:
        train_l, val_l, test_l = get_pclt20k_loaders_cipa_aligned(
            config.root,
            image_size=config.image_size_2d,
            batch_size=8,
            num_workers=getattr(config, 'num_workers', 4),
            random_state=getattr(config, 'random_state', 2023),
        )
    else:
        train_l, val_l, test_l = get_pclt20k_loaders(
            config.root,
            image_size=config.image_size_2d,
            batch_size=8,
            num_workers=getattr(config, 'num_workers', 4),
            missing_rate=getattr(config, 'missing_rate', 0.3),
            val_ratio=getattr(config, 'val_ratio', 0.1),
            random_state=getattr(config, 'random_state', 2023),
            use_case_split=getattr(config, 'use_case_split', True),
        )
    return {'train': train_l, 'val': val_l, 'test': test_l}[split]


# ─────────────────────────────────────────────────────────────
# 特征收集
# ─────────────────────────────────────────────────────────────

def collect_features(networks, loader, device, config, max_samples=800):
    """
    推理 loader，全局平均池化后返回四组特征向量：
      z_ct_g  (B,C)  CT  通用特征
      z_pet_g (B,C)  PET 通用特征
      z_ct    (B,C)  CT  专属特征
      z_pet   (B,C)  PET 专属特征
    """
    for v in networks.values():
        v.eval()

    buckets = {'z_ct_g': [], 'z_pet_g': [], 'z_ct': [], 'z_pet': []}
    n_collected = 0

    with torch.no_grad():
        for batch in loader:
            if n_collected >= max_samples:
                break
            ct   = batch['ct'].float().to(device)
            pet  = batch['pet'].float().to(device)
            mask = batch['mask'].float().to(device)

            out = _teacher_forward(networks, ct, pet, mask.shape[-2:], config)

            # 空间维度全局平均池化：(B,C,H,W) → (B,C)
            def gap(t):
                return t.mean(dim=[2, 3]).cpu().float().numpy()

            buckets['z_ct_g'].append(gap(out['z_mri_g']))
            buckets['z_pet_g'].append(gap(out['z_pet_g']))
            buckets['z_ct'].append(gap(out['z_mri']))
            buckets['z_pet'].append(gap(out['z_pet']))
            n_collected += ct.size(0)

    for v in networks.values():
        v.train()

    result = {k: np.concatenate(v, axis=0)[:max_samples] for k, v in buckets.items()}
    print(f"  收集样本数: {len(result['z_ct_g'])}  特征维度: {result['z_ct_g'].shape[1]}")
    return result


# ─────────────────────────────────────────────────────────────
# t-SNE 绘图
# ─────────────────────────────────────────────────────────────

_PALETTE = {
    'z_ct_g':  ('#1976D2', '#BBDEFB'),   # 蓝色（实心点 / 背景）
    'z_pet_g': ('#E53935', '#FFCDD2'),   # 红色
    'z_ct':    ('#43A047', '#C8E6C9'),   # 绿色
    'z_pet':   ('#8E24AA', '#E1BEE7'),   # 紫色
}
_LABEL = {
    'z_ct_g':  r'$z_{CT,g}$  (CT 通用)',
    'z_pet_g': r'$z_{PET,g}$ (PET 通用)',
    'z_ct':    r'$z_{CT}$    (CT 专属)',
    'z_pet':   r'$z_{PET}$   (PET 专属)',
}


def _run_tsne(arrays, perplexity=40, n_iter=1200, seed=42):
    """拼接并执行 t-SNE，返回各组的 2D 嵌入"""
    sizes = [len(a) for a in arrays]
    joined = np.concatenate(arrays, axis=0)
    tsne = TSNE(n_components=2, random_state=seed,
                perplexity=perplexity, n_iter=n_iter, init='pca')
    emb = tsne.fit_transform(joined)
    splits = []
    offset = 0
    for s in sizes:
        splits.append(emb[offset:offset + s])
        offset += s
    return splits


def _scatter(ax, emb, color, label, alpha=0.55, s=12):
    ax.scatter(emb[:, 0], emb[:, 1], c=color, alpha=alpha, s=s,
               linewidths=0, label=label)


def plot_fig1_alignment(feats, out_path, perplexity=40):
    """图1：z_ct_g vs z_pet_g ——验证通用特征对齐"""
    print("  运行 t-SNE (图1：通用特征对齐)…")
    e_ctg, e_petg = _run_tsne(
        [feats['z_ct_g'], feats['z_pet_g']], perplexity=perplexity)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')
    _scatter(ax, e_ctg,  _PALETTE['z_ct_g'][0],  _LABEL['z_ct_g'])
    _scatter(ax, e_petg, _PALETTE['z_pet_g'][0], _LABEL['z_pet_g'])
    ax.set_title('通用特征对齐验证\n(z_CT,g  vs  z_PET,g)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=10, markerscale=2, framealpha=0.85,
              loc='upper right', edgecolor='#CCCCCC')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  → {out_path}")


def plot_fig2_separation(feats, out_path, perplexity=40):
    """图2：z_ct_g vs z_ct ——验证同模态通用 vs 专属分离"""
    print("  运行 t-SNE (图2：通用 vs 专属分离)…")
    e_ctg, e_ct = _run_tsne(
        [feats['z_ct_g'], feats['z_ct']], perplexity=perplexity)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')
    _scatter(ax, e_ctg, _PALETTE['z_ct_g'][0], _LABEL['z_ct_g'])
    _scatter(ax, e_ct,  _PALETTE['z_ct'][0],   _LABEL['z_ct'])
    ax.set_title('CT 通用特征 vs 专属特征分离验证\n(z_CT,g  vs  z_CT)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=10, markerscale=2, framealpha=0.85,
              loc='upper right', edgecolor='#CCCCCC')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  → {out_path}")


def plot_fig3_overview(feats, out_path, perplexity=40):
    """图3：四组综合对比（论文主图）"""
    print("  运行 t-SNE (图3：四组综合)…")
    keys = ['z_ct_g', 'z_pet_g', 'z_ct', 'z_pet']
    embs = _run_tsne([feats[k] for k in keys], perplexity=perplexity)

    fig, ax = plt.subplots(figsize=(8, 7.5))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')
    for k, emb in zip(keys, embs):
        _scatter(ax, emb, _PALETTE[k][0], _LABEL[k])

    ax.set_title('解耦特征空间可视化 (t-SNE)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.legend(fontsize=10, markerscale=2.2, framealpha=0.88,
              loc='best', edgecolor='#BBBBBB')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  → {out_path}")


def plot_fig4_2x2(feats, out_path, perplexity=40):
    """图4：2×2 子图，适合论文双栏排版"""
    print("  运行 t-SNE (图4：2×2 子图)…")
    keys = ['z_ct_g', 'z_pet_g', 'z_ct', 'z_pet']
    embs = _run_tsne([feats[k] for k in keys], perplexity=perplexity)
    emb_map = dict(zip(keys, embs))

    pairs = [
        ('z_ct_g', 'z_pet_g', '通用特征对齐\n(CT‑g  vs  PET‑g)'),
        ('z_ct',   'z_pet',   '专属特征差异\n(CT  vs  PET)'),
        ('z_ct_g', 'z_ct',    'CT 通用 vs 专属'),
        ('z_pet_g','z_pet',   'PET 通用 vs 专属'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    fig.patch.set_facecolor('#F5F5F5')
    for ax, (k1, k2, title) in zip(axes.flat, pairs):
        ax.set_facecolor('#F5F5F5')
        _scatter(ax, emb_map[k1], _PALETTE[k1][0], _LABEL[k1])
        _scatter(ax, emb_map[k2], _PALETTE[k2][0], _LABEL[k2])
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, markerscale=2, loc='upper right',
                  framealpha=0.85, edgecolor='#CCCCCC')
        ax.axis('off')
    fig.suptitle('MDT 解耦特征 t-SNE 可视化', fontsize=14,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  → {out_path}")


# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='MDT 解耦特征 t-SNE 可视化',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--ckpt_dir',    required=True, help='训练输出目录（含 configs.json）')
    p.add_argument('--ckpt_file',   default='ckpt.best.pth.tar', help='checkpoint 文件名')
    p.add_argument('--root',        default=None,
                   help='PCLT20K 数据根目录（默认从 configs.json 读取）')
    p.add_argument('--split',       default='val', choices=('train', 'val', 'test'),
                   help='用于提取特征的数据集划分')
    p.add_argument('--max_samples', type=int, default=800,
                   help='最多使用多少个样本做 t-SNE（越多越准但越慢）')
    p.add_argument('--perplexity',  type=float, default=40,
                   help='t-SNE perplexity；样本数 < 100 时应调小到 20-30')
    p.add_argument('--out_dir',     default='vis_tsne',
                   help='可视化图片输出目录')
    p.add_argument('--gpu',         default='0', help='使用的 GPU ID')
    return p.parse_args()


def main():
    args = parse_args()

    # ── 设备 ──────────────────────────────────────────────────
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # ── 加载 config ───────────────────────────────────────────
    ckpt_dir = args.ckpt_dir
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(_ROOT, ckpt_dir)
    print(f"checkpoint 目录: {ckpt_dir}")
    config = load_config_from_dir(ckpt_dir)
    if args.root:
        config.root = args.root
    if not getattr(config, 'root', None):
        config.root = '../data/PCLT20K'
    config.gpus = [int(args.gpu)]
    # 确保解耦分支处于开启状态（alpha 值不影响推理）
    config.use_specific = getattr(config, 'use_specific', True)

    # ── 构建模型并加载权重 ────────────────────────────────────
    print("构建模型…")
    networks = build_mdt_seg_teacher(config)
    for v in networks.values():
        v.to(device)

    ckpt_path = os.path.join(ckpt_dir, args.ckpt_file)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"未找到 checkpoint: {ckpt_path}")
    print(f"加载权重: {ckpt_path}")
    load_checkpoint(networks, ckpt_path, device)

    # ── 构建 DataLoader ───────────────────────────────────────
    print(f"数据集: {args.split}  根目录: {config.root}")
    loader = get_loader(config, split=args.split)
    print(f"  总 batch: {len(loader)}")

    # ── 特征收集 ──────────────────────────────────────────────
    print(f"收集特征（最多 {args.max_samples} 个样本）…")
    feats = collect_features(networks, loader, device, config,
                             max_samples=args.max_samples)

    # ── 输出目录 ──────────────────────────────────────────────
    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(ckpt_dir, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"输出目录: {out_dir}")

    # ── 绘图 ──────────────────────────────────────────────────
    perp = args.perplexity
    plot_fig1_alignment(feats, os.path.join(out_dir, 'fig1_general_alignment.png'), perp)
    plot_fig2_separation(feats, os.path.join(out_dir, 'fig2_ct_general_vs_specific.png'), perp)
    plot_fig3_overview(feats, os.path.join(out_dir, 'fig3_overview.png'), perp)
    plot_fig4_2x2(feats, os.path.join(out_dir, 'fig4_2x2.png'), perp)

    print("\n全部完成！生成图片：")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith('.png'):
            print(f"  {os.path.join(out_dir, f)}")


if __name__ == '__main__':
    main()
