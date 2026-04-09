# -*- coding: utf-8 -*-
"""教师分割结果可视化：CT + GT + 预测 叠加"""

import os

import numpy as np
import torch

from tasks.mdt_seg import _teacher_forward


@torch.no_grad()
def save_teacher_prediction_grid(
    task,
    loader,
    out_dir,
    num_samples=6,
    threshold=0.5,
    device=None,
):
    """
    从 loader 取前 num_samples 张（按 batch 展开），保存 PNG：
    每行：CT | GT | Pred_prob | 叠加(CT 灰度 + 预测红 + GT 绿)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[vis_teacher] 需要 matplotlib，跳过可视化')
        return

    device = device or task.device
    os.makedirs(out_dir, exist_ok=True)
    task_nets = task.networks
    cfg = task.config

    saved = 0
    for batch in loader:
        ct = batch['ct'].float().to(device)
        pet = batch['pet'].float().to(device)
        mask = batch['mask'].float().to(device)
        target_size = mask.shape[-2:]
        out = _teacher_forward(task_nets, ct, pet, target_size, cfg)
        logit = out['seg_logit']
        prob = torch.sigmoid(logit).squeeze(1).cpu().numpy()
        ct_np = ct.squeeze(1).cpu().numpy()
        gt_np = mask.squeeze(1).cpu().numpy()

        B = ct_np.shape[0]
        for i in range(B):
            if saved >= num_samples:
                break
            c = ct_np[i]
            g = gt_np[i]
            p = prob[i]
            c = (c - c.min()) / (c.max() - c.min() + 1e-8)
            pred_bin = (p > threshold).astype(np.float32)

            h, w = c.shape
            rgb = np.stack([c, c, c], axis=-1)
            overlay = rgb.copy()
            overlay[..., 0] = np.clip(overlay[..., 0] + 0.5 * pred_bin, 0, 1)
            overlay[..., 1] = np.clip(overlay[..., 1] + 0.45 * g, 0, 1)

            fig, ax = plt.subplots(1, 4, figsize=(14, 3.5))
            ax[0].imshow(c, cmap='gray')
            ax[0].set_title('CT')
            ax[1].imshow(g, cmap='gray', vmin=0, vmax=1)
            ax[1].set_title('GT')
            ax[2].imshow(p, cmap='hot', vmin=0, vmax=1)
            ax[2].set_title('Pred prob')
            ax[3].imshow(overlay)
            ax[3].set_title('Overlay (R=pred G=GT)')
            for a in ax:
                a.axis('off')
            plt.suptitle(f'sample_{saved}')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'vis_{saved:02d}.png'), dpi=120)
            plt.close()
            saved += 1
        if saved >= num_samples:
            break

    print(f'[vis_teacher] 已保存 {saved} 张可视化 → {out_dir}')
