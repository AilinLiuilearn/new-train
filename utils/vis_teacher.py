# -*- coding: utf-8 -*-
"""Qualitative visualization for segmentation outputs."""

import os

import numpy as np
import torch


@torch.no_grad()
def save_segmentation_diagnostics(task, loader, out_dir, num_samples=8, threshold=0.5):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[vis_teacher] matplotlib not installed, skip diagnostics')
        return

    os.makedirs(out_dir, exist_ok=True)
    saved = 0

    for batch in loader:
        ct = batch['ct'].float().to(task.device)
        pet = batch['pet'].float().to(task.device)
        mask = batch['mask'].float().to(task.device)
        outputs = task.networks['model'](ct, pet, target_size=mask.shape[-2:])
        preds = outputs['preds'] if isinstance(outputs, dict) else outputs
        logit = preds[0]

        prob = torch.sigmoid(logit).squeeze(1).cpu().numpy()
        gt = mask.squeeze(1).cpu().numpy()
        ct_np = ct.cpu().numpy()
        pet_np = pet.cpu().numpy()

        for i in range(ct_np.shape[0]):
            if saved >= num_samples:
                break

            ct_img = _to_display(ct_np[i])
            pet_img = _to_display(pet_np[i])
            prob_img = prob[i]
            gt_img = (gt[i] > 0.5).astype(np.float32)
            pred_bin = (prob_img > threshold).astype(np.float32)
            fn = np.logical_and(gt_img > 0.5, pred_bin < 0.5)
            fp = np.logical_and(gt_img < 0.5, pred_bin > 0.5)

            err = np.zeros((gt_img.shape[0], gt_img.shape[1], 3), dtype=np.float32)
            err[..., 0] = fn.astype(np.float32)
            err[..., 2] = fp.astype(np.float32)

            gt_rgb = np.stack([ct_img, ct_img, ct_img], axis=-1)
            gt_rgb[..., 0] = np.clip(gt_rgb[..., 0] + 1.0 * gt_img, 0, 1)
            gt_rgb[..., 1] = np.clip(gt_rgb[..., 1] * (1.0 - 0.85 * gt_img), 0, 1)
            gt_rgb[..., 2] = np.clip(gt_rgb[..., 2] * (1.0 - 0.85 * gt_img), 0, 1)
            gt_binary = np.stack([gt_img, gt_img, gt_img], axis=-1)

            panels = [
                ('CT', ct_img, 'gray'),
                ('PET', pet_img, 'inferno'),
                ('Confidence', prob_img, 'jet'),
                ('GT Overlay', gt_rgb, None),
                ('GT Mask', gt_binary, None),
                ('FN/FP', err, None),
            ]

            ncols = 4
            nrows = int(np.ceil(len(panels) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
            axes = np.atleast_1d(axes).reshape(nrows, ncols)

            for ax, (title, img, cmap) in zip(axes.flat, panels):
                if img.ndim == 3 and img.shape[-1] == 3:
                    ax.imshow(img)
                else:
                    ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if cmap else None)
                ax.set_title(title)
                ax.axis('off')

            for ax in axes.flat[len(panels):]:
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'diagnostic_{saved:03d}.png'), dpi=140)
            plt.close(fig)
            saved += 1

        if saved >= num_samples:
            break

    print(f'[vis_teacher] saved {saved} diagnostics to {out_dir}')


def _to_display(img_3ch):
    if img_3ch.shape[0] == 3:
        img = np.transpose(img_3ch, (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = img * std + mean
        img = np.clip(img, 0, 1)
        return img.mean(axis=2)
    img = img_3ch.squeeze()
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img
