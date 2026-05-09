# -*- coding: utf-8 -*-
"""Qualitative visualization for segmentation outputs."""

import os

import numpy as np
import torch
import torch.nn.functional as F


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
        logit = outputs['preds'] if isinstance(outputs, dict) else outputs
        if isinstance(logit, (list, tuple)):
            logit = logit[0]

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

            tp = np.logical_and(gt_img > 0.5, pred_bin > 0.5)
            fn = np.logical_and(gt_img > 0.5, pred_bin < 0.5)
            fp = np.logical_and(gt_img < 0.5, pred_bin > 0.5)

            gt_binary = np.stack([gt_img, gt_img, gt_img], axis=-1)
            pred_binary = np.zeros((gt_img.shape[0], gt_img.shape[1], 3), dtype=np.float32)
            pred_binary[..., 1] = pred_bin.astype(np.float32)

            gt_overlay = _overlay_mask_on_gray(ct_img, gt_img, color=(1.0, 0.0, 0.0), alpha=0.85)
            pred_overlay = _overlay_mask_on_gray(ct_img, pred_bin, color=(0.0, 1.0, 0.0), alpha=0.90)
            compare_overlay = _overlay_gt_pred_on_gray(ct_img, gt_img, pred_bin)
            error_map = _make_error_map(tp, fn, fp)
            zoom_error = _make_zoom_error_panel(ct_img, gt_img, pred_bin, tp, fn, fp)
            fusion_panels = _extract_fusion_panels(task.networks['model'], i)
            cudm_panels = _extract_cudm_panels(outputs, i, target_size=gt_img.shape, ct_img=ct_img)

            panels = [
                ('CT', ct_img, 'gray'),
                ('PET', pet_img, 'inferno'),
                ('Confidence', prob_img, 'jet'),
                ('GT Overlay', gt_overlay, None),
                ('Pred Overlay', pred_overlay, None),
                ('GT vs Pred', compare_overlay, None),
                ('GT Mask', gt_binary, None),
                ('Pred Mask', pred_binary, None),
                ('FN/FP/TP', error_map, None),
                ('Zoomed Error', zoom_error, None),
            ]
            panels.extend(cudm_panels)
            panels.extend(fusion_panels[:10])

            ncols = 5
            nrows = int(np.ceil(len(panels) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.0 * nrows))
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
            plt.savefig(os.path.join(out_dir, f'diagnostic_{saved:03d}.png'), dpi=160)
            plt.close(fig)
            saved += 1

        if saved >= num_samples:
            break

    print(f'[vis_teacher] saved {saved} diagnostics to {out_dir}')


def _extract_cudm_panels(outputs, sample_idx, target_size, ct_img):
    if not isinstance(outputs, dict) or not outputs.get('fusion_aux'):
        return []

    panels = []
    for idx, aux in enumerate(outputs['fusion_aux'], start=1):
        common = aux.get('common') if isinstance(aux, dict) else None
        tumor = aux.get('tumor') if isinstance(aux, dict) else None
        if not isinstance(common, torch.Tensor) or not isinstance(tumor, torch.Tensor):
            continue

        tumor_map = _feature_energy_map(tumor, sample_idx, target_size)
        common_map = _feature_energy_map(common, sample_idx, target_size)
        panels.append((f'CUDM S{idx} Tumor', tumor_map, 'magma'))
        panels.append((f'CUDM S{idx} Common', common_map, 'viridis'))

        if idx == len(outputs['fusion_aux']):
            overlay = _overlay_heatmap_on_gray(ct_img, tumor_map, cmap_name='magma', alpha=0.55)
            panels.append((f'CUDM S{idx} Tumor Overlay', overlay, None))
    return panels


def _feature_energy_map(tensor, sample_idx, target_size):
    idx = min(sample_idx, tensor.shape[0] - 1)
    feat = tensor[idx:idx + 1].detach().float()
    energy = feat.abs().mean(dim=1, keepdim=True)
    energy = F.interpolate(energy, size=target_size, mode='bilinear', align_corners=False)
    arr = energy.squeeze().cpu().numpy()
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    return arr


def _overlay_heatmap_on_gray(gray_img, heatmap, cmap_name='magma', alpha=0.55):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap_name)
    heat_rgb = cmap(np.clip(heatmap, 0.0, 1.0))[..., :3].astype(np.float32)
    gray_rgb = np.stack([gray_img, gray_img, gray_img], axis=-1).astype(np.float32)
    weighted_alpha = alpha * np.clip(heatmap, 0.0, 1.0)[..., None]
    out = gray_rgb * (1.0 - weighted_alpha) + heat_rgb * weighted_alpha
    return np.clip(out, 0.0, 1.0)


def _extract_fusion_panels(model, sample_idx):
    if not hasattr(model, 'get_fusion_visuals'):
        return []
    visuals = model.get_fusion_visuals()
    if not visuals:
        return []

    panels = []
    for stage_name in ('fuse1', 'fuse2', 'fuse3', 'fuse4'):
        if stage_name not in visuals:
            continue
        stage = visuals[stage_name]
        for key in sorted(stage.keys()):
            if key == 'scale':
                continue
            val = stage[key]
            if not isinstance(val, torch.Tensor):
                continue
            if val.dim() < 3:
                continue
            cmap = 'viridis' if 'ct' in key else ('inferno' if 'pet' in key else 'magma')
            panels.append((f'{stage_name} {key}', _tensor_map(val, sample_idx), cmap))
    return panels


def _tensor_map(tensor, sample_idx):
    idx = min(sample_idx, tensor.shape[0] - 1)
    arr = tensor[idx].detach().float().cpu().numpy()
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    return arr


def _overlay_mask_on_gray(gray_img, mask, color=(0.0, 1.0, 0.0), alpha=0.85):
    rgb = np.stack([gray_img, gray_img, gray_img], axis=-1).astype(np.float32)
    mask = (mask > 0.5).astype(np.float32)[..., None]
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    rgb = rgb * (1.0 - alpha * mask) + color_arr * (alpha * mask) + rgb * (0.15 * mask)
    return np.clip(rgb, 0.0, 1.0)


def _overlay_gt_pred_on_gray(gray_img, gt_mask, pred_mask):
    rgb = np.stack([gray_img, gray_img, gray_img], axis=-1).astype(np.float32)
    gt_mask = gt_mask > 0.5
    pred_mask = pred_mask > 0.5

    tp = np.logical_and(gt_mask, pred_mask)
    fn = np.logical_and(gt_mask, np.logical_not(pred_mask))
    fp = np.logical_and(np.logical_not(gt_mask), pred_mask)

    rgb[tp] = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    rgb[fn] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    rgb[fp] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return np.clip(rgb, 0.0, 1.0)


def _make_error_map(tp, fn, fp):
    err = np.zeros((tp.shape[0], tp.shape[1], 3), dtype=np.float32)
    err[..., 0] = fn.astype(np.float32)
    err[..., 1] = np.maximum(tp.astype(np.float32), fp.astype(np.float32) * 0.95)
    err[..., 2] = fp.astype(np.float32)
    return np.clip(err, 0.0, 1.0)


def _make_zoom_error_panel(ct_img, gt_img, pred_bin, tp, fn, fp, pad=24, min_size=96):
    union = np.logical_or(gt_img > 0.5, pred_bin > 0.5)
    if union.any():
        ys, xs = np.where(union)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        y0 = max(0, y0 - pad)
        x0 = max(0, x0 - pad)
        y1 = min(ct_img.shape[0], y1 + pad)
        x1 = min(ct_img.shape[1], x1 + pad)
    else:
        h, w = ct_img.shape
        cy, cx = h // 2, w // 2
        half = min_size // 2
        y0, y1 = max(0, cy - half), min(h, cy + half)
        x0, x1 = max(0, cx - half), min(w, cx + half)

    crop_ct = ct_img[y0:y1, x0:x1]
    crop_tp = tp[y0:y1, x0:x1]
    crop_fn = fn[y0:y1, x0:x1]
    crop_fp = fp[y0:y1, x0:x1]

    base = np.stack([crop_ct, crop_ct, crop_ct], axis=-1).astype(np.float32)
    base[crop_tp] = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    base[crop_fn] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    base[crop_fp] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return np.clip(base, 0.0, 1.0)


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
