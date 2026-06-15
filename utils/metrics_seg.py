# -*- coding: utf-8 -*-
"""
分割评估指标（对齐 CIPA pred.py）：
- IoU / Dice / Sensitivity / Specificity / Precision / F1
- Acc：与 CIPA 一致 mean(TP/(TP+FN), TN/(TN+FP))，即 (Sens+Spec)/2，避免像素 Acc 虚高 ~98%
- Acc_pixel：像素准确率 (TP+TN)/全部，仅作参考
- HD95：逐图计算后取平均；优先 medpy，否则 scipy 边界距离近似
"""

import numpy as np
import torch


def _hd95_scipy(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """pred_bin, gt_bin: bool (H,W)，与 CIPA 一致的空集处理。"""
    from scipy.ndimage import binary_erosion, distance_transform_edt

    pred = pred_bin.astype(bool)
    gt = gt_bin.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        pred = pred.copy()
        h, w = pred.shape
        pred[h // 2, w // 2] = True

    def _border(m):
        if not m.any():
            return np.zeros_like(m, dtype=bool)
        return m & ~binary_erosion(m)

    bp, bg = _border(pred), _border(gt)
    if not bp.any() or not bg.any():
        return float(np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2))

    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)
    d_p2g = dt_gt[bp]
    d_g2p = dt_pred[bg]
    d = np.concatenate([d_p2g.ravel(), d_g2p.ravel()])
    if d.size == 0:
        return 0.0
    return float(np.percentile(d, 95))


def _hd95_numpy_fallback(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    pred = pred_bin.astype(bool)
    gt = gt_bin.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float(np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2))

    def _border_coords(mask):
        padded = np.pad(mask.astype(np.uint8), ((1, 1), (1, 1)), mode='constant')
        neighbor = np.zeros_like(mask, dtype=np.uint8)
        for dy in range(3):
            for dx in range(3):
                neighbor += padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
        border = mask & (neighbor < 9)
        coords = np.argwhere(border)
        if coords.size == 0:
            coords = np.argwhere(mask)
        return coords.astype(np.float32)

    cp = _border_coords(pred)
    cg = _border_coords(gt)
    if cp.size == 0 or cg.size == 0:
        return float(np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2))
    # Chunked all-pairs distance to avoid excessive memory on large contours.
    dists = []
    chunk = 1024
    for start in range(0, cp.shape[0], chunk):
        diff = cp[start:start + chunk, None, :] - cg[None, :, :]
        dists.append(np.sqrt((diff ** 2).sum(axis=2)).min(axis=1))
    for start in range(0, cg.shape[0], chunk):
        diff = cg[start:start + chunk, None, :] - cp[None, :, :]
        dists.append(np.sqrt((diff ** 2).sum(axis=2)).min(axis=1))
    return float(np.percentile(np.concatenate(dists), 95))


def compute_hd95_pair(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """单张 2D 二值图 HD95；优先 medpy，其次 scipy，最后 numpy fallback。"""
    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)
    try:
        from medpy.metric.binary import hd95 as medpy_hd95
        return float(medpy_hd95(pred_bin.astype(np.uint8), gt_bin.astype(np.uint8)))
    except Exception:
        try:
            return _hd95_scipy(pred_bin, gt_bin)
        except Exception:
            return _hd95_numpy_fallback(pred_bin, gt_bin)


def _compute_metrics_from_counts(tp, fp, fn, tn):
    denom_iou = tp + fp + fn
    denom_dice = 2 * tp + fp + fn
    iou = 1.0 if denom_iou == 0 else (tp / denom_iou)
    dice = 1.0 if denom_dice == 0 else ((2 * tp) / denom_dice)
    total = tp + fp + fn + tn
    acc_pixel = (tp + tn) / total if total > 0 else 0.0
    sensitivity = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    # CIPA pred.py: acc = mean( TP/(TP+FN), TN/(TN+FP) )
    acc = 0.5 * (sensitivity + specificity)
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
    return {
        'dice': dice, 'iou': iou,
        'acc': acc,
        'acc_pixel': acc_pixel,
        'sensitivity': sensitivity, 'specificity': specificity,
        'precision': precision, 'f1': f1,
    }


def search_threshold(pred_probs, targets, thresholds=None, metric='dice'):
    if thresholds is None:
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    if pred_probs.dim() == 4:
        pred_probs = pred_probs.squeeze(1)
    if targets.dim() == 4:
        targets = targets.squeeze(1)
    if targets.shape[-2:] != pred_probs.shape[-2:]:
        targets = torch.nn.functional.interpolate(
            targets.unsqueeze(1).float(), size=pred_probs.shape[-2:], mode='nearest'
        ).squeeze(1)
    pred_flat = pred_probs.reshape(-1)
    target_flat = targets.float().reshape(-1)

    best_threshold = 0.5
    best_metrics = {}
    best_score = -1.0
    all_results = []

    for t in thresholds:
        pred_bin = (pred_flat > t).float()
        tp = (pred_bin * target_flat).sum().item()
        fp = (pred_bin * (1 - target_flat)).sum().item()
        fn = ((1 - pred_bin) * target_flat).sum().item()
        tn = ((1 - pred_bin) * (1 - target_flat)).sum().item()
        metrics = _compute_metrics_from_counts(tp, fp, fn, tn)
        all_results.append((t, metrics))
        score = metrics.get(metric, metrics['dice'])
        if score > best_score:
            best_score = score
            best_threshold = t
            best_metrics = metrics

    return best_threshold, best_metrics, all_results


def compute_metrics_at_threshold(pred_probs, targets, threshold):
    if pred_probs.dim() == 4:
        pred_probs = pred_probs.squeeze(1)
    if targets.dim() == 4:
        targets = targets.squeeze(1)
    if targets.shape[-2:] != pred_probs.shape[-2:]:
        targets = torch.nn.functional.interpolate(
            targets.unsqueeze(1).float(), size=pred_probs.shape[-2:], mode='nearest'
        ).squeeze(1)
    pred_bin = (pred_probs > threshold).float()
    pred_flat = pred_bin.reshape(-1)
    target_flat = targets.float().reshape(-1)
    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()
    return _compute_metrics_from_counts(tp, fp, fn, tn)


def segmentation_metrics_cipa(pred_logits, target, threshold=0.5):
    if pred_logits.dim() == 4:
        pred_logits = pred_logits.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    if target.shape[-2:] != pred_logits.shape[-2:]:
        target = torch.nn.functional.interpolate(
            target.unsqueeze(1).float(), size=pred_logits.shape[-2:], mode='nearest'
        ).squeeze(1)
    pred = torch.sigmoid(pred_logits)
    pred_bin = (pred > threshold).float()
    target = target.float()
    pred_flat = pred_bin.reshape(-1)
    target_flat = target.reshape(-1)
    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()
    out = _compute_metrics_from_counts(tp, fp, fn, tn)
    # 单张无法定义 batch HD95；置 0
    out['hd95'] = 0.0
    return out


class SegmentationMetricsCIPA(torch.nn.Module):
    """累积 TP/FP/FN/TN + 逐样本 HD95（与 CIPA 测试脚本一致）。"""

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.hd95_list = []

    @torch.no_grad()
    def update(self, pred_logits, target):
        if pred_logits.dim() == 4:
            pred_logits = pred_logits.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)
        if target.shape[-2:] != pred_logits.shape[-2:]:
            target = torch.nn.functional.interpolate(
                target.unsqueeze(1).float(), size=pred_logits.shape[-2:], mode='nearest'
            ).squeeze(1)
        pred = torch.sigmoid(pred_logits)
        pred_bin = (pred > self.threshold).float()
        target = target.float()
        pred_flat = pred_bin.reshape(-1)
        target_flat = target.reshape(-1)
        self.tp += (pred_flat * target_flat).sum().item()
        self.fp += (pred_flat * (1 - target_flat)).sum().item()
        self.fn += ((1 - pred_flat) * target_flat).sum().item()
        self.tn += ((1 - pred_flat) * (1 - target_flat)).sum().item()

        pb = pred_bin.cpu().numpy()
        tb = target.cpu().numpy()
        for b in range(pb.shape[0]):
            self.hd95_list.append(compute_hd95_pair(pb[b] > 0.5, tb[b] > 0.5))

    def compute(self):
        denom_iou = self.tp + self.fp + self.fn
        denom_dice = 2 * self.tp + self.fp + self.fn
        iou = 1.0 if denom_iou == 0 else (self.tp / denom_iou)
        dice = 1.0 if denom_dice == 0 else ((2 * self.tp) / denom_dice)
        total = self.tp + self.fp + self.fn + self.tn
        acc_pixel = (self.tp + self.tn) / total if total > 0 else 0.0
        sensitivity = (self.tp / (self.tp + self.fn)) if (self.tp + self.fn) > 0 else 0.0
        specificity = (self.tn / (self.tn + self.fp)) if (self.tn + self.fp) > 0 else 0.0
        acc = 0.5 * (sensitivity + specificity)
        precision = (self.tp / (self.tp + self.fp)) if (self.tp + self.fp) > 0 else 0.0
        f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
        hd95 = float(np.mean(self.hd95_list)) if self.hd95_list else 0.0
        return {
            'dice': dice, 'iou': iou,
            'acc': acc,
            'acc_pixel': acc_pixel,
            'sensitivity': sensitivity, 'specificity': specificity,
            'precision': precision, 'f1': f1,
            'hd95': hd95,
        }
