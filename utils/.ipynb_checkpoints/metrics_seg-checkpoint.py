# -*- coding: utf-8 -*-
"""
分割评估指标：对齐 CIPA 的 evaluate。
整体 TP/FP/FN/TN 统计后算 Dice、IoU、sensitivity、specificity、precision、F1。
"""

import torch
import numpy as np


def segmentation_metrics_cipa(pred_logits, target, threshold=0.5):
    """
    与 CIPA train_utils/train_and_eval.evaluate 一致：
    先 sigmoid + 阈值二值化，再整体统计 TP/FP/FN/TN，最后算 Dice、IoU 等。
    pred_logits: (N,1,H,W) 或 (N,H,W)
    target: (N,1,H,W) 或 (N,H,W)，0/1
    """
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
    # CIPA 公式
    denom_iou = tp + fp + fn
    denom_dice = 2 * tp + fp + fn
    iou = 1.0 if denom_iou == 0 else (tp / denom_iou)
    dice = 1.0 if denom_dice == 0 else ((2 * tp) / denom_dice)
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total > 0 else 0.0
    sensitivity = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
    return {
        'dice': dice,
        'iou': iou,
        'acc': acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
    }


class SegmentationMetricsCIPA(torch.nn.Module):
    """逐 batch 累积 TP/FP/FN/TN，最后一次性算指标（与 CIPA 一致）"""

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

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

    def compute(self):
        denom_iou = self.tp + self.fp + self.fn
        denom_dice = 2 * self.tp + self.fp + self.fn
        iou = 1.0 if denom_iou == 0 else (self.tp / denom_iou)
        dice = 1.0 if denom_dice == 0 else ((2 * self.tp) / denom_dice)
        total = self.tp + self.fp + self.fn + self.tn
        acc = (self.tp + self.tn) / total if total > 0 else 0.0
        sensitivity = (self.tp / (self.tp + self.fn)) if (self.tp + self.fn) > 0 else 0.0
        specificity = (self.tn / (self.tn + self.fp)) if (self.tn + self.fp) > 0 else 0.0
        precision = (self.tp / (self.tp + self.fp)) if (self.tp + self.fp) > 0 else 0.0
        f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
        return {
            'dice': dice, 'iou': iou, 'acc': acc,
            'sensitivity': sensitivity, 'specificity': specificity,
            'precision': precision, 'f1': f1,
        }
