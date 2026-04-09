# -*- coding: utf-8 -*-
"""
分割损失：对齐 CIPA 的 Dice+BCE。
参考：https://github.com/mj129/CIPA/blob/main/train_utils/loss.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """Dice + BCEWithLogits，与 CIPA dice_bce_loss 一致（二分类分割）"""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def soft_dice_coeff(self, y_true, y_pred):
        # y_true, y_pred: (B,1,H,W) 或 (B,H,W)，float
        if y_pred.dim() == 4:
            y_pred = y_pred.view(y_pred.size(0), -1)
            y_true = y_true.view(y_true.size(0), -1)
        intersection = (y_true * y_pred).sum(dim=1)
        union = y_true.sum(dim=1) + y_pred.sum(dim=1)
        score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        return 1.0 - self.soft_dice_coeff(y_true, y_pred)

    def forward(self, logits, target):
        # logits: (B,1,H,W) 未激活
        # target: (B,1,H,W) 或 (B,H,W)，0/1 float
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()
        # 尺寸不一致时对齐到 logits
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode='nearest')
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='mean')
        pred = torch.sigmoid(logits)
        dice = self.soft_dice_loss(target, pred)
        return bce + dice


class DiceLoss(nn.Module):
    """仅 Dice loss（二分类）"""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode='nearest')
        pred = torch.sigmoid(logits)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice
