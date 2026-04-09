# -*- coding: utf-8 -*-
"""
分割损失：对齐 CIPA 的 Dice+BCE。
参考：https://github.com/mj129/CIPA/blob/main/train_utils/loss.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """Dice + BCEWithLogits，支持 pos_weight 缓解小病灶类别不平衡"""

    def __init__(self, smooth=1.0, pos_weight=None, bce_weight=1.0, dice_weight=1.0):
        """
        Args:
            smooth: Dice 平滑项
            pos_weight: 正类权重，用于 BCE。None 表示不加权；float 如 10 表示正类权重为负类的 10 倍
            bce_weight: BCE 在总损失中的权重
            dice_weight: Dice 在总损失中的权重
        """
        super().__init__()
        self.smooth = smooth
        self.pos_weight = pos_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def soft_dice_coeff(self, y_true, y_pred):
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
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode='nearest')
        pw = self.pos_weight
        if pw is not None:
            if isinstance(pw, (int, float)):
                pw = torch.tensor(pw, dtype=logits.dtype, device=logits.device)
            bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction='mean')
        else:
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction='mean')
        pred = torch.sigmoid(logits)
        dice = self.soft_dice_loss(target, pred)
        return self.bce_weight * bce + self.dice_weight * dice


class AdaptiveDiceBCELoss(nn.Module):
    """
    Kendall 多任务不确定性加权：自动学习 Dice / BCE 相对权重。
    loss = exp(-s_d)*L_dice + s_d + exp(-s_b)*L_bce + s_b，s 为可学习 log-方差。
    """

    def __init__(self, smooth=1.0, pos_weight=None, init_log_var=0.0):
        super().__init__()
        self.smooth = smooth
        self._pos_weight = float(pos_weight) if pos_weight is not None else None
        self.log_vars = nn.Parameter(torch.full((2,), float(init_log_var)))

    def soft_dice_coeff(self, y_true, y_pred):
        if y_pred.dim() == 4:
            y_pred = y_pred.view(y_pred.size(0), -1)
            y_true = y_true.view(y_true.size(0), -1)
        intersection = (y_true * y_pred).sum(dim=1)
        union = y_true.sum(dim=1) + y_pred.sum(dim=1)
        score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return score.mean()

    def forward(self, logits, target):
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode='nearest')
        pw = self._pos_weight
        if pw is not None:
            pw = torch.tensor(pw, device=logits.device, dtype=logits.dtype)
            bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction='mean')
        else:
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction='mean')
        pred = torch.sigmoid(logits)
        dice = 1.0 - self.soft_dice_coeff(target, pred)
        s_d, s_b = self.log_vars[0], self.log_vars[1]
        return torch.exp(-s_d) * dice + s_d + torch.exp(-s_b) * bce + s_b


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
