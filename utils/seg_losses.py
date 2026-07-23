# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


def prepare_binary_target(logits, target):
    if target.dim() == 3:
        target = target.unsqueeze(1)
    target = target.float()
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target, size=logits.shape[-2:], mode='nearest')
    return (target >= 0.5).to(dtype=logits.dtype, device=logits.device)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1.0, pos_weight=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.pos_weight = pos_weight

    def forward(self, logits, target):
        target = prepare_binary_target(logits, target)
        pw = self.pos_weight
        if pw is not None and isinstance(pw, (int, float)):
            pw = torch.tensor(pw, dtype=logits.dtype, device=logits.device)

        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        pred = torch.sigmoid(logits).float()
        target_f = target.float()
        intersection = (pred * target_f).sum(dim=(1, 2, 3))
        pred_sum = pred.sum(dim=(1, 2, 3))
        target_sum = target_f.sum(dim=(1, 2, 3))
        denom = (pred_sum + target_sum + float(self.smooth)).clamp_min(1e-6)

        dice = 1.0 - (2.0 * intersection + float(self.smooth)) / denom
        dice = dice.mean().to(dtype=bce.dtype)

        total = self.bce_weight * bce + self.dice_weight * dice
        stats = {
            'loss_bce': bce.detach(),
            'loss_dice': dice.detach(),
        }
        return total, stats
