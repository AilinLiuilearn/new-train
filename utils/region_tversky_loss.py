# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveRegionSpecificTverskyLoss(nn.Module):
    def __init__(
        self,
        smooth=1e-5,
        num_region_per_axis=(16, 16),
        batch_dice=True,
        A=0.3,
        B=0.4,
        apply_nonlin=True,
    ):
        super().__init__()
        if len(num_region_per_axis) != 2:
            raise ValueError('Current segmentation task is 2D, num_region_per_axis must have length 2.')
        self.smooth = smooth
        self.batch_dice = batch_dice
        self.num_region_per_axis = tuple(num_region_per_axis)
        self.A = A
        self.B = B
        self.apply_nonlin = apply_nonlin

    @staticmethod
    def _prepare_binary_target(logits, target):
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode='nearest')
        return (target >= 0.5).to(dtype=logits.dtype, device=logits.device)

    def _region_mean(self, x):
        grid_h, grid_w = self.num_region_per_axis
        h, w = x.shape[-2:]
        if h % grid_h != 0 or w % grid_w != 0:
            raise ValueError(
                f'Input size {(h, w)} must be divisible by region grid {(grid_h, grid_w)} '
                'when deterministic region pooling is enabled.'
            )
        patch_h = h // grid_h
        patch_w = w // grid_w
        x = x.reshape(x.shape[0], x.shape[1], grid_h, patch_h, grid_w, patch_w)
        return x.mean(dim=(3, 5))

    def forward(self, logits, target):
        target = self._prepare_binary_target(logits, target)

        if logits.shape[1] == 1:
            pred = torch.sigmoid(logits) if self.apply_nonlin else logits
            target_onehot = target
        elif logits.shape[1] == 2:
            pred = torch.softmax(logits, dim=1)[:, 1:2] if self.apply_nonlin else logits[:, 1:2]
            target_onehot = target
        else:
            raise ValueError(f'Expected binary logits with 1 or 2 channels, got {logits.shape[1]}.')

        tp = pred * target_onehot
        fp = pred * (1.0 - target_onehot)
        fn = (1.0 - pred) * target_onehot

        region_tp = self._region_mean(tp)
        region_fp = self._region_mean(fp)
        region_fn = self._region_mean(fn)

        if self.batch_dice:
            region_tp = region_tp.sum(0)
            region_fp = region_fp.sum(0)
            region_fn = region_fn.sum(0)

        alpha = self.A + self.B * (region_fp + self.smooth) / (region_fp + region_fn + self.smooth)
        beta = self.A + self.B * (region_fn + self.smooth) / (region_fp + region_fn + self.smooth)

        region_tversky = (region_tp + self.smooth) / (
            region_tp + alpha * region_fp + beta * region_fn + self.smooth
        )
        return (1.0 - region_tversky).mean()


class BCEAdaptiveRegionTverskyLoss(nn.Module):
    def __init__(
        self,
        bce_weight=0.5,
        region_tversky_weight=1.0,
        pos_weight=None,
        **region_tversky_kwargs,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.region_tversky_weight = region_tversky_weight
        self.pos_weight = pos_weight
        self.region_tversky = AdaptiveRegionSpecificTverskyLoss(**region_tversky_kwargs)

    @staticmethod
    def _prepare_target(logits, target):
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode='nearest')
        return (target >= 0.5).to(dtype=logits.dtype, device=logits.device)

    def forward(self, logits, target):
        target = self._prepare_target(logits, target)
        pw = self.pos_weight
        if pw is not None and isinstance(pw, (int, float)):
            pw = torch.tensor(pw, dtype=logits.dtype, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        region_tversky = self.region_tversky(logits, target)
        total = self.bce_weight * bce + self.region_tversky_weight * region_tversky
        stats = {
            'loss_bce': bce.detach(),
            'loss_region_tversky': region_tversky.detach(),
        }
        return total, stats


Adaptive_Region_Specific_TverskyLoss = AdaptiveRegionSpecificTverskyLoss
