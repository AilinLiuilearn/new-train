# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics_seg import SegmentationMetricsCIPA
from utils.optimization import get_optimizer


class WeightedIoULoss(nn.Module):
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
        inter = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
        iou = (inter + self.smooth) / (union + self.smooth)
        return (1.0 - iou).mean()


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, target):
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode='nearest')
        pw = self.pos_weight
        if pw is not None and isinstance(pw, (int, float)):
            pw = torch.tensor(pw, dtype=logits.dtype, device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)


class MutationLoss(nn.Module):
    def __init__(self, pos_weight=None, smooth=1.0):
        super().__init__()
        self.bce = WeightedBCELoss(pos_weight=pos_weight)
        self.iou = WeightedIoULoss(smooth=smooth)

    def _single(self, pred, target):
        return self.bce(pred, target) + self.iou(pred, target)

    def forward(self, preds, target):
        losses = [self._single(p, target) for p in preds]
        psum = preds[0]
        for p in preds[1:]:
            psum = psum + p
        losses.append(self._single(psum, target))
        return sum(losses), losses


def _forward(nets, ct, pet, target_size):
    return nets['model'](ct, pet, target_size=target_size)


class MDTSegStudent:
    def __init__(self, networks, config):
        self.networks = {k: v for k, v in networks.items() if v is not None}
        self.config = config
        self.device = torch.device('cuda', int(config.gpus[0]))
        for v in self.networks.values():
            v.to(self.device)

        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None
        self.loss_seg = MutationLoss(
            pos_weight=getattr(config, 'pos_weight', None),
            smooth=getattr(config, 'dice_smooth', 1.0),
        ).to(self.device)

        lr = getattr(config, 'decoder_lr', None) or config.learning_rate
        params = list(self.networks['model'].parameters())
        self.optimizer = get_optimizer(
            [{'params': params, 'lr': lr}],
            config.optimizer,
            config.learning_rate,
            config.weight_decay,
        )
        self.scheduler = None

    def train_step(self, batch):
        ct = batch['ct'].float().to(self.device)
        pet = batch['pet'].float().to(self.device)
        mask = batch['mask'].float().to(self.device)
        preds = _forward(self.networks, ct, pet, mask.shape[-2:])
        loss, _ = self.loss_seg(preds, mask)
        return loss, preds[0], mask, {'loss_seg': loss.detach()}

    @torch.no_grad()
    def evaluate(self, loader, threshold=None):
        for v in self.networks.values():
            v.eval()
        th = threshold or getattr(self.config, 'eval_threshold', 0.5)
        m = SegmentationMetricsCIPA(threshold=th).to(self.device)
        m.reset()
        total_loss, n = 0.0, 0
        for batch in loader:
            ct = batch['ct'].float().to(self.device)
            pet = batch['pet'].float().to(self.device)
            mask = batch['mask'].float().to(self.device)
            preds = _forward(self.networks, ct, pet, mask.shape[-2:])
            loss, _ = self.loss_seg(preds, mask)
            total_loss += loss.item() * ct.size(0)
            n += ct.size(0)
            m.update(preds[0], mask)
        for v in self.networks.values():
            v.train()
        out = m.compute()
        out['total_loss'] = total_loss / max(n, 1)
        return out

    def save_checkpoint(self, path, epoch):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {k: v.state_dict() for k, v in self.networks.items()}
        ckpt['epoch'] = epoch
        ckpt['optimizer'] = self.optimizer.state_dict()
        torch.save(ckpt, path)
        print('Saved:', path)
