# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics_seg import SegmentationMetricsCIPA
from utils.optimization import get_optimizer


def _prepare_target(logits, target):
    if target.dim() == 3:
        target = target.unsqueeze(1)
    target = target.float()
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target, size=logits.shape[-2:], mode='nearest')
    return target


def _compute_overlap_alpha(intersection_sum, ref_sum, smooth):
    dis = torch.pow((intersection_sum - ref_sum) / 2.0, 2)
    alpha = (torch.minimum(intersection_sum, ref_sum) + dis + smooth) / (
        torch.maximum(intersection_sum, ref_sum) + dis + smooth
    )
    return alpha


class WeightedIoULoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        target = _prepare_target(logits, target)
        pred = torch.sigmoid(logits)
        intersection_sum = (pred * target).sum(dim=(1, 2, 3))
        pred_sum = pred.sum(dim=(1, 2, 3))
        target_sum = target.sum(dim=(1, 2, 3))
        union_sum = pred_sum + target_sum - intersection_sum
        alpha = _compute_overlap_alpha(intersection_sum, union_sum, self.smooth)
        loss = 1.0 - alpha * (intersection_sum + self.smooth) / (union_sum + self.smooth)
        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        target = _prepare_target(logits, target)
        pred = torch.sigmoid(logits)
        intersection_sum = (pred * target).sum(dim=(1, 2, 3))
        pred_sum = pred.sum(dim=(1, 2, 3))
        target_sum = target.sum(dim=(1, 2, 3))
        denom_sum = pred_sum + target_sum
        alpha = _compute_overlap_alpha(intersection_sum, denom_sum, self.smooth)
        loss = 1.0 - alpha * (2.0 * intersection_sum + self.smooth) / (denom_sum + self.smooth)
        return loss.mean()


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, target, pixel_weight=None):
        target = _prepare_target(logits, target)
        pw = self.pos_weight
        if pw is not None and isinstance(pw, (int, float)):
            pw = torch.tensor(pw, dtype=logits.dtype, device=logits.device)
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction='none')
        if pixel_weight is not None:
            if pixel_weight.shape[-2:] != loss.shape[-2:]:
                pixel_weight = F.interpolate(pixel_weight, size=loss.shape[-2:], mode='bilinear', align_corners=False)
            loss = loss * pixel_weight.to(device=loss.device, dtype=loss.dtype)
        loss = loss.mean(dim=(1, 2, 3))
        return loss.mean()


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, target):
        target = _prepare_target(logits, target)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        prob = torch.sigmoid(logits)
        pt = prob * target + (1.0 - prob) * (1.0 - target)
        focal_factor = (1.0 - pt).pow(self.gamma)

        if self.pos_weight is not None:
            if isinstance(self.pos_weight, (int, float)):
                pw = torch.tensor(self.pos_weight, dtype=logits.dtype, device=logits.device)
            else:
                pw = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
            alpha = target * pw + (1.0 - target)
            loss = alpha * focal_factor * bce
        else:
            loss = focal_factor * bce
        loss = loss.mean(dim=(1, 2, 3))
        return loss.mean()


class TeacherSegLoss(nn.Module):
    def __init__(
        self,
        loss_type='bce_iou',
        pos_weight=None,
        smooth=1.0,
        bce_weight=1.0,
        dice_weight=1.0,
        iou_weight=1.0,
        focal_weight=1.0,
        focal_gamma=2.0,
        p_sum_weights=(0.5, 0.2, 0.2, 0.1),
        p_sum_loss_weight=0.3,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.focal_weight = focal_weight
        self.p_sum_weights = tuple(float(x) for x in p_sum_weights)
        self.p_sum_loss_weight = float(p_sum_loss_weight)

        self.bce = WeightedBCELoss(pos_weight=pos_weight)
        self.iou = WeightedIoULoss(smooth=smooth)
        self.dice = DiceLoss(smooth=smooth)
        self.focal = BinaryFocalLoss(gamma=focal_gamma, pos_weight=pos_weight)

    def _single(self, pred, target, pixel_weight=None):
        parts = {}
        if self.loss_type == 'bce_iou':
            parts['bce'] = self.bce_weight * self.bce(pred, target, pixel_weight=pixel_weight)
            parts['iou'] = self.iou_weight * self.iou(pred, target)
        elif self.loss_type == 'bce_dice':
            parts['bce'] = self.bce_weight * self.bce(pred, target, pixel_weight=pixel_weight)
            parts['dice'] = self.dice_weight * self.dice(pred, target)
        elif self.loss_type == 'bce_dice_focal':
            parts['bce'] = self.bce_weight * self.bce(pred, target, pixel_weight=pixel_weight)
            parts['dice'] = self.dice_weight * self.dice(pred, target)
            parts['focal'] = self.focal_weight * self.focal(pred, target)
        else:
            raise ValueError(f'Unsupported loss_type: {self.loss_type}')
        total = sum(parts.values())
        return total, parts

    def _build_p_sum(self, preds):
        weights = self.p_sum_weights
        if len(weights) < len(preds):
            weights = weights + (0.0,) * (len(preds) - len(weights))
        elif len(weights) > len(preds):
            weights = weights[:len(preds)]
        psum = preds[0] * weights[0]
        for p, w in zip(preds[1:], weights[1:]):
            psum = psum + p * w
        return psum

    def forward(self, preds, target, pixel_weight=None):
        p1 = preds[0]
        psum = self._build_p_sum(preds)

        loss_p1, parts_p1 = self._single(p1, target, pixel_weight=pixel_weight)
        loss_sum, parts_sum = self._single(psum, target, pixel_weight=pixel_weight)
        total = loss_p1 + self.p_sum_loss_weight * loss_sum

        stats = {
            'loss_p1': loss_p1.detach(),
            'loss_sum': loss_sum.detach(),
        }
        for name, value in parts_p1.items():
            stats[f'{name}_p1'] = value.detach()
        for name, value in parts_sum.items():
            stats[f'{name}_sum'] = value.detach()
        return total, stats


import copy


def _forward(nets, ct, pet, target_size):
    return nets['model'](ct, pet, target_size=target_size)


class MDTSegTeacher:
    def __init__(self, networks, config):
        self.networks = {k: v for k, v in networks.items() if v is not None}
        self.config = config
        self.device = torch.device('cuda', int(config.gpus[0]))
        for v in self.networks.values():
            v.to(self.device)

        self.ema_decay = float(getattr(config, 'ema_decay', 0.999))
        self.ema_warmup_epochs = int(getattr(config, 'ema_warmup_epochs', 3))
        self.use_ema = self.ema_decay > 0
        self._ema_step_count = 0
        self._current_epoch = 0
        if self.use_ema:
            self.ema_model = copy.deepcopy(self.networks['model'])
            self.ema_model.to(self.device)
            self.ema_model.eval()
            for p in self.ema_model.parameters():
                p.requires_grad = False
            self._ema_initialized = False
            print(f'[+] EMA enabled, decay={self.ema_decay}, warmup_epochs={self.ema_warmup_epochs}')
        else:
            self.ema_model = None
            self._ema_initialized = True

        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None
        self.loss_seg = TeacherSegLoss(
            loss_type=getattr(config, 'loss_type', 'bce_iou'),
            pos_weight=getattr(config, 'pos_weight', None),
            smooth=getattr(config, 'dice_smooth', 1.0),
            bce_weight=getattr(config, 'bce_weight', 1.0),
            dice_weight=getattr(config, 'dice_weight', 1.0),
            iou_weight=getattr(config, 'iou_weight', 1.0),
            focal_weight=getattr(config, 'focal_weight', 1.0),
            focal_gamma=getattr(config, 'focal_gamma', 2.0),
            p_sum_weights=getattr(config, 'p_sum_weights', (0.5, 0.2, 0.2, 0.1)),
            p_sum_loss_weight=getattr(config, 'p_sum_loss_weight', 0.3),
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

    @torch.no_grad()
    def update_ema(self):
        if not self.use_ema:
            return
        if not self._ema_initialized:
            for ema_p, model_p in zip(self.ema_model.parameters(), self.networks['model'].parameters()):
                ema_p.data.copy_(model_p.data)
            for ema_b, model_b in zip(self.ema_model.buffers(), self.networks['model'].buffers()):
                ema_b.data.copy_(model_b.data)
            self._ema_initialized = True
            return
        self._ema_step_count += 1
        alpha = min(self.ema_decay, 1.0 - 1.0 / (self._ema_step_count + 1))
        for ema_p, model_p in zip(self.ema_model.parameters(), self.networks['model'].parameters()):
            ema_p.data.mul_(alpha).add_(model_p.data, alpha=1.0 - alpha)
        for ema_b, model_b in zip(self.ema_model.buffers(), self.networks['model'].buffers()):
            ema_b.data.copy_(model_b.data)

    def set_epoch(self, epoch):
        self._current_epoch = epoch

    def _compute_total_loss(self, outputs, mask, pixel_weight=None):
        preds = outputs['preds'] if isinstance(outputs, dict) else outputs
        loss_seg, loss_stats = self.loss_seg(preds, mask, pixel_weight=pixel_weight)
        loss_dict = {
            'loss_seg': loss_seg.detach(),
            'loss_total': loss_seg.detach(),
        }
        loss_dict.update(loss_stats)
        return loss_seg, preds, loss_dict

    def train_step(self, batch):
        ct = batch['ct'].float().to(self.device)
        pet = batch['pet'].float().to(self.device)
        mask = batch['mask'].float().to(self.device)
        outputs = _forward(self.networks, ct, pet, mask.shape[-2:])
        loss, preds, loss_dict = self._compute_total_loss(outputs, mask)
        return loss, preds[0], mask, loss_dict

    def _get_eval_model(self):
        if self.use_ema and self.ema_model is not None and self._current_epoch > self.ema_warmup_epochs:
            return self.ema_model
        return self.networks['model']

    @torch.no_grad()
    def evaluate(self, loader, threshold=None, use_ema=True):
        use_ema_actual = use_ema and self.use_ema and self._current_epoch > self.ema_warmup_epochs
        eval_model = self.ema_model if use_ema_actual else self.networks['model']
        eval_model.eval()
        th = threshold or getattr(self.config, 'eval_threshold', 0.5)
        m = SegmentationMetricsCIPA(threshold=th).to(self.device)
        m.reset()
        total_loss, n = 0.0, 0
        for batch in loader:
            ct = batch['ct'].float().to(self.device)
            pet = batch['pet'].float().to(self.device)
            mask = batch['mask'].float().to(self.device)
            outputs = eval_model(ct, pet, target_size=mask.shape[-2:])
            preds = outputs['preds'] if isinstance(outputs, dict) else outputs
            loss_seg, _ = self.loss_seg(preds, mask)
            total_loss += loss_seg.item() * ct.size(0)
            n += ct.size(0)
            m.update(preds[0], mask)
        eval_model.train() if not use_ema_actual else None
        if not use_ema_actual:
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
        if self.use_ema and self.ema_model is not None:
            ckpt['ema_model'] = self.ema_model.state_dict()
        torch.save(ckpt, path)
        print('Saved:', path)
