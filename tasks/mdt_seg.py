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


class BCEIoULoss(nn.Module):
    def __init__(self, pos_weight=None, smooth=1.0, bce_weight=1.0, iou_weight=1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight

    def forward(self, logits, target):
        target = _prepare_target(logits, target)
        pw = self.pos_weight
        if pw is not None and isinstance(pw, (int, float)):
            pw = torch.tensor(pw, dtype=logits.dtype, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        pred = torch.sigmoid(logits)
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
        iou = 1.0 - (intersection + self.smooth) / (union + self.smooth)
        iou = iou.mean()
        total = self.bce_weight * bce + self.iou_weight * iou
        stats = {
            'loss_bce': bce.detach(),
            'loss_iou': iou.detach(),
        }
        return total, stats


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
            import copy
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
        self.loss_seg = BCEIoULoss(
            pos_weight=getattr(config, 'pos_weight', None),
            smooth=getattr(config, 'iou_smooth', 1.0),
            bce_weight=getattr(config, 'bce_weight', 1.0),
            iou_weight=getattr(config, 'iou_weight', 1.0),
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
        pred = outputs['preds'] if isinstance(outputs, dict) else outputs
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        loss_seg, loss_stats = self.loss_seg(pred, mask)
        loss_dict = {
            'loss_seg': loss_seg.detach(),
            'loss_total': loss_seg.detach(),
        }
        loss_dict.update(loss_stats)
        return loss_seg, pred, loss_dict

    def train_step(self, batch):
        ct = batch['ct'].float().to(self.device)
        pet = batch['pet'].float().to(self.device)
        mask = batch['mask'].float().to(self.device)
        outputs = _forward(self.networks, ct, pet, mask.shape[-2:])
        loss, pred, loss_dict = self._compute_total_loss(outputs, mask)
        return loss, pred, mask, loss_dict

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
            pred = eval_model(ct, pet, target_size=mask.shape[-2:])
            loss_seg, _ = self.loss_seg(pred, mask)
            total_loss += loss_seg.item() * ct.size(0)
            n += ct.size(0)
            m.update(pred, mask)
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
