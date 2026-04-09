# -*- coding: utf-8 -*-
"""
CIPA VMamba baseline 教师分割任务：双流 VMamba + 简单 add 融合，仅分割损失。
"""

import os
import torch

from utils.loss_seg import DiceBCELoss
from utils.metrics_seg import SegmentationMetricsCIPA
from utils.optimization import get_optimizer, get_cosine_scheduler


class CIPAVMambaBaselineTeacherSeg:
    def __init__(self, networks, config):
        self.networks = {k: v for k, v in networks.items() if v is not None}
        self.config = config
        self.device = torch.device('cuda', config.gpus[0] if isinstance(config.gpus[0], int) else int(config.gpus[0]))
        for v in self.networks.values():
            v.to(self.device)
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None
        self.seg_metrics = SegmentationMetricsCIPA(threshold=0.5).to(self.device)
        self.loss_seg = DiceBCELoss(
            smooth=getattr(config, 'dice_smooth', 1.0),
            pos_weight=getattr(config, 'pos_weight', None),
            bce_weight=getattr(config, 'bce_weight', 1.0),
            dice_weight=getattr(config, 'dice_weight', 1.0),
        )
        self._build_optimizer()

    def _build_optimizer(self):
        model = self.networks['model']
        self.optimizer = get_optimizer(
            list(model.parameters()),
            self.config.optimizer,
            self.config.learning_rate,
            self.config.weight_decay,
        )
        self.scheduler = None

    def train_step(self, batch):
        ct = batch['ct'].float().to(self.device)
        pet = batch['pet'].float().to(self.device)
        mask = batch['mask'].float().to(self.device)

        seg_logit = self.networks['model'](ct, pet)
        loss_seg = self.loss_seg(seg_logit, mask)
        return loss_seg, seg_logit, mask, {'loss_seg': loss_seg}

    @torch.no_grad()
    def evaluate(self, loader):
        for v in self.networks.values():
            v.eval()
        self.seg_metrics.reset()
        total_loss = 0.0
        n = 0
        for batch in loader:
            ct = batch['ct'].float().to(self.device)
            pet = batch['pet'].float().to(self.device)
            mask = batch['mask'].float().to(self.device)
            seg_logit = self.networks['model'](ct, pet)
            loss_seg = self.loss_seg(seg_logit, mask)
            total_loss += loss_seg.item() * ct.size(0)
            n += ct.size(0)
            self.seg_metrics.update(seg_logit, mask)
        for v in self.networks.values():
            v.train()
        metrics = self.seg_metrics.compute()
        metrics['total_loss'] = total_loss / max(n, 1)
        return metrics

    def save_checkpoint(self, path, epoch):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {'model': self.networks['model'].state_dict(), 'epoch': epoch}
        if self.optimizer is not None:
            ckpt['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            ckpt['scheduler'] = self.scheduler.state_dict()
        torch.save(ckpt, path)
        print("保存 checkpoint (CIPA VMamba baseline 教师):", path)
