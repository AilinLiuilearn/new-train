# -*- coding: utf-8 -*-
"""
MDT+ 阶段：增强教师模型。学生向教师的 CT 分支传递知识。
- 损失：频域教师（seg + FDMF 辅助项）+ loss_kd_repr（h_mri 对齐 h_mri_s）
"""

import os
import torch

from utils.loss_seg import DiceBCELoss
from utils.metrics_seg import SegmentationMetricsCIPA
from utils.optimization import get_optimizer, get_cosine_scheduler
from utils.loss_mdt import SimCosineLoss, flatten_feature
from tasks.mdt_seg import _teacher_forward


class MDTSegPlus:
    def __init__(self, teacher_networks, student_networks, config):
        self.config = config
        self.device = torch.device('cuda', config.gpus[0] if isinstance(config.gpus[0], int) else int(config.gpus[0]))
        self.networks = {k: v for k, v in teacher_networks.items() if v is not None}
        self.student = {k: v for k, v in student_networks.items() if v is not None}
        for v in self.networks.values():
            v.to(self.device)
        for v in self.student.values():
            v.to(self.device)
            v.eval()
        for p in self.student.values():
            for x in p.parameters():
                x.requires_grad = False
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None
        self.seg_metrics = SegmentationMetricsCIPA(
            threshold=getattr(config, 'eval_threshold', 0.5),
        ).to(self.device)
        self._build_loss()
        self._build_optimizer()

    def _build_loss(self):
        self.loss_seg = DiceBCELoss(
            smooth=getattr(self.config, 'dice_smooth', 1.0),
            pos_weight=getattr(self.config, 'pos_weight', None),
            bce_weight=getattr(self.config, 'bce_weight', 1.0),
            dice_weight=getattr(self.config, 'dice_weight', 1.0),
        )
        self.loss_kd_repr_fn = SimCosineLoss()
        self.alpha_kd_repr = getattr(self.config, 'alpha_kd_repr', 1.0)

    def _build_optimizer(self):
        params = []
        for name, net in self.networks.items():
            lr = self.config.learning_rate * 0.1 if name.startswith(
                ('encoder_', 'decoder_', 'segmentor', 'frm_', 'ffm_', 'lfgf_', 'shallow_fuse_', 'umsd_')
            ) else self.config.learning_rate
            params.append({'params': net.parameters(), 'lr': lr})
        self.optimizer = get_optimizer(params, self.config.optimizer, self.config.learning_rate, self.config.weight_decay)
        self.scheduler = None

    def train_step(self, batch):
        ct = batch['ct'].float().to(self.device)
        pet = batch['pet'].float().to(self.device)
        mask = batch['mask'].float().to(self.device)
        target_size = mask.shape[-2:]
        out = _teacher_forward(
            self.networks, ct, pet, target_size, self.config,
            mask=mask,
        )
        h_mri = out['h_mri']

        with torch.no_grad():
            feats_s = self.student['extractor'](ct, return_list=True)
            h_mri_s = self.student['projector'](feats_s[-1])
        flat_t = flatten_feature(h_mri)
        flat_s = flatten_feature(h_mri_s)
        loss_kd_repr = self.loss_kd_repr_fn(flat_t, flat_s)

        loss_seg = self.loss_seg(out['seg_logit'], mask)
        total_loss = self.config.alpha_seg * loss_seg + self.alpha_kd_repr * loss_kd_repr
        loss_dict = {'loss_seg': loss_seg, 'loss_kd_repr': loss_kd_repr}
        lf_net = self.networks.get('lfgf_fusion')
        if lf_net is not None and getattr(lf_net, 'fdmf_mi_loss', None) is not None:
            a_mi = getattr(self.config, 'alpha_fdmf_mi', 0.01)
            b_low = getattr(self.config, 'alpha_fdmf_low_mi', 0.05)
            scale = getattr(self.config, 'fdmf_loss_scale', 1.0)
            loss_fdmf_aux = scale * (a_mi * lf_net.fdmf_mi_loss - b_low * lf_net.fdmf_low_mi_lb)
            total_loss = total_loss + loss_fdmf_aux
            loss_dict['loss_fdmf_aux'] = loss_fdmf_aux.detach()
            loss_dict['loss_fdmf_mi'] = lf_net.fdmf_mi_loss.detach()
            loss_dict['loss_fdmf_low_mi_lb'] = lf_net.fdmf_low_mi_lb.detach()

        return total_loss, out['seg_logit'], mask, loss_dict

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
            target_size = mask.shape[-2:]
            out = _teacher_forward(self.networks, ct, pet, target_size, self.config)
            seg_logit = out['seg_logit']
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
        ckpt = {k: v.state_dict() for k, v in self.networks.items()}
        ckpt['epoch'] = epoch
        ckpt['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            ckpt['scheduler'] = self.scheduler.state_dict()
        torch.save(ckpt, path)
        print("保存 checkpoint (教师):", path)
