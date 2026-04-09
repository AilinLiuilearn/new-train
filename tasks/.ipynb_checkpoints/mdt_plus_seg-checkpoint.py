# -*- coding: utf-8 -*-
"""
MDT+ 阶段：增强教师模型。学生向教师的 CT 分支传递知识。
- 训练对象：教师（可训练）
- 学生：冻结，提供 h_mri_s = projector(feats[-1]) 作为教师 h_mri 的蒸馏目标（对齐 MKD）
- 数据：配对双模态（complete），与教师/学生使用相同 missing_rate 划分，保持现实缺失率
- 损失：教师完整损失（seg + sim + diff + recon）+ loss_kd_repr（教师 h_mri 对齐学生 h_mri_s）
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.loss_seg import DiceBCELoss
from utils.metrics_seg import SegmentationMetricsCIPA
from utils.optimization import get_optimizer, get_cosine_scheduler
from utils.loss_mdt import SimCosineLoss, DiffCosineLoss, DiffFrobeniusLoss, DiffMSELoss, SimCMDLoss, flatten_feature


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
        self.seg_metrics = SegmentationMetricsCIPA(threshold=0.5).to(self.device)
        self._build_loss()
        self._build_optimizer()

    def _build_loss(self):
        self.loss_seg = DiceBCELoss(smooth=getattr(self.config, 'dice_smooth', 1.0))
        if self.config.loss_sim == 'cosine':
            self.loss_sim_fn = SimCosineLoss()
        elif self.config.loss_sim == 'cmd':
            self.loss_sim_fn = SimCMDLoss(n_moments=getattr(self.config, 'n_moments', 5))
        else:
            self.loss_sim_fn = SimCosineLoss()
        if self.config.loss_diff == 'cosine':
            self.loss_diff_fn = DiffCosineLoss()
        elif self.config.loss_diff == 'fro':
            self.loss_diff_fn = DiffFrobeniusLoss()
        else:
            self.loss_diff_fn = DiffMSELoss()
        self.loss_recon_fn = nn.MSELoss()
        self.loss_kd_repr_fn = SimCosineLoss()
        self.alpha_kd_repr = getattr(self.config, 'alpha_kd_repr', 1.0)

    def _build_optimizer(self):
        params = []
        for name, net in self.networks.items():
            lr = self.config.learning_rate * 0.1 if name.startswith(('encoder_', 'decoder_', 'segmentor')) else self.config.learning_rate
            params.append({'params': net.parameters(), 'lr': lr})
        self.optimizer = get_optimizer(params, self.config.optimizer, self.config.learning_rate, self.config.weight_decay)
        self.scheduler = None

    def train_step(self, batch):
        ct = batch['ct'].float().to(self.device)
        pet = batch['pet'].float().to(self.device)
        mask = batch['mask'].float().to(self.device)
        target_size = mask.shape[-2:]

        feats_mri = self.networks['extractor_mri'](ct, return_list=True)
        feats_pet = self.networks['extractor_pet'](pet, return_list=True)
        h_mri = feats_mri[-1]
        h_pet = feats_pet[-1]
        if self.networks.get('projector_mri') is not None:
            h_mri = self.networks['projector_mri'](h_mri)
            h_pet = self.networks['projector_pet'](h_pet)
        z_mri_g = self.networks['encoder_general'](h_mri)
        z_pet_g = self.networks['encoder_general'](h_pet)
        z_mri = self.networks['encoder_mri'](h_mri)
        z_pet = self.networks['encoder_pet'](h_pet)

        loss_sim = self.loss_sim_fn(flatten_feature(z_mri_g), flatten_feature(z_pet_g))
        loss_diff_spec = self.loss_diff_fn(flatten_feature(z_mri), flatten_feature(z_pet))
        loss_diff_mri = self.loss_diff_fn(flatten_feature(z_mri), flatten_feature(z_mri_g))
        loss_diff_pet = self.loss_diff_fn(flatten_feature(z_pet), flatten_feature(z_pet_g))
        h_mri_recon = self.networks['decoder_mri'](z_mri_g + z_mri)
        h_pet_recon = self.networks['decoder_pet'](z_pet_g + z_pet)
        loss_recon_mri = self.loss_recon_fn(h_mri_recon, h_mri)
        loss_recon_pet = self.loss_recon_fn(h_pet_recon, h_pet)

        if self.config.use_specific:
            fusion3 = torch.cat([z_mri_g, z_pet_g, z_mri, z_pet], dim=1)
        else:
            fusion3 = z_mri_g + z_pet_g
        fpn_input_list = [feats_mri[0], feats_mri[1], feats_mri[2], fusion3]
        seg_logit = self.networks['segmentor'](fpn_input_list, target_size=target_size)
        loss_seg = self.loss_seg(seg_logit, mask)

        with torch.no_grad():
            feats_s = self.student['extractor'](ct, return_list=True)
            h_mri_s = self.student['projector'](feats_s[-1])
        flat_t = flatten_feature(h_mri)
        flat_s = flatten_feature(h_mri_s)
        loss_kd_repr = self.loss_kd_repr_fn(flat_t, flat_s)

        total_loss = (
            self.config.alpha_seg * loss_seg
            + self.config.alpha_sim * loss_sim
            + self.config.alpha_diff * (loss_diff_spec + loss_diff_mri + loss_diff_pet)
            + self.config.alpha_recon * (loss_recon_mri + loss_recon_pet)
            + self.alpha_kd_repr * loss_kd_repr
        )
        loss_dict = {
            'loss_seg': loss_seg, 'loss_sim': loss_sim,
            'loss_diff_spec': loss_diff_spec, 'loss_diff_mri': loss_diff_mri, 'loss_diff_pet': loss_diff_pet,
            'loss_recon_mri': loss_recon_mri, 'loss_recon_pet': loss_recon_pet,
            'loss_kd_repr': loss_kd_repr,
        }
        return total_loss, seg_logit, mask, loss_dict

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
            feats_mri = self.networks['extractor_mri'](ct, return_list=True)
            feats_pet = self.networks['extractor_pet'](pet, return_list=True)
            h_mri = feats_mri[-1]
            h_pet = feats_pet[-1]
            if self.networks.get('projector_mri') is not None:
                h_mri = self.networks['projector_mri'](h_mri)
                h_pet = self.networks['projector_pet'](h_pet)
            z_mri_g = self.networks['encoder_general'](h_mri)
            z_pet_g = self.networks['encoder_general'](h_pet)
            z_mri = self.networks['encoder_mri'](h_mri)
            z_pet = self.networks['encoder_pet'](h_pet)
            if self.config.use_specific:
                fusion3 = torch.cat([z_mri_g, z_pet_g, z_mri, z_pet], dim=1)
            else:
                fusion3 = z_mri_g + z_pet_g
            fpn_input_list = [feats_mri[0], feats_mri[1], feats_mri[2], fusion3]
            seg_logit = self.networks['segmentor'](fpn_input_list, target_size=target_size)
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
