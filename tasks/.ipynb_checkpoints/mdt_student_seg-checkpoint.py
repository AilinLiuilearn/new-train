# -*- coding: utf-8 -*-
"""
MDT 学生分割任务：仅输入 CT，多尺度 encoder + FPN 解码（与 SegModel 一致，无解耦）。
支持 mkd 式双 DataLoader：batch_paired（配对→蒸馏+seg）、batch_mri（缺失 PET→仅 seg）。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.loss_seg import DiceBCELoss
from utils.metrics_seg import SegmentationMetricsCIPA
from utils.optimization import get_optimizer
from utils.loss_mdt import flatten_feature


class MDTSegStudent:
    """学生：单模态 CT，多尺度 + FPN；支持双 DataLoader（paired + mri_only）与 mkd 一致。"""

    def __init__(self, student_networks, teacher_networks, config):
        self.config = config
        self.device = torch.device('cuda', config.gpus[0] if isinstance(config.gpus[0], int) else int(config.gpus[0]))
        self.networks = {k: v for k, v in student_networks.items() if v is not None}
        for v in self.networks.values():
            v.to(self.device)
        self.teacher = {k: v for k, v in teacher_networks.items() if v is not None}
        for v in self.teacher.values():
            v.to(self.device)
            v.eval()
        for p in self.teacher.values():
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
        # 特征蒸馏用 L2 归一化 + 余弦距离，避免教师/学生特征尺度差导致 MSE 爆炸、梯度冲垮分割头
        self.loss_feat_fn = nn.CosineEmbeddingLoss(reduction='mean')
        self.loss_logit_fn = nn.MSELoss()

    def _build_optimizer(self):
        params = []
        for name, net in self.networks.items():
            lr = self.config.learning_rate * 0.1 if name.startswith(('segmentor', 'feat_distill_proj')) else self.config.learning_rate
            params.append({'params': net.parameters(), 'lr': lr})
        self.optimizer = get_optimizer(params, self.config.optimizer, self.config.learning_rate, self.config.weight_decay)
        self.scheduler = None

    def _student_forward(self, ct, target_size):
        """多尺度 encoder + FPN，返回用于特征蒸馏的投影特征与分割 logit。"""
        feats = self.networks['extractor_mri'](ct, return_list=True)
        logit_s = self.networks['segmentor'](feats, target_size=target_size)
        feat_deep = feats[-1]
        feat_proj = self.networks['feat_distill_proj'](feat_deep)
        return feat_proj, logit_s

    @torch.no_grad()
    def _teacher_forward_full(self, ct, pet, target_size):
        """教师双模态：FPN logit + z_mri_g, z_mri（仅用于配对 batch）。"""
        feats_mri = self.teacher['extractor_mri'](ct, return_list=True)
        feats_pet = self.teacher['extractor_pet'](pet, return_list=True)
        h_mri = feats_mri[-1]
        h_pet = feats_pet[-1]
        if self.teacher.get('projector_mri') is not None:
            h_mri = self.teacher['projector_mri'](h_mri)
            h_pet = self.teacher['projector_pet'](h_pet)
        z_mri_g = self.teacher['encoder_general'](h_mri)
        z_pet_g = self.teacher['encoder_general'](h_pet)
        z_mri = self.teacher['encoder_mri'](h_mri)
        z_pet = self.teacher['encoder_pet'](h_pet)
        if getattr(self.config, 'use_specific', True):
            fusion3 = torch.cat([z_mri_g, z_pet_g, z_mri, z_pet], dim=1)
        else:
            fusion3 = z_mri_g + z_pet_g
        fpn_input_list = [feats_mri[0], feats_mri[1], feats_mri[2], fusion3]
        logit_t = self.teacher['segmentor'](fpn_input_list, target_size=target_size)
        return z_mri_g, z_mri, logit_t

    def train_step(self, batch_paired=None, batch_mri=None):
        """
        双 DataLoader 模式（对齐 mkd）：batch_paired 有 PET（蒸馏+seg），batch_mri 缺失 PET（仅 seg）。
        若仅传一个，则只算对应分支。
        """
        alpha_feat = getattr(self.config, 'alpha_feat', 0.5)
        alpha_logit = getattr(self.config, 'alpha_logit', 0.5)
        loss_seg_sum, loss_feat_sum, loss_logit_sum, n_total = 0.0, 0.0, 0.0, 0

        if batch_paired is not None:
            ct = batch_paired['ct'].float().to(self.device)
            pet = batch_paired['pet'].float().to(self.device)
            mask = batch_paired['mask'].float().to(self.device)
            target_size = mask.shape[-2:]
            n_p = ct.size(0)
            z_mri_g_t, z_mri_t, logit_t = self._teacher_forward_full(ct, pet, target_size)
            feat_proj_s, logit_s = self._student_forward(ct, target_size)
            loss_seg_p = self.loss_seg(logit_s, mask)
            z_teacher = torch.cat([z_mri_g_t, z_mri_t], dim=1)
            flat_s = flatten_feature(feat_proj_s)
            flat_t = flatten_feature(z_teacher)
            # 归一化后按余弦相似度监督，避免尺度差导致 MSE 爆炸（loss 有界）
            target_sim = torch.ones(n_p, device=flat_s.device, dtype=flat_s.dtype)
            loss_feat_p = self.loss_feat_fn(flat_s, flat_t, target_sim)
            loss_logit_p = self.loss_logit_fn(torch.sigmoid(logit_s), torch.sigmoid(logit_t))
            loss_seg_sum += loss_seg_p.item() * n_p
            loss_feat_sum += loss_feat_p.item() * n_p
            loss_logit_sum += loss_logit_p.item() * n_p
            n_total += n_p
        else:
            n_p = 0
            loss_seg_p = loss_feat_p = loss_logit_p = None

        if batch_mri is not None:
            ct_n = batch_mri['ct'].float().to(self.device)
            mask_n = batch_mri['mask'].float().to(self.device)
            target_size_n = mask_n.shape[-2:]
            n_m = ct_n.size(0)
            _, logit_s_n = self._student_forward(ct_n, target_size_n)
            loss_seg_m = self.loss_seg(logit_s_n, mask_n)
            loss_seg_sum += loss_seg_m.item() * n_m
            n_total += n_m
        else:
            n_m = 0
            loss_seg_m = None

        if n_total == 0:
            total_loss = torch.tensor(0.0, device=self.device)
            loss_dict_t = {
                'loss_seg': torch.tensor(0.0, device=self.device),
                'loss_feat': torch.tensor(0.0, device=self.device),
                'loss_logit': torch.tensor(0.0, device=self.device),
            }
            return total_loss, None, None, loss_dict_t

        scale = 1.0 / n_total
        if batch_paired is not None and batch_mri is not None:
            total_loss = (
                (self.config.alpha_seg * loss_seg_p + alpha_feat * loss_feat_p + alpha_logit * loss_logit_p) * (n_p / n_total)
                + self.config.alpha_seg * loss_seg_m * (n_m / n_total)
            )
        elif batch_paired is not None:
            total_loss = self.config.alpha_seg * loss_seg_p + alpha_feat * loss_feat_p + alpha_logit * loss_logit_p
        else:
            total_loss = self.config.alpha_seg * loss_seg_m

        loss_dict_t = {
            'loss_seg': torch.tensor(loss_seg_sum * scale, device=self.device),
            'loss_feat': torch.tensor(loss_feat_sum * scale, device=self.device),
            'loss_logit': torch.tensor(loss_logit_sum * scale, device=self.device),
        }
        if batch_paired is not None:
            _, logit_out = self._student_forward(batch_paired['ct'].float().to(self.device), batch_paired['mask'].shape[-2:])
            mask_out = batch_paired['mask'].float().to(self.device)
        elif batch_mri is not None:
            _, logit_out = self._student_forward(batch_mri['ct'].float().to(self.device), batch_mri['mask'].shape[-2:])
            mask_out = batch_mri['mask'].float().to(self.device)
        else:
            logit_out = mask_out = None
        return total_loss, logit_out, mask_out, loss_dict_t

    @torch.no_grad()
    def evaluate(self, loader):
        for v in self.networks.values():
            v.eval()
        self.seg_metrics.reset()
        total_loss = 0.0
        n = 0
        for batch in loader:
            ct = batch['ct'].float().to(self.device)
            mask = batch['mask'].float().to(self.device)
            _, logit_s = self._student_forward(ct, mask.shape[-2:])
            loss_seg = self.loss_seg(logit_s, mask)
            total_loss += loss_seg.item() * ct.size(0)
            n += ct.size(0)
            self.seg_metrics.update(logit_s, mask)
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
        print("保存 checkpoint:", path)
