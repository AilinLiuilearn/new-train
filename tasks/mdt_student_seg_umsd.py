# -*- coding: utf-8 -*-
"""Student distillation task for UMSD x4."""

import os
import torch
import torch.nn.functional as F

from models.umsd import hsic_loss
from tasks.mdt_seg import _teacher_forward
from utils.loss_seg import DiceBCELoss
from utils.metrics_seg import SegmentationMetricsCIPA
from utils.optimization import get_optimizer


def _cosine_distill(a, b):
    """BCHW 或任意形状：展平后做余弦蒸馏。"""
    a = F.normalize(a.flatten(1), dim=1)
    b = F.normalize(b.detach().flatten(1), dim=1)
    return (1.0 - (a * b).sum(dim=1)).mean()


class MDTSegStudent:
    def __init__(self, student_networks, teacher_networks, config, teacher_config=None):
        self.config = config
        self.teacher_config = teacher_config if teacher_config is not None else config
        self.device = torch.device("cuda", config.gpus[0] if isinstance(config.gpus[0], int) else int(config.gpus[0]))
        self.networks = {k: v for k, v in student_networks.items() if v is not None}
        self.teacher = {k: v for k, v in teacher_networks.items() if v is not None}
        for v in self.networks.values():
            v.to(self.device)
        for v in self.teacher.values():
            v.to(self.device)
            v.eval()
        for n in self.teacher.values():
            for p in n.parameters():
                p.requires_grad = False
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler("cuda") if config.mixed_precision else None
        self.seg_metrics = SegmentationMetricsCIPA(
            threshold=getattr(config, "eval_threshold", 0.5),
        ).to(self.device)
        self.loss_seg = DiceBCELoss(
            smooth=getattr(self.config, "dice_smooth", 1.0),
            pos_weight=getattr(self.config, "pos_weight", None),
            bce_weight=getattr(self.config, "bce_weight", 1.0),
            dice_weight=getattr(self.config, "dice_weight", 1.0),
        )
        self._build_optimizer()

    def _build_optimizer(self):
        params = []
        for name, net in self.networks.items():
            lr = self.config.learning_rate * 0.1 if name.startswith(("segmentor", "sumsd_")) else self.config.learning_rate
            params.append({"params": net.parameters(), "lr": lr})
        self.optimizer = get_optimizer(params, self.config.optimizer, self.config.learning_rate, self.config.weight_decay)

    def _student_forward(self, ct, target_size):
        feats = self.networks["extractor"](ct, return_list=True)
        h = feats[-1]
        if self.networks.get("projector") is not None:
            h = self.networks["projector"](h)

        f0, zg0, zs0 = self.networks["sumsd_0"](feats[0])
        f1, zg1, zs1 = self.networks["sumsd_1"](feats[1])
        f2, zg2, zs2 = self.networks["sumsd_2"](feats[2])
        f3, zg3, zs3 = self.networks["sumsd_3"](h)
        logit = self.networks["segmentor"]([f0, f1, f2, f3], target_size=target_size)
        return logit, [zg0, zg1, zg2, zg3], [zs0, zs1, zs2, zs3], [f0, f1, f2, f3]

    @torch.no_grad()
    def _teacher_forward(self, ct, pet, target_size):
        return _teacher_forward(self.teacher, ct, pet, target_size, self.teacher_config)

    def train_step(self, batch_paired=None, batch_mri=None):
        alpha_distill = getattr(self.config, "alpha_distill", 1.0)
        alpha_hsic_s = getattr(self.config, "alpha_hsic_s", 0.05)
        distill_ws = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device)
        distill_ws = distill_ws / distill_ws.sum()

        loss_seg_sum = 0.0
        loss_distill_sum = 0.0
        loss_hsic_s_sum = 0.0
        n_total = 0

        if batch_paired is not None:
            ct = batch_paired["ct"].float().to(self.device)
            pet = batch_paired["pet"].float().to(self.device)
            mask = batch_paired["mask"].float().to(self.device)
            n_p = ct.size(0)

            tout = self._teacher_forward(ct, pet, mask.shape[-2:])
            logit_s, z_general_s, z_specific_s, feat_s = self._student_forward(ct, mask.shape[-2:])
            loss_seg_p = self.loss_seg(logit_s, mask)

            # 全4层蒸馏：z_general_i_s ↔ z_shared_i_t（选择性解剖低频蒸馏）
            loss_distill_p = tout['seg_logit'].new_zeros(1).squeeze() if 'seg_logit' in tout else ct.new_zeros(1).squeeze()
            for i in range(4):
                loss_distill_p = loss_distill_p + distill_ws[i] * _cosine_distill(z_general_s[i], tout[f'z_shared_{i}'])

            loss_hsic_s_p = hsic_loss(z_general_s[3], z_specific_s[3])
            loss_paired_total = self.config.alpha_seg * loss_seg_p + alpha_distill * loss_distill_p + alpha_hsic_s * loss_hsic_s_p

            loss_seg_sum += loss_seg_p.item() * n_p
            loss_distill_sum += float(loss_distill_p.item()) * n_p
            loss_hsic_s_sum += float(loss_hsic_s_p.item()) * n_p
            n_total += n_p
        else:
            loss_paired_total = None
            n_p = 0

        if batch_mri is not None:
            ct_n = batch_mri["ct"].float().to(self.device)
            mask_n = batch_mri["mask"].float().to(self.device)
            n_m = ct_n.size(0)
            logit_n, _, _, _ = self._student_forward(ct_n, mask_n.shape[-2:])
            loss_seg_m = self.loss_seg(logit_n, mask_n)
            loss_seg_sum += loss_seg_m.item() * n_m
            n_total += n_m
        else:
            n_m = 0
            loss_seg_m = None

        if n_total == 0:
            dummy = torch.tensor(0.0, device=self.device)
            return dummy, None, None, {"loss_seg": dummy, "loss_distill": dummy, "loss_hsic_s": dummy}

        if batch_paired is not None and batch_mri is not None:
            total_loss = loss_paired_total * (n_p / n_total) + self.config.alpha_seg * loss_seg_m * (n_m / n_total)
            logit_out, mask_out = logit_s, batch_paired["mask"].float().to(self.device)
        elif batch_paired is not None:
            total_loss = loss_paired_total
            logit_out, mask_out = logit_s, batch_paired["mask"].float().to(self.device)
        else:
            total_loss = self.config.alpha_seg * loss_seg_m
            logit_out, mask_out = logit_n, batch_mri["mask"].float().to(self.device)

        scale = 1.0 / n_total
        loss_dict = {
            "loss_seg": torch.tensor(loss_seg_sum * scale, device=self.device),
            "loss_distill": torch.tensor(loss_distill_sum * scale, device=self.device),
            "loss_hsic_s": torch.tensor(loss_hsic_s_sum * scale, device=self.device),
        }
        return total_loss, logit_out, mask_out, loss_dict

    @torch.no_grad()
    def evaluate(self, loader):
        for v in self.networks.values():
            v.eval()
        self.seg_metrics.reset()
        total_loss = 0.0
        n = 0
        for batch in loader:
            ct = batch["ct"].float().to(self.device)
            mask = batch["mask"].float().to(self.device)
            logit, _, _, _ = self._student_forward(ct, mask.shape[-2:])
            loss_seg = self.loss_seg(logit, mask)
            total_loss += loss_seg.item() * ct.size(0)
            n += ct.size(0)
            self.seg_metrics.update(logit, mask)
        for v in self.networks.values():
            v.train()
        metrics = self.seg_metrics.compute()
        metrics["total_loss"] = total_loss / max(n, 1)
        return metrics

    def save_checkpoint(self, path, epoch):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {k: v.state_dict() for k, v in self.networks.items()}
        ckpt["epoch"] = epoch
        ckpt["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            ckpt["scheduler"] = self.scheduler.state_dict()
        torch.save(ckpt, path)
        print("保存 checkpoint:", path)
