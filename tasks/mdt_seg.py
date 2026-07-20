import json
import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F


def dice_bce_loss(logits, mask):
    bce = F.binary_cross_entropy_with_logits(logits, mask)
    prob = torch.sigmoid(logits)
    inter = (prob * mask).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3)) + 1e-6
    dice = 1 - ((2 * inter + 1) / den).mean()
    return dice + bce


def _flatten_grads(grads):
    flat = []
    for g in grads:
        if g is None:
            continue
        flat.append(g.reshape(-1))
    return torch.cat(flat) if flat else torch.zeros(1)


class MDTSegTeacher:
    def __init__(self, networks, config):
        self.networks = networks
        self.config = config
        self.model = networks['model']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(config.mixed_precision))
        self.global_batch_step = 0

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _amp_ctx(self):
        return torch.cuda.amp.autocast(enabled=bool(self.config.mixed_precision)) if torch.cuda.is_available() else nullcontext()

    def train_step(self, batch, forward_mode='full'):
        ct = batch['ct'].to(self.device, non_blocking=True)
        pet = batch['pet'].to(self.device, non_blocking=True) if forward_mode == 'full' else None
        mask = batch['mask'].to(self.device, non_blocking=True).float()
        with self._amp_ctx():
            logits = self.model(ct, pet=pet, forward_mode=forward_mode)
            loss = dice_bce_loss(logits, mask)
        return loss, logits, logits, {'loss_total': loss.detach(), 'loss_seg': loss.detach(), 'loss_boundary': torch.tensor(0.0, device=loss.device)}

    def _metrics(self, logits, mask):
        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).float()
        inter = (pred * mask).sum().item()
        union = ((pred + mask) > 0).float().sum().item()
        dice = (2 * inter + 1) / (pred.sum().item() + mask.sum().item() + 1)
        iou = (inter + 1) / (union + 1)
        acc = (pred == mask).float().mean().item()
        return dice, iou, acc, 0.0

    @torch.no_grad()
    def evaluate(self, loader, eval_mode='full', tag='val'):
        was_training = self.model.training
        self.model.eval()
        dices, ious, accs, hd95s = [], [], [], []
        for batch in loader:
            ct = batch['ct'].to(self.device, non_blocking=True)
            pet = batch['pet'].to(self.device, non_blocking=True) if eval_mode == 'full' else None
            mask = batch['mask'].to(self.device, non_blocking=True).float()
            logits = self.model(ct, pet=pet, forward_mode='full' if eval_mode == 'full' else 'missing')
            dice, iou, acc, hd95 = self._metrics(logits, mask)
            dices.append(dice)
            ious.append(iou)
            accs.append(acc)
            hd95s.append(hd95)
        self.model.train(was_training)
        return {'dice': float(np.mean(dices)), 'iou': float(np.mean(ious)), 'acc': float(np.mean(accs)), 'hd95': float(np.mean(hd95s)), 'total_loss': 0.0}

    def _module_param_grads(self, module):
        return [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]

    def gradient_diagnostics(self, batch):
        was_training = self.model.training
        self.model.eval()
        bn_states = []
        for m in self.model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                bn_states.append((m, m.track_running_stats))
                m.track_running_stats = False
        try:
            ct = batch['ct'].to(self.device, non_blocking=True)
            mask = batch['mask'].to(self.device, non_blocking=True).float()
            pet = batch['pet'].to(self.device, non_blocking=True)
            params_shared = list(self.model.enc_ct.parameters()) + list(self.model.ct_align.parameters()) + list(self.model.decoder.parameters())
            params_ct = list(self.model.enc_ct.parameters())
            params_align = list(self.model.ct_align.parameters())
            params_dec = list(self.model.decoder.parameters())
            with torch.cuda.amp.autocast(enabled=False):
                logits_full = self.model(ct, pet=pet, forward_mode='full')
                loss_full = dice_bce_loss(logits_full.float(), mask.float())
                logits_missing = self.model(ct, pet=None, forward_mode='missing')
                loss_missing = dice_bce_loss(logits_missing.float(), mask.float())
            g_full_shared = torch.autograd.grad(loss_full, params_shared, retain_graph=True, allow_unused=True)
            g_missing_shared = torch.autograd.grad(loss_missing, params_shared, retain_graph=True, allow_unused=True)
            g_full_ct = torch.autograd.grad(loss_full, params_ct, retain_graph=True, allow_unused=True)
            g_missing_ct = torch.autograd.grad(loss_missing, params_ct, retain_graph=True, allow_unused=True)
            g_full_align = torch.autograd.grad(loss_full, params_align, retain_graph=True, allow_unused=True)
            g_missing_align = torch.autograd.grad(loss_missing, params_align, retain_graph=True, allow_unused=True)
            g_full_dec = torch.autograd.grad(loss_full, params_dec, retain_graph=True, allow_unused=True)
            g_missing_dec = torch.autograd.grad(loss_missing, params_dec, retain_graph=True, allow_unused=True)
            full_vec = _flatten_grads(g_full_shared)
            missing_vec = _flatten_grads(g_missing_shared)
            eps = 1e-8
            cosine = float(torch.dot(full_vec, missing_vec) / (full_vec.norm() * missing_vec.norm() + eps))
            ct_cos = float(torch.dot(_flatten_grads(g_full_ct), _flatten_grads(g_missing_ct)) / (_flatten_grads(g_full_ct).norm() * _flatten_grads(g_missing_ct).norm() + eps))
            align_cos = float(torch.dot(_flatten_grads(g_full_align), _flatten_grads(g_missing_align)) / (_flatten_grads(g_full_align).norm() * _flatten_grads(g_missing_align).norm() + eps))
            dec_cos = float(torch.dot(_flatten_grads(g_full_dec), _flatten_grads(g_missing_dec)) / (_flatten_grads(g_full_dec).norm() * _flatten_grads(g_missing_dec).norm() + eps))
            stats = {
                'shared_grad_cosine_total': cosine,
                'ct_encoder_grad_cosine': ct_cos,
                'ct_alignment_grad_cosine': align_cos,
                'shared_decoder_grad_cosine': dec_cos,
                'diagnostic_full_grad_norm': float(full_vec.norm()),
                'diagnostic_missing_grad_norm': float(missing_vec.norm()),
                'diagnostic_grad_norm_ratio': float(full_vec.norm() / (missing_vec.norm() + eps)),
                'negative_cosine_layer_ratio': float(np.mean([ct_cos < 0, align_cos < 0, dec_cos < 0])),
            }
            return stats
        finally:
            for m, state in bn_states:
                m.track_running_stats = state
            self.model.train(was_training)

    def save_checkpoint(self, path, epoch, best_joint=None, best_full=None, best_missing=None, best_joint_epoch=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            'epoch': epoch,
            'global_batch_step': self.global_batch_step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': None if self.scheduler is None else self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best_joint': best_joint,
            'best_full': best_full,
            'best_missing': best_missing,
            'best_joint_epoch': best_joint_epoch,
            'seed': getattr(self.config, 'random_state', None),
            'config': vars(self.config),
        }
        torch.save(payload, path)
