import json
import os
import random

import numpy as np
import torch

from utils.seg_losses import BCEDiceLoss
from utils.metrics_seg import SegmentationMetricsCIPA


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
        self.optimizer = self._build_optimizer(config)
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(config.mixed_precision))
        self.global_batch_step = 0
        self.criterion = BCEDiceLoss(smooth=config.loss_smooth, bce_weight=config.bce_weight, dice_weight=config.dice_weight)
        self.metrics = SegmentationMetricsCIPA()

    def _build_optimizer(self, config):
        """
        Two LR groups only:
          - Stage-1 modules (encoders / align / CPPI trainable / calibration / decoder)
          - TRDF fusion
        """
        old_lr = float(getattr(config, 'old_module_lr', 2e-5))
        trdf_lr = float(getattr(config, 'learning_rate', 8e-5))
        weight_decay = float(getattr(config, 'weight_decay', 1e-4))

        trdf_params = [p for p in self.model.fusion.parameters() if p.requires_grad]
        trdf_ids = {id(p) for p in trdf_params}
        old_params = [
            p for p in self.model.parameters()
            if p.requires_grad and id(p) not in trdf_ids
        ]
        if not trdf_params:
            raise RuntimeError('TRDF param group is empty; fusion module missing trainable params')
        if not old_params:
            raise RuntimeError('Stage-1 param group is empty')

        param_groups = [
            {'params': old_params, 'lr': old_lr, 'name': 'stage1_modules'},
            {'params': trdf_params, 'lr': trdf_lr, 'name': 'trdf'},
        ]
        return torch.optim.AdamW(param_groups, lr=trdf_lr, weight_decay=weight_decay)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def train_step(self, batch, forward_mode='full'):
        ct = batch['ct'].to(self.device, non_blocking=True)
        pet = batch['pet'].to(self.device, non_blocking=True)
        mask = batch['mask'].to(self.device, non_blocking=True).float()
        outputs = self.model(ct, pet=pet, mask=mask, forward_mode=forward_mode)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        loss, loss_stats = self.criterion(logits, mask)
        stats = {
            'loss_total': loss.detach(),
            'loss_seg': loss_stats.get('loss_dice', loss.detach()),
            'loss_boundary': torch.tensor(0.0, device=loss.device),
        }
        return loss, logits, outputs, stats

    @torch.no_grad()
    def evaluate(self, loader, eval_mode='full', tag='val'):
        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        sample_count = 0
        self.metrics.reset()
        for batch in loader:
            ct = batch['ct'].to(self.device, non_blocking=True)
            mask = batch['mask'].to(self.device, non_blocking=True).float()
            batch_size = ct.shape[0]
            if eval_mode == 'full':
                pet = batch['pet'].to(self.device, non_blocking=True)
                forward_mode = 'full'
                pet_available = None
            elif eval_mode == 'fixed_missing':
                pet = batch['pet'].to(self.device, non_blocking=True)
                forward_mode = 'missing'
                pet_available = None
            else:
                pet = batch['pet'].to(self.device, non_blocking=True)
                forward_mode = 'auto'
                pet_available = batch.get('pet_available')
                if pet_available is not None:
                    pet_available = pet_available.to(self.device, non_blocking=True)
            outputs = self.model(ct, pet=pet, pet_available=pet_available, forward_mode=forward_mode, mask=None)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            loss, _ = self.criterion(logits, mask)
            self.metrics.update(logits, mask)
            total_loss += float(loss) * batch_size
            sample_count += batch_size
        out = self.metrics.compute()
        out['total_loss'] = total_loss / max(1, sample_count)
        self.model.train(was_training)
        return out

    def _module_param_grads(self, module):
        return [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]

    def gradient_diagnostics(self, batch, max_samples=1):
        was_training = self.model.training
        self.model.eval()
        bn_states = []
        for m in self.model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                bn_states.append((m, m.track_running_stats))
                m.track_running_stats = False
        try:
            ct = batch['ct'][:max_samples].to(self.device, non_blocking=True)
            mask = batch['mask'][:max_samples].to(self.device, non_blocking=True).float()
            pet = batch['pet'][:max_samples].to(self.device, non_blocking=True)
            params_shared = list(self.model.enc_ct.parameters()) + list(self.model.ct_align.parameters()) + list(self.model.decoder.parameters())
            params_ct = list(self.model.enc_ct.parameters())
            params_align = list(self.model.ct_align.parameters())
            params_dec = list(self.model.decoder.parameters())
            outputs_full = self.model(ct, pet=pet, forward_mode='full', mask=None)
            outputs_missing = self.model(ct, pet=pet, forward_mode='missing', mask=None)
            logits_full = outputs_full['logits'] if isinstance(outputs_full, dict) else outputs_full
            logits_missing = outputs_missing['logits'] if isinstance(outputs_missing, dict) else outputs_missing
            loss_full, _ = self.criterion(logits_full.float(), mask.float())
            loss_missing, _ = self.criterion(logits_missing.float(), mask.float())
            g_full_shared = torch.autograd.grad(loss_full, params_shared, retain_graph=True, allow_unused=True)
            g_missing_shared = torch.autograd.grad(loss_missing, params_shared, retain_graph=True, allow_unused=True)
            g_full_ct = torch.autograd.grad(loss_full, params_ct, retain_graph=True, allow_unused=True)
            g_missing_ct = torch.autograd.grad(loss_missing, params_ct, retain_graph=True, allow_unused=True)
            g_full_align = torch.autograd.grad(loss_full, params_align, retain_graph=True, allow_unused=True)
            g_missing_align = torch.autograd.grad(loss_missing, params_align, retain_graph=True, allow_unused=True)
            g_full_dec = torch.autograd.grad(loss_full, params_dec, retain_graph=True, allow_unused=True)
            g_missing_dec = torch.autograd.grad(loss_missing, params_dec, retain_graph=True, allow_unused=True)

            def cos(a, b):
                a = _flatten_grads(a)
                b = _flatten_grads(b)
                eps = 1e-8
                return float(torch.dot(a, b) / (a.norm() * b.norm() + eps))

            full_vec = _flatten_grads(g_full_shared)
            missing_vec = _flatten_grads(g_missing_shared)
            stats = {
                'shared_grad_cosine_total': cos(g_full_shared, g_missing_shared),
                'ct_encoder_grad_cosine': cos(g_full_ct, g_missing_ct),
                'ct_alignment_grad_cosine': cos(g_full_align, g_missing_align),
                'shared_decoder_grad_cosine': cos(g_full_dec, g_missing_dec),
                'full_shared_grad_norm': float(full_vec.norm()),
                'missing_shared_grad_norm': float(missing_vec.norm()),
                'full_missing_grad_norm_ratio': float(full_vec.norm() / (missing_vec.norm() + 1e-8)),
                'negative_parameter_tensor_ratio': float(np.mean([x < 0 for x in [cos(g_full_ct, g_missing_ct), cos(g_full_align, g_missing_align), cos(g_full_dec, g_missing_dec)]])),
            }
            return stats
        finally:
            for m, state in bn_states:
                m.track_running_stats = state
            self.model.train(was_training)

    def save_checkpoint(self, path, epoch, best_joint=None, best_full=None, best_missing=None, best_joint_epoch=None, val_full=None, val_missing=None, joint_dice=None):
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
            'val_full': val_full,
            'val_missing': val_missing,
            'joint_dice': joint_dice,
            'random_state': getattr(self.config, 'random_state', None),
            'seed': getattr(self.config, 'random_state', None),
            'config': vars(self.config),
            'random_state_python': random.getstate(),
            'random_state_numpy': np.random.get_state(),
            'random_state_torch': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload['random_state_cuda'] = torch.cuda.get_rng_state_all()
        torch.save(payload, path)
