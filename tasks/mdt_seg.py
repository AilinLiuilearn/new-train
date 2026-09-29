import json
import os
import random

import numpy as np
import torch

from utils.seg_losses import BCEDiceLoss
from utils.metrics_seg import SegmentationMetricsCIPA
from utils.ema import ModelEMA


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
        self.criterion = BCEDiceLoss(smooth=config.loss_smooth, bce_weight=config.bce_weight, dice_weight=config.dice_weight)
        self.metrics = SegmentationMetricsCIPA()
        self.ema = None
        self.ema_active = False
        self.ema_start_epoch = int(getattr(config, 'ema_start_epoch', 0))
        if bool(getattr(config, 'ema_enabled', False)):
            self.ema = ModelEMA(
                self.model,
                decay=float(getattr(config, 'ema_decay', 0.999)),
                warmup=bool(getattr(config, 'ema_decay_warmup', True)),
                device=self.device,
            )
            self.ema_active = self.ema_start_epoch <= 0

    def begin_epoch(self, epoch):
        """Activate the EMA after the warmup epochs, hard-syncing on the switch."""
        if self.ema is None or self.ema_active:
            return
        if int(epoch) > self.ema_start_epoch:
            self.ema.reset(self.model)
            self.ema_active = True

    def update_ema(self):
        if self.ema is not None and self.ema_active:
            return self.ema.update(self.model)
        return None

    def eval_model(self):
        """Model used for evaluation: EMA copy when active, else the model."""
        if self.ema is not None and self.ema_active:
            return self.ema.model
        return self.model

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def train_step(self, batch):
        ct = batch['ct'].to(self.device, non_blocking=True)
        pet = batch['pet'].to(self.device, non_blocking=True)
        mask = batch['mask'].to(self.device, non_blocking=True).float()
        outputs = self.model(ct, pet=pet)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        loss, loss_stats = self.criterion(logits, mask)
        stats = {
            'loss_total': loss.detach(),
            'loss_seg': loss_stats.get('loss_dice', loss.detach()),
        }
        return loss, logits, outputs, stats

    @torch.no_grad()
    def evaluate(self, loader, tag='val', model=None):
        active = model if model is not None else self.model
        was_training = self.model.training
        active.eval()
        if model is None:
            self.model.eval()
        total_loss = 0.0
        sample_count = 0
        self.metrics.reset()
        for batch in loader:
            ct = batch['ct'].to(self.device, non_blocking=True)
            pet = batch['pet'].to(self.device, non_blocking=True)
            mask = batch['mask'].to(self.device, non_blocking=True).float()
            batch_size = ct.shape[0]
            outputs = active(ct, pet=pet)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            loss, _ = self.criterion(logits, mask)
            self.metrics.update(logits, mask)
            total_loss += float(loss) * batch_size
            sample_count += batch_size
        out = self.metrics.compute()
        out['total_loss'] = total_loss / max(1, sample_count)
        if model is None:
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
            outputs_full = self.model(ct, pet=pet)
            logits_full = outputs_full['logits'] if isinstance(outputs_full, dict) else outputs_full
            loss_full, _ = self.criterion(logits_full.float(), mask.float())
            g_full_shared = torch.autograd.grad(loss_full, params_shared, retain_graph=True, allow_unused=True)
            g_full_ct = torch.autograd.grad(loss_full, params_ct, retain_graph=True, allow_unused=True)
            g_full_align = torch.autograd.grad(loss_full, params_align, retain_graph=True, allow_unused=True)
            g_full_dec = torch.autograd.grad(loss_full, params_dec, retain_graph=True, allow_unused=True)

            full_vec = _flatten_grads(g_full_shared)
            stats = {
                'full_shared_grad_norm': float(full_vec.norm()),
                'full_ct_grad_norm': float(_flatten_grads(g_full_ct).norm()),
                'full_align_grad_norm': float(_flatten_grads(g_full_align).norm()),
                'full_dec_grad_norm': float(_flatten_grads(g_full_dec).norm()),
            }
            return stats
        finally:
            for m, state in bn_states:
                m.track_running_stats = state
            self.model.train(was_training)

    def save_checkpoint(self, path, epoch, best_joint=None, best_full=None, best_joint_epoch=None, val_full=None, joint_dice=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            'epoch': epoch,
            'global_batch_step': self.global_batch_step,
            'train_batch_mode': 'full',
            'model': self.model.state_dict(),
            'model_ema': self.ema.state_dict() if self.ema is not None else None,
            'ema_updates': self.ema.updates if self.ema is not None else 0,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': None if self.scheduler is None else self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best_joint': best_joint,
            'best_full': best_full,
            'best_joint_epoch': best_joint_epoch,
            'val_full': val_full,
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
