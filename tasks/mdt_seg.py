# -*- coding: utf-8 -*-
"""Segmentation training task shared by the two clean baselines.

- ``train_step_ct``: CT-only forward, one BCE+Dice loss.
- ``train_step_mixed``: one auto forward on an exact-half batch, then
  0.5 * full_subset_loss + 0.5 * missing_subset_loss, one backward upstream.
- ``evaluate``: ``full`` / ``missing`` / ``ct`` modes.
- Single AdamW optimizer with one unified learning rate (no decoder_lr).
- Non-finite logits/loss raise with their location instead of being masked.
"""
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
    """Task object shared by both runners ("Teacher" is a legacy name).

    The name does not imply distillation: this is plain supervised training.
    """

    def __init__(self, networks, config):
        self.networks = networks
        self.config = config
        self.model = networks['model']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(config.mixed_precision))
        self.global_batch_step = 0
        self.criterion = BCEDiceLoss()
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
        """Activate the EMA after the warmup epochs, hard-syncing on switch."""
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

    def eval_weights_tag(self):
        return 'ema' if (self.ema is not None and self.ema_active) else 'raw'

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _finish_step(self, logits, mask, where):
        if not torch.isfinite(logits).all():
            raise RuntimeError(f'[NaN/Inf] logits at {where}')
        loss, loss_stats = self.criterion(logits, mask)
        if not torch.isfinite(loss).all():
            raise RuntimeError(f'[NaN/Inf] loss at {where}')
        return loss, loss_stats

    def train_step_ct(self, batch):
        """CT-only step: every sample trains the CT path."""
        ct = batch['ct'].to(self.device, non_blocking=True)
        mask = batch['mask'].to(self.device, non_blocking=True).float()
        outputs = self.model(ct)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        loss, loss_stats = self._finish_step(logits, mask, 'train_step_ct')
        stats = {
            'loss_total': loss.detach(),
            'loss_seg': loss_stats.get('loss_dice', loss.detach()),
        }
        return loss, logits, outputs, stats

    def train_step_mixed(self, batch, pet_available):
        """Mixed step: exact-half batch, 0.5*full + 0.5*missing, one backward."""
        ct = batch['ct'].to(self.device, non_blocking=True)
        pet = batch['pet'].to(self.device, non_blocking=True)
        mask = batch['mask'].to(self.device, non_blocking=True).float()
        raw_state = torch.as_tensor(pet_available, device=ct.device)
        if raw_state.numel() != ct.shape[0]:
            raise ValueError(
                f'pet_available must contain one state per sample: '
                f'got {raw_state.numel()} for batch {ct.shape[0]}')
        if raw_state.dtype == torch.bool:
            state = raw_state.long().view(-1)
        elif raw_state.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            if not torch.all((raw_state == 0) | (raw_state == 1)):
                raise ValueError('pet_available values must be 0 or 1')
            state = raw_state.long().view(-1)
        else:
            raise ValueError('pet_available must be 0/1 integers or bools, no silent float truncation')
        full_index = state.eq(1)
        missing_index = state.eq(0)
        num_full = int(full_index.sum())
        num_missing = int(missing_index.sum())
        if num_full == 0 or num_missing == 0:
            raise ValueError(
                f'mixed batch requires non-empty full and missing subsets: '
                f'got full={num_full} missing={num_missing}')
        outputs = self.model(ct, pet=pet, pet_available=state, forward_mode='auto')
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        if not torch.isfinite(logits).all():
            raise RuntimeError('[NaN/Inf] logits at train_step_mixed')
        full_loss, full_stats = self.criterion(logits[full_index], mask[full_index])
        missing_loss, missing_stats = self.criterion(logits[missing_index], mask[missing_index])
        total_loss = 0.5 * full_loss + 0.5 * missing_loss
        if not torch.isfinite(total_loss).all():
            raise RuntimeError('[NaN/Inf] loss at train_step_mixed')
        stats = {
            'loss_total': total_loss.detach(),
            'loss_full': full_loss.detach(),
            'loss_missing': missing_loss.detach(),
            'num_full': num_full,
            'num_missing': num_missing,
            'full_bce': full_stats.get('loss_bce', full_loss.detach()).detach(),
            'full_dice': full_stats.get('loss_dice', full_loss.detach()).detach(),
            'missing_bce': missing_stats.get('loss_bce', missing_loss.detach()).detach(),
            'missing_dice': missing_stats.get('loss_dice', missing_loss.detach()).detach(),
        }
        return total_loss, logits, outputs, stats

    @torch.no_grad()
    def evaluate(self, loader, eval_mode='full', tag='val', model=None):
        """eval_mode: 'full' | 'missing' | 'ct'. Missing passes pet=None."""
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
            mask = batch['mask'].to(self.device, non_blocking=True).float()
            batch_size = ct.shape[0]
            if eval_mode == 'full':
                pet = batch['pet'].to(self.device, non_blocking=True)
                outputs = active(ct, pet=pet, forward_mode='full')
            elif eval_mode == 'missing':
                outputs = active(ct, pet=None, forward_mode='missing')
            elif eval_mode == 'ct':
                outputs = active(ct)
            else:
                raise ValueError(f'Unsupported eval_mode={eval_mode!r}')
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            loss, _ = self.criterion(logits, mask)
            self.metrics.update(logits, mask)
            total_loss += float(loss) * batch_size
            sample_count += batch_size
        out = self.metrics.compute()
        out['total_loss'] = total_loss / max(1, sample_count)
        out['eval_mode'] = eval_mode
        out['weights'] = self.eval_weights_tag() if model is None or model is self.eval_model() else 'explicit'
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
            pet = batch.get('pet')
            if pet is not None:
                pet = pet[:max_samples].to(self.device, non_blocking=True)
            params_shared = (list(self.model.enc_ct.parameters())
                             + list(self.model.ct_align.parameters())
                             + list(self.model.decoder.parameters()))
            params_ct = list(self.model.enc_ct.parameters())
            params_align = list(self.model.ct_align.parameters())
            params_dec = list(self.model.decoder.parameters())
            params_pet = list(getattr(self.model, 'enc_pet', []))
            if not params_pet:
                params_pet = []
            else:
                params_pet = list(self.model.enc_pet.parameters())
            outputs_full = self.model(ct, pet=pet, forward_mode='full') if pet is not None \
                else self.model(ct)
            logits_full = outputs_full['logits'] if isinstance(outputs_full, dict) else outputs_full
            loss_full, _ = self.criterion(logits_full.float(), mask.float())
            g_full_shared = torch.autograd.grad(loss_full, params_shared, retain_graph=True, allow_unused=True)
            if pet is None:
                full_vec = _flatten_grads(g_full_shared)
                return {
                    'full_shared_grad_norm': float(full_vec.norm()),
                    'full_ct_grad_norm': float(_flatten_grads(
                        torch.autograd.grad(loss_full, params_ct, retain_graph=True, allow_unused=True)).norm()),
                    'full_align_grad_norm': float(_flatten_grads(
                        torch.autograd.grad(loss_full, params_align, retain_graph=True, allow_unused=True)).norm()),
                    'full_dec_grad_norm': float(_flatten_grads(
                        torch.autograd.grad(loss_full, params_dec, retain_graph=True, allow_unused=True)).norm()),
                }
            outputs_missing = self.model(ct, pet=None, forward_mode='missing')
            logits_missing = outputs_missing['logits'] if isinstance(outputs_missing, dict) else outputs_missing
            loss_missing, _ = self.criterion(logits_missing.float(), mask.float())
            g_missing_shared = torch.autograd.grad(loss_missing, params_shared, retain_graph=True, allow_unused=True)
            g_full_ct = torch.autograd.grad(loss_full, params_ct, retain_graph=True, allow_unused=True)
            g_missing_ct = torch.autograd.grad(loss_missing, params_ct, retain_graph=True, allow_unused=True)
            g_full_align = torch.autograd.grad(loss_full, params_align, retain_graph=True, allow_unused=True)
            g_missing_align = torch.autograd.grad(loss_missing, params_align, retain_graph=True, allow_unused=True)
            g_full_dec = torch.autograd.grad(loss_full, params_dec, retain_graph=True, allow_unused=True)
            g_missing_dec = torch.autograd.grad(loss_missing, params_dec, retain_graph=True, allow_unused=True)
            g_missing_pet = torch.autograd.grad(loss_missing, params_pet, allow_unused=True) if params_pet else []

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
                'missing_pet_grad_norm': float(_flatten_grads(g_missing_pet).norm()),
                'negative_parameter_tensor_ratio': float(np.mean([x < 0 for x in [
                    cos(g_full_ct, g_missing_ct),
                    cos(g_full_align, g_missing_align),
                    cos(g_full_dec, g_missing_dec)]])),
            }
            return stats
        finally:
            for m, state in bn_states:
                m.track_running_stats = state
            self.model.train(was_training)

    def save_checkpoint(self, path, epoch, best=None, best_epoch=None, val=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            'epoch': epoch,
            'global_batch_step': self.global_batch_step,
            'model': self.model.state_dict(),
            'model_ema': self.ema.state_dict() if self.ema is not None else None,
            'ema_updates': self.ema.updates if self.ema is not None else 0,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': None if self.scheduler is None else self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best': best,
            'best_epoch': best_epoch,
            'val': val,
            'eval_weights': self.eval_weights_tag(),
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

    @staticmethod
    def load_state_dicts(path, map_location='cpu'):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)