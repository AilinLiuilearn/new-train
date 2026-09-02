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


def _is_stage2_model(model) -> bool:
    return bool(getattr(model, 'is_fgms_stage2', False))


STAGE2_BASE_LRS = {
    'stage2_moe': 8e-5,
    'stage2_decoder': 2e-5,
    'stage1_boundary': 5e-6,
}


class MDTSegTeacher:
    def __init__(self, networks, config):
        self.networks = networks
        self.config = config
        self.model = networks['model']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.is_stage2 = _is_stage2_model(self.model)
        if self.is_stage2:
            self.optimizer = self._build_stage2_optimizer()
            self.print_stage2_optimizer_groups()
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(config.mixed_precision))
        self.global_batch_step = 0
        self.criterion = BCEDiceLoss(smooth=config.loss_smooth, bce_weight=config.bce_weight, dice_weight=config.dice_weight)
        self.metrics = SegmentationMetricsCIPA()

    def _build_stage2_optimizer(self):
        moe_params = list(self.model.stage2_moe.parameters())
        dec_params = list(self.model.stage2_decoder.parameters())
        boundary_params = (
            list(self.model.stage1.pet_calibration.parameters())
            + list(self.model.stage1.fusion.parameters())
        )
        if not moe_params:
            raise RuntimeError('stage2_moe has no parameters for optimizer.')
        if not dec_params:
            raise RuntimeError('stage2_decoder has no parameters for optimizer.')
        if not boundary_params:
            raise RuntimeError('stage1 boundary has no parameters for optimizer.')
        return torch.optim.AdamW(
            [
                {'name': 'stage2_moe', 'params': moe_params, 'lr': self.config.learning_rate},
                {'name': 'stage2_decoder', 'params': dec_params, 'lr': self.config.decoder_lr},
                {
                    'name': 'stage1_boundary',
                    'params': boundary_params,
                    'lr': float(getattr(self.config, 'stage1_boundary_lr', 5e-6)),
                },
            ],
            weight_decay=self.config.weight_decay,
        )

    def print_stage2_optimizer_groups(self):
        print('[OPTIMIZER][STAGE2] param groups:', flush=True)
        for group in self.optimizer.param_groups:
            name = group.get('name', 'unknown')
            lr = float(group['lr'])
            expected = STAGE2_BASE_LRS.get(name)
            print(f'  {name:16s} base_lr={lr:.8g}', flush=True)
            if expected is not None and abs(lr - expected) > 1e-12:
                raise RuntimeError(
                    f'Stage2 optimizer group {name} base_lr={lr}, expected {expected}'
                )

    def verify_stage2_lr_ratio(self, tag=''):
        snap = {
            group.get('name'): float(group['lr'])
            for group in self.optimizer.param_groups
            if group.get('name') in STAGE2_BASE_LRS
        }
        if len(snap) != 3:
            raise RuntimeError(f'Missing stage2 optimizer groups: {snap}')
        mult = snap['stage2_moe'] / STAGE2_BASE_LRS['stage2_moe']
        dec_ratio = snap['stage2_decoder'] / STAGE2_BASE_LRS['stage2_decoder']
        bnd_ratio = snap['stage1_boundary'] / STAGE2_BASE_LRS['stage1_boundary']
        if not (abs(dec_ratio - mult) < 1e-4 and abs(bnd_ratio - mult) < 1e-4):
            raise RuntimeError(
                f'Stage2 LR multiplier mismatch {tag}: moe_mult={mult:.6f} '
                f'dec_mult={dec_ratio:.6f} boundary_mult={bnd_ratio:.6f}'
            )
        print(
            f'[OPTIMIZER][STAGE2] LR ratio ok {tag}: '
            f'mult={mult:.6f} moe={snap["stage2_moe"]:.8g} '
            f'dec={snap["stage2_decoder"]:.8g} boundary={snap["stage1_boundary"]:.8g}',
            flush=True,
        )
        return mult

    def get_lr_by_name(self, name: str):
        for group in self.optimizer.param_groups:
            if group.get('name') == name:
                return float(group['lr'])
        return None

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def train_step(self, batch, forward_mode='full'):
        ct = batch['ct'].to(self.device, non_blocking=True)
        pet = batch['pet'].to(self.device, non_blocking=True)
        mask = batch['mask'].to(self.device, non_blocking=True).float()
        outputs = self.model(ct, pet=pet, mask=mask, forward_mode=forward_mode)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        seg_loss, loss_stats = self.criterion(logits, mask)
        total_loss = seg_loss
        balance_loss = torch.tensor(0.0, device=seg_loss.device)
        if isinstance(outputs, dict) and outputs.get('balance_loss') is not None:
            balance_loss = outputs['balance_loss']
            total_loss = seg_loss + balance_loss
        stats = {
            'loss_total': total_loss.detach(),
            'loss_seg': loss_stats.get('loss_dice', seg_loss.detach()),
            'loss_balance': balance_loss.detach() if torch.is_tensor(balance_loss) else torch.tensor(0.0, device=seg_loss.device),
            'loss_boundary': torch.tensor(0.0, device=seg_loss.device),
        }
        return total_loss, logits, outputs, stats

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
        if self.is_stage2:
            return self._gradient_diagnostics_stage2(batch, max_samples=max_samples)
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

    def _gradient_diagnostics_stage2(self, batch, max_samples=1):
        was_training = self.model.training
        self.model.train()
        try:
            ct = batch['ct'][:max_samples].to(self.device, non_blocking=True)
            mask = batch['mask'][:max_samples].to(self.device, non_blocking=True).float()
            pet = batch['pet'][:max_samples].to(self.device, non_blocking=True)
            params_moe = [p for p in self.model.stage2_moe.parameters() if p.requires_grad]
            params_dec = [p for p in self.model.stage2_decoder.parameters() if p.requires_grad]
            outputs_full = self.model(ct, pet=pet, forward_mode='full', mask=None)
            outputs_missing = self.model(ct, pet=pet, forward_mode='missing', mask=None)
            logits_full = outputs_full['logits']
            logits_missing = outputs_missing['logits']
            loss_full, _ = self.criterion(logits_full.float(), mask.float())
            loss_missing, _ = self.criterion(logits_missing.float(), mask.float())
            if outputs_full.get('balance_loss') is not None:
                loss_full = loss_full + outputs_full['balance_loss']
            if outputs_missing.get('balance_loss') is not None:
                loss_missing = loss_missing + outputs_missing['balance_loss']
            g_full_moe = torch.autograd.grad(loss_full, params_moe, retain_graph=True, allow_unused=True)
            g_missing_moe = torch.autograd.grad(loss_missing, params_moe, retain_graph=True, allow_unused=True)
            g_full_dec = torch.autograd.grad(loss_full, params_dec, retain_graph=True, allow_unused=True)
            g_missing_dec = torch.autograd.grad(loss_missing, params_dec, retain_graph=True, allow_unused=True)

            def cos(a, b):
                a = _flatten_grads(a)
                b = _flatten_grads(b)
                eps = 1e-8
                return float(torch.dot(a, b) / (a.norm() * b.norm() + eps))

            return {
                'stage2_moe_grad_cosine': cos(g_full_moe, g_missing_moe),
                'stage2_decoder_grad_cosine': cos(g_full_dec, g_missing_dec),
                'full_stage2_moe_grad_norm': float(_flatten_grads(g_full_moe).norm()),
                'missing_stage2_moe_grad_norm': float(_flatten_grads(g_missing_moe).norm()),
                'full_stage2_decoder_grad_norm': float(_flatten_grads(g_full_dec).norm()),
                'missing_stage2_decoder_grad_norm': float(_flatten_grads(g_missing_dec).norm()),
                'forbidden_stage1_grad_nonzero_count': float(self.model.count_forbidden_stage1_nonzero_grads()),
                'boundary_grad_nonzero_count': float(self.model.count_boundary_nonzero_grads()),
            }
        finally:
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
