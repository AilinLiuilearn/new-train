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


def resolve_stage2_affine_lr(config):
    explicit = getattr(config, 'stage2_affine_learning_rate', None)
    if explicit is None:
        return float(config.learning_rate) * 0.1
    return float(explicit)


def stage2_strategy_allows_affine(strategy):
    return str(strategy).strip() in ('paired_joint_affine', 'paired_anga_affine')


class MDTSegTeacher:
    def __init__(self, networks, config):
        self.networks = networks
        self.config = config
        self.model = networks['model']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.stage2_train_strategy = str(
            getattr(config, 'stage2_train_strategy', 'alternating_frozen')
        ).strip()
        self.is_stage2 = hasattr(self.model, 'role_fusion') and hasattr(self.model, 'stage1')
        self.allow_affine = bool(
            self.is_stage2 and stage2_strategy_allows_affine(self.stage2_train_strategy)
        )
        self.affine_lr = resolve_stage2_affine_lr(config) if self.allow_affine else None
        self.affine_warmup_epochs = int(getattr(config, 'stage2_affine_warmup_epochs', 1))
        self.anga_tau = float(getattr(config, 'stage2_anga_tau', 0.7))

        self.optimizer = self._build_optimizer()
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(config.mixed_precision))
        self.global_batch_step = 0
        self.criterion = BCEDiceLoss(smooth=config.loss_smooth, bce_weight=config.bce_weight, dice_weight=config.dice_weight)
        self.metrics = SegmentationMetricsCIPA()

    def _build_optimizer(self):
        wd = float(self.config.weight_decay)
        lr = float(self.config.learning_rate)

        if not self.is_stage2:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if not trainable_params:
                raise RuntimeError('No trainable parameters found for optimizer')
            return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)

        stage2_params = [
            p for p in list(self.model.role_fusion.parameters()) + list(self.model.decoder_adapters.parameters())
            if p.requires_grad
        ]
        if not stage2_params:
            raise RuntimeError('Stage2 optimizer requires trainable role_fusion/decoder_adapters')

        param_groups = [{
            'name': 'stage2',
            'params': stage2_params,
            'lr': lr,
            'weight_decay': wd,
        }]

        if self.allow_affine:
            affine_params = [
                p for p in self.model.stage1.pet_calibration.parameters() if p.requires_grad
            ]
            if not affine_params:
                raise RuntimeError('Affine strategy selected but no trainable affine params')
            param_groups.append({
                'name': 'stage1_affine',
                'params': affine_params,
                'lr': float(self.affine_lr),
                'weight_decay': wd,
            })

        # Validate mutual exclusion / coverage of all trainable params.
        ids_in_opt = []
        for g in param_groups:
            for p in g['params']:
                ids_in_opt.append(id(p))
        if len(ids_in_opt) != len(set(ids_in_opt)):
            raise RuntimeError('Optimizer param groups contain duplicated parameters')
        trainable_ids = {id(p) for p in self.model.parameters() if p.requires_grad}
        opt_ids = set(ids_in_opt)
        if trainable_ids != opt_ids:
            missing = trainable_ids - opt_ids
            extra = opt_ids - trainable_ids
            raise RuntimeError(
                f'Optimizer param coverage mismatch: missing={len(missing)} extra={len(extra)}'
            )

        print(
            f'[OPT] strategy={self.stage2_train_strategy} '
            f'stage2_params={sum(p.numel() for p in stage2_params)} lr={lr}',
            flush=True,
        )
        if self.allow_affine:
            print(
                f'[OPT] affine_params={sum(p.numel() for p in param_groups[1]["params"])} '
                f'lr={self.affine_lr} warmup_epochs={self.affine_warmup_epochs} tau={self.anga_tau}',
                flush=True,
            )
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def optimizer_group_lrs(self):
        out = {}
        for g in self.optimizer.param_groups:
            name = g.get('name', 'default')
            out[name] = float(g['lr'])
        return out

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
        """Stage-1 shared-param diagnostics, or Stage-2 affine-head diagnostics."""
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

            if self.is_stage2:
                if not self.allow_affine:
                    return {
                        'stage2_affine_diagnostics_skipped': 1.0,
                        'reason_frozen_strategy': 1.0,
                    }
                # Temporarily enable grads on affine heads for diagnosis.
                calib = self.model.stage1.pet_calibration
                params = [p for p in calib.parameters() if p.requires_grad]
                if not params:
                    return {'stage2_affine_diagnostics_skipped': 1.0, 'reason_no_affine_params': 1.0}

                self.model.train(True)
                self.model.zero_grad(set_to_none=True)
                outputs_full = self.model(ct, pet=pet, forward_mode='full', mask=None)
                loss_full, _ = self.criterion(outputs_full['logits'].float(), mask.float())
                g_full = torch.autograd.grad(loss_full, params, retain_graph=False, allow_unused=True)

                self.model.zero_grad(set_to_none=True)
                outputs_missing = self.model(ct, pet=pet, forward_mode='missing', mask=None)
                loss_missing, _ = self.criterion(outputs_missing['logits'].float(), mask.float())
                g_missing = torch.autograd.grad(loss_missing, params, retain_graph=False, allow_unused=True)

                def cos(a, b):
                    a = _flatten_grads(a)
                    b = _flatten_grads(b)
                    eps = 1e-8
                    return float(torch.dot(a, b) / (a.norm() * b.norm() + eps))

                full_vec = _flatten_grads(g_full)
                missing_vec = _flatten_grads(g_missing)
                return {
                    'affine_grad_cosine_total': cos(g_full, g_missing),
                    'full_affine_grad_norm': float(full_vec.norm()),
                    'missing_affine_grad_norm': float(missing_vec.norm()),
                    'full_missing_affine_grad_norm_ratio': float(
                        full_vec.norm() / (missing_vec.norm() + 1e-8)
                    ),
                }

            # Original Stage-1 shared diagnostics (trainable enc/align/decoder).
            params_shared = list(self.model.enc_ct.parameters()) + list(self.model.ct_align.parameters()) + list(self.model.decoder.parameters())
            params_shared = [p for p in params_shared if p.requires_grad]
            if not params_shared:
                return {'stage1_diagnostics_skipped': 1.0, 'reason_no_trainable_shared': 1.0}

            params_ct = [p for p in self.model.enc_ct.parameters() if p.requires_grad]
            params_align = [p for p in self.model.ct_align.parameters() if p.requires_grad]
            params_dec = [p for p in self.model.decoder.parameters() if p.requires_grad]
            outputs_full = self.model(ct, pet=pet, forward_mode='full', mask=None)
            outputs_missing = self.model(ct, pet=pet, forward_mode='missing', mask=None)
            logits_full = outputs_full['logits'] if isinstance(outputs_full, dict) else outputs_full
            logits_missing = outputs_missing['logits'] if isinstance(outputs_missing, dict) else outputs_missing
            loss_full, _ = self.criterion(logits_full.float(), mask.float())
            loss_missing, _ = self.criterion(logits_missing.float(), mask.float())
            g_full_shared = torch.autograd.grad(loss_full, params_shared, retain_graph=True, allow_unused=True)
            g_missing_shared = torch.autograd.grad(loss_missing, params_shared, retain_graph=True, allow_unused=True)
            g_full_ct = torch.autograd.grad(loss_full, params_ct, retain_graph=True, allow_unused=True) if params_ct else ()
            g_missing_ct = torch.autograd.grad(loss_missing, params_ct, retain_graph=True, allow_unused=True) if params_ct else ()
            g_full_align = torch.autograd.grad(loss_full, params_align, retain_graph=True, allow_unused=True) if params_align else ()
            g_missing_align = torch.autograd.grad(loss_missing, params_align, retain_graph=True, allow_unused=True) if params_align else ()
            g_full_dec = torch.autograd.grad(loss_full, params_dec, retain_graph=True, allow_unused=True) if params_dec else ()
            g_missing_dec = torch.autograd.grad(loss_missing, params_dec, retain_graph=True, allow_unused=True) if params_dec else ()

            def cos(a, b):
                a = _flatten_grads(a)
                b = _flatten_grads(b)
                eps = 1e-8
                return float(torch.dot(a, b) / (a.norm() * b.norm() + eps))

            full_vec = _flatten_grads(g_full_shared)
            missing_vec = _flatten_grads(g_missing_shared)
            stats = {
                'shared_grad_cosine_total': cos(g_full_shared, g_missing_shared),
                'ct_encoder_grad_cosine': cos(g_full_ct, g_missing_ct) if params_ct else 0.0,
                'ct_alignment_grad_cosine': cos(g_full_align, g_missing_align) if params_align else 0.0,
                'shared_decoder_grad_cosine': cos(g_full_dec, g_missing_dec) if params_dec else 0.0,
                'full_shared_grad_norm': float(full_vec.norm()),
                'missing_shared_grad_norm': float(missing_vec.norm()),
                'full_missing_grad_norm_ratio': float(full_vec.norm() / (missing_vec.norm() + 1e-8)),
                'negative_parameter_tensor_ratio': float(np.mean([
                    x < 0 for x in [
                        cos(g_full_ct, g_missing_ct) if params_ct else 0.0,
                        cos(g_full_align, g_missing_align) if params_align else 0.0,
                        cos(g_full_dec, g_missing_dec) if params_dec else 0.0,
                    ]
                ])),
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
            'stage2_train_strategy': self.stage2_train_strategy,
            'stage2_affine_learning_rate': self.affine_lr,
            'stage2_affine_warmup_epochs': self.affine_warmup_epochs,
            'stage2_anga_tau': self.anga_tau,
            'random_state_python': random.getstate(),
            'random_state_numpy': np.random.get_state(),
            'random_state_torch': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload['random_state_cuda'] = torch.cuda.get_rng_state_all()
        torch.save(payload, path)
