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
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(config.mixed_precision))
        self.global_batch_step = 0
        self.criterion = BCEDiceLoss(smooth=config.loss_smooth, bce_weight=config.bce_weight, dice_weight=config.dice_weight)
        self.metrics = SegmentationMetricsCIPA()

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def train_step(self, batch, forward_mode='full', collect_module1_candidates=True):
        ct = batch['ct'].to(self.device, non_blocking=True)
        pet = batch['pet'].to(self.device, non_blocking=True)
        mask = batch['mask'].to(self.device, non_blocking=True).float()
        outputs = self.model(
            ct,
            pet=pet,
            forward_mode=forward_mode,
            mask=mask,
            collect_module1_candidates=collect_module1_candidates,
        )
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        seg_loss, loss_stats = self.criterion(logits, mask)
        # Clean Module-1: loss_total = seg + lambda_p * PET-proto (raw).
        # First round (bank not ready) yields exactly 0.
        proto_raw = outputs.get('prototype_contrastive_loss', seg_loss.new_zeros(()))
        if proto_raw.dim() > 0:
            proto_raw = proto_raw.reshape(())
        proto_weight = float(getattr(self.config, 'pspi_proto_contrastive_weight', 0.01))
        proto_weighted = proto_weight * proto_raw
        total_loss = seg_loss + proto_weighted
        stats = {
            'loss_total': total_loss.detach(),
            'loss_seg': loss_stats.get('loss_dice', seg_loss.detach()),
            'loss_seg_total': seg_loss.detach(),
            'loss_proto': proto_raw.detach(),
            'loss_proto_weighted': proto_weighted.detach(),
            'proto_num_terms': outputs.get('prototype_contrastive_num_terms', 0),
            'loss_boundary': torch.tensor(0.0, device=total_loss.device),
        }
        return total_loss, logits, outputs, stats

    def train_batch_full_missing(self, batch, full_weight=0.5, missing_weight=0.5):
        """Within-batch alternating step: Full + Missing forwards, one update.

        Both branches share the same batch and are evaluated at the same
        parameter state theta_t. Gradients of `full_weight * L_full` and
        `missing_weight * L_missing` are accumulated into .grad, then a single
        optimizer update is applied. The scheduler is NOT stepped here.
        """
        full_weight = float(full_weight)
        missing_weight = float(missing_weight)
        if full_weight <= 0.0 or missing_weight <= 0.0:
            raise ValueError('within-batch weights must be > 0')
        if abs(full_weight + missing_weight - 1.0) > 1e-6:
            raise ValueError(
                f'within-batch weights must sum to 1.0, got {full_weight} + {missing_weight}'
            )

        if task_module1_ref := getattr(self.model, 'module1', None):
            task_module1_ref.training_collection_calls = 0

        self.optimizer.zero_grad(set_to_none=True)
        amp_enabled = bool(getattr(self.config, 'mixed_precision', False)) and self.device.type == 'cuda'

        def _branch_backward(loss):
            # Run forward -> backward immediately for each branch so that the
            # two full-resolution autograd graphs never coexist (memory).
            # Both weighted grads accumulate into .grad before one unscale/step.
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            full_loss, full_logits, full_outputs, full_stats = self.train_step(
                batch,
                forward_mode='full',
                collect_module1_candidates=True,
            )
        if not torch.isfinite(full_loss):
            raise RuntimeError('full loss became non-finite')
        _branch_backward(full_weight * full_loss)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            missing_loss, missing_logits, missing_outputs, missing_stats = self.train_step(
                batch,
                forward_mode='missing',
                collect_module1_candidates=False,
            )
        if not torch.isfinite(missing_loss):
            raise RuntimeError('missing loss became non-finite')
        _branch_backward(missing_weight * missing_loss)

        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)

        combined_stats = self._within_batch_grad_norms()
        grad_clip = float(getattr(self.config, 'grad_clip', 0.0))
        total_grad_norm = (
            torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), grad_clip)
            if grad_clip > 0 else torch.tensor(0.0)
        )

        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        combined_loss = full_weight * full_loss.detach() + missing_weight * missing_loss.detach()
        stats = {
            'full_loss': full_loss.detach(),
            'missing_loss': missing_loss.detach(),
            'combined_loss': combined_loss,
            'full_seg_loss': full_stats['loss_seg_total'],
            'missing_seg_loss': missing_stats['loss_seg_total'],
            'full_proto_loss': full_stats['loss_proto'],
            'missing_proto_loss': missing_stats['loss_proto'],
            'full_proto_loss_weighted': full_stats['loss_proto_weighted'],
            'missing_proto_loss_weighted': missing_stats['loss_proto_weighted'],
            'total_grad_norm': float(total_grad_norm) if torch.is_tensor(total_grad_norm) else float(total_grad_norm),
            'full_outputs': full_outputs,
            'missing_outputs': missing_outputs,
            **combined_stats,
        }
        return combined_loss, full_logits, missing_logits, stats

    def _within_batch_grad_norms(self):
        def module_norm(module):
            if module is None:
                return 0.0
            total = None
            for p in module.parameters():
                if p.requires_grad and p.grad is not None:
                    val = p.grad.detach().float().pow(2).sum()
                    total = val if total is None else total + val
            return float(total.sqrt().item()) if total is not None else 0.0

        def scalar_norm(param):
            if param is None or param.grad is None:
                return 0.0
            return float(param.grad.detach().float().pow(2).sum().sqrt().item())

        module1 = getattr(self.model, 'module1', None)
        attention = getattr(module1, 'attention', None) if module1 is not None else None
        return {
            'grad_combined_enc_ct': module_norm(self.model.enc_ct),
            'grad_combined_enc_pet': module_norm(self.model.enc_pet),
            'grad_combined_ct_align': module_norm(self.model.ct_align),
            'grad_combined_decoder': module_norm(self.model.decoder),
            'grad_combined_module1_retrieval': module_norm(attention),
            'grad_combined_prior_scale': scalar_norm(getattr(self.model, 'missing_prior_logits', None)),
        }

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
                pet = None
                forward_mode = 'missing'
                pet_available = None
            else:
                pet = batch['pet'].to(self.device, non_blocking=True)
                forward_mode = 'auto'
                pet_available = batch.get('pet_available')
                if pet_available is not None:
                    pet_available = pet_available.to(self.device, non_blocking=True)
            outputs = self.model(
                ct,
                pet=pet,
                pet_available=pet_available,
                forward_mode=forward_mode,
                mask=mask,
            )
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
            outputs_full = self.model(ct, pet=pet, forward_mode='full')
            outputs_missing = self.model(ct, pet=pet, forward_mode='missing')
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
            'train_mode': str(getattr(self.config, 'train_mode', 'within_batch_alternating')),
            'within_batch_full_weight': float(getattr(self.config, 'within_batch_full_weight', 0.5)),
            'within_batch_missing_weight': float(getattr(self.config, 'within_batch_missing_weight', 0.5)),
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
