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
        self.is_joint = hasattr(self.model, 'role_fusion') and hasattr(self.model, 'stage1')

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError('No trainable parameters found for optimizer')
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(config.mixed_precision))
        self.global_batch_step = 0
        self.criterion = BCEDiceLoss(smooth=config.loss_smooth, bce_weight=config.bce_weight, dice_weight=config.dice_weight)
        self.metrics = SegmentationMetricsCIPA()
        print(
            f'[OPT] joint={self.is_joint} trainable_tensors={len(trainable_params)} '
            f'trainable_params={sum(p.numel() for p in trainable_params)} lr={config.learning_rate}',
            flush=True,
        )

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
            # mask=None so evaluation never updates the prototype bank.
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

    def gradient_diagnostics(self, batch, max_samples=1):
        """Report Full/Missing grads on active joint modules (or Stage-1 shared)."""
        was_training = self.model.training
        self.model.train(True)
        try:
            ct = batch['ct'][:max_samples].to(self.device, non_blocking=True)
            mask = batch['mask'][:max_samples].to(self.device, non_blocking=True).float()
            pet = batch['pet'][:max_samples].to(self.device, non_blocking=True)

            def _module_grad_norm(module):
                total = None
                for p in module.parameters():
                    if p.grad is None:
                        continue
                    val = p.grad.detach().float().pow(2).sum()
                    total = val if total is None else total + val
                return float(total.sqrt().item()) if total is not None else 0.0

            def _run(route):
                self.model.zero_grad(set_to_none=True)
                out = self.model(ct, pet=pet, mask=mask, forward_mode=route)
                loss, _ = self.criterion(out['logits'].float(), mask.float())
                loss.backward()
                if self.is_joint:
                    attn = self.model.stage1.prototype_memory.attention
                    return {
                        f'{route}_enc_ct': _module_grad_norm(self.model.stage1.enc_ct),
                        f'{route}_enc_pet': _module_grad_norm(self.model.stage1.enc_pet),
                        f'{route}_ct_align': _module_grad_norm(self.model.stage1.ct_align),
                        f'{route}_prototype_attention': _module_grad_norm(attn),
                        f'{route}_calibration': _module_grad_norm(self.model.stage1.pet_calibration),
                        f'{route}_role_fusion': _module_grad_norm(self.model.role_fusion),
                        f'{route}_decoder': _module_grad_norm(self.model.stage1.decoder),
                        f'{route}_legacy_fusion': _module_grad_norm(self.model.stage1.fusion),
                    }
                return {
                    f'{route}_enc_ct': _module_grad_norm(self.model.enc_ct),
                    f'{route}_ct_align': _module_grad_norm(self.model.ct_align),
                    f'{route}_decoder': _module_grad_norm(self.model.decoder),
                }

            stats = {}
            stats.update(_run('full'))
            stats.update(_run('missing'))
            return stats
        finally:
            self.model.zero_grad(set_to_none=True)
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
