import json
import os
import random

import numpy as np
import torch

from utils.seg_losses import BCEDiceLoss
from utils.metrics_seg import SegmentationMetricsCIPA


def _check_loss(name, value, forward_mode, global_batch_step):
    if not torch.isfinite(value).all():
        raise RuntimeError(
            f'[NaN/Inf] {name} is non-finite: '
            f'route={forward_mode}, '
            f'global_batch_step={global_batch_step}, '
            f'dtype={value.dtype}, '
            f'value={value.detach().float().item()}'
        )


class MDTSegTeacher:
    def __init__(self, networks, config):
        self.networks = networks
        self.config = config
        self.model = networks['model']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = None
        amp_init_scale = float(getattr(config, 'amp_init_scale', 4096.0))
        if amp_init_scale <= 0:
            raise ValueError('amp_init_scale must be positive')
        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(config.mixed_precision), init_scale=amp_init_scale)
        self.global_batch_step = 0
        self.criterion = BCEDiceLoss(smooth=config.loss_smooth, bce_weight=config.bce_weight, dice_weight=config.dice_weight)
        self.metrics = SegmentationMetricsCIPA()

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def train_step(self, batch, forward_mode='full'):
        ct = batch['ct'].to(self.device, non_blocking=True)
        pet = batch['pet'].to(self.device, non_blocking=True) if forward_mode == 'full' else None
        mask = batch['mask'].to(self.device, non_blocking=True).float()
        outputs = self.model(ct, pet=pet, forward_mode=forward_mode, mask=mask if forward_mode == 'full' else None)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        loss_seg, loss_stats = self.criterion(logits, mask)
        _check_loss('loss_seg', loss_seg, forward_mode, self.global_batch_step)
        loss = loss_seg
        _check_loss('loss_total', loss, forward_mode, self.global_batch_step)
        stats = {
            'loss_total': loss.detach(),
            'loss_seg': loss_seg.detach(),
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
            if torch.isnan(ct).any() or torch.isinf(ct).any():
                raise RuntimeError(f"CT input contains NaN/Inf during evaluation (tag={tag}, eval_mode={eval_mode})")
            if torch.isnan(mask).any() or torch.isinf(mask).any():
                raise RuntimeError(f"Mask contains NaN/Inf during evaluation (tag={tag}, eval_mode={eval_mode})")
            if eval_mode == 'full':
                pet = batch['pet'].to(self.device, non_blocking=True)
                if torch.isnan(pet).any() or torch.isinf(pet).any():
                    raise RuntimeError(f"PET input contains NaN/Inf during evaluation (tag={tag}, eval_mode={eval_mode})")
                forward_mode = 'full'
                pet_available = None
            elif eval_mode == 'fixed_missing':
                pet = None
                forward_mode = 'missing'
                pet_available = None
            else:
                pet = batch['pet'].to(self.device, non_blocking=True)
                if torch.isnan(pet).any() or torch.isinf(pet).any():
                    raise RuntimeError(f"PET input contains NaN/Inf during evaluation (tag={tag}, eval_mode={eval_mode})")
                forward_mode = 'auto'
                pet_available = batch.get('pet_available')
            outputs = self.model(ct, pet=pet, pet_available=pet_available, forward_mode=forward_mode)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            loss, _ = self.criterion(logits, mask)
            self.metrics.update(logits, mask)
            total_loss += float(loss) * batch_size
            sample_count += batch_size
        out = self.metrics.compute()
        out['total_loss'] = total_loss / max(1, sample_count)
        self.model.train(was_training)
        return out


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
