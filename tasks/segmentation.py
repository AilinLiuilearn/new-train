"""Full-only supervised training, evaluation, EMA and checkpoint utilities."""
import os
import random
import numpy as np
import torch
from utils.seg_losses import BCEDiceLoss
from utils.metrics_seg import SegmentationMetricsCIPA
from utils.ema import ModelEMA


class SegmentationTask:
    """Supervised segmentation; EMA is used only for evaluation."""

    def __init__(self, networks, config):
        self.networks = networks
        self.config = config
        self.model = networks['model']
        requested = getattr(config, 'device', 'auto')
        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(config.mixed_precision) and self.device.type == 'cuda')
        self.global_batch_step = 0
        self.criterion = BCEDiceLoss()
        self.metrics = SegmentationMetricsCIPA(
            hd95_mode=getattr(config, 'hd95_mode', 'legacy'))
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