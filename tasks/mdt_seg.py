# -*- coding: utf-8 -*-
import os
import torch
import torch.nn.functional as F
from utils.metrics_seg import SegmentationMetricsCIPA
from utils.optimization import get_optimizer
from utils.seg_losses import BCEDiceLoss


def _forward(nets, ct, pet, target_size):
    return nets['model'](ct, pet, target_size=target_size)


class MDTSegTeacher:
    def __init__(self, networks, config):
        self.networks = {k: v for k, v in networks.items() if v is not None}
        self.config = config
        self.device = torch.device('cuda', int(config.gpus[0]))
        for v in self.networks.values():
            v.to(self.device)

        self.ema_decay = float(getattr(config, 'ema_decay', 0.999))
        self.ema_warmup_epochs = int(getattr(config, 'ema_warmup_epochs', 3))
        self.use_ema = self.ema_decay > 0
        self._ema_step_count = 0
        self._current_epoch = 0
        if self.use_ema:
            import copy
            self.ema_model = copy.deepcopy(self.networks['model'])
            self.ema_model.to(self.device)
            self.ema_model.eval()
            for p in self.ema_model.parameters():
                p.requires_grad = False
            self._ema_initialized = False
            print(f'[+] EMA enabled, decay={self.ema_decay}, warmup_epochs={self.ema_warmup_epochs}')
        else:
            self.ema_model = None
            self._ema_initialized = True

        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None
        self.loss_seg = BCEDiceLoss(
            bce_weight=getattr(config, 'bce_weight', 1.0),
            dice_weight=getattr(config, 'dice_weight', 1.0),
            smooth=getattr(config, 'loss_smooth', 1.0),
            pos_weight=getattr(config, 'pos_weight', None),
        ).to(self.device)

        lr = getattr(config, 'decoder_lr', None) or config.learning_rate
        params = list(self.networks['model'].parameters())
        self.optimizer = get_optimizer(
            [{'params': params, 'lr': lr}],
            config.optimizer,
            config.learning_rate,
            config.weight_decay,
        )
        self.scheduler = None

    @torch.no_grad()
    def update_ema(self):
        if not self.use_ema:
            return
        if not self._ema_initialized:
            for ema_p, model_p in zip(self.ema_model.parameters(), self.networks['model'].parameters()):
                ema_p.data.copy_(model_p.data)
            for ema_b, model_b in zip(self.ema_model.buffers(), self.networks['model'].buffers()):
                ema_b.data.copy_(model_b.data)
            self._ema_initialized = True
            return
        self._ema_step_count += 1
        alpha = min(self.ema_decay, 1.0 - 1.0 / (self._ema_step_count + 1))
        for ema_p, model_p in zip(self.ema_model.parameters(), self.networks['model'].parameters()):
            ema_p.data.mul_(alpha).add_(model_p.data, alpha=1.0 - alpha)
        for ema_b, model_b in zip(self.ema_model.buffers(), self.networks['model'].buffers()):
            ema_b.data.copy_(model_b.data)

    def set_epoch(self, epoch):
        self._current_epoch = epoch

    def _select_main_pred(self, outputs):
        if isinstance(outputs, dict):
            if 'pred' in outputs:
                return outputs['pred']
            preds = outputs.get('preds')
            if isinstance(preds, (list, tuple)):
                return preds[0]
        if isinstance(outputs, (list, tuple)):
            return outputs[0]
        return outputs

    def _compute_cudm_disentangle_loss(self, outputs, mask):
        bg_weight = float(getattr(self.config, 'cudm_bg_weight', 0.0))
        legacy_tumor_weight = float(getattr(self.config, 'cudm_tumor_weight', 0.0))
        if bg_weight <= 0 and legacy_tumor_weight > 0:
            bg_weight = legacy_tumor_weight
        orth_weight = float(getattr(self.config, 'cudm_orth_weight', 0.0))
        start_stage = max(1, int(getattr(self.config, 'cudm_loss_start_stage', 3)))
        if bg_weight <= 0 and orth_weight <= 0:
            zero = mask.new_tensor(0.0)
            return zero, {
                'loss_cudm_bg': zero.detach(),
                'loss_cudm_orth': zero.detach(),
            }
        if not isinstance(outputs, dict) or not outputs.get('fusion_aux'):
            zero = mask.new_tensor(0.0)
            return zero, {
                'loss_cudm_bg': zero.detach(),
                'loss_cudm_orth': zero.detach(),
            }

        bg_losses, orth_losses = [], []
        eps = 1e-6
        for stage_idx, aux in enumerate(outputs['fusion_aux'], start=1):
            if stage_idx < start_stage:
                continue
            common = aux['common'].float()
            tumor = aux['tumor'].float()
            mask_s = F.interpolate(mask.float(), size=tumor.shape[-2:], mode='nearest')

            energy = tumor.abs().mean(dim=1, keepdim=True)
            energy = energy / (energy.mean(dim=(2, 3), keepdim=True) + eps)
            background = 1.0 - mask_s
            bg_loss = (energy * background).sum(dim=(1, 2, 3)) / (background.sum(dim=(1, 2, 3)) + eps)
            bg_losses.append(bg_loss.mean())

            common_desc = F.adaptive_avg_pool2d(common, 1).flatten(1)
            tumor_desc = F.adaptive_avg_pool2d(tumor, 1).flatten(1)
            common_desc = common_desc - common_desc.mean(dim=1, keepdim=True)
            tumor_desc = tumor_desc - tumor_desc.mean(dim=1, keepdim=True)
            cosine = F.cosine_similarity(common_desc, tumor_desc, dim=1, eps=eps)
            orth_losses.append((cosine ** 2).mean())

        loss_bg = torch.stack(bg_losses).mean() if bg_losses else mask.new_tensor(0.0)
        loss_orth = torch.stack(orth_losses).mean() if orth_losses else mask.new_tensor(0.0)
        loss = bg_weight * loss_bg + orth_weight * loss_orth
        return loss, {
            'loss_cudm_bg': loss_bg.detach(),
            'loss_cudm_orth': loss_orth.detach(),
        }

    def _compute_total_loss(self, outputs, mask, pixel_weight=None):
        pred = self._select_main_pred(outputs)
        loss_seg, loss_stats = self.loss_seg(pred, mask)
        loss_cudm, cudm_stats = self._compute_cudm_disentangle_loss(outputs, mask)
        loss_total = loss_seg + loss_cudm
        loss_dict = {
            'loss_seg': loss_seg.detach(),
            'loss_cudm': loss_cudm.detach(),
            'loss_total': loss_total.detach(),
        }
        loss_dict.update(loss_stats)
        loss_dict.update(cudm_stats)
        return loss_total, pred, loss_dict

    def train_step(self, batch):
        ct = batch['ct'].float().to(self.device)
        pet = batch['pet'].float().to(self.device)
        mask = batch['mask'].float().to(self.device)
        outputs = _forward(self.networks, ct, pet, mask.shape[-2:])
        loss, pred, loss_dict = self._compute_total_loss(outputs, mask)
        return loss, pred, mask, loss_dict

    def _get_eval_model(self):
        if self.use_ema and self.ema_model is not None and self._current_epoch > self.ema_warmup_epochs:
            return self.ema_model
        return self.networks['model']

    @torch.no_grad()
    def evaluate(self, loader, threshold=None, use_ema=True):
        use_ema_actual = use_ema and self.use_ema and self._current_epoch > self.ema_warmup_epochs
        eval_model = self.ema_model if use_ema_actual else self.networks['model']
        eval_model.eval()
        th = threshold or getattr(self.config, 'eval_threshold', 0.5)
        m = SegmentationMetricsCIPA(threshold=th).to(self.device)
        m.reset()
        total_loss, n = 0.0, 0
        for batch in loader:
            ct = batch['ct'].float().to(self.device)
            pet = batch['pet'].float().to(self.device)
            mask = batch['mask'].float().to(self.device)
            outputs = eval_model(ct, pet, target_size=mask.shape[-2:])
            pred = self._select_main_pred(outputs)
            loss_seg, _ = self.loss_seg(pred, mask)
            total_loss += loss_seg.item() * ct.size(0)
            n += ct.size(0)
            m.update(pred, mask)
        eval_model.train() if not use_ema_actual else None
        if not use_ema_actual:
            for v in self.networks.values():
                v.train()
        out = m.compute()
        out['total_loss'] = total_loss / max(n, 1)
        return out

    def save_checkpoint(self, path, epoch):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {k: v.state_dict() for k, v in self.networks.items()}
        ckpt['epoch'] = epoch
        ckpt['optimizer'] = self.optimizer.state_dict()
        if self.use_ema and self.ema_model is not None:
            ckpt['ema_model'] = self.ema_model.state_dict()
        torch.save(ckpt, path)
        print('Saved:', path)
