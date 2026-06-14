# -*- coding: utf-8 -*-
import os
import torch
import torch.nn.functional as F
from utils.metrics_seg import SegmentationMetricsCIPA
from utils.optimization import get_optimizer
from utils.seg_losses import BCEDiceLoss


def mask_to_boundary(mask):
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = (mask > 0.5).float()
    kernel = torch.ones((1, 1, 3, 3), dtype=mask.dtype, device=mask.device)
    neighbor_sum = F.conv2d(mask, kernel, padding=1)
    boundary = ((neighbor_sum > 0) & (neighbor_sum < 9)).float()
    return boundary


def _forward(nets, ct, pet, target_size, pet_available=None):
    if pet_available is not None:
        try:
            return nets['model'](ct, pet, pet_available=pet_available, target_size=target_size)
        except TypeError:
            return nets['model'](ct, pet, target_size=target_size)
    return nets['model'](ct, pet, target_size=target_size)


class MDTSegTeacher:
    def __init__(self, networks, config):
        self.networks = {k: v for k, v in networks.items() if v is not None}
        self.config = config
        self.device = torch.device('cuda', int(config.gpus[0]))
        self.data_parallel = torch.cuda.is_available() and len(getattr(config, 'gpus', [])) > 1
        for key, v in list(self.networks.items()):
            v.to(self.device)
            if self.data_parallel:
                self.networks[key] = torch.nn.DataParallel(v, device_ids=list(config.gpus), output_device=int(config.gpus[0]))
        if self.data_parallel:
            print(f'[+] DataParallel enabled on GPUs: {list(config.gpus)}')

        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        self.loss_seg = BCEDiceLoss(
            bce_weight=getattr(config, 'bce_weight', 1.0),
            dice_weight=getattr(config, 'dice_weight', 1.0),
            smooth=getattr(config, 'loss_smooth', 1.0),
            pos_weight=getattr(config, 'pos_weight', None),
        ).to(self.device)

        lr = getattr(config, 'decoder_lr', None) or config.learning_rate
        if getattr(config, 'freeze_non_adc', False):
            for name, param in self.networks['model'].named_parameters():
                param.requires_grad = 'adc_mac' in name
            trainable = [p for p in self.networks['model'].parameters() if p.requires_grad]
            if not trainable:
                raise ValueError('freeze_non_adc=True but no ADC-MAC parameters were found. Set use_adc_mac=True.')
            print(f'[+] freeze_non_adc enabled, trainable ADC-MAC params={sum(p.numel() for p in trainable)}')
        else:
            trainable = [p for p in self.networks['model'].parameters() if p.requires_grad]
        self.optimizer = get_optimizer(
            [{'params': trainable, 'lr': lr}],
            config.optimizer,
            config.learning_rate,
            config.weight_decay,
        )
        self.scheduler = None

    def set_epoch(self, epoch):
        self._current_epoch = epoch

    def _select_main_pred(self, outputs):
        if isinstance(outputs, dict):
            if 'logits' in outputs:
                return outputs['logits']
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

    def _compute_fnet_sparse_aux_loss(self, outputs, mask):
        recon_weight = float(getattr(self.config, 'fnet_aux_recon_weight', 0.0))
        sparse_weight = float(getattr(self.config, 'fnet_aux_sparse_weight', 0.0))
        decor_weight = float(getattr(self.config, 'fnet_aux_decor_weight', 0.0))
        edge_weight = float(getattr(self.config, 'fnet_aux_edge_weight', 0.0))
        if recon_weight <= 0 and sparse_weight <= 0 and decor_weight <= 0 and edge_weight <= 0:
            zero = mask.new_tensor(0.0)
            return zero, {
                'loss_fnet_recon': zero.detach(),
                'loss_fnet_sparse': zero.detach(),
                'loss_fnet_decor': zero.detach(),
                'loss_fnet_edge': zero.detach(),
            }
        if not isinstance(outputs, dict) or not outputs.get('fusion_aux'):
            zero = mask.new_tensor(0.0)
            return zero, {
                'loss_fnet_recon': zero.detach(),
                'loss_fnet_sparse': zero.detach(),
                'loss_fnet_decor': zero.detach(),
                'loss_fnet_edge': zero.detach(),
            }
        try:
            from models.fnet_sparse_fusion import fnet_sparse_auxiliary_loss
        except ImportError:
            zero = mask.new_tensor(0.0)
            return zero, {
                'loss_fnet_recon': zero.detach(),
                'loss_fnet_sparse': zero.detach(),
                'loss_fnet_decor': zero.detach(),
                'loss_fnet_edge': zero.detach(),
            }
        return fnet_sparse_auxiliary_loss(
            outputs['fusion_aux'],
            mask=mask,
            recon_weight=recon_weight,
            sparse_weight=sparse_weight,
            decor_weight=decor_weight,
            edge_weight=edge_weight,
        )

    def _compute_segmentation_loss(self, outputs, mask):
        pred = self._select_main_pred(outputs)
        loss_seg, loss_stats = self.loss_seg(pred, mask)
        if not getattr(self.config, 'deep_supervision', False):
            return loss_seg, pred, loss_stats
        if not isinstance(outputs, dict) or not isinstance(outputs.get('preds'), (list, tuple)):
            return loss_seg, pred, loss_stats

        preds = list(outputs['preds'])
        weights = list(getattr(self.config, 'deep_supervision_weights', [0.5, 0.25, 0.125, 0.125]))
        if len(weights) < len(preds):
            weights.extend([weights[-1]] * (len(preds) - len(weights)))
        weights = weights[:len(preds)]
        weight_sum = sum(weights) if sum(weights) > 0 else 1.0

        ds_loss = pred.new_tensor(0.0)
        for w, p in zip(weights, preds):
            stage_loss, _ = self.loss_seg(p, mask)
            ds_loss = ds_loss + float(w) * stage_loss
        ds_loss = ds_loss / weight_sum
        loss_stats = dict(loss_stats)
        loss_stats['loss_deep_supervision'] = ds_loss.detach()
        return ds_loss, pred, loss_stats

    def _compute_total_loss(self, outputs, mask, pixel_weight=None):
        loss_seg, pred, loss_stats = self._compute_segmentation_loss(outputs, mask)
        loss_cudm, cudm_stats = self._compute_cudm_disentangle_loss(outputs, mask)
        loss_fnet, fnet_stats = self._compute_fnet_sparse_aux_loss(outputs, mask)
        loss_boundary = mask.new_tensor(0.0)
        boundary_weight = float(getattr(self.config, 'boundary_loss_weight', 0.0))
        if boundary_weight > 0 and isinstance(outputs, dict) and outputs.get('boundary_logits') is not None:
            boundary_logits = outputs['boundary_logits']
            boundary_target = mask_to_boundary(mask)
            if boundary_logits.shape[-2:] != boundary_target.shape[-2:]:
                boundary_logits = F.interpolate(boundary_logits, size=boundary_target.shape[-2:], mode='bilinear', align_corners=False)
            loss_boundary = F.binary_cross_entropy_with_logits(boundary_logits, boundary_target)
        loss_total = loss_seg + loss_cudm + loss_fnet + boundary_weight * loss_boundary
        loss_dict = {
            'loss_seg': loss_seg.detach(),
            'loss_boundary': loss_boundary.detach(),
            'loss_cudm': loss_cudm.detach(),
            'loss_fnet': loss_fnet.detach(),
            'loss_total': loss_total.detach(),
        }
        loss_dict.update(loss_stats)
        loss_dict.update(cudm_stats)
        loss_dict.update(fnet_stats)
        return loss_total, pred, loss_dict

    def train_step(self, batch):
        ct = batch['ct'].float().to(self.device)
        pet = batch['pet'].float().to(self.device)
        mask = batch['mask'].float().to(self.device)
        pet_available = batch.get('pet_available')
        if pet_available is not None:
            pet_available = pet_available.to(self.device)
        outputs = _forward(self.networks, ct, pet, mask.shape[-2:], pet_available=pet_available)
        loss, pred, loss_dict = self._compute_total_loss(outputs, mask)
        return loss, pred, mask, loss_dict

    @torch.no_grad()
    def evaluate(self, loader, threshold=None, force_missing_pet=False, tag="val"):
        eval_model = self.networks['model']
        eval_model.eval()
        th = threshold or getattr(self.config, 'eval_threshold', 0.5)
        mode = 'missing' if force_missing_pet else 'full'
        print(f'[evaluate] tag={tag} mode={mode}')
        m = SegmentationMetricsCIPA(threshold=th).to(self.device)
        m.reset()
        total_loss, n = 0.0, 0
        gate_sum = {'pet_gate_mean': 0.0, 'text_gate_mean': 0.0, 'prior_gate_mean': 0.0}
        gate_n = 0
        for batch in loader:
            ct = batch['ct'].float().to(self.device)
            pet = batch['pet'].float().to(self.device)
            mask = batch['mask'].float().to(self.device)
            pet_available = batch.get('pet_available')
            if pet_available is not None:
                pet_available = pet_available.to(self.device)
            if force_missing_pet:
                pet = torch.zeros_like(pet)
                pet_available = torch.zeros(ct.shape[0], device=self.device, dtype=torch.float32)
            outputs = _forward(self.networks, ct, pet, mask.shape[-2:], pet_available=pet_available)
            pred = self._select_main_pred(outputs)
            loss_seg, _ = self.loss_seg(pred, mask)
            total_loss += loss_seg.item() * ct.size(0)
            n += ct.size(0)
            m.update(pred, mask)
            if isinstance(outputs, dict) and isinstance(outputs.get('aux'), dict):
                aux = outputs['aux']
                for key in gate_sum:
                    val = aux.get(key)
                    if torch.is_tensor(val):
                        gate_sum[key] += float(val.detach().mean().cpu())
                    elif val is not None:
                        gate_sum[key] += float(val)
                gate_n += 1
        for v in self.networks.values():
            v.train()
        out = m.compute()
        out['total_loss'] = total_loss / max(n, 1)
        if gate_n > 0:
            out['pet_gate_mean'] = gate_sum['pet_gate_mean'] / gate_n
            out['text_gate_mean'] = gate_sum['text_gate_mean'] / gate_n
            out['prior_gate_mean'] = gate_sum['prior_gate_mean'] / gate_n
        return out

    def _unwrap(self, model):
        return model.module if isinstance(model, torch.nn.DataParallel) else model

    def model_state_dict(self, model):
        return self._unwrap(model).state_dict()

    def load_model_state_dict(self, model, state_dict, strict=False):
        return self._unwrap(model).load_state_dict(state_dict, strict=strict)

    def save_checkpoint(self, path, epoch):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {k: self.model_state_dict(v) for k, v in self.networks.items()}
        ckpt['epoch'] = epoch
        ckpt['optimizer'] = self.optimizer.state_dict()
        torch.save(ckpt, path)
        print('Saved:', path)
