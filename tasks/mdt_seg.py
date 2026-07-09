# -*- coding: utf-8 -*-
import os
import torch
import torch.nn.functional as F
from utils.metrics_seg import SegmentationMetricsCIPA
from utils.optimization import get_optimizer
from utils.seg_losses import BCEDiceLoss

try:
    from models.pet_prompted_ct_decoder import PET_PROMPT_LOG_KEYS
except ImportError:
    PET_PROMPT_LOG_KEYS = []

try:
    from models.pet_lap_hgl_prior import PET_LAP_HGL_LOG_KEYS
except ImportError:
    PET_LAP_HGL_LOG_KEYS = []

try:
    from models.pet_mrp_gsa import PET_MRP_GSA_LOG_KEYS
except ImportError:
    PET_MRP_GSA_LOG_KEYS = []


def mask_to_boundary(mask):
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = (mask > 0.5).float()
    kernel = torch.ones((1, 1, 3, 3), dtype=mask.dtype, device=mask.device)
    neighbor_sum = F.conv2d(mask, kernel, padding=1)
    boundary = ((neighbor_sum > 0) & (neighbor_sum < 9)).float()
    return boundary


def _forward(nets, ct, pet, target_size, pet_available=None, return_aux=False, forward_mode=None):
    kwargs = {'target_size': target_size}
    if pet_available is not None:
        kwargs['pet_available'] = pet_available
    if forward_mode is not None:
        kwargs['forward_mode'] = forward_mode
    if return_aux:
        kwargs['return_aux'] = return_aux
    return nets['model'](ct, pet, **kwargs)


def _is_finite_tensor(x):
    return torch.is_tensor(x) and torch.isfinite(x).all()


def _check_finite(name, x):
    if torch.is_tensor(x) and not torch.isfinite(x).all():
        raise RuntimeError(f'[NaN/Inf] {name} contains invalid values')


def _use_deep_supervision(config):
    if getattr(config, 'use_deep_supervision', False):
        return True
    return bool(getattr(config, 'deep_supervision', False))


def _deep_supervision_weights(config):
    raw = list(getattr(config, 'deep_supervision_weights', [1.0, 0.5, 0.25, 0.125]))
    if len(raw) < 4:
        raw.extend([raw[-1]] * (4 - len(raw)))
    raw = raw[:4]
    total = sum(raw)
    if total <= 0:
        return [0.25, 0.25, 0.25, 0.25]
    return [w / total for w in raw]


def _upsample_logits_to_target(logits, target):
    if logits.shape[-2:] != target.shape[-2:]:
        return F.interpolate(
            logits,
            size=target.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
    return logits


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

    def trainable_parameters(self):
        return [p for net in self.networks.values() for p in net.parameters() if p.requires_grad]

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
        use_ds = _use_deep_supervision(self.config)

        if not use_ds:
            loss_seg, loss_stats = self.loss_seg(pred, mask)
            loss_stats = dict(loss_stats)
            loss_stats['loss_seg'] = loss_seg.detach()
            loss_stats['use_deep_supervision'] = 0.0
            return loss_seg, pred, loss_stats

        aux_logits = outputs.get('aux_logits') if isinstance(outputs, dict) else None
        if not isinstance(aux_logits, (list, tuple)) or len(aux_logits) == 0:
            loss_seg, loss_stats = self.loss_seg(pred, mask)
            loss_stats = dict(loss_stats)
            loss_stats['loss_seg'] = loss_seg.detach()
            loss_stats['use_deep_supervision'] = 0.0
            return loss_seg, pred, loss_stats

        weights = _deep_supervision_weights(self.config)
        logits_list = [pred] + list(aux_logits)
        stage_names = ['main', 'aux_d2', 'aux_d3', 'aux_d4']

        total_loss = pred.new_tensor(0.0)
        stage_losses = {}
        for weight, name, logits in zip(weights, stage_names, logits_list):
            logits_up = _upsample_logits_to_target(logits, mask)
            stage_loss, _ = self.loss_seg(logits_up, mask)
            total_loss = total_loss + float(weight) * stage_loss
            stage_losses[name] = stage_loss.detach()

        loss_stats = {
            'loss_seg': total_loss.detach(),
            'use_deep_supervision': 1.0,
            'loss_main': stage_losses['main'],
            'loss_aux_d2': stage_losses['aux_d2'],
            'loss_aux_d3': stage_losses['aux_d3'],
            'loss_aux_d4': stage_losses['aux_d4'],
            'ds_weight_main': float(weights[0]),
            'ds_weight_d2': float(weights[1]),
            'ds_weight_d3': float(weights[2]),
            'ds_weight_d4': float(weights[3]),
        }
        return total_loss, pred, loss_stats

    def _compute_proxy_loss(self, outputs):
        weight = float(getattr(self.config, 'proxy_loss_weight', 0.0))
        if weight <= 0 or not isinstance(outputs, dict):
            zero = next(iter(self.networks['model'].parameters())).new_tensor(0.0)
            return zero, {'loss_proxy': zero.detach()}
        required = ('pet_low_proxy', 'pet_high_proxy', 'pet_low_real', 'pet_high_real', 'pet_available')
        if any(k not in outputs for k in required):
            zero = next(iter(self.networks['model'].parameters())).new_tensor(0.0)
            return zero, {'loss_proxy': zero.detach()}
        pet_available = outputs['pet_available'].float().view(-1, 1, 1, 1)
        available_count = pet_available.sum()
        if float(available_count.detach().cpu()) <= 0.0:
            zero = outputs['pet_low_proxy'].new_tensor(0.0)
            return zero, {'loss_proxy': zero.detach()}
        elements_per_sample = outputs['pet_low_proxy'][0].numel()
        denom = available_count.clamp_min(1.0) * float(elements_per_sample)
        low_l1 = (torch.abs(outputs['pet_low_proxy'] - outputs['pet_low_real'].detach()) * pet_available).sum() / denom
        high_l1 = (torch.abs(outputs['pet_high_proxy'] - outputs['pet_high_real'].detach()) * pet_available).sum() / denom
        loss_proxy = weight * (low_l1 + high_l1)
        return loss_proxy, {'loss_proxy': loss_proxy.detach()}

    def _compute_total_loss(self, outputs, mask, pixel_weight=None):
        loss_seg, pred, loss_stats = self._compute_segmentation_loss(outputs, mask)
        loss_cudm, cudm_stats = self._compute_cudm_disentangle_loss(outputs, mask)
        loss_fnet, fnet_stats = self._compute_fnet_sparse_aux_loss(outputs, mask)
        loss_proxy, proxy_stats = self._compute_proxy_loss(outputs)
        loss_mem = mask.new_tensor(0.0)
        if isinstance(outputs, dict) and isinstance(outputs.get('aux'), dict):
            aux = outputs['aux']
            if torch.is_tensor(aux.get('L_mem')):
                loss_mem = aux['L_mem']
        loss_boundary = mask.new_tensor(0.0)
        boundary_weight = float(getattr(self.config, 'boundary_loss_weight', 0.0))
        boundary_logits = None
        if boundary_weight > 0 and isinstance(outputs, dict) and outputs.get('boundary_logits') is not None:
            boundary_logits = outputs['boundary_logits']
            _check_finite('boundary_logits', boundary_logits)
            boundary_target = mask_to_boundary(mask)
            if boundary_logits.shape[-2:] != boundary_target.shape[-2:]:
                boundary_logits = F.interpolate(boundary_logits, size=boundary_target.shape[-2:], mode='bilinear', align_corners=False)
            loss_boundary = F.binary_cross_entropy_with_logits(boundary_logits, boundary_target)
        _check_finite('pred', pred)
        _check_finite('loss_seg', loss_seg)
        _check_finite('loss_mem', loss_mem)
        _check_finite('loss_boundary', loss_boundary)
        _check_finite('loss_proxy', loss_proxy)
        loss_total = loss_seg + loss_mem + loss_cudm + loss_fnet + loss_proxy + boundary_weight * loss_boundary
        _check_finite('loss_total', loss_total)
        loss_dict = {
            'loss_seg': loss_seg.detach(),
            'loss_mem': loss_mem.detach(),
            'loss_boundary': loss_boundary.detach(),
            'loss_cudm': loss_cudm.detach(),
            'loss_fnet': loss_fnet.detach(),
            'loss_proxy': loss_proxy.detach(),
            'loss_total': loss_total.detach(),
        }
        loss_dict.update(loss_stats)
        loss_dict.update(cudm_stats)
        loss_dict.update(fnet_stats)
        loss_dict.update(proxy_stats)
        return loss_total, pred, loss_dict

    def train_step(self, batch, forward_mode):
        if forward_mode not in ('full', 'missing'):
            raise ValueError(f'Unsupported training route: {forward_mode}')
        ct = batch['ct'].float().to(self.device)
        mask = batch['mask'].float().to(self.device)
        if forward_mode == 'full':
            pet = batch['pet'].float().to(self.device)
        else:
            pet = None
        outputs = _forward(
            self.networks,
            ct,
            pet,
            mask.shape[-2:],
            forward_mode=forward_mode,
        )
        loss_seg, pred, loss_stats = self._compute_segmentation_loss(outputs, mask)
        if forward_mode == 'missing':
            missing_weight = float(getattr(self.config, 'missing_loss_weight', 1.0))
            loss_total = missing_weight * loss_seg
        else:
            loss_total = loss_seg
        zero = loss_seg.detach().new_tensor(0.0)
        loss_dict = dict(loss_stats)
        loss_dict.update({
            'loss_seg': loss_seg.detach(),
            'loss_total': loss_total.detach(),
            'loss_full': loss_seg.detach() if forward_mode == 'full' else zero,
            'loss_missing': loss_seg.detach() if forward_mode == 'missing' else zero,
            'train_route_full': 1.0 if forward_mode == 'full' else 0.0,
            'train_route_missing': 1.0 if forward_mode == 'missing' else 0.0,
        })
        _check_finite('loss_seg', loss_dict.get('loss_seg', loss_total))
        _check_finite('loss_total', loss_dict.get('loss_total', loss_total))
        return loss_total, pred, mask, loss_dict

    @torch.no_grad()
    def evaluate(self, loader, threshold=None, eval_mode="full", random_pet_drop_prob=0.0, random_seed=2026, tag="val", force_missing_pet=None):
        eval_model = self.networks['model']
        eval_model.eval()
        th = threshold or getattr(self.config, 'eval_threshold', 0.5)
        mode_alias = {
            'full': 'full',
            'full_pet': 'full',
            'missing': 'fixed_missing',
            'fixed_missing': 'fixed_missing',
            'fixed_missing_pet': 'fixed_missing',
            'random': 'random_missing',
            'random_missing': 'random_missing',
            'random_missing_pet': 'random_missing',
        }
        eval_mode = mode_alias.get(str(eval_mode), str(eval_mode))
        if eval_mode not in ('full', 'fixed_missing', 'random_missing'):
            raise ValueError(f'Unsupported eval_mode={eval_mode}.')
        print(f'[evaluate] tag={tag} mode={eval_mode}')
        rng = torch.Generator(device=self.device)
        rng.manual_seed(int(random_seed))
        log_dmome = bool(getattr(self.config, 'log_dmome_weights', True))
        model_unwrapped = self._unwrap(eval_model)
        use_dmome = bool(getattr(model_unwrapped, 'use_dmome', False))
        lap_hgl_prior = getattr(model_unwrapped, 'pet_prior_type', None)
        need_aux = (use_dmome and log_dmome) or (lap_hgl_prior not in (None, 'none'))
        m = SegmentationMetricsCIPA(threshold=th).to(self.device)
        m.reset()
        total_loss, n = 0.0, 0
        gate_sum = {'pet_gate_mean': 0.0, 'prior_gate_mean': 0.0}
        gate_n = 0
        prompt_sum = {k: 0.0 for k in PET_PROMPT_LOG_KEYS}
        prompt_n = 0
        lap_hgl_sum = {k: 0.0 for k in PET_LAP_HGL_LOG_KEYS}
        lap_hgl_n = 0
        pet_mrp_sum = {k: 0.0 for k in PET_MRP_GSA_LOG_KEYS}
        pet_mrp_n = 0
        dmome_weight_acc = {}
        dmome_weight_batches = 0
        for batch_idx, batch in enumerate(loader):
            ct = batch['ct'].float().to(self.device)
            pet = batch['pet'].float().to(self.device)
            mask = batch['mask'].float().to(self.device)
            if eval_mode == 'full':
                pet_available = torch.ones(ct.shape[0], device=self.device, dtype=torch.long)
            elif eval_mode == 'fixed_missing':
                pet_available = torch.zeros(ct.shape[0], device=self.device, dtype=torch.long)
            else:
                prob = float(random_pet_drop_prob)
                pet_available = (torch.rand(ct.shape[0], device=self.device, generator=rng) >= prob).long()
            if force_missing_pet is not None:
                pet_available = torch.zeros(ct.shape[0], device=self.device, dtype=torch.long) if force_missing_pet else torch.ones(ct.shape[0], device=self.device, dtype=torch.long)
            outputs = _forward(
                self.networks,
                ct,
                pet,
                mask.shape[-2:],
                pet_available=pet_available,
                forward_mode='auto',
            )
            pred = self._select_main_pred(outputs)
            loss_seg, _ = self.loss_seg(pred, mask)
            total_loss += loss_seg.item() * ct.size(0)
            n += ct.size(0)
            m.update(pred, mask)
            if isinstance(outputs, dict) and isinstance(outputs.get('fusion_aux'), list):
                from models.simmlm_dmome_fusion import summarize_dmome_weights
                batch_summary = summarize_dmome_weights(outputs['fusion_aux'])
                for key, value in batch_summary.items():
                    dmome_weight_acc[key] = dmome_weight_acc.get(key, 0.0) + float(value)
                dmome_weight_batches += 1
            if isinstance(outputs, dict) and isinstance(outputs.get('aux'), dict):
                aux = outputs['aux']
                for key in gate_sum:
                    val = aux.get(key)
                    if torch.is_tensor(val):
                        gate_sum[key] += float(val.detach().mean().cpu())
                    elif val is not None:
                        gate_sum[key] += float(val)
                gate_n += 1
                for key in PET_PROMPT_LOG_KEYS:
                    val = aux.get(key)
                    if torch.is_tensor(val):
                        prompt_sum[key] += float(val.detach().cpu())
                    elif val is not None:
                        prompt_sum[key] += float(val)
                if any(k in aux for k in PET_PROMPT_LOG_KEYS):
                    prompt_n += 1
                for key in PET_LAP_HGL_LOG_KEYS:
                    val = aux.get(key)
                    if torch.is_tensor(val):
                        lap_hgl_sum[key] += float(val.detach().cpu())
                    elif val is not None:
                        lap_hgl_sum[key] += float(val)
                if any(k in aux for k in PET_LAP_HGL_LOG_KEYS):
                    lap_hgl_n += 1
                for key in PET_MRP_GSA_LOG_KEYS:
                    val = aux.get(key)
                    if torch.is_tensor(val):
                        pet_mrp_sum[key] += float(val.detach().cpu())
                    elif val is not None:
                        pet_mrp_sum[key] += float(val)
                if any(k in aux for k in PET_MRP_GSA_LOG_KEYS):
                    pet_mrp_n += 1
        for v in self.networks.values():
            v.train()
        out = m.compute()
        out['total_loss'] = total_loss / max(n, 1)
        if gate_n > 0:
            out['pet_gate_mean'] = gate_sum['pet_gate_mean'] / gate_n
            out['prior_gate_mean'] = gate_sum['prior_gate_mean'] / gate_n
        if prompt_n > 0:
            for key, total in prompt_sum.items():
                out[key] = total / prompt_n
        if lap_hgl_n > 0:
            for key, total in lap_hgl_sum.items():
                out[key] = total / lap_hgl_n
        if pet_mrp_n > 0:
            for key, total in pet_mrp_sum.items():
                out[key] = total / pet_mrp_n
            print(
                f'[evaluate] PET-MRP-GSA ({tag}): '
                + ' '.join(f'{k}={out[k]:.4f}' for k in PET_MRP_GSA_LOG_KEYS if k in out)
            )
        if dmome_weight_batches > 0:
            dmome_summary = {k: v / dmome_weight_batches for k, v in dmome_weight_acc.items()}
            out.update(dmome_summary)
            if log_dmome:
                summary_text = ' '.join(f'{k}={v:.4f}' for k, v in sorted(dmome_summary.items()))
                print(f'[evaluate] DMoME weights ({tag}): {summary_text}')
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
