# -*- coding: utf-8 -*-
import math
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


def _forward(nets, ct, pet, target_size, pet_available=None, return_aux=False, forward_mode=None, mask=None):
    kwargs = {'target_size': target_size}
    if pet_available is not None:
        kwargs['pet_available'] = pet_available
    if forward_mode is not None:
        kwargs['forward_mode'] = forward_mode
    if return_aux:
        kwargs['return_aux'] = return_aux
    if mask is not None:
        import inspect
        try:
            if 'mask' in inspect.signature(nets['model'].forward).parameters:
                kwargs['mask'] = mask
        except (TypeError, ValueError):
            pass
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
        return F.interpolate(logits, size=target.shape[-2:], mode='bilinear', align_corners=False)
    return logits


class MDTSegTeacher:
    def __init__(self, networks, config):
        self.networks = {k: v for k, v in networks.items() if v is not None}
        self.config = config
        if torch.cuda.is_available() and getattr(config, 'gpus', None):
            self.device = torch.device('cuda', int(config.gpus[0]))
        else:
            self.device = torch.device('cpu')
        self.data_parallel = torch.cuda.is_available() and len(getattr(config, 'gpus', [])) > 1
        for key, v in list(self.networks.items()):
            v.to(self.device)
            if self.data_parallel:
                self.networks[key] = torch.nn.DataParallel(v, device_ids=list(config.gpus), output_device=int(config.gpus[0]))
        if self.data_parallel:
            print(f'[+] DataParallel enabled on GPUs: {list(config.gpus)}')

        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        self.loss_seg = BCEDiceLoss(bce_weight=getattr(config, 'bce_weight', 1.0), dice_weight=getattr(config, 'dice_weight', 1.0), smooth=getattr(config, 'loss_smooth', 1.0), pos_weight=getattr(config, 'pos_weight', None)).to(self.device)
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
        self.optimizer = get_optimizer([{'params': trainable, 'lr': lr}], config.optimizer, config.learning_rate, config.weight_decay)
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
        zero = mask.new_tensor(0.0)
        return zero, {'loss_cudm_bg': zero.detach(), 'loss_cudm_orth': zero.detach()}

    def _compute_fnet_sparse_aux_loss(self, outputs, mask):
        zero = mask.new_tensor(0.0)
        return zero, {'loss_fnet_recon': zero.detach(), 'loss_fnet_sparse': zero.detach(), 'loss_fnet_decor': zero.detach(), 'loss_fnet_edge': zero.detach()}

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
        loss_stats = {'loss_seg': total_loss.detach(), 'use_deep_supervision': 1.0, 'loss_main': stage_losses['main'], 'loss_aux_d2': stage_losses['aux_d2'], 'loss_aux_d3': stage_losses['aux_d3'], 'loss_aux_d4': stage_losses['aux_d4'], 'ds_weight_main': float(weights[0]), 'ds_weight_d2': float(weights[1]), 'ds_weight_d3': float(weights[2]), 'ds_weight_d4': float(weights[3])}
        return total_loss, pred, loss_stats

    def _compute_proxy_loss(self, outputs):
        zero = next(iter(self.networks['model'].parameters())).new_tensor(0.0)
        return zero, {'loss_proxy': zero.detach()}

    def _compute_hatr_residual_loss(self, outputs, mask):
        zero = mask.new_tensor(0.0)
        required = ['hatr_teacher_full_logits', 'hatr_anchor_logits', 'hatr_teacher_full_states', 'hatr_counterfactual_states', 'hatr_pred_residuals']
        if not isinstance(outputs, dict) or not all(k in outputs for k in required):
            return zero, {'loss_hatr': zero.detach()}
        full_logits = outputs['hatr_teacher_full_logits']
        anchor_logits = outputs['hatr_anchor_logits']
        full_states = outputs['hatr_teacher_full_states']
        ct_cf_states = outputs['hatr_counterfactual_states']
        pred_residuals = outputs['hatr_pred_residuals']
        with torch.no_grad():
            p_f = torch.sigmoid(full_logits.float())
            p_a = torch.sigmoid(anchor_logits.detach().float())
            y = mask.float()
            e_f = (p_f - y).pow(2)
            e_a = (p_a - y).pow(2)
            advantage = F.relu(e_a - e_f) / (e_a + e_f + 1e-6)
        loss_total = zero
        stats = {'loss_hatr': zero.detach(), 'hatr_advantage_mean': advantage.mean().detach(), 'hatr_advantage_active_ratio': (advantage > 0).float().mean().detach(), 'hatr_full_error_mean': e_f.mean().detach(), 'hatr_anchor_error_mean': e_a.mean().detach()}
        for idx, (full_state, ct_cf_state, pred_residual) in enumerate(zip(full_states, ct_cf_states, pred_residuals), start=1):
            raw_residual = full_state.detach() - ct_cf_state.detach()
            advantage_s = F.interpolate(advantage, size=full_state.shape[-2:], mode='bilinear', align_corners=False)
            target_s = advantage_s * raw_residual
            loss_s = F.smooth_l1_loss(pred_residual.float(), target_s.float())
            loss_total = loss_total + loss_s
            stats[f'hatr_s{idx}_raw_residual_rms'] = raw_residual.float().pow(2).mean().sqrt().detach()
            stats[f'hatr_s{idx}_target_rms'] = target_s.float().pow(2).mean().sqrt().detach()
            stats[f'hatr_s{idx}_pred_rms'] = pred_residual.float().pow(2).mean().sqrt().detach()
            stats[f'hatr_s{idx}_hidden_rms'] = outputs['hatr_hidden_states'][idx - 1].float().pow(2).mean().sqrt().detach() if 'hatr_hidden_states' in outputs else zero.detach()
            stats[f'hatr_s{idx}_loss'] = loss_s.detach()
        loss_total = loss_total / 4.0
        stats['loss_hatr'] = loss_total.detach()
        return loss_total, stats

    def _masked_patch_mean(self, value, mask):
        denom = mask.sum().clamp_min(1.0)
        return (value * mask).sum() / denom, (mask.sum() > 0).float()

    def _compute_balanced_patch_risk(self, logits, mask, patch_size):
        pixel_loss = F.binary_cross_entropy_with_logits(logits.float(), mask.float(), reduction='none')
        fg_mass = F.adaptive_avg_pool2d(mask.float(), patch_size)
        fg_error = F.adaptive_avg_pool2d(pixel_loss * mask.float(), patch_size) / (fg_mass + 1e-6)
        bg_mask = 1.0 - mask.float()
        bg_mass = F.adaptive_avg_pool2d(bg_mask, patch_size)
        bg_error = F.adaptive_avg_pool2d(pixel_loss * bg_mask, patch_size) / (bg_mass + 1e-6)
        has_fg = (fg_mass > 0).float()
        has_bg = (bg_mass > 0).float()
        return (has_fg * fg_error + has_bg * bg_error) / (has_fg + has_bg).clamp_min(1.0), fg_mass, bg_mass

    def _compute_pgmr_loss(self, outputs, mask):
        ct_logits = outputs['gvtc_ct_logits']
        full_logits = outputs['gvtc_full_logits']
        comp_logits = outputs['gvtc_comp_logits']
        patch_resolution = outputs['gvtc_delta_d4'].shape[-2:]
        risk_ct, fg_mass, bg_mass = self._compute_balanced_patch_risk(ct_logits, mask, patch_resolution)
        risk_full, _, _ = self._compute_balanced_patch_risk(full_logits, mask, patch_resolution)
        risk_comp, _, _ = self._compute_balanced_patch_risk(comp_logits, mask, patch_resolution)
        risk_best = torch.minimum(risk_ct.detach(), risk_full.detach())
        violation = F.relu(risk_comp - risk_best)
        fg_patch = (F.adaptive_max_pool2d(mask.float(), patch_resolution) > 0).float()
        bg_patch = 1.0 - fg_patch
        loss_pgmr_fg, valid_fg = self._masked_patch_mean(violation, fg_patch)
        loss_pgmr_bg, valid_bg = self._masked_patch_mean(violation, bg_patch)
        loss_pgmr = (valid_fg * loss_pgmr_fg + valid_bg * loss_pgmr_bg) / (valid_fg + valid_bg).clamp_min(1.0)
        diagnostics = {
            'loss_pgmr_fg': loss_pgmr_fg.detach(),
            'loss_pgmr_bg': loss_pgmr_bg.detach(),
            'pgmr_positive_pet_gain_ratio': (risk_full < risk_ct).float().mean().detach(),
            'pgmr_positive_pet_gain_fg_ratio': ((risk_full < risk_ct) * fg_patch).sum().div(fg_patch.sum().clamp_min(1.0)).detach(),
            'pgmr_positive_pet_gain_bg_ratio': ((risk_full < risk_ct) * bg_patch).sum().div(bg_patch.sum().clamp_min(1.0)).detach(),
            'pgmr_ct_dominant_ratio': (risk_ct <= risk_full).float().mean().detach(),
            'pgmr_violation_ratio': (risk_comp > risk_best).float().mean().detach(),
            'pgmr_violation_fg_ratio': ((risk_comp > risk_best) * fg_patch).sum().div(fg_patch.sum().clamp_min(1.0)).detach(),
            'pgmr_violation_bg_ratio': ((risk_comp > risk_best) * bg_patch).sum().div(bg_patch.sum().clamp_min(1.0)).detach(),
            'pgmr_comp_better_than_ct_ratio': (risk_comp < risk_ct).float().mean().detach(),
            'pgmr_comp_better_than_ct_fg_ratio': ((risk_comp < risk_ct) * fg_patch).sum().div(fg_patch.sum().clamp_min(1.0)).detach(),
            'pgmr_comp_better_than_ct_bg_ratio': ((risk_comp < risk_ct) * bg_patch).sum().div(bg_patch.sum().clamp_min(1.0)).detach(),
            'pgmr_comp_better_than_full_ratio': (risk_comp < risk_full).float().mean().detach(),
            'risk_ct': risk_ct.mean().detach(),
            'risk_full': risk_full.mean().detach(),
            'risk_comp': risk_comp.mean().detach(),
        }
        return loss_pgmr, diagnostics

    def _compute_ptgc_joint_loss(self, outputs, mask):
        mode = str(outputs.get('gvtc_ablation_mode', 'gvtc_pgmr'))
        config_mode = str(getattr(self.config, 'ptgc_ablation_mode', mode))
        if mode != config_mode:
            raise RuntimeError(f'Model mode {mode!r} does not match config mode {config_mode!r}')
        ct_logits = outputs['gvtc_ct_logits']
        full_logits = outputs['gvtc_full_logits']
        comp_logits = outputs.get('gvtc_comp_logits', ct_logits)
        loss_ct, _ = self.loss_seg(ct_logits, mask)
        loss_full, _ = self.loss_seg(full_logits, mask)
        loss_comp, _ = self.loss_seg(comp_logits, mask)
        loss_seg_joint = 0.5 * loss_full + 0.5 * loss_comp if mode != 'baseline' else 0.5 * loss_full + 0.5 * loss_ct
        stats = {
            'loss_gvtc_ct': loss_ct.detach(),
            'loss_gvtc_full': loss_full.detach(),
            'loss_gvtc_comp': loss_comp.detach(),
            'loss_seg_joint': loss_seg_joint.detach(),
        }
        if mode == 'baseline':
            return loss_seg_joint, {**stats, 'loss_pgmr': loss_ct.new_tensor(0.0).detach(), 'weighted_loss_pgmr': loss_ct.new_tensor(0.0).detach(), 'loss_pgmr_fg': loss_ct.new_tensor(0.0).detach(), 'loss_pgmr_bg': loss_ct.new_tensor(0.0).detach()}
        if mode == 'gvtc':
            return loss_seg_joint, {**stats, 'loss_pgmr': loss_ct.new_tensor(0.0).detach(), 'weighted_loss_pgmr': loss_ct.new_tensor(0.0).detach(), 'loss_pgmr_fg': loss_ct.new_tensor(0.0).detach(), 'loss_pgmr_bg': loss_ct.new_tensor(0.0).detach()}
        if mode == 'gvtc_pgmr':
            loss_pgmr, pgmr_stats = self._compute_pgmr_loss(outputs, mask)
            weighted_loss_pgmr = float(getattr(self.config, 'pgmr_weight', 0.1)) * loss_pgmr
            loss_total = loss_seg_joint + weighted_loss_pgmr
            return loss_total, {**stats, 'loss_pgmr': loss_pgmr.detach(), 'weighted_loss_pgmr': weighted_loss_pgmr.detach(), **pgmr_stats}
        raise ValueError(f'Unsupported PTGC ablation mode: {mode}')

    def _compute_total_loss(self, outputs, mask, pixel_weight=None):
        if isinstance(outputs, dict) and 'gvtc_ablation_mode' in outputs:
            loss_total, stats = self._compute_ptgc_joint_loss(outputs, mask)
            loss_dict = {'loss_total': loss_total.detach(), **stats}
            return loss_total, outputs['logits'], loss_dict
        loss_seg, pred, loss_stats = self._compute_segmentation_loss(outputs, mask)
        loss_dict = {'loss_seg': loss_seg.detach(), 'loss_total': loss_seg.detach()}
        loss_dict.update(loss_stats)
        return loss_seg, pred, loss_dict

    def train_step(self, batch, forward_mode):
        if forward_mode not in ('full', 'missing'):
            raise ValueError(f'Unsupported training route: {forward_mode}')
        ct = batch['ct'].float().to(self.device)
        mask = batch['mask'].float().to(self.device)
        pet = batch['pet'].float().to(self.device) if forward_mode == 'full' else None
        outputs = _forward(self.networks, ct, pet, mask.shape[-2:], forward_mode=forward_mode, mask=mask)
        loss_total, pred, loss_dict = self._compute_total_loss(outputs, mask)
        aux_losses = outputs.get('aux_losses', {}) if isinstance(outputs, dict) else {}
        diagnostics = outputs.get('diagnostics', {}) if isinstance(outputs, dict) else {}
        zero = loss_total.new_tensor(0.0)
        route_loss = aux_losses.get('pg_mtr_route_loss', zero)
        mem_loss = aux_losses.get('pg_mtr_mem_loss', zero)
        bank_loss = aux_losses.get('mtib_bank_loss', zero)
        comp_loss = aux_losses.get('mtib_comp_loss', zero)
        hatr_loss, hatr_stats = self._compute_hatr_residual_loss(outputs, mask)
        route_weight = float(getattr(self.config, 'pg_mtr_route_weight', 0.1))
        mem_weight = float(getattr(self.config, 'pg_mtr_mem_weight', 0.05))
        bank_weight = float(getattr(self.config, 'mtib_bank_weight', 0.05))
        comp_weight = float(getattr(self.config, 'mtib_comp_weight', 0.05))
        hatr_weight = float(getattr(self.config, 'hatr_weight', 0.1))
        if forward_mode == 'missing' and getattr(self.config, 'model_arch', '') != 'dual_decoder_ptgc':
            missing_weight = float(getattr(self.config, 'missing_loss_weight', 1.0))
            loss_total = missing_weight * loss_total
        elif 'hatr_pred_residuals' in outputs:
            loss_total = loss_total + hatr_weight * hatr_loss
        elif 'mtib_bank_loss' in aux_losses or 'mtib_comp_loss' in aux_losses:
            loss_total = loss_total + bank_weight * bank_loss + comp_weight * comp_loss
        elif getattr(self.config, 'model_arch', '') not in ('dual_decoder_ptgc',):
            loss_total = loss_total + route_weight * route_loss + mem_weight * mem_loss
        loss_dict.update({'loss_seg': loss_dict.get('loss_seg', loss_total.detach()), 'loss_total': loss_total.detach(), 'loss_full': loss_total.detach() if forward_mode == 'full' else zero, 'loss_missing': loss_total.detach() if forward_mode == 'missing' else zero, 'train_route_full': 1.0 if forward_mode == 'full' else 0.0, 'train_route_missing': 1.0 if forward_mode == 'missing' else 0.0, 'loss_pg_mtr_route': route_loss.detach(), 'loss_pg_mtr_mem': mem_loss.detach(), 'weighted_loss_pg_mtr_route': (route_weight * route_loss).detach(), 'weighted_loss_pg_mtr_mem': (mem_weight * mem_loss).detach(), 'loss_mtib_bank': bank_loss.detach(), 'loss_mtib_comp': comp_loss.detach(), 'weighted_loss_mtib_bank': (bank_weight * bank_loss).detach(), 'weighted_loss_mtib_comp': (comp_weight * comp_loss).detach(), 'loss_hatr': hatr_loss.detach(), 'weighted_loss_hatr': (hatr_weight * hatr_loss).detach()})
        loss_dict.update(hatr_stats)
        for key, value in diagnostics.items():
            if torch.is_tensor(value) and value.numel() > 0:
                loss_dict[f'diag_{key}'] = value.detach().float().mean()
        return loss_total, pred, mask, loss_dict

    def _accumulate_pg_diagnostics(self, pg_diag_sum, pg_diag_count, diagnostics):
        for key, value in diagnostics.items():
            scalar = None
            if torch.is_tensor(value):
                if value.numel() == 0:
                    continue
                scalar = float(value.detach().float().mean().cpu())
            else:
                try:
                    scalar = float(value)
                except (TypeError, ValueError):
                    continue
            if not math.isfinite(scalar):
                continue
            pg_diag_sum[key] = pg_diag_sum.get(key, 0.0) + scalar
            pg_diag_count[key] = pg_diag_count.get(key, 0) + 1

    def _print_pg_diag_summary(self, tag, pg_diag_avg):
        if not pg_diag_avg:
            return
        print(f'[evaluate] PG-MTR diagnostics ({tag}):')

    @torch.no_grad()
    def evaluate(self, loader, threshold=None, eval_mode="full", random_pet_drop_prob=0.0, random_seed=2026, tag="val", force_missing_pet=None):
        eval_model = self.networks['model']
        eval_model.eval()
        th = threshold or getattr(self.config, 'eval_threshold', 0.5)
        mode_alias = {'full': 'full', 'full_pet': 'full', 'missing': 'fixed_missing', 'fixed_missing': 'fixed_missing', 'fixed_missing_pet': 'fixed_missing', 'random': 'random_missing', 'random_missing': 'random_missing', 'random_missing_pet': 'random_missing'}
        eval_mode = mode_alias.get(str(eval_mode), str(eval_mode))
        if eval_mode not in ('full', 'fixed_missing', 'random_missing'):
            raise ValueError(f'Unsupported eval_mode={eval_mode}.')
        rng = torch.Generator(device=self.device); rng.manual_seed(int(random_seed))
        m = SegmentationMetricsCIPA(threshold=th).to(self.device); m.reset(); total_loss, n = 0.0, 0
        pg_diag_sum, pg_diag_count = {}, {}
        for batch in loader:
            ct = batch['ct'].float().to(self.device)
            pet = batch['pet'].float().to(self.device)
            mask = batch['mask'].float().to(self.device)
            if eval_mode == 'full':
                pet_available = torch.ones(ct.shape[0], device=self.device, dtype=torch.long)
            elif eval_mode == 'fixed_missing':
                pet_available = torch.zeros(ct.shape[0], device=self.device, dtype=torch.long)
            else:
                pet_available = (torch.rand(ct.shape[0], device=self.device, generator=rng) >= float(random_pet_drop_prob)).long()
            if force_missing_pet is not None:
                pet_available = torch.zeros(ct.shape[0], device=self.device, dtype=torch.long) if force_missing_pet else torch.ones(ct.shape[0], device=self.device, dtype=torch.long)
            outputs = _forward(self.networks, ct, pet, mask.shape[-2:], pet_available=pet_available, forward_mode='auto', mask=mask)
            pred = self._select_main_pred(outputs)
            loss_seg, _ = self.loss_seg(pred, mask)
            total_loss += loss_seg.item() * ct.size(0); n += ct.size(0)
            m.update(pred, mask)
            diagnostics = outputs.get('diagnostics', {}) if isinstance(outputs, dict) else {}
            self._accumulate_pg_diagnostics(pg_diag_sum, pg_diag_count, diagnostics)
        for v in self.networks.values(): v.train()
        out = m.compute(); out['total_loss'] = total_loss / max(n, 1)
        if pg_diag_sum:
            out.update({k: pg_diag_sum[k] / max(pg_diag_count.get(k, 1), 1) for k in pg_diag_sum})
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
