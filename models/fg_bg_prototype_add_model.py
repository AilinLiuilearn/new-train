"""Aligned Add baseline with scale-supervised FG/BG decoder memories.

Full always bypasses memory. Missing never encodes PET. GT is accepted only
by prepare_candidates(), never by forward(). Auxiliary heads are shared by
Full/Missing/CT-reference and are not evaluated during ordinary inference.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline, _validate_state, _check_finite
from models.decoder_dual_prototype import gain_weights
from models.decoder_fg_bg_prototype import ForegroundBackgroundMemory


class ForegroundBackgroundAddModel(DualSharedAddPETCTBaseline):
    def __init__(self, *, prototype_enabled=True, prototype_dim=32, prototype_clusters=4,
                 prototype_capacity=256, prototype_min_candidates=24, prototype_decay=.999,
                 prototype_refresh_steps=20, prototype_warmup_steps=100,
                 prototype_gain_mode='per_scale', prototype_adaptive_sampling=True,
                 prototype_sampling_decay=.99, prototype_logit_scale=2.,
                 prototype_correction_alpha=.1, prototype_min_gain=.01,
                 prototype_max_patches=16, use_deep_supervision=True, **baseline_kwargs):
        if prototype_gain_mode not in ('per_scale', 'final'):
            raise ValueError('gain_mode must be per_scale or final')
        if prototype_gain_mode == 'per_scale' and not use_deep_supervision:
            raise ValueError('per_scale gain requires trained auxiliary prediction heads')
        if not 0 <= prototype_min_gain < 1:
            raise ValueError('min_gain must be in [0,1)')
        super().__init__(use_deep_supervision=use_deep_supervision, **baseline_kwargs)
        channels = [getattr(self.decoder, f'proj{i}')[0].out_channels for i in (4, 3, 2, 1)]
        self.prototype = ForegroundBackgroundMemory(channels, warmup_steps=prototype_warmup_steps,
            dim=prototype_dim, clusters=prototype_clusters, capacity=prototype_capacity,
            min_candidates=prototype_min_candidates, decay=prototype_decay,
            refresh_steps=prototype_refresh_steps, adaptive_sampling=prototype_adaptive_sampling,
            sampling_decay=prototype_sampling_decay, logit_scale=prototype_logit_scale,
            correction_alpha=prototype_correction_alpha, max_patches=prototype_max_patches)
        self.prototype_enabled, self.gain_mode = bool(prototype_enabled), prototype_gain_mode
        self.min_gain = float(prototype_min_gain)

    def _decode_trace(self, features, target_size, missing_index=None, diagnostics=None):
        if len(features) != 4: raise ValueError('expected four encoder levels')
        dec, trace = self.decoder, []

        def finish(x, level):
            if self.prototype_enabled and missing_index is not None and missing_index.numel():
                corrected = self.prototype.scales[level](x.index_select(0, missing_index),
                    diagnostics[level] if diagnostics is not None else None)
                x = x.index_copy(0, missing_index, corrected)
            trace.append(x)
            return x

        x1, x2, x3, x4 = features
        d4 = finish(dec.proj4(x4), 0)
        s3 = dec.proj3(x3)
        d3 = finish(dec.fuse3(torch.cat([F.interpolate(d4, size=s3.shape[-2:],
                              mode='bilinear', align_corners=False), s3], 1)), 1)
        s2 = dec.proj2(x2)
        d2 = finish(dec.fuse2(torch.cat([F.interpolate(d3, size=s2.shape[-2:],
                              mode='bilinear', align_corners=False), s2], 1)), 2)
        s1 = dec.proj1(x1)
        d1 = finish(dec.fuse1(torch.cat([F.interpolate(d2, size=s1.shape[-2:],
                              mode='bilinear', align_corners=False), s1], 1)), 3)
        logits = F.interpolate(dec.seg_head(d1), size=target_size, mode='bilinear', align_corners=False)
        _check_finite('fg_bg_decoder_logits', logits)
        return logits, trace

    def _scale_predictions(self, trace, final, target_size):
        if not self.use_deep_supervision:
            return [final] * 4
        # Heads consume live traces before candidate tensors are detached.
        logits = [F.interpolate(getattr(self.decoder, f'aux_head_d{4-i}')(trace[i]),
                    size=target_size, mode='bilinear', align_corners=False) for i in range(3)] + [final]
        _check_finite('scale_logits', logits)
        return logits

    def forward(self, ct, pet=None, pet_available=None, target_size=None, forward_mode='auto',
                return_knowledge=False, return_scale_logits=False, return_diagnostics=False):
        if (return_knowledge or return_scale_logits) and not self.training:
            raise RuntimeError('auxiliary supervision and candidate collection are training-only')
        batch = ct.shape[0]
        if forward_mode in ('full', 'missing'):
            state = torch.full((batch,), int(forward_mode == 'full'), device=ct.device, dtype=torch.long)
        elif forward_mode == 'auto':
            state = _validate_state(torch.ones(batch, dtype=torch.long) if pet_available is None
                                    else pet_available, batch).to(ct.device)
        else: raise ValueError(f'unsupported mode: {forward_mode}')
        full = state.eq(1).nonzero(as_tuple=True)[0]
        missing = state.eq(0).nonzero(as_tuple=True)[0]
        if full.numel() and pet is None: raise ValueError('Full rows require PET')
        target_size = ct.shape[-2:] if target_size is None else target_size
        ct_feats = self._encode_ct(ct)
        fused = list(ct_feats)
        if full.numel():
            ct_full = [x.index_select(0, full) for x in ct_feats]
            pet_full = self._encode_pet(pet.index_select(0, full).contiguous())
            full_fused = self.fusion(ct_full, pet_full)
            fused = [c.index_copy(0, full, f.to(c.dtype)) for c, f in zip(ct_feats, full_fused)]
        diagnostics = [{} for _ in range(4)] if return_diagnostics else None
        logits, trace = self._decode_trace(fused, target_size, missing, diagnostics)
        out = dict(logits=logits, pred=logits, aux={}, pet_available=state.detach().cpu(),
                   num_full=int(full.numel()), num_missing=int(missing.numel()))
        if diagnostics is not None: out['prototype_reads'] = diagnostics
        need_aux = return_scale_logits or (return_knowledge and self.gain_mode == 'per_scale')
        predictions = self._scale_predictions(trace, logits, target_size) if need_aux else [logits] * 4
        if return_scale_logits: out['scale_logits'] = predictions
        if return_knowledge and full.numel():
            reference, ref_trace = self._decode_trace(ct_full, target_size)
            ref_predictions = self._scale_predictions(ref_trace, reference, target_size) if need_aux else [reference] * 4
            out['knowledge'] = dict(full_index=full,
                full_features=[x.index_select(0, full).detach() for x in trace],
                reference_features=[x.detach() for x in ref_trace],
                full_logits_by_scale=[x.index_select(0, full).detach() for x in predictions],
                reference_logits_by_scale=ref_predictions)
        return out

    @torch.no_grad()
    def prepare_candidates(self, knowledge, mask, return_diagnostics=False):
        if not self.training: raise RuntimeError('candidate extraction is training-only')
        target = mask.index_select(0, knowledge['full_index'])
        if target.ndim == 3: target = target[:, None]
        values, diagnostics = [], {}
        for i, scale in enumerate(self.prototype.scales):
            stats = {} if return_diagnostics else None
            index = i if self.gain_mode == 'per_scale' else 3
            weight = gain_weights(knowledge['full_logits_by_scale'][index],
                knowledge['reference_logits_by_scale'][index], target,
                foreground_weight=1., background_weight=1., min_gain=self.min_gain, stats=stats)
            values.append(scale.candidates(knowledge['full_features'][i],
                knowledge['reference_features'][i], weight, target, stats=stats))
            if stats is not None:
                diagnostics.update({f'd{4-i}_{key}': value for key, value in stats.items()})
        return (values, diagnostics) if return_diagnostics else values


def build_fg_bg_model(cfg):
    baseline = dict(ct_backbone=cfg.ct_backbone, pet_backbone=cfg.pet_backbone,
        ct_pretrained_path=cfg.ct_pretrained_path, pet_pretrained_path=cfg.pet_pretrained_path,
        decoder_channels=tuple(cfg.decoder_channels), pretrained=cfg.pretrained,
        use_deep_supervision=bool(cfg.use_deep_supervision or cfg.deep_supervision))
    names = ('enabled', 'dim', 'clusters', 'capacity', 'min_candidates', 'decay',
             'refresh_steps', 'warmup_steps', 'gain_mode', 'adaptive_sampling', 'sampling_decay',
             'logit_scale', 'correction_alpha', 'min_gain', 'max_patches')
    baseline.update({f'prototype_{name}': getattr(cfg, f'prototype_{name}') for name in names})
    model = ForegroundBackgroundAddModel(**baseline)
    print(f'[fg_bg_prototype] fusion=AddFusion gain_mode={model.gain_mode} '
          f'deep_supervision={model.use_deep_supervision} banks=foreground,background '
          f'K={cfg.prototype_clusters} adaptive_sampling={cfg.prototype_adaptive_sampling} '
          f'logit_scale={cfg.prototype_logit_scale} alpha={cfg.prototype_correction_alpha} '
          f'knowledge_enabled={model.prototype_enabled}', flush=True)
    return {'model': model}
