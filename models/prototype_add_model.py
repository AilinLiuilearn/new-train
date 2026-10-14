"""Pure Add baseline + Missing-only correction at decoder D4,D3,D2,D1.

The original baseline/decoder are imported unchanged. ``return_knowledge``
is explicit, training-only and never supplied by validation/inference. It
adds a supervised CT counterfactual decode of Full rows only; no extra
encoder call, PET imputation, custom fusion or target-dependent forward.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline, _validate_state, _check_finite
from models.decoder_dual_prototype import DecoderDualPrototype, gain_weights


class PrototypeAddModel(DualSharedAddPETCTBaseline):
    def __init__(self, *, prototype_enabled=True, prototype_dim=32,
                 prototype_clusters=6, prototype_capacity=256,
                 prototype_min_candidates=24, prototype_decay=.999,
                 prototype_refresh_steps=20, prototype_warmup_steps=100,
                 prototype_spatial=True, prototype_channel=True,
                 prototype_foreground_weight=2., prototype_background_weight=1.,
                 prototype_min_gain=.01, prototype_outlier_filter=False, **baseline_kwargs):
        if baseline_kwargs.get('use_deep_supervision', False):
            raise ValueError('prototype experiment preserves baseline deep_supervision=False')
        super().__init__(**baseline_kwargs)
        if prototype_foreground_weight <= 0 or prototype_background_weight <= 0:
            raise ValueError('benefit weights must be positive')
        if not 0 <= prototype_min_gain < 1:
            raise ValueError('prototype_min_gain must be in [0,1)')
        channels = [self.decoder.proj4[0].out_channels, self.decoder.proj3[0].out_channels,
                    self.decoder.proj2[0].out_channels, self.decoder.proj1[0].out_channels]
        self.prototype = DecoderDualPrototype(
            channels, prototype_dim, prototype_clusters, prototype_capacity,
            prototype_min_candidates, prototype_decay, prototype_refresh_steps,
            prototype_warmup_steps, prototype_spatial, prototype_channel,
            outlier_filter=prototype_outlier_filter)
        self.prototype_enabled = bool(prototype_enabled)
        self.foreground_weight = float(prototype_foreground_weight)
        self.background_weight = float(prototype_background_weight)
        self.min_gain = float(prototype_min_gain)

    def _decode_trace(self, features, target_size, missing_index=None):
        """Same ops as UNetStyleDecoder; extra seam follows each decode block.

        features and trace order differ deliberately: encoder [S1..S4],
        decoder trace [D4..D1] with channels [512,256,128,64] by default.
        """
        if len(features) != 4:
            raise ValueError('decoder expects four encoder scales')
        dec = self.decoder
        trace = []

        def finish(x, level):
            if self.prototype_enabled and missing_index is not None and missing_index.numel():
                rows = x.index_select(0, missing_index)
                corrected = self.prototype.scales[level](rows)
                x = x.index_copy(0, missing_index, corrected)
            trace.append(x)
            return x

        x1, x2, x3, x4 = features
        d4 = finish(dec.proj4(x4), 0)
        s3 = dec.proj3(x3)
        d3 = finish(dec.fuse3(torch.cat([F.interpolate(
            d4, size=s3.shape[-2:], mode='bilinear', align_corners=False), s3], 1)), 1)
        s2 = dec.proj2(x2)
        d2 = finish(dec.fuse2(torch.cat([F.interpolate(
            d3, size=s2.shape[-2:], mode='bilinear', align_corners=False), s2], 1)), 2)
        s1 = dec.proj1(x1)
        d1 = finish(dec.fuse1(torch.cat([F.interpolate(
            d2, size=s1.shape[-2:], mode='bilinear', align_corners=False), s1], 1)), 3)
        logits = F.interpolate(dec.seg_head(d1), size=target_size,
                               mode='bilinear', align_corners=False)
        _check_finite('prototype_decoder_logits', logits)
        return logits, trace

    def forward(self, ct, pet=None, pet_available=None, target_size=None,
                forward_mode='auto', return_knowledge=False):
        if return_knowledge and not self.training:
            raise RuntimeError('knowledge collection is training-only')
        batch = ct.shape[0]
        if forward_mode == 'full':
            state = torch.ones(batch, dtype=torch.long, device=ct.device)
        elif forward_mode == 'missing':
            state = torch.zeros(batch, dtype=torch.long, device=ct.device)
        elif forward_mode == 'auto':
            state = _validate_state(torch.ones(batch, dtype=torch.long)
                                    if pet_available is None else pet_available, batch).to(ct.device)
        else:
            raise ValueError(f'Unsupported forward_mode={forward_mode!r}')
        full_idx = state.eq(1).nonzero(as_tuple=True)[0]
        missing_idx = state.eq(0).nonzero(as_tuple=True)[0]
        if full_idx.numel() and pet is None:
            raise ValueError('Full rows require PET')
        if target_size is None:
            target_size = ct.shape[-2:]
        # Exactly one CT encoder call on the entire mixed batch; preserve
        # original CT-align BatchNorm statistics and exact-half protocol.
        ct_feats = self._encode_ct(ct)
        fused = list(ct_feats)
        if full_idx.numel():
            pet_full = self._encode_pet(pet.index_select(0, full_idx).contiguous())
            ct_full = [x.index_select(0, full_idx) for x in ct_feats]
            full_fused = self.fusion(ct_full, pet_full)  # strictly AddFusion
            fused = [c.index_copy(0, full_idx, f.to(c.dtype)) for c, f in zip(ct_feats, full_fused)]
        logits, trace = self._decode_trace(fused, target_size, missing_idx)
        out = dict(logits=logits, pred=logits, aux={}, pet_available=state.detach().cpu(),
                   num_full=int(full_idx.numel()), num_missing=int(missing_idx.numel()))
        if return_knowledge and full_idx.numel():
            reference_logits, reference_trace = self._decode_trace(ct_full, target_size)
            # Trace collection itself carries no gradient. Reference logits
            # retain their graph ONLY for the explicit CT auxiliary loss.
            out['knowledge'] = dict(
                full_index=full_idx,
                full_features=[x.index_select(0, full_idx).detach() for x in trace],
                reference_features=[x.detach() for x in reference_trace],
                full_logits=logits.index_select(0, full_idx).detach(),
                reference_logits=reference_logits)
        return out

    @torch.no_grad()
    def prepare_candidates(self, knowledge, mask):
        if not self.training:
            raise RuntimeError('candidate extraction is training-only')
        target = mask.index_select(0, knowledge['full_index'])
        if target.ndim == 3:
            target = target[:, None]
        weight = gain_weights(knowledge['full_logits'], knowledge['reference_logits'], target,
                              self.foreground_weight, self.background_weight, self.min_gain)
        return [scale.candidates(full, reference, weight, target)
                for scale, full, reference in zip(self.prototype.scales,
                                                 knowledge['full_features'],
                                                 knowledge['reference_features'])]


def build_prototype_model(cfg):
    baseline = dict(
        ct_backbone=cfg.ct_backbone, pet_backbone=cfg.pet_backbone,
        ct_pretrained_path=cfg.ct_pretrained_path, pet_pretrained_path=cfg.pet_pretrained_path,
        decoder_channels=tuple(cfg.decoder_channels), pretrained=cfg.pretrained,
        use_deep_supervision=bool(cfg.use_deep_supervision or cfg.deep_supervision))
    names = ('enabled', 'dim', 'clusters', 'capacity', 'min_candidates', 'decay',
             'refresh_steps', 'warmup_steps', 'spatial', 'channel',
             'foreground_weight', 'background_weight', 'min_gain')
    baseline.update({f'prototype_{name}': getattr(cfg, f'prototype_{name}') for name in names})
    # New CLI configs explicitly default False. Old saved configs lack this
    # field and used trimming, so preserve their evaluation/resume semantics.
    baseline['prototype_outlier_filter'] = getattr(cfg, 'prototype_outlier_filter', True)
    model = PrototypeAddModel(**baseline)
    extra = sum(p.numel() for p in model.prototype.parameters())
    print(f'[prototype_add] fusion=AddFusion mixed_ratio=0.5 '
          f'decoder_levels=D4,D3,D2,D1 extra_parameters={extra} '
          f'knowledge_enabled={model.prototype_enabled} '
          f'outlier_filter={model.prototype.scales[0].spatial_bank.outlier_filter}', flush=True)
    return {'model': model}
