# -*- coding: utf-8 -*-
"""Dual-modal Full/Missing baseline with pure per-scale addition.

- full:    aligned_CT + PET_features -> shared decoder (PET required).
- missing: aligned_CT + zeros_like(aligned_CT) -> same decoder.
           Missing rows' PET is never encoded, so Missing outputs cannot
           depend on Missing rows' PET content. ``pet=None`` is allowed.
- auto:    routes each sample by a length-B bool / 0-1-int state, encodes PET
           for Full rows only, then restores the original sample order.

Only the Full path goes through :class:`AddFusion` (the future fusion seam).
"""
import torch
import torch.nn as nn

from models.components.add_fusion import AddFusion
from models.components.backbones import create_feature_backbone, load_local_weights_safe
from models.components.channel_align import StageChannelAlign
from models.components.group_unet_decoder import UNetStyleDecoder


DECODER_INPUT_CHANNELS = (64, 128, 320, 512)


def _check_finite(name, xs):
    if isinstance(xs, (list, tuple)):
        for i, x in enumerate(xs):
            if not torch.isfinite(x).all():
                raise RuntimeError(f'[NaN/Inf] {name}[{i}] contains invalid values')
    elif not torch.isfinite(xs).all():
        raise RuntimeError(f'[NaN/Inf] {name} contains invalid values')


def _validate_state(pet_available, batch, name='pet_available'):
    """Validate the raw state without silent truncation: bool or 0/1 ints."""
    raw = torch.as_tensor(pet_available)
    if raw.numel() != batch:
        raise ValueError(f'{name} must contain one state per sample: '
                         f'got {raw.numel()} for batch {batch}')
    if raw.dtype == torch.bool:
        return raw.long().view(-1)
    if raw.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        if not torch.all((raw == 0) | (raw == 1)):
            raise ValueError(f'{name} values must be 0 or 1')
        return raw.long().view(-1)
    raise ValueError(f'{name} must be bools or 0/1 integers, no silent float truncation')


class DualSharedAddPETCTBaseline(nn.Module):
    def __init__(self, ct_backbone='convnextv2_nano', pet_backbone='mit_b1',
                 ct_pretrained_path=None, pet_pretrained_path=None,
                 in_channels=3, out_channels=1,
                 decoder_channels=(512, 256, 128, 64),
                 use_deep_supervision=False, pretrained=True,
                 check_finite=True):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)
        # Intermediate NaN/Inf checks (encoder/align/fused tensors) can be
        # disabled for speed; the logits check in _decode always stays on.
        # Default True preserves the original behavior exactly.
        self.check_finite = bool(check_finite)
        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        self.enc_pet = create_feature_backbone(pet_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path,
                                name='CT_Encoder', required=bool(pretrained))
        load_local_weights_safe(self.enc_pet, pet_pretrained_path,
                                name='PET_Encoder', required=bool(pretrained))
        ct_channels = list(self.enc_ct.feature_info.channels())
        self.ct_align = StageChannelAlign(ct_channels, list(DECODER_INPUT_CHANNELS))
        self.fusion = AddFusion()
        self.decoder = UNetStyleDecoder(
            list(DECODER_INPUT_CHANNELS), decoder_channels=decoder_channels,
            out_channels=out_channels, use_deep_supervision=self.use_deep_supervision)

    def _maybe_check_finite(self, name, xs):
        if self.check_finite:
            _check_finite(name, xs)

    @staticmethod
    def _to_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _encode_ct(self, ct):
        feats = self.enc_ct(self._to_3ch(ct))
        self._maybe_check_finite('ct_feats', feats)
        aligned = self.ct_align(feats)
        self._maybe_check_finite('aligned_ct', aligned)
        return aligned

    def _encode_pet(self, pet):
        feats = self.enc_pet(self._to_3ch(pet))
        self._maybe_check_finite('pet_feats', feats)
        return feats

    def _decode(self, fused_feats, target_size):
        out = self.decoder(fused_feats, target_size)
        _check_finite('logits', out['logits'])
        out['pred'] = out['logits']
        out['aux'] = {}
        return out

    def _forward_full(self, ct, pet, target_size):
        if pet is None:
            raise ValueError('Full path requires a valid PET tensor, got pet=None')
        ct_feats = self._encode_ct(ct)
        pet_feats = self._encode_pet(pet)
        return self._decode(self.fusion(ct_feats, pet_feats), target_size)

    def _forward_missing(self, ct, pet, target_size):
        # Missing never encodes PET: aligned_CT goes to the decoder with a
        # zero increment, so the result cannot depend on any PET content.
        _ = pet
        ct_feats = self._encode_ct(ct)
        fused = [c + torch.zeros_like(c) for c in ct_feats]
        self._maybe_check_finite('fused_feats', fused)
        return self._decode(fused, target_size)

    def _forward_auto(self, ct, pet, pet_available, target_size):
        batch = ct.shape[0]
        state = _validate_state(pet_available, batch).to(ct.device)
        full_idx = state.eq(1).nonzero(as_tuple=True)[0]
        num_full = int(full_idx.numel())
        ct_feats = self._encode_ct(ct)
        if num_full > 0:
            if pet is None:
                raise ValueError(f'auto path has {num_full} Full rows but pet=None')
            pet_full = self._encode_pet(pet.index_select(0, full_idx).contiguous())
            ct_full = [c.index_select(0, full_idx) for c in ct_feats]
            fused_full = self.fusion(ct_full, pet_full)
        # Missing rows: aligned_CT, zero increment. Full rows replaced below.
        fused = [c + torch.zeros_like(c) for c in ct_feats]
        if num_full > 0:
            fused = [base.index_copy(0, full_idx, f.to(dtype=base.dtype))
                     for base, f in zip(fused, fused_full)]
        self._maybe_check_finite('fused_feats', fused)
        out = self._decode(fused, target_size)
        out['pet_available'] = state.detach().cpu()
        out['num_full'] = num_full
        out['num_missing'] = batch - num_full
        return out

    def forward(self, ct, pet=None, pet_available=None, target_size=None,
                forward_mode='auto'):
        if target_size is None:
            target_size = ct.shape[-2:]
        if forward_mode == 'full':
            return self._forward_full(ct, pet, target_size)
        if forward_mode == 'missing':
            return self._forward_missing(ct, pet, target_size)
        if forward_mode == 'auto':
            if pet_available is None:
                pet_available = torch.ones(ct.shape[0], dtype=torch.long)
            return self._forward_auto(ct, pet, pet_available, target_size)
        raise ValueError(f'Unsupported forward_mode={forward_mode!r}')