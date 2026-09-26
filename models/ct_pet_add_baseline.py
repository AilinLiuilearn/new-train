# -*- coding: utf-8 -*-
"""Clean CT+PET per-scale addition baseline.

CT and PET go through the same heterogeneous encoders as the main model
(ConvNeXtV2-nano for CT, MiT-B1 for PET), CT is channel-aligned to the PET
pyramid, and the two pyramids are added scale by scale before the shared
decoder. No text, no gates, no residuals beyond the plain sum.

Missing contract (same as the main line): real PET is still encoded, then
zeroed (encode-then-zero), so Missing rows get exactly the CT features and
the output equals the CT-only decode.

The module exposes ``enc_ct`` / ``ct_align`` / ``decoder`` so the training
task (optimizer, EMA, gradient logging, diagnostics) works unchanged.
"""
import torch
from torch import nn

from models.baseline_blocks import UNetStyleDecoder
from models.build_mdt_seg import create_feature_backbone, load_local_weights_safe
from models.dual_shared_add_baseline import StageChannelAlign


class CTPETAddBaseline(nn.Module):
    def __init__(self, ct_backbone='convnextv2_nano', pet_backbone='mit_b1',
                 ct_pretrained_path=None, pet_pretrained_path=None,
                 in_channels=3, out_channels=1,
                 decoder_channels=(512, 256, 128, 64),
                 use_deep_supervision=False, decoder_norm='bn'):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)
        if decoder_norm not in ('bn', 'group'):
            raise ValueError(f'Unsupported decoder_norm={decoder_norm!r}')
        self.decoder_norm = decoder_norm
        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        self.enc_pet = create_feature_backbone(pet_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='CT_Encoder')
        load_local_weights_safe(self.enc_pet, pet_pretrained_path, name='PET_Encoder')
        ct_channels = list(self.enc_ct.feature_info.channels())
        pet_channels = list(self.enc_pet.feature_info.channels())
        self.ct_align = StageChannelAlign(ct_channels, pet_channels)
        self.decoder = UNetStyleDecoder(
            pet_channels, decoder_channels=decoder_channels,
            out_channels=out_channels,
            use_deep_supervision=self.use_deep_supervision,
            norm_type=decoder_norm)

    @staticmethod
    def _to_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _encode_ct(self, ct):
        return self.ct_align(self.enc_ct(self._to_3ch(ct)))

    def _encode_pet(self, pet):
        if pet is None:
            raise ValueError('Baseline requires PET input before fusion-time masking')
        return self.enc_pet(self._to_3ch(pet))

    @staticmethod
    def _add_fused(ct_feats, pet_feats):
        fused = []
        for c, p in zip(ct_feats, pet_feats):
            if tuple(c.shape) != tuple(p.shape):
                raise ValueError('CT/PET feature shapes must match per scale for addition')
            if p.dtype != c.dtype:
                p = p.to(dtype=c.dtype)
            fused.append(c + p)
        return fused

    def _decode(self, fused_feats, target_size):
        out = self.decoder(fused_feats, target_size)
        out['pred'] = out['logits']
        out['aux'] = {}
        return out

    def _forward_full(self, ct, pet, target_size):
        fused = self._add_fused(self._encode_ct(ct), self._encode_pet(pet))
        return self._decode(fused, target_size)

    def _forward_missing(self, ct, pet, target_size):
        ct_feats = self._encode_ct(ct)
        pet_feats = [torch.zeros_like(f) for f in self._encode_pet(pet)]
        return self._decode(self._add_fused(ct_feats, pet_feats), target_size)

    def _forward_auto(self, ct, pet, pet_available, target_size):
        ct_feats = self._encode_ct(ct)
        pet_feats_real = self._encode_pet(pet)
        pet_available = torch.as_tensor(pet_available, device=ct.device).long().view(-1)
        if pet_available.numel() != ct.shape[0]:
            raise ValueError('pet_available must contain one state per sample')
        if not torch.all((pet_available == 0) | (pet_available == 1)):
            raise ValueError('pet_available values must be 0 or 1')
        mask = pet_available.to(dtype=pet_feats_real[0].dtype).view(-1, 1, 1, 1)
        pet_feats = [f * mask.to(device=f.device) for f in pet_feats_real]
        out = self._decode(self._add_fused(ct_feats, pet_feats), target_size)
        out['pet_available'] = pet_available.detach().cpu()
        out['num_full'] = int(pet_available.eq(1).sum())
        out['num_missing'] = int(pet_available.eq(0).sum())
        return out

    def forward(self, ct, pet, pet_available=None, target_size=None, forward_mode='auto'):
        if target_size is None:
            target_size = ct.shape[-2:]
        if forward_mode == 'full':
            return self._forward_full(ct, pet, target_size)
        if forward_mode == 'missing':
            return self._forward_missing(ct, pet, target_size)
        if forward_mode == 'auto':
            if pet_available is None:
                pet_available = torch.ones(ct.shape[0], device=ct.device, dtype=torch.long)
            return self._forward_auto(ct, pet, pet_available, target_size)
        raise ValueError(f'Unsupported forward_mode={forward_mode!r}')
