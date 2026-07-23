# -*- coding: utf-8 -*-
import json
import os
import random
import subprocess
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.seg_mdt import SegMDTConfig
from models.apsf_module import APSF
from models.baseline_blocks import UNetStyleDecoder, _check_tensor, _check_tensor_list
from models.build_mdt_seg import build_mdt_seg_teacher, create_feature_backbone, load_local_weights_safe
from models.mppc import MPPC
from tasks.mdt_seg import MDTSegTeacher
from utils.optimization import get_cosine_scheduler
from utils.train_logger import append_epoch_log, init_train_log


class StageChannelAlign(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ) for c_in, c_out in zip(in_channels_list, out_channels_list)
        ])

    def forward(self, feats):
        return [proj(feat) for proj, feat in zip(self.proj, feats)]


class DualSharedAddPETCTBaseline(nn.Module):
    def __init__(self, ct_backbone='convnextv2_nano', pet_backbone='mit_b1', ct_pretrained_path=None, pet_pretrained_path=None, in_channels=3, out_channels=1, decoder_channels=(512, 256, 128, 64), use_deep_supervision=False):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)
        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        self.enc_pet = create_feature_backbone(pet_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='CT_Encoder')
        load_local_weights_safe(self.enc_pet, pet_pretrained_path, name='PET_Encoder')
        ct_channels = list(self.enc_ct.feature_info.channels())
        pet_channels = list(self.enc_pet.feature_info.channels())
        self.ct_align = StageChannelAlign(ct_channels, pet_channels)
        self.mppc = MPPC(channels=pet_channels, num_slots=3, momentum=0.9, temperature=0.1, gate_init_logit=-6.0)
        self.apsf = APSF(channels=pet_channels)
        self.decoder = UNetStyleDecoder(pet_channels, decoder_channels=decoder_channels, out_channels=out_channels, use_deep_supervision=self.use_deep_supervision)

    @staticmethod
    def _to_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _encode_ct(self, ct):
        ct_feats = self.enc_ct(self._to_3ch(ct))
        _check_tensor_list('ct_feats', ct_feats)
        return self.ct_align(ct_feats)

    def _encode_pet(self, pet):
        pet_feats = self.enc_pet(self._to_3ch(pet))
        _check_tensor_list('pet_feats', pet_feats)
        return pet_feats

    def _align_pet_to_ct(self, ct_feats, pet_feats):
        aligned = []
        for ct_feat, pet_feat in zip(ct_feats, pet_feats):
            if pet_feat.shape[-2:] != ct_feat.shape[-2:]:
                pet_feat = F.interpolate(pet_feat, size=ct_feat.shape[-2:], mode='bilinear', align_corners=False)
            aligned.append(pet_feat)
        return aligned

    def _fuse_features(self, ct_feats, aux_feats, forward_mode):
        if forward_mode == 'full':
            fused_feats = self.apsf.forward_full(ct_feats, aux_feats)
        elif forward_mode == 'missing':
            fused_feats = self.apsf.forward_missing(ct_feats, aux_feats)
        else:
            raise ValueError(f'Unsupported forward_mode={forward_mode!r}')
        _check_tensor_list('apsf_fused_output', fused_feats)
        return fused_feats

    def _decode(self, fused_feats, target_size):
        out = self.decoder(fused_feats, target_size)
        _check_tensor('decoder_logits', out['logits'])
        out['pred'] = out['logits']
        out['aux'] = {}
        return out

    def _forward_full(self, ct, pet, target_size, mask=None):
        if pet is None:
            raise ValueError('Full forward requires pet')
        aligned_ct = self._encode_ct(ct)
        pet_feats = self._align_pet_to_ct(aligned_ct, self._encode_pet(pet))
        with torch.autocast(device_type=aligned_ct[0].device.type, enabled=False):
            pet_for_fusion = self.mppc(aligned_ct, pet_features=pet_feats, target=mask, mode='full', update_bank=self.training)
        fused_feats = self._fuse_features(aligned_ct, pet_for_fusion, 'full')
        return self._decode(fused_feats, target_size)

    def _forward_missing(self, ct, target_size):
        aligned_ct = self._encode_ct(ct)
        pet_compensation = self.mppc(aligned_ct, pet_features=None, target=None, mode='missing')
        fused_feats = self._fuse_features(aligned_ct, pet_compensation, 'missing')
        return self._decode(fused_feats, target_size)

    def _forward_auto(self, ct, pet, pet_available, target_size):
        pet_available = pet_available.long().view(-1)
        if pet_available.numel() != ct.shape[0]:
            raise ValueError(f'pet_available length must match batch size, got {pet_available.numel()} vs {ct.shape[0]}')
        uniq = set(int(v) for v in pet_available.unique().tolist())
        if not uniq.issubset({0, 1}):
            raise ValueError(f'pet_available must contain only 0/1 values, got {sorted(uniq)}')
        if torch.all(pet_available == 1):
            return self._forward_full(ct, pet, target_size, mask=None)
        if torch.all(pet_available == 0):
            return self._forward_missing(ct, target_size)
        aligned_ct = self._encode_ct(ct)
        fused = [feat.clone() for feat in aligned_ct]
        full_idx = torch.nonzero(pet_available == 1, as_tuple=False).flatten()
        missing_idx = torch.nonzero(pet_available == 0, as_tuple=False).flatten()
        if full_idx.numel() > 0 and pet is None:
            raise ValueError('full_idx is non-empty but pet is None')
        if full_idx.numel() > 0:
            ct_full = [feat.index_select(0, full_idx) for feat in aligned_ct]
            pet_full = self._align_pet_to_ct(ct_full, self._encode_pet(pet.index_select(0, full_idx)))
            pet_for_fusion = self.mppc(ct_full, pet_features=pet_full, target=None, mode='full', update_bank=False)
            fused_full = self._fuse_features(ct_full, pet_for_fusion, 'full')
            for i, feat in enumerate(fused_full):
                fused[i].index_copy_(0, full_idx, feat)
        if missing_idx.numel() > 0:
            ct_missing = [feat.index_select(0, missing_idx) for feat in aligned_ct]
            pet_compensation = self.mppc(ct_missing, pet_features=None, target=None, mode='missing')
            fused_missing = self._fuse_features(ct_missing, pet_compensation, 'missing')
            for i, feat in enumerate(fused_missing):
                fused[i].index_copy_(0, missing_idx, feat)
        return self._decode(fused, target_size)

    def forward(self, ct, pet=None, pet_available=None, target_size=None, forward_mode='auto', mask=None):
        if target_size is None:
            target_size = ct.shape[-2:]
        if forward_mode == 'full':
            return self._forward_full(ct, pet, target_size, mask=mask)
        if forward_mode == 'missing':
            return self._forward_missing(ct, target_size)
        if forward_mode == 'auto':
            if pet_available is None:
                pet_available = torch.ones(ct.shape[0], device=ct.device, dtype=torch.long)
            return self._forward_auto(ct, pet, pet_available, target_size)
        raise ValueError(f'Unsupported forward_mode={forward_mode!r}')
