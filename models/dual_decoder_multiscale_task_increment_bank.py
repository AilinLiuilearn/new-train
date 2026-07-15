# -*- coding: utf-8 -*-
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from models.baseline_petct_unet import UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import ConvBNAct, create_feature_backbone, load_local_weights_safe
from models.multiscale_task_increment_bank import SharedMultiScaleTaskIncrementBank
from models.pg_mtr import _valid_group_count


class StageChannelAlign(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        super().__init__()
        self.proj = nn.ModuleList([ConvBNAct(c_in, c_out, kernel_size=1) for c_in, c_out in zip(in_channels_list, out_channels_list)])

    def forward(self, feats):
        return [proj(feat) for proj, feat in zip(self.proj, feats)]


class ZeroInitResidualTaskRefine(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(_valid_group_count(channels), channels, affine=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(self.norm(x))))


class DualDecoderMultiScaleTaskIncrementBank(nn.Module):
    def __init__(self, ct_backbone='convnextv2_nano', pet_backbone='mit_b1', ct_pretrained_path=None, pet_pretrained_path=None, in_channels=3, out_channels=1, decoder_channels=(512, 256, 128, 64), use_deep_supervision=False, mtib_stages='all', mtib_num_tokens=8, mtib_temperature=0.07):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)
        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        self.enc_pet = create_feature_backbone(pet_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='CT_Encoder')
        load_local_weights_safe(self.enc_pet, pet_pretrained_path, name='PET_Encoder')
        ct_channels = list(self.enc_ct.feature_info.channels())
        pet_channels = list(self.enc_pet.feature_info.channels())
        self.ct_align = StageChannelAlign(ct_channels, pet_channels)
        self.full_decoder = UNetStyleDecoder(pet_channels, decoder_channels=decoder_channels, out_channels=out_channels, use_deep_supervision=self.use_deep_supervision)
        self.missing_decoder = copy.deepcopy(self.full_decoder)
        self.task_refine = nn.ModuleDict({str(s): ZeroInitResidualTaskRefine(pet_channels[s - 1]) for s in (1, 2, 3, 4)})
        self.mtib = SharedMultiScaleTaskIncrementBank(pet_channels, num_tokens=mtib_num_tokens, temperature=mtib_temperature, stage_mode=mtib_stages)

    @staticmethod
    def _to_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _encode_ct(self, ct):
        ct = self._to_3ch(ct)
        feats = self.enc_ct(ct)
        _check_tensor_list('ct_feats', feats)
        return self.ct_align(feats)

    def _encode_pet(self, pet):
        pet = self._to_3ch(pet)
        feats = self.enc_pet(pet)
        _check_tensor_list('pet_feats', feats)
        return feats

    def _finalize_decoder_output(self, dec_out):
        if isinstance(dec_out, dict):
            out = {'logits': _sanitize(dec_out['logits'])}
            if 'aux_logits' in dec_out:
                out['aux_logits'] = [_sanitize(x) for x in dec_out['aux_logits']]
            return out
        return {'logits': _sanitize(dec_out)}

    def _decode_with(self, decoder, features, target_size):
        out = self._finalize_decoder_output(decoder(features, target_size))
        _check_tensor('logits', out['logits'])
        out['pred'] = out['logits']
        out['aux'] = {}
        return out

    def _true_increment(self, ct_feats, pet_feats):
        joint_feats = []
        full_feats = []
        increments = {}
        for s in (1, 2, 3, 4):
            i = s - 1
            c = ct_feats[i]
            p = pet_feats[i]
            j = c + p
            r = self.task_refine[str(s)](j)
            z = j + r
            d_star = z - c
            joint_feats.append(j)
            full_feats.append(z)
            increments[s] = d_star
        return joint_feats, full_feats, increments

    def _forward_full(self, ct, pet, target_size):
        ct_feats = self._encode_ct(ct)
        pet_feats = self._encode_pet(pet)
        joint_feats, full_feats, true_inc = self._true_increment(ct_feats, pet_feats)
        out = self._decode_with(self.full_decoder, full_feats, target_size)
        bank_out, bank_loss, bank_diag = self.mtib.forward_full(joint_feats, true_inc)
        comp_out, comp_loss, comp_diag = self.mtib.forward_ct_comp(ct_feats, true_inc)
        out['aux_losses'] = {'mtib_bank_loss': bank_loss, 'mtib_comp_loss': comp_loss}
        out['diagnostics'] = {**bank_diag, **comp_diag}
        out['mtib_bank'] = bank_out
        out['mtib_comp'] = comp_out
        return out

    def _forward_missing(self, ct, target_size):
        ct_feats = self._encode_ct(ct)
        missing_inc, diag = self.mtib.forward_missing(ct_feats)
        missing_feats = [ct_feats[i] + missing_inc[i + 1] for i in range(4)]
        out = self._decode_with(self.missing_decoder, missing_feats, target_size)
        out['aux_losses'] = {}
        out['diagnostics'] = diag
        return out

    def _forward_auto(self, ct, pet, pet_available, target_size):
        pa = pet_available.long().view(-1)
        if torch.all(pa == 1):
            return self._forward_full(ct, pet, target_size)
        if torch.all(pa == 0):
            return self._forward_missing(ct, target_size)
        ct_feats = self._encode_ct(ct)
        full_idx = torch.nonzero(pa == 1, as_tuple=False).flatten()
        missing_idx = torch.nonzero(pa == 0, as_tuple=False).flatten()
        full_out = None
        missing_out = None
        if full_idx.numel() > 0:
            full_ct = [f.index_select(0, full_idx) for f in ct_feats]
            full_pet = self._encode_pet(pet.index_select(0, full_idx))
            joint_feats, full_feats, true_inc = self._true_increment(full_ct, full_pet)
            full_out = self._decode_with(self.full_decoder, full_feats, target_size)
            bank_out, bank_loss, bank_diag = self.mtib.forward_full(joint_feats, true_inc)
            comp_out, comp_loss, comp_diag = self.mtib.forward_ct_comp(full_ct, true_inc)
            full_out['aux_losses'] = {'mtib_bank_loss': bank_loss, 'mtib_comp_loss': comp_loss}
            full_out['diagnostics'] = {**bank_diag, **comp_diag}
            full_out['mtib_bank'] = bank_out
            full_out['mtib_comp'] = comp_out
        if missing_idx.numel() > 0:
            miss_ct = [f.index_select(0, missing_idx) for f in ct_feats]
            miss_inc, miss_diag = self.mtib.forward_missing(miss_ct)
            miss_feats = [miss_ct[i] + miss_inc[i + 1] for i in range(4)]
            missing_out = self._decode_with(self.missing_decoder, miss_feats, target_size)
            missing_out['diagnostics'] = miss_diag
            missing_out['aux_losses'] = {}
        if full_out is None:
            return missing_out
        if missing_out is None:
            return full_out
        merged = {}
        logits = full_out['logits'].new_zeros((ct.shape[0],) + tuple(full_out['logits'].shape[1:]))
        if full_idx.numel() > 0:
            logits = logits.index_copy(0, full_idx, full_out['logits'])
        if missing_idx.numel() > 0:
            logits = logits.index_copy(0, missing_idx, missing_out['logits'])
        merged['logits'] = logits
        merged['pred'] = logits
        merged['aux'] = {}
        merged['aux_losses'] = {}
        merged['diagnostics'] = {}
        return merged

    def forward(self, ct, pet, pet_available=None, target_size=None, forward_mode='auto', mask=None):
        if target_size is None:
            target_size = ct.shape[-2:]
        if forward_mode == 'full':
            return self._forward_full(ct, pet, target_size)
        if forward_mode == 'missing':
            return self._forward_missing(ct, target_size)
        if forward_mode == 'auto':
            if pet_available is None:
                pet_available = torch.ones(ct.shape[0], device=ct.device, dtype=torch.long)
            return self._forward_auto(ct, pet, pet_available, target_size)
        raise ValueError(forward_mode)
