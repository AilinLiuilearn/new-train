import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline_petct_unet import UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import create_feature_backbone, load_local_weights_safe
from models.hatr_task_residual import CorrectionAdapter, HierarchicalTaskResidualRecovery


class StageChannelAlign(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        super().__init__()
        from models.build_mdt_seg import ConvBNAct
        self.proj = nn.ModuleList([ConvBNAct(c_in, c_out, kernel_size=1) for c_in, c_out in zip(in_channels_list, out_channels_list)])

    def forward(self, feats):
        return [proj(feat) for proj, feat in zip(self.proj, feats)]


class DualDecoderHATRTaskResidual(nn.Module):
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
        self.full_decoder = UNetStyleDecoder(pet_channels, decoder_channels=decoder_channels, out_channels=out_channels, use_deep_supervision=self.use_deep_supervision)
        self.missing_decoder = copy.deepcopy(self.full_decoder)
        self.hatr_recovery = HierarchicalTaskResidualRecovery(decoder_channels, pet_channels)
        d4, d3, d2, d1 = decoder_channels
        self.correction_adapter4 = CorrectionAdapter(d4)
        self.correction_adapter3 = CorrectionAdapter(d3)
        self.correction_adapter2 = CorrectionAdapter(d2)
        self.correction_adapter1 = CorrectionAdapter(d1)
        self.decoder_channels = tuple(decoder_channels)

    @staticmethod
    def _to_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _encode_ct(self, ct):
        ct = self._to_3ch(ct)
        ct_feats = self.enc_ct(ct)
        _check_tensor_list('ct_feats', ct_feats)
        return self.ct_align(ct_feats)

    def _encode_pet(self, pet):
        pet = self._to_3ch(pet)
        pet_feats = self.enc_pet(pet)
        _check_tensor_list('pet_feats', pet_feats)
        return pet_feats

    def _finalize_decoder_output(self, dec_out):
        if isinstance(dec_out, dict):
            out = {'logits': _sanitize(dec_out['logits'])}
            if 'aux_logits' in dec_out:
                out['aux_logits'] = [_sanitize(x) for x in dec_out['aux_logits']]
            return out
        return {'logits': _sanitize(dec_out)}

    def _decode_with_states(self, decoder, features, target_size):
        x1, x2, x3, x4 = features
        d4 = decoder.proj4(x4)
        s3 = decoder.proj3(x3)
        d3 = decoder.fuse3(torch.cat([F.interpolate(d4, size=s3.shape[-2:], mode='bilinear', align_corners=False), s3], dim=1))
        s2 = decoder.proj2(x2)
        d2 = decoder.fuse2(torch.cat([F.interpolate(d3, size=s2.shape[-2:], mode='bilinear', align_corners=False), s2], dim=1))
        s1 = decoder.proj1(x1)
        d1 = decoder.fuse1(torch.cat([F.interpolate(d2, size=s1.shape[-2:], mode='bilinear', align_corners=False), s1], dim=1))
        logits = decoder.seg_head(d1)
        logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)
        out = {'logits': logits}
        if decoder.use_deep_supervision:
            out['aux_logits'] = [decoder.aux_head_d2(d2), decoder.aux_head_d3(d3), decoder.aux_head_d4(d4)]
        return out, [d1, d2, d3, d4]

    def _decode_missing_with_residuals(self, decoder, features, pred_residuals, target_size):
        x1, x2, x3, x4 = features
        r1, r2, r3, r4 = pred_residuals
        d4_base = decoder.proj4(x4)
        d4 = d4_base + self.correction_adapter4(r4)
        s3 = decoder.proj3(x3)
        d3_base = decoder.fuse3(torch.cat([F.interpolate(d4, size=s3.shape[-2:], mode='bilinear', align_corners=False), s3], dim=1))
        d3 = d3_base + self.correction_adapter3(r3)
        s2 = decoder.proj2(x2)
        d2_base = decoder.fuse2(torch.cat([F.interpolate(d3, size=s2.shape[-2:], mode='bilinear', align_corners=False), s2], dim=1))
        d2 = d2_base + self.correction_adapter2(r2)
        s1 = decoder.proj1(x1)
        d1_base = decoder.fuse1(torch.cat([F.interpolate(d2, size=s1.shape[-2:], mode='bilinear', align_corners=False), s1], dim=1))
        d1 = d1_base + self.correction_adapter1(r1)
        logits = F.interpolate(decoder.seg_head(d1), size=target_size, mode='bilinear', align_corners=False)
        out = {'logits': logits}
        if decoder.use_deep_supervision:
            out['aux_logits'] = [decoder.aux_head_d2(d2), decoder.aux_head_d3(d3), decoder.aux_head_d4(d4)]
        return out, [d1, d2, d3, d4], [d1_base, d2_base, d3_base, d4_base]

    def _forward_full(self, ct, pet, target_size):
        ct_feats = self._encode_ct(ct)
        pet_feats = self._encode_pet(pet)
        fused_feats = [c + p for c, p in zip(ct_feats, pet_feats)]
        full_out, full_states = self._decode_with_states(self.full_decoder, fused_feats, target_size)
        if self.training:
            with torch.no_grad():
                ct_cf_out, ct_cf_states = self._decode_with_states(self.full_decoder, [c.detach() for c in ct_feats], target_size)
            pred_residuals = self.hatr_recovery([c.detach() for c in ct_feats])
            full_out['hatr_counterfactual_logits'] = ct_cf_out['logits'].detach()
            full_out['hatr_full_states'] = [x.detach() for x in full_states]
            full_out['hatr_counterfactual_states'] = [x.detach() for x in ct_cf_states]
            full_out['hatr_pred_residuals'] = pred_residuals
        return full_out

    def _forward_missing(self, ct, target_size):
        ct_feats = self._encode_ct(ct)
        pred_residuals = self.hatr_recovery([c.detach() for c in ct_feats])
        out, _, _ = self._decode_missing_with_residuals(self.missing_decoder, ct_feats, pred_residuals, target_size)
        return out

    def forward(self, ct, pet, pet_available=None, target_size=None, forward_mode='auto'):
        if target_size is None:
            target_size = ct.shape[-2:]
        if forward_mode == 'full':
            return self._forward_full(ct, pet, target_size)
        if forward_mode == 'missing':
            return self._forward_missing(ct, target_size)
        if forward_mode == 'auto':
            if pet_available is None:
                pet_available = torch.ones(ct.shape[0], device=ct.device, dtype=torch.long)
            pet_available = pet_available.long().view(-1)
            if torch.all(pet_available == 1):
                return self._forward_full(ct, pet, target_size)
            if torch.all(pet_available == 0):
                return self._forward_missing(ct, target_size)
            ct_feats = self._encode_ct(ct)
            full_idx = torch.nonzero(pet_available == 1, as_tuple=False).flatten()
            missing_idx = torch.nonzero(pet_available == 0, as_tuple=False).flatten()
            full_outputs = None
            missing_outputs = None
            if full_idx.numel() > 0:
                full_ct_feats = [feat.index_select(0, full_idx) for feat in ct_feats]
                pet_full = pet.index_select(0, full_idx)
                pet_feats = self._encode_pet(pet_full)
                fused = [c + p for c, p in zip(full_ct_feats, pet_feats)]
                full_outputs, _ = self._decode_with_states(self.full_decoder, fused, target_size)
            if missing_idx.numel() > 0:
                missing_ct_feats = [feat.index_select(0, missing_idx) for feat in ct_feats]
                pred_residuals = self.hatr_recovery([c.detach() for c in missing_ct_feats])
                missing_outputs, _, _ = self._decode_missing_with_residuals(self.missing_decoder, missing_ct_feats, pred_residuals, target_size)
            if full_outputs is None:
                return missing_outputs
            if missing_outputs is None:
                return full_outputs
            logits = full_outputs['logits'].new_zeros((ct.shape[0],) + tuple(full_outputs['logits'].shape[1:]))
            logits.index_copy_(0, full_idx, full_outputs['logits'])
            logits.index_copy_(0, missing_idx, missing_outputs['logits'])
            return {'logits': logits}
        raise ValueError(f'Unsupported forward_mode={forward_mode!r}')
