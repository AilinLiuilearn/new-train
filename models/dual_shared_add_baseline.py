import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline_blocks import AddFusion, UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import create_feature_backbone, load_local_weights_safe
from models.mtpi_compensation import MTPICompensation


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
    def __init__(self, ct_backbone='convnextv2_nano', pet_backbone='mit_b1', ct_pretrained_path=None, pet_pretrained_path=None, in_channels=3, out_channels=1, decoder_channels=(512, 256, 128, 64), use_deep_supervision=False, mtpi_filter_ratio=0.05):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)
        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        self.enc_pet = create_feature_backbone(pet_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='CT_Encoder')
        load_local_weights_safe(self.enc_pet, pet_pretrained_path, name='PET_Encoder')
        ct_channels = list(self.enc_ct.feature_info.channels())
        pet_channels = list(self.enc_pet.feature_info.channels())
        self.ct_align = StageChannelAlign(ct_channels, pet_channels)
        self.mtpi = MTPICompensation(channels=pet_channels, filter_ratio=mtpi_filter_ratio)
        self.fusion = AddFusion()
        self.decoder = UNetStyleDecoder(pet_channels, decoder_channels=decoder_channels, out_channels=out_channels, use_deep_supervision=self.use_deep_supervision)

    @staticmethod
    def _to_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    @staticmethod
    def _match_pet_spatial(ct_feats, pet_feats):
        matched = []
        for ct_feat, pet_feat in zip(ct_feats, pet_feats):
            if pet_feat.shape[-2:] != ct_feat.shape[-2:]:
                pet_feat = F.interpolate(pet_feat, size=ct_feat.shape[-2:], mode='bilinear', align_corners=False)
            matched.append(pet_feat)
        return matched

    def _encode_ct(self, ct):
        ct_feats = self.enc_ct(self._to_3ch(ct))
        _check_tensor_list('ct_feats', ct_feats)
        return self.ct_align(ct_feats)

    def _encode_pet(self, pet):
        pet_feats = self.enc_pet(self._to_3ch(pet))
        _check_tensor_list('pet_feats', pet_feats)
        return pet_feats

    def _decode(self, fused_feats, target_size):
        out = self.decoder(fused_feats, target_size)
        _check_tensor('logits', out['logits'])
        out['pred'] = out['logits']
        out['aux'] = {}
        return out

    def _forward_full(self, ct, pet, target_size, target=None):
        ct_feats = self._encode_ct(ct)
        pet_feats = self._encode_pet(pet)
        pet_feats = self._match_pet_spatial(ct_feats, pet_feats)
        with torch.no_grad():
            self.mtpi(ct_feats, pet_features=pet_feats, modality_state='full', target=target)
        fused_feats = self.fusion(ct_feats, pet_feats, None)
        out = self._decode(fused_feats, target_size)
        out['mtpi_reference_loss'] = ct_feats[0].new_zeros(())
        return out

    def _forward_missing(self, ct, target_size, target=None):
        ct_feats = self._encode_ct(ct)
        compensated_pet_feats = self.mtpi(ct_feats, pet_features=None, modality_state='missing')
        fused_feats = self.fusion(ct_feats, compensated_pet_feats, None)
        out = self._decode(fused_feats, target_size)
        if target is not None:
            out['mtpi_reference_loss'] = self.mtpi.reference_loss(compensated_pet_feats, target)
        else:
            out['mtpi_reference_loss'] = ct_feats[0].new_zeros(())
        return out

    def _forward_auto(self, ct, pet, pet_available, target_size):
        pet_available = pet_available.long().view(-1)
        if pet_available.numel() != ct.shape[0]:
            raise ValueError('pet_available length must match batch size')
        if not torch.all((pet_available == 0) | (pet_available == 1)):
            raise ValueError('pet_available must contain only 0 and 1')
        if torch.all(pet_available == 1):
            return self._forward_full(ct, pet, target_size)
        if torch.all(pet_available == 0):
            return self._forward_missing(ct, target_size)
        ct_feats = self._encode_ct(ct)
        full_idx = torch.nonzero(pet_available == 1, as_tuple=False).flatten()
        missing_idx = torch.nonzero(pet_available == 0, as_tuple=False).flatten()
        fused = [feat.clone() for feat in ct_feats]
        if full_idx.numel() > 0:
            pet_full = pet.index_select(0, full_idx)
            pet_feats = self._match_pet_spatial([feat.index_select(0, full_idx) for feat in ct_feats], self._encode_pet(pet_full))
            pet_feats = self.mtpi([feat.index_select(0, full_idx) for feat in ct_feats], pet_features=pet_feats, modality_state='full', target=None)
            fused_full = self.fusion([feat.index_select(0, full_idx) for feat in ct_feats], pet_feats, None)
            for i, feat in enumerate(fused_full):
                fused[i].index_copy_(0, full_idx, feat)
        if missing_idx.numel() > 0:
            missing_pet = self.mtpi([feat.index_select(0, missing_idx) for feat in ct_feats], pet_features=None, modality_state='missing')
            fused_missing = self.fusion([feat.index_select(0, missing_idx) for feat in ct_feats], missing_pet, None)
            for i, feat in enumerate(fused_missing):
                fused[i].index_copy_(0, missing_idx, feat)
        out = self._decode(fused, target_size)
        out['mtpi_reference_loss'] = ct_feats[0].new_zeros(())
        return out

    def forward(self, ct, pet, pet_available=None, target_size=None, forward_mode='auto', target=None):
        if target_size is None:
            target_size = ct.shape[-2:]
        if forward_mode == 'full':
            return self._forward_full(ct, pet, target_size, target=target)
        if forward_mode == 'missing':
            return self._forward_missing(ct, target_size, target=target)
        if forward_mode == 'auto':
            if pet_available is None:
                pet_available = torch.ones(ct.shape[0], device=ct.device, dtype=torch.long)
            return self._forward_auto(ct, pet, pet_available, target_size)
        raise ValueError(f'Unsupported forward_mode={forward_mode!r}')
