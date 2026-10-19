"""Independent CT-only or heterogeneous aligned-add segmentation baseline.
CT-only allocates no PET encoder, PET aligner or fusion module.
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


class DualSharedAddPETCTBaseline(nn.Module):
    def __init__(self, ct_backbone='convnextv2_nano', pet_backbone='mit_b1',
                 ct_pretrained_path=None, pet_pretrained_path=None,
                 in_channels=3, out_channels=1,
                 decoder_channels=(512, 256, 128, 64),
                 use_deep_supervision=False, pretrained=True, baseline="full_add"):
        super().__init__()
        self.baseline = baseline
        if baseline not in ("ct_only", "full_add"):
            raise ValueError(baseline)
        self.use_deep_supervision = bool(use_deep_supervision)
        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path,
                                name='CT_Encoder', required=bool(pretrained))
        ct_channels = list(self.enc_ct.feature_info.channels())
        self.ct_align = StageChannelAlign(ct_channels, list(DECODER_INPUT_CHANNELS))
        if baseline == "full_add":
            self.enc_pet = create_feature_backbone(pet_backbone, in_channels=in_channels)
            load_local_weights_safe(self.enc_pet, pet_pretrained_path,
                                    name='PET_Encoder', required=bool(pretrained))
            self.pet_align = StageChannelAlign(
                list(self.enc_pet.feature_info.channels()), list(DECODER_INPUT_CHANNELS))
            self.fusion = AddFusion()
        self.decoder = UNetStyleDecoder(
            list(DECODER_INPUT_CHANNELS), decoder_channels=decoder_channels,
            out_channels=out_channels, use_deep_supervision=self.use_deep_supervision)

    @staticmethod
    def _to_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _encode_ct(self, ct):
        feats = self.enc_ct(self._to_3ch(ct))
        _check_finite('ct_feats', feats)
        aligned = self.ct_align(feats)
        _check_finite('aligned_ct', aligned)
        return aligned

    def _encode_pet(self, pet):
        feats = self.enc_pet(self._to_3ch(pet))
        _check_finite('pet_feats', feats)
        aligned = self.pet_align(feats)
        _check_finite('aligned_pet', aligned)
        return aligned

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
        _check_finite('fused_feats', fused)
        return self._decode(fused, target_size)

    def forward(self, ct, pet=None, target_size=None, forward_mode=None):
        target_size = ct.shape[-2:] if target_size is None else target_size
        if self.baseline == "ct_only":
            return self._decode(self._encode_ct(ct), target_size)
        if forward_mode == "missing":
            return self._forward_missing(ct, None, target_size)
        return self._forward_full(ct, pet, target_size)
