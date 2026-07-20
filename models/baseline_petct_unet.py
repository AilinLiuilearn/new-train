import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline_blocks import AddFusion, UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import create_feature_backbone, load_local_weights_safe


class ConcatConvFusion(nn.Module):
    def __init__(self, ct_channels, pet_channels, out_channels=None):
        super().__init__()
        if len(ct_channels) != len(pet_channels):
            raise ValueError('ct_channels and pet_channels must have same length')
        out_channels = list(out_channels or ct_channels)
        self.fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c_ct + c_pet, c_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ) for c_ct, c_pet, c_out in zip(ct_channels, pet_channels, out_channels)
        ])

    def forward(self, ct_feats, pet_feats):
        fused = []
        for ct_feat, pet_feat, fuse in zip(ct_feats, pet_feats, self.fuse):
            if pet_feat.shape[-2:] != ct_feat.shape[-2:]:
                pet_feat = F.interpolate(pet_feat, size=ct_feat.shape[-2:], mode='bilinear', align_corners=False)
            fused.append(_sanitize(fuse(torch.cat([ct_feat, pet_feat], dim=1))))
        return fused


class PETCTBaselineUNet(nn.Module):
    def __init__(self, ct_backbone='mit_b1', pet_backbone='mit_b1', ct_pretrained_path=None, pet_pretrained_path=None, in_channels=3, out_channels=1, decoder_channels=(512, 256, 128, 64), fusion_type='concat_conv', use_deep_supervision=False, **kwargs):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)
        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        self.enc_pet = create_feature_backbone(pet_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='CT_Encoder')
        load_local_weights_safe(self.enc_pet, pet_pretrained_path, name='PET_Encoder')
        ct_channels = self.enc_ct.feature_info.channels()
        pet_channels = self.enc_pet.feature_info.channels()
        if fusion_type == 'add':
            self.fusion = AddFusion()
        else:
            self.fusion = ConcatConvFusion(ct_channels, pet_channels, out_channels=ct_channels)
        self.decoder = UNetStyleDecoder(ct_channels, decoder_channels=decoder_channels, out_channels=out_channels, use_deep_supervision=self.use_deep_supervision)

    @staticmethod
    def _to_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def forward(self, ct, pet, pet_available=None, target_size=None, return_aux=False):
        if target_size is None:
            target_size = ct.shape[-2:]
        ct = self._to_3ch(ct)
        pet = self._to_3ch(pet)
        ct_feats = self.enc_ct(ct)
        pet_feats = self.enc_pet(pet)
        fused_feats = self.fusion(ct_feats, pet_feats)
        out = self.decoder(fused_feats, target_size)
        out['pred'] = out['logits']
        out['aux'] = {}
        return out
