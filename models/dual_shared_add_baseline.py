import torch
import torch.nn as nn

from models.baseline_blocks import AddFusion, UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import create_feature_backbone, load_local_weights_safe
from models.ssppc_module import SpatialSemanticPairedPrototypeCompensation


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
    def __init__(self, ct_backbone='convnextv2_nano', pet_backbone='mit_b1', ct_pretrained_path=None, pet_pretrained_path=None, in_channels=3, out_channels=1, decoder_channels=(512, 256, 128, 64), use_deep_supervision=False, use_ssppc=True, ssppc_outlier_ratio=0.05, ssppc_cache_on_cpu=True):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)
        self.use_ssppc = bool(use_ssppc)
        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        self.enc_pet = create_feature_backbone(pet_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='CT_Encoder')
        load_local_weights_safe(self.enc_pet, pet_pretrained_path, name='PET_Encoder')
        ct_channels = list(self.enc_ct.feature_info.channels())
        pet_channels = list(self.enc_pet.feature_info.channels())
        self.ct_align = StageChannelAlign(ct_channels, pet_channels)
        self.fusion = AddFusion()
        self.decoder = UNetStyleDecoder(pet_channels, decoder_channels=decoder_channels, out_channels=out_channels, use_deep_supervision=self.use_deep_supervision)
        self.ssppc = SpatialSemanticPairedPrototypeCompensation(channels=pet_channels, outlier_ratio=ssppc_outlier_ratio, cache_on_cpu=ssppc_cache_on_cpu) if self.use_ssppc else None

    @staticmethod
    def _to_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _encode_ct(self, ct):
        ct_feats = self.enc_ct(self._to_3ch(ct))
        _check_tensor_list('ct_feats', ct_feats)
        return self.ct_align(ct_feats)

    def _encode_pet(self, pet):
        if pet is None:
            raise ValueError('API-style baseline requires PET input before fusion-time masking')
        pet_feats = self.enc_pet(self._to_3ch(pet))
        _check_tensor_list('pet_feats', pet_feats)
        return pet_feats

    def _encode_pair(self, ct, pet, mask=None):
        ct_feats = self._encode_ct(ct)
        pet_feats_real = self._encode_pet(pet)
        if self.training and self.use_ssppc and self.ssppc is not None and mask is not None:
            self.ssppc.collect(ct_feats=ct_feats, pet_feats_real=pet_feats_real, mask=mask)
        return ct_feats, pet_feats_real

    def _decode(self, fused_feats, target_size):
        out = self.decoder(fused_feats, target_size)
        _check_tensor('logits', out['logits'])
        out['pred'] = out['logits']
        out['aux'] = {}
        return out

    def _forward_full(self, ct, pet, target_size, mask=None, return_ssppc_debug=False):
        ct_feats, pet_feats_real = self._encode_pair(ct, pet, mask)
        if return_ssppc_debug and self.use_ssppc and self.ssppc is not None:
            _, ssppc_debug = self.ssppc(ct_feats, return_debug=True)
        else:
            ssppc_debug = None
        fused_feats = self.fusion(ct_feats, pet_feats_real, None)
        out = self._decode(fused_feats, target_size)
        if return_ssppc_debug and ssppc_debug is not None:
            out['ssppc_debug'] = ssppc_debug
        return out

    def _forward_missing(self, ct, pet, target_size, mask=None, return_ssppc_debug=False):
        ct_feats, pet_feats_real = self._encode_pair(ct, pet, mask)
        if self.use_ssppc and self.ssppc is not None:
            if return_ssppc_debug:
                pet_feats_comp, ssppc_debug = self.ssppc(ct_feats, return_debug=True)
            else:
                pet_feats_comp = self.ssppc(ct_feats, return_debug=False)
                ssppc_debug = None
        else:
            pet_feats_comp = [torch.zeros_like(feat) for feat in pet_feats_real]
            ssppc_debug = None
        fused_feats = self.fusion(ct_feats, pet_feats_comp, None)
        out = self._decode(fused_feats, target_size)
        if return_ssppc_debug and ssppc_debug is not None:
            out['ssppc_debug'] = ssppc_debug
        return out

    def _forward_auto(self, ct, pet, pet_available, target_size, mask=None, return_ssppc_debug=False):
        ct_feats, pet_feats_real = self._encode_pair(ct, pet, mask)
        pet_available = pet_available.to(device=ct.device).long().view(-1)
        if pet_available.numel() != ct.shape[0]:
            raise ValueError('pet_available must contain one state per sample')
        if not torch.all((pet_available == 0) | (pet_available == 1)):
            raise ValueError('pet_available values must be 0 or 1')
        if self.use_ssppc and self.ssppc is not None:
            pet_feats_comp = self.ssppc(ct_feats, return_debug=False)
            pet_feats_aux = self.ssppc.route_pet_features(pet_feats_real=pet_feats_real, pet_feats_comp=pet_feats_comp, pet_missing=(pet_available == 0))
            if return_ssppc_debug:
                _, ssppc_debug = self.ssppc(ct_feats, return_debug=True)
            else:
                ssppc_debug = None
        else:
            pet_feats_aux = []
            for feat in pet_feats_real:
                availability_mask = pet_available.to(device=feat.device, dtype=feat.dtype).view(-1, 1, 1, 1)
                pet_feats_aux.append(feat * availability_mask)
            ssppc_debug = None
        fused_feats = self.fusion(ct_feats, pet_feats_aux, None)
        out = self._decode(fused_feats, target_size)
        if return_ssppc_debug and ssppc_debug is not None:
            out['ssppc_debug'] = ssppc_debug
        return out

    def forward(self, ct, pet, pet_available=None, target_size=None, forward_mode='auto', mask=None, return_ssppc_debug=False):
        if target_size is None:
            target_size = ct.shape[-2:]
        if forward_mode == 'full':
            return self._forward_full(ct, pet, target_size, mask=mask, return_ssppc_debug=return_ssppc_debug)
        if forward_mode == 'missing':
            return self._forward_missing(ct, pet, target_size, mask=mask, return_ssppc_debug=return_ssppc_debug)
        if forward_mode == 'auto':
            if pet_available is None:
                pet_available = torch.ones(ct.shape[0], device=ct.device, dtype=torch.long)
            return self._forward_auto(ct, pet, pet_available, target_size, mask=mask, return_ssppc_debug=return_ssppc_debug)
        raise ValueError(f'Unsupported forward_mode={forward_mode!r}')
