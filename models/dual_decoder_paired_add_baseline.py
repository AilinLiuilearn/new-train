import copy

import torch
import torch.nn as nn

from models.baseline_petct_unet import UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import ConvBNAct, create_feature_backbone, load_local_weights_safe


class StageChannelAlign(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        super().__init__()
        self.proj = nn.ModuleList(
            [ConvBNAct(c_in, c_out, kernel_size=1) for c_in, c_out in zip(in_channels_list, out_channels_list)]
        )

    def forward(self, feats):
        return [proj(feat) for proj, feat in zip(self.proj, feats)]


class DualDecoderPairedAddPETCTBaseline(nn.Module):
    def __init__(
        self,
        ct_backbone='convnextv2_nano',
        pet_backbone='mit_b1',
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        in_channels=3,
        out_channels=1,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False,
    ):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)
        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        self.enc_pet = create_feature_backbone(pet_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='CT_Encoder')
        load_local_weights_safe(self.enc_pet, pet_pretrained_path, name='PET_Encoder')

        ct_channels = list(self.enc_ct.feature_info.channels())
        pet_channels = list(self.enc_pet.feature_info.channels())
        if len(ct_channels) != 4 or len(pet_channels) != 4:
            raise ValueError('Both encoders must output 4 stage features.')

        self.ct_align = StageChannelAlign(ct_channels, pet_channels)
        self.full_decoder = UNetStyleDecoder(pet_channels, decoder_channels=decoder_channels, out_channels=out_channels, use_deep_supervision=self.use_deep_supervision)
        self.missing_decoder = copy.deepcopy(self.full_decoder)

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

    def _decode(self, decoder, feats, target_size):
        dec_out = decoder(feats, target_size)
        if isinstance(dec_out, dict):
            logits = _sanitize(dec_out['logits'])
            outputs = {'logits': logits}
            if 'aux_logits' in dec_out:
                outputs['aux_logits'] = [_sanitize(x) for x in dec_out['aux_logits']]
        else:
            outputs = {'logits': _sanitize(dec_out)}
        _check_tensor('logits', outputs['logits'])
        outputs['pred'] = outputs['logits']
        return outputs

    def _forward_full_only(self, ct, pet, target_size):
        ct_feats = self._encode_ct(ct)
        pet_feats = self._encode_pet(pet)
        fused = [_sanitize(c + p) for c, p in zip(ct_feats, pet_feats)]
        return self._decode(self.full_decoder, fused, target_size)

    def _forward_missing_only(self, ct, target_size):
        ct_feats = self._encode_ct(ct)
        return self._decode(self.missing_decoder, ct_feats, target_size)

    def forward(self, ct, pet, pet_available=None, target_size=None, forward_mode='auto'):
        if target_size is None:
            target_size = ct.shape[-2:]
        forward_mode = str(forward_mode)
        if forward_mode == 'full':
            if self.training:
                if pet is None:
                    raise ValueError('forward_mode="full" requires pet input during training.')
                full_logits = self._forward_full_only(ct, pet, target_size)
                missing_logits = self._forward_missing_only(ct, target_size)
                return {
                    'logits': missing_logits['logits'],
                    'paired_joint': True,
                    'paired_full_logits': full_logits['logits'],
                    'paired_missing_logits': missing_logits['logits'],
                }
            if pet is None:
                raise ValueError('forward_mode="full" requires pet input.')
            return self._forward_full_only(ct, pet, target_size)
        if forward_mode == 'missing':
            return self._forward_missing_only(ct, target_size)
        raise ValueError('forward_mode must be full or missing for paired baseline')
