import torch
import torch.nn as nn

from models.baseline_petct_unet import UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import create_feature_backbone, load_local_weights_safe
from models.dual_decoder_add_baseline import StageChannelAlign


class FullPETCTAddUNet(nn.Module):
    def __init__(
        self,
        ct_backbone='convnextv2_nano',
        pet_backbone='mit_b1',
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        in_channels=3,
        out_channels=1,
        decoder_channels=(512, 256, 128, 64),
        **kwargs,
    ):
        super().__init__()
        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        self.enc_pet = create_feature_backbone(pet_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='CT_Encoder')
        load_local_weights_safe(self.enc_pet, pet_pretrained_path, name='PET_Encoder')

        ct_channels = list(self.enc_ct.feature_info.channels())
        pet_channels = list(self.enc_pet.feature_info.channels())
        if len(ct_channels) != 4 or len(pet_channels) != 4:
            raise ValueError('Both encoders must output 4 stage features.')
        if pet_channels != [64, 128, 320, 512]:
            raise ValueError(f'PET encoder channels must be [64, 128, 320, 512], got {pet_channels}.')

        self.ct_align = StageChannelAlign(ct_channels, [64, 128, 320, 512])
        self.fusion = AddFusion()
        self.decoder = UNetStyleDecoder(
            encoder_channels=[64, 128, 320, 512],
            decoder_channels=tuple(decoder_channels),
            out_channels=out_channels,
            use_deep_supervision=False,
        )

    @staticmethod
    def _to_3ch(x):
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x

    def _encode_ct(self, ct):
        ct = self._to_3ch(ct)
        with torch.cuda.amp.autocast(enabled=False):
            ct_feats = self.enc_ct(ct)
            ct_feats = [feat.float() for feat in ct_feats]
            _check_tensor_list('ct_feats', ct_feats)
            return self.ct_align(ct_feats)

    def _encode_pet(self, pet):
        pet = self._to_3ch(pet)
        with torch.cuda.amp.autocast(enabled=False):
            pet_feats = self.enc_pet(pet)
            pet_feats = [feat.float() for feat in pet_feats]
            _check_tensor_list('pet_feats', pet_feats)
            return pet_feats

    def _finalize_decoder_output(self, dec_out):
        if isinstance(dec_out, dict):
            out = {'logits': _sanitize(dec_out['logits'])}
            return out
        return {'logits': _sanitize(dec_out)}

    def forward(self, ct, pet, pet_available=None, target_size=None, forward_mode='auto'):
        if pet is None:
            raise ValueError('pet_contribution_full requires PET input.')
        forward_mode = str(forward_mode)
        if forward_mode == 'missing':
            raise ValueError('pet_contribution_full does not support missing PET.')
        if target_size is None:
            target_size = ct.shape[-2:]
        ct_feats = self._encode_ct(ct)
        pet_feats = self._encode_pet(pet)
        fused_feats = self.fusion(ct_feats, pet_feats, pet_available=pet_available)
        dec_out = self.decoder(fused_feats, target_size)
        outputs = self._finalize_decoder_output(dec_out)
        _check_tensor('logits', outputs['logits'])
        outputs['pred'] = outputs['logits']
        outputs['aux'] = {}
        return outputs
