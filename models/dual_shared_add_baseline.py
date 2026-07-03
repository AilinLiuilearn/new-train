import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline_petct_unet import AddFusion, UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import ConvBNAct, create_feature_backbone, load_local_weights_safe
from models.simmlm_dmome_fusion import make_full_pet_available


class StageChannelAlign(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        super().__init__()
        self.proj = nn.ModuleList(
            [ConvBNAct(c_in, c_out, kernel_size=1) for c_in, c_out in zip(in_channels_list, out_channels_list)]
        )

    def forward(self, feats):
        return [proj(feat) for proj, feat in zip(self.proj, feats)]


class DualSharedAddPETCTBaseline(nn.Module):
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
        self.fusion = AddFusion()
        self.decoder = UNetStyleDecoder(
            pet_channels,
            decoder_channels=decoder_channels,
            out_channels=out_channels,
            use_deep_supervision=self.use_deep_supervision,
        )

    @staticmethod
    def _to_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def forward(self, ct, pet, pet_available=None, target_size=None):
        if target_size is None:
            target_size = ct.shape[-2:]
        ct = self._to_3ch(ct)
        pet = self._to_3ch(pet)
        pet_available = make_full_pet_available(ct.shape[0], ct.device) if pet_available is None else pet_available.long().view(-1)

        ct_feats = self.enc_ct(ct)
        pet_feats = self.enc_pet(pet)
        _check_tensor_list('ct_feats', ct_feats)
        _check_tensor_list('pet_feats', pet_feats)

        fused_feats = self.fusion(self.ct_align(ct_feats), pet_feats, pet_available=pet_available)
        dec_out = self.decoder(fused_feats, target_size)
        outputs = self._finalize_decoder_output(dec_out)
        _check_tensor('logits', outputs['logits'])
        outputs['pred'] = outputs['logits']
        outputs['aux'] = {}
        return outputs

    @staticmethod
    def _finalize_decoder_output(dec_out):
        if isinstance(dec_out, dict):
            out = {'logits': _sanitize(dec_out['logits'])}
            if 'aux_logits' in dec_out:
                out['aux_logits'] = [_sanitize(x) for x in dec_out['aux_logits']]
            return out
        return {'logits': _sanitize(dec_out)}
