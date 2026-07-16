import torch
import torch.nn as nn

from models.baseline_petct_unet import UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import create_feature_backbone, load_local_weights_safe
from models.dual_decoder_add_baseline import StageChannelAlign


class CTOnlyConvNeXtUNet(nn.Module):
    def __init__(
        self,
        ct_backbone='convnextv2_nano',
        ct_pretrained_path=None,
        in_channels=3,
        out_channels=1,
        decoder_channels=(512, 256, 128, 64),
        **kwargs,
    ):
        super().__init__()
        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='CT_Encoder')

        ct_channels = list(self.enc_ct.feature_info.channels())
        if len(ct_channels) != 4:
            raise ValueError('CT encoder must output 4 stage features.')

        self.ct_align = StageChannelAlign(ct_channels, [64, 128, 320, 512])
        self.decoder = UNetStyleDecoder(
            encoder_channels=[64, 128, 320, 512],
            decoder_channels=tuple(decoder_channels),
            out_channels=out_channels,
            use_deep_supervision=False,
        )
        self.ct_align = self.ct_align

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
        ct_feats = [feat.detach().float().contiguous() for feat in ct_feats]
        _check_tensor_list('ct_feats', ct_feats)
        return self.ct_align(ct_feats)

    def _finalize_decoder_output(self, dec_out):
        if isinstance(dec_out, dict):
            out = {'logits': _sanitize(dec_out['logits'])}
            return out
        return {'logits': _sanitize(dec_out)}

    def forward(self, ct, pet=None, pet_available=None, target_size=None, forward_mode='auto'):
        if target_size is None:
            target_size = ct.shape[-2:]
        features = self._encode_ct(ct)
        dec_out = self.decoder(features, target_size)
        outputs = self._finalize_decoder_output(dec_out)
        _check_tensor('logits', outputs['logits'])
        outputs['pred'] = outputs['logits']
        outputs['aux'] = {}
        return outputs
