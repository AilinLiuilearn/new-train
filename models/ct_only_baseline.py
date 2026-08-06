import torch
import torch.nn as nn

from models.baseline_blocks import UNetStyleDecoder, _check_tensor, _check_tensor_list
from models.build_mdt_seg import create_feature_backbone, load_local_weights_safe


class CTOnlyUNetStyle(nn.Module):
    def __init__(
        self,
        ct_backbone='convnextv2_nano',
        ct_pretrained_path=None,
        in_channels=3,
        out_channels=1,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False,
    ):
        super().__init__()
        self.is_ct_only = True
        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='CT_Encoder')
        ct_channels = list(self.enc_ct.feature_info.channels())
        self.ct_align = nn.Identity()
        self.decoder = UNetStyleDecoder(
            ct_channels,
            decoder_channels=decoder_channels,
            out_channels=out_channels,
            use_deep_supervision=use_deep_supervision,
        )

    @staticmethod
    def _to_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _encode_ct(self, ct):
        ct_feats = self.enc_ct(self._to_3ch(ct))
        _check_tensor_list('ct_feats', ct_feats)
        return self.ct_align(ct_feats)

    def _decode(self, ct_feats, target_size):
        out = self.decoder(ct_feats, target_size)
        _check_tensor('logits', out['logits'])
        out['pred'] = out['logits']
        out['aux'] = {}
        return out

    def forward(
        self,
        ct,
        pet=None,
        pet_available=None,
        target_size=None,
        forward_mode='ct_only',
        mask=None,
    ):
        del pet, pet_available, mask
        if target_size is None:
            target_size = ct.shape[-2:]
        if forward_mode not in ('ct_only', 'full', 'missing', 'auto'):
            raise ValueError(f'Unsupported forward_mode={forward_mode!r}')
        return self._decode(self._encode_ct(ct), target_size)

    @torch.no_grad()
    def finalize_cppi_epoch(self, *args, **kwargs):
        del args, kwargs
        return {
            'status': 'ct_only_no_cppi',
            'bank_version_before': 0,
            'bank_version_after': 0,
            'ready_count': 0,
            'ready_slots': 0,
            'classes': {},
        }
