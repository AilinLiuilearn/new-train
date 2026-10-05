# -*- coding: utf-8 -*-
"""CT-only segmentation baseline.

ConvNeXtV2-Nano CT encoder -> StageChannelAlign -> shared GroupNorm decoder.
No PET encoder is instantiated, no PET file is read, no fusion is used.
"""
import torch
import torch.nn as nn

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


class CTOnlySegmentationModel(nn.Module):
    """CT -> align -> decode. Forward takes CT only."""

    def __init__(self, ct_backbone='convnextv2_nano', ct_pretrained_path=None,
                 in_channels=3, out_channels=1,
                 decoder_channels=(512, 256, 128, 64),
                 use_deep_supervision=False, pretrained=True):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)
        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path,
                                name='CT_Encoder', required=bool(pretrained))
        ct_channels = list(self.enc_ct.feature_info.channels())
        self.ct_align = StageChannelAlign(ct_channels, list(DECODER_INPUT_CHANNELS))
        self.decoder = UNetStyleDecoder(
            list(DECODER_INPUT_CHANNELS), decoder_channels=decoder_channels,
            out_channels=out_channels, use_deep_supervision=self.use_deep_supervision)

    @staticmethod
    def _to_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _encode_ct(self, ct):
        ct_feats = self.enc_ct(self._to_3ch(ct))
        _check_finite('ct_feats', ct_feats)
        aligned = self.ct_align(ct_feats)
        _check_finite('aligned_ct', aligned)
        return aligned

    def forward(self, ct, target_size=None):
        if target_size is None:
            target_size = ct.shape[-2:]
        out = self.decoder(self._encode_ct(ct), target_size)
        _check_finite('logits', out['logits'])
        out['pred'] = out['logits']
        out['aux'] = {}
        return out