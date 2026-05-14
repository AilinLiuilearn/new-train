# -*- coding: utf-8 -*-
"""
Single-modality student model for knowledge distillation.

Encoder: ConvNeXt family (atto / femto / pico / nano) via timm.
Decoder: Configurable — same AttentionUNetDecoder as the teacher,
         or a lighter LightConcatUNetDecoder.

Supported backbones and their feature channels:
    convnext_atto  : [40,  80, 160, 320]
    convnext_femto : [48,  96, 192, 384]
    convnext_pico  : [64, 128, 256, 512]
    convnext_nano  : [80, 160, 320, 640]
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.build_mdt_seg import (
    AttentionUNetDecoder,
    LightConcatUNetDecoder,
    ConvBNAct,
    load_local_weights_safe,
)


STUDENT_BACKBONES = {
    'convnext_atto':  (0, 1, 2, 3),
    'convnext_femto': (0, 1, 2, 3),
    'convnext_pico':  (0, 1, 2, 3),
    'convnext_nano':  (0, 1, 2, 3),
}

DECODER_PRESETS = {
    'convnext_atto':  (256, 128, 64, 32),
    'convnext_femto': (256, 128, 64, 32),
    'convnext_pico':  (384, 192, 96, 48),
    'convnext_nano':  (512, 256, 128, 64),
}

DECODER_PRESETS_LIGHT = {
    'convnext_atto':  (128, 64, 32, 16),
    'convnext_femto': (192, 96, 48, 24),
    'convnext_pico':  (256, 128, 64, 32),
    'convnext_nano':  (384, 192, 96, 48),
}


def _get_student_out_indices(backbone):
    if backbone in STUDENT_BACKBONES:
        return STUDENT_BACKBONES[backbone]
    raise ValueError(
        f'Unsupported student backbone: {backbone}. '
        f'Supported: {list(STUDENT_BACKBONES.keys())}'
    )


def create_student_backbone(backbone, in_channels=3):
    return timm.create_model(
        backbone,
        pretrained=False,
        features_only=True,
        out_indices=_get_student_out_indices(backbone),
        in_chans=in_channels,
    )


class SingleModalityStudent(nn.Module):
    """Single-modality (CT-only) student for knowledge distillation.

    Parameters
    ----------
    backbone : str
        One of convnext_atto / convnext_femto / convnext_pico / convnext_nano.
    pretrained_path : str or None
        Path to local pretrained weights for the encoder.
    in_channels : int
        Number of input channels (1 for raw CT, 3 for pseudo-RGB).
    out_channels : int
        Number of segmentation output channels (1 for binary).
    decoder_type : str
        'attention' → same AttentionUNetDecoder as teacher;
        'light'     → LightConcatUNetDecoder (lighter alternative).
    decoder_channels : tuple or None
        Override decoder channel widths.  If None, use preset defaults.
    """

    def __init__(
        self,
        backbone='convnext_pico',
        pretrained_path=None,
        in_channels=3,
        out_channels=1,
        decoder_type='attention',
        decoder_channels=None,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.decoder_type = decoder_type

        self.encoder = create_student_backbone(backbone, in_channels=in_channels)
        if pretrained_path:
            load_local_weights_safe(self.encoder, pretrained_path, name='Student_Encoder')

        enc_channels = self.encoder.feature_info.channels()

        if decoder_channels is None:
            if decoder_type == 'light':
                decoder_channels = DECODER_PRESETS_LIGHT.get(
                    backbone, (256, 128, 64, 32)
                )
            else:
                decoder_channels = DECODER_PRESETS.get(
                    backbone, (512, 256, 128, 64)
                )

        if decoder_type == 'attention':
            self.decoder = AttentionUNetDecoder(
                enc_channels, out_channels=out_channels,
                decoder_channels=decoder_channels,
            )
        elif decoder_type == 'light':
            self.decoder = LightConcatUNetDecoder(
                enc_channels, out_channels=out_channels,
                decoder_channels=decoder_channels,
            )
        else:
            raise ValueError(
                f'Unknown decoder_type={decoder_type}. '
                f'Supported: attention, light'
            )

    @staticmethod
    def _to_3ch(x):
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def forward(self, ct, target_size=None):
        if isinstance(target_size, torch.Tensor):
            raise TypeError(
                'SingleModalityStudent is CT-only. Use model(ct, target_size=...), '
                'do not pass PET as the second positional argument.'
            )
        ct = self._to_3ch(ct)
        features = self.encoder(ct)
        if target_size is None:
            target_size = ct.shape[-2:]
        outputs = self.decoder(features, target_size)
        return outputs

    def get_encoder_features(self, ct):
        """Return intermediate encoder features (for distillation)."""
        ct = self._to_3ch(ct)
        return self.encoder(ct)


def build_student_seg(config):
    """Build student segmentation model from config."""
    model = SingleModalityStudent(
        backbone=getattr(config, 'student_backbone', 'convnext_pico'),
        pretrained_path=getattr(config, 'student_pretrained_path', None),
        in_channels=3,
        out_channels=1,
        decoder_type=getattr(config, 'student_decoder_type', 'attention'),
        decoder_channels=getattr(config, 'student_decoder_channels', None),
    )
    return dict(model=model)
