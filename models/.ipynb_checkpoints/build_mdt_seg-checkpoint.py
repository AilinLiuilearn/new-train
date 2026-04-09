# -*- coding: utf-8 -*-
"""
MDT 分割网络构建：EfficientB2 双流 + 解耦（projector / encoder_general / encoder_mri / encoder_pet / decoder）+ 分割头。
保留 mkd 的解耦结构，骨干换为 EfficientNet-B2 单尺度特征图。
"""

import torch
import torch.nn as nn
from models.seg_backbone import SegEncoderEfficientB2


def build_mdt_seg_teacher(config):
    """
    构建 MDT 教师：extractor_ct, extractor_pet（共享结构的 EfficientB2），
    projector_mri/projector_pet, encoder_general, encoder_mri, encoder_pet,
    decoder_mri, decoder_pet, segmentor.
    """
    hidden = getattr(config, 'hidden', 256)
    add_type = getattr(config, 'add_type', 'add')
    use_projector = getattr(config, 'use_projector', True)
    use_specific = getattr(config, 'use_specific', True)
    backbone_name = getattr(config, 'backbone', 'efficientnet_b2')

    # 单尺度特征（最后一层），用于解耦
    extractor_ct = SegEncoderEfficientB2(backbone_name=backbone_name, out_indices=(3,), pretrained=True)
    extractor_pet = SegEncoderEfficientB2(backbone_name=backbone_name, out_indices=(3,), pretrained=True)
    C = extractor_ct.out_channels  # 352 for eff_b2

    if use_projector:
        projector_mri = nn.Sequential(
            nn.Conv2d(C, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        projector_pet = nn.Sequential(
            nn.Conv2d(C, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        in_ch = hidden
    else:
        projector_mri = projector_pet = None
        in_ch = C

    half = in_ch // 2
    encoder_general = nn.Sequential(
        nn.Conv2d(in_ch, half, 1),
        nn.BatchNorm2d(half),
        nn.ReLU(inplace=True),
    )
    encoder_mri = nn.Sequential(
        nn.Conv2d(in_ch, half, 1),
        nn.BatchNorm2d(half),
        nn.ReLU(inplace=True),
    )
    encoder_pet = nn.Sequential(
        nn.Conv2d(in_ch, half, 1),
        nn.BatchNorm2d(half),
        nn.ReLU(inplace=True),
    )

    decoder_mri = nn.Sequential(
        nn.Conv2d(half, in_ch, 1),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
    )
    decoder_pet = nn.Sequential(
        nn.Conv2d(half, in_ch, 1),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
    )

    if use_specific:
        seg_in = half * 2  # z_general_ct + z_general_pet + z_mri + z_pet -> 2*half
    else:
        seg_in = half  # z_general_ct + z_general_pet -> 1*half (add)
    segmentor = nn.Sequential(
        nn.Conv2d(seg_in, half, 1),
        nn.BatchNorm2d(half),
        nn.ReLU(inplace=True),
        nn.Conv2d(half, 1, 1),
    )

    networks = dict(
        extractor_mri=extractor_ct,
        extractor_pet=extractor_pet,
        projector_mri=projector_mri,
        projector_pet=projector_pet,
        encoder_general=encoder_general,
        encoder_mri=encoder_mri,
        encoder_pet=encoder_pet,
        decoder_mri=decoder_mri,
        decoder_pet=decoder_pet,
        segmentor=segmentor,
    )
    return networks
