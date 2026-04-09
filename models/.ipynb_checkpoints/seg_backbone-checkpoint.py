# -*- coding: utf-8 -*-
"""
EfficientNet-B2 骨干 + FPN 解码器（单模态分割用）。
与用户提供的 SegModel 一致：1→3 通道适配、timm 预训练、FPN 上采样到原图尺寸。
用于 MDT 的 extractor 时只使用 encoder 部分，输出多尺度或单尺度特征。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FPNDecoder(nn.Module):
    """FPN 风格解码器，上采样到 target_size 后 1×1 分割头"""

    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for in_ch in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in in_channels_list
        ])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.seg_head = nn.Conv2d(out_channels, 1, 1)

    def forward(self, x_list, target_size):
        lateral_outs = [conv(x) for conv, x in zip(self.lateral_convs, x_list)]
        fpn_outs = [lateral_outs[-1]]
        for lat in reversed(lateral_outs[:-1]):
            up_feat = self.upsample(fpn_outs[-1])
            if up_feat.shape[-2:] != lat.shape[-2:]:
                up_feat = F.interpolate(up_feat, size=lat.shape[-2:], mode='bilinear', align_corners=False)
            fpn_outs.append(up_feat + lat)
        fpn_outs = list(reversed(fpn_outs))
        fpn_outs = [conv(x) for conv, x in zip(self.fpn_convs, fpn_outs)]
        out = fpn_outs[0]
        if target_size is not None:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        return self.seg_head(out)


class SegEncoderEfficientB2(nn.Module):
    """
    单模态编码器：EfficientNet-B2 (features_only) + 1→3 通道适配。
    输出多尺度特征列表，供 FPN 或 MDT 使用。
    """

    def __init__(self, backbone_name='efficientnet_b2', out_indices=(0, 1, 2, 3), pretrained=True):
        super().__init__()
        self.channel_adapt = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        self.out_indices = out_indices
        if 'efficientnetv2' in backbone_name.lower():
            pretrained = False
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=True,
            # checkpoint_path='/root/autodl-tmp/mkd-main/new-train/model.safetensors',
            features_only=True,
            out_indices=out_indices,
        )
        # self.encoder.load_state_dict(torch.load('/root/autodl-tmp/mkd-main/new-train/model.safetensors'), strict=False)
        self._channels = self.encoder.feature_info.channels()

    @property
    def out_channels(self):
        """最后一层通道数（用于 MDT 单尺度）"""
        return self._channels[-1]

    @property
    def feature_channels(self):
        return self._channels

    def forward(self, x, return_list=True):
        """
        x: (B, 1, H, W)
        return_list=True: 返回多尺度列表；False: 只返回最后一层 (B, C, h, w)
        """
        x = self.channel_adapt(x)
        feats = self.encoder(x)
        if return_list:
            return feats
        return feats[-1]


class SegModel(nn.Module):
    """单模态 CT 分割完整模型：EfficientB2 + FPN → 1×H×W mask（与用户提供一致）"""

    def __init__(self, backbone_name='efficientnet_b2'):
        super().__init__()
        self.encoder_module = SegEncoderEfficientB2(backbone_name=backbone_name, pretrained=True)
        self.decoder = FPNDecoder(self.encoder_module.feature_channels)

    def forward(self, x):
        original_size = x.shape[-2:]
        feats = self.encoder_module(x, return_list=True)
        return self.decoder(feats, target_size=original_size)
