# -*- coding: utf-8 -*-
"""
EfficientNet / MiT 系列骨干 + FPN 解码器（单模态分割用）。
支持 efficientnet_b2/b3/b4/b5、tf_efficientnetv2_s（timm）；mit_b0/mit_b1/mit_b2（HuggingFace SegformerModel），1→3 通道适配。
用于 MDT 的 extractor 时只使用 encoder 部分，输出多尺度或单尺度特征。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# HuggingFace MiT 配置（mit_b* 通过 transformers 的 SegformerModel 加载）
_MIT_CONFIGS = {
    'mit_b0': {'hf': 'nvidia/mit-b0', 'out_chs': [32, 64, 160, 256]},
    'mit_b1': {'hf': 'nvidia/mit-b1', 'out_chs': [64, 128, 320, 512]},
    'mit_b2': {'hf': 'nvidia/mit-b2', 'out_chs': [64, 128, 320, 512]},
    'mit_b3': {'hf': 'nvidia/mit-b3', 'out_chs': [64, 128, 320, 512]},
    'mit_b4': {'hf': 'nvidia/mit-b4', 'out_chs': [64, 128, 320, 512]},
    'mit_b5': {'hf': 'nvidia/mit-b5', 'out_chs': [64, 128, 320, 512]},
}

# 离线用 SegformerConfig 参数（pretrained=False 时避免从 HuggingFace 下载）
_MIT_SEGFORMER_CONFIGS = {
    'mit_b0': {'hidden_sizes': [32, 64, 160, 256], 'num_attention_heads': [1, 2, 5, 8]},
    'mit_b1': {'hidden_sizes': [64, 128, 320, 512], 'num_attention_heads': [1, 2, 5, 8]},
    'mit_b2': {'hidden_sizes': [64, 128, 320, 512], 'num_attention_heads': [1, 2, 5, 8]},
    'mit_b3': {'hidden_sizes': [64, 128, 320, 512], 'num_attention_heads': [1, 2, 5, 8]},
    'mit_b4': {'hidden_sizes': [64, 128, 320, 512], 'num_attention_heads': [1, 2, 5, 8]},
    'mit_b5': {'hidden_sizes': [64, 128, 320, 512], 'num_attention_heads': [1, 2, 5, 8]},
}


class _HFMiTEncoder(nn.Module):
    """HuggingFace SegFormer (MiT) encoder，输出 4 个尺度特征列表"""

    def __init__(self, hf_name: str, pretrained: bool = True, local_path: str = None, mit_name: str = None):
        super().__init__()
        try:
            from transformers import SegformerModel, SegformerConfig
        except ImportError:
            raise ImportError("使用 mit_b* 需安装 transformers: pip install transformers")
        load_path = local_path if (local_path and os.path.isdir(local_path)) else hf_name
        if pretrained and (local_path and os.path.isdir(local_path)):
            self.model = SegformerModel.from_pretrained(load_path)
        elif pretrained:
            try:
                self.model = SegformerModel.from_pretrained(load_path)
            except Exception as e:
                if mit_name and mit_name in _MIT_SEGFORMER_CONFIGS:
                    print(f"[seg_backbone] HuggingFace 下载失败 ({e})，使用本地 config 初始化 mit_b*")
                    cfg = SegformerConfig(**_MIT_SEGFORMER_CONFIGS[mit_name])
                    self.model = SegformerModel(cfg)
                else:
                    raise
        else:
            if mit_name and mit_name in _MIT_SEGFORMER_CONFIGS:
                cfg = SegformerConfig(**_MIT_SEGFORMER_CONFIGS[mit_name])
                self.model = SegformerModel(cfg)
            else:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(hf_name)
                self.model = SegformerModel(config)
        self.model.config.output_hidden_states = True

    def forward(self, x):
        out = self.model(pixel_values=x, return_dict=True)
        hs = out.hidden_states
        feats = list(hs[-4:])
        return feats


def _load_pretrained_strict_false(model, checkpoint_path):
    """从本地 checkpoint 加载预训练，strict=False 以兼容 features_only 与完整分类器权重的差异"""
    if str(checkpoint_path).endswith('.safetensors'):
        try:
            from safetensors.torch import load_file
            ckpt = load_file(checkpoint_path, device='cpu')
        except ImportError:
            raise ImportError('pip install safetensors 以加载 .safetensors')
    else:
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']
    incompatible = model.load_state_dict(ckpt, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        pass  # 预期：classifier/bn2 等键在 features_only 中不存在


class FPNDecoder(nn.Module):
    """
    标准 FPN 结构分割解码器（Semantic FPN 风格）。
    结构：Lateral 1×1 → Top-down 融合 → 每层 3×3 平滑 → 多尺度融合 → 分割头。
    输入：x_list = [C2, C3, C4, C5] 多尺度特征。
    """

    def __init__(self, in_channels_list, out_channels=256, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()
        # Lateral: 1×1 将各层统一到 out_channels
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for in_ch in in_channels_list
        ])
        # Top-down 后的 3×3 平滑（FPN 标准操作）
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in in_channels_list
        ])
        # 多尺度融合：每层经 seg_block 后上采样到最高分辨率相加（Semantic FPN）
        self.seg_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in in_channels_list
        ])
        self.seg_head = nn.Conv2d(out_channels, 1, 1)

    def forward(self, x_list, target_size):
        # Lateral connections
        lateral_outs = [conv(x) for conv, x in zip(self.lateral_convs, x_list)]
        # Top-down pathway: 从最深层开始，逐级上采样并融合
        fpn_outs = [lateral_outs[-1]]
        for lat in reversed(lateral_outs[:-1]):
            up_feat = F.interpolate(fpn_outs[-1], size=lat.shape[-2:], mode='bilinear', align_corners=False)
            fpn_outs.append(up_feat + lat)
        fpn_outs = list(reversed(fpn_outs))
        # FPN 每层 3×3 平滑
        fpn_outs = [conv(x) for conv, x in zip(self.fpn_convs, fpn_outs)]
        # 多尺度融合：每层经 seg_block 后上采样到最高分辨率相加
        h_ref = fpn_outs[0].shape[-2:]
        fused = None
        for i, feat in enumerate(fpn_outs):
            pred = self.seg_blocks[i](feat)
            if pred.shape[-2:] != h_ref:
                pred = F.interpolate(pred, size=h_ref, mode='bilinear', align_corners=False)
            fused = pred if fused is None else fused + pred
        if target_size is not None and fused.shape[-2:] != target_size:
            fused = F.interpolate(fused, size=target_size, mode='bilinear', align_corners=False)
        fused = self.dropout(fused)
        return self.seg_head(fused)


class UNetDecoder(nn.Module):
    """
    UNet 风格分割解码器（与 FPN 同接口，替换教师 FPN 头）。
    自瓶颈（最深层）逐级双线性上采样，与编码器金字塔的浅层特征 concat 后做双卷积，
    形成经典 U 形跳跃连接；最后 1×1 输出 logits。
    输入 x_list = [C2, C3, C4, C5] 与 FPNDecoder 一致：索引 0 分辨率最高，3 为瓶颈。
    """

    def __init__(self, in_channels_list, out_channels=256, dropout=0.0):
        super().__init__()
        if len(in_channels_list) != 4:
            raise ValueError('UNetDecoder 需要四尺度 in_channels_list')
        self.dropout = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for in_ch in in_channels_list
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.up_blocks = nn.ModuleList([
            self._make_up_block(out_channels * 2, out_channels) for _ in range(3)
        ])
        self.seg_head = nn.Conv2d(out_channels, 1, 1)

    @staticmethod
    def _make_up_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_list, target_size):
        lat = [conv(x) for conv, x in zip(self.lateral_convs, x_list)]
        x = self.bottleneck(lat[3])
        for step, idx in enumerate((2, 1, 0)):
            x = F.interpolate(
                x, size=x_list[idx].shape[-2:], mode='bilinear', align_corners=False
            )
            x = torch.cat([x, lat[idx]], dim=1)
            x = self.up_blocks[step](x)
        if target_size is not None and x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = self.dropout(x)
        return self.seg_head(x)


class SegFormerMLPDecoder(nn.Module):
    """
    SegFormer 类分割头（与 MiT/SegFormer 常用设定一致）：
    四尺度各 1×1 嵌入到统一通道 → 全部上采样到最高分辨率 → concat → 融合卷积 → 1×1 logits。
    比纯 UNet 拼接路径更轻、与 Transformer 编码器搭配更常见。
    接口与 FPNDecoder/UNetDecoder 相同。
    """

    def __init__(self, in_channels_list, out_channels=256, dropout=0.0):
        super().__init__()
        if len(in_channels_list) != 4:
            raise ValueError('SegFormerMLPDecoder 需要四尺度 in_channels_list')
        self.embed = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for c in in_channels_list
        ])
        fuse_in = out_channels * 4
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()
        self.seg_head = nn.Conv2d(out_channels, 1, 1)

    def forward(self, x_list, target_size):
        h0, w0 = x_list[0].shape[-2:]
        feats = []
        for i, x in enumerate(x_list):
            e = self.embed[i](x)
            if e.shape[-2:] != (h0, w0):
                e = F.interpolate(e, size=(h0, w0), mode='bilinear', align_corners=False)
            feats.append(e)
        x = torch.cat(feats, dim=1)
        x = self.fuse(x)
        if target_size is not None and x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = self.dropout(x)
        return self.seg_head(x)


class SegEncoderEfficientB2(nn.Module):
    """
    单模态编码器：EfficientNet / MiT (features_only) + 1→3 通道适配。
    输出多尺度特征列表，供 FPN 或 MDT 使用。
    支持 backbone: efficientnet_b2/b3/b4/b5、tf_efficientnetv2_s（timm）；mit_b0/mit_b1/mit_b2（HuggingFace）。
    """

    def __init__(self, backbone_name='efficientnet_b2', out_indices=(0, 1, 2, 3), pretrained=True, pretrained_path=None):
        super().__init__()
        self.channel_adapt = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        self.out_indices = out_indices
        self._use_hf_mit = backbone_name in _MIT_CONFIGS

        if self._use_hf_mit:
            cfg = _MIT_CONFIGS[backbone_name]
            local_path = pretrained_path if (pretrained_path and os.path.isdir(pretrained_path)) else None
            self.encoder = _HFMiTEncoder(cfg['hf'], pretrained=pretrained, local_path=local_path, mit_name=backbone_name)
            self._channels = cfg['out_chs']
            if local_path:
                print(f"[seg_backbone] 已从本地加载 MiT: {local_path}")
            elif pretrained:
                print(f"[seg_backbone] 已从 HuggingFace 加载 MiT: {cfg['hf']}")
            if pretrained_path and os.path.isfile(pretrained_path):
                _load_pretrained_strict_false(self.encoder, pretrained_path)
                print(f"[seg_backbone] 已从本地覆盖预训练: {pretrained_path}")
        else:
            kwargs = dict(features_only=True, out_indices=out_indices)
            if pretrained and pretrained_path and os.path.isfile(pretrained_path):
                self.encoder = timm.create_model(backbone_name, pretrained=False, **kwargs)
                _load_pretrained_strict_false(self.encoder, pretrained_path)
                print(f"[seg_backbone] 已从本地加载预训练: {pretrained_path}")
            else:
                self.encoder = timm.create_model(backbone_name, pretrained=pretrained, **kwargs)
            fi = getattr(self.encoder, 'feature_info', None)
            if fi is not None:
                self._channels = fi.channels()
            else:
                with torch.no_grad():
                    dummy = torch.randn(1, 3, 64, 64)
                    out = self.encoder(dummy)
                self._channels = [o.shape[1] for o in out] if isinstance(out, (list, tuple)) else [out.shape[1]]

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
