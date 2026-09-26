import torch
import torch.nn as nn
import torch.nn.functional as F

from models.build_mdt_seg import ConvBNAct


def _check_tensor(name, x):
    if torch.is_tensor(x) and not torch.isfinite(x).all():
        raise RuntimeError(f'[NaN/Inf] {name} contains invalid values')


def _check_tensor_list(name, xs):
    for i, x in enumerate(xs):
        _check_tensor(f'{name}[{i}]', x)


def _sanitize(x):
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


def _norm_layer(channels, norm_type='group'):
    if norm_type == 'group':
        # Largest power-of-two group count (>=2) dividing the channels; GroupNorm
        # normalizes per-sample so Full/Missing rows are decoupled.
        groups = 8
        while groups > 1 and channels % groups != 0:
            groups //= 2
        return nn.GroupNorm(groups, channels)
    raise ValueError(f'Unsupported decoder norm_type: {norm_type!r}')


def _conv_group_act(in_channels, out_channels, kernel_size=1, stride=1, dilation=1):
    """Conv + GroupNorm + ReLU, used only for norm_type='group'."""
    padding = (kernel_size // 2) * dilation
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
        _norm_layer(out_channels, 'group'),
        nn.ReLU(inplace=True),
    )


def _decoder_block(in_channels, out_channels, norm_type, kernel_size=1, stride=1, dilation=1):
    # norm_type='bn' keeps the original ConvBNAct (with its .block wrapper) so
    # the module hierarchy, parameter keys and compute order are identical to
    # the old baseline: old checkpoints load with strict=True.
    if norm_type == 'bn':
        return ConvBNAct(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)
    if norm_type == 'group':
        return _conv_group_act(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)
    raise ValueError(f'Unsupported decoder norm_type: {norm_type!r}')


class UNetStyleDecoder(nn.Module):
    def __init__(self, encoder_channels=(64, 128, 320, 512), decoder_channels=(512, 256, 128, 64), out_channels=1, use_deep_supervision=False, norm_type='bn'):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        d4, d3, d2, d1 = decoder_channels
        self.use_deep_supervision = bool(use_deep_supervision)
        self.norm_type = norm_type
        self.proj4 = _decoder_block(c4, d4, norm_type, kernel_size=1)
        self.proj3 = _decoder_block(c3, d3, norm_type, kernel_size=1)
        self.proj2 = _decoder_block(c2, d2, norm_type, kernel_size=1)
        self.proj1 = _decoder_block(c1, d1, norm_type, kernel_size=1)
        self.fuse3 = nn.Sequential(_decoder_block(d4 + d3, d3, norm_type, kernel_size=3), _decoder_block(d3, d3, norm_type, kernel_size=3))
        self.fuse2 = nn.Sequential(_decoder_block(d3 + d2, d2, norm_type, kernel_size=3), _decoder_block(d2, d2, norm_type, kernel_size=3))
        self.fuse1 = nn.Sequential(_decoder_block(d2 + d1, d1, norm_type, kernel_size=3), _decoder_block(d1, d1, norm_type, kernel_size=3))
        self.seg_head = nn.Conv2d(d1, out_channels, kernel_size=1)
        if self.use_deep_supervision:
            self.aux_head_d2 = nn.Conv2d(d2, out_channels, kernel_size=1)
            self.aux_head_d3 = nn.Conv2d(d3, out_channels, kernel_size=1)
            self.aux_head_d4 = nn.Conv2d(d4, out_channels, kernel_size=1)
        if norm_type != 'bn':
            print(f'[UNetStyleDecoder] norm_type={norm_type} (per-sample; decouples Full/Missing rows)')

    def forward(self, features, target_size):
        x1, x2, x3, x4 = features
        d4 = self.proj4(x4)
        s3 = self.proj3(x3)
        d3 = self.fuse3(torch.cat([F.interpolate(d4, size=s3.shape[-2:], mode='bilinear', align_corners=False), s3], dim=1))
        s2 = self.proj2(x2)
        d2 = self.fuse2(torch.cat([F.interpolate(d3, size=s2.shape[-2:], mode='bilinear', align_corners=False), s2], dim=1))
        s1 = self.proj1(x1)
        d1 = self.fuse1(torch.cat([F.interpolate(d2, size=s1.shape[-2:], mode='bilinear', align_corners=False), s1], dim=1))
        logits = self.seg_head(d1)
        final_logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)
        if not self.use_deep_supervision:
            return {'logits': final_logits}
        return {'logits': final_logits, 'aux_logits': [self.aux_head_d2(d2), self.aux_head_d3(d3), self.aux_head_d4(d4)]}


class AddFusion(nn.Module):
    def forward(self, ct_feats, pet_feats, pet_available=None):
        fused = []
        for ct_feat, pet_feat in zip(ct_feats, pet_feats):
            if pet_feat.shape[-2:] != ct_feat.shape[-2:]:
                pet_feat = F.interpolate(pet_feat, size=ct_feat.shape[-2:], mode='bilinear', align_corners=False)
            fused.append(_sanitize(ct_feat + pet_feat))
        return fused
