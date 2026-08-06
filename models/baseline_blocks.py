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


class UNetStyleDecoder(nn.Module):
    def __init__(self, encoder_channels=(64, 128, 320, 512), decoder_channels=(512, 256, 128, 64), out_channels=1, use_deep_supervision=False):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        d4, d3, d2, d1 = decoder_channels
        self.use_deep_supervision = bool(use_deep_supervision)
        self.proj4 = ConvBNAct(c4, d4, kernel_size=1)
        self.proj3 = ConvBNAct(c3, d3, kernel_size=1)
        self.proj2 = ConvBNAct(c2, d2, kernel_size=1)
        self.proj1 = ConvBNAct(c1, d1, kernel_size=1)
        self.fuse3 = nn.Sequential(ConvBNAct(d4 + d3, d3, kernel_size=3), ConvBNAct(d3, d3, kernel_size=3))
        self.fuse2 = nn.Sequential(ConvBNAct(d3 + d2, d2, kernel_size=3), ConvBNAct(d2, d2, kernel_size=3))
        self.fuse1 = nn.Sequential(ConvBNAct(d2 + d1, d1, kernel_size=3), ConvBNAct(d1, d1, kernel_size=3))
        self.seg_head = nn.Conv2d(d1, out_channels, kernel_size=1)
        if self.use_deep_supervision:
            self.aux_head_d2 = nn.Conv2d(d2, out_channels, kernel_size=1)
            self.aux_head_d3 = nn.Conv2d(d3, out_channels, kernel_size=1)
            self.aux_head_d4 = nn.Conv2d(d4, out_channels, kernel_size=1)

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


class StateAwareWeightedAddFusion(nn.Module):
    def __init__(self, num_scales=4):
        super().__init__()
        self.num_scales = int(num_scales)
        self.raw_alpha_full = nn.Parameter(torch.zeros(self.num_scales))
        self.raw_alpha_missing = nn.Parameter(
            torch.full((self.num_scales,), -2.944)
        )

    @property
    def alpha_full(self):
        return 2.0 * torch.sigmoid(self.raw_alpha_full)

    @property
    def alpha_missing(self):
        return 2.0 * torch.sigmoid(self.raw_alpha_missing)

    def forward(self, ct_feats, pet_feats, mode, pet_available=None):
        if len(ct_feats) != self.num_scales or len(pet_feats) != self.num_scales:
            raise ValueError(
                f'Expected {self.num_scales} CT/PET scales, got '
                f'{len(ct_feats)} and {len(pet_feats)}'
            )
        if mode == 'full':
            alpha = self.alpha_full
        elif mode == 'missing':
            alpha = self.alpha_missing
        elif mode == 'auto':
            if pet_available is None:
                raise ValueError('pet_available is required for auto fusion')
            pet_available = pet_available.view(-1, 1, 1, 1)
            alpha = None
        else:
            raise ValueError(f'Unsupported fusion mode: {mode}')

        fused = []
        for scale_idx, (ct_feat, pet_feat) in enumerate(zip(ct_feats, pet_feats)):
            if pet_feat.shape[-2:] != ct_feat.shape[-2:]:
                pet_feat = F.interpolate(
                    pet_feat,
                    size=ct_feat.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            if mode == 'auto':
                availability = pet_available.to(device=ct_feat.device, dtype=ct_feat.dtype)
                full_weight = self.alpha_full[scale_idx].to(device=ct_feat.device, dtype=ct_feat.dtype)
                missing_weight = self.alpha_missing[scale_idx].to(device=ct_feat.device, dtype=ct_feat.dtype)
                weight = availability * full_weight + (1.0 - availability) * missing_weight
            else:
                weight = alpha[scale_idx].to(device=ct_feat.device, dtype=ct_feat.dtype)
            fused.append(_sanitize(ct_feat + weight * pet_feat))
        return fused
