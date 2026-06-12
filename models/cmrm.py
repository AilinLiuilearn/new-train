import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelWeights(nn.Module):
    """KTB Feature Rectify Module channel branch."""

    def __init__(self, dim, reduction=1):
        super().__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        b, _, _, _ = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(b, self.dim * 2)
        max_ = self.max_pool(x).view(b, self.dim * 2)
        y = torch.cat((avg, max_), dim=1)
        y = self.mlp(y).view(b, self.dim * 2, 1)
        channel_weights = y.reshape(b, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)
        return channel_weights


class SpatialWeights(nn.Module):
    """KTB Feature Rectify Module spatial branch."""

    def __init__(self, dim, reduction=1):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        b, _, h, w = x1.shape
        x = torch.cat((x1, x2), dim=1)
        spatial_weights = self.mlp(x).reshape(b, 2, 1, h, w).permute(1, 0, 2, 3, 4)
        return spatial_weights


class FeatureRectifyModule(nn.Module):
    """KTB FRM: channel and spatial mutual rectification."""

    def __init__(self, dim, reduction=1, lambda_c=0.5, lambda_s=0.5):
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1, out_x2


class CMRM(nn.Module):
    """
    Cross-Modal Rectification Module.

    It aligns heterogeneous CT/PET encoder channels first, then reuses KTB FRM
    to perform channel-spatial mutual rectification.
    """

    def __init__(self, ct_channels, pet_channels, out_channels, reduction=1, lambda_c=0.5, lambda_s=0.5):
        super().__init__()
        self.ct_proj = nn.Sequential(
            nn.Conv2d(ct_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pet_proj = nn.Sequential(
            nn.Conv2d(pet_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.frm = FeatureRectifyModule(out_channels, reduction=reduction, lambda_c=lambda_c, lambda_s=lambda_s)

    def forward(self, ct_feat, pet_feat):
        ct = self.ct_proj(ct_feat)
        pet = self.pet_proj(pet_feat)
        if pet.shape[-2:] != ct.shape[-2:]:
            pet = F.interpolate(pet, size=ct.shape[-2:], mode='bilinear', align_corners=False)
        return self.frm(ct, pet)


class CMRMSumFusion(nn.Module):
    """Ablation-friendly fusion: CMRM rectification followed by summation."""

    def __init__(self, ct_channels, pet_channels, out_channels, reduction=1, lambda_c=0.5, lambda_s=0.5):
        super().__init__()
        if not (len(ct_channels) == len(pet_channels) == len(out_channels) == 4):
            raise ValueError('CMRMSumFusion expects 4 feature stages.')
        self.blocks = nn.ModuleList([
            CMRM(ct_ch, pet_ch, out_ch, reduction=reduction, lambda_c=lambda_c, lambda_s=lambda_s)
            for ct_ch, pet_ch, out_ch in zip(ct_channels, pet_channels, out_channels)
        ])

    def forward(self, ct_feats, pet_feats):
        fused = []
        for block, ct_feat, pet_feat in zip(self.blocks, ct_feats, pet_feats):
            ct_r, pet_r = block(ct_feat, pet_feat)
            fused.append(ct_r + pet_r)
        return fused
