# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FPA(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.master = ConvBNReLU(in_ch, in_ch, k=1, p=0)
        self.b3 = ConvBNReLU(in_ch, in_ch, k=3, p=1)
        self.b5 = ConvBNReLU(in_ch, in_ch, k=5, p=2)
        self.b7 = ConvBNReLU(in_ch, in_ch, k=7, p=3)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.out = ConvBNReLU(in_ch, in_ch, k=1, p=0)

    def forward(self, x):
        m = self.master(x)
        attn = self.b3(x) + self.b5(x) + self.b7(x)
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=x.shape[-2:], mode='bilinear', align_corners=False)
        y = m * torch.sigmoid(attn) + gp
        return self.out(y)


class GAU(nn.Module):
    def __init__(self, low_ch, high_ch):
        super().__init__()
        self.low_proj = ConvBNReLU(low_ch, low_ch, k=3, p=1)
        self.high_proj = ConvBNReLU(high_ch, low_ch, k=1, p=0)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_ch, low_ch, 1, bias=False),
            nn.Sigmoid(),
        )
        self.out = ConvBNReLU(low_ch, low_ch, k=3, p=1)

    def forward(self, low_feat, high_feat):
        low_enhanced = self.low_proj(low_feat) * self.attn(high_feat)
        high_up = F.interpolate(self.high_proj(high_feat), size=low_feat.shape[-2:], mode='bilinear', align_corners=False)
        return self.out(low_enhanced + high_up)


class ResNetPAN(nn.Module):
    def __init__(self, backbone_name='resnet34', pretrained=True, pretrained_path=None, in_channels=2, out_channels=1):
        super().__init__()
        self.in_channels = in_channels

        if backbone_name == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if (pretrained and not pretrained_path) else None
            backbone = resnet18(weights=weights)
            c5 = 512
        else:
            weights = ResNet34_Weights.IMAGENET1K_V1 if (pretrained and not pretrained_path) else None
            backbone = resnet34(weights=weights)
            c5 = 512

        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location='cpu')
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            elif isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
                ckpt = ckpt['model']
            if isinstance(ckpt, dict):
                ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
                backbone.load_state_dict(ckpt, strict=False)
                print(f'[Backbone init] loaded {backbone_name} weights from: {pretrained_path}')

        old_conv = backbone.conv1
        new_conv = nn.Conv2d(in_channels, old_conv.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            mean_w = old_conv.weight.mean(dim=1, keepdim=True)
            for c in range(in_channels):
                new_conv.weight[:, c:c+1] = mean_w
        backbone.conv1 = new_conv

        self.backbone = backbone
        self.fpa = FPA(c5)
        self.gau4 = GAU(low_ch=256, high_ch=512)
        self.gau3 = GAU(low_ch=128, high_ch=256)
        self.gau2 = GAU(low_ch=64, high_ch=128)

        self.seg_head = nn.Sequential(
            ConvBNReLU(64, 64, k=3, p=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, out_channels, 1),
        )

    def encode(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        p2 = self.backbone.layer1(x)
        p3 = self.backbone.layer2(p2)
        p4 = self.backbone.layer3(p3)
        p5 = self.backbone.layer4(p4)
        return p2, p3, p4, p5

    def forward(self, ct, pet=None, target_size=None):
        if self.in_channels == 1:
            x = ct
        else:
            if pet is None:
                raise ValueError('PET input is required when in_channels=2')
            x = torch.cat([ct, pet], dim=1)

        p2, p3, p4, p5 = self.encode(x)
        p5 = self.fpa(p5)
        d4 = self.gau4(p4, p5)
        d3 = self.gau3(p3, d4)
        d2 = self.gau2(p2, d3)

        logit = self.seg_head(d2)
        out_size = target_size if target_size is not None else ct.shape[-2:]
        return F.interpolate(logit, size=out_size, mode='bilinear', align_corners=False)


# backward-compatible alias
ResNet34PAN = ResNetPAN
