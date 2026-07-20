import torch
import torch.nn as nn
import torch.nn.functional as F

from models.build_mdt_seg import ConvBNAct


class AddFusion(nn.Module):
    def forward(self, ct_feats, pet_feats):
        return [c + p for c, p in zip(ct_feats, pet_feats)]


class SharedDecoder(nn.Module):
    def __init__(self, channels, out_channels=1):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.up4 = ConvBNAct(c4, c3)
        self.up3 = ConvBNAct(c3, c2)
        self.up2 = ConvBNAct(c2, c1)
        self.out = nn.Conv2d(c1, out_channels, 1)

    def forward(self, feats, target_size):
        x = feats[-1]
        for skip, block in zip(reversed(feats[:-1]), [self.up4, self.up3, self.up2]):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = block(x + skip)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return self.out(x)
