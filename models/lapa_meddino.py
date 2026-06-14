import torch
import torch.nn as nn
import torch.nn.functional as F


class _PriorProjector(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(x)


class LAPA(nn.Module):
    """Lesion-Aware Prior Adapter.

    Lite implementation with frozen/replaceable prior interface.
    """

    def __init__(self, ct_channels, prior_channels=None):
        super().__init__()
        if prior_channels is None:
            prior_channels = ct_channels
        if len(ct_channels) != len(prior_channels):
            raise ValueError('ct_channels and prior_channels must have the same length.')
        self.projectors = nn.ModuleList([
            _PriorProjector(pin, cout) for pin, cout in zip(prior_channels, ct_channels)
        ])
        self.spatial_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c * 2, 1, kernel_size=3, padding=1, bias=True),
                nn.Sigmoid(),
            ) for c in ct_channels
        ])
        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in ct_channels])

    def forward(self, ct_feats, prior_feats, prior_gates=None):
        if prior_gates is None:
            prior_gates = [1.0] * len(ct_feats)
        enhanced = []
        for feat, prior, projector, gate, alpha, prior_gate in zip(
            ct_feats, prior_feats, self.projectors, self.spatial_gates, self.alpha, prior_gates
        ):
            prior_proj = projector(prior)
            if prior_proj.shape[-2:] != feat.shape[-2:]:
                prior_proj = F.interpolate(prior_proj, size=feat.shape[-2:], mode='bilinear', align_corners=False)
            spatial_gate = gate(torch.cat([feat, prior_proj], dim=1))
            enhanced.append(feat + alpha * prior_gate * spatial_gate * prior_proj)
        return enhanced
