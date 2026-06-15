import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_tensor(name, x):
    if torch.is_tensor(x) and not torch.isfinite(x).all():
        raise RuntimeError(f'[NaN/Inf] {name} contains invalid values')


def _check_tensor_list(name, xs):
    for i, x in enumerate(xs):
        _check_tensor(f'{name}[{i}]', x)


def _sanitize(x):
    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    return torch.clamp(x, -1e4, 1e4)


def _make_norm(channels, norm):
    norm = str(norm).lower()
    if norm == 'bn':
        return nn.BatchNorm2d(channels)
    if norm == 'gn':
        groups = min(32, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm == 'none':
        return nn.Identity()
    raise ValueError(f'Unsupported lapa_norm={norm}')


class _PriorProjector(nn.Module):
    def __init__(self, in_channels, out_channels, norm='gn'):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            _make_norm(out_channels, norm),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            _make_norm(out_channels, norm),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = _sanitize(x)
        x = self.proj(x)
        return _sanitize(x)


class LAPA(nn.Module):
    """Lesion-Aware Prior Adapter.

    Lite implementation with frozen/replaceable prior interface.
    """

    def __init__(self, ct_channels, prior_channels=None, norm='gn'):
        super().__init__()
        if prior_channels is None:
            prior_channels = ct_channels
        if len(ct_channels) != len(prior_channels):
            raise ValueError('ct_channels and prior_channels must have the same length.')
        self.projectors = nn.ModuleList([
            _PriorProjector(pin, cout, norm=norm) for pin, cout in zip(prior_channels, ct_channels)
        ])
        self.spatial_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c * 2, 1, kernel_size=3, padding=1, bias=True),
                nn.Sigmoid(),
            ) for c in ct_channels
        ])
        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in ct_channels])

    def forward(self, ct_feats, prior_feats, prior_gates=None):
        _check_tensor_list('ct_feats', ct_feats)
        _check_tensor_list('prior_feats', prior_feats)
        if prior_gates is None:
            prior_gates = [1.0] * len(ct_feats)
        enhanced = []
        for idx, (feat, prior, projector, gate, alpha, prior_gate) in enumerate(zip(
            ct_feats, prior_feats, self.projectors, self.spatial_gates, self.alpha, prior_gates
        )):
            prior = _sanitize(prior)
            prior_proj = projector(prior)
            if prior_proj.shape[-2:] != feat.shape[-2:]:
                prior_proj = F.interpolate(prior_proj, size=feat.shape[-2:], mode='bilinear', align_corners=False)
            prior_proj = _sanitize(prior_proj)
            _check_tensor(f'prior_proj[{idx}]', prior_proj)
            spatial_gate = gate(torch.cat([_sanitize(feat), prior_proj], dim=1))
            out = _sanitize(feat + alpha * prior_gate * spatial_gate * prior_proj)
            enhanced.append(out)
        _check_tensor_list('enhanced_ct_feats', enhanced)
        return enhanced
