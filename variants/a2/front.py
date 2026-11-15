"""A2: paired deep plus same-scale conditioning, otherwise original A.

Conceptual inspiration: FreqFusion (Chen et al., TPAMI 2024),
https://github.com/Linwei-Chen/FreqFusion/blob/3fb0c70637a3c194fb74294d3ce4681958b26241/FreqFusion.py.
Independent pure-PyTorch formulation; no source code or CARAFE copied.
"""
import torch
from torch import nn
from torch.nn import functional as F


class PreModule(nn.Module):
    def __init__(self, text_features, channels=(64, 128, 320, 512)):
        super().__init__()
        text = torch.as_tensor(text_features, dtype=torch.float32).detach().clone()
        if text.shape != (2, 512):
            raise ValueError('text_features must have shape (2, 512)')
        if not torch.isfinite(text).all() or (text.norm(dim=1) == 0).any():
            raise ValueError('text_features must be finite nonzero vectors')
        self.register_buffer('text_features', F.normalize(text, dim=1))
        self.channels = tuple(channels)
        if len(self.channels) != 4:
            raise ValueError('four scales required')
        self.shallow = nn.ModuleList(nn.ModuleList(nn.Conv2d(self.channels[s], 16, 1, bias=False) for _ in range(2)) for s in range(3))
        self.deep = nn.ModuleList(nn.ModuleList(nn.Conv2d(self.channels[s + 1], 16, 1, bias=False) for _ in range(2)) for s in range(3))
        self.text_map = nn.ModuleList(nn.Linear(512, 16) for _ in range(3))
        self.joint = nn.ModuleList(nn.Conv2d(32, 16, 1) for _ in range(3))
        self.kernel = nn.ModuleList(nn.ModuleList(nn.Conv2d(16, 9, 1) for _ in range(2)) for s in range(3))
        self.out = nn.ModuleList(nn.ModuleList(nn.Conv2d(16, self.channels[s], 1, bias=False) for _ in range(2)) for s in range(3))
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)) and layer.bias is not None:
                nn.init.zeros_(layer.bias)
        for pair in self.out:
            for layer in pair:
                nn.init.zeros_(layer.weight)

    @staticmethod
    def _difference(z, k):
        padded = F.pad(z, (1, 1, 1, 1), mode='replicate')
        h, w = z.shape[-2:]
        result = torch.zeros_like(z)
        for i in range(3):
            for j in range(3):
                result = result + k[:, i * 3 + j:i * 3 + j + 1] * (padded[:, :, i:i+h, j:j+w] - z)
        return result

    def forward(self, c_list, p_list, capture=False):
        if len(c_list) != 4 or len(p_list) != 4:
            raise ValueError('expected four aligned tensors for each modality')
        co, po, diag = [], [], []
        for s in range(3):
            inputs = (c_list[s], p_list[s])
            with torch.autocast(device_type=inputs[0].device.type, enabled=False):
                z = [self.shallow[s][m](inputs[m].float()) for m in range(2)]
                size = inputs[0].shape[-2:]
                deep = [F.interpolate(self.deep[s][m]((c_list, p_list)[m][s+1].float()), size=size, mode='bilinear', align_corners=False) for m in range(2)]
                role = 2 * torch.sigmoid(self.text_map[s](self.text_features.float()))
                q = F.gelu(self.joint[s](torch.cat([(deep[m] + z[m]) * role[m][None, :, None, None] for m in range(2)], dim=1)))
                kernels = [torch.softmax(self.kernel[s][m](q), dim=1) for m in range(2)]
                diff = [self._difference(z[m], kernels[m]) for m in range(2)]
                delta = [self.out[s][m](diff[m]) for m in range(2)]
            co.append((inputs[0].float() + delta[0]).to(inputs[0].dtype))
            po.append((inputs[1].float() + delta[1]).to(inputs[1].dtype))
            if capture:
                with torch.no_grad():
                    d = {}
                    for m, name in enumerate(('ct', 'pet')):
                        k = kernels[m].detach()
                        d[name + '_delta'] = delta[m].detach()
                        d[name + '_kernel_entropy'] = (-(k * k.clamp_min(1e-12).log()).sum(1)).mean()
                        d[name + '_kernel_spatialstd'] = k.flatten(2).std(dim=2, unbiased=False).mean()
                        d[name + '_kernel_sum_error'] = (k.sum(1) - 1).abs().max()
                        d[name + '_constant_response'] = self._difference(torch.ones_like(z[m]), k).abs().max()
                        d[name + '_rolemod'] = role[m].detach()
                        d[name + '_current_to_deep_condition_RMS'] = z[m].square().mean().sqrt() / (deep[m].square().mean().sqrt() + 1e-8)
                    diag.append(d)
        co.append(c_list[3])
        po.append(p_list[3])
        if capture:
            diag.append({'ct_delta': torch.zeros_like(c_list[3]), 'pet_delta': torch.zeros_like(p_list[3])})
        return co, po, diag
