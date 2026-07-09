import torch
import torch.nn as nn
import torch.nn.functional as F


def _truncated_normal_(tensor, mean=0.0, std=0.02, a=-2.0, b=2.0):
    with torch.no_grad():
        try:
            return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
        except Exception:
            tmp = tensor.new_empty(tensor.shape + (4,)).normal_()
            valid = (tmp > a) & (tmp < b)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor


class StageAnatomyConditionedMissingPETRouter(nn.Module):
    def __init__(self, in_channels, num_tokens=8, residual_scale_init=0.1):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.local_router = nn.Sequential(
            nn.GroupNorm(num_groups=min(8, in_channels), num_channels=in_channels),
            nn.Conv2d(in_channels, self.num_tokens, kernel_size=1, bias=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_router = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels, self.num_tokens, kernel_size=1, bias=True),
        )
        self.token_bank = nn.Parameter(torch.empty(self.num_tokens, in_channels))
        self.refine = nn.Sequential(
            nn.GroupNorm(num_groups=min(8, in_channels), num_channels=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self._reset_parameters()

    def _reset_parameters(self):
        _truncated_normal_(self.token_bank, std=0.02)
        with torch.no_grad():
            self.token_bank.sub_(self.token_bank.mean(dim=0, keepdim=True))
        for m in self.local_router.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.global_router.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.refine.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (1, 1):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ct_feat):
        local_logits = self.local_router(ct_feat)
        global_logits = self.global_router(self.global_pool(ct_feat))
        routing = F.softmax((local_logits.float() + global_logits.float()), dim=1).to(dtype=ct_feat.dtype)
        missing = torch.einsum('bkhw,kc->bchw', routing, self.token_bank)
        missing = self.refine(missing)
        missing = missing * self.residual_scale
        diagnostics = {
            'local_logits': local_logits,
            'global_logits': global_logits,
            'routing': routing,
            'token_bank': self.token_bank,
            'residual_scale': self.residual_scale,
        }
        return missing, diagnostics


class AnatomyConditionedHierarchicalMissingPETRouter(nn.Module):
    def __init__(self, channels_list, num_tokens=8, residual_scale_init=0.1):
        super().__init__()
        self.stages = nn.ModuleList([
            StageAnatomyConditionedMissingPETRouter(c, num_tokens=num_tokens, residual_scale_init=residual_scale_init)
            for c in channels_list
        ])

    def forward(self, ct_feats):
        if len(ct_feats) != len(self.stages):
            raise ValueError(f'Expected {len(self.stages)} stage features, got {len(ct_feats)}')
        residuals = []
        diagnostics = []
        for feat, stage in zip(ct_feats, self.stages):
            residual, diag = stage(feat)
            residuals.append(residual)
            diagnostics.append(diag)
        return residuals, diagnostics
