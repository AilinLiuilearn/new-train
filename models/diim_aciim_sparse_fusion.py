import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=padding, bias=False, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FeatureRectifyLite(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.channel_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim * 2, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_mlp = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        c = self.channel_mlp(x)
        c1, c2 = c.chunk(2, dim=1)
        s = self.spatial_mlp(x)
        s1 = s[:, 0:1]
        s2 = s[:, 1:2]
        out1 = x1 + c2 * x2 + s2 * x2
        out2 = x2 + c1 * x1 + s1 * x1
        return out1, out2


class DynamicSparseGate(nn.Module):
    def __init__(self, dim, keep_ratio=0.25, min_keep=64):
        super().__init__()
        hidden = max(dim // 4, 8)
        self.keep_ratio = float(keep_ratio)
        self.min_keep = int(min_keep)
        self.score = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        self.temp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, diff, common):
        logits = self.score(torch.cat([diff, common], dim=1))
        temp = 0.5 + self.temp(torch.cat([diff, common], dim=1))
        logits = logits / temp
        b, _, h, w = logits.shape
        flat = logits.flatten(1)
        k = max(int(h * w * self.keep_ratio), self.min_keep)
        k = min(k, h * w)
        topk_vals, _ = torch.topk(flat, k=k, dim=1)
        thresh = topk_vals[:, -1].view(b, 1, 1, 1)
        hard = (logits >= thresh).to(dtype=logits.dtype)
        soft = torch.sigmoid(logits - thresh)
        return hard + (1.0 - hard) * soft


class DIIMACIIMSparseFusionBlock(nn.Module):
    def __init__(self, ct_channels, pet_channels, out_channels,
                 keep_ratio=0.25, min_keep=64, diff_weight=0.5):
        super().__init__()
        self.ct_proj = ConvBNAct(ct_channels, out_channels, kernel_size=1)
        self.pet_proj = ConvBNAct(pet_channels, out_channels, kernel_size=1)
        self.rectify = FeatureRectifyLite(out_channels)
        self.diff_refine = nn.Sequential(
            ConvBNAct(out_channels, out_channels, kernel_size=3),
            ConvBNAct(out_channels, out_channels, kernel_size=3),
        )
        self.common_refine = nn.Sequential(
            ConvBNAct(out_channels, out_channels, kernel_size=1),
            ConvBNAct(out_channels, out_channels, kernel_size=3),
        )
        self.sparse_gate = DynamicSparseGate(out_channels, keep_ratio=keep_ratio, min_keep=min_keep)
        self.balance_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 2, max(out_channels // 8, 8), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(out_channels // 8, 8), 2, kernel_size=1),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.diff_weight = nn.Parameter(torch.tensor(float(diff_weight)))
        self._last_aux = None

    def forward(self, ct_feat, pet_feat):
        ct = self.ct_proj(ct_feat)
        pet = self.pet_proj(pet_feat)
        if pet.shape[-2:] != ct.shape[-2:]:
            pet = F.interpolate(pet, size=ct.shape[-2:], mode='bilinear', align_corners=False)

        ct_r, pet_r = self.rectify(ct, pet)
        common = 0.5 * (ct_r + pet_r)
        diff = torch.abs(ct_r - pet_r)
        common = self.common_refine(common)
        diff = self.diff_refine(diff)

        balance = torch.softmax(self.balance_head(torch.cat([common, diff], dim=1)).flatten(1), dim=1)
        balance = balance.view(-1, 2, 1, 1)
        common_w = balance[:, 0:1]
        diff_w = balance[:, 1:2]

        mask = self.sparse_gate(diff, common)
        lesion = diff * mask
        background = common * (1.0 - 0.5 * mask)
        diff_gain = torch.sigmoid(self.diff_weight)

        fused = self.mix(torch.cat([
            background * (1.0 + common_w),
            lesion * (1.0 + diff_w) * (0.5 + diff_gain),
        ], dim=1))
        fused = fused + 0.1 * (ct + pet)

        self._last_aux = {
            'balance': balance.detach(),
            'mask': mask.detach(),
            'mask_ratio': mask.detach().mean(),
            'diff_gain': diff_gain.detach(),
        }
        return fused


class DIIMACIIMSparseFusion(nn.Module):
    def __init__(self, ct_channels, pet_channels, out_channels,
                 keep_ratio=(0.35, 0.25, 0.2, 0.15), min_keep=(128, 64, 32, 16), diff_weight=0.5):
        super().__init__()
        if not (len(ct_channels) == len(pet_channels) == len(out_channels) == 4):
            raise ValueError('DIIMACIIMSparseFusion expects 4-stage features.')
        self.blocks = nn.ModuleList([
            DIIMACIIMSparseFusionBlock(
                ct_ch, pet_ch, out_ch,
                keep_ratio=keep_ratio[i],
                min_keep=min_keep[i],
                diff_weight=diff_weight,
            )
            for i, (ct_ch, pet_ch, out_ch) in enumerate(zip(ct_channels, pet_channels, out_channels))
        ])
        self._last_aux = {}

    def forward(self, ct_feats, pet_feats):
        fused = []
        aux = {}
        for idx, (block, ct_feat, pet_feat) in enumerate(zip(self.blocks, ct_feats, pet_feats), start=1):
            fused.append(block(ct_feat, pet_feat))
            aux[f'stage{idx}'] = block._last_aux
        self._last_aux = aux
        return fused, {'diim_aciim_sparse': aux}

    def get_fusion_visuals(self):
        if not self._last_aux:
            return {}
        visuals = {}
        for stage, item in self._last_aux.items():
            if not item:
                continue
            visuals[f'{stage}_mask'] = item['mask']
            visuals[f'{stage}_mask_ratio'] = item['mask_ratio']
            visuals[f'{stage}_balance'] = item['balance']
            visuals[f'{stage}_diff_gain'] = item['diff_gain']
        return visuals
