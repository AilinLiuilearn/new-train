# -*- coding: utf-8 -*-
"""FNet-style sparse residual feature fusion for CT/PET segmentation.

This module adapts the sparse-coding decomposition idea from image-level
multi-modal fusion to multi-stage encoder feature fusion. It keeps the output
shape identical to each encoder stage so it can replace simple CT+PET summation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class SparseThreshold(nn.Module):
    def __init__(self, init_theta=-2.0, alpha=0.1, sharpness=20.0):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(float(init_theta)))
        self.alpha = float(alpha)
        self.sharpness = float(sharpness)
        self.softplus = nn.Softplus()

    def forward(self, x):
        theta = self.softplus(self.theta)
        magnitude = x.abs()
        gate = torch.sigmoid(self.sharpness * (magnitude - theta))
        shrink = torch.relu(magnitude - self.alpha * theta)
        return x.sign() * shrink * gate


class SparseCodingBlock(nn.Module):
    def __init__(self, channels, hidden_channels=None, n_iter=2):
        super().__init__()
        hidden = hidden_channels or max(16, min(64, channels // 4))
        self.n_iter = max(1, int(n_iter))
        self.encode = nn.Conv2d(channels, hidden, 1, bias=False)
        self.refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, 1, bias=False),
            )
            for _ in range(self.n_iter)
        ])
        self.threshold = SparseThreshold()
        self.decode = nn.Conv2d(hidden, channels, 1, bias=False)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        code = self.threshold(self.encode(x))
        for block in self.refine:
            code = self.threshold(code + self.res_scale * block(code))
        rec = self.decode(code)
        return code, rec


class FNetSparseFusionStage(nn.Module):
    def __init__(self, channels, hidden_channels=None, n_iter=2, init_gamma=0.1):
        super().__init__()
        hidden = hidden_channels or max(16, min(64, channels // 4))
        self.ct_sparse = SparseCodingBlock(channels, hidden_channels=hidden, n_iter=n_iter)
        self.pet_sparse = SparseCodingBlock(channels, hidden_channels=hidden, n_iter=n_iter)
        self.joint_sparse = SparseCodingBlock(channels * 2, hidden_channels=hidden, n_iter=n_iter)
        self.decoder = nn.Sequential(
            LayerNorm2d(hidden * 3),
            nn.Conv2d(hidden * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.balance_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, max(8, channels // 8), 1, bias=False),
            nn.GELU(),
            nn.Conv2d(max(8, channels // 8), channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, ct_feature, pet_feature):
        ct_code, ct_rec = self.ct_sparse(ct_feature)
        pet_code, pet_rec = self.pet_sparse(pet_feature)
        ct_res = ct_feature - ct_rec
        pet_res = pet_feature - pet_rec
        joint_code, joint_rec = self.joint_sparse(torch.cat([ct_res, pet_res], dim=1))

        enhanced = self.decoder(torch.cat([ct_code, pet_code, joint_code], dim=1))
        gate = self.balance_gate(torch.cat([ct_feature, pet_feature], dim=1))
        base = ct_feature * (1.0 - gate) + pet_feature * gate
        out = base + self.gamma * enhanced

        aux = {
            'ct_code': ct_code,
            'pet_code': pet_code,
            'joint_code': joint_code,
            'ct_rec': ct_rec,
            'pet_rec': pet_rec,
            'joint_rec': joint_rec,
            'ct_res': ct_res,
            'pet_res': pet_res,
            'enhanced': enhanced,
            'gate': gate,
            'out': out,
        }
        return out, aux


class MultiStageFNetSparseFusion(nn.Module):
    def __init__(self, encoder_channels, hidden_ratio=0.25, max_hidden=64, n_iter=2, init_gamma=0.1):
        super().__init__()
        self.stages = nn.ModuleList()
        for channels in encoder_channels:
            hidden = max(16, min(int(max_hidden), int(channels * hidden_ratio)))
            self.stages.append(
                FNetSparseFusionStage(
                    channels,
                    hidden_channels=hidden,
                    n_iter=n_iter,
                    init_gamma=init_gamma,
                )
            )

    def forward(self, ct_feats, pet_feats):
        fused = []
        aux = []
        for stage, ct_feature, pet_feature in zip(self.stages, ct_feats, pet_feats):
            out, stage_aux = stage(ct_feature, pet_feature)
            fused.append(out)
            aux.append(stage_aux)
        return fused, aux

    def get_fusion_visuals(self):
        visuals = {}
        for idx, stage in enumerate(self.stages, start=1):
            stage_visuals = getattr(stage, 'last_visuals', None)
            if stage_visuals:
                visuals[f'fnet_sparse{idx}'] = stage_visuals
        return visuals


def fnet_sparse_auxiliary_loss(fusion_aux, mask=None, recon_weight=0.0, sparse_weight=0.0, decor_weight=0.0, edge_weight=0.0):
    if not fusion_aux:
        device = mask.device if mask is not None else torch.device('cpu')
        return torch.tensor(0.0, device=device), {}

    recon_losses = []
    sparse_losses = []
    decor_losses = []
    edge_losses = []
    eps = 1e-6

    for aux in fusion_aux:
        ct_res = aux['ct_res'].float()
        pet_res = aux['pet_res'].float()
        joint_rec = aux['joint_rec'].float()
        ct_code = aux['ct_code'].float()
        pet_code = aux['pet_code'].float()
        joint_code = aux['joint_code'].float()

        if recon_weight > 0:
            target_res = torch.cat([ct_res.detach(), pet_res.detach()], dim=1)
            recon_losses.append(F.smooth_l1_loss(joint_rec, target_res))

        if sparse_weight > 0:
            sparse_losses.append((ct_code.abs().mean() + pet_code.abs().mean() + joint_code.abs().mean()) / 3.0)

        if decor_weight > 0:
            ct_desc = F.adaptive_avg_pool2d(ct_code, 1).flatten(1)
            pet_desc = F.adaptive_avg_pool2d(pet_code, 1).flatten(1)
            ct_desc = ct_desc - ct_desc.mean(dim=1, keepdim=True)
            pet_desc = pet_desc - pet_desc.mean(dim=1, keepdim=True)
            decor_losses.append((F.cosine_similarity(ct_desc, pet_desc, dim=1, eps=eps) ** 2).mean())

        if edge_weight > 0 and mask is not None:
            enhanced = aux['enhanced'].float()
            mask_s = F.interpolate(mask.float(), size=enhanced.shape[-2:], mode='nearest')
            dx = mask_s[:, :, :, 1:] - mask_s[:, :, :, :-1]
            dy = mask_s[:, :, 1:, :] - mask_s[:, :, :-1, :]
            edge = F.pad(dx.abs(), (0, 1, 0, 0)) + F.pad(dy.abs(), (0, 0, 0, 1))
            if edge.sum() > 0:
                energy = enhanced.abs().mean(dim=1, keepdim=True)
                energy = energy / (energy.mean(dim=(2, 3), keepdim=True) + eps)
                edge_losses.append((energy * edge).sum() / (edge.sum() + eps))

    device = fusion_aux[0]['ct_res'].device
    zero = torch.tensor(0.0, device=device)
    loss_recon = torch.stack(recon_losses).mean() if recon_losses else zero
    loss_sparse = torch.stack(sparse_losses).mean() if sparse_losses else zero
    loss_decor = torch.stack(decor_losses).mean() if decor_losses else zero
    loss_edge = torch.stack(edge_losses).mean() if edge_losses else zero
    total = recon_weight * loss_recon + sparse_weight * loss_sparse + decor_weight * loss_decor - edge_weight * loss_edge
    stats = {
        'loss_fnet_recon': loss_recon.detach(),
        'loss_fnet_sparse': loss_sparse.detach(),
        'loss_fnet_decor': loss_decor.detach(),
        'loss_fnet_edge': loss_edge.detach(),
    }
    return total, stats
