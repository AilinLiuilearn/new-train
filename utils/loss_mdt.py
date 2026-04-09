# -*- coding: utf-8 -*-
"""MDT 表示损失：相似性、差异性、重构"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def flatten_feature(x):
    """(B,C,H,W) -> (B, C*H*W)"""
    return x.flatten(1)


class SimCosineLoss(nn.Module):
    def forward(self, x1, x2):
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=1)
        cos = (x1 * x2).sum(dim=1)
        return (1 - cos).mean()


class DiffCosineLoss(nn.Module):
    def forward(self, x1, x2):
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=1)
        cos = (x1 * x2).sum(dim=1)
        return (1 + cos).mean()


class DiffFrobeniusLoss(nn.Module):
    def forward(self, x1, x2):
        batch_size = x1.size(0)
        x1 = x1.view(batch_size, -1)
        x2 = x2.view(batch_size, -1)
        n1 = torch.norm(x1, p=2, dim=1, keepdim=True).detach() + 1e-6
        n2 = torch.norm(x2, p=2, dim=1, keepdim=True).detach() + 1e-6
        x1 = x1 / n1
        x2 = x2 / n2
        return (x1.t() @ x2).mean()


class DiffMSELoss(nn.Module):
    def forward(self, x1, x2):
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=1)
        return F.mse_loss(x1, x2)


def _rbf_kernel(x):
    """RBF 核矩阵：x (B, N) → K (B, B)，带宽用中位数启发式。"""
    dist = torch.cdist(x, x)
    sigma = dist.median().clamp(min=1e-6)
    return torch.exp(-dist ** 2 / (2 * sigma ** 2))


def hsic_loss(z1, z2):
    """
    HSIC 独立性损失：度量 z1 与 z2 的统计相关性，值越小越独立。
    用于约束 z_mri（CT专属）与 z_mri_g（通用）之间的独立性。
    计算方式：先 avg pool 到 (B, C)，再用 RBF 核计算 HSIC。
    HSIC = Tr(Kx H Ky H) / (B-1)^2，H 为中心化矩阵。
    """
    B = z1.shape[0]
    if B < 2:
        return z1.new_zeros(1).squeeze()
    x = F.adaptive_avg_pool2d(z1, 1).view(B, -1)
    y = F.adaptive_avg_pool2d(z2, 1).view(B, -1)
    Kx = _rbf_kernel(x)
    Ky = _rbf_kernel(y)
    H = torch.eye(B, device=x.device) - 1.0 / B
    return torch.trace(Kx @ H @ Ky @ H) / ((B - 1) ** 2)


def mmd_loss(z1, z2):
    """
    MMD 分布对齐损失：使 z_mri_g 与 z_pet_g 的分布接近（通用特征跨模态一致）。
    MMD = E[Kxx] + E[Kyy] - 2·E[Kxy]，用 RBF 核。
    """
    B = z1.shape[0]
    if B < 2:
        return z1.new_zeros(1).squeeze()
    x = F.adaptive_avg_pool2d(z1, 1).view(B, -1)
    y = F.adaptive_avg_pool2d(z2, 1).view(B, -1)
    xy = torch.cat([x, y], dim=0)
    K = _rbf_kernel(xy)
    return K[:B, :B].mean() + K[B:, B:].mean() - 2 * K[:B, B:].mean()


class SimCMDLoss(nn.Module):
    def __init__(self, n_moments=5):
        super().__init__()
        self.n_moments = n_moments

    def _matchnorm(self, x1, x2):
        return torch.sqrt(((x1 - x2) ** 2).sum() + 1e-8)

    def _scm(self, x1, x2, k):
        s1 = torch.mean(torch.pow(x1, k), 0)
        s2 = torch.mean(torch.pow(x2, k), 0)
        return self._matchnorm(s1, s2)

    def forward(self, x1, x2):
        loss = self._matchnorm(x1, x2)
        for i in range(self.n_moments - 1):
            loss = loss + self._scm(x1, x2, i + 2)
        return loss
