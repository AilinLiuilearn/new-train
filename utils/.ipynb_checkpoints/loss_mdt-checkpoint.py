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
