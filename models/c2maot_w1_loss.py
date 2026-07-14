# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def one_dimensional_empirical_wasserstein_1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.shape != y.shape:
        raise RuntimeError(f'Shape mismatch for W1 loss: source={tuple(x.shape)} target={tuple(y.shape)}')
    with torch.cuda.amp.autocast(enabled=False):
        x_flat = x.float().flatten(start_dim=1)
        y_flat = y.float().flatten(start_dim=1)
        if x_flat.shape != y_flat.shape:
            raise RuntimeError(f'Flattened shape mismatch for W1 loss: source={tuple(x_flat.shape)} target={tuple(y_flat.shape)}')
        x_sorted = torch.sort(x_flat, dim=1).values
        y_sorted = torch.sort(y_flat, dim=1).values
        w1_sample = (x_sorted - y_sorted).abs().mean(dim=1)
        return w1_sample.mean()


class C2MAOTHierarchicalW1Loss(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        alpha = float(alpha)
        if alpha <= 1.0:
            raise ValueError(f'alpha must be > 1 for hierarchical weighting, got {alpha}')
        self.alpha = alpha

    def forward(
        self,
        source_feats: Dict[int, torch.Tensor],
        target_feats: Dict[int, torch.Tensor],
        active_stage_numbers: Tuple[int, ...],
    ):
        total = None
        diagnostics = {}
        for stage in active_stage_numbers:
            if stage not in source_feats or stage not in target_feats:
                raise RuntimeError(f'Missing stage {stage} in source/target features')
            source = source_feats[stage]
            target = target_feats[stage]
            if source.shape != target.shape:
                raise RuntimeError(
                    f'Stage {stage} shape mismatch: source={tuple(source.shape)} target={tuple(target.shape)}'
                )
            w1 = one_dimensional_empirical_wasserstein_1(source, target)
            weight = source.new_tensor(self.alpha ** stage)
            weighted = weight * w1
            diagnostics[f'pg_mtr_ot_s{stage}_w1'] = w1.detach()
            diagnostics[f'pg_mtr_ot_s{stage}_weight'] = weight.detach()
            diagnostics[f'pg_mtr_ot_s{stage}_weighted'] = weighted.detach()
            total = weighted if total is None else total + weighted
        if total is None:
            total = next(iter(source_feats.values())).new_tensor(0.0)
        diagnostics['pg_mtr_ot_total'] = total.detach()
        return total, diagnostics
