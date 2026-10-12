# -*- coding: utf-8 -*-
"""Install weighted addition before the existing task creates optimizer/EMA."""
from models.build_mdt_seg import build_dual_model
from models.dual_shared_add_baseline import DECODER_INPUT_CHANNELS
from models.weighted_add_fusion import WeightedAddFusion


def build_full_weighted_add_model(cfg):
    networks = build_dual_model(cfg)
    model = networks['model']
    if bool(getattr(cfg, 'weighted_fusion_enabled', True)):
        model.fusion = WeightedAddFusion(DECODER_INPUT_CHANNELS)
    print(f'[full_weighted_add] fusion={type(model.fusion).__name__} '
          f'fusion_params={sum(p.numel() for p in model.fusion.parameters())}',
          flush=True)
    return networks

