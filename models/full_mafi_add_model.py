# -*- coding: utf-8 -*-
"""Build MAFI+Add using the unchanged baseline's encoders and state routing.

Install fusion before MDTSegTeacher constructs its optimizer and EMA.
The inherited Missing route calls neither the PET encoder nor MAFI.
"""
from models.build_mdt_seg import build_dual_model
from models.dual_shared_add_baseline import DECODER_INPUT_CHANNELS
from models.mafi_add_fusion import MAFIAddFusion


def build_full_mafi_add_model(cfg):
    networks = build_dual_model(cfg)
    model = networks['model']
    if bool(getattr(cfg, 'mafi_enabled', True)):
        model.fusion = MAFIAddFusion(DECODER_INPUT_CHANNELS)
    print(f'[full_mafi_add] enabled={bool(getattr(cfg, "mafi_enabled", True))} '
          f'fusion={type(model.fusion).__name__} '
          f'fusion_params={sum(p.numel() for p in model.fusion.parameters())}',
          flush=True)
    return networks

