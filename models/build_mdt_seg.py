# -*- coding: utf-8 -*-
"""Model builder for the clean Full/Missing Add baseline."""
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline


def _get(cfg, name, default):
    return getattr(cfg, name, default)


def build_dual_model(cfg):
    model = DualSharedAddPETCTBaseline(
        baseline=cfg.baseline,
        ct_backbone=_get(cfg, 'ct_backbone', 'convnextv2_nano'),
        pet_backbone=_get(cfg, 'pet_backbone', 'mit_b1'),
        ct_pretrained_path=_get(cfg, 'ct_pretrained_path', None) if _get(cfg, 'pretrained', True) else None,
        pet_pretrained_path=_get(cfg, 'pet_pretrained_path', None) if _get(cfg, 'pretrained', True) else None,
        in_channels=3,
        out_channels=1,
        decoder_channels=tuple(_get(cfg, 'decoder_channels', (512, 256, 128, 64))),
        use_deep_supervision=bool(_get(cfg, 'use_deep_supervision', False)
                                  or _get(cfg, 'deep_supervision', False)),
        pretrained=bool(_get(cfg, 'pretrained', True)),
    )
    return {'model': model}
