# -*- coding: utf-8 -*-
"""Dedicated builder for the local-contrast fusion model (MDTSegTeacher-compatible)."""
from models.dual_shared_local_contrast_fusion import (
    DEFAULT_FUSION_KWARGS,
    MODEL_ARCH,
    DualSharedLocalContrastFusionModel,
)


def _get(cfg, name, default):
    return getattr(cfg, name, default)


def build_local_contrast_fusion_model(cfg):
    arch = str(_get(cfg, 'model_arch', MODEL_ARCH))
    if arch != MODEL_ARCH:
        raise ValueError(
            f'build_local_contrast_fusion_model expects model_arch={MODEL_ARCH!r}, got {arch!r}')
    fusion_kwargs = dict(DEFAULT_FUSION_KWARGS)
    fusion_kwargs.update({
        'dim': int(_get(cfg, 'fusion_dim', 32)),
        'heads': int(_get(cfg, 'fusion_heads', 4)),
        'window': int(_get(cfg, 'fusion_window', 5)),
        'chunk_rows': int(_get(cfg, 'fusion_chunk_rows', 16)),
        'checkpoint_chunks': bool(_get(cfg, 'fusion_checkpoint_chunks', True)),
        'position_bias_beta': float(_get(cfg, 'fusion_position_bias_beta', 0.0)),
    })
    model = DualSharedLocalContrastFusionModel(
        ct_backbone=_get(cfg, 'ct_backbone', 'convnextv2_nano'),
        pet_backbone=_get(cfg, 'pet_backbone', 'mit_b1'),
        ct_pretrained_path=_get(cfg, 'ct_pretrained_path', None),
        pet_pretrained_path=_get(cfg, 'pet_pretrained_path', None),
        in_channels=3,
        out_channels=1,
        decoder_channels=tuple(_get(cfg, 'decoder_channels', (512, 256, 128, 64))),
        use_deep_supervision=bool(_get(cfg, 'use_deep_supervision', False)
                                  or _get(cfg, 'deep_supervision', False)),
        pretrained=bool(_get(cfg, 'pretrained', True)),
        check_finite=bool(_get(cfg, 'check_finite', True)),
        fusion_kwargs=fusion_kwargs,
    )
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f'[local_contrast_fusion] fusion_class={type(model.fusion).__name__} '
          f'fusion_params={fusion_params} params_total={total_params}', flush=True)
    return {'model': model}