# -*- coding: utf-8 -*-
"""Builder for the spatial-fusion baseline (does not touch build_dual_model)."""
from models.dual_shared_spatial_fusion import (
    DEFAULT_FUSION_KWARGS,
    MODEL_ARCH,
    DualSharedSpatialFusionPETCTModel,
)


def _get(cfg, name, default):
    return getattr(cfg, name, default)


def build_spatial_fusion_model(cfg):
    arch = str(_get(cfg, 'model_arch', MODEL_ARCH))
    if arch != MODEL_ARCH:
        raise ValueError(f'build_spatial_fusion_model expects model_arch={MODEL_ARCH!r}, got {arch!r}')
    fusion_kwargs = dict(DEFAULT_FUSION_KWARGS)
    fusion_kwargs.update({
        'attention_dim': int(_get(cfg, 'fusion_attention_dim', 64)),
        'num_heads': int(_get(cfg, 'fusion_num_heads', 4)),
        'local_kernel_size': int(_get(cfg, 'fusion_local_kernel_size', 5)),
        'max_axis_length': int(_get(cfg, 'fusion_max_axis_length', 128)),
        'axis_chunk_size': int(_get(cfg, 'fusion_axis_chunk_size', 32)),
        'mode': str(_get(cfg, 'fusion_mode', 'full')),
        'use_checkpoint': bool(_get(cfg, 'fusion_use_checkpoint', True)),
        'check_finite': True,
        'resize_pet': True,
    })
    model = DualSharedSpatialFusionPETCTModel(
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
        fusion_enabled=bool(_get(cfg, 'fusion_enabled', True)),
        fusion_kwargs=fusion_kwargs,
    )
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f'[spatial_fusion] enabled={model.fusion_enabled} '
          f'fusion_class={type(model.fusion).__name__} mode={model.fusion_mode} '
          f'fusion_params={fusion_params} params_total={total_params}', flush=True)
    return {'model': model}