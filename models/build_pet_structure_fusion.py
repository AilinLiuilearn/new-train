# -*- coding: utf-8 -*-
"""Dedicated builder for the PET-structure fusion model (task-compatible)."""
from models.components.add_fusion import AddFusion
from models.dual_shared_add_baseline import DECODER_INPUT_CHANNELS
from models.dual_shared_pet_structure_fusion import (
    DEFAULT_FUSION_KWARGS,
    MODEL_ARCH,
    DualSharedPETStructureFusionModel,
)
from models.pet_guided_structure_fusion import MultiScalePETGuidedStructureFusion


def _get(cfg, name, default):
    return getattr(cfg, name, default)


def build_pet_structure_model(cfg):
    arch = str(_get(cfg, 'model_arch', MODEL_ARCH))
    if arch != MODEL_ARCH:
        raise ValueError(
            f'build_pet_structure_model expects model_arch={MODEL_ARCH!r}, got {arch!r}')
    fusion_type = str(_get(cfg, 'fusion_type', 'pet_structure'))
    fusion_kwargs = dict(DEFAULT_FUSION_KWARGS)
    fusion_kwargs.update({
        'inner_channels': int(_get(cfg, 'fusion_inner_channels', 32)),
        'heads': int(_get(cfg, 'fusion_heads', 4)),
        'kernel_size': int(_get(cfg, 'fusion_kernel_size', 5)),
        'chunk_rows': int(_get(cfg, 'fusion_chunk_rows', 16)),
        'checkpoint_chunks': bool(_get(cfg, 'fusion_checkpoint_chunks', True)),
        'beta_init': float(_get(cfg, 'fusion_beta_init', 0.0)),
        'structure_strength_init': float(_get(cfg, 'fusion_structure_strength_init', 0.1)),
        'use_pet_guidance': bool(_get(cfg, 'fusion_use_pet_guidance', True)),
        'use_structure_constraint': bool(_get(cfg, 'fusion_use_structure_constraint', True)),
        'ct_update_type': str(_get(cfg, 'fusion_ct_update_type', 'difference')),
        'check_finite': bool(_get(cfg, 'fusion_check_finite', True)),
    })
    model = DualSharedPETStructureFusionModel(
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
        fusion_type=fusion_type,
        fusion_kwargs=fusion_kwargs,
    )
    if fusion_type == 'pet_structure' and not isinstance(
            model.fusion, MultiScalePETGuidedStructureFusion):
        raise RuntimeError('fusion_type=pet_structure must build '
                           'MultiScalePETGuidedStructureFusion, '
                           f'got {type(model.fusion).__name__}')
    if fusion_type == 'add' and not isinstance(model.fusion, AddFusion):
        raise RuntimeError('fusion_type=add must keep AddFusion, '
                           f'got {type(model.fusion).__name__}')
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f'[pet_structure_fusion] fusion_type={fusion_type} '
          f'fusion_class={type(model.fusion).__name__} '
          f'channels={tuple(DECODER_INPUT_CHANNELS)} '
          f'fusion_params={fusion_params} params_total={total_params}', flush=True)
    return {'model': model}