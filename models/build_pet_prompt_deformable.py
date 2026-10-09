# -*- coding: utf-8 -*-
"""Dedicated builder for the PET-prompt deformable fusion model."""
from models.components.add_fusion import AddFusion
from models.dual_shared_add_baseline import DECODER_INPUT_CHANNELS
from models.dual_shared_pet_prompt_deformable import (
    DEFAULT_FUSION_KWARGS,
    MODEL_ARCH,
    DualSharedPETPromptDeformableModel,
)
from models.pet_prompt_deformable_fusion import MultiScalePETPromptDeformableFusion


def _get(cfg, name, default):
    return getattr(cfg, name, default)


def build_pet_prompt_model(cfg):
    arch = str(_get(cfg, 'model_arch', MODEL_ARCH))
    if arch != MODEL_ARCH:
        raise ValueError(
            f'build_pet_prompt_model expects model_arch={MODEL_ARCH!r}, got {arch!r}')
    fusion_type = str(_get(cfg, 'fusion_type', 'pet_prompt'))
    fusion_kwargs = dict(DEFAULT_FUSION_KWARGS)
    fusion_kwargs.update({
        'inner_channels': int(_get(cfg, 'fusion_inner_channels', 32)),
        'heads': int(_get(cfg, 'fusion_heads', 4)),
        'offset_radius': float(_get(cfg, 'fusion_offset_radius', 2.0)),
        'chunk_rows': int(_get(cfg, 'fusion_chunk_rows', 16)),
        'checkpoint_chunks': bool(_get(cfg, 'fusion_checkpoint_chunks', True)),
        'use_gaussian': bool(_get(cfg, 'fusion_use_gaussian', True)),
        'use_pet_prompt': bool(_get(cfg, 'fusion_use_pet_prompt', True)),
        'use_deformable': bool(_get(cfg, 'fusion_use_deformable', True)),
        'use_ct_correction': bool(_get(cfg, 'fusion_use_ct_correction', True)),
        'check_finite': bool(_get(cfg, 'fusion_check_finite', True)),
    })
    model = DualSharedPETPromptDeformableModel(
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
    if fusion_type == 'pet_prompt' and not isinstance(
            model.fusion, MultiScalePETPromptDeformableFusion):
        raise RuntimeError('fusion_type=pet_prompt must build '
                           'MultiScalePETPromptDeformableFusion, '
                           f'got {type(model.fusion).__name__}')
    if fusion_type == 'add' and not isinstance(model.fusion, AddFusion):
        raise RuntimeError('fusion_type=add must keep AddFusion, '
                           f'got {type(model.fusion).__name__}')
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f'[pet_prompt_fusion] fusion_type={fusion_type} '
          f'fusion_class={type(model.fusion).__name__} '
          f'channels={tuple(DECODER_INPUT_CHANNELS)} '
          f'fusion_params={fusion_params} params_total={total_params}', flush=True)
    return {'model': model}
