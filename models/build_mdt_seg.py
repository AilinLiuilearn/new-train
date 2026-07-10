import os

import torch
import torch.nn as nn
import timm

try:
    from transformers import SegformerConfig, SegformerModel, ConvNextConfig, ConvNextModel
except ImportError:
    SegformerConfig = None
    SegformerModel = None
    ConvNextConfig = None
    ConvNextModel = None


def _unwrap_state_dict(state_dict):
    if isinstance(state_dict, dict):
        for key in ('state_dict', 'model', 'module'):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
    return state_dict


def _map_convnext_hf_to_timm_key(key):
    if not key.startswith('convnext.'):
        return None
    key = key[len('convnext.'):]
    if key in ('stem_weight', 'embeddings.patch_embeddings.weight'):
        return 'stem_0.weight'
    if key in ('stem_bias', 'embeddings.patch_embeddings.bias'):
        return 'stem_0.bias'
    if key == 'embeddings.layernorm.weight':
        return 'stem_1.weight'
    if key == 'embeddings.layernorm.bias':
        return 'stem_1.bias'
    if key.startswith('encoder.stages.'):
        key = key[len('encoder.'):]
    key = key.replace('stages.', 'stages_')
    key = key.replace('layers.', 'blocks.')
    key = key.replace('layer_scale_parameter', 'gamma')
    key = key.replace('dwconv.', 'conv_dw.')
    key = key.replace('pwconv1.', 'mlp.fc1.')
    key = key.replace('pwconv2.', 'mlp.fc2.')
    key = key.replace('downsampling_layer.', 'downsample.')
    key = key.replace('layernorm.', 'norm.')
    return key


def build_mdt_seg_teacher(config):
    model_arch = getattr(config, 'model_arch', 'dual_shared_add_baseline')
    if model_arch == 'dual_shared_add_baseline':
        from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
        model = DualSharedAddPETCTBaseline(
            ct_backbone=getattr(config, 'ct_backbone', 'convnextv2_nano'),
            pet_backbone=getattr(config, 'pet_backbone', 'mit_b1'),
            ct_pretrained_path=getattr(config, 'ct_pretrained_path', None),
            pet_pretrained_path=getattr(config, 'pet_pretrained_path', None),
            in_channels=3,
            out_channels=1,
            decoder_channels=getattr(config, 'decoder_channels', (512, 256, 128, 64)),
            use_deep_supervision=_resolve_use_deep_supervision(config),
            missing_mode=getattr(config, 'missing_mode', 'ct'),
            pg_mtr_num_tokens=getattr(config, 'pg_mtr_num_tokens', 8),
            pg_mtr_temperature=getattr(config, 'pg_mtr_temperature', 0.07),
        )
        print(
            f'[dual_shared_add_baseline] ct={getattr(config, "ct_backbone", "convnextv2_nano")} '
            f'pet={getattr(config, "pet_backbone", "mit_b1")} '
            f'fusion=add shared_decoder=UNetStyleDecoder '
            f'deep_supervision={_resolve_use_deep_supervision(config)} '
            f'missing_mode={getattr(config, "missing_mode", "ct")}'
        )
        return dict(model=model)
    raise ValueError(
        f'Unsupported model_arch={model_arch}. '
        'Only dual_shared_add_baseline is kept for this cleaned checkpoint.'
    )


def _resolve_use_deep_supervision(config):
    if getattr(config, 'use_deep_supervision', False):
        return True
    return bool(getattr(config, 'deep_supervision', False))
