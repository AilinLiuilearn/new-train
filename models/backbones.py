# -*- coding: utf-8 -*-
"""Encoder construction and offline pretrained-weight loading.

Strict by design:
- Missing timm/transformers raises instead of falling back to a toy encoder.
- A user-supplied pretrained path that does not exist, cannot be parsed, or
  matches zero tensors raises (no silent random init).
- Weights are loaded from local files only; nothing is downloaded.
- Loading reports the source file, per-tensor/per-element load ratio and any
  skipped keys, and verifies that the core stages actually loaded.
"""
import os

import torch
import torch.nn as nn

try:
    import timm
except Exception:  # pragma: no cover - guarded by strict checks below
    timm = None

try:
    from transformers import SegformerConfig, SegformerModel, ConvNextConfig, ConvNextModel
except Exception:  # pragma: no cover - guarded by strict checks below
    SegformerConfig = SegformerModel = None
    ConvNextConfig = ConvNextModel = None


class SimpleFeatureInfo:
    def __init__(self, channels):
        self._channels = list(channels)

    def channels(self):
        return list(self._channels)


def _normalize_backbone_name(backbone):
    backbone = str(backbone).strip().replace('\u200b', '').replace('\ufeff', '')
    aliases = {
        'mit-b1': 'mit_b1',
        'segformer-b1': 'mit_b1',
        'nvidia/mit-b1': 'mit_b1',
        'convnextv2-nano': 'convnextv2_nano',
        'convnext_v2_nano': 'convnextv2_nano',
    }
    return aliases.get(backbone, backbone)


def _unwrap_state_dict(state_dict):
    if isinstance(state_dict, dict):
        for key in ('state_dict', 'model', 'module'):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
    return state_dict


def _sanitize_state_dict(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ('module.', 'backbone.', 'visual.'):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    return cleaned


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


def _state_key_candidates(key):
    candidates = [key]
    prefixes = ('model.', 'module.', 'backbone.', 'encoder.', 'visual.', 'segformer.', 'convnext.')
    for prefix in prefixes:
        if key.startswith(prefix):
            candidates.append(key[len(prefix):])
    if key.startswith('segformer.'):
        suffix = key[len('segformer.'):]
        candidates.extend(['model.' + suffix, suffix])
    if key.startswith('convnext.'):
        suffix = key[len('convnext.'):]
        candidates.extend([suffix, 'model.' + suffix])
        if suffix.startswith('encoder.'):
            enc_suffix = suffix[len('encoder.'):]
            candidates.extend(['model.encoder.' + enc_suffix, 'model.' + enc_suffix])
        elif suffix.startswith('embeddings.'):
            candidates.append('model.' + suffix)
    if key.startswith('convnext.encoder.'):
        suffix = key[len('convnext.encoder.'):]
        candidates.extend([suffix, 'model.encoder.' + suffix, 'model.' + suffix])
    for prefix in ('model.', 'model.encoder.', 'encoder.', 'segformer.', 'segformer.encoder.', 'convnext.', 'convnext.encoder.'):
        candidates.append(prefix + key)
    normalized = []
    for cand in candidates:
        normalized.extend([
            cand,
            cand.replace('stages.', 'stages_'),
            cand.replace('stages_', 'stages.'),
            cand.replace('stem.', 'stem_'),
            cand.replace('stem_', 'stem.'),
            cand.replace('embeddings.patch_embeddings.', 'stem_'),
            cand.replace('embeddings.patch_embeddings.', 'model.embeddings.patch_embeddings.'),
            cand.replace('encoder.stages.', 'stages_'),
            cand.replace('encoder.stages.', 'stages.'),
            cand.replace('layernorm.', 'norm.'),
            cand.replace('layers.', 'blocks.'),
            cand.replace('blocks.', 'layers.'),
            cand.replace('layer_scale_parameter', 'gamma'),
            cand.replace('gamma', 'layer_scale_parameter'),
            cand.replace('dwconv.', 'conv_dw.'),
            cand.replace('conv_dw.', 'dwconv.'),
            cand.replace('pwconv1.', 'mlp.fc1.'),
            cand.replace('mlp.fc1.', 'pwconv1.'),
            cand.replace('pwconv2.', 'mlp.fc2.'),
            cand.replace('mlp.fc2.', 'pwconv2.'),
            cand.replace('downsampling_layer.', 'downsample.'),
            cand.replace('downsample.', 'downsampling_layer.'),
        ])
    return list(dict.fromkeys(normalized))


def _resolve_weight_file(path, name):
    """Resolve a pretrained path to a concrete weight file.

    Accepts either a file or a directory containing a supported weight file.
    Raises if the path does not exist or no supported file is found.
    """
    if not path:
        raise ValueError(f'{name}: pretrained path is empty; pass a real local path '
                         'or explicitly set pretrained=False for a smoke test')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{name}: pretrained path not found: {path}')
    if os.path.isdir(path):
        candidates = (
            'pytorch_model.bin', 'model.safetensors',
            'mit_b1.pth', 'mit_b1.bin', 'mit_b1.pt', 'mit-b1.pth', 'mit-b1.bin', 'mit-b1.pt',
            'convnextv2_nano.pth', 'convnextv2_nano.bin', 'convnextv2_nano.pt',
        )
        for cand in candidates:
            full = os.path.join(path, cand)
            if os.path.exists(full):
                return full
        raise FileNotFoundError(f'{name}: no supported weight file under directory {path}')
    return path


def _load_tensor_dict(path):
    if str(path).endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(path, device='cpu')
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        try:
            return torch.load(path, map_location='cpu')
        except Exception:
            from safetensors.torch import load_file
            return load_file(path, device='cpu')


def load_local_weights_safe(model, path, name='Encoder', required=True):
    """Load offline pretrained weights into ``model``.

    ``required=True`` (default) raises on any failure so a formal experiment
    never silently runs from random init. ``required=False`` is only for smoke
    tests that explicitly opt out of pretrained weights.
    """
    if path is None or path == '':
        if required:
            raise ValueError(f'{name}: pretrained path is required for a formal run; '
                             'pass pretrained=False for a smoke test')
        print(f'[-] {name}: pretrained path not provided; training from scratch (smoke)')
        return
    try:
        weight_file = _resolve_weight_file(path, name)
    except (FileNotFoundError, ValueError) as e:
        if required:
            raise
        print(f'[-] {name}: {e}; training from scratch (smoke)')
        return
    print(f'[+] {name}: loading local weights from {weight_file}')
    try:
        state_dict = _load_tensor_dict(weight_file)
    except Exception as e:
        if required:
            raise RuntimeError(f'{name}: failed to parse weights at {weight_file}: {e}')
        print(f'[-] {name}: failed to parse weights at {weight_file}: {e}; training from scratch (smoke)')
        return
    state_dict = _sanitize_state_dict(_unwrap_state_dict(state_dict))
    model_state = model.state_dict()
    loadable = {}
    skipped = []
    for k, v in state_dict.items():
        matched_key = None
        mapped_key = _map_convnext_hf_to_timm_key(k)
        mapped_candidates = []
        if mapped_key is not None:
            mapped_candidates.extend([mapped_key, f'model.{mapped_key}'])
        for cand in mapped_candidates:
            if cand in model_state and model_state[cand].shape == v.shape:
                matched_key = cand
                break
        if matched_key is None:
            for cand in _state_key_candidates(k):
                if cand in model_state and model_state[cand].shape == v.shape:
                    matched_key = cand
                    break
        if matched_key is not None:
            loadable[matched_key] = v
        else:
            skipped.append(k)
    loaded_tensors = len(loadable)
    loaded_elements = sum(v.numel() for v in loadable.values())
    ckpt_total = len(state_dict)
    ckpt_elements = sum(v.numel() for v in state_dict.values() if torch.is_tensor(v))
    model_total = len(model_state)
    model_elements = sum(v.numel() for v in model_state.values())
    print(f'[PRETRAIN] {name}: weight_file={weight_file}')
    print(f'[PRETRAIN] {name}: checkpoint_tensors={ckpt_total} model_tensors={model_total}')
    print(f'[PRETRAIN] {name}: loaded_tensors={loaded_tensors} '
          f'loaded_elements={loaded_elements}/{model_elements} '
          f'({100.0 * loaded_elements / max(1, model_elements):.1f}% of model)')
    print(f'[PRETRAIN] {name}: skipped_tensors={len(skipped)} '
          f'skipped_elements={sum(v.numel() for k, v in state_dict.items() if k in skipped and torch.is_tensor(v))}')
    if not loadable:
        if required:
            raise RuntimeError(f'{name}: zero compatible tensors loaded from {weight_file}; '
                               'refusing to train from random init')
        print(f'[-] {name}: zero compatible tensors; training from scratch (smoke)')
        return
    msg = model.load_state_dict(loadable, strict=False)
    print(f'[PRETRAIN] {name}: missing_after_load={len(msg.missing_keys)} '
          f'unexpected_after_load={len(msg.unexpected_keys)}')
    if skipped:
        print(f'[PRETRAIN] {name}: skipped_examples={skipped[:8]}')
    _verify_core_stages_loaded(name, model_state, msg.missing_keys)


def _verify_core_stages_loaded(name, model_state, missing_keys):
    """Ensure the stem and all four encoder stages actually received weights."""
    missing = set(missing_keys)
    def stage_loaded(prefix):
        return not any(k.startswith(prefix) for k in missing)
    if _has_stage(model_state, 'stem'):
        if not stage_loaded('stem'):
            raise RuntimeError(f'{name}: stem weights missing after load; refusing random init')
    for i in range(4):
        prefixes = (f'stages.{i}.', f'stages_{i}.', f'encoder.stages.{i}.', f'encoder.stages_{i}.')
        if any(any(k.startswith(p) for p in prefixes) for k in model_state):
            if not stage_loaded(prefixes[0]) and not stage_loaded(prefixes[1]):
                raise RuntimeError(f'{name}: stage {i} weights missing after load; refusing random init')


def _has_stage(model_state, prefix):
    return any(k.startswith(prefix) for k in model_state)


class SegformerFeatureBackbone(nn.Module):
    """MiT (SegFormer) encoder, feature_info exposes the four stage channels."""

    def __init__(self, variant='mit_b1', in_channels=3):
        super().__init__()
        variant = _normalize_backbone_name(variant)
        settings = {
            'mit_b0': dict(depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256], heads=[1, 2, 5, 8]),
            'mit_b1': dict(depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 320, 512], heads=[1, 2, 5, 8]),
        }
        if variant not in settings:
            raise ValueError(f'Unsupported MiT variant: {variant}')
        if SegformerConfig is None or SegformerModel is None:
            raise ImportError('Segformer MiT backbone requires transformers. Install: pip install transformers')
        s = settings[variant]
        config = SegformerConfig(
            num_channels=in_channels, depths=s['depths'], sr_ratios=[8, 4, 2, 1],
            hidden_sizes=s['hidden_sizes'], patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2],
            num_attention_heads=s['heads'], mlp_ratios=[4, 4, 4, 4], hidden_act='gelu',
            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
            classifier_dropout_prob=0.1, initializer_range=0.02, drop_path_rate=0.1,
            reshape_last_stage=True, output_hidden_states=True,
        )
        self.model = SegformerModel(config)
        self.feature_info = SimpleFeatureInfo(config.hidden_sizes)

    def forward(self, x):
        outputs = self.model(pixel_values=x, output_hidden_states=True, return_dict=True)
        hidden = list(outputs.hidden_states or [])
        if len(hidden) >= 5:
            return hidden[1:5]
        if len(hidden) == 4:
            return hidden
        raise ValueError(f'Segformer encoder must output 4 stage features, got {len(hidden)}')


class ConvNextFeatureBackbone(nn.Module):
    """HuggingFace ConvNeXt encoder (convnextv2_nano via timm below)."""

    def __init__(self, variant='convnext_tiny', in_channels=3):
        super().__init__()
        variant = _normalize_backbone_name(variant)
        settings = {'convnext_tiny': dict(depths=[3, 3, 9, 3], hidden_sizes=[96, 192, 384, 768])}
        if variant not in settings:
            raise ValueError(f'Unsupported HuggingFace ConvNeXt variant: {variant}')
        if ConvNextConfig is None or ConvNextModel is None:
            raise ImportError('HuggingFace ConvNeXt backbone requires transformers. Install: pip install transformers')
        s = settings[variant]
        config = ConvNextConfig(
            num_channels=in_channels, depths=s['depths'], hidden_sizes=s['hidden_sizes'],
            patch_size=4, out_features=['stage1', 'stage2', 'stage3', 'stage4'],
        )
        self.model = ConvNextModel(config)
        self.feature_info = SimpleFeatureInfo(config.hidden_sizes)

    def forward(self, x):
        outputs = self.model(pixel_values=x, output_hidden_states=True, return_dict=True)
        hidden = list(outputs.hidden_states)
        if len(hidden) >= 5:
            return hidden[1:5]
        return hidden[-4:]


class TimmFeatureBackbone(nn.Module):
    """timm features_only backbone (used for convnextv2_nano)."""

    def __init__(self, variant, in_channels=3):
        super().__init__()
        if timm is None:
            raise ImportError('timm is required for convnextv2_nano. Install: pip install timm')
        self.model = timm.create_model(
            variant, pretrained=False, features_only=True,
            out_indices=(0, 1, 2, 3), in_chans=in_channels,
        )
        self.feature_info = SimpleFeatureInfo(self.model.feature_info.channels())

    def forward(self, x):
        return self.model(x)


def create_feature_backbone(backbone, in_channels=3):
    """Build a real encoder; never returns a fallback toy backbone."""
    backbone = _normalize_backbone_name(backbone)
    if backbone in ('mit_b0', 'mit_b1'):
        if SegformerConfig is None or SegformerModel is None:
            raise ImportError('MiT encoder requires transformers. Install: pip install transformers')
        return SegformerFeatureBackbone(backbone, in_channels=in_channels)
    if backbone == 'convnext_tiny':
        if ConvNextConfig is None or ConvNextModel is None:
            raise ImportError('ConvNeXt encoder requires transformers. Install: pip install transformers')
        return ConvNextFeatureBackbone(backbone, in_channels=in_channels)
    if backbone in ('convnextv2_nano',):
        if timm is None:
            raise ImportError('convnextv2_nano requires timm. Install: pip install timm')
        return TimmFeatureBackbone(backbone, in_channels=in_channels)
    raise ValueError(f'Unsupported backbone: {backbone}. Supported: mit_b1, convnextv2_nano.')