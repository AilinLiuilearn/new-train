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


def _sanitize_state_dict(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ('module.', 'backbone.', 'visual.'):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    return cleaned


def load_local_weights_safe(model, path, name='Encoder'):
    if not path:
        print(f'[-] {name}: pretrained path not provided; training from scratch')
        return
    if not os.path.exists(path):
        print(f'[-] {name}: pretrained path not found: {path}; training from scratch')
        return
    source_path = path
    if os.path.isdir(path):
        print(f'[+] {name}: pretrained path is a directory: {path}')
        found = False
        for cand in ('pytorch_model.bin', 'model.safetensors', 'mit_b0.pth', 'mit_b0.bin', 'mit_b0.pt', 'mit-b0.pth', 'mit-b0.bin', 'mit-b0.pt', 'mit_b1.pth', 'mit_b1.bin', 'mit_b1.pt', 'mit-b1.pth', 'mit-b1.bin', 'mit-b1.pt', 'pvt_v2_b1.pth', 'pvt_v2_b1.bin', 'pvt_v2_b1.pt', 'convnext_tiny.pth', 'convnext_tiny.bin', 'convnext_tiny.pt', 'convnext_nano.pth', 'convnext_nano.bin', 'convnext_nano.pt', 'convnextv2_nano.pth', 'convnextv2_nano.bin', 'convnextv2_nano.pt'):
            full = os.path.join(path, cand)
            if os.path.exists(full):
                path = full
                found = True
                break
        if not found:
            print(f'[-] {name}: no supported weight file found under {path}; training from scratch')
            return
        print(f'[+] {name}: resolved weight file {path}')
    else:
        print(f'[+] {name}: pretrained file {path}')
    print(f'[+] {name}: loading local weights from {path}')
    if str(path).endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(path, device='cpu')
    else:
        try:
            state_dict = torch.load(path, map_location='cpu', weights_only=False)
        except Exception:
            try:
                state_dict = torch.load(path, map_location='cpu')
            except Exception:
                from safetensors.torch import load_file
                state_dict = load_file(path, device='cpu')
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
    model_total = len(model_state)
    ckpt_total = len(state_dict)
    if loadable:
        msg = model.load_state_dict(loadable, strict=False)
        loaded_tensors = len(loadable)
        loaded_elements = sum(v.numel() for v in loadable.values())
        skipped_tensors = len(skipped)
        skipped_elements = sum(v.numel() for k, v in state_dict.items() if k in skipped and torch.is_tensor(v))
        print(f'[PRETRAIN] {name}: success=True')
        print(f'[PRETRAIN] {name}: source={source_path}')
        print(f'[PRETRAIN] {name}: weight_file={path}')
        print(f'[PRETRAIN] {name}: checkpoint_tensors={ckpt_total}, model_tensors={model_total}')
        print(f'[PRETRAIN] {name}: loaded_tensors={loaded_tensors}, loaded_elements={loaded_elements}')
        print(f'[PRETRAIN] {name}: skipped_tensors={skipped_tensors}, skipped_elements={skipped_elements}')
        print(f'[PRETRAIN] {name}: missing_after_load={len(msg.missing_keys)}, unexpected_after_load={len(msg.unexpected_keys)}')
        if skipped:
            print(f'[PRETRAIN] {name}: skipped_examples={skipped[:8]}')
    else:
        print(f'[PRETRAIN] {name}: success=False')
        print(f'[PRETRAIN] {name}: source={source_path}')
        print(f'[PRETRAIN] {name}: weight_file={path}')
        print(f'[PRETRAIN] {name}: checkpoint_tensors={ckpt_total}, model_tensors={model_total}')
        print(f'[PRETRAIN] {name}: loaded_tensors=0, skipped_tensors={len(skipped)}')
        if skipped:
            print(f'[PRETRAIN] {name}: skipped_examples={skipped[:8]}')
        print(f'[-] {name}: no compatible tensors were loaded; training this encoder from scratch')


def _normalize_backbone_name(backbone):
    backbone = str(backbone).strip().replace('\u200b', '').replace('\ufeff', '')
    aliases = {
        'mit-b0': 'mit_b0',
        'segformer-b0': 'mit_b0',
        'nvidia/mit-b0': 'mit_b0',
        'mit-b1': 'mit_b1',
        'segformer-b1': 'mit_b1',
        'nvidia/mit-b1': 'mit_b1',
        'convnext-t': 'convnext_tiny',
        'convnext-tiny': 'convnext_tiny',
        'convnext_t': 'convnext_tiny',
        'convnextv2-nano': 'convnextv2_nano',
        'convnext_v2_nano': 'convnextv2_nano',
        'pvt-b1': 'pvt_v2_b1',
        'pvt_b1': 'pvt_v2_b1',
    }
    return aliases.get(backbone, backbone)


class SimpleFeatureInfo:
    def __init__(self, channels):
        self._channels = list(channels)

    def channels(self):
        return self._channels


class SegformerFeatureBackbone(nn.Module):
    def __init__(self, variant='mit_b0', in_channels=3):
        super().__init__()
        variant = _normalize_backbone_name(variant)
        mit_settings = {
            'mit_b0': dict(depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256], num_attention_heads=[1, 2, 5, 8], drop_path_rate=0.1),
            'mit_b1': dict(depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 320, 512], num_attention_heads=[1, 2, 5, 8], drop_path_rate=0.1),
        }
        if variant not in mit_settings:
            raise ValueError(f'Unsupported MiT variant: {variant}')
        if SegformerConfig is None or SegformerModel is None:
            raise ImportError('Segformer MiT backbone requires transformers. Install it with: pip install transformers')
        settings = mit_settings[variant]
        config = SegformerConfig(
            num_channels=in_channels,
            depths=settings['depths'],
            sr_ratios=[8, 4, 2, 1],
            hidden_sizes=settings['hidden_sizes'],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            num_attention_heads=settings['num_attention_heads'],
            mlp_ratios=[4, 4, 4, 4],
            hidden_act='gelu',
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            classifier_dropout_prob=0.1,
            initializer_range=0.02,
            drop_path_rate=settings['drop_path_rate'],
            reshape_last_stage=True,
            output_hidden_states=True,
        )
        self.model = SegformerModel(config)
        self.feature_info = SimpleFeatureInfo(config.hidden_sizes)

    def forward(self, x):
        outputs = self.model(pixel_values=x, output_hidden_states=True, return_dict=True)
        hidden_states = list(outputs.hidden_states or [])
        if len(hidden_states) >= 5:
            return hidden_states[1:5]
        if len(hidden_states) == 4:
            return hidden_states
        raise ValueError(f'Segformer encoder must output 4 stage features, got {len(hidden_states)}')


class ConvNextFeatureBackbone(nn.Module):
    def __init__(self, variant='convnext_tiny', in_channels=3):
        super().__init__()
        variant = _normalize_backbone_name(variant)
        convnext_settings = {
            'convnext_tiny': dict(depths=[3, 3, 9, 3], hidden_sizes=[96, 192, 384, 768]),
        }
        if variant not in convnext_settings:
            raise ValueError(f'Unsupported HuggingFace ConvNeXt variant: {variant}')
        if ConvNextConfig is None or ConvNextModel is None:
            raise ImportError('HuggingFace ConvNeXt backbone requires transformers. Install it with: pip install transformers')
        settings = convnext_settings[variant]
        config = ConvNextConfig(num_channels=in_channels, depths=settings['depths'], hidden_sizes=settings['hidden_sizes'], patch_size=4, out_features=['stage1', 'stage2', 'stage3', 'stage4'])
        self.model = ConvNextModel(config)
        self.feature_info = SimpleFeatureInfo(config.hidden_sizes)

    def forward(self, x):
        outputs = self.model(pixel_values=x, output_hidden_states=True, return_dict=True)
        hidden_states = list(outputs.hidden_states)
        if len(hidden_states) >= 5:
            return hidden_states[1:5]
        return hidden_states[-4:]


def _get_backbone_out_indices(backbone):
    backbone = _normalize_backbone_name(backbone)
    if backbone in ('pvt_v2_b1', 'mit_b0', 'mit_b1'):
        return (0, 1, 2, 3)
    if backbone in ('convnext_tiny', 'convnext_nano', 'convnextv2_nano', 'convnextv2_atto', 'convnextv2_femto', 'convnextv2_pico'):
        return (0, 1, 2, 3)
    raise ValueError(f'Unsupported backbone: {backbone}. Supported: pvt_v2_b1, mit_b0, mit_b1, convnext_tiny, convnext_nano, convnextv2_nano.')


class FallbackFeatureBackbone(nn.Module):
    def __init__(self, in_channels=3, channels=(32, 64, 160, 256)):
        super().__init__()
        self.feature_info = SimpleFeatureInfo(channels)
        self.stem = ConvBNAct(in_channels, channels[0], kernel_size=3, stride=2)
        self.stage1 = ConvBNAct(channels[0], channels[0], kernel_size=3, stride=2)
        self.stage2 = ConvBNAct(channels[0], channels[1], kernel_size=3, stride=2)
        self.stage3 = ConvBNAct(channels[1], channels[2], kernel_size=3, stride=2)
        self.stage4 = ConvBNAct(channels[2], channels[3], kernel_size=3, stride=2)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


def create_feature_backbone(backbone, in_channels=3):
    backbone = _normalize_backbone_name(backbone)
    if backbone in ('mit_b0', 'mit_b1'):
        if SegformerConfig is None or SegformerModel is None:
            return FallbackFeatureBackbone(in_channels=in_channels, channels=(32, 64, 160, 256) if backbone == 'mit_b0' else (64, 128, 320, 512))
        return SegformerFeatureBackbone(backbone, in_channels=in_channels)
    if backbone == 'convnext_tiny':
        if ConvNextConfig is None or ConvNextModel is None:
            return FallbackFeatureBackbone(in_channels=in_channels, channels=(96, 192, 384, 768))
        return ConvNextFeatureBackbone(backbone, in_channels=in_channels)
    if timm is None:
        return FallbackFeatureBackbone(in_channels=in_channels)
    return timm.create_model(backbone, pretrained=False, features_only=True, out_indices=_get_backbone_out_indices(backbone), in_chans=in_channels)


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


_STAGE1_CONFIG_KEYS = (
    'ct_backbone',
    'pet_backbone',
    'cppi_num_clusters',
    'cppi_build_stage',
    'decoder_channels',
)


def _peek_checkpoint_payload(checkpoint_path):
    try:
        payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location='cpu')
    return payload if isinstance(payload, dict) else {'model': payload}


def _align_stage1_config_from_checkpoint(config, stage1_checkpoint):
    """Prefer Stage-1 checkpoint config for architecture-critical fields."""
    payload = _peek_checkpoint_payload(stage1_checkpoint)
    ckpt_cfg = payload.get('config')
    if not isinstance(ckpt_cfg, dict):
        print(
            f'[STAGE2][WARN] stage1_checkpoint has no embedded config; '
            f'using current CLI values as-is: {stage1_checkpoint}',
            flush=True,
        )
        return payload

    mismatches = []
    for key in _STAGE1_CONFIG_KEYS:
        if key not in ckpt_cfg:
            continue
        cur = getattr(config, key, None)
        ckpt_val = ckpt_cfg[key]
        # Normalize list/tuple comparisons for decoder_channels etc.
        cur_cmp = list(cur) if isinstance(cur, (list, tuple)) else cur
        ckpt_cmp = list(ckpt_val) if isinstance(ckpt_val, (list, tuple)) else ckpt_val
        if cur_cmp != ckpt_cmp:
            mismatches.append((key, cur, ckpt_val))
            setattr(config, key, ckpt_val)

    if mismatches:
        print('[STAGE2][ERROR] Stage1 checkpoint config mismatch; overriding with checkpoint values:', flush=True)
        for key, cur, ckpt_val in mismatches:
            print(f'  - {key}: cli/default={cur} -> checkpoint={ckpt_val}', flush=True)
    else:
        print('[STAGE2] Stage1 checkpoint config matches current architecture fields', flush=True)
    return payload


def _build_dual_shared_add_baseline(config):
    from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
    cppi_num_clusters = getattr(config, 'cppi_num_clusters', 6)
    cppi_build_stage = getattr(config, 'cppi_build_stage', 4)
    cppi_output_dir = os.path.join(config.checkpoint_dir, 'cppi')
    no_encoder_pretrained = bool(getattr(config, 'no_encoder_pretrained', False))
    model = DualSharedAddPETCTBaseline(
        ct_backbone=getattr(config, 'ct_backbone', 'convnextv2_nano'),
        pet_backbone=getattr(config, 'pet_backbone', 'mit_b1'),
        ct_pretrained_path=None if no_encoder_pretrained else getattr(config, 'ct_pretrained_path', None),
        pet_pretrained_path=None if no_encoder_pretrained else getattr(config, 'pet_pretrained_path', None),
        in_channels=3,
        out_channels=1,
        decoder_channels=getattr(config, 'decoder_channels', (512, 256, 128, 64)),
        use_deep_supervision=bool(getattr(config, 'use_deep_supervision', False) or getattr(config, 'deep_supervision', False)),
        cppi_num_clusters=cppi_num_clusters,
        cppi_build_stage=cppi_build_stage,
        cppi_output_dir=cppi_output_dir,
    )
    print(
        f'[dual_shared_add_baseline] ct={getattr(config, "ct_backbone", "convnextv2_nano")} '
        f'pet={getattr(config, "pet_backbone", "mit_b1")} '
        f'fusion=state_aware_weighted_add shared_decoder=UNetStyleDecoder '
        f'deep_supervision={bool(getattr(config, "use_deep_supervision", False) or getattr(config, "deep_supervision", False))}'
    )
    print(f'[CPPI] enabled=True')
    print(f'[CPPI] num_clusters={cppi_num_clusters}')
    print(f'[CPPI] build_stage={cppi_build_stage}')
    print(f'[CPPI] output_dir={cppi_output_dir}')
    return model


def _summarize_trainable_names(names, max_groups=24):
    groups = []
    for name in names:
        top = name.split('.', 2)[0]
        if top not in groups:
            groups.append(top)
        if len(groups) >= max_groups:
            break
    return groups


def _build_prompt_role_expert_stage2(config):
    from models.stage2_prompt_role_expert_fusion import (
        PromptRoleExpertStage2Seg,
        load_stage1_checkpoint,
    )

    stage1_checkpoint = getattr(config, 'stage1_checkpoint', None)
    if not stage1_checkpoint:
        raise ValueError(
            'model_arch=prompt_role_expert_stage2 requires --stage1_checkpoint '
            '(frozen Stage-1 best checkpoint). Do NOT pass Stage-1 weights via --resume_checkpoint.'
        )
    if not os.path.exists(stage1_checkpoint):
        raise FileNotFoundError(f'stage1_checkpoint not found: {stage1_checkpoint}')

    # Align architecture fields with the Stage-1 checkpoint before construction.
    _align_stage1_config_from_checkpoint(config, stage1_checkpoint)

    strategy = str(getattr(config, 'stage2_train_strategy', 'alternating_frozen')).strip()
    allow_affine = strategy in ('paired_joint_affine', 'paired_anga_affine')
    if strategy == 'paired_anga_affine':
        tau = float(getattr(config, 'stage2_anga_tau', 0.7))
        if not (0.0 < tau < 1.0):
            raise ValueError(f'stage2_anga_tau must satisfy 0 < tau < 1, got {tau}')

    stage1 = _build_dual_shared_add_baseline(config)
    load_stage1_checkpoint(stage1, stage1_checkpoint, strict=True)

    pm = stage1.prototype_memory
    bank_ready = bool(pm.bank_ready)
    bank_version = int(pm.bank_version.item()) if hasattr(pm, 'bank_version') else -1
    ready = getattr(pm, 'prototype_ready', None)
    ready_count = int(ready.sum().item()) if torch.is_tensor(ready) else 0

    print(f'[STAGE2] stage1_checkpoint={stage1_checkpoint}', flush=True)
    print(f'[STAGE2] train_strategy={strategy} allow_affine_trainable={allow_affine}', flush=True)
    print(f'[STAGE2] CPPI bank_ready={bank_ready} bank_version={bank_version} ready_slots={ready_count}', flush=True)
    if not bank_ready:
        raise RuntimeError(
            'Stage-2 requires a ready frozen CPPI bank in stage1_checkpoint, but '
            f'prototype_memory.bank_ready=False (bank_version={bank_version}).'
        )

    model = PromptRoleExpertStage2Seg(
        stage1_model=stage1,
        channels=(64, 128, 320, 512),
        decoder_channels=getattr(config, 'decoder_channels', (512, 256, 128, 64)),
        expert_dim=int(getattr(config, 'stage2_expert_dim', 128)),
        atom_num=int(getattr(config, 'stage2_atom_num', 32)),
        atom_dim=int(getattr(config, 'stage2_atom_dim', 256)),
        prompt_hidden_channels=int(getattr(config, 'stage2_prompt_hidden_channels', 64)),
        mlp_ratio=float(getattr(config, 'stage2_mlp_ratio', 2.0)),
        dropout=float(getattr(config, 'stage2_dropout', 0.0)),
        adapter_bottlenecks=getattr(config, 'stage2_adapter_bottlenecks', (64, 32, 16, 8)),
        external_prompt_dim=None,  # Text OFF for the first experiment.
        require_ready_cppi_for_missing=True,
        allow_affine_trainable=allow_affine,
    )
    model.assert_stage1_boundary_contract(allow_affine=allow_affine)

    stage1_total = sum(p.numel() for p in model.stage1.parameters())
    affine_trainable = sum(p.numel() for p in model.stage1.pet_calibration.parameters() if p.requires_grad)
    stage1_other_trainable = sum(
        p.numel()
        for n, p in model.stage1.named_parameters()
        if p.requires_grad and not n.startswith('pet_calibration.')
    )
    stage2_modules_trainable = sum(
        p.numel() for p in list(model.role_fusion.parameters()) + list(model.decoder_adapters.parameters()) if p.requires_grad
    )
    trainable_names = model.trainable_parameter_names()
    print(f'[STAGE2] Stage1 total params={stage1_total}', flush=True)
    print(f'[STAGE2] Stage1 affine trainable params={affine_trainable}', flush=True)
    print(f'[STAGE2] Stage1 non-affine trainable params={stage1_other_trainable}', flush=True)
    print(f'[STAGE2] SPRE+adapters trainable params={stage2_modules_trainable}', flush=True)
    print(
        f'[STAGE2] Trainable name groups={_summarize_trainable_names(trainable_names)} '
        f'(num_tensors={len(trainable_names)})',
        flush=True,
    )
    if stage1_other_trainable != 0:
        raise RuntimeError(f'Stage-1 non-affine must stay frozen, got trainable={stage1_other_trainable}')
    if allow_affine and affine_trainable <= 0:
        raise RuntimeError('Affine strategy selected but pet_calibration has no trainable params')
    if not allow_affine and affine_trainable != 0:
        raise RuntimeError(f'Frozen strategy but affine trainable={affine_trainable}')
    if stage2_modules_trainable <= 0:
        raise RuntimeError('Stage-2 has no trainable SPRE/adapter parameters')
    return model


def build_mdt_seg_teacher(config):
    model_arch = str(getattr(config, 'model_arch', 'dual_shared_add_baseline')).strip()
    if model_arch == 'dual_shared_add_baseline':
        model = _build_dual_shared_add_baseline(config)
        return {'model': model}
    if model_arch == 'prompt_role_expert_stage2':
        model = _build_prompt_role_expert_stage2(config)
        return {'model': model}
    raise ValueError(
        f'Unsupported model_arch={model_arch!r}; '
        f'expected dual_shared_add_baseline or prompt_role_expert_stage2'
    )
