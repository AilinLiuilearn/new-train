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


def _strict_load_into_module(module, state_dict, name):
    msg = module.load_state_dict(state_dict, strict=True)
    if msg.missing_keys or msg.unexpected_keys:
        raise RuntimeError(
            f"[STAGE1 INIT] {name}: missing={msg.missing_keys} "
            f"unexpected={msg.unexpected_keys}"
        )
    return len(state_dict)


def _extract_legacy_unimodal_state(checkpoint, prefix):
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        return None
    return {
        key[len(prefix):]: value
        for key, value in model_state.items()
        if key.startswith(prefix)
    }


def _resolve_stage1_module_state(checkpoint, modality, strict):
    """Return the module state dict for a Stage-1 unimodal checkpoint.

    New Stage-1 checkpoints store explicit 'encoder' / 'ct_align' entries.
    Legacy CT-only checkpoints only store the full 'model' state dict; in that
    case the encoder (and ct_align) tensors are extracted by key prefix.
    """
    if modality == "ct":
        encoder_state = checkpoint.get("encoder")
        if not isinstance(encoder_state, dict):
            encoder_state = _extract_legacy_unimodal_state(checkpoint, "enc_ct.")
            if encoder_state is None:
                if strict:
                    raise RuntimeError(
                        "[STAGE1 INIT] CT checkpoint has no encoder weights"
                    )
                return None, None
        align_state = checkpoint.get("ct_align")
        if not isinstance(align_state, dict):
            align_state = _extract_legacy_unimodal_state(checkpoint, "ct_align.")
            if align_state is None and strict:
                raise RuntimeError(
                    "[STAGE1 INIT] CT checkpoint has no ct_align weights"
                )
        return encoder_state, align_state
    encoder_state = checkpoint.get("encoder")
    if not isinstance(encoder_state, dict):
        encoder_state = _extract_legacy_unimodal_state(checkpoint, "enc_pet.")
        if encoder_state is None:
            if strict:
                raise RuntimeError(
                    "[STAGE1 INIT] PET checkpoint has no encoder weights"
                )
            return None, None
    return encoder_state, None


def load_stage1_unimodal_initialization(model, ct_checkpoint, pet_checkpoint, strict=True):
    """Load Stage-1 unimodal expert weights into the Stage-2 joint model.

    Only modality-specific encoders (+ CT align) are loaded. Stage-1 decoders
    are NEVER loaded: the Stage-2 shared decoder keeps its seed-determined
    initialization so this experiment isolates encoder pretraining only.
    """
    report = {
        "ct_encoder": False,
        "ct_align": False,
        "pet_encoder": False,
        "shared_decoder": False,
    }
    if ct_checkpoint:
        if not os.path.isfile(ct_checkpoint):
            raise FileNotFoundError(ct_checkpoint)
        checkpoint = torch.load(ct_checkpoint, map_location="cpu", weights_only=False)
        encoder_state, align_state = _resolve_stage1_module_state(
            checkpoint, "ct", strict
        )
        if encoder_state is not None:
            n = _strict_load_into_module(model.enc_ct, encoder_state, "ct_encoder")
            report["ct_encoder"] = True
            print(f"[STAGE1 INIT] ct_encoder loaded_tensors={n}", flush=True)
        if align_state is not None:
            n = _strict_load_into_module(model.ct_align, align_state, "ct_align")
            report["ct_align"] = True
            print(f"[STAGE1 INIT] ct_align loaded_tensors={n}", flush=True)
        print(f"[STAGE1 INIT] ct_checkpoint={ct_checkpoint}", flush=True)
    if pet_checkpoint:
        if not os.path.isfile(pet_checkpoint):
            raise FileNotFoundError(pet_checkpoint)
        checkpoint = torch.load(pet_checkpoint, map_location="cpu", weights_only=False)
        encoder_state, _ = _resolve_stage1_module_state(checkpoint, "pet", strict)
        if encoder_state is not None:
            n = _strict_load_into_module(model.enc_pet, encoder_state, "pet_encoder")
            report["pet_encoder"] = True
            print(f"[STAGE1 INIT] pet_encoder loaded_tensors={n}", flush=True)
        print(f"[STAGE1 INIT] pet_checkpoint={pet_checkpoint}", flush=True)

    print(
        "[STAGE1 INIT] "
        f"ct_encoder={report['ct_encoder']} "
        f"ct_align={report['ct_align']} "
        f"pet_encoder={report['pet_encoder']} "
        "shared_decoder_loaded=False "
        "(Stage-1 decoders are never transferred; encoders remain trainable)",
        flush=True,
    )
    return report


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


def build_mdt_seg_teacher(config, model_state_dict=None):
    from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
    from models.state_guided_expert_fusion import FORMAT_VERSION as MODULE2_FORMAT_VERSION
    from models.state_guided_expert_fusion import PROMPTS as MODULE2_PROMPTS
    from models.state_guided_expert_fusion import load_text_cache
    module2_enabled = bool(getattr(config, 'module2_enabled', False))
    pspi_enabled = bool(getattr(config, 'pspi_enabled', True))
    if module2_enabled and not pspi_enabled:
        raise ValueError('module2_enabled=True requires pspi_enabled=True')
    module2_experts = getattr(config, 'module2_experts_per_group', 2)
    if module2_enabled:
        if type(module2_experts) is not int or module2_experts < 1:
            raise ValueError('module2_experts_per_group must be a positive integer')
    module2_kwargs = None
    module2_from_checkpoint = False
    if module2_enabled:
        # Full-checkpoint recovery: reuse archived text vectors/metadata so
        # evaluation never depends on the external text cache file.
        ckpt_module2_prefix = 'module2.'
        ckpt_text_key = ckpt_module2_prefix + 'text_embeddings'
        ckpt_extra_key = ckpt_module2_prefix + '_extra_state'
        has_ckpt_module2 = (
            isinstance(model_state_dict, dict)
            and ckpt_text_key in model_state_dict
            and ckpt_extra_key in model_state_dict
        )
        if has_ckpt_module2:
            extra = model_state_dict[ckpt_extra_key]
            if not isinstance(extra, dict):
                raise ValueError('module2 _extra_state in checkpoint must be a dict')
            if extra.get('version') != MODULE2_FORMAT_VERSION:
                raise ValueError('module2 checkpoint version mismatch')
            if list(extra.get('prompts', [])) != list(MODULE2_PROMPTS):
                raise ValueError('module2 checkpoint prompt content/order mismatch; rebuild cache')
            ckpt_cfg = dict(extra.get('config', {}))
            if int(ckpt_cfg.get('experts_per_group', -1)) != int(module2_experts):
                raise ValueError(
                    f"module2 experts_per_group mismatch: config={module2_experts} "
                    f"checkpoint={ckpt_cfg.get('experts_per_group')}"
                )
            use_text = bool(ckpt_cfg.get('use_text', False))
            if use_text != bool(getattr(config, 'module2_use_text', True)):
                raise ValueError(
                    f"module2 text-mode mismatch: config={bool(getattr(config, 'module2_use_text', True))} "
                    f"checkpoint={use_text}"
                )
            archived = model_state_dict[ckpt_text_key]
            if not torch.is_tensor(archived):
                raise ValueError('module2 text_embeddings in checkpoint must be a Tensor')
            module2_kwargs = dict(ckpt_cfg)
            module2_kwargs['text_embeddings'] = archived.detach().float().cpu().clone() if use_text else None
            module2_kwargs['text_metadata'] = dict(extra.get('text_metadata', {}))
            module2_from_checkpoint = True
            print('[Module2] text vectors restored from checkpoint state_dict (no cache file needed)')
        elif bool(getattr(config, 'module2_use_text', True)):
            cache_path = getattr(config, 'module2_text_cache', None)
            if not cache_path:
                raise ValueError(
                    'module2_use_text=True requires module2_text_cache; generate with: '
                    'python models/state_guided_expert_fusion.py --cache-text pretrained/module2_text_cache.pt '
                    '--backend biomedclip --text-tower-path pretrained/biomedbert_text_tower '
                    '--biomedclip-path pretrained/biomedclip_model --device cpu'
                )
            if not os.path.isfile(cache_path):
                raise FileNotFoundError(
                    f'Module-2 text cache not found: {cache_path}. Generate with: '
                    'python models/state_guided_expert_fusion.py --cache-text pretrained/module2_text_cache.pt '
                    '--backend biomedclip --text-tower-path pretrained/biomedbert_text_tower '
                    '--biomedclip-path pretrained/biomedclip_model --device cpu'
                )
            embeddings, metadata = load_text_cache(cache_path)
            module2_kwargs = {
                'use_text': True,
                'text_embeddings': embeddings,
                'text_metadata': metadata,
                'experts_per_group': int(module2_experts),
            }
            print(f'[Module2] text cache loaded: {cache_path} dim={embeddings.shape[1]}')
        else:
            module2_kwargs = {
                'use_text': False,
                'experts_per_group': int(module2_experts),
            }
            print('[Module2] text disabled; no cache file accessed')
    model = DualSharedAddPETCTBaseline(
        ct_backbone=getattr(config, 'ct_backbone', 'convnextv2_nano'),
        pet_backbone=getattr(config, 'pet_backbone', 'mit_b1'),
        ct_pretrained_path=getattr(config, 'ct_pretrained_path', None),
        pet_pretrained_path=getattr(config, 'pet_pretrained_path', None),
        in_channels=3,
        out_channels=1,
        decoder_channels=getattr(config, 'decoder_channels', (512, 256, 128, 64)),
        use_deep_supervision=bool(getattr(config, 'use_deep_supervision', False) or getattr(config, 'deep_supervision', False)),
        pspi_enabled=getattr(config, 'pspi_enabled', True),
        pspi_num_clusters=getattr(config, 'pspi_num_clusters', 6),
        pspi_build_stage=getattr(config, 'pspi_build_stage', 4),
        pspi_cluster_max_iter=getattr(config, 'pspi_cluster_max_iter', 25),
        pspi_outlier_discard_rate=getattr(config, 'pspi_outlier_discard_rate', 0.05),
        pspi_bank_update_mode=getattr(config, 'pspi_bank_update_mode', 'direct'),
        pspi_ema_momentum=getattr(config, 'pspi_ema_momentum', 0.95),
        pspi_retrieval_temperature=getattr(config, 'pspi_retrieval_temperature', 0.1),
        pspi_proto_temperature=getattr(config, 'pspi_proto_temperature', 0.02),
        pspi_collect_candidates=getattr(config, 'pspi_collect_candidates', True),
        pspi_prior_scale_enabled=getattr(config, 'pspi_prior_scale_enabled', True),
        pspi_prior_scale_init=getattr(config, 'pspi_prior_scale_init', 0.1),
        module2_enabled=module2_enabled,
        module2_kwargs=module2_kwargs,
    )
    if bool(getattr(config, 'stage1_init_enabled', False)):
        load_stage1_unimodal_initialization(
            model,
            getattr(config, 'stage1_ct_checkpoint', None),
            getattr(config, 'stage1_pet_checkpoint', None),
            strict=bool(getattr(config, 'stage1_init_strict', True)),
        )
        assert all(p.requires_grad for p in model.enc_ct.parameters())
        assert all(p.requires_grad for p in model.enc_pet.parameters())
        assert all(p.requires_grad for p in model.ct_align.parameters())
    pspi_enabled = bool(getattr(config, 'pspi_enabled', True))
    module2_enabled = bool(getattr(config, 'module2_enabled', False))
    module2_params = sum(p.numel() for p in model.module2.parameters()) if model.module2 is not None else 0
    module2_trainable = sum(p.numel() for p in model.module2.parameters() if p.requires_grad) if model.module2 is not None else 0
    if model.module2 is not None:
        fusion_name = type(model.module2).__name__
    else:
        fusion_name = 'AddFusion'
    print(
        f'[dual_shared_add_baseline] ct={getattr(config, "ct_backbone", "convnextv2_nano")} '
        f'pet={getattr(config, "pet_backbone", "mit_b1")} '
        f'fusion={fusion_name} '
        f'shared_decoder=UNetStyleDecoder '
        f'deep_supervision={bool(getattr(config, "use_deep_supervision", False) or getattr(config, "deep_supervision", False))}'
    )
    print(
        f'[Module2] enabled={module2_enabled} '
        f'use_text={bool(getattr(config, "module2_use_text", True))} '
        f'experts_per_group={getattr(config, "module2_experts_per_group", 2)} '
        f'params={module2_params} trainable={module2_trainable} '
        f'anatomical_personalization={"module2_spatial_gamma_beta" if model.module2 is not None and model.module2.personalization else ("none" if model.module2 is None else "disabled")} '
        f'from_checkpoint={module2_from_checkpoint}'
    )
    if model.module2 is not None:
        print(
            f'[Module2][PriorScale] requested_enabled={model.requested_prior_scale_enabled} '
            f'effective_enabled={model.effective_prior_scale_enabled} '
            f'missing_prior_logits=None (replaced by routing weight a_P)'
        )
    if pspi_enabled:
        fusion_desc = 'downstream_fusion=AddFusion' if model.module2 is None else 'downstream_fusion=StateGuidedExpertFusion'
    else:
        fusion_desc = 'baseline_fusion=AddFusion'
    if model.module2 is not None:
        prior_scale_desc = (
            f'prior_scale_type=module2_routing_aP '
            f'prior_scale_requested={model.requested_prior_scale_enabled} '
            f'prior_scale_effective=False '
            f'missing_prior_alpha=disabled_by_module2 '
            f'module2_base_pet=raw_prior'
        )
    else:
        prior_scale_desc = (
            f'prior_scale_type=per_scale_scalar '
            f'prior_scale_init={getattr(config, "pspi_prior_scale_init", 0.1)} '
            f'prior_scale_enabled={bool(getattr(config, "pspi_prior_scale_enabled", True))}'
        )
    print(
        f'[PSPI] enabled={pspi_enabled} '
        f'module1=paired_ct_pet_prototype_prior_retrieval '
        f'clustering=spherical_cosine '
        f'cluster_init=deterministic_mean_farthest '
        f'outlier_filter=cosine_top5_percent '
        f'retrieval=cosine_soft '
        f'retrieval_temperature={getattr(config, "pspi_retrieval_temperature", 0.1)} '
        f'personalization=none '
        f'(module1_personalization=none; Module-2 anatomical personalization logged separately) '
        f'prototype_loss=pet_multi_positive_contrastive '
        f'proto_temperature={getattr(config, "pspi_proto_temperature", 0.02)} '
        f'proto_weight={getattr(config, "pspi_proto_contrastive_weight", 0.01)} '
        f'reconstruction_loss=none '
        f'cold_start=epoch1 '
        f'bank_update={getattr(config, "pspi_bank_update_mode", "direct")} '
        f'ema_momentum={getattr(config, "pspi_ema_momentum", 0.95)} '
        f'K={getattr(config, "pspi_num_clusters", 6)} '
        f'build_stage=S{getattr(config, "pspi_build_stage", 4)} '
        f'full_path={"module2_weighted_CT_plus_real_PET" if model.module2 is not None else "raw_CT_plus_real_PET"} '
        f'missing_boundary={"module2_weighted_CT_plus_raw_prior" if model.module2 is not None else "CT_plus_scale_weighted_PET_prior"} '
        f'{prior_scale_desc} '
        f'{fusion_desc} '
        f'decoder=UNetStyleDecoder'
    )
    return {'model': model}
