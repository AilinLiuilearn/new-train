import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.mpa_bioclip import MPABioCLIPSumFusion, get_bioclip_text_feature

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
        print(f'[-] {name}: No pretrained path provided. Training from scratch.')
        return
    if not os.path.exists(path):
        print(f'[-] {name}: Path not found {path}. Training from scratch.')
        return
    if os.path.isdir(path):
        found = False
        for cand in ('pytorch_model.bin', 'model.safetensors',
                     'mit_b0.pth', 'mit_b0.bin', 'mit_b0.pt',
                     'mit-b0.pth', 'mit-b0.bin', 'mit-b0.pt',
                     'mit_b1.pth', 'mit_b1.bin', 'mit_b1.pt',
                     'mit-b1.pth', 'mit-b1.bin', 'mit-b1.pt',
                     'pvt_v2_b1.pth', 'pvt_v2_b1.bin', 'pvt_v2_b1.pt',
                     'convnext_tiny.pth', 'convnext_tiny.bin', 'convnext_tiny.pt',
                     'convnext_nano.pth', 'convnext_nano.bin', 'convnext_nano.pt'):
            full = os.path.join(path, cand)
            if os.path.exists(full):
                path = full
                found = True
                break
        if not found:
            print(f'[-] {name}: No supported weight file found in {path}. Training from scratch.')
            return
    print(f'[+] {name}: Loading local weights from {path}')
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
    msg = model.load_state_dict(loadable, strict=False)
    print(f'[+] {name} loaded params: {len(loadable)}, skipped: {len(skipped)}')
    if skipped:
        print(f'[+] {name} skipped examples: {skipped[:8]}')
    print(f'[+] {name} load status: {msg}')


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
            'mit_b0': dict(
                depths=[2, 2, 2, 2],
                hidden_sizes=[32, 64, 160, 256],
                num_attention_heads=[1, 2, 5, 8],
                drop_path_rate=0.1,
            ),
            'mit_b1': dict(
                depths=[2, 2, 2, 2],
                hidden_sizes=[64, 128, 320, 512],
                num_attention_heads=[1, 2, 5, 8],
                drop_path_rate=0.1,
            ),
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
        config = ConvNextConfig(
            num_channels=in_channels,
            depths=settings['depths'],
            hidden_sizes=settings['hidden_sizes'],
            patch_size=4,
            out_features=['stage1', 'stage2', 'stage3', 'stage4'],
        )
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
    if backbone in ('convnext_tiny', 'convnext_nano'):
        return (0, 1, 2, 3)
    raise ValueError(f'Unsupported backbone: {backbone}. Supported: pvt_v2_b1, mit_b0, mit_b1, convnext_tiny, convnext_nano.')


def create_feature_backbone(backbone, in_channels=3):
    backbone = _normalize_backbone_name(backbone)
    if backbone in ('mit_b0', 'mit_b1'):
        return SegformerFeatureBackbone(backbone, in_channels=in_channels)
    if backbone == 'convnext_tiny':
        return ConvNextFeatureBackbone(backbone, in_channels=in_channels)
    return timm.create_model(
        backbone,
        pretrained=False,
        features_only=True,
        out_indices=_get_backbone_out_indices(backbone),
        in_chans=in_channels,
    )


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LightConcatUNetDecoder(nn.Module):
    def __init__(self, encoder_channels, out_channels=1, decoder_channels=(512, 256, 128, 64)):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        d4, d3, d2, d1 = decoder_channels
        self.proj4 = ConvBNAct(c4, d4, kernel_size=1)
        self.proj3 = ConvBNAct(c3, d3, kernel_size=1)
        self.proj2 = ConvBNAct(c2, d2, kernel_size=1)
        self.proj1 = ConvBNAct(c1, d1, kernel_size=1)
        self.fuse3 = nn.Sequential(ConvBNAct(d4 + d3, d3), ConvBNAct(d3, d3))
        self.fuse2 = nn.Sequential(ConvBNAct(d3 + d2, d2), ConvBNAct(d2, d2))
        self.fuse1 = nn.Sequential(ConvBNAct(d2 + d1, d1), ConvBNAct(d1, d1))
        self.head4 = nn.Conv2d(d4, out_channels, 1)
        self.head3 = nn.Conv2d(d3, out_channels, 1)
        self.head2 = nn.Conv2d(d2, out_channels, 1)
        self.head1 = nn.Conv2d(d1, out_channels, 1)

    @staticmethod
    def _upsample_to(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    @staticmethod
    def _upsample_size(x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, features, target_size):
        x1, x2, x3, x4 = features
        d4 = self.proj4(x4)
        s3 = self.proj3(x3)
        d3 = self.fuse3(torch.cat([self._upsample_to(d4, s3), s3], dim=1))
        s2 = self.proj2(x2)
        d2 = self.fuse2(torch.cat([self._upsample_to(d3, s2), s2], dim=1))
        s1 = self.proj1(x1)
        d1 = self.fuse1(torch.cat([self._upsample_to(d2, s1), s1], dim=1))
        p1 = self._upsample_size(self.head1(d1), target_size)
        p2 = self._upsample_size(self.head2(d2), target_size)
        p3 = self._upsample_size(self.head3(d3), target_size)
        p4 = self._upsample_size(self.head4(d4), target_size)
        return {'preds': [p1, p2, p3, p4], 'pred': p1}


class AdditiveProjectionFusion(nn.Module):
    def __init__(self, ct_channels, pet_channels, out_channels):
        super().__init__()
        if not (len(ct_channels) == len(pet_channels) == len(out_channels)):
            raise ValueError('ct_channels, pet_channels and out_channels must have the same length.')
        self.ct_proj = nn.ModuleList([ConvBNAct(cin, cout, kernel_size=1) for cin, cout in zip(ct_channels, out_channels)])
        self.pet_proj = nn.ModuleList([ConvBNAct(cin, cout, kernel_size=1) for cin, cout in zip(pet_channels, out_channels)])

    def forward(self, ct_feats, pet_feats):
        fused = []
        for ct_proj, pet_proj, ct_feat, pet_feat in zip(self.ct_proj, self.pet_proj, ct_feats, pet_feats):
            ct_aligned = ct_proj(ct_feat)
            pet_aligned = pet_proj(pet_feat)
            if pet_aligned.shape[-2:] != ct_aligned.shape[-2:]:
                pet_aligned = F.interpolate(pet_aligned, size=ct_aligned.shape[-2:], mode='bilinear', align_corners=False)
            fused.append(ct_aligned + pet_aligned)
        return fused


class HeterogeneousDualBackboneUNet(nn.Module):
    def __init__(self, ct_backbone='convnext_tiny', pet_backbone='mit_b0',
                 ct_pretrained_path=None, pet_pretrained_path=None,
                 in_channels=3, out_channels=1, decoder_type='light', fusion_channels=None,
                 fusion_type='mpa_bioclip_sum', bioclip_model_path=None,
                 bioclip_text_tower_path='/root/autodl-tmp/mkd-main/new-train/pretrained/biomedbert_text_tower',
                 bioclip_text='focal abnormal metabolic lung lesion on PET-CT scan'):
        super().__init__()
        ct_backbone = _normalize_backbone_name(ct_backbone)
        pet_backbone = _normalize_backbone_name(pet_backbone)
        self.backbone = f'{ct_backbone}+{pet_backbone}'
        self.ct_backbone = ct_backbone
        self.pet_backbone = pet_backbone
        self.decoder_type = decoder_type
        self.fusion_type = fusion_type
        self.use_tcpm = False
        self.use_adc_mac = False

        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        self.enc_pet = create_feature_backbone(pet_backbone, in_channels=in_channels)
        if ct_pretrained_path:
            load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='Teacher_CT_ConvNeXt_Encoder')
        if pet_pretrained_path:
            load_local_weights_safe(self.enc_pet, pet_pretrained_path, name='Teacher_PET_MiT_Encoder')

        ct_channels = self.enc_ct.feature_info.channels()
        pet_channels = self.enc_pet.feature_info.channels()
        if fusion_channels is None:
            fusion_channels = pet_channels
        fusion_channels = tuple(int(ch) for ch in fusion_channels)

        if fusion_type == 'project_sum':
            self.fusion = AdditiveProjectionFusion(ct_channels, pet_channels, fusion_channels)
        elif fusion_type in ('mpa_bioclip_sum', 'bcg_pa_sum'):
            if not bioclip_model_path:
                raise ValueError('fusion_type=mpa_bioclip_sum/bcg_pa_sum requires --bioclip_model_path.')
            print(f'[+] MPA-BioCLIP/BCG-PA: Encoding text prompt from {bioclip_model_path}')
            print(f'[+] MPA-BioCLIP/BCG-PA: Using local text tower {bioclip_text_tower_path}')
            text_feat = get_bioclip_text_feature(
                bioclip_model_path,
                bioclip_text,
                text_tower_path=bioclip_text_tower_path,
            )
            print(f'[+] MPA-BioCLIP/BCG-PA text feature shape: {tuple(text_feat.shape)}')
            self.fusion = MPABioCLIPSumFusion(ct_channels, pet_channels, fusion_channels, text_feat)
        else:
            raise ValueError(
                f'Unsupported fusion_type: {fusion_type}. '
                'Supported for hetero_convnext_mit: project_sum, mpa_bioclip_sum, bcg_pa_sum.'
            )

        if decoder_type != 'light':
            raise ValueError(f'Unsupported decoder_type: {decoder_type}. The cleaned hetero model only supports light.')
        self.decoder = LightConcatUNetDecoder(fusion_channels, out_channels=out_channels)

    @staticmethod
    def _to_3ch(x):
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def forward(self, ct, pet, target_size=None):
        ct = self._to_3ch(ct)
        pet = self._to_3ch(pet)
        ct_feats = self.enc_ct(ct)
        pet_feats = self.enc_pet(pet)
        if len(ct_feats) != 4 or len(pet_feats) != 4:
            raise ValueError(f'Heterogeneous encoders must output 4 stages, got CT={len(ct_feats)} PET={len(pet_feats)}')
        fused_feats = self.fusion(ct_feats, pet_feats)
        if len(fused_feats) != 4:
            raise ValueError(f'Heterogeneous fusion must output 4 stages, got {len(fused_feats)}')
        if target_size is None:
            target_size = ct.shape[-2:]
        return self.decoder(fused_feats, target_size)

    def set_epoch(self, epoch):
        return None

    def set_adc_mac_visuals(self, enabled):
        if self.fusion is not None and hasattr(self.fusion, 'set_visuals'):
            self.fusion.set_visuals(bool(enabled))

    def get_adc_mac_visuals(self):
        return {}

    def get_fusion_visuals(self):
        if self.fusion is not None and hasattr(self.fusion, 'get_fusion_visuals'):
            return self.fusion.get_fusion_visuals()
        return {}


class DualBackboneUNet(nn.Module):
    def __init__(self, backbone='pvt_v2_b1', pretrained_path=None,
                 in_channels=3, out_channels=1, use_tcpm=False, decoder_type='light',
                 fusion_type='sum', use_adc_mac=False):
        super().__init__()
        backbone = _normalize_backbone_name(backbone)
        self.backbone = backbone
        self.use_tcpm = False
        self.decoder_type = decoder_type
        self.use_adc_mac = False
        self.enc_ct = create_feature_backbone(backbone, in_channels=in_channels)
        self.enc_pet = create_feature_backbone(backbone, in_channels=in_channels)
        if pretrained_path:
            load_local_weights_safe(self.enc_ct, pretrained_path, name='Teacher_CT_Encoder')
            load_local_weights_safe(self.enc_pet, pretrained_path, name='Teacher_PET_Encoder')
        enc_channels = self.enc_ct.feature_info.channels()
        if decoder_type != 'light':
            raise ValueError(f'Unsupported decoder_type: {decoder_type}. The cleaned dual baseline only supports light.')
        self.decoder = LightConcatUNetDecoder(enc_channels, out_channels=out_channels)
        self.fusion_type = fusion_type
        if fusion_type not in ('sum', 'auto'):
            raise ValueError(f'Unsupported fusion_type for cleaned dual baseline: {fusion_type}. Supported: sum, auto.')

    @staticmethod
    def _to_3ch(x):
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def forward(self, ct, pet, target_size=None):
        ct = self._to_3ch(ct)
        pet = self._to_3ch(pet)
        ct_feats = self.enc_ct(ct)
        pet_feats = self.enc_pet(pet)
        fused_feats = [c + p for c, p in zip(ct_feats, pet_feats)]
        if target_size is None:
            target_size = ct.shape[-2:]
        return self.decoder(fused_feats, target_size)

    def set_epoch(self, epoch):
        return None

    def set_adc_mac_visuals(self, enabled):
        return None

    def get_adc_mac_visuals(self):
        return {}

    def get_fusion_visuals(self):
        return {}


def build_mdt_seg_teacher(config):
    if getattr(config, 'model_arch', 'dual') == 'hetero_convnext_mit':
        model = HeterogeneousDualBackboneUNet(
            ct_backbone=getattr(config, 'ct_backbone', 'convnext_tiny'),
            pet_backbone=getattr(config, 'pet_backbone', 'mit_b0'),
            ct_pretrained_path=getattr(config, 'ct_pretrained_path', None),
            pet_pretrained_path=getattr(config, 'pet_pretrained_path', None),
            in_channels=3,
            out_channels=1,
            decoder_type=getattr(config, 'decoder_type', 'light'),
            fusion_channels=getattr(config, 'fusion_channels', None),
            fusion_type=getattr(config, 'fusion_type', 'mpa_bioclip_sum'),
            bioclip_model_path=getattr(config, 'bioclip_model_path', None),
            bioclip_text_tower_path=getattr(config, 'bioclip_text_tower_path', '/root/autodl-tmp/mkd-main/new-train/pretrained/biomedbert_text_tower'),
            bioclip_text=getattr(config, 'bioclip_text', 'focal abnormal metabolic lung lesion on PET-CT scan'),
        )
        return dict(model=model)

    model = DualBackboneUNet(
        backbone=getattr(config, 'backbone', 'pvt_v2_b1'),
        pretrained_path=getattr(config, 'pretrained_path', None),
        in_channels=3,
        out_channels=1,
        decoder_type=getattr(config, 'decoder_type', 'light'),
        fusion_type=getattr(config, 'fusion_type', 'sum'),
    )
    return dict(model=model)
