import os

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    if key == 'stem_weight':
        return 'stem_0.weight'
    if key == 'stem_bias':
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
        if nk.startswith('convnext.encoder.'):
            nk = nk[len('convnext.encoder.'):]
        nk = nk.replace('stages.', 'stages_')
        nk = nk.replace('stem.', 'stem_')
        nk = nk.replace('embeddings.patch_embeddings.', 'stem_')
        nk = nk.replace('encoder.stages.', 'stages_')
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
                     'mit_b1.pth', 'mit_b1.bin', 'mit_b1.pt',
                     'mit-b1.pth', 'mit-b1.bin', 'mit-b1.pt',
                     'pvt_v2_b1.pth', 'pvt_v2_b1.bin', 'pvt_v2_b1.pt',
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
        if mapped_key is not None and mapped_key in model_state and model_state[mapped_key].shape == v.shape:
            matched_key = mapped_key
        else:
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
    def __init__(self, variant='mit_b1', in_channels=3, pretrained=False, use_adc_mac=False):
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
            raise ImportError(
                'Segformer MiT backbone requires transformers. Install it with: pip install transformers'
            )
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
        self.use_adc_mac = use_adc_mac
        self.save_adc_mac_visuals = False
        self._last_adc_mac_features = {}
        self._last_adc_mac_weights = {}
        if self.use_adc_mac:
            from models.adc_mac import ADCMAC
            adc_cfg = [
                (config.hidden_sizes[0], 5, 5, 3),
                (config.hidden_sizes[1], 5, 5, 3),
                (config.hidden_sizes[2], 3, 3, 3),
                (config.hidden_sizes[3], 3, 3, 3),
            ]
            self.adc_mac = nn.ModuleList([
                ADCMAC(ch, ch, one=one, two=two, three=three)
                for ch, one, two, three in adc_cfg
            ])
        else:
            self.adc_mac = nn.ModuleList()

    def forward(self, x):
        if not self.use_adc_mac:
            self._last_adc_mac_features = {}
            self._last_adc_mac_weights = {}
            outputs = self.model(pixel_values=x, output_hidden_states=True, return_dict=True)
            return list(outputs.hidden_states)

        encoder = self.model.encoder
        hidden_states = x
        features = []
        if self.save_adc_mac_visuals:
            self._last_adc_mac_features = {}
            self._last_adc_mac_weights = {}
        for stage_idx, patch_embed in enumerate(encoder.patch_embeddings):
            hidden_states, height, width = patch_embed(hidden_states)
            for block in encoder.block[stage_idx]:
                hidden_states = block(hidden_states, height, width)[0]
            hidden_states = encoder.layer_norm[stage_idx](hidden_states)

            batch_size, _, channels = hidden_states.shape
            stage_feature = hidden_states.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()
            if stage_idx < len(self.adc_mac):
                self.adc_mac[stage_idx].cache_direction_weights = self.save_adc_mac_visuals
                stage_feature = self.adc_mac[stage_idx](stage_feature)
                if self.save_adc_mac_visuals and self.adc_mac[stage_idx]._last_direction_weights is not None:
                    self._last_adc_mac_weights[f'stage{stage_idx + 1}'] = self.adc_mac[stage_idx]._last_direction_weights
            if self.save_adc_mac_visuals:
                self._last_adc_mac_features[f'stage{stage_idx + 1}'] = stage_feature.detach().cpu()
            features.append(stage_feature)
            hidden_states = stage_feature
        return features


def _get_backbone_out_indices(backbone):
    backbone = _normalize_backbone_name(backbone)
    if backbone in ('pvt_v2_b1', 'mit_b0', 'mit_b1'):
        return (0, 1, 2, 3)
    if backbone in (
        'convnext_tiny', 'convnext_nano',
        'convnextv2_atto', 'convnextv2_femto', 'convnextv2_pico', 'convnextv2_nano',
        'convnextv2_tiny', 'convnextv2_small', 'convnextv2_base', 'convnextv2_large', 'convnextv2_huge',
    ):
        return (0, 1, 2, 3)
    if backbone in ('efficientnet_b3', 'efficientnet_b4'):
        return (1, 2, 3, 4)
    raise ValueError(
        f'Unsupported backbone: {backbone}. Supported: pvt_v2_b1, mit-b0/mit_b0, mit-b1/mit_b1, convnext_tiny, convnext_nano, '
        'convnextv2_atto/femto/pico/nano/tiny/small/base/large/huge, efficientnet_b3, efficientnet_b4.'
    )


def create_feature_backbone(backbone, in_channels=3, use_adc_mac=False):
    backbone = _normalize_backbone_name(backbone)
    if backbone in ('mit_b0', 'mit_b1'):
        return SegformerFeatureBackbone(backbone, in_channels=in_channels, use_adc_mac=use_adc_mac)
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


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = x.mean(dim=(2, 3))
        mx = x.amax(dim=(2, 3))
        w = torch.sigmoid(self.fc(avg) + self.fc(mx))
        return x * w.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        desc = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(desc))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.sa(self.ca(x))


class LightASPP(nn.Module):
    """Lightweight ASPP for multi-scale context at the bottleneck."""

    def __init__(self, in_channels, out_channels, dilations=(1, 6, 12)):
        super().__init__()
        branch_ch = out_channels // len(dilations)
        self.branches = nn.ModuleList([
            ConvBNAct(in_channels, branch_ch, kernel_size=3 if d > 1 else 1, dilation=d)
            for d in dilations
        ])
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNAct(in_channels, branch_ch, kernel_size=1),
        )
        total_ch = branch_ch * (len(dilations) + 1)
        self.fuse = ConvBNAct(total_ch, out_channels, kernel_size=1)

    def forward(self, x):
        outs = [br(x) for br in self.branches]
        gap = self.gap(x)
        gap = F.interpolate(gap, size=x.shape[-2:], mode='bilinear', align_corners=False)
        outs.append(gap)
        return self.fuse(torch.cat(outs, dim=1))


class SESkip(nn.Module):
    """Squeeze-and-Excitation gate on skip connections to recalibrate channels."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        return x * self.se(x).view(b, c, 1, 1)


class AttentionUNetDecoder(nn.Module):
    """UNet decoder with CBAM attention, lightweight ASPP bottleneck, and SE-gated skips."""

    def __init__(self, encoder_channels, out_channels=1, decoder_channels=(512, 256, 128, 64)):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        d4, d3, d2, d1 = decoder_channels

        self.bottleneck = LightASPP(c4, d4, dilations=(1, 6, 12))

        self.skip3 = nn.Sequential(ConvBNAct(c3, d3, kernel_size=1), SESkip(d3))
        self.skip2 = nn.Sequential(ConvBNAct(c2, d2, kernel_size=1), SESkip(d2))
        self.skip1 = nn.Sequential(ConvBNAct(c1, d1, kernel_size=1), SESkip(d1))

        self.fuse3 = nn.Sequential(ConvBNAct(d4 + d3, d3), ConvBNAct(d3, d3), CBAM(d3))
        self.fuse2 = nn.Sequential(ConvBNAct(d3 + d2, d2), ConvBNAct(d2, d2), CBAM(d2))
        self.fuse1 = nn.Sequential(ConvBNAct(d2 + d1, d1), ConvBNAct(d1, d1), CBAM(d1))

        self.head4 = nn.Conv2d(d4, out_channels, 1)
        self.head3 = nn.Conv2d(d3, out_channels, 1)
        self.head2 = nn.Conv2d(d2, out_channels, 1)
        self.head1 = nn.Conv2d(d1, out_channels, 1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    @staticmethod
    def _up_size(x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, features, target_size):
        x1, x2, x3, x4 = features

        d4 = self.bottleneck(x4)

        s3 = self.skip3(x3)
        d3 = self.fuse3(torch.cat([self._up(d4, s3), s3], dim=1))

        s2 = self.skip2(x2)
        d2 = self.fuse2(torch.cat([self._up(d3, s2), s2], dim=1))

        s1 = self.skip1(x1)
        d1 = self.fuse1(torch.cat([self._up(d2, s1), s1], dim=1))

        p1 = self._up_size(self.head1(d1), target_size)
        p2 = self._up_size(self.head2(d2), target_size)
        p3 = self._up_size(self.head3(d3), target_size)
        p4 = self._up_size(self.head4(d4), target_size)
        return {'preds': [p1, p2, p3, p4], 'pred': p1}


class TextGuidedCrossAttentionBlock(nn.Module):
    def __init__(self, channels, text_dim=512, num_text_tokens=4, num_heads=4, pool_size=8):
        super().__init__()
        self.channels = channels
        self.num_text_tokens = num_text_tokens
        self.pool_size = pool_size
        self.text_project = nn.Sequential(
            nn.Linear(text_dim, num_text_tokens * channels, bias=False),
            nn.GELU(),
            nn.LayerNorm(num_text_tokens * channels),
        )
        self.vis_norm = nn.LayerNorm(channels)
        self.txt_norm = nn.LayerNorm(channels)
        self.txt_update_norm = nn.LayerNorm(channels)
        self.vis_to_txt = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.txt_to_vis = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(channels, channels, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(0.01))

    def _text_tokens(self, text_code):
        b = text_code.shape[0]
        return self.text_project(text_code).view(b, self.num_text_tokens, self.channels)

    def forward(self, x, text_code):
        b, c, h, w = x.shape
        visual = x.flatten(2).transpose(1, 2)
        txt = self._text_tokens(text_code)

        pooled = F.adaptive_avg_pool2d(x, output_size=(min(self.pool_size, h), min(self.pool_size, w)))
        pooled = pooled.flatten(2).transpose(1, 2)
        txt_delta, _ = self.vis_to_txt(
            query=self.txt_norm(txt),
            key=self.vis_norm(pooled),
            value=pooled,
        )
        txt = txt + self.scale * self.txt_update_norm(txt_delta)

        visual_delta, _ = self.txt_to_vis(
            query=self.vis_norm(visual),
            key=self.txt_norm(txt),
            value=txt,
        )
        guided = visual + self.scale * visual_delta
        guided = guided.transpose(1, 2).view(b, c, h, w)

        txt_desc = txt.mean(dim=1)
        channel = self.channel_gate(txt_desc).view(b, c, 1, 1)
        spatial = self.spatial_gate(torch.cat([x, guided], dim=1))
        return x + self.scale * guided * channel * spatial


class LightConcatUNetDecoder(nn.Module):
    def __init__(self, encoder_channels, out_channels=1, decoder_channels=(512, 256, 128, 64)):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        d4, d3, d2, d1 = decoder_channels

        self.proj4 = ConvBNAct(c4, d4, kernel_size=1)
        self.proj3 = ConvBNAct(c3, d3, kernel_size=1)
        self.proj2 = ConvBNAct(c2, d2, kernel_size=1)
        self.proj1 = ConvBNAct(c1, d1, kernel_size=1)

        self.fuse3 = nn.Sequential(
            ConvBNAct(d4 + d3, d3),
            ConvBNAct(d3, d3),
        )
        self.fuse2 = nn.Sequential(
            ConvBNAct(d3 + d2, d2),
            ConvBNAct(d2, d2),
        )
        self.fuse1 = nn.Sequential(
            ConvBNAct(d2 + d1, d1),
            ConvBNAct(d1, d1),
        )

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


class NnUNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class NnUNetUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = NnUNetConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class NnUNetDecoder(nn.Module):
    def __init__(self, encoder_channels, out_channels=1, decoder_channels=(512, 256, 128, 64, 32)):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        d4, d3, d2, d1, d0 = decoder_channels

        self.bottleneck = NnUNetConvBlock(c4, d4)
        self.up3 = NnUNetUpBlock(d4, c3, d3)
        self.up2 = NnUNetUpBlock(d3, c2, d2)
        self.up1 = NnUNetUpBlock(d2, c1, d1)
        self.up0 = nn.ConvTranspose2d(d1, d0, kernel_size=2, stride=2)
        self.up_full = nn.ConvTranspose2d(d0, d0, kernel_size=2, stride=2)
        self.final_conv = NnUNetConvBlock(d0, d0)

        self.head3 = nn.Conv2d(d3, out_channels, kernel_size=1)
        self.head2 = nn.Conv2d(d2, out_channels, kernel_size=1)
        self.head1 = nn.Conv2d(d1, out_channels, kernel_size=1)
        self.head0 = nn.Conv2d(d0, out_channels, kernel_size=1)

    @staticmethod
    def _upsample_size(x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, features, target_size):
        x1, x2, x3, x4 = features

        d4 = self.bottleneck(x4)
        d3 = self.up3(d4, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x1)
        d0 = self.up0(d1)
        d0 = self.up_full(d0)
        if d0.shape[-2:] != target_size:
            d0 = self._upsample_size(d0, target_size)
        d0 = self.final_conv(d0)

        p0 = self.head0(d0)
        p1 = self.head1(d1)
        p2 = self.head2(d2)
        p3 = self.head3(d3)
        return {'preds': [p0, p1, p2, p3], 'pred': p0}


class TextGuidedLightConcatUNetDecoder(nn.Module):
    def __init__(self, encoder_channels, out_channels=1, decoder_channels=(512, 256, 128, 64), text_dim=512):
        super().__init__()
        from models.cudm_text_fusion import FixedPromptEmbedding, TUMOR_PROMPT

        c1, c2, c3, c4 = encoder_channels
        d4, d3, d2, d1 = decoder_channels
        self.text_encoder = FixedPromptEmbedding(prompt=TUMOR_PROMPT)

        self.proj4 = ConvBNAct(c4, d4, kernel_size=1)
        self.proj3 = ConvBNAct(c3, d3, kernel_size=1)
        self.proj2 = ConvBNAct(c2, d2, kernel_size=1)
        self.proj1 = ConvBNAct(c1, d1, kernel_size=1)

        self.tg4 = TextGuidedCrossAttentionBlock(d4, text_dim=text_dim, num_text_tokens=4, num_heads=8, pool_size=8)
        self.tg3 = TextGuidedCrossAttentionBlock(d3, text_dim=text_dim, num_text_tokens=4, num_heads=4, pool_size=8)
        self.tg2 = TextGuidedCrossAttentionBlock(d2, text_dim=text_dim, num_text_tokens=4, num_heads=4, pool_size=8)
        self.tg1 = TextGuidedCrossAttentionBlock(d1, text_dim=text_dim, num_text_tokens=4, num_heads=2, pool_size=8)

        self.fuse3 = nn.Sequential(
            ConvBNAct(d4 + d3, d3),
            ConvBNAct(d3, d3),
        )
        self.fuse2 = nn.Sequential(
            ConvBNAct(d3 + d2, d2),
            ConvBNAct(d2, d2),
        )
        self.fuse1 = nn.Sequential(
            ConvBNAct(d2 + d1, d1),
            ConvBNAct(d1, d1),
        )

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
        b = x1.shape[0]
        text_code = self.text_encoder(b, x1.device)

        d4 = self.tg4(self.proj4(x4), text_code)
        s3 = self.proj3(x3)
        d3 = self.fuse3(torch.cat([self._upsample_to(d4, s3), s3], dim=1))
        d3 = self.tg3(d3, text_code)

        s2 = self.proj2(x2)
        d2 = self.fuse2(torch.cat([self._upsample_to(d3, s2), s2], dim=1))
        d2 = self.tg2(d2, text_code)

        s1 = self.proj1(x1)
        d1 = self.fuse1(torch.cat([self._upsample_to(d2, s1), s1], dim=1))
        d1 = self.tg1(d1, text_code)

        p1 = self._upsample_size(self.head1(d1), target_size)
        p2 = self._upsample_size(self.head2(d2), target_size)
        p3 = self._upsample_size(self.head3(d3), target_size)
        p4 = self._upsample_size(self.head4(d4), target_size)
        return {'preds': [p1, p2, p3, p4], 'pred': p1}


class ConvNextFeatureBackbone(nn.Module):
    def __init__(self, variant='convnext_tiny', in_channels=3):
        super().__init__()
        variant = _normalize_backbone_name(variant)
        convnext_settings = {
            'convnext_tiny': dict(hidden_sizes=[96, 192, 384, 768]),
        }
        if variant not in convnext_settings:
            raise ValueError(f'Unsupported ConvNeXt variant: {variant}')

        if ConvNextConfig is None or ConvNextModel is None:
            raise ImportError(
                'HuggingFace ConvNeXt backbone requires transformers. Install it with: pip install transformers'
            )
        settings = convnext_settings[variant]
        config = ConvNextConfig(
            num_channels=in_channels,
            depths=[3, 3, 9, 3],
            hidden_sizes=settings['hidden_sizes'],
            patch_size=4,
            out_features=['stage1', 'stage2', 'stage3', 'stage4'],
        )
        self.model = ConvNextModel(config)
        self.feature_info = SimpleFeatureInfo(settings['hidden_sizes'])

    def forward(self, x):
        outputs = self.model(pixel_values=x, output_hidden_states=True, return_dict=True)
        hidden_states = list(outputs.hidden_states)
        if len(hidden_states) >= 5:
            return hidden_states[1:5]
        return hidden_states[-4:]


class AdditiveProjectionFusion(nn.Module):
    def __init__(self, ct_channels, pet_channels, out_channels):
        super().__init__()
        if not (len(ct_channels) == len(pet_channels) == len(out_channels)):
            raise ValueError('ct_channels, pet_channels and out_channels must have the same length.')
        self.ct_proj = nn.ModuleList([
            ConvBNAct(cin, cout, kernel_size=1) for cin, cout in zip(ct_channels, out_channels)
        ])
        self.pet_proj = nn.ModuleList([
            ConvBNAct(cin, cout, kernel_size=1) for cin, cout in zip(pet_channels, out_channels)
        ])

    def forward(self, ct_feats, pet_feats):
        fused = []
        for ct_proj, pet_proj, ct_feat, pet_feat in zip(self.ct_proj, self.pet_proj, ct_feats, pet_feats):
            ct_aligned = ct_proj(ct_feat)
            pet_aligned = pet_proj(pet_feat)
            if pet_aligned.shape[-2:] != ct_aligned.shape[-2:]:
                pet_aligned = F.interpolate(
                    pet_aligned,
                    size=ct_aligned.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            fused.append(ct_aligned + pet_aligned)
        return fused


class HeterogeneousDualBackboneUNet(nn.Module):
    def __init__(self, ct_backbone='convnext_tiny', pet_backbone='mit_b0',
                 ct_pretrained_path=None, pet_pretrained_path=None,
                 in_channels=3, out_channels=1, decoder_type='light', fusion_channels=None,
                 use_adc_mac=False):
        super().__init__()
        ct_backbone = _normalize_backbone_name(ct_backbone)
        pet_backbone = _normalize_backbone_name(pet_backbone)
        self.backbone = f'{ct_backbone}+{pet_backbone}'
        self.ct_backbone = ct_backbone
        self.pet_backbone = pet_backbone
        self.decoder_type = decoder_type
        self.fusion_type = 'project_sum'
        self.use_tcpm = False
        self.use_adc_mac = use_adc_mac and pet_backbone == 'mit_b1'
        if use_adc_mac and pet_backbone != 'mit_b1':
            print(f'[-] ADC-MAC is currently intended for mit_b1 PET encoder, ignored for pet_backbone={pet_backbone}')

        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels, use_adc_mac=False)
        self.enc_pet = create_feature_backbone(pet_backbone, in_channels=in_channels, use_adc_mac=self.use_adc_mac)
        if ct_pretrained_path:
            load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='Teacher_CT_ConvNeXt_Encoder')
        if pet_pretrained_path:
            load_local_weights_safe(self.enc_pet, pet_pretrained_path, name='Teacher_PET_MiT_Encoder')

        ct_channels = self.enc_ct.feature_info.channels()
        pet_channels = self.enc_pet.feature_info.channels()
        if fusion_channels is None:
            fusion_channels = pet_channels
        fusion_channels = tuple(int(ch) for ch in fusion_channels)
        self.fusion = AdditiveProjectionFusion(ct_channels, pet_channels, fusion_channels)

        if decoder_type == 'light':
            self.decoder = LightConcatUNetDecoder(fusion_channels, out_channels=out_channels)
        elif decoder_type == 'nnunet':
            self.decoder = NnUNetDecoder(fusion_channels, out_channels=out_channels)
        elif decoder_type == 'text_guided_light':
            self.decoder = TextGuidedLightConcatUNetDecoder(fusion_channels, out_channels=out_channels)
        elif decoder_type == 'attention':
            self.decoder = AttentionUNetDecoder(fusion_channels, out_channels=out_channels)
        else:
            raise ValueError(f'Unsupported decoder_type: {decoder_type}')

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
        fused_feats = self.fusion(ct_feats, pet_feats)
        if target_size is None:
            target_size = ct.shape[-2:]
        return self.decoder(fused_feats, target_size)

    def set_epoch(self, epoch):
        return None

    def set_adc_mac_visuals(self, enabled):
        if hasattr(self.enc_pet, 'save_adc_mac_visuals'):
            self.enc_pet.save_adc_mac_visuals = bool(enabled)

    def get_adc_mac_visuals(self):
        visuals = {}
        if hasattr(self.enc_pet, '_last_adc_mac_features') and self.enc_pet._last_adc_mac_features:
            visuals['pet'] = {
                'features': self.enc_pet._last_adc_mac_features,
                'weights': getattr(self.enc_pet, '_last_adc_mac_weights', {}),
            }
        return visuals

    def get_fusion_visuals(self):
        return {}


class DualBackboneUNet(nn.Module):
    """Dual CT/PET teacher with configurable timm backbone and UNet decoder."""

    def __init__(self, backbone='pvt_v2_b1', pretrained_path=None,
                 in_channels=3, out_channels=1, use_tcpm=False, decoder_type='attention',
                 fusion_type='auto', wavelet_window_sizes=(8, 8, 4, 4), wavelet_heads=(1, 2, 4, 8),
                 wavelet_sr_ratios=(4, 4, 2, 1), wavelet_attn_ratio=0.25, wavelet_conv_ratio=0.25,
                 fnet_sparse_hidden_ratio=0.25, fnet_sparse_max_hidden=64, fnet_sparse_iters=2,
                 fnet_sparse_init_gamma=0.1, use_adc_mac=False):
        super().__init__()
        backbone = _normalize_backbone_name(backbone)
        self.backbone = backbone
        self.use_tcpm = use_tcpm
        self.decoder_type = decoder_type
        self.use_adc_mac = use_adc_mac and backbone == 'mit_b1'
        if use_adc_mac and backbone != 'mit_b1':
            print(f'[-] ADC-MAC is currently implemented for mit_b1 only, ignored for backbone={backbone}')
        self.enc_ct = create_feature_backbone(backbone, in_channels=in_channels, use_adc_mac=self.use_adc_mac)
        self.enc_pet = create_feature_backbone(backbone, in_channels=in_channels, use_adc_mac=self.use_adc_mac)
        if pretrained_path:
            load_local_weights_safe(self.enc_ct, pretrained_path, name='Teacher_CT_Encoder')
            load_local_weights_safe(self.enc_pet, pretrained_path, name='Teacher_PET_Encoder')

        enc_channels = self.enc_ct.feature_info.channels()
        if decoder_type == 'light':
            self.decoder = LightConcatUNetDecoder(enc_channels, out_channels=out_channels)
        elif decoder_type == 'nnunet':
            self.decoder = NnUNetDecoder(enc_channels, out_channels=out_channels)
        elif decoder_type == 'text_guided_light':
            self.decoder = TextGuidedLightConcatUNetDecoder(enc_channels, out_channels=out_channels)
        elif decoder_type == 'attention':
            self.decoder = AttentionUNetDecoder(enc_channels, out_channels=out_channels)
        else:
            raise ValueError(f'Unsupported decoder_type: {decoder_type}')

        fusion_type = 'cudm_text' if fusion_type == 'auto' and use_tcpm else fusion_type
        fusion_type = 'sum' if fusion_type == 'auto' else fusion_type
        self.fusion_type = fusion_type

        if fusion_type == 'cudm_text':
            from models.cudm_text_fusion import MultiStageCUDMTextFusion
            self.fusion = MultiStageCUDMTextFusion(enc_channels)
        elif fusion_type == 'pet_window_wavelet':
            from models.pet_window_wavelet_fusion import MultiStagePETWindowWaveletFusion
            self.fusion = MultiStagePETWindowWaveletFusion(
                enc_channels,
                window_sizes=tuple(wavelet_window_sizes),
                heads_per_stage=tuple(wavelet_heads),
                attn_ratio=wavelet_attn_ratio,
                wavelet_ratio=wavelet_conv_ratio,
                sr_ratios=tuple(wavelet_sr_ratios),
            )
        elif fusion_type == 'fnet_sparse':
            from models.fnet_sparse_fusion import MultiStageFNetSparseFusion
            self.fusion = MultiStageFNetSparseFusion(
                enc_channels,
                hidden_ratio=fnet_sparse_hidden_ratio,
                max_hidden=fnet_sparse_max_hidden,
                n_iter=fnet_sparse_iters,
                init_gamma=fnet_sparse_init_gamma,
            )
        elif fusion_type == 'heccm_all':
            from models.heccm_fusion import MultiStageHECCMFusion
            self.fusion = MultiStageHECCMFusion(
                enc_channels,
                num_classes=2,
                window_sizes=(16, 8, 8, 4),
                embed_dim=16,
                num_heads=2,
                init_gamma=0.01,
            )
        elif fusion_type in ('edl_gcm_plus', 'edl_gcm_plus_ct'):
            from models.edl_gcm_plus_fusion import EDLGCMPlusBottleneckFusion
            self.fusion = EDLGCMPlusBottleneckFusion(
                enc_channels,
                num_groups=8,
                init_gamma=0.01,
                shallow_mode='ct' if fusion_type == 'edl_gcm_plus_ct' else 'sum',
            )
        elif fusion_type == 'edl_spmc_s3':
            from models.edl_spmc_stage3_fusion import EDLSPMCStage3Fusion
            self.fusion = EDLSPMCStage3Fusion(enc_channels, init_gamma=0.01)
        elif fusion_type in ('spgc_s12', 'spgc_s12_edl_spmc_s3'):
            from models.spgc_fusion import SPGCFusion
            self.fusion = SPGCFusion(
                enc_channels,
                init_gamma=0.01,
                use_stage3_spmc=fusion_type == 'spgc_s12_edl_spmc_s3',
            )
        elif fusion_type == 'concat':
            self.fusion = nn.ModuleList([
                nn.Conv2d(ch * 2, ch, kernel_size=1, bias=False)
                for ch in enc_channels
            ])
        elif fusion_type == 'sum':
            self.fusion = None
        else:
            raise ValueError(f'Unsupported fusion_type: {fusion_type}')

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

        if self.fusion_type == 'concat':
            fused_feats = [
                proj(torch.cat([c, p], dim=1))
                for proj, c, p in zip(self.fusion, ct_feats, pet_feats)
            ]
            fusion_aux = None
        elif self.fusion is not None:
            fusion_out = self.fusion(ct_feats, pet_feats)
            if isinstance(fusion_out, tuple):
                fused_feats, fusion_aux = fusion_out
            else:
                fused_feats, fusion_aux = fusion_out, None
        else:
            fused_feats = [c + p for c, p in zip(ct_feats, pet_feats)]
            fusion_aux = None

        if target_size is None:
            target_size = ct.shape[-2:]
        outputs = self.decoder(fused_feats, target_size)
        if fusion_aux is not None and isinstance(outputs, dict):
            outputs['fusion_aux'] = fusion_aux
        return outputs

    def set_epoch(self, epoch):
        return None

    def set_adc_mac_visuals(self, enabled):
        for encoder in (self.enc_ct, self.enc_pet):
            if hasattr(encoder, 'save_adc_mac_visuals'):
                encoder.save_adc_mac_visuals = bool(enabled)

    def get_adc_mac_visuals(self):
        visuals = {}
        if hasattr(self.enc_ct, '_last_adc_mac_features') and self.enc_ct._last_adc_mac_features:
            visuals['ct'] = {
                'features': self.enc_ct._last_adc_mac_features,
                'weights': getattr(self.enc_ct, '_last_adc_mac_weights', {}),
            }
        if hasattr(self.enc_pet, '_last_adc_mac_features') and self.enc_pet._last_adc_mac_features:
            visuals['pet'] = {
                'features': self.enc_pet._last_adc_mac_features,
                'weights': getattr(self.enc_pet, '_last_adc_mac_weights', {}),
            }
        return visuals

    def get_fusion_visuals(self):
        if self.fusion is not None and hasattr(self.fusion, 'get_fusion_visuals'):
            return self.fusion.get_fusion_visuals()
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
            use_adc_mac=getattr(config, 'use_adc_mac', False),
        )
        return dict(model=model)

    model = DualBackboneUNet(
        backbone=getattr(config, 'backbone', 'pvt_v2_b1'),
        pretrained_path=getattr(config, 'pretrained_path', None),
        in_channels=3,
        out_channels=1,
        use_tcpm=getattr(config, 'use_tcpm', False),
        decoder_type=getattr(config, 'decoder_type', 'attention'),
        fusion_type=getattr(config, 'fusion_type', 'auto'),
        wavelet_window_sizes=getattr(config, 'wavelet_window_sizes', (8, 8, 4, 4)),
        wavelet_heads=getattr(config, 'wavelet_heads', (1, 2, 4, 8)),
        wavelet_sr_ratios=getattr(config, 'wavelet_sr_ratios', (4, 4, 2, 1)),
        wavelet_attn_ratio=getattr(config, 'wavelet_attn_ratio', 0.25),
        wavelet_conv_ratio=getattr(config, 'wavelet_conv_ratio', 0.25),
        fnet_sparse_hidden_ratio=getattr(config, 'fnet_sparse_hidden_ratio', 0.25),
        fnet_sparse_max_hidden=getattr(config, 'fnet_sparse_max_hidden', 64),
        fnet_sparse_iters=getattr(config, 'fnet_sparse_iters', 2),
        fnet_sparse_init_gamma=getattr(config, 'fnet_sparse_init_gamma', 0.1),
        use_adc_mac=getattr(config, 'use_adc_mac', False),
    )
    return dict(model=model)
