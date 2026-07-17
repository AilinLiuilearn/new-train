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
    if not path or not os.path.exists(path):
        return
    if os.path.isdir(path):
        for cand in ('pytorch_model.bin', 'model.safetensors', 'mit_b1.pth', 'mit-b1.pth', 'convnextv2_nano.pth'):
            full = os.path.join(path, cand)
            if os.path.exists(full):
                path = full
                break
    if str(path).endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(path, device='cpu')
    else:
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = _sanitize_state_dict(_unwrap_state_dict(state_dict))
    model.load_state_dict(state_dict, strict=False)


def _normalize_backbone_name(backbone):
    aliases = {'convnextv2-nano': 'convnextv2_nano', 'convnext-v2-nano': 'convnextv2_nano', 'mit-b1': 'mit_b1'}
    return aliases.get(str(backbone), str(backbone))


class SimpleFeatureInfo:
    def __init__(self, channels): self._channels = list(channels)
    def channels(self): return self._channels


class SegformerFeatureBackbone(nn.Module):
    def __init__(self, variant='mit_b1', in_channels=3):
        super().__init__(); variant = _normalize_backbone_name(variant)
        settings = {'mit_b1': dict(depths=[2,2,2,2], hidden_sizes=[64,128,320,512], num_attention_heads=[1,2,5,8], drop_path_rate=0.1)}
        if SegformerConfig is None or SegformerModel is None: raise ImportError('transformers required')
        cfg = SegformerConfig(num_channels=in_channels, depths=settings[variant]['depths'], sr_ratios=[8,4,2,1], hidden_sizes=settings[variant]['hidden_sizes'], patch_sizes=[7,3,3,3], strides=[4,2,2,2], num_attention_heads=settings[variant]['num_attention_heads'], mlp_ratios=[4,4,4,4], hidden_act='gelu', hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, classifier_dropout_prob=0.1, initializer_range=0.02, drop_path_rate=settings[variant]['drop_path_rate'], reshape_last_stage=True, output_hidden_states=True)
        self.model = SegformerModel(cfg); self.feature_info = SimpleFeatureInfo(cfg.hidden_sizes)
    def forward(self, x):
        out = self.model(pixel_values=x, output_hidden_states=True, return_dict=True); hs = list(out.hidden_states or [])
        return hs[1:5] if len(hs) >= 5 else hs


class ConvNextFeatureBackbone(nn.Module):
    def __init__(self, variant='convnext_tiny', in_channels=3):
        super().__init__(); self.feature_info = SimpleFeatureInfo([96,192,384,768])
        if ConvNextConfig is None or ConvNextModel is None: raise ImportError('transformers required')
        cfg = ConvNextConfig(num_channels=in_channels, depths=[3,3,9,3], hidden_sizes=[96,192,384,768], patch_size=4, out_features=['stage1','stage2','stage3','stage4'])
        self.model = ConvNextModel(cfg)
    def forward(self, x):
        out = self.model(pixel_values=x, output_hidden_states=True, return_dict=True); hs=list(out.hidden_states); return hs[1:5] if len(hs)>=5 else hs[-4:]


def _get_backbone_out_indices(backbone): return (0,1,2,3)


class FallbackFeatureBackbone(nn.Module):
    def __init__(self, in_channels=3, channels=(32,64,160,256)):
        super().__init__(); self.feature_info=SimpleFeatureInfo(channels); self.net=nn.ModuleList([nn.Conv2d(in_channels,channels[0],3,2,1),nn.Conv2d(channels[0],channels[1],3,2,1),nn.Conv2d(channels[1],channels[2],3,2,1),nn.Conv2d(channels[2],channels[3],3,2,1)])
    def forward(self,x):
        feats=[]; y=x
        for layer in self.net:
            y=torch.relu(layer(y)); feats.append(y)
        return feats


def create_feature_backbone(backbone, in_channels=3):
    backbone = _normalize_backbone_name(backbone)
    if backbone == 'mit_b1':
        return SegformerFeatureBackbone(backbone, in_channels=in_channels) if SegformerConfig is not None and SegformerModel is not None else FallbackFeatureBackbone(in_channels=in_channels, channels=(64,128,320,512))
    if backbone == 'convnextv2_nano':
        return ConvNextFeatureBackbone(backbone, in_channels=in_channels) if ConvNextConfig is not None and ConvNextModel is not None else FallbackFeatureBackbone(in_channels=in_channels, channels=(96,192,384,768))
    return FallbackFeatureBackbone(in_channels=in_channels)


def _resolve_use_deep_supervision(config):
    return bool(getattr(config, 'use_deep_supervision', getattr(config, 'deep_supervision', False)))


def build_mdt_seg_teacher(config):
    model_arch = getattr(config, 'model_arch', 'dual_shared_add_baseline')
    common_kwargs = dict(ct_backbone=getattr(config, 'ct_backbone', 'convnextv2_nano'), pet_backbone=getattr(config, 'pet_backbone', 'mit_b1'), ct_pretrained_path=getattr(config, 'ct_pretrained_path', None), pet_pretrained_path=getattr(config, 'pet_pretrained_path', None), in_channels=3, out_channels=1, decoder_channels=getattr(config, 'decoder_channels', (512, 256, 128, 64)), use_deep_supervision=_resolve_use_deep_supervision(config))
    if model_arch == 'dual_decoder_ptgc':
        from models.dual_decoder_ptgc import DualDecoderPTGC
        model = DualDecoderPTGC(**common_kwargs, ptgc_ablation_mode=getattr(config, 'ptgc_ablation_mode', 'gvtc_pgmr'))
        print(f'[dual_decoder_ptgc] method=GVTC mode={getattr(config, "ptgc_ablation_mode", "gvtc_pgmr")} task_state=CT_S4_plus_prediction routing=region_local_sparsemax virtual_nodes=1_null_plus_8_operators operator=low_rank_16 insertion=S4 training=joint_from_scratch pgmr_weight={getattr(config, "pgmr_weight", 0.1)}')
        return dict(model=model)
    raise ValueError(f'Unsupported model_arch={model_arch}. Supported: dual_shared_add_baseline, dual_decoder_add_baseline, dual_decoder_pg_mtr_retrieval, dual_decoder_multiscale_task_increment_bank, dual_decoder_hatr_task_residual, dual_decoder_ptgc.')
