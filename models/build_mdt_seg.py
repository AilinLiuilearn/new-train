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


class SimpleFeatureInfo:
    def __init__(self, channels):
        self._channels = list(channels)

    def channels(self):
        return self._channels


def create_feature_backbone(backbone, in_channels=3):
    backbone = str(backbone).replace('-', '_')
    if backbone == 'convnextv2_nano':
        return FallbackFeatureBackbone(in_channels=in_channels, channels=(96, 192, 384, 512))
    if backbone == 'mit_b1':
        return FallbackFeatureBackbone(in_channels=in_channels, channels=(64, 128, 320, 512))
    if timm is None:
        return FallbackFeatureBackbone(in_channels=in_channels)
    try:
        return timm.create_model(backbone, pretrained=False, features_only=True, out_indices=(0, 1, 2, 3), in_chans=in_channels)
    except Exception:
        return FallbackFeatureBackbone(in_channels=in_channels)


def load_local_weights_safe(model, path, name='Encoder'):
    if not path or not os.path.exists(path):
        return
    if str(path).endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(path, device='cpu')
    else:
        state_dict = torch.load(path, map_location='cpu')
    state_dict = _sanitize_state_dict(_unwrap_state_dict(state_dict))
    model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}, strict=False)


def _resolve_use_deep_supervision(config):
    return bool(getattr(config, 'use_deep_supervision', False) or getattr(config, 'deep_supervision', False))


class FallbackFeatureBackbone(nn.Module):
    def __init__(self, in_channels=3, channels=(32, 64, 160, 256)):
        super().__init__()
        self.feature_info = SimpleFeatureInfo(channels)
        self.stem = ConvBNAct(in_channels, channels[0], stride=2)
        self.stage1 = ConvBNAct(channels[0], channels[0], stride=2)
        self.stage2 = ConvBNAct(channels[0], channels[1], stride=2)
        self.stage3 = ConvBNAct(channels[1], channels[2], stride=2)
        self.stage4 = ConvBNAct(channels[2], channels[3], stride=2)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


def build_mdt_seg_teacher(config):
    model_arch = getattr(config, 'model_arch', 'dual_shared_add_baseline')
    common_kwargs = dict(ct_backbone=getattr(config, 'ct_backbone', 'convnextv2_nano'), pet_backbone=getattr(config, 'pet_backbone', 'mit_b1'), ct_pretrained_path=getattr(config, 'ct_pretrained_path', None), pet_pretrained_path=getattr(config, 'pet_pretrained_path', None), in_channels=3, out_channels=1, decoder_channels=getattr(config, 'decoder_channels', (512, 256, 128, 64)), use_deep_supervision=_resolve_use_deep_supervision(config))
    if model_arch == 'dual_shared_add_baseline':
        from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
        return dict(model=DualSharedAddPETCTBaseline(**common_kwargs))
    if model_arch == 'dual_decoder_add_baseline':
        from models.dual_decoder_add_baseline import DualDecoderAddPETCTBaseline
        return dict(model=DualDecoderAddPETCTBaseline(**common_kwargs))
    if model_arch == 'dual_decoder_pg_mtr_retrieval':
        from models.dual_decoder_pg_mtr_retrieval import DualDecoderPGMTRRetrieval
        return dict(model=DualDecoderPGMTRRetrieval(**common_kwargs))
    if model_arch == 'dual_decoder_multiscale_task_increment_bank':
        from models.dual_decoder_multiscale_task_increment_bank import DualDecoderMultiScaleTaskIncrementBank
        return dict(model=DualDecoderMultiScaleTaskIncrementBank(**common_kwargs))
    if model_arch == 'dual_decoder_hatr_task_residual':
        from models.dual_decoder_hatr_task_residual import DualDecoderHATRTaskResidual
        return dict(model=DualDecoderHATRTaskResidual(**common_kwargs))
    if model_arch == 'dual_decoder_ptgc':
        from models.dual_decoder_ptgc import DualDecoderPTGC
        model = DualDecoderPTGC(**common_kwargs, ptgc_ablation_mode=getattr(config, 'ptgc_ablation_mode', 'gvtc_pgmr'))
        print(f'[dual_decoder_ptgc] method=GVTC mode={getattr(config, "ptgc_ablation_mode", "gvtc_pgmr")} task_state=CT_S4_plus_prediction routing=region_local_sparsemax virtual_nodes=1_null_plus_8_operators operator=low_rank_16 insertion=S4 training=joint_from_scratch pgmr_weight={getattr(config, "pgmr_weight", 0.1)}')
        return dict(model=model)
    raise ValueError(f'Unsupported model_arch={model_arch}.')
