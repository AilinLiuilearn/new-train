import os
import torch
import torch.nn as nn
import timm


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
    def __init__(self, channels): self._channels = list(channels)
    def channels(self): return self._channels


def load_local_weights_safe(model, path, name='Encoder'):
    if not path or not os.path.exists(path): return
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state: state = state['state_dict']
    model.load_state_dict(state, strict=False)


def create_feature_backbone(backbone, in_channels=3):
    if timm is None: raise ImportError('timm required')
    model = timm.create_model(backbone, pretrained=False, features_only=True, out_indices=(0,1,2,3), in_chans=in_channels)
    return model


def build_mdt_seg_teacher(config):
    from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
    return {'model': DualSharedAddPETCTBaseline(config)}
