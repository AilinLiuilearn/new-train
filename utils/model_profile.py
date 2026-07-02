# -*- coding: utf-8 -*-

import copy

import torch
import torch.nn as nn


def count_params_m(networks_dict):
    total = sum(p.numel() for net in networks_dict.values() for p in net.parameters())
    return total / 1e6


class BaselineTeacherWrapper(nn.Module):
    def __init__(self, networks):
        super().__init__()
        for k, v in networks.items():
            setattr(self, k, v)

    def forward(self, ct, pet):
        return self.model(ct, pet, target_size=(ct.shape[-2], ct.shape[-1]))


def _get_flops_params_gm(model_wrapper, input_args):
    params_m = sum(p.numel() for p in model_wrapper.parameters()) / 1e6
    try:
        from thop import profile
        flops, _ = profile(model_wrapper, inputs=input_args)
        return flops / 1e9, params_m
    except ImportError:
        return None, params_m


def _count_module_params(module):
    if module is None:
        return 0, 0, 0
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def print_baseline_profile(networks, config, image_size=None, tag='MODEL PROFILE'):
    image_size = image_size or getattr(config, 'image_size_2d', 512)
    device = torch.device('cuda', int(config.gpus[0])) if torch.cuda.is_available() else torch.device('cpu')

    model = networks.get('model')
    total, trainable, frozen = _count_module_params(model)
    print_trainable_only = bool(getattr(config, 'print_trainable_only', True))
    headline = trainable if print_trainable_only else total
    print('\n================ MODEL PROFILE ================')
    print(f'HEADLINE PARAMS ({"trainable" if print_trainable_only else "total"}): {headline / 1e6:.2f}M')
    print(f'TOTAL PARAMS: {total / 1e6:.2f}M')
    print(f'TRAINABLE PARAMS: {trainable / 1e6:.2f}M')
    print(f'FROZEN PARAMS: {frozen / 1e6:.2f}M')
    for name in ('enc_ct', 'enc_pet', 'pet_guides', 'fusion', 'stage_fusion', 'decoder', 'boundary_head'):
        module = getattr(model, name, None) if model is not None else None
        mt, mtr, mf = _count_module_params(module)
        if module is not None:
            print(f'{name}: total={mt / 1e6:.2f}M trainable={mtr / 1e6:.2f}M frozen={mf / 1e6:.2f}M')
    print('==============================================')

    flops_g = None
    try:
        nets_copy = {k: copy.deepcopy(v) for k, v in networks.items()}
        wrapper = BaselineTeacherWrapper(nets_copy).to(device).eval()
        ct = torch.randn(1, 1, image_size, image_size, device=device)
        pet = torch.randn(1, 1, image_size, image_size, device=device)
        with torch.no_grad():
            flops_g, _ = _get_flops_params_gm(wrapper, (ct, pet))
        del wrapper, nets_copy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f'  [profile] FLOPs 统计失败: {e}')
    if flops_g is not None:
        print(f'[{tag}] FLOPs: {flops_g:.2f}G')
    else:
        print(f'[{tag}] FLOPs 需 thop: pip install thop')
    return total / 1e6, flops_g
