# -*- coding: utf-8 -*-

import copy

import torch
import torch.nn as nn


def count_params_m(networks_dict, trainable_only=False, exclude_text_encoder=False):
    total = 0
    for net in networks_dict.values():
        for name, p in net.named_parameters():
            if trainable_only and not p.requires_grad:
                continue
            if exclude_text_encoder and name.startswith('text_encoder.'):
                continue
            total += p.numel()
    return total / 1e6


class BaselineTeacherWrapper(nn.Module):
    def __init__(self, networks):
        super().__init__()
        for k, v in networks.items():
            setattr(self, k, v)

    def forward(self, ct, pet):
        model = self.model
        text_code = None
        if hasattr(model, 'text_encoder'):
            text_code = torch.zeros(ct.shape[0], 512, device=ct.device, dtype=ct.dtype)
        return model(ct, pet, target_size=(ct.shape[-2], ct.shape[-1]), text_code=text_code)


def _get_flops_params_gm(model_wrapper, input_args):
    params_m = sum(p.numel() for p in model_wrapper.parameters()) / 1e6
    try:
        from thop import profile
        flops, _ = profile(model_wrapper, inputs=input_args)
        return flops / 1e9, params_m
    except ImportError:
        return None, params_m


def print_baseline_profile(networks, config, image_size=None, tag='teacher_baseline'):
    image_size = image_size or getattr(config, 'image_size_2d', 512)
    device = torch.device('cuda', int(config.gpus[0]))

    params_m_total = count_params_m(networks, trainable_only=False, exclude_text_encoder=False)
    params_m = count_params_m(networks, trainable_only=True, exclude_text_encoder=False)
    params_m_no_text = count_params_m(networks, trainable_only=False, exclude_text_encoder=True)
    flops_g = None

    try:
        nets_copy = {k: copy.deepcopy(v) for k, v in networks.items()}
        wrapper = BaselineTeacherWrapper(nets_copy).to(device).eval()
        ct = torch.randn(1, 1, image_size, image_size, device=device)
        pet = torch.randn(1, 1, image_size, image_size, device=device)
        with torch.no_grad():
            flops_g, _ = _get_flops_params_gm(wrapper, (ct, pet))
        del wrapper, nets_copy
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'  [profile] FLOPs 统计失败: {e}')

    if flops_g is not None:
        print(f'[{tag}] Trainable Params: {params_m:.2f}M  Params(no text encoder): {params_m_no_text:.2f}M  Total Params: {params_m_total:.2f}M  FLOPs(no text encoder): {flops_g:.2f}G')
    else:
        print(f'[{tag}] Trainable Params: {params_m:.2f}M  Params(no text encoder): {params_m_no_text:.2f}M  Total Params: {params_m_total:.2f}M  (FLOPs 需 thop: pip install thop)')
    return params_m, flops_g
