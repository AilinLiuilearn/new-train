# -*- coding: utf-8 -*-
"""
模型 FLOPs (G) 与 Params (M) 统计工具。
依赖: pip install thop（可选，无则仅输出 Params）
注意：thop.profile 会向模型注入 total_ops/total_params，故用 deepcopy 避免污染原始网络。
"""

import copy
import torch
import torch.nn as nn


def count_params_m(networks_dict):
    """统计参数量，单位 M"""
    total = sum(p.numel() for net in networks_dict.values() for p in net.parameters())
    return total / 1e6


class TeacherSegWrapper(nn.Module):
    """教师完整前向包装，供 thop 统计 FLOPs（与 tasks.mdt_seg._teacher_forward 一致：频域 FDMF + 可选浅层 FSF）"""

    def __init__(self, networks, use_cmx=True):
        super().__init__()
        for k, v in networks.items():
            setattr(self, k, v)
        self.use_cmx = use_cmx

    def forward(self, ct, pet):
        feats_mri = self.extractor_mri(ct, return_list=True)
        feats_pet = self.extractor_pet(pet, return_list=True)
        h_mri = feats_mri[-1]
        h_pet = feats_pet[-1]
        if self.projector_mri is not None:
            h_mri = self.projector_mri(h_mri)
            h_pet = self.projector_pet(h_pet)
        fusion3 = self.lfgf_fusion(h_pet, h_mri, return_separated=False)
        if self.use_cmx and hasattr(self, 'frm_ffm_0'):
            fused_0, _ = self.frm_ffm_0(feats_mri[0], feats_pet[0], return_attn=True)
            fused_1, _ = self.frm_ffm_1(feats_mri[1], feats_pet[1], return_attn=True)
            fused_2, _ = self.frm_ffm_2(feats_mri[2], feats_pet[2], return_attn=True)
        else:
            fused_0 = feats_mri[0] + feats_pet[0]
            fused_1 = feats_mri[1] + feats_pet[1]
            fused_2 = feats_mri[2] + feats_pet[2]
        fpn_input_list = [fused_0, fused_1, fused_2, fusion3]
        seg_logit = self.segmentor(fpn_input_list, target_size=(ct.shape[-2], ct.shape[-1]))
        return seg_logit


class StudentSegWrapper(nn.Module):
    """学生完整前向包装，供 thop 统计 FLOPs"""

    def __init__(self, networks, use_specific=True):
        super().__init__()
        for k, v in networks.items():
            setattr(self, k, v)
        self.use_specific = use_specific

    def forward(self, ct):
        feats = self.extractor(ct, return_list=True)
        h = feats[-1]
        if self.projector is not None:
            h = self.projector(h)
        z_general = self.encoder_general(h)
        z_mri = self.encoder_mri(h)
        if self.use_specific:
            fusion_s = torch.cat([z_general, z_mri], dim=1)
        else:
            fusion_s = z_general
        fpn_input_list = [feats[0], feats[1], feats[2], fusion_s]
        logit_s = self.segmentor(fpn_input_list, target_size=(ct.shape[-2], ct.shape[-1]))
        return logit_s


def get_flops_params_gm(model_wrapper, input_args, input_kwargs=None):
    """
    返回 (flops_g, params_m)，若 thop 不可用则 flops_g 为 None。
    input_args: 如 (ct, pet) 或 (ct,)
    input_kwargs: 可选
    """
    params_m = sum(p.numel() for p in model_wrapper.parameters()) / 1e6
    try:
        from thop import profile
        input_kwargs = input_kwargs or {}
        flops, _ = profile(model_wrapper, inputs=input_args, **input_kwargs)
        return flops / 1e9, params_m
    except ImportError:
        return None, params_m


def print_model_profile(stage_name, networks, config, is_teacher=True, image_size=None):
    """打印阶段模型的 FLOPs (G) 和 Params (M)"""
    image_size = image_size or getattr(config, 'image_size_2d', 512)
    device = torch.device('cuda', config.gpus[0] if isinstance(config.gpus[0], int) else int(config.gpus[0]))
    use_cmx = getattr(config, 'use_cmx', True)
    use_specific = getattr(config, 'use_specific', True)

    params_m = count_params_m(networks)
    flops_g = None

    try:
        # 使用 deepcopy 避免 thop 注入 total_ops/total_params 污染原始网络
        nets_copy = {k: copy.deepcopy(v) for k, v in networks.items()}
        if is_teacher:
            wrapper = TeacherSegWrapper(nets_copy, use_cmx).to(device).eval()
            ct = torch.randn(1, 1, image_size, image_size, device=device)
            pet = torch.randn(1, 1, image_size, image_size, device=device)
            with torch.no_grad():
                flops_g, _ = get_flops_params_gm(wrapper, (ct, pet))
        else:
            wrapper = StudentSegWrapper(nets_copy, use_specific).to(device).eval()
            ct = torch.randn(1, 1, image_size, image_size, device=device)
            with torch.no_grad():
                flops_g, _ = get_flops_params_gm(wrapper, (ct,))
        del wrapper, nets_copy
    except Exception:
        pass  # flops_g 保持 None，仅输出 Params

    if flops_g is not None:
        print(f"[{stage_name}] Params: {params_m:.2f}M, FLOPs: {flops_g:.2f}G")
    else:
        print(f"[{stage_name}] Params: {params_m:.2f}M (FLOPs 需 thop: pip install thop)")


class BaselineTeacherWrapper(nn.Module):
    """
    当前 baseline 教师前向包装（enc_ct/enc_pet/fuse_*/segmentor），供 thop 统计 FLOPs。
    """
    def __init__(self, networks):
        super().__init__()
        for k, v in networks.items():
            setattr(self, k, v)

    def forward(self, ct, pet):
        if hasattr(self, 'model'):
            out = self.model(ct, pet, target_size=(ct.shape[-2], ct.shape[-1]))
            return out
        feats_ct  = self.enc_ct(ct,  return_list=True)
        feats_pet = self.enc_pet(pet, return_list=True)
        if hasattr(self, 'feature_dropout'):
            feats_ct  = [self.feature_dropout(f) for f in feats_ct]
            feats_pet = [self.feature_dropout(f) for f in feats_pet]
        fused = [getattr(self, f'fuse_{i}')(feats_ct[i], feats_pet[i]) for i in range(4)]
        return self.segmentor(fused, target_size=(ct.shape[-2], ct.shape[-1]))


def print_baseline_profile(networks, config, image_size=None, tag='baseline'):
    """
    打印 baseline 教师网络的参数量 (M) 和 FLOPs (G)。
    在 run_mdt_seg.py 中 build_mdt_seg_teacher 之后调用。
    """
    image_size = image_size or getattr(config, 'image_size_2d', 512)
    device = torch.device('cuda', int(config.gpus[0]))

    params_m = count_params_m(networks)
    flops_g = None

    try:
        nets_copy = {k: copy.deepcopy(v) for k, v in networks.items()}
        wrapper = BaselineTeacherWrapper(nets_copy).to(device).eval()
        ct  = torch.randn(1, 1, image_size, image_size, device=device)
        pet = torch.randn(1, 1, image_size, image_size, device=device)
        with torch.no_grad():
            flops_g, _ = get_flops_params_gm(wrapper, (ct, pet))
        del wrapper, nets_copy
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'  [profile] FLOPs 统计失败: {e}')

    if flops_g is not None:
        print(f'[{tag}] Params: {params_m:.2f}M  FLOPs: {flops_g:.2f}G')
    else:
        print(f'[{tag}] Params: {params_m:.2f}M  (FLOPs 需 thop: pip install thop)')
    return params_m, flops_g
