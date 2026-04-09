# -*- coding: utf-8 -*-
"""
CIPA VMamba Baseline 教师：使用 CIPA 的 dual_vmamba_baseline（VMamba + 简单 add 融合，无 CRM/DCIM）。
依赖：CIPA-main 项目、selective_scan 已安装。
"""

import os
import sys

# 将 CIPA 项目加入 path（优先于 new-train 的 models，避免命名冲突）
_CIPA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'CIPA-main', 'CIPA-main')
if _CIPA_ROOT not in sys.path:
    sys.path.insert(0, _CIPA_ROOT)

import torch
import torch.nn as nn


class _EasyDict(dict):
    def __getattr__(self, k):
        return self[k]
    def __setattr__(self, k, v):
        self[k] = v


def _to_3ch(x):
    """将 (B,1,H,W) 转为 (B,3,H,W)，CIPA backbone 需要 3 通道"""
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x


def build_cipa_vmamba_baseline_teacher(config):
    """
    构建 CIPA VMamba baseline 教师：sigma_tiny_baseline + MambaDecoder。
    返回 networks = {'model': CIPA_EncoderDecoder_wrapper}
    """
    cipa_root = getattr(config, 'cipa_root', None) or _CIPA_ROOT
    pretrained_path = getattr(config, 'cipa_pretrained', None)
    if pretrained_path is None:
        pretrained_path = os.path.join(cipa_root, 'pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth')
    elif not os.path.isabs(pretrained_path):
        pretrained_path = os.path.join(cipa_root, pretrained_path)
    # 若预训练文件不存在则跳过加载（随机初始化）
    if not os.path.isfile(pretrained_path):
        pretrained_path = None

    cfg = _EasyDict()
    cfg.backbone = 'sigma_tiny_baseline'
    cfg.pretrained_model = pretrained_path
    cfg.decoder = 'MambaDecoder'
    cfg.decoder_embed_dim = 512
    cfg.image_height = getattr(config, 'image_size_2d', 512)
    cfg.image_width = getattr(config, 'image_size_2d', 512)
    cfg.bn_eps = 1e-3
    cfg.bn_momentum = 0.1
    cfg.num_classes = 1

    # 在 CIPA 目录下加载 CIPA 模型（临时切换 path 与 modules）
    _old_cwd = os.getcwd()
    _saved = {k: sys.modules.pop(k) for k in list(sys.modules.keys()) if k in ('models', 'utils', 'train_utils') or (k.startswith('models.') or k.startswith('utils.') or k.startswith('train_utils.'))}
    try:
        sys.path.insert(0, cipa_root)
        os.chdir(cipa_root)
        from models.builder import EncoderDecoder as CIPAEncoderDecoder
        cipa_model = CIPAEncoderDecoder(cfg=cfg, norm_layer=nn.BatchNorm2d)
    finally:
        if cipa_root in sys.path:
            sys.path.remove(cipa_root)
        os.chdir(_old_cwd)
        for k, v in _saved.items():
            sys.modules[k] = v

    class CIPAWrapper(nn.Module):
        """适配 new-train 的 (B,1,H,W) 输入"""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, ct, pet):
            ct_3ch = _to_3ch(ct)
            pet_3ch = _to_3ch(pet)
            return self.model(ct_3ch, pet_3ch)

    wrapper = CIPAWrapper(cipa_model)
    return dict(model=wrapper)
