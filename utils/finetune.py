# -*- coding: utf-8 -*-
"""Two-stage CMGF fine-tuning helpers.

finetune_mode:
  none          - train all params with a single learning_rate
  cmgf_only     - freeze encoder/CPPI/calibration/decoder; train fusion only
  joint_diff_lr - unfreeze all; use backbone/fusion/decoder learning rates
"""

from __future__ import annotations

import torch
import torch.nn as nn


BACKBONE_ATTRS = (
    'enc_ct',
    'enc_pet',
    'ct_align',
    'pet_calibration',
    'prototype_memory',
)
FUSION_ATTRS = ('fusion',)
DECODER_ATTRS = ('decoder',)


def _module_params(module):
    return [p for p in module.parameters()]


def _set_requires_grad(module, enabled):
    for param in module.parameters():
        param.requires_grad = bool(enabled)


def _set_norm_eval(module):
    for child in module.modules():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            child.eval()


def apply_finetune_mode(model, finetune_mode='none'):
    mode = str(finetune_mode or 'none')
    if mode not in {'none', 'cmgf_only', 'joint_diff_lr'}:
        raise ValueError(
            f'Unsupported finetune_mode={finetune_mode!r}; '
            "expected one of: 'none', 'cmgf_only', 'joint_diff_lr'"
        )

    if mode == 'cmgf_only':
        for name in BACKBONE_ATTRS + DECODER_ATTRS:
            _set_requires_grad(getattr(model, name), False)
        for name in FUSION_ATTRS:
            _set_requires_grad(getattr(model, name), True)
    else:
        for name in BACKBONE_ATTRS + FUSION_ATTRS + DECODER_ATTRS:
            _set_requires_grad(getattr(model, name), True)

    return mode


def sync_frozen_modules_eval(model, finetune_mode='none'):
    """Keep frozen modules in eval so BN running stats stay fixed."""
    if str(finetune_mode or 'none') != 'cmgf_only':
        return
    for name in BACKBONE_ATTRS + DECODER_ATTRS:
        module = getattr(model, name)
        module.eval()
        _set_norm_eval(module)


def build_finetune_param_groups(model, cfg):
    mode = str(getattr(cfg, 'finetune_mode', 'none') or 'none')
    base_lr = float(cfg.learning_rate)
    backbone_lr = float(getattr(cfg, 'backbone_lr', None) or base_lr)
    fusion_lr = float(getattr(cfg, 'fusion_lr', None) or base_lr)
    decoder_lr = float(getattr(cfg, 'decoder_lr', None) or base_lr)
    weight_decay = float(cfg.weight_decay)

    def collect(attrs):
        params = []
        for name in attrs:
            params.extend(_module_params(getattr(model, name)))
        return params

    if mode == 'cmgf_only':
        fusion_params = [p for p in collect(FUSION_ATTRS) if p.requires_grad]
        if not fusion_params:
            raise RuntimeError('cmgf_only mode found no trainable fusion parameters')
        groups = [{'params': fusion_params, 'lr': fusion_lr, 'weight_decay': weight_decay, 'name': 'fusion'}]
        default_lr = fusion_lr
    elif mode == 'joint_diff_lr':
        groups = [
            {'params': collect(BACKBONE_ATTRS), 'lr': backbone_lr, 'weight_decay': weight_decay, 'name': 'backbone'},
            {'params': collect(FUSION_ATTRS), 'lr': fusion_lr, 'weight_decay': weight_decay, 'name': 'fusion'},
            {'params': collect(DECODER_ATTRS), 'lr': decoder_lr, 'weight_decay': weight_decay, 'name': 'decoder'},
        ]
        groups = [g for g in groups if len(g['params']) > 0]
        default_lr = backbone_lr
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        groups = [{'params': params, 'lr': base_lr, 'weight_decay': weight_decay, 'name': 'all'}]
        default_lr = base_lr

    return groups, default_lr


def summarize_finetune_setup(model, cfg, param_groups):
    mode = str(getattr(cfg, 'finetune_mode', 'none') or 'none')
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    lines = [
        f'[FINETUNE] mode={mode}',
        f'[FINETUNE] trainable_params={trainable} frozen_params={frozen}',
    ]
    for group in param_groups:
        n = sum(p.numel() for p in group['params'])
        lines.append(
            f"[FINETUNE] group={group.get('name', 'unnamed')} "
            f"lr={group['lr']:.8g} params={n}"
        )
    return '\n'.join(lines)


def create_finetune_optimizer(model, cfg):
    apply_finetune_mode(model, getattr(cfg, 'finetune_mode', 'none'))
    sync_frozen_modules_eval(model, getattr(cfg, 'finetune_mode', 'none'))
    param_groups, default_lr = build_finetune_param_groups(model, cfg)
    optimizer = torch.optim.AdamW(param_groups, lr=default_lr, weight_decay=float(cfg.weight_decay))
    print(summarize_finetune_setup(model, cfg, param_groups), flush=True)
    return optimizer
