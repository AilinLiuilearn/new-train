# -*- coding: utf-8 -*-
"""优化器与学习率调度"""

import math
import torch


def get_optimizer(params, name='adamw', lr=1e-4, weight_decay=1e-4):
    if isinstance(params, list) and len(params) > 0 and isinstance(params[0], dict):
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def get_cosine_scheduler(optimizer, epochs, warmup_steps=0, cycles=1, min_lr=1e-6, steps_per_epoch=None):
    """
    按 step 调度的 cosine 学习率。
    若提供 steps_per_epoch，则 total_steps = epochs * steps_per_epoch；否则用 epochs * 1000 近似。
    """
    if steps_per_epoch is not None:
        total_steps = epochs * steps_per_epoch
    else:
        total_steps = epochs * 1000

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        mul = 0.5 * (1 + math.cos(math.pi * cycles * progress))
        return max(min_lr / optimizer.defaults['lr'], mul)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
