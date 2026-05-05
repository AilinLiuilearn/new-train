# -*- coding: utf-8 -*-
"""优化器与学习率调度"""

import math
import torch


def get_optimizer(params, name='adamw', lr=1e-4, weight_decay=1e-4):
    if isinstance(params, list) and len(params) > 0 and isinstance(params[0], dict):
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def get_cosine_scheduler(optimizer, epochs, warmup_steps=0, cycles=1, min_lr=1e-6, steps_per_epoch=None, flat_ratio=0.0):
    """
    Cosine LR with optional flat tail.
    flat_ratio: fraction of total steps at the end where lr stays at min_lr plateau.
                0.0 = pure cosine (default), 0.3 = last 30% is flat at a higher floor.
    """
    if steps_per_epoch is not None:
        total_steps = epochs * steps_per_epoch
    else:
        total_steps = epochs * 1000

    cosine_steps = int(total_steps * (1.0 - flat_ratio))

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        if step >= cosine_steps:
            return max(min_lr / optimizer.defaults['lr'], 0.5 * (1 + math.cos(math.pi * cycles)))
        progress = (step - warmup_steps) / max(1, cosine_steps - warmup_steps)
        progress = min(1.0, progress)
        mul = 0.5 * (1 + math.cos(math.pi * cycles * progress))
        return max(min_lr / optimizer.defaults['lr'], mul)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
