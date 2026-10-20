# -*- coding: utf-8 -*-
"""Shared optimizer-step and random-seed helpers.
"""
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def module_grad_norm(module):
    total = None
    for p in module.parameters():
        if p.grad is None:
            continue
        value = p.grad.detach().float().pow(2).sum()
        total = value if total is None else total + value
    return float(total.sqrt().item()) if total is not None else 0.0
