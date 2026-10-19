# -*- coding: utf-8 -*-
"""Shared optimizer-step and random-seed helpers.
"""
import csv
import json
import os
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


def optimizer_step_succeeded(task):
    """AMP-aware optimizer step. Returns False when the step was skipped."""
    if not task.scaler.is_enabled():
        task.optimizer.step()
        return True
    before = task.scaler.get_scale()
    task.scaler.step(task.optimizer)
    task.scaler.update()
    after = task.scaler.get_scale()
    return bool(after >= before)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
