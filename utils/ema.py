# -*- coding: utf-8 -*-
"""Exponential moving average of model weights for evaluation.

EMA smooths the parameter trajectory produced by the (warmup + cosine) LR
schedule, which is the main source of the epoch-to-epoch validation jitter.
Evaluate/save with the EMA weights to get a flatter plateau.
"""
import copy

import torch


class ModelEMA:
    """Maintain an exponential moving average of ``model``'s parameters/buffers.

    ``update`` must be called once per optimizer step. ``model`` is the EMA copy
    meant for evaluation; the source model is never modified.
    """

    def __init__(self, model, decay=0.999, warmup=True, device=None):
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.updates = 0
        self.model = copy.deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.model.to(device)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        if self.warmup:
            # Ramp the decay early so the EMA converges quickly at the start.
            decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        else:
            decay = self.decay
        ema_state = self.model.state_dict()
        src_state = model.state_dict()
        for key, ema_val in ema_state.items():
            src_val = src_state[key].detach()
            if ema_val.dtype.is_floating_point:
                ema_val.mul_(decay).add_(src_val, alpha=1.0 - decay)
            else:
                ema_val.copy_(src_val)
        return decay

    def state_dict(self):
        return self.model.state_dict()
