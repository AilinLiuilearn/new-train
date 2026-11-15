# -*- coding: utf-8 -*-
"""Exponential moving average of model weights for evaluation.

Both baselines share one EMA policy. Evaluation checkpoints record whether
raw or EMA weights were used.
"""
import copy

import torch


class ModelEMA:
    """Maintain an EMA of ``model``'s parameters/buffers.

    ``update`` must be called once per successful optimizer step. ``model``
    is the EMA copy meant for evaluation; the source model is never modified.
    Only tensors are tracked; anything else in the state dict raises loudly.
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
            decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        else:
            decay = self.decay
        ema_state = self.model.state_dict()
        src_state = model.state_dict()
        if set(ema_state.keys()) != set(src_state.keys()):
            raise RuntimeError(
                f'EMA/state key mismatch: only_ema={sorted(set(ema_state) - set(src_state))} '
                f'only_src={sorted(set(src_state) - set(ema_state))}')
        for key in ema_state.keys():
            ema_val = ema_state[key]
            src_val = src_state[key]
            if not torch.is_tensor(ema_val) or not torch.is_tensor(src_val):
                raise RuntimeError(f'EMA state entry {key!r} is not a tensor')
            if ema_val.dtype.is_floating_point:
                ema_val.mul_(decay).add_(src_val.detach(), alpha=1.0 - decay)
            else:
                ema_val.copy_(src_val)
        return decay

    @torch.no_grad()
    def reset(self, model):
        """Hard-sync the EMA copy to ``model`` and restart the decay ramp."""
        self.model.load_state_dict(model.state_dict())
        self.updates = 0

    def state_dict(self):
        return self.model.state_dict()