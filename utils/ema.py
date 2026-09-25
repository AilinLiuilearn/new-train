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

    ``update`` must be called once per successful optimizer step. ``model`` is
    the EMA copy meant for evaluation; the source model is never modified.

    Handles non-tensor state entries (e.g. fusion ``_extra_state`` dicts):
    those are compared for contract consistency and refreshed via the owning
    submodule's ``set_extra_state``, never treated as tensors.
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
        if set(ema_state.keys()) != set(src_state.keys()):
            raise RuntimeError(
                f'EMA/state key mismatch: only_ema={sorted(set(ema_state) - set(src_state))} '
                f'only_src={sorted(set(src_state) - set(ema_state))}')
        for key in ema_state.keys():
            ema_val = ema_state[key]
            src_val = src_state[key]
            if isinstance(ema_val, dict) or isinstance(src_val, dict):
                if ema_val != src_val:
                    # Refresh the EMA copy's contracted extra state via the
                    # owning submodule so buffers/attrs stay consistent.
                    self._sync_extra_state(key, src_val)
                continue
            if not torch.is_tensor(ema_val) or not torch.is_tensor(src_val):
                raise RuntimeError(f'EMA state entry {key!r} is not a tensor/dict')
            if ema_val.dtype.is_floating_point and key != 'fusion.text_embeddings':
                ema_val.mul_(decay).add_(src_val.detach(), alpha=1.0 - decay)
            else:
                # Integer/bool buffers and the frozen text cache copy exactly.
                ema_val.copy_(src_val)
        return decay

    @torch.no_grad()
    def _sync_extra_state(self, key, src_extra):
        import copy as _copy
        # key looks like 'fusion._extra_state'; find owning submodule.
        parts = key.split('.')
        owner = self.model
        for part in parts[:-1]:
            owner = getattr(owner, part, None)
            if owner is None:
                raise RuntimeError(f'EMA cannot locate owner of extra state {key!r}')
        setter = getattr(owner, 'set_extra_state', None)
        if setter is None:
            raise RuntimeError(f'EMA extra state {key!r} differs but owner has no set_extra_state')
        setter(_copy.deepcopy(src_extra))

    @torch.no_grad()
    def reset(self, model):
        """Hard-sync the EMA copy to ``model`` and restart the decay ramp.

        Used when the EMA is (re)activated after a warmup delay, so the EMA
        does not blend in stale weights from before it started tracking.
        """
        self.model.load_state_dict(model.state_dict())
        self.updates = 0

    def state_dict(self):
        return self.model.state_dict()
