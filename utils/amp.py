"""AMP compatibility with the existing CUDA environment; no torch upgrade needed."""
from contextlib import nullcontext
import torch


def make_grad_scaler(device, enabled=True):
    enabled = bool(enabled and torch.device(device).type == 'cuda')
    modern = getattr(getattr(torch, 'amp', None), 'GradScaler', None)
    if modern is not None:
        return modern('cuda', enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device, enabled=True):
    if not enabled or torch.device(device).type != 'cuda':
        return nullcontext()
    modern = getattr(getattr(torch, 'amp', None), 'autocast', None)
    if modern is not None:
        return modern(device_type='cuda', enabled=True)
    return torch.cuda.amp.autocast(enabled=True)
