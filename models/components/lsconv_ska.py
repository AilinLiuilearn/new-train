"""SKA backend adapter; the CUDA implementation is the unmodified official file.

The reference uses the source kernel's modulo channel sharing (not contiguous
repeat_interleave), zero padding, no softmax, and no channel mixing in SKA.
"""
import importlib

import torch
from torch import nn
from torch.nn import functional as F

BACKENDS = ('auto', 'torch', 'triton')


def ska_torch(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """O(HW) differentiable reference, streaming nine shifts without unfold."""
    if x.ndim != 4 or w.ndim != 5:
        raise ValueError('SKA expects x[B,C,H,W], w[B,C/8,9,H,W]')
    b, c, h, width = x.shape
    wc = w.shape[1]
    if (wc < 1 or c % wc or w.shape != (b, wc, 9, h, width)
            or x.device != w.device or not x.is_floating_point()
            or not w.is_floating_point()):
        raise ValueError('SKA weight shape/device/dtype does not match input')
    dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    padded = F.pad(x.to(dtype), (1, 1, 1, 1))
    weights = w.to(dtype)
    out = torch.zeros_like(x, dtype=dtype)
    for kh in range(3):
        for kw in range(3):
            # Official Triton selects kernel index ci % wc.
            kernel = weights[:, :, kh * 3 + kw].repeat(1, c // wc, 1, 1)
            out = out + padded[:, :, kh:kh + h, kw:kw + width] * kernel
    return out.to(x.dtype)


class SKA(nn.Module):
    def __init__(self, backend='auto'):
        super().__init__()
        if backend not in BACKENDS:
            raise ValueError(f'Unsupported SKA backend {backend!r}')
        self.backend = backend

    def forward(self, x, w):
        if self.backend == 'torch' or (self.backend == 'auto' and not x.is_cuda):
            return ska_torch(x, w)
        if not x.is_cuda:
            raise RuntimeError('Official Triton SKA requires CUDA; use --lsconv_backend torch on CPU')
        if x.dtype not in (torch.float16, torch.float32):
            raise TypeError('Official Triton SKA supports float16/float32 here; use torch backend for other dtypes')
        try:
            source = importlib.import_module('.lsconv_ska_triton', __package__)
        except ImportError as exc:
            raise RuntimeError('Official SKA requires a compatible Triton installation; '
                               'run the CUDA preflight before training') from exc
        # No silent CUDA fallback: source compilation/gradient errors must surface.
        return source.SkaFn.apply(x, w)
