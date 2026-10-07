"""RASFE-inspired Gaussian response descriptors for PET/CT feature matching.

Repository destination: models/gaussian_response_descriptor.py
Tensor contract: (B, dim, H, W) -> (B, dim, H, W), without resizing.
Requires Python >= 3.10 and PyTorch >= 2.1.

Retained RASFE mechanism: three normalized 3x3 Gaussian response branches,
with (sigma, dilation) = (1.0, 1), (1.5, 1), (2.0, 2). Fixed kernels are
persistent buffers; learnable kernels are Parameters initialized identically.
The learnable version does NOT constrain kernels to remain Gaussian.

Explicit task adaptation: channel-wise filtering of projected feature maps,
concat -> GroupNorm -> GELU -> pointwise projection -> descriptor residual.
This is neither a full RAC-Net reproduction nor physical HU processing.
The original contrast descriptor remains in the existing fusion file.
No attention, fusion, missing-modality generation or loss is added here.
"""
from __future__ import annotations

import argparse
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

__all__ = ["GaussianDepthwiseResponse", "GaussianResponseDescriptor",
           "sync_fixed_gaussian_buffers"]


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_feature(x: Tensor, dim: int) -> None:
    if not isinstance(x, Tensor) or x.ndim != 4:
        raise ValueError("Expected a BCHW feature tensor")
    if x.shape[1] != dim or min(x.shape[0], x.shape[2], x.shape[3]) < 1:
        raise ValueError(f"Expected nonempty BCHW features with {dim} channels")
    if not x.is_floating_point():
        raise TypeError("Feature tensor must be floating point")


class GaussianDepthwiseResponse(nn.Module):
    """Zero-padded, depthwise 3x3 Gaussian filtering at one dilation.

    sigma is measured on the undilated 3x3 kernel grid. Dilation enlarges the
    sampling footprint, not the kernel size. Boundary responses use zero
    padding, so a spatial constant is preserved only away from boundaries.
    """

    def __init__(self, dim: int, sigma: float, dilation: int,
                 trainable: bool = False) -> None:
        super().__init__()
        self.dim = _positive_integer(dim, "dim")
        self.dilation = _positive_integer(dilation, "dilation")
        self.sigma = float(sigma)
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        if not isinstance(trainable, bool):
            raise TypeError("trainable must be bool")

        # Construct deterministically in float64, normalize, then store float32.
        # No random Conv2d kernel is allocated/discarded during initialization.
        coordinates = torch.arange(-1, 2, dtype=torch.float64)
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        kernel = torch.exp(-(xx.square() + yy.square()) / (2 * self.sigma**2))
        kernel = (kernel / kernel.sum()).to(dtype=torch.float32)
        weight = kernel.reshape(1, 1, 3, 3).repeat(self.dim, 1, 1, 1).contiguous()
        if trainable:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight, persistent=True)

    def forward(self, x: Tensor) -> Tensor:
        _validate_feature(x, self.dim)
        if x.device != self.weight.device:
            raise ValueError("Move descriptor and input to the same device")
        # Do not use no_grad: a fixed filter must still backpropagate into x.
        # Standard autocast handles conv dtype; there is no forced FP32 copy.
        return F.conv2d(x, self.weight, bias=None, stride=1,
                        padding=self.dilation, dilation=self.dilation,
                        groups=self.dim)

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, sigma={self.sigma}, dilation={self.dilation}, "
                f"trainable={isinstance(self.weight, nn.Parameter)}")


class GaussianResponseDescriptor(nn.Module):
    """Three-branch Gaussian response descriptor, preserving raw features.

    E(X) = X + Conv1x1(GELU(GN(concat(G1(X), G2(X), G3(X)))))

    rasfe_fixed: only GN and Conv1x1 learn; kernels remain fixed.
    rasfe_learnable: identical architecture/initialization; kernels also learn.
    CT and PET should receive independent instances, at every scale.

    Unlike the fusion update projection, mix is NOT zero initialized. The
    surrounding fusion's existing zero-initialized updates preserve its
    initial CT+PET output. This descriptor alone is not initialized as X.

    State dictionaries contain kernel values but cannot identify buffer vs
    Parameter semantics. Validate descriptor_type in experiment/checkpoint
    metadata before loading. A strict tensor load alone cannot detect that.
    """

    BRANCH_SPECS = ((1.0, 1), (1.5, 1), (2.0, 2))
    TYPES = ("rasfe_fixed", "rasfe_learnable")

    def __init__(self, dim: int, descriptor_type: str = "rasfe_fixed") -> None:
        super().__init__()
        self.dim = _positive_integer(dim, "dim")
        if descriptor_type not in self.TYPES:
            raise ValueError(f"descriptor_type must be one of {self.TYPES}, got {descriptor_type!r}")
        self.descriptor_type = descriptor_type
        trainable = descriptor_type == "rasfe_learnable"
        self.branches = nn.ModuleList([
            GaussianDepthwiseResponse(self.dim, sigma, dilation, trainable)
            for sigma, dilation in self.BRANCH_SPECS
        ])
        response_channels = 3 * self.dim
        self.norm = nn.GroupNorm(math.gcd(response_channels, 8), response_channels)
        self.mix = nn.Conv2d(response_channels, self.dim, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        _validate_feature(x, self.dim)
        response = torch.cat([branch(x) for branch in self.branches], dim=1)
        return x + self.mix(F.gelu(self.norm(response)))

    def extra_repr(self) -> str:
        return f"dim={self.dim}, descriptor_type={self.descriptor_type!r}"


@torch.no_grad()
def sync_fixed_gaussian_buffers(target: nn.Module, source: nn.Module) -> None:
    """Copy only fixed Gaussian kernels after an EMA update.

    The repository's EMA averages all floating buffers. Even equal constants
    can drift by roundoff under multiply/add. This helper leaves all learned
    parameters and all other buffers under the existing EMA policy.
    Both models must have the same descriptor configuration/module paths.
    """
    for name, branch in source.named_modules():
        if not isinstance(branch, GaussianDepthwiseResponse):
            continue
        if isinstance(branch.weight, nn.Parameter):
            continue
        peer = target.get_submodule(name)
        if (not isinstance(peer, GaussianDepthwiseResponse)
                or isinstance(peer.weight, nn.Parameter)
                or peer.weight.shape != branch.weight.shape
                or peer.sigma != branch.sigma
                or peer.dilation != branch.dilation):
            raise ValueError(f"Fixed Gaussian EMA configuration mismatch at {name!r}")
        peer.weight.copy_(branch.weight.detach())


def _smoke() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--descriptor-type", choices=GaussianResponseDescriptor.TYPES,
                        default="rasfe_fixed")
    args = parser.parse_args()
    torch.manual_seed(2023)
    torch.set_num_threads(1)
    m = GaussianResponseDescriptor(args.dim, args.descriptor_type).to(args.device)
    x = torch.randn(2, args.dim, 17, 19, device=args.device, requires_grad=True)
    y = m(x)
    y.square().mean().backward()
    if y.shape != x.shape or not bool(torch.isfinite(y).all()):
        raise RuntimeError("Forward smoke check failed")
    if x.grad is None or not bool(torch.isfinite(x.grad).all()):
        raise RuntimeError("Backward smoke check failed")
    print(f"OK {args.descriptor_type}: shape={tuple(y.shape)}, "
          f"parameters={sum(p.numel() for p in m.parameters())}, "
          f"buffers={sum(b.numel() for b in m.buffers())}")


if __name__ == "__main__":
    _smoke()
