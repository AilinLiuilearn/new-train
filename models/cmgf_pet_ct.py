"""CMGF fusion adapted for paired CT/PET feature maps.

This is a clean, standalone PyTorch adaptation of the Complete
Modality-guided Fusion (CMGF) module from VideoFusion:

Paper: https://arxiv.org/abs/2503.23359
Code:  https://github.com/Linfeng-Tang/VideoFusion

The official CMGF computation is preserved:

    common_query = ct_feature + pet_feature
    ct_branch     = shared_cross_transformer(common_query, ct_feature)
    pet_branch    = shared_cross_transformer(common_query, pet_feature)
    fused_feature = ct_branch + pet_branch

The same CrossTransformerBlock instance is deliberately reused for both
modalities, matching the official implementation. Attention is computed over
channel subspaces (head_dim x head_dim), not over all spatial token pairs
(HW x HW). Therefore, no global spatial attention matrix is allocated.

Intended interface for both training paths:

    Full:    fused = module(ct_feature, calibrated_real_pet_feature)
    Missing: fused = module(ct_feature, calibrated_proxy_pet_feature)

Both inputs and the output have shape [B, C, H, W]. No Full/Missing flag,
prototype, text, temporal frame, deformable sampler, or auxiliary loss is
required by this module.
"""

from __future__ import annotations

import argparse
import time
from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _to_tokens(x: Tensor) -> Tensor:
    """Convert [B, C, H, W] to [B, H*W, C]."""

    return x.flatten(2).transpose(1, 2)


def _to_feature_map(x: Tensor, height: int, width: int) -> Tensor:
    """Convert [B, H*W, C] back to [B, C, H, W]."""

    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], height, width)


class BiasFreeLayerNorm(nn.Module):
    """Bias-free LayerNorm used by the official Transformer utility code."""

    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        return x * torch.rsqrt(variance + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    """LayerNorm with learnable scale and bias."""

    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) * torch.rsqrt(variance + 1e-5)
        return normalized * self.weight + self.bias


class FeatureLayerNorm(nn.Module):
    """Apply LayerNorm along channels independently at every spatial point."""

    def __init__(self, channels: int, layer_norm_type: str = "WithBias") -> None:
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(channels)
        elif layer_norm_type == "WithBias":
            self.body = WithBiasLayerNorm(channels)
        else:
            raise ValueError(
                "layer_norm_type must be 'BiasFree' or 'WithBias', "
                f"but received {layer_norm_type!r}."
            )

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        return _to_feature_map(self.body(_to_tokens(x)), height, width)


class GatedDepthwiseFeedForward(nn.Module):
    """The gated depthwise-convolution FFN used in VideoFusion CMGF."""

    def __init__(
        self,
        channels: int,
        expansion_factor: float = 2.66,
        bias: bool = False,
    ) -> None:
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        if hidden_channels < 1:
            raise ValueError("The expanded FFN width must be positive.")

        self.project_in = nn.Conv2d(
            channels, hidden_channels * 2, kernel_size=1, bias=bias
        )
        self.depthwise = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_channels * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(
            hidden_channels, channels, kernel_size=1, bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(self.project_in(x))
        x_gated, x_linear = x.chunk(2, dim=1)
        return self.project_out(F.gelu(x_gated) * x_linear)


class CrossChannelAttention(nn.Module):
    """Cross attention with a common query and modality-specific key/value.

    The attention matrix shape is [B, heads, C/heads, C/heads]. This retains
    the official transposed/channel-attention logic and avoids an HW-by-HW
    spatial attention matrix.
    """

    def __init__(self, channels: int, num_heads: int = 4, bias: bool = False) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads "
                f"({num_heads})."
            )

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.query_projection = nn.Conv2d(
            channels, channels, kernel_size=1, bias=bias
        )
        self.query_depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=bias,
        )
        self.key_value_projection = nn.Conv2d(
            channels, channels * 2, kernel_size=1, bias=bias
        )
        self.key_value_depthwise = nn.Conv2d(
            channels * 2,
            channels * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels * 2,
            bias=bias,
        )
        self.output_projection = nn.Conv2d(
            channels, channels, kernel_size=1, bias=bias
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, _, height, width = x.shape
        return x.reshape(
            batch, self.num_heads, self.head_dim, height * width
        )

    def forward(self, query_feature: Tensor, source_feature: Tensor) -> Tensor:
        if query_feature.shape != source_feature.shape:
            raise ValueError(
                "query_feature and source_feature must have identical shapes; "
                f"received {tuple(query_feature.shape)} and "
                f"{tuple(source_feature.shape)}."
            )

        batch, _, height, width = query_feature.shape
        query = self._split_heads(
            self.query_depthwise(self.query_projection(query_feature))
        )
        key, value = self.key_value_depthwise(
            self.key_value_projection(source_feature)
        ).chunk(2, dim=1)
        key = self._split_heads(key)
        value = self._split_heads(value)

        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        attention = (query @ key.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)

        output = attention @ value
        output = output.reshape(batch, self.channels, height, width)
        return self.output_projection(output)


class CrossTransformerBlock(nn.Module):
    """Shared CMGF cross-transformer block: cross attention + gated FFN."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "WithBias",
    ) -> None:
        super().__init__()
        self.norm_attention = FeatureLayerNorm(channels, layer_norm_type)
        self.cross_attention = CrossChannelAttention(channels, num_heads, bias)
        self.norm_ffn = FeatureLayerNorm(channels, layer_norm_type)
        self.ffn = GatedDepthwiseFeedForward(
            channels, ffn_expansion_factor, bias
        )

    def forward(self, common_query: Tensor, source_feature: Tensor) -> Tensor:
        # The same normalization weights process both inputs, as in the source.
        output = common_query + self.cross_attention(
            self.norm_attention(common_query),
            self.norm_attention(source_feature),
        )
        return output + self.ffn(self.norm_ffn(output))


class CMGFPETCTFusion(nn.Module):
    """Standalone CMGF module for CT/PET feature fusion.

    Args:
        channels: Channel count C shared by CT, PET, and fused features.
        num_heads: Number of channel-attention heads. C must be divisible by it.
        ffn_expansion_factor: Width multiplier of the gated FFN.
        bias: Whether CMGF convolutions use bias.
        layer_norm_type: ``"WithBias"`` or ``"BiasFree"``.

    Input:
        ct_feature:  [B, C, H, W]
        pet_feature: [B, C, H, W], either calibrated real PET or proxy PET.

    Output:
        fused_feature: [B, C, H, W]
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "WithBias",
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive.")

        self.channels = channels
        # Intentionally one shared block, not separate CT/PET parameters.
        self.shared_cross_transformer = CrossTransformerBlock(
            channels=channels,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type,
        )

    def _validate_inputs(self, ct_feature: Tensor, pet_feature: Tensor) -> None:
        if ct_feature.ndim != 4 or pet_feature.ndim != 4:
            raise ValueError(
                "CMGF expects CT and PET tensors in [B, C, H, W] format."
            )
        if ct_feature.shape != pet_feature.shape:
            raise ValueError(
                "CT and PET features must have identical shapes; received "
                f"{tuple(ct_feature.shape)} and {tuple(pet_feature.shape)}."
            )
        if ct_feature.shape[1] != self.channels:
            raise ValueError(
                f"The module was built for {self.channels} channels, but the "
                f"input has {ct_feature.shape[1]}."
            )
        if ct_feature.device != pet_feature.device:
            raise ValueError("CT and PET features must be on the same device.")
        if ct_feature.dtype != pet_feature.dtype:
            raise ValueError("CT and PET features must have the same dtype.")

    def forward(self, ct_feature: Tensor, pet_feature: Tensor) -> Tensor:
        self._validate_inputs(ct_feature, pet_feature)

        # Official CMGF public/comprehensive query.
        common_query = ct_feature + pet_feature

        # Two modality retrievals use the same transformer parameters.
        ct_guided = self.shared_cross_transformer(common_query, ct_feature)
        pet_guided = self.shared_cross_transformer(common_query, pet_feature)

        # Preserve the official code path exactly: sum the two branch outputs.
        return ct_guided + pet_guided


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return device


def _parameter_counts(module: nn.Module) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def run_self_test(args: argparse.Namespace) -> None:
    """Run standalone shape, finite-value, backward, and symmetry checks."""

    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    module = CMGFPETCTFusion(
        channels=args.channels,
        num_heads=args.num_heads,
        ffn_expansion_factor=args.ffn_expansion_factor,
        bias=args.bias,
        layer_norm_type=args.layer_norm_type,
    ).to(device)
    module.train()

    shape = (args.batch_size, args.channels, args.height, args.width)
    ct_feature = torch.randn(shape, device=device, requires_grad=True)
    pet_feature = torch.randn(shape, device=device, requires_grad=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=use_amp,
    ):
        fused_feature = module(ct_feature, pet_feature)
        loss = fused_feature.square().mean()

    if not torch.isfinite(fused_feature).all():
        raise RuntimeError("The forward result contains NaN or Inf values.")
    if fused_feature.shape != ct_feature.shape:
        raise RuntimeError(
            f"Output shape {tuple(fused_feature.shape)} does not match the "
            f"input shape {tuple(ct_feature.shape)}."
        )

    loss.backward()
    if ct_feature.grad is None or pet_feature.grad is None:
        raise RuntimeError("Backward did not produce gradients for both inputs.")
    if not torch.isfinite(ct_feature.grad).all() or not torch.isfinite(
        pet_feature.grad
    ).all():
        raise RuntimeError("The backward result contains NaN or Inf gradients.")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time

    # Official shared-branch CMGF is invariant to swapping its two inputs.
    module.eval()
    with torch.no_grad():
        original_order = module(ct_feature.detach(), pet_feature.detach())
        swapped_order = module(pet_feature.detach(), ct_feature.detach())
        symmetry_error = (original_order - swapped_order).abs().max().item()

    total_parameters, trainable_parameters = _parameter_counts(module)
    attention_side = args.channels // args.num_heads

    print("CMGF CT/PET standalone test: PASS")
    print(f"device: {device}")
    print(f"AMP enabled: {use_amp}")
    print(f"input/output shape: {tuple(fused_feature.shape)}")
    print(f"output dtype: {fused_feature.dtype}")
    print(f"parameters: {total_parameters:,} total / {trainable_parameters:,} trainable")
    print(
        "attention matrix per head: "
        f"[{attention_side}, {attention_side}] (channel attention, not HW x HW)"
    )
    print(f"CT/PET swap max error: {symmetry_error:.3e}")
    print(f"forward + backward time: {elapsed:.3f} s")
    if device.type == "cuda":
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"peak allocated CUDA memory: {peak_mib:.1f} MiB")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone CMGF test for paired CT/PET feature maps."
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-expansion-factor", type=float, default=2.66)
    parser.add_argument(
        "--layer-norm-type",
        choices=("WithBias", "BiasFree"),
        default="WithBias",
    )
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=7)
    return parser


if __name__ == "__main__":
    run_self_test(build_argument_parser().parse_args())