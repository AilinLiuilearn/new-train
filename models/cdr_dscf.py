"""Standalone CDR-DSCF fusion module for CT/PET feature fusion.

Old implementation:
    Explicit continuous pairwise relative-position bias over the full QxN
    offset tensor, with query/sample chunking and activation recomputation.

New implementation:
    Dynamic CDR offsets -> four-way CT/PET cross-grid sampling -> modal
    competition -> Q/K coordinate position embeddings -> PyTorch SDPA sparse
    multi-head attention -> residual output.

Retained:
    CDR offset regulation, four-way cross sampling, modal competition, sparse
    multi-head attention (Query = full-resolution fused features; Key/Value =
    dynamically sampled features), and residual output.

Position modeling change:
    Pairwise continuous position bias is replaced by additive coordinate
    embeddings on Query and Key. This is not claimed to be mathematically
    equivalent to the old bias path. The goal is to avoid materializing explicit
    QxN position tensors and to let scaled_dot_product_attention use an
    efficient backend.

``use_position_bias`` is kept for constructor compatibility; it now enables
coordinate position embedding rather than pairwise bias.

The public module accepts only two tensors and returns one tensor:

    fused = CDRDSCFFusion2D(...)(ct_feature, pet_feature)

Run this file directly for a shape and gradient self-test:

    python -m models.cdr_dscf
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class CDRDSCFDebugOutput:
    """Detached diagnostic tensors returned when ``return_debug=True``."""

    alpha: Tensor
    ct_raw_offset: Tensor
    pet_raw_offset: Tensor
    consensus_offset: Tensor
    difference_offset: Tensor
    ct_regulated_offset: Tensor
    pet_regulated_offset: Tensor
    modal_weights: Tensor


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for a BCHW feature map."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class OffsetPredictor(nn.Module):
    """Predict grouped 2-D offsets in ``(dy, dx)`` channel order."""

    def __init__(self, group_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("offset_kernel_size must be odd")

        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(
                group_channels,
                group_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=group_channels,
                bias=True,
            ),
            LayerNorm2d(group_channels),
            nn.GELU(),
            nn.Conv2d(group_channels, 2, kernel_size=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CDRDeformableSparseAttention2D(nn.Module):
    """CDR-regulated dynamic sparse cross-modal attention."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        num_groups: int = 4,
        sampling_stride: int = 4,
        offset_kernel_size: int = 5,
        use_position_bias: bool = True,
        attention_residual_init: float = 1e-3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        if channels % num_groups != 0:
            raise ValueError("channels must be divisible by num_groups")
        if num_heads % num_groups != 0:
            raise ValueError("num_heads must be divisible by num_groups")
        if sampling_stride < 1:
            raise ValueError("sampling_stride must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.channels = channels
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.group_channels = channels // num_groups
        self.head_channels = channels // num_heads
        self.heads_per_group = num_heads // num_groups
        self.scale = self.head_channels**-0.5
        self.sampling_stride = sampling_stride
        # Compatibility name: enables Q/K coordinate position embeddings.
        self.use_position_bias = use_position_bias

        self.ct_offset_predictor = OffsetPredictor(
            self.group_channels, offset_kernel_size, sampling_stride
        )
        self.pet_offset_predictor = OffsetPredictor(
            self.group_channels, offset_kernel_size, sampling_stride
        )

        self.alpha_logit = nn.Parameter(torch.zeros(1, num_groups, 2, 1, 1))

        self.query_fusion = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.query_projection = nn.Conv2d(channels, channels, kernel_size=1)

        self.modal_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, kernel_size=1),
        )

        self.key_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.value_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.output_projection = nn.Conv2d(channels, channels, kernel_size=1)

        if use_position_bias:
            self.query_position_projection = nn.Linear(2, self.head_channels, bias=True)
            self.key_position_projection = nn.Linear(2, self.head_channels, bias=True)
            nn.init.zeros_(self.query_position_projection.weight)
            nn.init.zeros_(self.query_position_projection.bias)
            nn.init.zeros_(self.key_position_projection.weight)
            nn.init.zeros_(self.key_position_projection.bias)
        else:
            self.query_position_projection = None
            self.key_position_projection = None

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        self.attention_weight = nn.Parameter(
            torch.full((channels,), float(attention_residual_init))
        )
        self.identity_weight = nn.Parameter(torch.ones(channels))

    def alpha(self) -> Tensor:
        return 1.0 + torch.tanh(self.alpha_logit)

    @staticmethod
    def _reference_grid(
        height: int,
        width: int,
        batch_groups: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        if height == 1:
            y = torch.zeros(1, dtype=dtype, device=device)
        else:
            y = torch.linspace(-1.0, 1.0, height, dtype=dtype, device=device)
        if width == 1:
            x = torch.zeros(1, dtype=dtype, device=device)
        else:
            x = torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        reference = torch.stack((yy, xx), dim=-1)
        return reference.unsqueeze(0).expand(batch_groups, -1, -1, -1)

    def _predict_and_regulate_offsets(
        self, ct: Tensor, pet: Tensor
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        b, c, h, w = ct.shape
        g = self.num_groups

        ct_grouped = ct.reshape(b, g, self.group_channels, h, w).reshape(
            b * g, self.group_channels, h, w
        )
        pet_grouped = pet.reshape(b, g, self.group_channels, h, w).reshape(
            b * g, self.group_channels, h, w
        )

        ct_raw_flat = self.ct_offset_predictor(ct_grouped)
        pet_raw_flat = self.pet_offset_predictor(pet_grouped)
        hs, ws = ct_raw_flat.shape[-2:]

        ct_raw = ct_raw_flat.reshape(b, g, 2, hs, ws)
        pet_raw = pet_raw_flat.reshape(b, g, 2, hs, ws)

        consensus = 0.5 * (ct_raw + pet_raw)
        difference = 0.5 * (pet_raw - ct_raw)
        alpha = self.alpha()

        ct_regulated = consensus - alpha * difference
        pet_regulated = consensus + alpha * difference

        debug = {
            "alpha": alpha,
            "ct_raw_offset": ct_raw,
            "pet_raw_offset": pet_raw,
            "consensus_offset": consensus,
            "difference_offset": difference,
            "ct_regulated_offset": ct_regulated,
            "pet_regulated_offset": pet_regulated,
        }

        return (
            ct_regulated.reshape(b * g, 2, hs, ws),
            pet_regulated.reshape(b * g, 2, hs, ws),
            debug,
        )

    def _sample(self, feature: Tensor, position_yx: Tensor) -> Tensor:
        b, _, h, w = feature.shape
        g = self.num_groups
        feature = feature.reshape(b, g, self.group_channels, h, w).reshape(
            b * g, self.group_channels, h, w
        )

        sampled = F.grid_sample(
            feature,
            position_yx[..., (1, 0)],
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        hs, ws = sampled.shape[-2:]
        sampled = sampled.reshape(b, g, self.group_channels, hs * ws)
        return sampled.reshape(b, self.channels, 1, hs * ws)

    def _sdpa_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        original_head_dim = query.shape[-1]
        padded_head_dim = ((original_head_dim + 7) // 8) * 8
        pad_dim = padded_head_dim - original_head_dim

        if pad_dim > 0:
            query = F.pad(query, (0, pad_dim))
            key = F.pad(key, (0, pad_dim))
            value = F.pad(value, (0, pad_dim))
            # Keep the original 1/sqrt(head_dim) scale after SDPA's padded scale.
            query = query * math.sqrt(padded_head_dim / original_head_dim)

        dropout_p = self.attention_dropout.p if self.training else 0.0
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=False,
        )

        if pad_dim > 0:
            output = output[..., :original_head_dim]

        return output

    def forward(
        self,
        ct: Tensor,
        pet: Tensor,
        return_debug: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, CDRDSCFDebugOutput]]:
        if ct.ndim != 4 or pet.ndim != 4:
            raise ValueError("ct and pet must both be 4-D BCHW tensors")
        if ct.shape != pet.shape:
            raise ValueError(
                f"ct and pet must have identical shapes, got {ct.shape} and {pet.shape}"
            )
        if ct.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {ct.shape[1]}")

        b, c, h, w = ct.shape
        identity = self.query_fusion(torch.cat([ct, pet], dim=1))
        query_map = self.query_projection(identity)

        ct_offset, pet_offset, debug_dict = self._predict_and_regulate_offsets(ct, pet)
        hs, ws = ct_offset.shape[-2:]
        bg = b * self.num_groups
        reference = self._reference_grid(hs, ws, bg, ct.dtype, ct.device)

        ct_position = (reference + ct_offset.permute(0, 2, 3, 1)).clamp(-1.0, 1.0)
        pet_position = (reference + pet_offset.permute(0, 2, 3, 1)).clamp(-1.0, 1.0)

        ct_at_ct = self._sample(ct, ct_position)
        pet_at_ct = self._sample(pet, ct_position)
        ct_at_pet = self._sample(ct, pet_position)
        pet_at_pet = self._sample(pet, pet_position)

        query_at_ct = self._sample(query_map, ct_position)
        query_at_pet = self._sample(query_map, pet_position)
        sampled_query = torch.cat([query_at_ct, query_at_pet], dim=-1)

        modal_logits = self.modal_gate(sampled_query)
        modal_weights = torch.softmax(modal_logits, dim=1)

        ct_samples = torch.cat([ct_at_ct, ct_at_pet], dim=-1)
        pet_samples = torch.cat([pet_at_ct, pet_at_pet], dim=-1)
        sparse_samples = modal_weights[:, 0:1] * ct_samples + modal_weights[:, 1:2] * pet_samples

        num_queries = h * w
        num_samples = 2 * hs * ws

        query = query_map.reshape(
            b,
            self.num_heads,
            self.head_channels,
            num_queries,
        ).permute(0, 1, 3, 2).contiguous()
        key = self.key_projection(sparse_samples).reshape(
            b,
            self.num_heads,
            self.head_channels,
            num_samples,
        ).permute(0, 1, 3, 2).contiguous()
        value = self.value_projection(sparse_samples).reshape(
            b,
            self.num_heads,
            self.head_channels,
            num_samples,
        ).permute(0, 1, 3, 2).contiguous()

        if (
            self.query_position_projection is not None
            and self.key_position_projection is not None
        ):
            query_grid = self._reference_grid(
                height=h,
                width=w,
                batch_groups=1,
                dtype=query.dtype,
                device=query.device,
            ).reshape(1, num_queries, 2)
            query_position = self.query_position_projection(query_grid).unsqueeze(1)
            query = query + query_position

            # Same CT-then-PET order as sparse_samples.
            sampled_positions = torch.cat(
                [
                    ct_position.reshape(b, self.num_groups, -1, 2),
                    pet_position.reshape(b, self.num_groups, -1, 2),
                ],
                dim=2,
            )
            key_position = self.key_position_projection(sampled_positions)
            key_position = key_position.repeat_interleave(self.heads_per_group, dim=1)
            key = key + key_position

        output = self._sdpa_attention(query, key, value)
        output = output.permute(0, 1, 3, 2).contiguous().reshape(b, c, h, w)
        output = self.output_dropout(self.output_projection(output))

        attention_weight = self.attention_weight.view(1, c, 1, 1)
        identity_weight = self.identity_weight.view(1, c, 1, 1)
        output = attention_weight * output + identity_weight * identity

        if not return_debug:
            return output

        debug_output = CDRDSCFDebugOutput(
            alpha=debug_dict["alpha"].detach(),
            ct_raw_offset=debug_dict["ct_raw_offset"].detach(),
            pet_raw_offset=debug_dict["pet_raw_offset"].detach(),
            consensus_offset=debug_dict["consensus_offset"].detach(),
            difference_offset=debug_dict["difference_offset"].detach(),
            ct_regulated_offset=debug_dict["ct_regulated_offset"].detach(),
            pet_regulated_offset=debug_dict["pet_regulated_offset"].detach(),
            modal_weights=modal_weights.detach(),
        )
        return output, debug_output


class CDRDSCFFusion2D(nn.Module):
    """Independent two-input/one-output CDR-DSCF fusion block."""

    def __init__(
        self,
        in_channels: int,
        low_rank_ratio: float = 0.125,
        low_channels: Optional[int] = None,
        num_heads: int = 4,
        num_groups: int = 4,
        sampling_stride: int = 4,
        offset_kernel_size: int = 5,
        use_position_bias: bool = True,
        attention_residual_init: float = 1e-3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        if low_channels is None:
            if not 0.0 < low_rank_ratio <= 1.0:
                raise ValueError("low_rank_ratio must be in (0, 1]")
            low_channels = max(num_heads, int(round(in_channels * low_rank_ratio)))

        divisor = math.lcm(num_heads, num_groups)
        if low_channels % divisor != 0:
            raise ValueError(
                f"low_channels={low_channels} must be divisible by lcm("
                f"num_heads={num_heads}, num_groups={num_groups})={divisor}. "
                "Pass low_channels explicitly or adjust low_rank_ratio."
            )

        self.in_channels = in_channels
        self.low_channels = low_channels

        self.ct_down = nn.Conv2d(in_channels, low_channels, kernel_size=1)
        self.pet_down = nn.Conv2d(in_channels, low_channels, kernel_size=1)

        self.sparse_fusion = CDRDeformableSparseAttention2D(
            channels=low_channels,
            num_heads=num_heads,
            num_groups=num_groups,
            sampling_stride=sampling_stride,
            offset_kernel_size=offset_kernel_size,
            use_position_bias=use_position_bias,
            attention_residual_init=attention_residual_init,
            dropout=dropout,
        )
        self.channel_up = nn.Conv2d(low_channels, in_channels, kernel_size=1)

    def get_alpha(self, detach: bool = True) -> Tensor:
        alpha = self.sparse_fusion.alpha()
        return alpha.detach() if detach else alpha

    def forward(
        self,
        ct_feature: Tensor,
        pet_feature: Tensor,
        return_debug: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, CDRDSCFDebugOutput]]:
        if ct_feature.ndim != 4 or pet_feature.ndim != 4:
            raise ValueError("inputs must use BCHW layout")
        if ct_feature.shape != pet_feature.shape:
            raise ValueError("CT and PET features must have identical shapes")
        if ct_feature.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {ct_feature.shape[1]}"
            )

        ct_low = self.ct_down(ct_feature)
        pet_low = self.pet_down(pet_feature)

        if return_debug:
            fused_low, debug = self.sparse_fusion(ct_low, pet_low, return_debug=True)
            return self.channel_up(fused_low), debug

        fused_low = self.sparse_fusion(ct_low, pet_low, return_debug=False)
        return self.channel_up(fused_low)


def _count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _self_test() -> None:
    """Run lightweight shape and gradient checks."""
    torch.manual_seed(7)

    block = CDRDSCFFusion2D(
        in_channels=64,
        low_channels=16,
        num_heads=4,
        num_groups=4,
        sampling_stride=4,
        offset_kernel_size=5,
        use_position_bias=True,
        dropout=0.0,
    )
    block.eval()

    assert block.sparse_fusion.query_position_projection is not None
    assert block.sparse_fusion.key_position_projection is not None
    assert block.sparse_fusion.query_position_projection.weight.abs().max().item() == 0.0
    assert block.sparse_fusion.key_position_projection.weight.abs().max().item() == 0.0

    ct = torch.randn(2, 64, 16, 16, requires_grad=True)
    real_pet = torch.randn(2, 64, 16, 16, requires_grad=True)
    compensated_pet = torch.randn(2, 64, 16, 16, requires_grad=True)

    full_output = block(ct, real_pet)
    missing_output = block(ct, compensated_pet)
    debug_output, debug = block(ct, real_pet, return_debug=True)

    assert full_output.shape == ct.shape
    assert missing_output.shape == ct.shape
    assert debug_output.shape == ct.shape

    max_output_error = (full_output - debug_output).abs().max()
    assert max_output_error.item() < 1e-6

    for tensor in [
        debug.alpha,
        debug.ct_raw_offset,
        debug.pet_raw_offset,
        debug.consensus_offset,
        debug.difference_offset,
        debug.ct_regulated_offset,
        debug.pet_regulated_offset,
        debug.modal_weights,
    ]:
        assert torch.isfinite(tensor).all()

    assert (debug.ct_regulated_offset - debug.ct_raw_offset).abs().max().item() < 1e-6
    assert (debug.pet_regulated_offset - debug.pet_raw_offset).abs().max().item() < 1e-6

    block.train()
    ct_train = torch.randn(2, 64, 16, 16, requires_grad=True)
    pet_train = torch.randn(2, 64, 16, 16, requires_grad=True)
    compensated_train = torch.randn(2, 64, 16, 16, requires_grad=True)
    full_train = block(ct_train, pet_train)
    missing_train = block(ct_train, compensated_train)
    loss = full_train.square().mean() + missing_train.square().mean()
    loss.backward()

    required_grads = {
        "ct_down": block.ct_down.weight.grad,
        "pet_down": block.pet_down.weight.grad,
        "ct_offset_predictor": block.sparse_fusion.ct_offset_predictor.net[0].weight.grad,
        "pet_offset_predictor": block.sparse_fusion.pet_offset_predictor.net[0].weight.grad,
        "alpha_logit": block.sparse_fusion.alpha_logit.grad,
        "modal_gate": block.sparse_fusion.modal_gate[0].weight.grad,
        "query_projection": block.sparse_fusion.query_projection.weight.grad,
        "key_projection": block.sparse_fusion.key_projection.weight.grad,
        "value_projection": block.sparse_fusion.value_projection.weight.grad,
        "query_position_projection": block.sparse_fusion.query_position_projection.weight.grad,
        "key_position_projection": block.sparse_fusion.key_position_projection.weight.grad,
        "output_projection": block.sparse_fusion.output_projection.weight.grad,
        "channel_up": block.channel_up.weight.grad,
    }
    for name, grad in required_grads.items():
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name

    alpha_grad = block.sparse_fusion.alpha_logit.grad
    assert alpha_grad is not None
    assert torch.isfinite(alpha_grad).all()

    print("CDR-DSCF standalone self-test passed")
    print(f"  parameters: {_count_parameters(block):,}")
    print(f"  Full output shape:    {tuple(full_output.shape)}")
    print(f"  Missing output shape: {tuple(missing_output.shape)}")
    print(f"  alpha shape: {tuple(block.get_alpha().shape)}")
    print(
        "  initial alpha (group x [dy, dx]):\n",
        block.get_alpha().squeeze(0).squeeze(-1).squeeze(-1),
    )
    print(f"  max output error:              {max_output_error.item():.3e}")
    print(f"  alpha gradient L1: {alpha_grad.abs().sum().item():.3e}")
    print(
        "  query/key position weight grad L1: "
        f"{block.sparse_fusion.query_position_projection.weight.grad.abs().sum().item():.3e} / "
        f"{block.sparse_fusion.key_position_projection.weight.grad.abs().sum().item():.3e}"
    )


if __name__ == "__main__":
    _self_test()
