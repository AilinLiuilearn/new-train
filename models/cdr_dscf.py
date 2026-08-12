"""Standalone CDR-DSCF fusion module for CT/PET feature fusion.

This file reimplements the central idea of the Dynamic Sparse Cross-modality
Fusion (DSCF) block from:

    Keep the Balance: A Parameter-Efficient Symmetrical Framework for RGB+X
    Semantic Segmentation, CVPR 2025.

Reference implementation (Apache-2.0):
    https://github.com/imcjx/KTB

The modification introduced here is Consensus-Difference Regulation (CDR).
The two modality-specific raw sampling offsets are decomposed into a common
center and a modality difference. A small, bounded, learnable alpha then
controls how much of that difference is retained before DSCF's cross-grid
sampling and sparse attention are performed.

The public module accepts only two tensors and returns one tensor:

    fused = CDRDSCFFusion2D(...)(ct_feature, pet_feature)

No Full/Missing flag, retrieval score, prototype feature, or reliability
estimate is required. The PET input may therefore be either a corrected real
PET feature or a corrected compensated PET feature.

Run this file directly for a shape/equivalence/gradient self-test:

    python cdr_dscf.py
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
    """Predict grouped 2-D offsets in ``(dy, dx)`` channel order.

    Modality groups are folded into the batch dimension before this module is
    called, matching the grouped offset prediction used by the reference DSCF.
    """

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


class ContinuousRelativePositionBias(nn.Module):
    """Continuous relative-position bias for sparse sampled coordinates.

    This is a resolution-independent version of the continuous position-bias
    branch used by deformable attention. It maps normalized ``(dy, dx)``
    displacements to one bias per attention head within a sampling group.
    """

    def __init__(self, heads_per_group: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, heads_per_group, bias=False),
        )

    def forward(self, displacement: Tensor) -> Tensor:
        # Log compression prevents large normalized displacements from
        # dominating the attention logits.
        displacement = (
            torch.sign(displacement)
            * torch.log2(torch.abs(displacement) + 1.0)
            / math.log2(8.0)
        )
        return self.mlp(displacement)


class CDRDeformableSparseAttention2D(nn.Module):
    """CDR-regulated dynamic sparse cross-modal attention.

    Parameters
    ----------
    channels:
        Channel count after the outer low-rank projections.
    num_heads:
        Number of sparse-attention heads.
    num_groups:
        Number of grouped offset fields. ``channels`` and ``num_heads`` must
        both be divisible by this value.
    sampling_stride:
        Spatial stride of the sparse offset predictors.
    offset_kernel_size:
        Kernel size used by each depth-wise offset predictor.
    use_position_bias:
        Add continuous relative-position bias to sparse attention.
    attention_residual_init:
        Initial scale of the deformable-attention residual branch. A small
        value reproduces the conservative early-stage behavior of DSCF.
    dropout:
        Dropout probability applied to attention weights and output features.

    Notes
    -----
    For every group and coordinate direction, CDR learns

        alpha = 1 + tanh(alpha_logit),  0 < alpha < 2.

    Zero initialization gives alpha=1, making regulated offsets exactly equal
    to the two raw offsets at initialization (the original DSCF behavior).
    """

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
        self.use_position_bias = use_position_bias

        self.ct_offset_predictor = OffsetPredictor(
            self.group_channels, offset_kernel_size, sampling_stride
        )
        self.pet_offset_predictor = OffsetPredictor(
            self.group_channels, offset_kernel_size, sampling_stride
        )

        # One bounded controller for each group and coordinate direction.
        # Shape: [1, G, (dy, dx), 1, 1].
        self.alpha_logit = nn.Parameter(torch.zeros(1, num_groups, 2, 1, 1))

        # Full-resolution bimodal query/identity feature.
        self.query_fusion = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.query_projection = nn.Conv2d(channels, channels, kernel_size=1)

        # Point-wise competition between CT and PET at the sampled locations.
        self.modal_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, kernel_size=1),
        )

        self.key_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.value_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.output_projection = nn.Conv2d(channels, channels, kernel_size=1)

        if use_position_bias:
            self.position_bias = ContinuousRelativePositionBias(
                heads_per_group=self.heads_per_group
            )
        else:
            self.position_bias = None

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        # Per-channel residual mixing, retained from DSCF.
        self.attention_weight = nn.Parameter(
            torch.full((channels,), float(attention_residual_init))
        )
        self.identity_weight = nn.Parameter(torch.ones(channels))

    def alpha(self) -> Tensor:
        """Return bounded CDR coefficients with shape ``[1,G,2,1,1]``."""
        return 1.0 + torch.tanh(self.alpha_logit)

    @staticmethod
    def _reference_grid(
        height: int,
        width: int,
        batch_groups: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """Create a normalized grid in ``(y, x)`` order."""
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
        """Predict raw offsets and apply consensus-difference regulation."""
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
        """Group-wise bilinear sampling.

        ``feature`` is BCHW. ``position_yx`` has shape ``[B*G,Hs,Ws,2]``
        in ``(y,x)`` order. The returned tensor is ``[B,C,1,Hs*Ws]``.
        """
        b, _, h, w = feature.shape
        g = self.num_groups
        feature = feature.reshape(b, g, self.group_channels, h, w).reshape(
            b * g, self.group_channels, h, w
        )

        sampled = F.grid_sample(
            feature,
            position_yx[..., (1, 0)],  # grid_sample expects (x, y)
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        hs, ws = sampled.shape[-2:]
        sampled = sampled.reshape(b, g, self.group_channels, hs * ws)
        return sampled.reshape(b, self.channels, 1, hs * ws)

    def _relative_position_bias(
        self,
        query_h: int,
        query_w: int,
        ct_position: Tensor,
        pet_position: Tensor,
        batch_size: int,
    ) -> Tensor:
        """Return bias with shape ``[B*num_heads, HW, 2*Ns]``."""
        if self.position_bias is None:
            raise RuntimeError("position bias is disabled")

        bg = batch_size * self.num_groups
        query_grid = self._reference_grid(
            query_h,
            query_w,
            bg,
            ct_position.dtype,
            ct_position.device,
        ).reshape(bg, query_h * query_w, 2)

        sampled_positions = torch.cat(
            [
                ct_position.reshape(bg, -1, 2),
                pet_position.reshape(bg, -1, 2),
            ],
            dim=1,
        )
        displacement = query_grid.unsqueeze(2) - sampled_positions.unsqueeze(1)
        bias = self.position_bias(displacement)

        # [B*G, HW, 2Ns, heads/group]
        bias = bias.reshape(
            batch_size,
            self.num_groups,
            query_h * query_w,
            sampled_positions.shape[1],
            self.heads_per_group,
        )
        bias = bias.permute(0, 1, 4, 2, 3).contiguous()
        return bias.reshape(
            batch_size * self.num_heads,
            query_h * query_w,
            sampled_positions.shape[1],
        )

    def forward(
        self,
        ct: Tensor,
        pet: Tensor,
        return_debug: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, CDRDSCFDebugOutput]]:
        """Fuse two same-shaped BCHW feature maps."""
        if ct.ndim != 4 or pet.ndim != 4:
            raise ValueError("ct and pet must both be 4-D BCHW tensors")
        if ct.shape != pet.shape:
            raise ValueError(
                f"ct and pet must have identical shapes, got {ct.shape} and {pet.shape}"
            )
        if ct.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {ct.shape[1]}"
            )

        b, c, h, w = ct.shape
        identity = self.query_fusion(torch.cat([ct, pet], dim=1))
        query_map = self.query_projection(identity)

        ct_offset, pet_offset, debug_dict = self._predict_and_regulate_offsets(
            ct, pet
        )
        hs, ws = ct_offset.shape[-2:]
        bg = b * self.num_groups
        reference = self._reference_grid(
            hs, ws, bg, ct.dtype, ct.device
        )

        # Offsets are predicted in normalized-grid units, as in the default
        # clamp-based branch of the reference deformable attention.
        ct_position = (
            reference + ct_offset.permute(0, 2, 3, 1)
        ).clamp(-1.0, 1.0)
        pet_position = (
            reference + pet_offset.permute(0, 2, 3, 1)
        ).clamp(-1.0, 1.0)

        # Four cross-grid samples: each modality is read at both coordinate sets.
        ct_at_ct = self._sample(ct, ct_position)
        pet_at_ct = self._sample(pet, ct_position)
        ct_at_pet = self._sample(ct, pet_position)
        pet_at_pet = self._sample(pet, pet_position)

        # Query samples determine point-wise CT/PET competition separately on
        # the CT-proposed and PET-proposed coordinate sets.
        query_at_ct = self._sample(query_map, ct_position)
        query_at_pet = self._sample(query_map, pet_position)
        sampled_query = torch.cat([query_at_ct, query_at_pet], dim=-1)

        modal_logits = self.modal_gate(sampled_query)
        modal_weights = torch.softmax(modal_logits, dim=1)

        ct_samples = torch.cat([ct_at_ct, ct_at_pet], dim=-1)
        pet_samples = torch.cat([pet_at_ct, pet_at_pet], dim=-1)
        sparse_samples = (
            modal_weights[:, 0:1] * ct_samples
            + modal_weights[:, 1:2] * pet_samples
        )

        num_samples = 2 * hs * ws
        query = query_map.reshape(
            b, self.num_heads, self.head_channels, h * w
        ).reshape(b * self.num_heads, self.head_channels, h * w)
        key = self.key_projection(sparse_samples).reshape(
            b, self.num_heads, self.head_channels, num_samples
        ).reshape(b * self.num_heads, self.head_channels, num_samples)
        value = self.value_projection(sparse_samples).reshape(
            b, self.num_heads, self.head_channels, num_samples
        ).reshape(b * self.num_heads, self.head_channels, num_samples)

        attention = torch.einsum("bcm,bcn->bmn", query, key) * self.scale
        if self.position_bias is not None:
            attention = attention + self._relative_position_bias(
                h, w, ct_position, pet_position, b
            )
        attention = self.attention_dropout(torch.softmax(attention, dim=-1))

        output = torch.einsum("bmn,bcn->bcm", attention, value)
        output = output.reshape(b, c, h, w)
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
    """Independent two-input/one-output CDR-DSCF fusion block.

    Input and output shapes are all ``[B, in_channels, H, W]``. Separate 1x1
    projections reproduce DSCF's modality-specific low-rank linear mappings;
    the final projection restores the original channel count.

    Example
    -------
    >>> block = CDRDSCFFusion2D(128, low_rank_ratio=0.125,
    ...                         num_heads=4, num_groups=4)
    >>> ct = torch.randn(2, 128, 32, 32)
    >>> pet = torch.randn(2, 128, 32, 32)
    >>> fused = block(ct, pet)
    >>> fused.shape
    torch.Size([2, 128, 32, 32])
    """

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

        # Linear projection on tokens is equivalent to 1x1 convolution on BCHW.
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
        """Return ``[1,G,2,1,1]`` CDR coefficients for inspection."""
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
                f"expected {self.in_channels} input channels, "
                f"got {ct_feature.shape[1]}"
            )

        ct_low = self.ct_down(ct_feature)
        pet_low = self.pet_down(pet_feature)

        if return_debug:
            fused_low, debug = self.sparse_fusion(
                ct_low, pet_low, return_debug=True
            )
            return self.channel_up(fused_low), debug

        fused_low = self.sparse_fusion(ct_low, pet_low, return_debug=False)
        return self.channel_up(fused_low)


def _count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _self_test() -> None:
    """Run lightweight Full/Missing, equivalence, and gradient checks."""
    torch.manual_seed(7)

    block = CDRDSCFFusion2D(
        in_channels=64,
        low_channels=16,
        num_heads=4,
        num_groups=4,
        sampling_stride=4,
        offset_kernel_size=5,
        use_position_bias=True,
    )
    block.train()

    ct = torch.randn(2, 64, 16, 16, requires_grad=True)
    real_pet = torch.randn(2, 64, 16, 16, requires_grad=True)
    compensated_pet = (
        F.avg_pool2d(torch.randn(2, 64, 18, 18), kernel_size=3, stride=1)
        .contiguous()
        .requires_grad_()
    )

    full_output, full_debug = block(ct, real_pet, return_debug=True)
    missing_output = block(ct, compensated_pet)

    expected_shape = ct.shape
    assert full_output.shape == expected_shape
    assert missing_output.shape == expected_shape

    # alpha=1 at initialization must exactly recover each raw offset.
    ct_equivalence_error = (
        full_debug.ct_regulated_offset - full_debug.ct_raw_offset
    ).abs().max()
    pet_equivalence_error = (
        full_debug.pet_regulated_offset - full_debug.pet_raw_offset
    ).abs().max()
    assert ct_equivalence_error.item() < 1e-6
    assert pet_equivalence_error.item() < 1e-6

    loss = full_output.square().mean() + missing_output.square().mean()
    loss.backward()

    alpha_grad = block.sparse_fusion.alpha_logit.grad
    assert alpha_grad is not None
    assert torch.isfinite(alpha_grad).all()

    print("CDR-DSCF standalone self-test passed")
    print(f"  parameters: { _count_parameters(block):,}")
    print(f"  Full output shape:    {tuple(full_output.shape)}")
    print(f"  Missing output shape: {tuple(missing_output.shape)}")
    print(f"  alpha shape: {tuple(block.get_alpha().shape)}")
    print(
        "  initial alpha (group x [dy, dx]):\n",
        block.get_alpha().squeeze(0).squeeze(-1).squeeze(-1),
    )
    print(f"  max CT offset equivalence error:  {ct_equivalence_error.item():.3e}")
    print(f"  max PET offset equivalence error: {pet_equivalence_error.item():.3e}")
    print(f"  alpha gradient L1: {alpha_grad.abs().sum().item():.3e}")


if __name__ == "__main__":
    _self_test()