"""Evidence-guided full-resolution CT/PET fusion with neighborhood attention.

This file is intentionally standalone.  It implements the complete fusion
module discussed for paired CT/PET segmentation features:

1. Select one of two fixed PET-state text embeddings (real or compensated).
2. Apply scale-specific residual multiplicative text modulation to PET.
3. Estimate PET evidential vacuity and CT/PET evidential conflict.
4. Build one PET trust map from those two quantities.
5. Correct PET from CT with full-resolution dilated-neighborhood cross-attention.
6. Transfer only trusted PET evidence back to CT with the same attention core.
7. Produce one reliability-centred fused feature for the shared decoder.

No Q/K/V pooling, token merging, feature interpolation, global HW-by-HW
attention matrix, ``unfold``, additional loss, or online text encoder is used.

The attention design is based on:

* Neighborhood Attention Transformer (CVPR 2023), which makes spatial
  attention linear in the number of image tokens by restricting every query
  to a sliding neighborhood.
* Dilated Neighborhood Attention Transformer, which expands the receptive
  field without increasing the number of attended keys.

For production training, install NATTEN and use ``attention_backend="natten"``
or ``"auto"`` on CUDA.  A chunked pure-PyTorch backend is included for
correctness tests and environments without NATTEN.  It deliberately avoids
``unfold`` and bounds temporary memory by processing a few query rows at a
time, but it is slower than the fused NATTEN kernel.

Expected four-scale project interface:

    channels = (64, 128, 320, 512)
    ct_feats  = [B,C1,H1,W1], ..., [B,C4,H4,W4]
    pet_feats = [B,C1,H1,W1], ..., [B,C4,H4,W4]

State mapping:

    0 = real/full PET
    1 = prototype-compensated/missing PET

In ``auto`` mode, ``pet_available=1`` selects state 0 and
``pet_available=0`` selects state 1 independently for every sample.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

BIOMEDCLIP_MODEL_PATH = (
    "/root/autodl-tmp/mkd-main/new-train/"
    "pretrained/biomedclip_model"
)
BIOMEDBERT_TEXT_TOWER_PATH = (
    "/root/autodl-tmp/mkd-main/new-train/"
    "pretrained/biomedbert_text_tower"
)

try:
    # NATTEN >= 0.21 exposes fused neighborhood attention at package level.
    from natten import na2d as _natten_na2d

    _NATTEN_AVAILABLE = True
except (ImportError, OSError):
    _natten_na2d = None
    _NATTEN_AVAILABLE = False


REAL_PET_PROMPT = (
    "Real PET preserves patient-specific metabolic details "
    "and heterogeneous lesion uptake."
)

PROXY_PET_PROMPT = (
    "Prototype-compensated PET provides a smooth metabolic "
    "prior with coarse lesion localization."
)

PET_PROMPTS = (
    REAL_PET_PROMPT,
    PROXY_PET_PROMPT,
)


@dataclass(frozen=True)
class ScaleAttentionConfig:
    """Attention configuration for one encoder scale."""

    channels: int
    internal_channels: int
    context_dilation: int


def _check_positive_odd(value: int, name: str) -> int:
    value = int(value)
    if value <= 0 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got {value}.")
    return value


def _zero_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)


def _rms(x: Tensor) -> Tensor:
    return x.detach().float().square().mean().sqrt()


class LayerNorm2d(nn.Module):
    """Per-pixel channel LayerNorm for a BCHW feature map."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"LayerNorm2d expected [B,{self.channels},H,W], "
                f"got {tuple(x.shape)}."
            )
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        return (
            normalized * self.weight.view(1, -1, 1, 1)
            + self.bias.view(1, -1, 1, 1)
        )


class ModalityAdapter(nn.Module):
    """Map one modality to the half-width attention space."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.normalization = LayerNorm2d(out_channels)
        self.activation = nn.GELU()

        nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.normalization(self.projection(x)))


class TextPETResidualModulation(nn.Module):
    """Scale-specific PET channel modulation from a fixed text embedding.

    The hidden width equals the scale's attention width instead of being a
    fixed constant:

        delta_T = tanh(W2 GELU(W1 LN(e_T)))
        PET_T   = PET + PET * delta_T

    The last layer starts from zero, so the module is initially an identity.
    """

    def __init__(
        self,
        channels: int,
        text_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if min(channels, text_dim, hidden_dim) <= 0:
            raise ValueError("channels, text_dim, and hidden_dim must be positive.")

        self.channels = int(channels)
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.text_to_channel = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, channels, bias=True),
        )
        _zero_module(self.text_to_channel[-1])

    def forward(
        self,
        pet_feature: Tensor,
        text_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if pet_feature.ndim != 4 or pet_feature.shape[1] != self.channels:
            raise ValueError(
                f"pet_feature must be [B,{self.channels},H,W], "
                f"got {tuple(pet_feature.shape)}."
            )
        if text_embedding.ndim != 2:
            raise ValueError("text_embedding must have shape [B,text_dim].")
        if text_embedding.shape != (pet_feature.shape[0], self.text_dim):
            raise ValueError(
                "The text batch and dimension must match PET; got "
                f"{tuple(text_embedding.shape)}."
            )

        parameter = self.text_to_channel[-1].weight
        text_embedding = text_embedding.to(
            device=pet_feature.device,
            dtype=parameter.dtype,
        )
        channel_delta = torch.tanh(self.text_to_channel(text_embedding))
        channel_delta = channel_delta.to(dtype=pet_feature.dtype).view(
            pet_feature.shape[0],
            self.channels,
            1,
            1,
        )
        return pet_feature + pet_feature * channel_delta, channel_delta


class SharedEvidentialHead(nn.Module):
    """Produce non-negative task evidence in one shared class frame.

    The same head is applied to adapted CT and PET features.  With evidence
    ``e >= 0`` and ``K`` classes:

        alpha = e + 1
        strength = sum(alpha)
        vacuity = K / strength
        belief = e / strength

    The final pointwise layer is zero-initialized for a stable, spatially
    uniform starting point.
    """

    def __init__(self, channels: int, num_classes: int = 2) -> None:
        super().__init__()
        if channels <= 0 or num_classes < 2:
            raise ValueError("channels must be positive and num_classes >= 2.")
        self.channels = int(channels)
        self.num_classes = int(num_classes)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )
        self.activation = nn.GELU()
        self.to_evidence = nn.Conv2d(
            channels,
            num_classes,
            kernel_size=1,
            bias=True,
        )

        nn.init.kaiming_normal_(self.depthwise.weight, nonlinearity="linear")
        nn.init.zeros_(self.depthwise.bias)
        _zero_module(self.to_evidence)

    def forward(self, feature: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        evidence = F.softplus(
            self.to_evidence(self.activation(self.depthwise(feature)))
        )
        strength = (evidence + 1.0).sum(dim=1, keepdim=True)
        belief = evidence / strength
        vacuity = float(self.num_classes) / strength
        return evidence, belief, vacuity.clamp(0.0, 1.0)


def evidential_conflict(ct_belief: Tensor, pet_belief: Tensor) -> Tensor:
    """Return cross-class CT/PET belief conflict in [0, 1]."""

    if ct_belief.shape != pet_belief.shape or ct_belief.ndim != 4:
        raise ValueError("CT and PET beliefs must have the same BCHW shape.")
    all_pairs = ct_belief.sum(dim=1, keepdim=True) * pet_belief.sum(
        dim=1,
        keepdim=True,
    )
    agreement = (ct_belief * pet_belief).sum(dim=1, keepdim=True)
    return (all_pairs - agreement).clamp(0.0, 1.0)


def _feature_to_heads(x: Tensor, num_heads: int) -> Tensor:
    """Convert [B,D,H,W] to NATTEN's [B,H,W,heads,head_dim]."""

    batch, channels, height, width = x.shape
    if channels % num_heads != 0:
        raise ValueError(
            f"channels ({channels}) must be divisible by heads ({num_heads})."
        )
    head_dim = channels // num_heads
    return (
        x.view(batch, num_heads, head_dim, height, width)
        .permute(0, 3, 4, 1, 2)
        .contiguous()
    )


def _heads_to_feature(x: Tensor) -> Tensor:
    """Convert [B,H,W,heads,head_dim] back to [B,D,H,W]."""

    if x.ndim != 5:
        raise ValueError(f"Head tensor must be 5-D, got {tuple(x.shape)}.")
    batch, height, width, num_heads, head_dim = x.shape
    return (
        x.permute(0, 3, 4, 1, 2)
        .contiguous()
        .view(batch, num_heads * head_dim, height, width)
    )


def _chunked_torch_na2d(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: int,
    dilation: int,
    chunk_rows: int,
) -> Tensor:
    """Memory-bounded reference implementation of 2-D cross-neighborhood attention.

    Q/K/V use the heads-last layout [B,H,W,heads,head_dim].  The function
    stores only the K^2 logits for ``chunk_rows`` query rows and never creates
    an unfolded K/V tensor.  It is intended for tests, not fast training.
    """

    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("Reference NA expects equal Q/K/V shapes.")
    if query.ndim != 5:
        raise ValueError("Reference NA expects [B,H,W,heads,head_dim].")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive.")

    batch, height, width, heads, head_dim = query.shape
    half_kernel = kernel_size // 2
    radius = half_kernel * dilation
    if kernel_size * dilation > min(height, width):
        raise ValueError(
            "kernel_size * dilation must not exceed either spatial side; "
            f"got kernel={kernel_size}, dilation={dilation}, "
            f"shape=({height},{width})."
        )

    key_cf = key.permute(0, 3, 4, 1, 2).contiguous()
    value_cf = value.permute(0, 3, 4, 1, 2).contiguous()
    padding = (radius, radius, radius, radius)
    key_padded = F.pad(key_cf, padding)
    value_padded = F.pad(value_cf, padding)
    scale = head_dim ** -0.5
    offsets = [
        (dy * dilation, dx * dilation)
        for dy in range(-half_kernel, half_kernel + 1)
        for dx in range(-half_kernel, half_kernel + 1)
    ]
    x_coordinates = torch.arange(width, device=query.device)
    output_chunks: List[Tensor] = []

    for row_start in range(0, height, chunk_rows):
        row_end = min(row_start + chunk_rows, height)
        rows = row_end - row_start
        query_chunk = query[:, row_start:row_end]
        y_coordinates = torch.arange(
            row_start,
            row_end,
            device=query.device,
        )
        logits_per_offset: List[Tensor] = []

        for y_offset, x_offset in offsets:
            y_start = radius + row_start + y_offset
            x_start = radius + x_offset
            key_slice = key_padded[
                :,
                :,
                :,
                y_start : y_start + rows,
                x_start : x_start + width,
            ].permute(0, 3, 4, 1, 2)
            logits = (query_chunk * key_slice).sum(dim=-1) * scale
            valid_y = (
                (y_coordinates + y_offset >= 0)
                & (y_coordinates + y_offset < height)
            )
            valid_x = (
                (x_coordinates + x_offset >= 0)
                & (x_coordinates + x_offset < width)
            )
            valid = valid_y[:, None] & valid_x[None, :]
            logits = logits.masked_fill(
                ~valid.view(1, rows, width, 1),
                torch.finfo(logits.dtype).min,
            )
            logits_per_offset.append(logits)

        logits = torch.stack(logits_per_offset, dim=-1)
        weights = torch.softmax(logits.float(), dim=-1).to(query.dtype)
        output_chunk = torch.zeros_like(query_chunk)

        # Re-read V slices instead of materializing all K^2 value neighborhoods.
        for offset_index, (y_offset, x_offset) in enumerate(offsets):
            y_start = radius + row_start + y_offset
            x_start = radius + x_offset
            value_slice = value_padded[
                :,
                :,
                :,
                y_start : y_start + rows,
                x_start : x_start + width,
            ].permute(0, 3, 4, 1, 2)
            output_chunk = output_chunk + (
                weights[..., offset_index].unsqueeze(-1) * value_slice
            )
        output_chunks.append(output_chunk)

    return torch.cat(output_chunks, dim=1)


class ScaleAwareDilatedNeighborhoodCrossAttention(nn.Module):
    """Shared bidirectional full-resolution CT/PET attention core.

    Half of the heads use contiguous 7x7 neighborhoods and the other half use
    the scale's dilated 7x7 neighborhood.  Dilation changes sampling positions
    but neither parameters nor the number of keys per query.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        local_heads: int = 2,
        kernel_size: int = 7,
        context_dilation: int = 2,
        attention_backend: str = "auto",
        natten_backend: Optional[str] = None,
        torch_chunk_rows: int = 4,
    ) -> None:
        super().__init__()
        kernel_size = _check_positive_odd(kernel_size, "kernel_size")
        if channels <= 0 or channels % num_heads != 0:
            raise ValueError("channels must be positive and divisible by num_heads.")
        if not 0 < local_heads < num_heads:
            raise ValueError("local_heads must be between 1 and num_heads - 1.")
        if context_dilation < 1:
            raise ValueError("context_dilation must be at least 1.")
        if attention_backend not in {"auto", "natten", "torch"}:
            raise ValueError("attention_backend must be auto, natten, or torch.")

        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.local_heads = int(local_heads)
        self.context_heads = self.num_heads - self.local_heads
        self.kernel_size = kernel_size
        self.context_dilation = int(context_dilation)
        self.attention_backend = attention_backend
        self.natten_backend = natten_backend
        self.torch_chunk_rows = int(torch_chunk_rows)

        # One projection set is deliberately reused in both directions.
        self.query_projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.key_projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.value_projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.output_projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (
            self.query_projection,
            self.key_projection,
            self.value_projection,
            self.output_projection,
        ):
            nn.init.xavier_uniform_(module.weight)

    def _use_natten(self, query: Tensor) -> bool:
        if self.attention_backend == "torch":
            return False
        if self.attention_backend == "natten":
            if not _NATTEN_AVAILABLE:
                raise RuntimeError(
                    "attention_backend='natten' was requested, but NATTEN "
                    "could not be imported. Install a NATTEN build matching "
                    "the local PyTorch/CUDA versions."
                )
            return True
        return bool(_NATTEN_AVAILABLE and query.is_cuda)

    def _attention_group(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dilation: int,
    ) -> Tensor:
        if self._use_natten(query):
            kwargs = {}
            if self.natten_backend is not None:
                kwargs["backend"] = self.natten_backend
            return _natten_na2d(
                query,
                key,
                value,
                kernel_size=self.kernel_size,
                stride=1,
                dilation=dilation,
                is_causal=False,
                **kwargs,
            )
        return _chunked_torch_na2d(
            query=query,
            key=key,
            value=value,
            kernel_size=self.kernel_size,
            dilation=dilation,
            chunk_rows=self.torch_chunk_rows,
        )

    def forward(
        self,
        query_feature: Tensor,
        source_feature: Tensor,
        source_reliability: Optional[Tensor] = None,
    ) -> Tensor:
        if query_feature.shape != source_feature.shape:
            raise ValueError(
                "query_feature and source_feature must have equal BCHW shapes."
            )
        if query_feature.ndim != 4 or query_feature.shape[1] != self.channels:
            raise ValueError(
                f"Attention expected [B,{self.channels},H,W], "
                f"got {tuple(query_feature.shape)}."
            )
        height, width = query_feature.shape[-2:]
        if self.kernel_size * self.context_dilation > min(height, width):
            raise ValueError(
                "The configured dilated neighborhood does not fit the feature "
                f"map: kernel={self.kernel_size}, "
                f"dilation={self.context_dilation}, shape=({height},{width})."
            )

        query = _feature_to_heads(
            self.query_projection(query_feature),
            self.num_heads,
        )
        key = _feature_to_heads(
            self.key_projection(source_feature),
            self.num_heads,
        )
        value_feature = self.value_projection(source_feature)
        if source_reliability is not None:
            expected_shape = (
                source_feature.shape[0],
                1,
                height,
                width,
            )
            if source_reliability.shape != expected_shape:
                raise ValueError(
                    "source_reliability must be [B,1,H,W], got "
                    f"{tuple(source_reliability.shape)}."
                )
            value_feature = value_feature * source_reliability.to(
                device=value_feature.device,
                dtype=value_feature.dtype,
            )
        value = _feature_to_heads(value_feature, self.num_heads)

        local_slice = slice(0, self.local_heads)
        context_slice = slice(self.local_heads, self.num_heads)
        local_output = self._attention_group(
            query[..., local_slice, :].contiguous(),
            key[..., local_slice, :].contiguous(),
            value[..., local_slice, :].contiguous(),
            dilation=1,
        )
        context_output = self._attention_group(
            query[..., context_slice, :].contiguous(),
            key[..., context_slice, :].contiguous(),
            value[..., context_slice, :].contiguous(),
            dilation=self.context_dilation,
        )
        output = torch.cat((local_output, context_output), dim=3)
        return self.output_projection(_heads_to_feature(output))


class EvidenceGuidedSDNCAScale(nn.Module):
    """Complete evidence/text/attention/fusion block for one encoder scale."""

    def __init__(
        self,
        channels: int,
        internal_channels: int,
        text_dim: int,
        context_dilation: int,
        num_heads: int = 4,
        local_heads: int = 2,
        kernel_size: int = 7,
        num_evidence_classes: int = 2,
        attention_backend: str = "auto",
        natten_backend: Optional[str] = None,
        torch_chunk_rows: int = 4,
    ) -> None:
        super().__init__()
        if internal_channels % num_heads != 0:
            raise ValueError(
                f"internal_channels={internal_channels} must be divisible by "
                f"num_heads={num_heads}."
            )
        self.channels = int(channels)
        self.internal_channels = int(internal_channels)
        self.context_dilation = int(context_dilation)

        self.text_modulator = TextPETResidualModulation(
            channels=channels,
            text_dim=text_dim,
            hidden_dim=internal_channels,
        )
        self.ct_adapter = ModalityAdapter(channels, internal_channels)
        self.pet_adapter = ModalityAdapter(channels, internal_channels)
        self.evidence_head = SharedEvidentialHead(
            channels=internal_channels,
            num_classes=num_evidence_classes,
        )
        self.cross_attention = ScaleAwareDilatedNeighborhoodCrossAttention(
            channels=internal_channels,
            num_heads=num_heads,
            local_heads=local_heads,
            kernel_size=kernel_size,
            context_dilation=context_dilation,
            attention_backend=attention_backend,
            natten_backend=natten_backend,
            torch_chunk_rows=torch_chunk_rows,
        )
        self.pet_delta_projection = nn.Conv2d(
            internal_channels,
            channels,
            kernel_size=1,
            bias=True,
        )
        self.ct_delta_projection = nn.Conv2d(
            internal_channels,
            channels,
            kernel_size=1,
            bias=True,
        )

        # Both cross-modal corrections start as exact residual identities.
        _zero_module(self.pet_delta_projection)
        _zero_module(self.ct_delta_projection)

    def _validate_inputs(self, ct_feature: Tensor, pet_feature: Tensor) -> None:
        if ct_feature.shape != pet_feature.shape:
            raise ValueError("CT and PET features must have identical shapes.")
        if ct_feature.ndim != 4 or ct_feature.shape[1] != self.channels:
            raise ValueError(
                f"Scale block expected [B,{self.channels},H,W], got "
                f"{tuple(ct_feature.shape)}."
            )
        if ct_feature.device != pet_feature.device:
            raise ValueError("CT and PET must be on the same device.")
        if ct_feature.dtype != pet_feature.dtype:
            raise ValueError("CT and PET must have the same dtype.")

    def forward(
        self,
        ct_feature: Tensor,
        pet_feature: Tensor,
        selected_text: Tensor,
        return_diagnostics: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        self._validate_inputs(ct_feature, pet_feature)

        # 1) State text acts only on PET and starts from an identity mapping.
        pet_text, text_channel_delta = self.text_modulator(
            pet_feature,
            selected_text,
        )

        # 2) One shared evidential class frame produces a single PET trust map.
        ct_internal = self.ct_adapter(ct_feature)
        pet_internal = self.pet_adapter(pet_text)
        _, ct_belief, _ = self.evidence_head(ct_internal)
        _, pet_belief, pet_vacuity = self.evidence_head(pet_internal)
        conflict = evidential_conflict(ct_belief, pet_belief)
        pet_trust = ((1.0 - pet_vacuity) * (1.0 - conflict)).clamp(0.0, 1.0)

        # 3) Untrusted PET receives a stronger residual structural correction.
        pet_delta_internal = self.cross_attention(
            query_feature=pet_internal,
            source_feature=ct_internal,
            source_reliability=None,
        )
        pet_delta = self.pet_delta_projection(pet_delta_internal)
        pet_corrected = pet_text + (1.0 - pet_trust) * pet_delta

        # 4) CT retrieves only trusted values from the corrected PET feature.
        pet_corrected_internal = self.pet_adapter(pet_corrected)
        ct_delta_internal = self.cross_attention(
            query_feature=ct_internal,
            source_feature=pet_corrected_internal,
            source_reliability=pet_trust,
        )
        ct_delta = self.ct_delta_projection(ct_delta_internal)
        ct_enhanced = ct_feature + ct_delta

        # 5) CT remains the anchor; PET contribution grows monotonically with
        # trust.  The denominator keeps the feature scale bounded.
        fused_feature = (
            ct_enhanced + pet_trust * pet_corrected
        ) / (1.0 + pet_trust)

        if not return_diagnostics:
            return fused_feature

        diagnostics = {
            "pet_vacuity_mean": pet_vacuity.detach().float().mean(),
            "ct_pet_conflict_mean": conflict.detach().float().mean(),
            "pet_trust_mean": pet_trust.detach().float().mean(),
            "pet_trust_min": pet_trust.detach().float().min(),
            "pet_trust_max": pet_trust.detach().float().max(),
            "text_channel_delta_abs_mean": (
                text_channel_delta.detach().float().abs().mean()
            ),
            "ct_rms": _rms(ct_feature),
            "pet_before_text_rms": _rms(pet_feature),
            "pet_after_text_rms": _rms(pet_text),
            "pet_corrected_rms": _rms(pet_corrected),
            "fused_rms": _rms(fused_feature),
            "context_dilation": torch.tensor(
                self.context_dilation,
                device=ct_feature.device,
            ),
        }
        return fused_feature, diagnostics


class MultiScaleEvidenceGuidedSDNCA(nn.Module):
    """Four-scale CT/PET fusion module for Full, Missing, and Auto paths.

    Args:
        text_embeddings: Frozen tensor [2,text_dim], ordered as real then proxy.
            Text must be encoded once outside this module.  The encoder is not
            retained or trained here.
        channels: Encoder feature widths.  Project defaults are MiT-B1 widths.
        context_dilations: Dilated-head spacing for the four scales.
        internal_ratio: Attention width relative to each scale width.  The
            default 0.5 preserves the agreed half-channel design.
    """

    def __init__(
        self,
        text_embeddings: Tensor,
        channels: Sequence[int] = (64, 128, 320, 512),
        context_dilations: Sequence[int] = (2, 3, 4, 2),
        internal_ratio: float = 0.5,
        num_heads: int = 4,
        local_heads: int = 2,
        kernel_size: int = 7,
        num_evidence_classes: int = 2,
        attention_backend: str = "auto",
        natten_backend: Optional[str] = None,
        torch_chunk_rows: int = 4,
    ) -> None:
        super().__init__()
        if text_embeddings.ndim != 2 or text_embeddings.shape[0] != 2:
            raise ValueError("text_embeddings must have shape [2,text_dim].")
        if len(channels) == 0 or len(channels) != len(context_dilations):
            raise ValueError("channels and context_dilations must have equal length.")
        if not 0.0 < internal_ratio <= 1.0:
            raise ValueError("internal_ratio must be in (0,1].")

        normalized_text = F.normalize(
            text_embeddings.detach().float(),
            dim=-1,
        )
        self.register_buffer("text_embeddings", normalized_text)
        self.channels = tuple(int(channel) for channel in channels)
        self.context_dilations = tuple(
            int(dilation) for dilation in context_dilations
        )

        scale_configs: List[ScaleAttentionConfig] = []
        scale_modules: List[nn.Module] = []
        for channel, dilation in zip(self.channels, self.context_dilations):
            internal = int(round(channel * internal_ratio))
            # Round up only as much as needed for equal-sized attention heads.
            internal = int(math.ceil(internal / num_heads) * num_heads)
            config = ScaleAttentionConfig(
                channels=channel,
                internal_channels=internal,
                context_dilation=dilation,
            )
            scale_configs.append(config)
            scale_modules.append(
                EvidenceGuidedSDNCAScale(
                    channels=channel,
                    internal_channels=internal,
                    text_dim=normalized_text.shape[1],
                    context_dilation=dilation,
                    num_heads=num_heads,
                    local_heads=local_heads,
                    kernel_size=kernel_size,
                    num_evidence_classes=num_evidence_classes,
                    attention_backend=attention_backend,
                    natten_backend=natten_backend,
                    torch_chunk_rows=torch_chunk_rows,
                )
            )
        self.scale_configs = tuple(scale_configs)
        self.scales = nn.ModuleList(scale_modules)

    @staticmethod
    def _state_ids(
        mode: str,
        batch_size: int,
        device: torch.device,
        pet_available: Optional[Tensor],
    ) -> Tensor:
        mode = str(mode).lower()
        if mode == "full":
            return torch.zeros(batch_size, device=device, dtype=torch.long)
        if mode == "missing":
            return torch.ones(batch_size, device=device, dtype=torch.long)
        if mode != "auto":
            raise ValueError("mode must be 'full', 'missing', or 'auto'.")
        if pet_available is None:
            raise ValueError("auto mode requires pet_available.")

        availability = pet_available.to(device=device).long().reshape(-1)
        if availability.numel() != batch_size:
            raise ValueError("pet_available must contain one value per sample.")
        if not torch.all((availability == 0) | (availability == 1)):
            raise ValueError("pet_available values must be 0 or 1.")
        # available=1 -> real text id 0; available=0 -> proxy text id 1.
        return 1 - availability

    def _validate_feature_lists(
        self,
        ct_features: Sequence[Tensor],
        pet_features: Sequence[Tensor],
    ) -> None:
        if len(ct_features) != len(self.scales) or len(pet_features) != len(
            self.scales
        ):
            raise ValueError(
                f"Expected {len(self.scales)} CT/PET scales, got "
                f"{len(ct_features)} and {len(pet_features)}."
            )
        batch_size = ct_features[0].shape[0]
        for index, (ct_feature, pet_feature, channel) in enumerate(
            zip(ct_features, pet_features, self.channels)
        ):
            if ct_feature.shape != pet_feature.shape:
                raise ValueError(f"Scale {index}: CT/PET shapes do not match.")
            if ct_feature.ndim != 4 or ct_feature.shape[1] != channel:
                raise ValueError(
                    f"Scale {index}: expected [B,{channel},H,W], got "
                    f"{tuple(ct_feature.shape)}."
                )
            if ct_feature.shape[0] != batch_size:
                raise ValueError("All scales must have the same batch size.")

    def forward(
        self,
        ct_features: Sequence[Tensor],
        pet_features: Sequence[Tensor],
        mode: str = "full",
        pet_available: Optional[Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Union[
        List[Tensor],
        Tuple[List[Tensor], Dict[str, Union[Tensor, List[Dict[str, Tensor]]]]],
    ]:
        self._validate_feature_lists(ct_features, pet_features)
        batch_size = ct_features[0].shape[0]
        device = ct_features[0].device
        state_ids = self._state_ids(
            mode=mode,
            batch_size=batch_size,
            device=device,
            pet_available=pet_available,
        )
        selected_text = self.text_embeddings.index_select(0, state_ids)

        fused_features: List[Tensor] = []
        scale_diagnostics: List[Dict[str, Tensor]] = []
        for scale, ct_feature, pet_feature in zip(
            self.scales,
            ct_features,
            pet_features,
        ):
            result = scale(
                ct_feature=ct_feature,
                pet_feature=pet_feature,
                selected_text=selected_text,
                return_diagnostics=return_diagnostics,
            )
            if return_diagnostics:
                fused_feature, diagnostics = result
                scale_diagnostics.append(diagnostics)
            else:
                fused_feature = result
            fused_features.append(fused_feature)

        if not return_diagnostics:
            return fused_features
        return fused_features, {
            "pet_state_ids": state_ids.detach(),
            "scales": scale_diagnostics,
        }


def count_parameters(module: nn.Module) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def build_dummy_text_embeddings(text_dim: int = 512) -> Tensor:
    """Deterministic stand-in used only by this file's self-test."""

    positions = torch.arange(text_dim, dtype=torch.float32)
    real = torch.sin(positions / 31.0)
    proxy = torch.cos(positions / 29.0)
    return torch.stack((real, proxy), dim=0)


def _missing_files(path: str, names: Sequence[str]) -> List[str]:
    return [name for name in names if not os.path.isfile(os.path.join(path, name))]


def _has_hf_model_files(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json")) and (
        os.path.isfile(os.path.join(path, "pytorch_model.bin"))
        or os.path.isfile(os.path.join(path, "model.safetensors"))
    )


def _has_hf_tokenizer_files(path: str) -> bool:
    has_vocab = os.path.isfile(os.path.join(path, "vocab.txt")) or os.path.isfile(
        os.path.join(path, "tokenizer.json")
    )
    has_config = os.path.isfile(os.path.join(path, "tokenizer_config.json"))
    return has_vocab and has_config


def _raise_local_load_error(
    biomedclip_model_path: str,
    biomedbert_text_tower_path: str,
    missing: Mapping[str, Sequence[str]],
    attempted: Sequence[str],
) -> None:
    lines = [
        "Failed to load local PET text embeddings offline.",
        f"Checked biomedclip_model_path={biomedclip_model_path}",
        f"Checked biomedbert_text_tower_path={biomedbert_text_tower_path}",
    ]
    for label, files in missing.items():
        if files:
            lines.append(f"Missing under {label}: {', '.join(files)}")
    lines.append("Attempted load methods: " + "; ".join(attempted))
    lines.append(
        "No online/Hugging Face hub fallback is allowed. "
        "Provide complete local tokenizer + text-model files."
    )
    raise FileNotFoundError("\n".join(lines))


@torch.no_grad()
def load_local_pet_text_embeddings(
    biomedclip_model_path: str = BIOMEDCLIP_MODEL_PATH,
    biomedbert_text_tower_path: str = BIOMEDBERT_TEXT_TOWER_PATH,
) -> Tensor:
    """Encode the two fixed PET prompts once with a local frozen text model.

    Returns a CPU tensor of shape ``[2, text_dim]`` ordered as
    ``[real/full, proxy/missing]``.  The text encoder is deleted before return.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Install transformers only for offline prompt encoding: "
            "pip install transformers"
        ) from error

    attempted: List[str] = []
    missing = {
        biomedbert_text_tower_path: [],
        biomedclip_model_path: [],
    }

    if not os.path.isdir(biomedbert_text_tower_path):
        missing[biomedbert_text_tower_path] = ["<directory missing>"]
    if not os.path.isdir(biomedclip_model_path):
        missing[biomedclip_model_path] = ["<directory missing>"]

    tokenizer_path: Optional[str] = None
    model_path: Optional[str] = None
    source = None

    bert_has_model = _has_hf_model_files(biomedbert_text_tower_path)
    bert_has_tok = _has_hf_tokenizer_files(biomedbert_text_tower_path)
    clip_has_tok = _has_hf_tokenizer_files(biomedclip_model_path)

    if bert_has_model and bert_has_tok:
        tokenizer_path = biomedbert_text_tower_path
        model_path = biomedbert_text_tower_path
        source = "hf_biomedbert_text_tower"
        attempted.append(
            "HF AutoTokenizer+AutoModel from biomedbert_text_tower (preferred)"
        )
    elif bert_has_model and clip_has_tok:
        tokenizer_path = biomedclip_model_path
        model_path = biomedbert_text_tower_path
        source = "hf_biomedbert_model_biomedclip_tokenizer"
        attempted.append(
            "HF AutoTokenizer from biomedclip_model + AutoModel from biomedbert_text_tower"
        )
    else:
        if not bert_has_model:
            miss_model = _missing_files(
                biomedbert_text_tower_path,
                ["config.json", "pytorch_model.bin"],
            )
            if not os.path.isfile(
                os.path.join(biomedbert_text_tower_path, "model.safetensors")
            ) and "pytorch_model.bin" in miss_model:
                miss_model.append("model.safetensors")
            missing[biomedbert_text_tower_path].extend(miss_model)
        if not bert_has_tok and not clip_has_tok:
            for path in (biomedbert_text_tower_path, biomedclip_model_path):
                miss = []
                if not os.path.isfile(os.path.join(path, "tokenizer_config.json")):
                    miss.append("tokenizer_config.json")
                if not (
                    os.path.isfile(os.path.join(path, "vocab.txt"))
                    or os.path.isfile(os.path.join(path, "tokenizer.json"))
                ):
                    miss.append("vocab.txt|tokenizer.json")
                missing[path].extend(miss)
        attempted.extend(
            [
                "HF AutoTokenizer+AutoModel from biomedbert_text_tower",
                "HF split tokenizer(biomedclip)+model(biomedbert)",
                "OpenCLIP BioMedCLIP local load (not selected: HF files incomplete)",
            ]
        )
        _raise_local_load_error(
            biomedclip_model_path,
            biomedbert_text_tower_path,
            missing,
            attempted,
        )

    assert tokenizer_path is not None and model_path is not None

    tokenizer = None
    text_model = None
    embeddings: Optional[Tensor] = None
    used_cuda = False
    try:
        attempted.append(
            f"AutoTokenizer.from_pretrained({tokenizer_path}, local_files_only=True)"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
        )
        attempted.append(
            f"AutoModel.from_pretrained({model_path}, local_files_only=True)"
        )
        text_model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
        )
        text_model.eval()
        text_model.requires_grad_(False)
        # Keep the text encoder on CPU so training GPUs stay free.
        text_model = text_model.to("cpu")

        tokens = tokenizer(
            list(PET_PROMPTS),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        tokens = {name: value.to("cpu") for name, value in tokens.items()}

        with torch.no_grad():
            outputs = text_model(**tokens)
            if (
                hasattr(outputs, "text_embeds")
                and outputs.text_embeds is not None
            ):
                embeddings = outputs.text_embeds
            else:
                # Prefer verified attention-mask mean pooling over unverified poolers.
                hidden = outputs.last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(
                    1.0
                )

            embeddings = F.normalize(embeddings.float(), dim=-1).cpu()

        assert embeddings.ndim == 2
        assert embeddings.shape[0] == 2
        assert torch.isfinite(embeddings).all()
        assert not torch.allclose(embeddings[0], embeddings[1])
    except (AssertionError, FileNotFoundError, RuntimeError, ValueError):
        raise
    except Exception as error:
        attempted.append(f"failed with {type(error).__name__}: {error}")
        _raise_local_load_error(
            biomedclip_model_path,
            biomedbert_text_tower_path,
            missing,
            attempted,
        )
    finally:
        del tokenizer
        del text_model
        gc.collect()
        if used_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert embeddings is not None
    print("[EDV] offline=True")
    print(f"[EDV] source={source}")
    print(f"[EDV] embedding_shape={tuple(embeddings.shape)}")
    print("[EDV] state_order=['real', 'proxy']")
    print("[EDV] text_encoder_retained=False")
    return embeddings


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def run_self_test(args: argparse.Namespace) -> None:
    """Run shape, state mapping, finite-value, gradient, and budget checks."""

    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    model = MultiScaleEvidenceGuidedSDNCA(
        text_embeddings=build_dummy_text_embeddings(args.text_dim),
        attention_backend=args.attention_backend,
        torch_chunk_rows=args.torch_chunk_rows,
    ).to(device)
    model.train()

    total_parameters, trainable_parameters = count_parameters(model)
    if total_parameters >= 5_000_000:
        raise RuntimeError(
            f"The standalone fusion module exceeds 5M parameters: "
            f"{total_parameters:,}."
        )

    # Project shapes for a 512x512 input.  Use --batch-size 16 on the target
    # GPU for the required AMP benchmark.
    shapes = (
        (args.batch_size, 64, 128, 128),
        (args.batch_size, 128, 64, 64),
        (args.batch_size, 320, 32, 32),
        (args.batch_size, 512, 16, 16),
    )
    ct_features = [
        torch.randn(shape, device=device, requires_grad=True) for shape in shapes
    ]
    pet_features = [
        torch.randn(shape, device=device, requires_grad=True) for shape in shapes
    ]
    availability = torch.arange(args.batch_size, device=device) % 2

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=use_amp,
    ):
        fused_features, diagnostics = model(
            ct_features,
            pet_features,
            mode="auto",
            pet_available=availability,
            return_diagnostics=True,
        )
        loss = sum(feature.square().mean() for feature in fused_features)

    for fused, expected in zip(fused_features, shapes):
        if fused.shape != expected or not torch.isfinite(fused).all():
            raise RuntimeError("Output shape or finite-value check failed.")
    expected_states = 1 - availability
    torch.testing.assert_close(diagnostics["pet_state_ids"], expected_states)
    loss.backward()
    if any(feature.grad is None for feature in ct_features + pet_features):
        raise RuntimeError("Backward did not reach every CT/PET scale.")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    print("Evidence-guided SDNCA standalone test: PASS")
    print(f"device: {device}")
    print(f"NATTEN import available: {_NATTEN_AVAILABLE}")
    print(f"attention backend request: {args.attention_backend}")
    print(f"AMP enabled: {use_amp}")
    print(f"parameters: {total_parameters:,} total")
    print(f"trainable parameters: {trainable_parameters:,}")
    print(f"forward + backward: {elapsed:.3f} s")
    if device.type == "cuda":
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"peak allocated CUDA memory: {peak_mib:.1f} MiB")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone full-resolution evidence-guided CT/PET SDNCA."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--text-dim", type=int, default=512)
    parser.add_argument(
        "--attention-backend",
        choices=("auto", "natten", "torch"),
        default="auto",
    )
    parser.add_argument("--torch-chunk-rows", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser


__all__ = [
    "PET_PROMPTS",
    "REAL_PET_PROMPT",
    "PROXY_PET_PROMPT",
    "BIOMEDCLIP_MODEL_PATH",
    "BIOMEDBERT_TEXT_TOWER_PATH",
    "TextPETResidualModulation",
    "SharedEvidentialHead",
    "ScaleAwareDilatedNeighborhoodCrossAttention",
    "EvidenceGuidedSDNCAScale",
    "MultiScaleEvidenceGuidedSDNCA",
    "count_parameters",
    "load_local_pet_text_embeddings",
]


if __name__ == "__main__":
    run_self_test(build_argument_parser().parse_args())