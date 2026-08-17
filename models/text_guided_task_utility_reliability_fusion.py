"""Standalone text-guided, task-utility reliability fusion for CT/PET.

This file implements the complete four-scale fusion module discussed for a
paired CT/PET segmentation model.  It is intentionally independent from the
encoders, CPPI prototype memory, PET calibration, and decoder so it can be
copied into an existing project without modifying those components.

Pipeline
--------
1. Encode two fixed PET-state prompts once with a local frozen BioMedCLIP /
   BioMedBERT text tower (optional and switchable).
2. Convert the selected text plus four CT/PET scale descriptors into a short
   sequence of length 9 (or 8 when text is disabled).
3. Use two official ``mamba_ssm.Mamba`` blocks in forward/reverse directions
   to estimate one PET reliability value per scale.
4. Use each reliability in one closed-loop dual path:

       robust_s  = CT_s + r_s * PET_s
       synergy_s = CrossAttention(CT_s, r_s * PET_s)
       output_s  = robust_s + r_s * synergy_s

5. Optionally calibrate reliability with a single Task-Utility Reliability
   Ranking (TURR) loss.  TURR is evaluated sparsely and uses one training-only,
   no-gradient counterfactual decoder pass.  It never aligns real PET with
   compensated PET and never introduces a second encoder or decoder.

Expected project interface
--------------------------
``ct_features`` and ``pet_features`` are equally-shaped four-element lists:

    [B, C1, H1, W1], ..., [B, C4, H4, W4]

State mapping:

    0 = real/full PET
    1 = prototype-compensated/missing PET

In ``auto`` mode, ``pet_available=1`` selects state 0 and
``pet_available=0`` selects state 1 independently for every sample.

Dependencies
------------
Required for the fusion module:

    torch
    mamba-ssm

Required only when the two local prompts must be encoded:

    transformers

The Mamba implementation is imported directly from the official
``mamba_ssm`` package.  No hand-written Mamba/SSM approximation is included.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import os
import time
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence
from typing import Tuple, Union

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

REAL_PET_PROMPT = (
    "Real PET provides metabolic uptake "
    "and patient-specific lesion details."
)

PROXY_PET_PROMPT = (
    "Compensated PET provides a metabolic prior "
    "and coarse lesion location."
)

PET_PROMPTS = (
    REAL_PET_PROMPT,
    PROXY_PET_PROMPT,
)


def _require_mamba() -> type[nn.Module]:
    """Import the official Mamba block lazily.

    Lazy import lets code paths with ``use_text=False`` still import this file
    before the CUDA extension environment is fully configured.  Constructing
    the fusion module always requires the real ``mamba_ssm`` package.
    """

    try:
        from mamba_ssm import Mamba
    except ImportError as error:
        raise RuntimeError(
            "The official mamba_ssm package is required. Install a build "
            "compatible with the project's PyTorch/CUDA versions, for example "
            "`pip install mamba-ssm[causal-conv1d] --no-build-isolation`."
        ) from error
    return Mamba


def _zero_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)


def _detach_tree(value: Any) -> Any:
    """Detach tensors inside common nested prediction containers."""

    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key: _detach_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    return value


def _count_parameters(module: nn.Module) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def _has_hf_model_files(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json")) and (
        os.path.isfile(os.path.join(path, "pytorch_model.bin"))
        or os.path.isfile(os.path.join(path, "model.safetensors"))
        or os.path.isfile(os.path.join(path, "pytorch_model.bin.index.json"))
        or os.path.isfile(os.path.join(path, "model.safetensors.index.json"))
    )


def _has_hf_tokenizer_files(path: str) -> bool:
    has_vocabulary = (
        os.path.isfile(os.path.join(path, "vocab.txt"))
        or os.path.isfile(os.path.join(path, "tokenizer.json"))
    )
    has_config = os.path.isfile(os.path.join(path, "tokenizer_config.json"))
    return has_vocabulary and has_config


@torch.no_grad()
def load_local_pet_text_embeddings(
    biomedclip_model_path: str = BIOMEDCLIP_MODEL_PATH,
    biomedbert_text_tower_path: str = BIOMEDBERT_TEXT_TOWER_PATH,
    prompts: Sequence[str] = PET_PROMPTS,
) -> Tensor:
    """Encode fixed PET prompts once using local, frozen HF model files.

    The function mirrors the project's existing offline loading route.  The
    text model is loaded on CPU with ``local_files_only=True`` and deleted
    before return.  No model-hub/network fallback is permitted.

    Returns:
        Normalized CPU float tensor ``[2, text_dim]`` ordered as
        ``[real/full, proxy/missing]``.
    """

    if len(prompts) != 2:
        raise ValueError(f"Exactly two PET prompts are required, got {len(prompts)}.")

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Offline prompt encoding requires transformers: "
            "`pip install transformers`."
        ) from error

    if not os.path.isdir(biomedbert_text_tower_path):
        raise FileNotFoundError(
            "Local BioMedBERT text tower directory was not found: "
            f"{biomedbert_text_tower_path}"
        )
    if not os.path.isdir(biomedclip_model_path):
        raise FileNotFoundError(
            "Local BioMedCLIP directory was not found: "
            f"{biomedclip_model_path}"
        )

    if not _has_hf_model_files(biomedbert_text_tower_path):
        raise FileNotFoundError(
            "The BioMedBERT text tower must contain config.json and either "
            "pytorch_model.bin or model.safetensors. Checked: "
            f"{biomedbert_text_tower_path}"
        )

    if _has_hf_tokenizer_files(biomedbert_text_tower_path):
        tokenizer_path = biomedbert_text_tower_path
        source = "biomedbert_text_tower"
    elif _has_hf_tokenizer_files(biomedclip_model_path):
        tokenizer_path = biomedclip_model_path
        source = "biomedclip_tokenizer+biomedbert_text_tower"
    else:
        raise FileNotFoundError(
            "No complete local tokenizer was found. Expected "
            "tokenizer_config.json plus vocab.txt or tokenizer.json under "
            f"{biomedbert_text_tower_path} or {biomedclip_model_path}."
        )

    tokenizer = None
    text_model = None
    embeddings: Optional[Tensor] = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
        )
        text_model = AutoModel.from_pretrained(
            biomedbert_text_tower_path,
            local_files_only=True,
        )
        text_model.eval()
        text_model.requires_grad_(False)
        text_model.to("cpu")

        tokens = tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        tokens = {name: value.to("cpu") for name, value in tokens.items()}
        outputs = text_model(**tokens)

        if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            embeddings = outputs.text_embeds
        else:
            hidden = outputs.last_hidden_state
            attention_mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            embeddings = (hidden * attention_mask).sum(dim=1)
            embeddings = embeddings / attention_mask.sum(dim=1).clamp_min(1.0)

        embeddings = F.normalize(embeddings.detach().float(), dim=-1).cpu()
        if embeddings.ndim != 2 or embeddings.shape[0] != 2:
            raise RuntimeError(
                "Unexpected local text embedding shape: "
                f"{tuple(embeddings.shape)}."
            )
        if not torch.isfinite(embeddings).all():
            raise RuntimeError("Local text embeddings contain NaN or Inf.")
        if torch.allclose(embeddings[0], embeddings[1]):
            raise RuntimeError("Real and compensated PET embeddings are identical.")
    finally:
        del tokenizer
        del text_model
        gc.collect()

    assert embeddings is not None
    print("[TG-TURF text] offline=True")
    print(f"[TG-TURF text] source={source}")
    print(f"[TG-TURF text] embedding_shape={tuple(embeddings.shape)}")
    print("[TG-TURF text] state_order=['real', 'proxy']")
    print("[TG-TURF text] encoder_retained=False")
    return embeddings


class LayerNorm2d(nn.Module):
    """Per-spatial-location channel LayerNorm for BCHW tensors."""

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


class GlobalScaleToken(nn.Module):
    """Compress one BCHW scale into one normalized sequence token."""

    def __init__(self, channels: int, d_model: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.projection = nn.Linear(channels, d_model, bias=True)
        self.normalization = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, feature: Tensor) -> Tensor:
        if feature.ndim != 4 or feature.shape[1] != self.channels:
            raise ValueError(
                f"Expected [B,{self.channels},H,W], got {tuple(feature.shape)}."
            )
        descriptor = feature.float().mean(dim=(2, 3))
        parameter_dtype = self.projection.weight.dtype
        descriptor = descriptor.to(dtype=parameter_dtype)
        return self.normalization(self.projection(descriptor))


class BidirectionalMambaContext(nn.Module):
    """A short-sequence bidirectional context block using official Mamba."""

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        if min(d_model, d_state, d_conv, expand) <= 0:
            raise ValueError("All Mamba dimensions must be positive.")

        Mamba = _require_mamba()
        self.input_norm = nn.LayerNorm(d_model)
        self.forward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.backward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                f"Mamba tokens must be [B,L,D], got {tuple(tokens.shape)}."
            )
        normalized = self.input_norm(tokens)
        forward_context = tokens + self.forward_mamba(normalized)

        reversed_tokens = torch.flip(tokens, dims=(1,))
        reversed_normalized = self.input_norm(reversed_tokens)
        backward_context = reversed_tokens + self.backward_mamba(
            reversed_normalized
        )
        backward_context = torch.flip(backward_context, dims=(1,))
        return self.output_norm(0.5 * (forward_context + backward_context))


def _pad_to_window(x: Tensor, window_size: int) -> Tuple[Tensor, int, int]:
    _, _, height, width = x.shape
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, height, width


def _partition_windows(x: Tensor, window_size: int) -> Tensor:
    """BCHW -> [B*num_windows, window_tokens, C]."""

    batch, channels, height, width = x.shape
    if height % window_size != 0 or width % window_size != 0:
        raise ValueError("Window partition requires divisible spatial dimensions.")
    x = x.view(
        batch,
        channels,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
    )
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(-1, window_size * window_size, channels)


def _reverse_windows(
    windows: Tensor,
    window_size: int,
    batch_size: int,
    height: int,
    width: int,
) -> Tensor:
    channels = windows.shape[-1]
    x = windows.view(
        batch_size,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        channels,
    )
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(batch_size, channels, height, width)


def _reshape_attention_tokens(tokens: Tensor, num_heads: int) -> Tensor:
    batch_windows, length, channels = tokens.shape
    if channels % num_heads != 0:
        raise ValueError(
            f"Attention channels={channels} must divide heads={num_heads}."
        )
    head_dim = channels // num_heads
    return tokens.view(batch_windows, length, num_heads, head_dim).transpose(1, 2)


def _global_sdpa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: int,
) -> Tensor:
    batch, channels, height, width = query.shape

    def flatten(x: Tensor) -> Tensor:
        tokens = x.flatten(2).transpose(1, 2)
        return _reshape_attention_tokens(tokens, num_heads)

    q = flatten(query)
    k = flatten(key)
    v = flatten(value)
    attended = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=False,
    )
    attended = attended.transpose(1, 2).reshape(batch, height * width, channels)
    return attended.transpose(1, 2).reshape(batch, channels, height, width)


def _windowed_sdpa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: int,
    window_size: int,
) -> Tensor:
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("Windowed attention requires equal Q/K/V shapes.")

    batch, channels, height, width = query.shape
    effective_window = max(1, min(int(window_size), height, width))
    query_pad, original_h, original_w = _pad_to_window(query, effective_window)
    key_pad, _, _ = _pad_to_window(key, effective_window)
    value_pad, _, _ = _pad_to_window(value, effective_window)
    padded_h, padded_w = query_pad.shape[-2:]

    q_tokens = _partition_windows(query_pad, effective_window)
    k_tokens = _partition_windows(key_pad, effective_window)
    v_tokens = _partition_windows(value_pad, effective_window)
    q = _reshape_attention_tokens(q_tokens, num_heads)
    k = _reshape_attention_tokens(k_tokens, num_heads)
    v = _reshape_attention_tokens(v_tokens, num_heads)

    attention_mask: Optional[Tensor] = None
    if padded_h != original_h or padded_w != original_w:
        valid = torch.ones(
            (batch, 1, original_h, original_w),
            device=query.device,
            dtype=torch.float32,
        )
        valid = F.pad(
            valid,
            (0, padded_w - original_w, 0, padded_h - original_h),
            value=0.0,
        )
        valid_tokens = _partition_windows(valid, effective_window).squeeze(-1) > 0.5
        attention_mask = torch.zeros(
            q.shape[0],
            1,
            1,
            q.shape[-2],
            device=q.device,
            dtype=q.dtype,
        )
        attention_mask = attention_mask.masked_fill(
            ~valid_tokens[:, None, None, :],
            torch.finfo(q.dtype).min,
        )

    attended = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
    )
    attended = attended.transpose(1, 2).reshape(
        q_tokens.shape[0],
        effective_window * effective_window,
        channels,
    )
    restored = _reverse_windows(
        attended,
        effective_window,
        batch,
        padded_h,
        padded_w,
    )
    return restored[:, :, :original_h, :original_w]


class ReliabilityGuidedDualPathScale(nn.Module):
    """One scale of CT-anchored robust/synergistic fusion."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: Optional[int],
    ) -> None:
        super().__init__()
        if channels <= 0 or num_heads <= 0:
            raise ValueError("channels and num_heads must be positive.")

        internal_channels = max(channels // 2, num_heads)
        internal_channels = (
            (internal_channels + num_heads - 1) // num_heads
        ) * num_heads

        self.channels = int(channels)
        self.internal_channels = int(internal_channels)
        self.num_heads = int(num_heads)
        self.window_size = None if window_size is None else int(window_size)

        self.ct_norm = LayerNorm2d(channels)
        self.pet_norm = LayerNorm2d(channels)
        self.query = nn.Conv2d(
            channels,
            internal_channels,
            kernel_size=1,
            bias=False,
        )
        self.key = nn.Conv2d(
            channels,
            internal_channels,
            kernel_size=1,
            bias=False,
        )
        self.value = nn.Conv2d(
            channels,
            internal_channels,
            kernel_size=1,
            bias=False,
        )
        self.output = nn.Conv2d(
            internal_channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        # The collaborative residual starts at zero while gradients can still
        # train the output projection on the first optimizer step.
        _zero_module(self.output)

    def forward(
        self,
        ct_feature: Tensor,
        pet_feature: Tensor,
        reliability: Tensor,
        return_diagnostics: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        if ct_feature.shape != pet_feature.shape:
            raise ValueError("CT/PET feature shapes must match at every scale.")
        if ct_feature.ndim != 4 or ct_feature.shape[1] != self.channels:
            raise ValueError(
                f"Expected [B,{self.channels},H,W], got {tuple(ct_feature.shape)}."
            )
        if reliability.shape != (ct_feature.shape[0], 1):
            raise ValueError(
                "Scale reliability must have shape [B,1], got "
                f"{tuple(reliability.shape)}."
            )

        gate = reliability.to(
            device=pet_feature.device,
            dtype=pet_feature.dtype,
        ).view(-1, 1, 1, 1)
        trusted_pet = gate * pet_feature
        robust_feature = ct_feature + trusted_pet

        normalized_ct = self.ct_norm(ct_feature)
        normalized_pet = self.pet_norm(trusted_pet)
        query = self.query(normalized_ct)
        key = self.key(normalized_pet)
        value = self.value(normalized_pet)
        if self.window_size is None:
            attended = _global_sdpa(
                query,
                key,
                value,
                num_heads=self.num_heads,
            )
        else:
            attended = _windowed_sdpa(
                query,
                key,
                value,
                num_heads=self.num_heads,
                window_size=self.window_size,
            )
        synergy_delta = self.output(attended)
        fused_feature = robust_feature + gate * synergy_delta

        if not return_diagnostics:
            return fused_feature
        return fused_feature, {
            "reliability": reliability.detach(),
            "ct_rms": ct_feature.detach().float().square().mean().sqrt(),
            "pet_rms": pet_feature.detach().float().square().mean().sqrt(),
            "trusted_pet_rms": trusted_pet.detach().float().square().mean().sqrt(),
            "synergy_rms": synergy_delta.detach().float().square().mean().sqrt(),
            "fused_rms": fused_feature.detach().float().square().mean().sqrt(),
        }


class TaskUtilityReliabilityRankingLoss(nn.Module):
    """Rank reliability logits by detached counterfactual task utility.

    A fixed ``(utility=0, logit=0)`` anchor makes ``r=0.5`` the neutral point:

        utility > 0  -> desired reliability > 0.5
        utility < 0  -> desired reliability < 0.5

    This is one loss, with no feature reconstruction, real/proxy alignment,
    MSE, SmoothL1, KL, contrastive term, or auxiliary latent encoder.
    """

    def __init__(self, tie_epsilon: float = 1e-8) -> None:
        super().__init__()
        self.tie_epsilon = float(tie_epsilon)

    def forward(
        self,
        reliability_logits: Tensor,
        utility: Tensor,
        scale_index: int,
    ) -> Tuple[Tensor, int]:
        if reliability_logits.ndim != 2:
            raise ValueError("reliability_logits must have shape [B,S].")
        if utility.ndim != 1 or utility.shape[0] != reliability_logits.shape[0]:
            raise ValueError("utility must have shape [B].")
        if not 0 <= int(scale_index) < reliability_logits.shape[1]:
            raise ValueError(f"Invalid scale_index={scale_index}.")

        scores = reliability_logits[:, int(scale_index)].float()
        utilities = utility.detach().float().to(device=scores.device)
        # Fixed neutral anchor: sigmoid(0) == 0.5.
        scores = torch.cat((scores, scores.new_zeros(1)), dim=0)
        utilities = torch.cat((utilities, utilities.new_zeros(1)), dim=0)

        preferred = utilities[:, None] > (
            utilities[None, :] + self.tie_epsilon
        )
        pair_count = int(preferred.sum().item())
        if pair_count == 0:
            return reliability_logits.sum() * 0.0, 0

        # preferred[p, q] means p should have a larger logit than q.
        pair_losses = F.softplus(scores[None, :] - scores[:, None])
        return pair_losses[preferred].mean(), pair_count


@contextmanager
def _preserve_batchnorm_running_stats(module: nn.Module) -> Iterator[None]:
    """Prevent a no-gradient counterfactual pass from updating BN buffers."""

    snapshots: List[Tuple[nn.Module, Optional[Tensor], Optional[Tensor], Optional[Tensor]]] = []
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            snapshots.append(
                (
                    child,
                    None if child.running_mean is None else child.running_mean.clone(),
                    None if child.running_var is None else child.running_var.clone(),
                    None
                    if child.num_batches_tracked is None
                    else child.num_batches_tracked.clone(),
                )
            )
    try:
        yield
    finally:
        for child, running_mean, running_var, tracked in snapshots:
            if running_mean is not None:
                child.running_mean.copy_(running_mean)
            if running_var is not None:
                child.running_var.copy_(running_var)
            if tracked is not None:
                child.num_batches_tracked.copy_(tracked)


class TextGuidedTaskUtilityReliabilityFusion(nn.Module):
    """Complete four-scale TG-TURF fusion module.

    Args:
        channels: Four encoder feature widths.
        use_text: Default text switch.  If ``False``, no text model is loaded.
        text_embeddings: Optional precomputed ``[2,text_dim]`` embeddings.
        use_turr_loss: Enable the training-only TURR helper by default.
        turr_interval: Evaluate one counterfactual scale every N optimizer steps.

    Text ablation:
        Construct with ``use_text=True`` so embeddings/projection exist, then
        pass ``text_enabled=False`` to ``forward`` or call
        ``set_text_enabled(False)``.  A module constructed with
        ``use_text=False`` cannot enable text later because no text parameters
        or embeddings were allocated.

    TURR ablation:
        Pass ``enabled=False`` to ``compute_turr_loss`` or call
        ``set_turr_enabled(False)``.  The returned loss is a graph-connected
        scalar zero, so training code needs no special-case branch.
    """

    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 320, 512),
        d_model: int = 64,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        num_attention_heads: int = 4,
        shallow_window_size: int = 8,
        use_text: bool = True,
        text_embeddings: Optional[Tensor] = None,
        biomedclip_model_path: str = BIOMEDCLIP_MODEL_PATH,
        biomedbert_text_tower_path: str = BIOMEDBERT_TEXT_TOWER_PATH,
        use_turr_loss: bool = True,
        turr_interval: int = 4,
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"Exactly four scales are required, got {len(channels)}.")
        if any(int(channel) <= 0 for channel in channels):
            raise ValueError("All channel counts must be positive.")
        if d_model <= 0 or num_attention_heads <= 0:
            raise ValueError("d_model and num_attention_heads must be positive.")
        if turr_interval <= 0:
            raise ValueError("turr_interval must be positive.")

        self.channels = tuple(int(channel) for channel in channels)
        self.num_scales = len(self.channels)
        self.d_model = int(d_model)
        self.default_text_enabled = bool(use_text)
        self.default_turr_enabled = bool(use_turr_loss)
        self.turr_interval = int(turr_interval)

        self.ct_tokens = nn.ModuleList(
            GlobalScaleToken(channel, d_model) for channel in self.channels
        )
        self.pet_tokens = nn.ModuleList(
            GlobalScaleToken(channel, d_model) for channel in self.channels
        )

        if use_text:
            if text_embeddings is None:
                text_embeddings = load_local_pet_text_embeddings(
                    biomedclip_model_path=biomedclip_model_path,
                    biomedbert_text_tower_path=biomedbert_text_tower_path,
                )
            if not torch.is_tensor(text_embeddings):
                raise TypeError("text_embeddings must be a Tensor.")
            if text_embeddings.ndim != 2 or text_embeddings.shape[0] != 2:
                raise ValueError(
                    "text_embeddings must have shape [2,text_dim], got "
                    f"{tuple(text_embeddings.shape)}."
                )
            if not text_embeddings.is_floating_point():
                raise TypeError("text_embeddings must be floating point.")
            normalized_text = F.normalize(
                text_embeddings.detach().float(),
                dim=-1,
            )
            if not torch.isfinite(normalized_text).all():
                raise ValueError("text_embeddings contain NaN or Inf.")
            if torch.allclose(normalized_text[0], normalized_text[1]):
                raise ValueError("Real/proxy text embedding rows must differ.")
            self.register_buffer("text_embeddings", normalized_text)
            self.text_projection: Optional[nn.Module] = nn.Sequential(
                nn.Linear(normalized_text.shape[1], d_model, bias=True),
                nn.LayerNorm(d_model),
            )
            nn.init.xavier_uniform_(self.text_projection[0].weight)
            nn.init.zeros_(self.text_projection[0].bias)
        else:
            self.register_buffer("text_embeddings", torch.empty(0, 0))
            self.text_projection = None

        self.context = BidirectionalMambaContext(
            d_model=d_model,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )
        self.reliability_head = nn.Linear(d_model, 1, bias=True)
        # Neutral start: a_s=0 and r_s=0.5.  First step trains the head; later
        # steps propagate reliability gradients into tokenizers and Mamba.
        _zero_module(self.reliability_head)

        scale_modules: List[nn.Module] = []
        for scale_index, channel in enumerate(self.channels):
            window_size = shallow_window_size if scale_index < 2 else None
            scale_modules.append(
                ReliabilityGuidedDualPathScale(
                    channels=channel,
                    num_heads=num_attention_heads,
                    window_size=window_size,
                )
            )
        self.scales = nn.ModuleList(scale_modules)
        self.turr_criterion = TaskUtilityReliabilityRankingLoss()

    @property
    def text_available(self) -> bool:
        return self.text_projection is not None and self.text_embeddings.numel() > 0

    def set_text_enabled(self, enabled: bool) -> None:
        if enabled and not self.text_available:
            raise RuntimeError(
                "Text cannot be enabled because this instance was constructed "
                "with use_text=False."
            )
        self.default_text_enabled = bool(enabled)

    def set_turr_enabled(self, enabled: bool) -> None:
        self.default_turr_enabled = bool(enabled)

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
        # available=1 -> real state 0; available=0 -> proxy state 1.
        return 1 - availability

    def _validate_features(
        self,
        ct_features: Sequence[Tensor],
        pet_features: Sequence[Tensor],
    ) -> None:
        if len(ct_features) != self.num_scales or len(pet_features) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} CT/PET scales, got "
                f"{len(ct_features)} and {len(pet_features)}."
            )
        batch_size = ct_features[0].shape[0]
        for scale_index, (ct_feature, pet_feature, channel) in enumerate(
            zip(ct_features, pet_features, self.channels)
        ):
            if ct_feature.shape != pet_feature.shape:
                raise ValueError(
                    f"Scale {scale_index}: CT/PET shapes differ: "
                    f"{tuple(ct_feature.shape)} vs {tuple(pet_feature.shape)}."
                )
            if ct_feature.ndim != 4 or ct_feature.shape[1] != channel:
                raise ValueError(
                    f"Scale {scale_index}: expected [B,{channel},H,W], got "
                    f"{tuple(ct_feature.shape)}."
                )
            if ct_feature.shape[0] != batch_size:
                raise ValueError("All feature scales must share the same batch size.")

    def _build_tokens(
        self,
        ct_features: Sequence[Tensor],
        pet_features: Sequence[Tensor],
        state_ids: Tensor,
        text_enabled: bool,
    ) -> Tuple[Tensor, List[int]]:
        sequence: List[Tensor] = []
        if text_enabled:
            assert self.text_projection is not None
            selected_text = self.text_embeddings.index_select(0, state_ids)
            parameter = self.text_projection[0].weight
            selected_text = selected_text.to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            sequence.append(self.text_projection(selected_text))

        pet_indices: List[int] = []
        for ct_tokenizer, pet_tokenizer, ct_feature, pet_feature in zip(
            self.ct_tokens,
            self.pet_tokens,
            ct_features,
            pet_features,
        ):
            sequence.append(ct_tokenizer(ct_feature))
            sequence.append(pet_tokenizer(pet_feature))
            pet_indices.append(len(sequence) - 1)
        return torch.stack(sequence, dim=1), pet_indices

    def predict_reliability(
        self,
        ct_features: Sequence[Tensor],
        pet_features: Sequence[Tensor],
        mode: str = "full",
        pet_available: Optional[Tensor] = None,
        text_enabled: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        self._validate_features(ct_features, pet_features)
        batch_size = ct_features[0].shape[0]
        device = ct_features[0].device
        state_ids = self._state_ids(
            mode=mode,
            batch_size=batch_size,
            device=device,
            pet_available=pet_available,
        )

        use_text_now = (
            self.default_text_enabled
            if text_enabled is None
            else bool(text_enabled)
        )
        if use_text_now and not self.text_available:
            raise RuntimeError(
                "text_enabled=True was requested, but this instance has no "
                "text embeddings/projection."
            )

        tokens, pet_indices = self._build_tokens(
            ct_features,
            pet_features,
            state_ids,
            text_enabled=use_text_now,
        )
        # Preserve autocast output dtype so the official Mamba kernels can run
        # in the training loop's selected mixed precision.
        tokens = tokens.to(device=device)
        contextualized = self.context(tokens)
        pet_context = contextualized[:, pet_indices, :]
        reliability_logits = self.reliability_head(pet_context).squeeze(-1)
        reliability = torch.sigmoid(reliability_logits)
        return reliability_logits, reliability, state_ids

    def fuse_with_reliability(
        self,
        ct_features: Sequence[Tensor],
        pet_features: Sequence[Tensor],
        reliability: Tensor,
        return_diagnostics: bool = False,
    ) -> Union[
        List[Tensor],
        Tuple[List[Tensor], List[Dict[str, Tensor]]],
    ]:
        self._validate_features(ct_features, pet_features)
        expected_shape = (ct_features[0].shape[0], self.num_scales)
        if reliability.shape != expected_shape:
            raise ValueError(
                f"reliability must have shape {expected_shape}, got "
                f"{tuple(reliability.shape)}."
            )

        fused_features: List[Tensor] = []
        scale_diagnostics: List[Dict[str, Tensor]] = []
        for scale_index, (scale, ct_feature, pet_feature) in enumerate(
            zip(self.scales, ct_features, pet_features)
        ):
            result = scale(
                ct_feature,
                pet_feature,
                reliability[:, scale_index : scale_index + 1],
                return_diagnostics=return_diagnostics,
            )
            if return_diagnostics:
                fused_feature, diagnostics = result
                scale_diagnostics.append(diagnostics)
            else:
                fused_feature = result
            fused_features.append(fused_feature)

        if return_diagnostics:
            return fused_features, scale_diagnostics
        return fused_features

    def forward(
        self,
        ct_features: Sequence[Tensor],
        pet_features: Sequence[Tensor],
        mode: str = "full",
        pet_available: Optional[Tensor] = None,
        text_enabled: Optional[bool] = None,
        reliability_override: Optional[Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Union[
        List[Tensor],
        Tuple[List[Tensor], Dict[str, Any]],
    ]:
        reliability_logits, predicted_reliability, state_ids = (
            self.predict_reliability(
                ct_features,
                pet_features,
                mode=mode,
                pet_available=pet_available,
                text_enabled=text_enabled,
            )
        )
        reliability = (
            predicted_reliability
            if reliability_override is None
            else reliability_override
        )
        use_text_now = (
            self.default_text_enabled
            if text_enabled is None
            else bool(text_enabled)
        )

        result = self.fuse_with_reliability(
            ct_features,
            pet_features,
            reliability,
            return_diagnostics=return_diagnostics,
        )
        if not return_diagnostics:
            return result

        fused_features, scale_diagnostics = result
        diagnostics: Dict[str, Any] = {
            # Keep logits attached: TURR must send gradients to this tensor.
            "reliability_logits": reliability_logits,
            "reliability": reliability.detach(),
            "predicted_reliability": predicted_reliability.detach(),
            "pet_state_ids": state_ids.detach(),
            "text_enabled": use_text_now,
            "scale_diagnostics": scale_diagnostics,
        }
        return fused_features, diagnostics

    def should_compute_turr(
        self,
        global_step: int,
        enabled: Optional[bool] = None,
    ) -> bool:
        use_turr = (
            self.default_turr_enabled if enabled is None else bool(enabled)
        )
        return bool(
            use_turr
            and self.training
            and int(global_step) % self.turr_interval == 0
        )

    def turr_scale_index(self, global_step: int) -> int:
        return (int(global_step) // self.turr_interval) % self.num_scales

    def compute_turr_loss(
        self,
        *,
        global_step: int,
        decoder: nn.Module,
        main_prediction: Any,
        target: Tensor,
        segmentation_loss_per_sample: Callable[[Any, Tensor], Tensor],
        ct_features: Sequence[Tensor],
        main_fused_features: Sequence[Tensor],
        reliability_logits: Tensor,
        decoder_args: Sequence[Any] = (),
        decoder_kwargs: Optional[Mapping[str, Any]] = None,
        enabled: Optional[bool] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute the optional sparse TURR loss.

        ``segmentation_loss_per_sample`` must return shape ``[B]``.  It should
        reproduce the project's existing segmentation criterion per sample;
        TURR does not prescribe or add a new segmentation loss.

        Example for a decoder with signature ``decoder(features, target_size)``::

            turr_loss, info = fusion.compute_turr_loss(
                global_step=global_step,
                decoder=model.decoder,
                decoder_args=(target_size,),
                main_prediction=output,
                target=mask,
                segmentation_loss_per_sample=project_loss_per_sample,
                ct_features=ct_feats,
                main_fused_features=fused_feats,
                reliability_logits=diag["reliability_logits"],
            )

        The counterfactual scale is replaced by CT exactly, which is the
        closed-loop fusion result at ``r_s=0``.  Encoders, calibration, text,
        Mamba, and the other scales are not recomputed.
        """

        zero = reliability_logits.sum() * 0.0
        if not self.should_compute_turr(global_step, enabled=enabled):
            return zero, {
                "computed": False,
                "scale_index": None,
                "pair_count": 0,
            }

        if len(ct_features) != self.num_scales:
            raise ValueError("ct_features must contain four scales.")
        if len(main_fused_features) != self.num_scales:
            raise ValueError("main_fused_features must contain four scales.")
        batch_size = reliability_logits.shape[0]
        if reliability_logits.shape != (batch_size, self.num_scales):
            raise ValueError(
                "reliability_logits must have shape [B,4], got "
                f"{tuple(reliability_logits.shape)}."
            )

        scale_index = self.turr_scale_index(global_step)
        counterfactual_features = [
            feature.detach() for feature in main_fused_features
        ]
        # At r_s=0: robust=CT and the gated synergy term is zero.
        counterfactual_features[scale_index] = ct_features[scale_index].detach()

        kwargs = {} if decoder_kwargs is None else dict(decoder_kwargs)
        detached_target = target.detach()
        with torch.no_grad():
            main_loss_per_sample = segmentation_loss_per_sample(
                _detach_tree(main_prediction),
                detached_target,
            )
            with _preserve_batchnorm_running_stats(decoder):
                counterfactual_prediction = decoder(
                    counterfactual_features,
                    *tuple(decoder_args),
                    **kwargs,
                )
            counterfactual_loss_per_sample = segmentation_loss_per_sample(
                counterfactual_prediction,
                detached_target,
            )

        if main_loss_per_sample.shape != (batch_size,):
            raise ValueError(
                "segmentation_loss_per_sample(main_prediction, target) must "
                f"return [B], got {tuple(main_loss_per_sample.shape)}."
            )
        if counterfactual_loss_per_sample.shape != (batch_size,):
            raise ValueError(
                "segmentation_loss_per_sample(counterfactual, target) must "
                f"return [B], got {tuple(counterfactual_loss_per_sample.shape)}."
            )

        utility = (
            counterfactual_loss_per_sample
            - main_loss_per_sample
        ).detach()
        turr_loss, pair_count = self.turr_criterion(
            reliability_logits,
            utility,
            scale_index,
        )
        return turr_loss, {
            "computed": True,
            "scale_index": scale_index,
            "pair_count": pair_count,
            "utility": utility,
            "main_loss_per_sample": main_loss_per_sample.detach(),
            "counterfactual_loss_per_sample": (
                counterfactual_loss_per_sample.detach()
            ),
        }


def binary_dice_bce_per_sample(
    prediction: Any,
    target: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Minimal example of a per-sample segmentation criterion.

    This helper is provided for the standalone self-test and integration
    examples.  Replace it with the project's existing per-sample loss so the
    main segmentation objective is unchanged.
    """

    if isinstance(prediction, dict):
        logits = prediction.get("logits", prediction.get("pred"))
    else:
        logits = prediction
    if not torch.is_tensor(logits):
        raise TypeError("prediction must be a Tensor or contain logits/pred.")
    if logits.shape != target.shape:
        raise ValueError(
            f"Binary logits/target shapes differ: {tuple(logits.shape)} vs "
            f"{tuple(target.shape)}."
        )

    target = target.to(device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    ).flatten(1).mean(dim=1)
    probability = torch.sigmoid(logits)
    intersection = (probability * target).flatten(1).sum(dim=1)
    denominator = probability.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    dice_loss = 1.0 - (2.0 * intersection + eps) / (denominator + eps)
    return bce + dice_loss


class _SelfTestDecoder(nn.Module):
    """Tiny decoder used only by this file's optional self-test."""

    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Conv2d(channel, 1, kernel_size=1) for channel in channels
        )

    def forward(self, features: Sequence[Tensor]) -> Dict[str, Tensor]:
        target_size = features[0].shape[-2:]
        logits = 0.0
        for projection, feature in zip(self.projections, features):
            value = projection(feature)
            if value.shape[-2:] != target_size:
                value = F.interpolate(
                    value,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            logits = logits + value
        return {"logits": logits}


def build_dummy_text_embeddings(text_dim: int = 512) -> Tensor:
    positions = torch.arange(text_dim, dtype=torch.float32)
    real = torch.sin(positions / 31.0)
    proxy = torch.cos(positions / 29.0)
    return torch.stack((real, proxy), dim=0)


def run_self_test(args: argparse.Namespace) -> None:
    """Run forward/backward, switches, state mapping, and TURR checks."""

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    use_amp = bool(args.amp and device.type == "cuda")

    channels = (32, 64, 128, 256)
    text_embeddings = build_dummy_text_embeddings(args.text_dim)
    module = TextGuidedTaskUtilityReliabilityFusion(
        channels=channels,
        use_text=True,
        text_embeddings=text_embeddings,
        use_turr_loss=True,
        turr_interval=1,
    ).to(device)
    decoder = _SelfTestDecoder(channels).to(device)
    module.train()
    decoder.train()

    shapes = (
        (args.batch_size, 32, 32, 32),
        (args.batch_size, 64, 16, 16),
        (args.batch_size, 128, 8, 8),
        (args.batch_size, 256, 4, 4),
    )
    ct_features = [
        torch.randn(shape, device=device, requires_grad=True) for shape in shapes
    ]
    pet_features = [
        torch.randn(shape, device=device, requires_grad=True) for shape in shapes
    ]
    pet_available = torch.arange(args.batch_size, device=device) % 2
    target = torch.randint(
        0,
        2,
        (args.batch_size, 1, 32, 32),
        device=device,
    ).float()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=use_amp,
    ):
        fused_features, diagnostics = module(
            ct_features,
            pet_features,
            mode="auto",
            pet_available=pet_available,
            text_enabled=True,
            return_diagnostics=True,
        )
        main_prediction = decoder(fused_features)
        segmentation_per_sample = binary_dice_bce_per_sample(
            main_prediction,
            target,
        )
        turr_loss, turr_info = module.compute_turr_loss(
            global_step=0,
            decoder=decoder,
            main_prediction=main_prediction,
            target=target,
            segmentation_loss_per_sample=binary_dice_bce_per_sample,
            ct_features=ct_features,
            main_fused_features=fused_features,
            reliability_logits=diagnostics["reliability_logits"],
        )
        total_loss = segmentation_per_sample.mean() + turr_loss
    total_loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    for output, expected in zip(fused_features, shapes):
        if output.shape != expected or not torch.isfinite(output).all():
            raise RuntimeError("Fusion output shape/finite check failed.")
    if diagnostics["reliability"].shape != (args.batch_size, 4):
        raise RuntimeError("Reliability shape check failed.")
    torch.testing.assert_close(
        diagnostics["pet_state_ids"],
        1 - pet_available,
    )
    if not torch.isfinite(total_loss):
        raise RuntimeError("Total loss contains NaN/Inf.")
    if module.reliability_head.weight.grad is None:
        raise RuntimeError("TURR/main loss did not reach reliability head.")

    # Verify runtime switches without constructing another module.
    with torch.no_grad():
        no_text_features = module(
            [feature.detach() for feature in ct_features],
            [feature.detach() for feature in pet_features],
            mode="full",
            text_enabled=False,
        )
        if any(not torch.isfinite(feature).all() for feature in no_text_features):
            raise RuntimeError("Text-disabled forward contains NaN/Inf.")
    zero_turr, skipped = module.compute_turr_loss(
        global_step=1,
        decoder=decoder,
        main_prediction=main_prediction,
        target=target,
        segmentation_loss_per_sample=binary_dice_bce_per_sample,
        ct_features=ct_features,
        main_fused_features=fused_features,
        reliability_logits=diagnostics["reliability_logits"],
        enabled=False,
    )
    if zero_turr.detach().abs().item() != 0.0 or skipped["computed"]:
        raise RuntimeError("TURR switch check failed.")

    total_parameters, trainable_parameters = _count_parameters(module)
    print("TG-TURF standalone self-test: PASS")
    print(f"device: {device}")
    print(f"AMP: {use_amp}")
    print(f"four-scale shapes: {shapes}")
    print(f"reliability shape: {tuple(diagnostics['reliability'].shape)}")
    print(f"TURR scale: {turr_info['scale_index']}")
    print(f"TURR ranking pairs: {turr_info['pair_count']}")
    print(f"parameters: {total_parameters:,} total")
    print(f"trainable parameters: {trainable_parameters:,}")
    print(f"forward + counterfactual + backward: {elapsed:.3f} s")
    if device.type == "cuda":
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"peak allocated CUDA memory: {peak_mib:.1f} MiB")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone text-guided task-utility CT/PET fusion."
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--text-dim", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser


__all__ = [
    "BIOMEDCLIP_MODEL_PATH",
    "BIOMEDBERT_TEXT_TOWER_PATH",
    "REAL_PET_PROMPT",
    "PROXY_PET_PROMPT",
    "PET_PROMPTS",
    "load_local_pet_text_embeddings",
    "LayerNorm2d",
    "GlobalScaleToken",
    "BidirectionalMambaContext",
    "ReliabilityGuidedDualPathScale",
    "TaskUtilityReliabilityRankingLoss",
    "TextGuidedTaskUtilityReliabilityFusion",
    "binary_dice_bce_per_sample",
    "build_dummy_text_embeddings",
]


if __name__ == "__main__":
    run_self_test(build_argument_parser().parse_args())
