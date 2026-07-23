"""Text-guided CT-anchored fusion for real PET and MPPC compensation.

This file implements one standalone four-scale fusion module:

    [CT, real-PET / MPPC-proxy]
        -> joint channel attention
        -> fixed BioMedCLIP text-guided channel modulation
        -> 1x1 channel compression
        -> CT-anchored deformable local aggregation
        -> CT residual update

The implementation borrows the *continuous channel weighting + text MLP* idea
from TCPM in UP-Fusion, but deliberately removes hard Top-k selection and
channel reordering.  It borrows the *cooperative offset prediction + local
sampling* idea from CDA in IDEA, but changes the task to CT-anchored retrieval:
CT is always the query/anchor and the auxiliary feature is only candidate
evidence.  There is no direct PET/proxy-to-output bypass.

References:
    UP-Fusion (AAAI 2026):
        https://github.com/ixilai/UP-Fusion
    IDEA / CDA (CVPR 2025):
        https://github.com/924973292/IDEA

Recommended prompts are intentionally short, include both CT and PET roles, and
serve tumor segmentation without relying on difficult medical terminology.

Typical use:

    fusion = TextGuidedCTAnchorFusion.from_local_biomedclip(
        biomedclip_dir="pretrained/biomedclip_model",
        text_tower_dir="pretrained/biomedbert_text_tower",
        channels=(64, 128, 320, 512),
    )

    # Full: auxiliary_features are real PET encoder features.
    fused = fusion(ct_features, pet_features, mode="full")

    # Missing: auxiliary_features are produced by MPPC; PET is not loaded.
    pet_proxy = mppc(ct_features, pet_features=None, mode="missing")
    fused = fusion(ct_features, pet_proxy, mode="missing")

For repeated experiments, precompute the two fixed prompt embeddings once:

    save_local_biomedclip_prompt_embeddings(
        output_path="pretrained/biomedclip_model/petct_fusion_prompts.pt",
        biomedclip_dir="pretrained/biomedclip_model",
        text_tower_dir="pretrained/biomedbert_text_tower",
    )
    fusion = TextGuidedCTAnchorFusion.from_embedding_file(
        "pretrained/biomedclip_model/petct_fusion_prompts.pt"
    )
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


FusionMode = Literal["full", "missing"]

DEFAULT_FULL_PROMPT = (
    "CT shows tumor structure, and acquired PET provides lesion activity cues "
    "for tumor segmentation."
)
DEFAULT_MISSING_PROMPT = (
    "CT shows tumor structure, and PET compensation provides uncertain "
    "lesion cues for tumor segmentation."
)


def _resolve_weight_file(model_dir: Path) -> Path:
    candidates = (
        "open_clip_pytorch_model.bin",
        "pytorch_model.bin",
        "model.safetensors",
    )
    for name in candidates:
        path = model_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No BioMedCLIP weight file found under {model_dir}. "
        f"Expected one of: {', '.join(candidates)}"
    )


@torch.no_grad()
def encode_prompts_with_local_biomedclip(
    biomedclip_dir: Union[str, os.PathLike],
    text_tower_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    full_prompt: str = DEFAULT_FULL_PROMPT,
    missing_prompt: str = DEFAULT_MISSING_PROMPT,
    device: Union[str, torch.device] = "cpu",
    context_length: int = 256,
) -> Tensor:
    """Encode the two fixed prompts with local BioMedCLIP weights.

    The loader follows the local-loading implementation distributed with
    BioMedCLIP.  The model is used only here and is not retained by the fusion
    module.  The returned tensor has shape ``[2, text_dim]`` in the order
    ``[full, missing]`` and is L2-normalized.

    ``text_tower_dir`` should point to the local
    ``biomedbert_text_tower`` directory when offline training is required.
    """

    try:
        from open_clip import create_model_and_transforms, get_tokenizer
        from open_clip.factory import _MODEL_CONFIGS
    except ImportError as exc:
        raise ImportError(
            "Local BioMedCLIP prompt encoding requires open-clip-torch. "
            "Install the same open_clip version used by your BioMedCLIP setup."
        ) from exc

    model_dir = Path(biomedclip_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"BioMedCLIP directory does not exist: {model_dir}")

    config_path = model_dir / "open_clip_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing BioMedCLIP config: {config_path}")
    weight_path = _resolve_weight_file(model_dir)

    with config_path.open("r", encoding="utf-8") as file_obj:
        config = json.load(file_obj)
    if "model_cfg" not in config:
        raise KeyError(f"{config_path} does not contain 'model_cfg'")

    model_cfg = copy.deepcopy(config["model_cfg"])
    preprocess_cfg = copy.deepcopy(config.get("preprocess_cfg", {}))

    if text_tower_dir is not None:
        local_text_dir = Path(text_tower_dir).expanduser().resolve()
        if not local_text_dir.is_dir():
            raise FileNotFoundError(
                f"Local BioMedBERT text tower does not exist: {local_text_dir}"
            )
        text_cfg = model_cfg.setdefault("text_cfg", {})
        text_cfg["hf_model_name"] = str(local_text_dir)
        text_cfg["hf_tokenizer_name"] = str(local_text_dir)

    # The BioMedCLIP README uses this registry route for fully local loading.
    model_name = "petct_biomedclip_local"
    _MODEL_CONFIGS[model_name] = model_cfg
    tokenizer = get_tokenizer(model_name)
    model, _, _ = create_model_and_transforms(
        model_name=model_name,
        pretrained=str(weight_path),
        **{f"image_{key}": value for key, value in preprocess_cfg.items()},
    )

    run_device = torch.device(device)
    model = model.to(run_device).eval()
    tokens = tokenizer(
        [full_prompt, missing_prompt],
        context_length=int(context_length),
    ).to(run_device)

    try:
        embeddings = model.encode_text(tokens, normalize=True)
    except TypeError:
        embeddings = F.normalize(model.encode_text(tokens), dim=-1)

    embeddings = embeddings.detach().float().cpu()
    if embeddings.ndim != 2 or embeddings.shape[0] != 2:
        raise RuntimeError(
            "BioMedCLIP must return two prompt embeddings shaped [2, D], "
            f"got {tuple(embeddings.shape)}"
        )
    if not torch.isfinite(embeddings).all():
        raise FloatingPointError("BioMedCLIP produced non-finite prompt embeddings")
    return F.normalize(embeddings, dim=-1)


def save_local_biomedclip_prompt_embeddings(
    output_path: Union[str, os.PathLike],
    biomedclip_dir: Union[str, os.PathLike],
    text_tower_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    full_prompt: str = DEFAULT_FULL_PROMPT,
    missing_prompt: str = DEFAULT_MISSING_PROMPT,
    device: Union[str, torch.device] = "cpu",
) -> Path:
    """Encode and save the fixed prompt embeddings for later training runs."""

    embeddings = encode_prompts_with_local_biomedclip(
        biomedclip_dir=biomedclip_dir,
        text_tower_dir=text_tower_dir,
        full_prompt=full_prompt,
        missing_prompt=missing_prompt,
        device=device,
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings,
            "prompts": {
                "full": full_prompt,
                "missing": missing_prompt,
            },
            "source_to_index": {"full": 0, "missing": 1},
        },
        output,
    )
    return output


def load_prompt_embeddings(
    embedding_path: Union[str, os.PathLike],
) -> Tuple[Tensor, Optional[Mapping[str, str]]]:
    """Load ``[full, missing]`` prompt embeddings from a local ``.pt`` file."""

    path = Path(embedding_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prompt embedding file does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    prompts: Optional[Mapping[str, str]] = None
    if torch.is_tensor(payload):
        embeddings = payload
    elif isinstance(payload, Mapping) and torch.is_tensor(payload.get("embeddings")):
        embeddings = payload["embeddings"]
        raw_prompts = payload.get("prompts")
        if isinstance(raw_prompts, Mapping):
            prompts = raw_prompts
    else:
        raise TypeError(
            f"{path} must contain a Tensor or a dict with Tensor key 'embeddings'"
        )

    embeddings = embeddings.detach().float()
    if embeddings.ndim == 3 and embeddings.shape[1] == 1:
        embeddings = embeddings[:, 0]
    if embeddings.ndim != 2 or embeddings.shape[0] != 2:
        raise ValueError(
            f"Prompt embeddings must have shape [2, D], got {tuple(embeddings.shape)}"
        )
    if not torch.isfinite(embeddings).all():
        raise FloatingPointError("Prompt embedding file contains NaN or Inf")
    return F.normalize(embeddings, dim=-1), prompts


def _valid_group_count(channels: int, preferred: int = 8) -> int:
    groups = min(int(preferred), int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


def _base_offsets(num_points: int) -> Tensor:
    """Return local sampling offsets in ``(dx, dy)`` pixel order."""

    patterns = {
        1: ((0.0, 0.0),),
        5: (
            (0.0, 0.0),
            (-1.0, 0.0),
            (1.0, 0.0),
            (0.0, -1.0),
            (0.0, 1.0),
        ),
        9: tuple(
            (float(dx), float(dy))
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        ),
    }
    if num_points not in patterns:
        raise ValueError(
            f"num_points must be one of {tuple(patterns)}, got {num_points}"
        )
    return torch.tensor(patterns[num_points], dtype=torch.float32)


class JointChannelAttention(nn.Module):
    """SE-style continuous channel attention over concatenated CT/aux features."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // int(reduction))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        # 2*sigmoid(0)=1: continuous CA starts as an identity rescaling.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        weights = 2.0 * torch.sigmoid(self.mlp(self.pool(x)))
        return x * weights, weights


class TextChannelModulation(nn.Module):
    """Map one fixed source prompt to differentiable visual channel gains."""

    def __init__(self, text_dim: int, visual_channels: int, hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, visual_channels),
        )
        # 1+tanh(0)=1: the text path is initially neutral but trainable.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: Tensor, text_embedding: Tensor) -> Tuple[Tensor, Tensor]:
        gains = 1.0 + torch.tanh(self.mlp(text_embedding))
        gains = gains[:, :, None, None].to(dtype=x.dtype)
        return x * gains, gains


class CTAnchoredDeformableLocalAggregation(nn.Module):
    """Retrieve local auxiliary evidence using CT queries and shared offsets.

    A zero-valued rejection candidate is included in the local softmax.  The
    auxiliary feature therefore cannot force a non-zero update: when the null
    candidate dominates, the aggregated residual approaches zero and the final
    output approaches the unchanged CT feature.
    """

    def __init__(
        self,
        channels: int,
        local_dim: int,
        num_heads: int,
        num_points: int = 5,
        max_offset: float = 1.0,
        residual_init: float = 0.1,
        null_init_bias: float = 1.0,
    ) -> None:
        super().__init__()
        if local_dim <= 0 or num_heads <= 0:
            raise ValueError("local_dim and num_heads must be positive")
        if local_dim % num_heads != 0:
            raise ValueError(
                f"local_dim={local_dim} must be divisible by num_heads={num_heads}"
            )
        if max_offset < 0:
            raise ValueError("max_offset must be >= 0")
        if residual_init <= 0:
            raise ValueError("residual_init must be > 0 to preserve branch gradients")

        self.channels = int(channels)
        self.local_dim = int(local_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.local_dim // self.num_heads
        self.num_points = int(num_points)
        self.max_offset = float(max_offset)

        self.q_proj = nn.Conv2d(channels, local_dim, kernel_size=1, bias=False)
        self.kv_proj = nn.Conv2d(
            channels, 2 * local_dim, kernel_size=1, bias=False
        )

        # CT and auxiliary evidence cooperatively predict one aligned sampling
        # grid.  The final layer is zero-initialized, so training starts from a
        # stable fixed local cross rather than random spatial jumps.
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(
                2 * local_dim,
                2 * local_dim,
                kernel_size=3,
                padding=1,
                groups=2 * local_dim,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(
                2 * local_dim,
                2 * num_points,
                kernel_size=1,
                bias=True,
            ),
        )
        nn.init.zeros_(self.offset_predictor[-1].weight)
        nn.init.zeros_(self.offset_predictor[-1].bias)

        # One null score per attention head. Its value is exactly zero.
        self.null_score = nn.Conv2d(
            2 * local_dim, num_heads, kernel_size=1, bias=True
        )
        nn.init.zeros_(self.null_score.weight)
        nn.init.constant_(self.null_score.bias, float(null_init_bias))

        self.out_proj = nn.Conv2d(local_dim, channels, kernel_size=1, bias=False)
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_init), dtype=torch.float32)
        )
        self.register_buffer(
            "base_offsets",
            _base_offsets(num_points),
            persistent=False,
        )

    @staticmethod
    def _identity_grid(
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if height > 1:
            y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        else:
            y = torch.zeros(1, device=device, dtype=dtype)
        if width > 1:
            x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        else:
            x = torch.zeros(1, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)

    @staticmethod
    def _pixels_to_normalized(
        pixel_offsets: Tensor,
        height: int,
        width: int,
    ) -> Tensor:
        normalized = pixel_offsets.clone()
        normalized[..., 0] *= 2.0 / max(width - 1, 1)
        normalized[..., 1] *= 2.0 / max(height - 1, 1)
        return normalized

    def _sampling_grid(
        self,
        learned_offsets: Tensor,
        point_idx: int,
        identity_grid: Tensor,
        height: int,
        width: int,
    ) -> Tensor:
        # [B, K, 2, H, W] -> [B, H, W, 2], coordinate order (dx, dy).
        offset = learned_offsets[:, point_idx].permute(0, 2, 3, 1)
        base = self.base_offsets[point_idx].to(
            device=offset.device,
            dtype=offset.dtype,
        )
        offset = offset + base.view(1, 1, 1, 2)
        return identity_grid + self._pixels_to_normalized(
            offset, height=height, width=width
        )

    def forward(
        self,
        ct: Tensor,
        evidence: Tensor,
        *,
        return_aux: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        batch, _, height, width = ct.shape
        q_raw = self.q_proj(ct)
        kv_raw = self.kv_proj(evidence)
        k_raw, _ = kv_raw.chunk(2, dim=1)

        offset_input = torch.cat((q_raw, k_raw), dim=1)
        learned_offsets = torch.tanh(self.offset_predictor(offset_input))
        learned_offsets = learned_offsets.view(
            batch, self.num_points, 2, height, width
        )
        learned_offsets = learned_offsets * self.max_offset

        q = q_raw.view(
            batch, self.num_heads, self.head_dim, height, width
        )
        q = F.normalize(q, dim=2, eps=1e-6)

        identity_grid = self._identity_grid(
            height,
            width,
            device=ct.device,
            dtype=ct.dtype,
        )

        candidate_scores: List[Tensor] = []
        sampled_values: List[Tensor] = []
        score_scale = 1.0 / math.sqrt(self.head_dim)

        for point_idx in range(self.num_points):
            grid = self._sampling_grid(
                learned_offsets,
                point_idx,
                identity_grid,
                height,
                width,
            )
            sampled_kv = F.grid_sample(
                kv_raw,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            sampled_k, sampled_v = sampled_kv.chunk(2, dim=1)
            sampled_k = sampled_k.view(
                batch, self.num_heads, self.head_dim, height, width
            )
            sampled_k = F.normalize(sampled_k, dim=2, eps=1e-6)
            sampled_v = sampled_v.view(
                batch, self.num_heads, self.head_dim, height, width
            )
            candidate_scores.append(
                (q * sampled_k).sum(dim=2) * score_scale
            )
            sampled_values.append(sampled_v)

        null_logits = self.null_score(offset_input)
        score_tensor = torch.stack(candidate_scores, dim=2)
        all_logits = torch.cat((null_logits.unsqueeze(2), score_tensor), dim=2)
        all_weights = F.softmax(all_logits, dim=2)
        candidate_weights = all_weights[:, :, 1:]

        aggregated = torch.zeros_like(q_raw).view(
            batch, self.num_heads, self.head_dim, height, width
        )
        for point_idx, sampled_v in enumerate(sampled_values):
            weight = candidate_weights[:, :, point_idx].unsqueeze(2)
            aggregated = aggregated + weight * sampled_v

        aggregated = aggregated.reshape(
            batch, self.local_dim, height, width
        )
        residual = self.out_proj(aggregated)
        residual = residual * torch.tanh(
            self.residual_scale
        ).to(dtype=residual.dtype)

        if not return_aux:
            return residual
        aux = {
            "acceptance": candidate_weights.sum(dim=2).mean(
                dim=1, keepdim=True
            ).detach(),
            "null_weight": all_weights[:, :, 0].mean(
                dim=1, keepdim=True
            ).detach(),
            "mean_offset_pixels": learned_offsets.norm(
                dim=2
            ).mean(dim=1, keepdim=True).detach(),
        }
        return residual, aux


class TextGuidedCTAnchorFusionScale(nn.Module):
    """One-scale visual/text conditioning followed by CT-anchored aggregation."""

    def __init__(
        self,
        channels: int,
        text_dim: int,
        local_dim: int,
        num_heads: int,
        *,
        ca_reduction: int = 8,
        text_hidden: Optional[int] = None,
        num_points: int = 5,
        max_offset: float = 1.0,
        residual_init: float = 0.1,
        null_init_bias: float = 1.0,
    ) -> None:
        super().__init__()
        joint_channels = 2 * int(channels)
        if text_hidden is None:
            text_hidden = min(128, max(32, channels // 2))

        self.channels = int(channels)
        self.channel_attention = JointChannelAttention(
            joint_channels,
            reduction=ca_reduction,
        )
        self.text_modulation = TextChannelModulation(
            text_dim=text_dim,
            visual_channels=joint_channels,
            hidden=int(text_hidden),
        )
        self.evidence_projection = nn.Sequential(
            nn.Conv2d(
                joint_channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                _valid_group_count(channels),
                channels,
            ),
            nn.GELU(),
        )
        self.local_aggregation = CTAnchoredDeformableLocalAggregation(
            channels=channels,
            local_dim=local_dim,
            num_heads=num_heads,
            num_points=num_points,
            max_offset=max_offset,
            residual_init=residual_init,
            null_init_bias=null_init_bias,
        )

    def forward(
        self,
        ct: Tensor,
        auxiliary: Tensor,
        text_embedding: Tensor,
        *,
        return_aux: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        if auxiliary.shape[-2:] != ct.shape[-2:]:
            auxiliary = F.interpolate(
                auxiliary,
                size=ct.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if ct.shape != auxiliary.shape:
            raise ValueError(
                "CT and auxiliary features must match after spatial alignment, "
                f"got CT={tuple(ct.shape)}, auxiliary={tuple(auxiliary.shape)}"
            )
        if ct.ndim != 4 or ct.shape[1] != self.channels:
            raise ValueError(
                f"Expected [B,{self.channels},H,W], got {tuple(ct.shape)}"
            )

        joint = torch.cat((ct, auxiliary), dim=1)
        attended, channel_weights = self.channel_attention(joint)
        modulated, text_gains = self.text_modulation(
            attended,
            text_embedding,
        )
        evidence = self.evidence_projection(modulated)

        local_result = self.local_aggregation(
            ct,
            evidence,
            return_aux=return_aux,
        )
        if return_aux:
            residual, aux = local_result
        else:
            residual = local_result

        # The only output skip is CT. Neither raw auxiliary nor projected
        # evidence has an additive bypass to the decoder.
        fused = ct + residual
        if not return_aux:
            return fused

        aux = dict(aux)
        aux["channel_weights"] = channel_weights.flatten(1).detach()
        aux["text_gains"] = text_gains.flatten(1).detach()
        aux["residual_abs_mean"] = residual.detach().abs().mean().view(1)
        return fused, aux


class TextGuidedCTAnchorFusion(nn.Module):
    """Four-scale fusion shared by Full real-PET and Missing MPPC-proxy routes."""

    SOURCE_TO_INDEX: Mapping[str, int] = {"full": 0, "missing": 1}

    def __init__(
        self,
        text_embeddings: Tensor,
        channels: Sequence[int] = (64, 128, 320, 512),
        local_dims: Sequence[int] = (16, 32, 64, 64),
        num_heads: Sequence[int] = (1, 2, 4, 4),
        *,
        ca_reduction: int = 8,
        num_points: int = 5,
        max_offset: float = 1.0,
        residual_init: float = 0.1,
        null_init_bias: float = 1.0,
        prompts: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__()
        self.channels = tuple(int(value) for value in channels)
        self.local_dims = tuple(int(value) for value in local_dims)
        self.num_heads = tuple(int(value) for value in num_heads)
        if not (
            len(self.channels)
            == len(self.local_dims)
            == len(self.num_heads)
            and len(self.channels) > 0
        ):
            raise ValueError(
                "channels, local_dims, and num_heads must have the same "
                "non-zero length"
            )
        if any(value <= 0 for value in self.channels):
            raise ValueError("All channel counts must be positive")

        embeddings = torch.as_tensor(text_embeddings).detach().float()
        if embeddings.ndim == 3 and embeddings.shape[1] == 1:
            embeddings = embeddings[:, 0]
        if embeddings.ndim != 2 or embeddings.shape[0] != 2:
            raise ValueError(
                "text_embeddings must be [2,D] in [full,missing] order, "
                f"got {tuple(embeddings.shape)}"
            )
        if embeddings.shape[1] <= 0:
            raise ValueError("text embedding dimension must be positive")
        if not torch.isfinite(embeddings).all():
            raise FloatingPointError("text_embeddings contain NaN or Inf")
        embeddings = F.normalize(embeddings, dim=-1)
        self.register_buffer(
            "source_text_embeddings",
            embeddings,
            persistent=True,
        )

        if prompts is None:
            prompts = {
                "full": DEFAULT_FULL_PROMPT,
                "missing": DEFAULT_MISSING_PROMPT,
            }
        self.prompts = {
            "full": str(prompts.get("full", DEFAULT_FULL_PROMPT)),
            "missing": str(prompts.get("missing", DEFAULT_MISSING_PROMPT)),
        }

        text_dim = int(embeddings.shape[1])
        self.scales = nn.ModuleList(
            [
                TextGuidedCTAnchorFusionScale(
                    channels=channel_count,
                    text_dim=text_dim,
                    local_dim=local_dim,
                    num_heads=head_count,
                    ca_reduction=ca_reduction,
                    num_points=num_points,
                    max_offset=max_offset,
                    residual_init=residual_init,
                    null_init_bias=null_init_bias,
                )
                for channel_count, local_dim, head_count in zip(
                    self.channels,
                    self.local_dims,
                    self.num_heads,
                )
            ]
        )

    @classmethod
    def from_local_biomedclip(
        cls,
        biomedclip_dir: Union[str, os.PathLike],
        text_tower_dir: Optional[Union[str, os.PathLike]] = None,
        *,
        full_prompt: str = DEFAULT_FULL_PROMPT,
        missing_prompt: str = DEFAULT_MISSING_PROMPT,
        encode_device: Union[str, torch.device] = "cpu",
        **fusion_kwargs: Any,
    ) -> "TextGuidedCTAnchorFusion":
        """Build the module after one-time local BioMedCLIP prompt encoding."""

        embeddings = encode_prompts_with_local_biomedclip(
            biomedclip_dir=biomedclip_dir,
            text_tower_dir=text_tower_dir,
            full_prompt=full_prompt,
            missing_prompt=missing_prompt,
            device=encode_device,
        )
        return cls(
            text_embeddings=embeddings,
            prompts={"full": full_prompt, "missing": missing_prompt},
            **fusion_kwargs,
        )

    @classmethod
    def from_embedding_file(
        cls,
        embedding_path: Union[str, os.PathLike],
        **fusion_kwargs: Any,
    ) -> "TextGuidedCTAnchorFusion":
        """Build the module from cached fixed BioMedCLIP prompt embeddings."""

        embeddings, prompts = load_prompt_embeddings(embedding_path)
        return cls(
            text_embeddings=embeddings,
            prompts=prompts,
            **fusion_kwargs,
        )

    def _source_embedding(
        self,
        mode: FusionMode,
        batch_size: int,
        reference: Tensor,
    ) -> Tensor:
        if mode not in self.SOURCE_TO_INDEX:
            raise ValueError(
                f"mode must be one of {tuple(self.SOURCE_TO_INDEX)}, got {mode!r}"
            )
        index = self.SOURCE_TO_INDEX[mode]
        embedding = self.source_text_embeddings[index]
        return embedding.to(
            device=reference.device,
            dtype=reference.dtype,
        ).unsqueeze(0).expand(batch_size, -1)

    def forward(
        self,
        ct_features: Sequence[Tensor],
        auxiliary_features: Sequence[Tensor],
        *,
        mode: FusionMode,
        return_aux: bool = False,
    ) -> Union[List[Tensor], Tuple[List[Tensor], Dict[str, Any]]]:
        """Fuse aligned CT features with real PET or MPPC proxy features.

        Args:
            ct_features: Four-scale aligned CT features.
            auxiliary_features: Real PET features for ``mode="full"`` or MPPC
                compensation features for ``mode="missing"``.
            mode: Batch-level source identity. It selects only the fixed prompt;
                both routes use the same trainable fusion parameters.
            return_aux: Return detached diagnostic maps for inspection.

        Returns:
            A feature list with the same shapes as ``ct_features``. Every scale
            is exactly ``CT + learned_local_residual``.
        """

        if not isinstance(ct_features, (list, tuple)):
            raise TypeError("ct_features must be a list or tuple")
        if not isinstance(auxiliary_features, (list, tuple)):
            raise TypeError("auxiliary_features must be a list or tuple")
        expected = len(self.scales)
        if len(ct_features) != expected or len(auxiliary_features) != expected:
            raise ValueError(
                f"Expected {expected} CT and auxiliary scales, got "
                f"{len(ct_features)} and {len(auxiliary_features)}"
            )

        outputs: List[Tensor] = []
        scale_aux: List[Dict[str, Tensor]] = []
        for scale_idx, (scale, ct, auxiliary) in enumerate(
            zip(self.scales, ct_features, auxiliary_features)
        ):
            if not torch.is_tensor(ct) or not torch.is_tensor(auxiliary):
                raise TypeError(f"Scale {scale_idx} inputs must be tensors")
            if ct.ndim != 4 or auxiliary.ndim != 4:
                raise ValueError(
                    f"Scale {scale_idx} inputs must be BCHW tensors"
                )
            if ct.shape[0] != auxiliary.shape[0]:
                raise ValueError(
                    f"Scale {scale_idx} batch sizes differ: "
                    f"{ct.shape[0]} vs {auxiliary.shape[0]}"
                )
            text_embedding = self._source_embedding(
                mode,
                batch_size=ct.shape[0],
                reference=ct,
            )
            result = scale(
                ct,
                auxiliary,
                text_embedding,
                return_aux=return_aux,
            )
            if return_aux:
                output, aux = result
                scale_aux.append(aux)
            else:
                output = result
            outputs.append(output)

        if not return_aux:
            return outputs
        return outputs, {
            "mode": mode,
            "prompt": self.prompts[mode],
            "scales": scale_aux,
        }

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
