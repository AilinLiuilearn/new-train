"""Text-guided OAF for CT/PET feature fusion.

This is a task-adapted, standalone reimplementation of the Operation-based
Adaptive Fusion (OAF) idea from TITA (ICCV 2025):

    https://github.com/huxingyuabc/TITA

The original OAF creates HPF, ADD and MUL candidates for each source and uses
six sample-wise weights to aggregate them.  This version keeps that core and
adds:

1. Identity-centered HPF/ADD/MUL candidates (operation-internal residuals);
2. A *non-affine* fixed-text PET channel residual multiplication applied
   before PET operations:
   ``PET_out = PET * (1 + tanh(MLP(text)))``.

The final fused feature is the six-way Softmax weighted sum of candidates.
There is no outer ``CT + PET + OAF`` residual and no feature-text similarity.

Input:
    ct_feature:  [B, C, H, W]
    pet_feature: [B, C, H, W]
    pet_state:   "real", "proxy", a sequence of those strings, or a [B]
                 integer tensor (0=real, 1=proxy)

Output:
    fused_feature: [B, C, H, W]

Only PyTorch is required for training/inference.  ``transformers`` is optional
and is used only once to export the two fixed text embeddings offline.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
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


REAL_PET_PROMPT = (
    "Real PET preserves patient-specific metabolic details "
    "and heterogeneous lesion uptake."
)
PROXY_PET_PROMPT = (
    "Prototype-compensated PET provides a smooth metabolic "
    "prior with coarse lesion localization."
)
PET_PROMPTS: Tuple[str, str] = (REAL_PET_PROMPT, PROXY_PET_PROMPT)
PET_STATE_TO_ID: Mapping[str, int] = {
    "real": 0,
    "full": 0,
    "proxy": 1,
    "missing": 1,
    "compensated": 1,
}


def _hamming_2d(kernel_size: int) -> Tensor:
    """Return a flattened 2-D Hamming window with shape [1, K*K, 1, 1]."""
    window_1d = torch.hamming_window(kernel_size, periodic=False)
    window_2d = torch.outer(window_1d, window_1d)
    return window_2d.reshape(1, kernel_size * kernel_size, 1, 1)


class OAFSourceBranch(nn.Module):
    """Generate identity-centered HPF, ADD and MUL candidates for one modality.

      HPF: X + (X - spatially_variant_low_pass(X))
      ADD: X + P_add(X)
      MUL: X * (1 + tanh(P_mul(X)))

    The spatially variant filtering is implemented with K*K shifted slices
    instead of a full ``unfold`` tensor.  This preserves the operation while
    avoiding a [B, C*K*K, H*W] allocation at high-resolution feature stages.
    """

    def __init__(
        self,
        channels: int,
        compressed_channels: Optional[int] = None,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        hidden = channels if compressed_channels is None else compressed_channels
        if hidden <= 0:
            raise ValueError("compressed_channels must be positive")

        self.channels = channels
        self.kernel_size = kernel_size
        self.compressor = nn.Conv2d(channels, hidden, kernel_size=1)

        # One normalized KxK low-pass kernel per spatial location.
        self.highpass_kernel_generator = nn.Conv2d(
            hidden, kernel_size * kernel_size, kernel_size=3, padding=1
        )
        # Unlike the original public file, output channels are explicitly C,
        # so compressed_channels may safely differ from channels.
        self.add_operand_generator = nn.Conv2d(
            hidden, channels, kernel_size=3, padding=1
        )
        self.mul_operand_generator = nn.Conv2d(
            hidden, channels, kernel_size=3, padding=1
        )

        self.register_buffer("hamming_window", _hamming_2d(kernel_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.compressor.weight)
        if self.compressor.bias is not None:
            nn.init.zeros_(self.compressor.bias)

        # ADD starts as identity: addition = x + 0
        nn.init.zeros_(self.add_operand_generator.weight)
        if self.add_operand_generator.bias is not None:
            nn.init.zeros_(self.add_operand_generator.bias)

        # MUL starts as identity: multiplication = x * (1 + tanh(0)) = x
        nn.init.zeros_(self.mul_operand_generator.weight)
        if self.mul_operand_generator.bias is not None:
            nn.init.zeros_(self.mul_operand_generator.bias)

        nn.init.normal_(self.highpass_kernel_generator.weight, mean=0.0, std=1e-3)
        if self.highpass_kernel_generator.bias is not None:
            nn.init.zeros_(self.highpass_kernel_generator.bias)

    def _normalized_spatial_kernels(self, hidden: Tensor) -> Tensor:
        raw_kernel = self.highpass_kernel_generator(hidden)
        kernel = F.softmax(raw_kernel, dim=1)
        window = self.hamming_window.to(dtype=kernel.dtype)
        kernel = kernel * window
        return kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _spatially_variant_lowpass(self, x: Tensor, kernel: Tensor) -> Tensor:
        """Apply per-pixel KxK kernels without materializing a large unfold."""
        _, _, height, width = x.shape
        radius = self.kernel_size // 2
        pad_mode = "reflect" if height > radius and width > radius else "replicate"
        padded = F.pad(x, (radius, radius, radius, radius), mode=pad_mode)

        output = torch.zeros_like(x)
        kernel_index = 0
        for row in range(self.kernel_size):
            for col in range(self.kernel_size):
                shifted = padded[:, :, row : row + height, col : col + width]
                output = output + shifted * kernel[:, kernel_index : kernel_index + 1]
                kernel_index += 1
        return output

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected [B,{self.channels},H,W], got {tuple(x.shape)}"
            )

        hidden = self.compressor(x)
        spatial_kernel = self._normalized_spatial_kernels(hidden)
        low_frequency = self._spatially_variant_lowpass(x, spatial_kernel)

        highpass_delta = x - low_frequency
        highpass = x + highpass_delta

        add_delta = self.add_operand_generator(hidden)
        addition = x + add_delta

        mul_delta = torch.tanh(self.mul_operand_generator(hidden))
        multiplication = x * (1.0 + mul_delta)
        return highpass, addition, multiplication


class TextPETResidualModulation(nn.Module):
    """Modulate PET channels with a fixed global text embedding.

    The modulation is multiplicative residual modulation:

        PET_out = PET + PET * channel_delta
                = PET * (1 + channel_delta)

    No feature-text similarity, additive PET bias, spatial attention,
    dynamic convolution or reliability estimation is used.
    """

    def __init__(self, channels: int, text_dim: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if text_dim <= 0:
            raise ValueError(f"text_dim must be positive, got {text_dim}")

        hidden_dim = min(int(channels), 32)
        self.channels = int(channels)
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)

        self.text_to_channel = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, channels, bias=True),
        )

        # Identity initialization: channel_delta=0 => modulated_pet=pet.
        nn.init.zeros_(self.text_to_channel[-1].weight)
        if self.text_to_channel[-1].bias is not None:
            nn.init.zeros_(self.text_to_channel[-1].bias)

    def forward(
        self,
        pet_feature: Tensor,
        text_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if pet_feature.ndim != 4:
            raise ValueError(
                "pet_feature must have shape [B,C,H,W], "
                f"got {tuple(pet_feature.shape)}"
            )
        if pet_feature.shape[1] != self.channels:
            raise ValueError(
                f"Expected PET channels={self.channels}, got {pet_feature.shape[1]}"
            )
        if text_embedding.ndim != 2:
            raise ValueError(
                "text_embedding must have shape [B,text_dim], "
                f"got {tuple(text_embedding.shape)}"
            )
        if text_embedding.shape[0] != pet_feature.shape[0]:
            raise ValueError("PET batch size and text batch size must match")

        text_embedding = text_embedding.to(
            device=pet_feature.device,
            dtype=self.text_to_channel[-1].weight.dtype,
        )
        channel_delta = torch.tanh(self.text_to_channel(text_embedding))
        channel_delta = channel_delta.to(
            device=pet_feature.device,
            dtype=pet_feature.dtype,
        ).view(pet_feature.shape[0], self.channels, 1, 1)

        modulated_pet = pet_feature + pet_feature * channel_delta
        return modulated_pet, channel_delta


PetState = Union[str, Sequence[str], Tensor]


class TextGuidedOAF(nn.Module):
    """Fuse CT and calibrated real/proxy PET features into one feature map.

    Branch order in diagnostics:
        modality dimension: 0=CT, 1=PET
        operation dimension: 0=HPF, 1=ADD, 2=MUL

    Final output is the six-way Softmax weighted sum of identity-centered
    operation candidates (no outer CT+PET residual).
    """

    def __init__(
        self,
        channels: int,
        text_embeddings: Tensor,
        compressed_channels: Optional[int] = None,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if text_embeddings.ndim != 2 or text_embeddings.shape[0] != 2:
            raise ValueError(
                "text_embeddings must have shape [2, text_dim] in the order "
                "[real PET, proxy PET]"
            )
        if not torch.is_floating_point(text_embeddings):
            text_embeddings = text_embeddings.float()

        self.channels = channels
        self.ct_operations = OAFSourceBranch(
            channels, compressed_channels, kernel_size
        )
        self.pet_operations = OAFSourceBranch(
            channels, compressed_channels, kernel_size
        )

        # Official TITA: GAP(cat(X1, X2)) -> MLP -> six logits -> Softmax.
        self.image_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            nn.Linear(channels, 6),
        )

        normalized_text = F.normalize(text_embeddings.detach().float(), dim=-1)
        self.register_buffer("text_embeddings", normalized_text)
        self.text_modulator = TextPETResidualModulation(
            channels=channels,
            text_dim=normalized_text.shape[1],
        )

    @classmethod
    def from_text_bank(
        cls,
        channels: int,
        text_bank_path: Union[str, Path],
        compressed_channels: Optional[int] = None,
        kernel_size: int = 3,
    ) -> "TextGuidedOAF":
        payload = torch.load(text_bank_path, map_location="cpu")
        if not isinstance(payload, dict) or "embeddings" not in payload:
            raise ValueError("Text bank must be a dict containing 'embeddings'")
        return cls(
            channels=channels,
            text_embeddings=payload["embeddings"],
            compressed_channels=compressed_channels,
            kernel_size=kernel_size,
        )

    @staticmethod
    def _state_ids(pet_state: PetState, batch_size: int, device: torch.device) -> Tensor:
        if isinstance(pet_state, str):
            states: Sequence[str] = [pet_state] * batch_size
            try:
                ids = [PET_STATE_TO_ID[state.lower()] for state in states]
            except KeyError as error:
                raise ValueError(f"Unknown PET state: {error.args[0]}") from error
            return torch.tensor(ids, device=device, dtype=torch.long)

        if isinstance(pet_state, Tensor):
            ids = pet_state.to(device=device, dtype=torch.long).reshape(-1)
            if ids.numel() == 1 and batch_size > 1:
                ids = ids.expand(batch_size)
            if ids.numel() != batch_size:
                raise ValueError(
                    f"pet_state has {ids.numel()} ids, but batch size is {batch_size}"
                )
            if not torch.all((ids == 0) | (ids == 1)):
                raise ValueError("Tensor PET states must contain only 0 (real) or 1 (proxy)")
            return ids

        if len(pet_state) != batch_size:
            raise ValueError(
                f"Received {len(pet_state)} PET states for batch size {batch_size}"
            )
        try:
            ids = [PET_STATE_TO_ID[state.lower()] for state in pet_state]
        except KeyError as error:
            raise ValueError(f"Unknown PET state: {error.args[0]}") from error
        return torch.tensor(ids, device=device, dtype=torch.long)

    def forward(
        self,
        ct_feature: Tensor,
        pet_feature: Tensor,
        pet_state: PetState,
        return_diagnostics: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        if ct_feature.shape != pet_feature.shape:
            raise ValueError(
                "CT and PET feature shapes must match, got "
                f"{tuple(ct_feature.shape)} and {tuple(pet_feature.shape)}"
            )
        if ct_feature.ndim != 4 or ct_feature.shape[1] != self.channels:
            raise ValueError(
                f"Expected CT/PET [B,{self.channels},H,W], got "
                f"{tuple(ct_feature.shape)}"
            )

        batch_size = ct_feature.shape[0]
        state_ids = self._state_ids(pet_state, batch_size, ct_feature.device)

        selected_text = self.text_embeddings.index_select(0, state_ids)
        pet_modulated, text_channel_delta = self.text_modulator(
            pet_feature=pet_feature,
            text_embedding=selected_text,
        )

        ct_candidates = self.ct_operations(ct_feature)
        pet_candidates = self.pet_operations(pet_modulated)

        image_logits = self.image_router(
            torch.cat((ct_feature, pet_modulated), dim=1)
        )
        routing_weights = F.softmax(image_logits, dim=1)

        fused_feature = torch.zeros_like(ct_feature)
        for operation_index, (ct_candidate, pet_candidate) in enumerate(
            zip(ct_candidates, pet_candidates)
        ):
            ct_weight = routing_weights[:, operation_index].view(-1, 1, 1, 1)
            pet_weight = routing_weights[:, operation_index + 3].view(-1, 1, 1, 1)
            fused_feature = (
                fused_feature + ct_weight * ct_candidate + pet_weight * pet_candidate
            )

        if not return_diagnostics:
            return fused_feature

        route_matrix = routing_weights.reshape(batch_size, 2, 3)
        modality_weights = route_matrix.sum(dim=2)
        ct_operation_weights = route_matrix[:, 0] / modality_weights[:, 0:1].clamp_min(
            1e-6
        )
        pet_operation_weights = route_matrix[:, 1] / modality_weights[:, 1:2].clamp_min(
            1e-6
        )

        diagnostics = {
            "routing_weights": route_matrix,
            "image_logits": image_logits,
            "modality_weights": modality_weights,
            "ct_operation_weights": ct_operation_weights,
            "pet_operation_weights": pet_operation_weights,
            "text_channel_delta": text_channel_delta.detach(),
            "text_channel_delta_abs_mean": text_channel_delta.detach()
            .float()
            .abs()
            .mean(),
            "text_channel_delta_min": text_channel_delta.detach().float().min(),
            "text_channel_delta_max": text_channel_delta.detach().float().max(),
            "pet_state_ids": state_ids,
            "ct_rms": ct_feature.detach().float().square().mean().sqrt(),
            "pet_before_text_rms": pet_feature.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "pet_after_text_rms": pet_modulated.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "fused_rms": fused_feature.detach().float().square().mean().sqrt(),
        }
        return fused_feature, diagnostics


class MultiScaleTextGuidedOAF(nn.Module):
    """Apply TextGuidedOAF independently at every encoder scale.

    Compatible with the existing baseline fusion call signature:
        fusion(ct_feats, pet_feats, mode="full"|"missing"|"auto", pet_available=...)
    """

    def __init__(
        self,
        channels: Sequence[int],
        text_embeddings: Tensor,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if len(channels) == 0:
            raise ValueError("channels must be a non-empty sequence")
        self.channels = list(channels)
        self.compressed_channels = [
            max(int(c) // 4, 32)
            for c in self.channels
        ]
        self.blocks = nn.ModuleList(
            [
                TextGuidedOAF(
                    channels=int(c),
                    text_embeddings=text_embeddings,
                    compressed_channels=int(hidden_c),
                    kernel_size=kernel_size,
                )
                for c, hidden_c in zip(
                    self.channels,
                    self.compressed_channels,
                )
            ]
        )

    def _resolve_state_ids(
        self,
        mode: str,
        batch_size: int,
        device: torch.device,
        pet_available: Optional[Tensor],
    ) -> Tensor:
        if mode == "full":
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if mode == "missing":
            return torch.ones(batch_size, dtype=torch.long, device=device)
        if mode == "auto":
            if pet_available is None:
                raise ValueError("pet_available is required for auto fusion")
            # Project convention: pet_available=1 -> real PET, 0 -> proxy PET.
            # TextGuidedOAF convention: 0=real, 1=proxy (inverted).
            state_ids = 1 - pet_available.to(device=device, dtype=torch.long).view(-1)
            if state_ids.numel() != batch_size:
                raise ValueError(
                    f"pet_available has {state_ids.numel()} values, "
                    f"but batch size is {batch_size}"
                )
            if not torch.all((state_ids == 0) | (state_ids == 1)):
                raise ValueError(
                    "auto fusion state_ids must contain only 0 (real) or 1 (proxy)"
                )
            return state_ids
        raise ValueError(f"Unsupported fusion mode: {mode!r}")

    def forward(
        self,
        ct_feats: Sequence[Tensor],
        pet_feats: Sequence[Tensor],
        mode: str,
        pet_available: Optional[Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Union[List[Tensor], Tuple[List[Tensor], Dict[str, object]]]:
        if not (len(ct_feats) == len(pet_feats) == len(self.blocks)):
            raise ValueError(
                "Expected matching scale counts, got "
                f"ct={len(ct_feats)}, pet={len(pet_feats)}, blocks={len(self.blocks)}"
            )

        batch_size = ct_feats[0].shape[0]
        device = ct_feats[0].device
        state_ids = self._resolve_state_ids(mode, batch_size, device, pet_available)

        fused_feats: List[Tensor] = []
        routing_weights: List[Tensor] = []
        modality_weights: List[Tensor] = []
        ct_operation_weights: List[Tensor] = []
        pet_operation_weights: List[Tensor] = []
        text_channel_delta: List[Tensor] = []
        text_channel_delta_abs_mean: List[Tensor] = []
        pet_before_text_rms: List[Tensor] = []
        pet_after_text_rms: List[Tensor] = []
        fused_rms: List[Tensor] = []

        for block, ct_feat, pet_feat in zip(self.blocks, ct_feats, pet_feats):
            if pet_feat.shape[1] != ct_feat.shape[1]:
                raise ValueError(
                    "CT/PET channel mismatch at fusion: "
                    f"ct={tuple(ct_feat.shape)}, pet={tuple(pet_feat.shape)}"
                )
            if pet_feat.shape[-2:] != ct_feat.shape[-2:]:
                pet_feat = F.interpolate(
                    pet_feat,
                    size=ct_feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            if return_diagnostics:
                fused, diagnostics = block(
                    ct_feat,
                    pet_feat,
                    pet_state=state_ids,
                    return_diagnostics=True,
                )
                routing_weights.append(diagnostics["routing_weights"])
                modality_weights.append(diagnostics["modality_weights"])
                ct_operation_weights.append(diagnostics["ct_operation_weights"])
                pet_operation_weights.append(diagnostics["pet_operation_weights"])
                text_channel_delta.append(diagnostics["text_channel_delta"])
                text_channel_delta_abs_mean.append(
                    diagnostics["text_channel_delta_abs_mean"]
                )
                pet_before_text_rms.append(diagnostics["pet_before_text_rms"])
                pet_after_text_rms.append(diagnostics["pet_after_text_rms"])
                fused_rms.append(diagnostics["fused_rms"])
            else:
                fused = block(ct_feat, pet_feat, pet_state=state_ids)
            fused_feats.append(fused)

        if not return_diagnostics:
            return fused_feats

        return fused_feats, {
            "routing_weights": routing_weights,
            "modality_weights": modality_weights,
            "ct_operation_weights": ct_operation_weights,
            "pet_operation_weights": pet_operation_weights,
            "text_channel_delta": text_channel_delta,
            "text_channel_delta_abs_mean": text_channel_delta_abs_mean,
            "pet_before_text_rms": pet_before_text_rms,
            "pet_after_text_rms": pet_after_text_rms,
            "fused_rms": fused_rms,
            "pet_state_ids": state_ids,
        }


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
    biomedclip_model_path: str,
    biomedbert_text_tower_path: str,
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
    print("[TextOAF] offline=True")
    print(f"[TextOAF] source={source}")
    print(f"[TextOAF] embedding_shape={tuple(embeddings.shape)}")
    print("[TextOAF] state_order=['real', 'proxy']")
    print("[TextOAF] text_encoder_retained=False")
    return embeddings


@torch.no_grad()
def export_fixed_text_bank(
    model_name_or_path: str,
    output_path: Union[str, Path],
    device: str = "cpu",
) -> Path:
    """Encode the two prompts once with a frozen local Hugging Face text model.

    This function is an offline utility.  The resulting ``.pt`` file is loaded
    by ``TextGuidedOAF.from_text_bank``; the language model is not present in
    the segmentation training graph.  ``model_name_or_path`` must be a local
    directory; remote Hugging Face model ids are rejected.
    """
    if not os.path.isdir(model_name_or_path):
        raise ValueError(
            "export_fixed_text_bank requires a local directory path; "
            f"got {model_name_or_path!r}. Remote model names are not allowed."
        )

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Install transformers only for offline prompt encoding: "
            "pip install transformers"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, local_files_only=True
    )
    text_model = (
        AutoModel.from_pretrained(model_name_or_path, local_files_only=True)
        .to(device)
        .eval()
    )
    text_model.requires_grad_(False)
    tokens = tokenizer(
        list(PET_PROMPTS), padding=True, truncation=True, return_tensors="pt"
    )
    tokens = {name: value.to(device) for name, value in tokens.items()}
    with torch.no_grad():
        outputs = text_model(**tokens)

        if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            embeddings = outputs.text_embeds
        else:
            hidden = outputs.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    payload = {
        "embeddings": F.normalize(embeddings.float(), dim=-1).cpu(),
        "prompts": list(PET_PROMPTS),
        "model_name_or_path": model_name_or_path,
        "state_order": ["real", "proxy"],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    del tokenizer
    del text_model
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_path


def _demo_text_embeddings(text_dim: int = 32) -> Tensor:
    """Deterministic placeholders used only by the local shape/gradient test."""
    if text_dim < 2:
        raise ValueError("demo text_dim must be at least 2")
    embeddings = torch.zeros(2, text_dim)
    embeddings[0, 0] = 1.0
    embeddings[1, 1] = 1.0
    return embeddings


def smoke_test() -> None:
    """Check shapes, normalized routing and backward propagation."""
    torch.manual_seed(7)
    batch, channels, height, width = 4, 32, 24, 24
    module = TextGuidedOAF(
        channels=channels,
        text_embeddings=_demo_text_embeddings(),
        compressed_channels=16,
    )
    ct = torch.randn(batch, channels, height, width, requires_grad=True)
    pet = torch.randn(batch, channels, height, width, requires_grad=True)
    states = ["real", "proxy", "full", "missing"]

    fused, diagnostics = module(ct, pet, states, return_diagnostics=True)
    weights = diagnostics["routing_weights"]
    assert fused.shape == ct.shape
    assert weights.shape == (batch, 2, 3)
    assert torch.allclose(
        weights.reshape(batch, 6).sum(dim=1),
        torch.ones(batch),
        atol=1e-6,
    )
    # Text identity at init.
    pet_mod, delta = module.text_modulator(pet, module.text_embeddings.index_select(
        0, diagnostics["pet_state_ids"]
    ))
    torch.testing.assert_close(delta, torch.zeros_like(delta))
    torch.testing.assert_close(pet_mod, pet)

    fused.square().mean().backward()
    assert ct.grad is not None and torch.isfinite(ct.grad).all()
    assert pet.grad is not None and torch.isfinite(pet.grad).all()
    assert module.text_modulator.text_to_channel[-1].weight.grad is not None

    print("Smoke test passed")
    print("fused shape:", tuple(fused.shape))
    print("weight shape [B, modality, operation]:", tuple(weights.shape))
    print("operation order: HPF, ADD, MUL")
    print("sample-0 CT weights:", weights[0, 0].detach().tolist())
    print("sample-0 PET weights:", weights[0, 1].detach().tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test", action="store_true", help="run the standalone module test"
    )
    parser.add_argument(
        "--export-text-bank",
        type=Path,
        help="output .pt path for offline fixed-text embeddings",
    )
    parser.add_argument(
        "--text-model",
        type=str,
        default=BIOMEDBERT_TEXT_TOWER_PATH,
        help="Local text-model directory used only with --export-text-bank",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.export_text_bank is not None:
        path = export_fixed_text_bank(args.text_model, args.export_text_bank, args.device)
        print(f"Saved fixed text bank to {path}")
        return

    # Running without arguments is intentionally useful.
    smoke_test()


if __name__ == "__main__":
    main()
