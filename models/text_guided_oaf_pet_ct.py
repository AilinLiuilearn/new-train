"""Text-guided OAF for CT/PET feature fusion.

This is a task-adapted, standalone reimplementation of the Operation-based
Adaptive Fusion (OAF) idea from TITA (ICCV 2025):

    https://github.com/huxingyuabc/TITA

The original OAF creates HPF, ADD and MUL candidates for each source and uses
six sample-wise weights to aggregate them.  This version keeps that core and
adds a *non-affine* fixed-text condition: a frozen Real/Proxy PET text embedding
adds a learned bias only to the three PET routing logits.  It never scales or
shifts the PET feature itself.

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
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


REAL_PET_PROMPT = (
    "Calibrated real PET preserves patient-specific metabolic patterns, "
    "lesion heterogeneity, and fine details."
)
PROXY_PET_PROMPT = (
    "Calibrated proxy PET provides a smooth and conservative prior that "
    "preserves coarse lesion location and metabolic semantics."
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
    """Generate HPF, ADD and MUL candidates for one modality.

    This follows the official TITA OAF logic:
      HPF: X - spatially_variant_low_pass(X)
      ADD: X + P_add(X)
      MUL: X * P_mul(X)

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
        for module in (
            self.compressor,
            self.add_operand_generator,
            self.mul_operand_generator,
        ):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        # The official implementation uses a small initialization for the
        # adaptive high-pass kernel generator.
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

        highpass = x - low_frequency
        addition = x + self.add_operand_generator(hidden)
        multiplication = x * self.mul_operand_generator(hidden)
        return highpass, addition, multiplication


class TextPETRouteBias(nn.Module):
    """Map a frozen text vector to biases for PET's HPF/ADD/MUL logits.

    The final layer is zero-initialized.  Therefore, before learning, this
    module is exactly neutral and TextGuidedOAF reduces to image-only OAF.
    """

    def __init__(self, text_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if text_dim <= 0 or hidden_dim <= 0:
            raise ValueError("text_dim and hidden_dim must be positive")
        self.net = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, text_embedding: Tensor) -> Tensor:
        return self.net(text_embedding)


PetState = Union[str, Sequence[str], Tensor]


class TextGuidedOAF(nn.Module):
    """Fuse CT and calibrated real/proxy PET features into one feature map.

    Branch order in diagnostics:
        modality dimension: 0=CT, 1=PET
        operation dimension: 0=HPF, 1=ADD, 2=MUL
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
        self.text_router = TextPETRouteBias(
            text_dim=normalized_text.shape[1], hidden_dim=channels
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

        ct_candidates = self.ct_operations(ct_feature)
        pet_candidates = self.pet_operations(pet_feature)

        image_logits = self.image_router(torch.cat((ct_feature, pet_feature), dim=1))
        selected_text = self.text_embeddings.index_select(0, state_ids)
        text_bias = self.text_router(selected_text).to(dtype=image_logits.dtype)

        # Text changes only PET operation selection; CT logits remain image-driven.
        routing_logits = torch.cat(
            (image_logits[:, :3], image_logits[:, 3:] + text_bias), dim=1
        )
        routing_weights = F.softmax(routing_logits, dim=1)

        fused = torch.zeros_like(ct_feature)
        for operation_index, (ct_candidate, pet_candidate) in enumerate(
            zip(ct_candidates, pet_candidates)
        ):
            ct_weight = routing_weights[:, operation_index].view(-1, 1, 1, 1)
            pet_weight = routing_weights[:, operation_index + 3].view(-1, 1, 1, 1)
            fused = fused + ct_weight * ct_candidate + pet_weight * pet_candidate

        if not return_diagnostics:
            return fused

        diagnostics = {
            "routing_weights": routing_weights.reshape(batch_size, 2, 3),
            "image_logits": image_logits,
            "pet_text_bias": text_bias,
            "pet_state_ids": state_ids,
        }
        return fused, diagnostics


@torch.no_grad()
def export_fixed_text_bank(
    model_name_or_path: str,
    output_path: Union[str, Path],
    device: str = "cpu",
) -> Path:
    """Encode the two prompts once with a frozen Hugging Face text model.

    This function is an offline utility.  The resulting ``.pt`` file is loaded
    by ``TextGuidedOAF.from_text_bank``; the language model is not present in
    the segmentation training graph.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Install transformers only for offline prompt encoding: "
            "pip install transformers"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    text_model = AutoModel.from_pretrained(model_name_or_path).to(device).eval()
    tokens = tokenizer(
        list(PET_PROMPTS), padding=True, truncation=True, return_tensors="pt"
    )
    tokens = {name: value.to(device) for name, value in tokens.items()}
    outputs = text_model(**tokens)

    if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
        embeddings = outputs.text_embeds
    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        embeddings = outputs.pooler_output
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

    fused.square().mean().backward()
    assert ct.grad is not None and torch.isfinite(ct.grad).all()
    assert pet.grad is not None and torch.isfinite(pet.grad).all()
    assert module.text_router.net[-1].weight.grad is not None

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
        default="emilyalsentzer/Bio_ClinicalBERT",
        help="Hugging Face text model used only with --export-text-bank",
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
