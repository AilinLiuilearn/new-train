"""Standalone Full CT/PET fusion; put this file under models/.

Requires PyTorch >= 2.1. Loading offline CLIP additionally requires transformers.
No repository-specific imports, online downloads, missing-modality inference,
auxiliary losses, final CT residual, or trainable residual scaling.

Example (construct BEFORE optimizer/DDP, then move with the parent model):
    fusion = FullPETCTAsymmetricFusion(
        clip_path="pretrained/clip-vit-base-patch32")
    fused = fusion(ct_features, pet_features)  # four shallow-to-deep tensors

Local CLIP directory must contain Hugging Face config, weights and tokenizer
files. A bare OpenAI .pt checkpoint is not this format. Frozen CLIP is run once
on CPU in eval/no_grad; only its two float32 pooler embeddings are retained as
persistent buffers. Text projections remain trainable. No CLIP call in forward.
The prompts and architecture settings are stored in checkpoint extra state.
To reconstruct without the original CLIP directory:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = FullPETCTAsymmetricFusion(text_embeddings=state["text_embeddings"])
    model.load_state_dict(state)  # use the same architecture constructor options

Verification:
    python full_petct_asymmetric_fusion.py --self-test
    python full_petct_asymmetric_fusion.py --smoke --device cpu
    python full_petct_asymmetric_fusion.py --smoke --device cuda --batch-size 16 \
        --amp --checkpoint-attention --clip-path pretrained/clip-vit-base-patch32
Smoke uses actual four-scale sizes and backward. Its memory excludes backbones,
decoder, optimizer states and their activations: it is NOT a full-model OOM test.

Source inspirations (adaptations, not reproductions):
DGNet model/dgnet/pwm.py: image/text multiplication and pooled+text channel gate.
WFANet net_torch.py: global attention on reduced spatial grids. Here pooling/
interpolation replace wavelet transforms; there are no frequency branches.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Sequence
import unittest
from unittest.mock import patch

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ["FullPETCTAsymmetricFusion"]

CT_PROMPT = "A CT image showing the anatomical structure and boundaries of lung tumors."
PET_PROMPT = "A PET image showing bright tumor regions in the lungs."


def _offline_clip_embeddings(path: str | Path, prompts: Sequence[str]) -> Tensor:
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Offline Hugging Face CLIP directory not found: {path}")
    try:
        from transformers import AutoTokenizer, CLIPTextModel
    except ImportError as exc:
        raise ImportError("Install transformers to encode local CLIP prompts, or supply text_embeddings.") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    encoder = CLIPTextModel.from_pretrained(str(path), local_files_only=True)
    encoder.to(device="cpu", dtype=torch.float32).requires_grad_(False).eval()
    tokens = tokenizer(list(prompts), padding=True, truncation=True,
                       max_length=encoder.config.max_position_embeddings,
                       return_tensors="pt")
    with torch.no_grad():
        # CLIP text pooler output, NOT projected/normalized get_text_features.
        embeddings = encoder(**tokens).pooler_output.detach().float().cpu().clone()
    return embeddings


class ChannelLayerNorm(nn.Module):
    """Per spatial location, normalize channels only; BCHW in/out."""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class TextChannelModulation(nn.Module):
    def __init__(self, channels: int, text_dim: int, pool: str, use_text: bool):
        super().__init__()
        self.pool = pool
        self.use_text = use_text
        if use_text:
            self.text_proj = nn.Sequential(nn.Linear(text_dim, channels),
                                           nn.LayerNorm(channels), nn.GELU())
        self.visual_conv = nn.Conv2d(channels, channels, 3, padding=1)
        hidden = max(channels // 16, 4)
        # Same MLP for visual descriptor and text within one modality.
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU(),
                                 nn.Conv2d(hidden, channels, 1))

    def forward(self, x: Tensor, text: Tensor | None) -> Tensor:
        if self.use_text:
            if text is None:
                raise ValueError("Text-enabled modulation requires cached embeddings.")
            t = self.text_proj(text).reshape(1, -1, 1, 1)
            u = self.visual_conv(x * t)
        else:
            u = self.visual_conv(x)
        descriptor = (F.adaptive_avg_pool2d(u, 1) if self.pool == "avg"
                      else F.adaptive_max_pool2d(u, 1))
        logits = self.mlp(descriptor)
        if self.use_text:
            logits = logits + self.mlp(t)
        return x * (1.0 + torch.tanh(logits))


class CTMultiScaleCorrection(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.in_proj = nn.Conv2d(channels, channels, 1)
        self.branches = nn.ModuleList([
            nn.Conv2d(channels, channels, k, padding=k // 2, groups=channels)
            for k in (3, 5, 7)])
        self.out_proj = nn.Conv2d(3 * channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.gelu(self.in_proj(x))
        return self.out_proj(F.gelu(torch.cat([branch(x) for branch in self.branches], dim=1)))


class PETGlobalCorrection(nn.Module):
    def __init__(self, channels: int, dim: int, heads: int,
                 grid_cap: int, checkpoint_attention: bool):
        super().__init__()
        self.heads, self.dim, self.grid_cap = heads, dim, grid_cap
        self.checkpoint_attention = checkpoint_attention
        self.in_proj = nn.Conv2d(channels, dim, 1)
        self.position_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm = ChannelLayerNorm(dim)
        self.qkv = nn.Conv2d(dim, 3 * dim, 1)
        self.attn_out = nn.Conv2d(dim, dim, 1)
        self.out_proj = nn.Conv2d(dim, channels, 1)

    def _attention(self, x: Tensor) -> Tensor:
        b, _, h, w = x.shape
        x = x + self.position_conv(x)
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, self.dim // self.heads, h * w)
        q, k, v = qkv.permute(1, 0, 2, 4, 3).unbind(0)
        # No attention weights returned, no manual N*N tensor retained.
        # PyTorch chooses the available backend; fused CUDA is not guaranteed.
        y = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(),
                                           dropout_p=0.0, is_causal=False)
        y = y.transpose(-2, -1).reshape(b, self.dim, h, w)
        return self.attn_out(y)

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        target = (min(h, self.grid_cap), min(w, self.grid_cap))
        if target != (h, w):
            x = F.adaptive_avg_pool2d(x, target)
        x = self.in_proj(x)
        if self.checkpoint_attention and self.training and torch.is_grad_enabled():
            y = checkpoint(self._attention, x, use_reentrant=False)
        else:
            y = self._attention(x)
        if target != (h, w):
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        return self.out_proj(y)


class JointSpatialFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.ct_proj = nn.Conv2d(channels, channels, 1)
        self.pet_proj = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Conv2d(4, 2, 3, padding=1)
        self.out_proj = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, ct: Tensor, pet: Tensor) -> Tensor:
        c, p = self.ct_proj(ct), self.pet_proj(pet)
        stats = torch.cat((c.mean(1, keepdim=True), c.amax(1, keepdim=True),
                           p.mean(1, keepdim=True), p.amax(1, keepdim=True)), dim=1)
        wc, wp = self.gate(stats).sigmoid().chunk(2, dim=1)
        # Independent sigmoid gates; no softmax competition and NO final CT skip.
        return self.out_proj(torch.cat((wc * c, wp * p), dim=1))


class FusionScale(nn.Module):
    def __init__(self, channels: int, text_dim: int, pet_dim: int, heads: int,
                 grid_cap: int, use_text: bool, checkpoint_attention: bool):
        super().__init__()
        self.ct_mod = TextChannelModulation(channels, text_dim, "avg", use_text)
        self.pet_mod = TextChannelModulation(channels, text_dim, "max", use_text)
        self.ct_correction = CTMultiScaleCorrection(channels)
        self.pet_correction = PETGlobalCorrection(channels, pet_dim, heads, grid_cap,
                                                 checkpoint_attention)
        self.fusion = JointSpatialFusion(channels)

    def forward(self, ct: Tensor, pet: Tensor, text: Tensor | None) -> Tensor:
        ct_mod = self.ct_mod(ct, None if text is None else text[0])
        pet_mod = self.pet_mod(pet, None if text is None else text[1])
        ct_out = ct + self.ct_correction(ct_mod)
        pet_out = pet + self.pet_correction(pet_mod)
        return self.fusion(ct_out, pet_out)


class FullPETCTAsymmetricFusion(nn.Module):
    """Full-only, shallow-to-deep four-scale fusion.

    Inputs are sequences of BCHW floating tensors. Corresponding CT/PET must
    have identical shape, dtype and device; all scales share batch/device/dtype.
    Spatial dimensions may differ from the documented default pyramid.
    Missing requests are rejected before any branch computation.
    Custom cached embeddings must be [2, text_dim], CT first, PET second.
    No exact identity/baseline equivalence is claimed at initialization.
    """
    def __init__(self, clip_path: str | Path | None = None, *,
                 channels: Sequence[int] = (64, 128, 320, 512),
                 pet_dims: Sequence[int] = (64, 128, 160, 256),
                 heads: int = 4, grid_cap: int = 32, text_dim: int = 512,
                 use_text: bool = True, text_embeddings: Tensor | None = None,
                 prompts: Sequence[str] = (CT_PROMPT, PET_PROMPT),
                 checkpoint_attention: bool = False):
        super().__init__()
        self.channels = tuple(channels)
        dims = tuple(pet_dims)
        if len(self.channels) != 4 or len(dims) != 4:
            raise ValueError("Exactly four scales are required, shallow-to-deep.")
        values = (*self.channels, *dims, heads, grid_cap, text_dim)
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in values):
            raise ValueError("Channels, dimensions, heads and grid_cap must be positive integers.")
        if any(d % heads for d in dims):
            raise ValueError("Each PET attention dimension must be divisible by heads.")
        if len(prompts) != 2 or any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError("Provide two nonempty prompts, CT then PET.")
        self.use_text = use_text
        self._contract = dict(version=1, channels=list(self.channels), pet_dims=list(dims),
                              heads=heads, grid_cap=grid_cap, text_dim=text_dim,
                              use_text=use_text, prompts=list(prompts))
        if use_text:
            if text_embeddings is None:
                if clip_path is None:
                    clip_path = Path(__file__).resolve().parent.parent / "pretrained" / "clip-vit-base-patch32"
                text_embeddings = _offline_clip_embeddings(clip_path, prompts)
            if not isinstance(text_embeddings, Tensor) or text_embeddings.shape != (2, text_dim):
                raise ValueError(f"Expected CLIP pooler embeddings [2,{text_dim}].")
            if not text_embeddings.is_floating_point() or not torch.isfinite(text_embeddings).all():
                raise ValueError("Text embeddings must be finite floating point values.")
            text_embeddings = text_embeddings.detach().to(device="cpu", dtype=torch.float32).clone()
        elif text_embeddings is not None:
            raise ValueError("Do not supply text_embeddings with use_text=False.")
        self.register_buffer("text_embeddings", text_embeddings, persistent=True)
        self.scales = nn.ModuleList([
            FusionScale(c, text_dim, d, heads, grid_cap, use_text, checkpoint_attention)
            for c, d in zip(self.channels, dims)])
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            if module.kernel_size == (1, 1):
                nn.init.xavier_uniform_(module.weight)
            else:
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_extra_state(self) -> dict:
        return dict(self._contract)

    def set_extra_state(self, state: dict) -> None:
        if state != self._contract:
            raise RuntimeError("Fusion checkpoint contract differs from constructor settings/prompts.")

    def forward(self, ct_features: Sequence[Tensor], pet_features: Sequence[Tensor],
                *, state: str = "full") -> list[Tensor]:
        if state != "full":
            raise ValueError("This module supports only state='full' with real CT and PET.")
        if not isinstance(ct_features, (list, tuple)) or not isinstance(pet_features, (list, tuple)):
            raise TypeError("Features must be lists/tuples of four BCHW tensors.")
        if len(ct_features) != 4 or len(pet_features) != 4:
            raise ValueError("Expected four CT and four PET feature maps.")
        first = ct_features[0]
        if not isinstance(first, Tensor) or first.ndim != 4:
            raise ValueError("CT features must be BCHW tensors.")
        for i, (c, p, channels) in enumerate(zip(ct_features, pet_features, self.channels)):
            if not isinstance(c, Tensor) or not isinstance(p, Tensor):
                raise TypeError(f"Scale {i}: inputs must be tensors.")
            if c.ndim != 4 or p.shape != c.shape or c.shape[1] != channels or min(c.shape) <= 0:
                raise ValueError(f"Scale {i}: matching nonempty BCHW tensors with {channels} channels required.")
            if not c.is_floating_point() or p.dtype != c.dtype or c.dtype != first.dtype:
                raise ValueError("All feature maps must share a floating dtype.")
            if c.device != p.device or c.device != first.device or c.shape[0] != first.shape[0]:
                raise ValueError("All feature maps must share batch size and device.")
        parameter = next(self.parameters())
        if parameter.device != first.device:
            raise ValueError("Move the fusion module to the input device before forward.")
        return [layer(c, p, self.text_embeddings)
                for layer, c, p in zip(self.scales, ct_features, pet_features)]


class ContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)
        self.kw = dict(channels=(8, 12, 16, 24), pet_dims=(8, 8, 12, 16),
                       text_dim=16, grid_cap=8)

    def model(self, **kwargs):
        return FullPETCTAsymmetricFusion(text_embeddings=torch.randn(2, 16), **self.kw, **kwargs)

    def features(self):
        return [torch.randn(2, c, h, w, requires_grad=True)
                for c, (h, w) in zip(self.kw["channels"], [(17, 15), (9, 8), (5, 4), (3, 2)])]

    def test_four_scales_backward_and_frozen_cache(self):
        model = self.model(checkpoint_attention=True)
        c, p = self.features(), self.features()
        out = model(c, p)
        self.assertEqual([x.shape for x in out], [x.shape for x in c])
        sum(x.square().mean() for x in out).backward()
        for x in c + p:
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertFalse(model.text_embeddings.requires_grad)
        self.assertIsNone(model.text_embeddings.grad)

    def test_no_text_never_loads_clip(self):
        with patch(__name__ + "._offline_clip_embeddings", side_effect=AssertionError("CLIP called")):
            model = FullPETCTAsymmetricFusion(use_text=False, **self.kw)
            out = model(self.features(), self.features())
            sum(x.mean() for x in out).backward()

    def test_missing_rejected_before_branches(self):
        model = self.model()
        with patch.object(model.scales[0], "forward", side_effect=AssertionError("branch called")):
            with self.assertRaises(ValueError):
                model([], [], state="missing")

    def test_checkpoint_round_trip(self):
        model = self.model().eval()
        c, p = self.features(), self.features()
        with torch.no_grad():
            expected = model(c, p)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        state = torch.load(stream, weights_only=True)
        restored = FullPETCTAsymmetricFusion(text_embeddings=state["text_embeddings"], **self.kw).eval()
        restored.load_state_dict(state)
        with torch.no_grad():
            actual = restored(c, p)
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_no_final_ct_residual(self):
        model = self.model()
        for scale in model.scales:
            nn.init.zeros_(scale.fusion.out_proj.weight)
            nn.init.zeros_(scale.fusion.out_proj.bias)
        for x in model(self.features(), self.features()):
            self.assertEqual(torch.count_nonzero(x).item(), 0)

    def test_grid_cap_and_shape_validation(self):
        model = self.model()
        seen = []
        handles = [s.pet_correction.qkv.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].shape[-2:])) for s in model.scales]
        c, p = self.features(), self.features()
        model(c, p)
        for handle in handles:
            handle.remove()
        self.assertEqual(seen, [(8, 8), (8, 8), (5, 4), (3, 2)])
        with self.assertRaises(ValueError):
            model(c, p[:3])
        p[0] = p[0][:, :-1]
        with self.assertRaises(ValueError):
            model(c, p)

    def test_checkpoint_matches_regular_gradients(self):
        regular, checked = self.model(), self.model(checkpoint_attention=True)
        checked.load_state_dict(regular.state_dict())
        c, p = self.features(), self.features()
        out1, out2 = regular(c, p), checked(c, p)
        sum(x.square().mean() for x in out1).backward()
        sum(x.square().mean() for x in out2).backward()
        for a, b in zip(regular.parameters(), checked.parameters()):
            torch.testing.assert_close(a.grad, b.grad)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpoint-attention", action="store_true")
    parser.add_argument("--clip-path", type=Path)
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
    if args.smoke:
        torch.manual_seed(19)
        torch.set_num_threads(4)
        device = torch.device(args.device)
        # Without a local path, synthetic embeddings test computation only.
        model = FullPETCTAsymmetricFusion(
            clip_path=args.clip_path,
            text_embeddings=None if args.clip_path else torch.randn(2, 512),
            checkpoint_attention=args.checkpoint_attention).to(device).train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        c = [torch.randn(args.batch_size, ch, size, size, device=device, requires_grad=True)
             for ch, size in zip(model.channels, (128, 64, 32, 16))]
        p = [torch.randn_like(x, requires_grad=True) for x in c]
        with torch.autocast(device_type=device.type, enabled=args.amp,
                            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
            out = model(c, p)
            loss = sum(x.float().square().mean() for x in out)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in c + p)
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters())
        print("Shapes:", [tuple(x.shape) for x in out])
        print("Trainable parameters:", sum(x.numel() for x in model.parameters()))
        print("CLIP source:", args.clip_path or "synthetic embeddings (NOT real CLIP validation)")
        print("Forward/backward: finite")
        if device.type == "cuda":
            print("Module-only peak allocated GiB:", torch.cuda.max_memory_allocated(device) / 2**30)
    if not args.self_test and not args.smoke:
        parser.print_help()


if __name__ == "__main__":
    _main()
