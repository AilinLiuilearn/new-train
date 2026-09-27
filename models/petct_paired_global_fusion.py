"""Standalone Full PET-CT fusion; destination: models/petct_paired_global_fusion.py.

Contract
--------
* Inputs: four aligned BCHW CT/PET tensors in fine-to-coarse S1..S4 order.
  Default channels: (64, 128, 320, 512). CT channel alignment belongs upstream.
* Outputs: four fused BCHW tensors with the same shapes, ready for a decoder.
* Full-only: Missing/mixed states are rejected BEFORE any feature computation.
  Stage-2 CT-to-bank retrieval is a separate module, not zero-PET fusion here.
* No encoder, decoder, prototype bank, loss, EMA, or training schedule is owned
  by this module. Returned features retain gradients; detach only when collecting
  bank entries outside the module.
* Text: frozen local Hugging Face CLIPTextModel pooler_output, two fixed prompts.
  Only normalized embeddings are persistent buffers. The tower is never retained.
  A trainable fixed-text projection is not proof of semantic usefulness.

Mechanism
---------
One bottleneck text bias -> visual channel gate -> separate CT 3/5/7 DWConv and
PET 3 DWConv -> raw-input branch residuals -> PET-scored paired CT/PET pooling
-> CT Q / paired CT K / paired PET V global attention -> local/global PET merge
-> spatial gate -> concat with CT -> output projection (NO final CT addition).

Dependencies: Python >=3.10, PyTorch >=2.1. transformers is needed ONLY to prepare
real CLIP embeddings. Attention uses SDPA; FlashAttention use depends on hardware,
dtype and PyTorch. Set an external torch.manual_seed for reproducible initialization.

Examples
--------
python petct_paired_global_fusion.py --prepare-text \
    --clip-path pretrained/clip-vit-base-patch32 \
    --text-cache pretrained/petct_fixed_text.pt
python petct_paired_global_fusion.py --smoke --use-text \
    --text-cache pretrained/petct_fixed_text.pt
python petct_paired_global_fusion.py --smoke --device cuda --batch-size 16 --amp

References (mechanism context, NOT a reproduction or a performance claim):
https://github.com/iLearn-Lab/MM26-DGNet/blob/main/model/dgnet/pwm.py
https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
https://huggingface.co/docs/transformers/model_doc/clip
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ['PETCTPairedGlobalFusion', 'prepare_clip_text_cache']

DEFAULT_CT_PROMPT = 'A CT image showing the anatomical structure and boundaries of lung tumors.'
DEFAULT_PET_PROMPT = 'A PET image showing bright tumor regions in the lungs.'
TEXT_FORMAT = 'petct-fixed-clip-pooler-v1'


def _normalize_text(embeddings: Tensor, text_dim: int) -> Tensor:
    """Make an owned, gradient-free CPU copy; never modify the caller's tensor."""
    if not isinstance(embeddings, Tensor) or embeddings.shape != (2, text_dim):
        raise ValueError(f'text embeddings must have shape (2, {text_dim}), CT then PET')
    t = embeddings.detach().to(device='cpu', dtype=torch.float32).clone()
    if not torch.isfinite(t).all() or (t.norm(dim=-1) < 1e-8).any():
        raise ValueError('text embeddings must be finite and nonzero')
    return F.normalize(t, p=2, dim=-1)


def _encode_local_clip(clip_path: str | Path, prompts: Sequence[str]) -> Tensor:
    path = Path(clip_path).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f'Offline CLIP directory not found: {path}')
    try:
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as exc:
        raise ImportError('Install transformers to prepare text; cached inference only needs torch') from exc
    tokenizer = CLIPTokenizer.from_pretrained(str(path), local_files_only=True)
    tower = CLIPTextModel.from_pretrained(str(path), local_files_only=True)
    tower = tower.to(device='cpu', dtype=torch.float32).eval()
    tower.requires_grad_(False)
    tokens = tokenizer(list(prompts), padding=True, truncation=True,
                       max_length=tower.config.max_position_embeddings,
                       return_tensors='pt')
    with torch.no_grad():
        # Deliberately use pooler_output, NOT CLIP's final contrastive projection.
        t = tower(**tokens).pooler_output.detach().float().cpu().clone()
    return _normalize_text(t, t.shape[-1])


def prepare_clip_text_cache(
    clip_path: str | Path, output_path: str | Path,
    ct_prompt: str = DEFAULT_CT_PROMPT, pet_prompt: str = DEFAULT_PET_PROMPT,
) -> Path:
    """One-time offline encoding. Call once before training, not in forward()."""
    embeddings = _encode_local_clip(clip_path, (ct_prompt, pet_prompt))
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'format': TEXT_FORMAT, 'embeddings': embeddings,
                'prompts': [ct_prompt, pet_prompt],
                'feature_type': 'CLIPTextModel.pooler_output'}, output)
    return output


class ConvNormAct(nn.Sequential):
    """GroupNorm(1) avoids running statistics and works for small Full sub-batches."""
    def __init__(self, cin: int, cout: int, kernel: int = 1, groups: int = 1):
        super().__init__(
            nn.Conv2d(cin, cout, kernel, padding=kernel // 2, groups=groups, bias=False),
            nn.GroupNorm(1, cout), nn.GELU(),
        )


class LightTextChannelGate(nn.Module):
    def __init__(self, channels: int, bottleneck: int, pool: str):
        super().__init__()
        if pool not in ('avg', 'max'):
            raise ValueError('pool must be avg or max')
        self.pool = pool
        self.down = nn.Linear(channels, bottleneck)
        self.up = nn.Linear(bottleneck, channels)
        # Only this gate is identity at initialization; the entire fusion is NOT.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: Tensor, text_bias: Tensor | None) -> tuple[Tensor, Tensor]:
        # Small descriptors in FP32 prevent half-precision pooling overflow.
        if self.pool == 'avg':
            v = x.float().mean(dim=(-2, -1)).to(x.dtype)
        else:
            v = x.amax(dim=(-2, -1))
        u = self.down(v)
        if text_bias is not None:
            u = u + text_bias.to(device=u.device, dtype=u.dtype).unsqueeze(0)
        gate = torch.tanh(self.up(F.gelu(u)))
        return x * (1 + gate.to(x.dtype)[:, :, None, None]), gate


class CTStructureProcessor(nn.Module):
    def __init__(self, channels: int, inner: int):
        super().__init__()
        self.project = ConvNormAct(channels, inner)
        self.branches = nn.ModuleList([ConvNormAct(inner, inner, k, groups=inner)
                                       for k in (3, 5, 7)])
        self.merge = ConvNormAct(3 * inner, inner)
        self.output = nn.Conv2d(inner, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        z = self.project(x)
        return self.output(self.merge(torch.cat([b(z) for b in self.branches], dim=1)))


class PETResponseProcessor(nn.Sequential):
    def __init__(self, channels: int, inner: int):
        super().__init__(ConvNormAct(channels, inner),
                         ConvNormAct(inner, inner, 3, groups=inner),
                         nn.Conv2d(inner, channels, 1))


class PairedRegionPool(nn.Module):
    """PET scores -> same within-region weights for CT and PET, no cross-patient data.

    Grid bins do not overlap. H/W must be divisible by the effective grid sides.
    Grid side is min(configured_side, feature_side); smaller maps are not enlarged.
    Standard 128/64/32/16 feature sizes with grid=16 satisfy this contract.
    """
    def __init__(self, channels: int, grid_size: int = 16):
        super().__init__()
        if grid_size < 1:
            raise ValueError('grid_size must be positive')
        self.grid_size = grid_size
        self.score = nn.Conv2d(channels, 1, 1)

    def grid(self, height: int, width: int) -> tuple[int, int]:
        gh, gw = min(height, self.grid_size), min(width, self.grid_size)
        if height % gh or width % gw:
            raise ValueError(f'Feature size {(height, width)} must be divisible by '
                             f'the effective region grid {(gh, gw)}; no implicit resize/padding')
        return gh, gw

    @staticmethod
    def _regions(x: Tensor, gh: int, gw: int) -> Tensor:
        b, c, h, w = x.shape
        # [B,N_regions,N_pixels_in_region,C], both indices in row-major order.
        return x.reshape(b, c, gh, h // gh, gw, w // gw).permute(
            0, 2, 4, 3, 5, 1).reshape(b, gh * gw, (h // gh) * (w // gw), c)

    def forward(self, ct: Tensor, pet: Tensor) -> tuple[Tensor, Tensor, Tensor, tuple[int, int]]:
        b, _, h, w = ct.shape
        gh, gw = self.grid(h, w)
        if gh == h and gw == w:
            # One pixel per region: softmax would be 1 and score has no effect.
            return (ct.flatten(2).transpose(1, 2), pet.flatten(2).transpose(1, 2),
                    ct.new_ones((b, gh * gw, 1), dtype=torch.float32), (gh, gw))
        score = self._regions(self.score(pet), gh, gw).squeeze(-1)
        weights = torch.softmax(score.float(), dim=-1)
        cr, pr = self._regions(ct, gh, gw), self._regions(pet, gh, gw)
        with torch.autocast(device_type=ct.device.type, enabled=False):
            tc = torch.einsum('bnk,bnkd->bnd', weights, cr.float())
            tp = torch.einsum('bnk,bnkd->bnd', weights, pr.float())
        return tc.to(ct.dtype), tp.to(pet.dtype), weights, (gh, gw)


class PairedGlobalAttention(nn.Module):
    def __init__(self, dim: int, heads: int, query_size: int,
                 checkpoint_attention: bool = False):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.query_size = query_size
        self.checkpoint_attention = checkpoint_attention
        self.q_norm, self.k_norm, self.v_norm = (nn.LayerNorm(dim) for _ in range(3))
        self.q, self.k, self.v = (nn.Linear(dim, dim) for _ in range(3))
        self.out = nn.Linear(dim, dim)

    def _attend(self, uc: Tensor, tc: Tensor, tp: Tensor) -> Tensor:
        b, d, h, w = uc.shape
        qh, qw = min(h, self.query_size), min(w, self.query_size)
        q_source = F.adaptive_avg_pool2d(uc, (qh, qw)).flatten(2).transpose(1, 2)
        def split(x: Tensor) -> Tensor:
            return x.reshape(b, -1, self.heads, self.head_dim).transpose(1, 2)
        q = split(self.q(self.q_norm(q_source)))
        k = split(self.k(self.k_norm(tc)))
        v = split(self.v(self.v_norm(tp)))
        result = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        result = result.transpose(1, 2).reshape(b, qh * qw, d)
        result = self.out(result).transpose(1, 2).reshape(b, d, qh, qw)
        if (qh, qw) != (h, w):
            result = F.interpolate(result, size=(h, w), mode='bilinear', align_corners=False)
        return result

    def forward(self, uc: Tensor, tc: Tensor, tp: Tensor) -> Tensor:
        if self.checkpoint_attention and self.training and torch.is_grad_enabled():
            return checkpoint(self._attend, uc, tc, tp, use_reentrant=False)
        return self._attend(uc, tc, tp)


class FusionScale(nn.Module):
    def __init__(self, channels: int, inner: int, heads: int, query_size: int,
                 kv_size: int, text_dim: int, text_reduction: int,
                 checkpoint_attention: bool):
        super().__init__()
        self.channels, self.inner = channels, inner
        r = max(8, channels // text_reduction)
        self.ct_gate = LightTextChannelGate(channels, r, 'avg')
        self.pet_gate = LightTextChannelGate(channels, r, 'max')
        # Shared across CT/PET in THIS scale; not shared across scales.
        self.text_projection = nn.Linear(text_dim, r, bias=False)
        self.ct_processor = CTStructureProcessor(channels, inner)
        self.pet_processor = PETResponseProcessor(channels, inner)
        self.ct_project = ConvNormAct(channels, inner)
        self.pet_project = ConvNormAct(channels, inner)
        self.pool = PairedRegionPool(inner, kv_size)
        self.attention = PairedGlobalAttention(inner, heads, query_size, checkpoint_attention)
        self.pet_merge = ConvNormAct(2 * inner, inner)
        self.gate_context = ConvNormAct(3 * inner, inner)
        self.spatial_score = nn.Conv2d(inner, 1, 3, padding=1)
        nn.init.zeros_(self.spatial_score.weight)
        nn.init.zeros_(self.spatial_score.bias)  # Initial spatial gate = 0.5.
        self.pet_output = nn.Conv2d(inner, channels, 1)
        self.fusion_output = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, ct: Tensor, pet: Tensor, text: Tensor | None,
                return_diagnostics: bool = False) -> tuple[Tensor, dict[str, Any] | None]:
        bias = None if text is None else torch.tanh(self.text_projection(text))
        cm, gc = self.ct_gate(ct, None if bias is None else bias[0])
        pm, gp = self.pet_gate(pet, None if bias is None else bias[1])
        # Residual origins are RAW aligned inputs, not channel-modulated inputs.
        c_str = ct + self.ct_processor(cm)
        p_met = pet + self.pet_processor(pm)
        uc, up = self.ct_project(c_str), self.pet_project(p_met)
        tc, tp, region_weights, grid = self.pool(uc, up)
        global_pet = self.attention(uc, tc, tp)
        local_global_pet = self.pet_merge(torch.cat([up, global_pet], dim=1))
        a = torch.sigmoid(self.spatial_score(self.gate_context(
            torch.cat([uc, up, global_pet], dim=1))))
        r_pet = self.pet_output(a * local_global_pet)
        fused = self.fusion_output(torch.cat([c_str, r_pet], dim=1))
        # Deliberately NO `fused + ct` or `fused + c_str` here.
        if not return_diagnostics:
            return fused, None
        b, _, h, w = ct.shape
        qhw = (min(h, self.attention.query_size), min(w, self.attention.query_size))
        return fused, {
            'ct_channel_gate': gc.detach(), 'pet_channel_gate': gp.detach(),
            'spatial_gate': a.detach(), 'region_weights': region_weights.detach(),
            'query_grid': qhw, 'kv_grid': grid,
            # Shape only; the full attention matrix is NOT requested/stored here.
            'attention_shape': (b, self.attention.heads, math.prod(qhw), math.prod(grid)),
        }


class PETCTPairedGlobalFusion(nn.Module):
    """Four-scale Full fusion, independent of the encoder/decoder implementation.

    Construct with a real `text_cache_path`, with `clip_path` for one-time offline
    encoding, or load a full state_dict (it includes the text buffers). Constructing
    use_text=True without text is allowed for checkpoint restoration, but forward
    raises until valid cached text is available. No random-text fallback exists.

    Mixed precision: PET is cast to each CT tensor's dtype WITHOUT detach. Autocast
    determines output dtype; output SHAPES match inputs. Keep module weights FP32
    and use autocast for normal mixed-precision training.
    """
    def __init__(
        self, channels: Sequence[int] = (64, 128, 320, 512),
        inner_channels: Sequence[int] = (64, 128, 160, 256),
        num_heads: int = 4, query_size: int = 32, kv_size: int = 16,
        use_text: bool = True, text_dim: int = 512, text_reduction: int = 16,
        text_embeddings: Tensor | None = None,
        text_cache_path: str | Path | None = None,
        clip_path: str | Path | None = None,
        ct_prompt: str = DEFAULT_CT_PROMPT, pet_prompt: str = DEFAULT_PET_PROMPT,
        checkpoint_attention: bool = False, check_finite: bool = False,
    ):
        super().__init__()
        if len(channels) != 4 or len(inner_channels) != 4:
            raise ValueError('Exactly four scales, S1..S4, are required')
        numbers = (*channels, *inner_channels, num_heads, query_size, kv_size,
                   text_dim, text_reduction)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in numbers):
            raise ValueError('Channels, dimensions, heads and grid sizes must be positive integers')
        if any(d % num_heads for d in inner_channels):
            raise ValueError('Every inner_channels entry must be divisible by num_heads')
        if any(d < 2 for d in inner_channels):
            raise ValueError('inner_channels must be >=2 for normalized singleton feature maps')
        if sum(x is not None for x in (text_embeddings, text_cache_path, clip_path)) > 1:
            raise ValueError('Provide only one of text_embeddings, text_cache_path, clip_path')
        self.channels = tuple(channels)
        self.inner_channels = tuple(inner_channels)
        self.use_text = bool(use_text)
        self.text_dim = text_dim
        self.prompts = (ct_prompt, pet_prompt)
        self.check_finite = bool(check_finite)
        self.register_buffer('text_embeddings', torch.zeros(2, text_dim), persistent=True)
        self.register_buffer('text_ready', torch.tensor(False), persistent=True)
        self.scales = nn.ModuleList([
            FusionScale(c, d, num_heads, query_size, kv_size, text_dim,
                        text_reduction, checkpoint_attention)
            for c, d in zip(channels, inner_channels)
        ])
        if text_embeddings is not None:
            self.set_text_embeddings(text_embeddings)
        elif text_cache_path is not None:
            self.load_text_cache(text_cache_path)
        elif clip_path is not None:
            self.set_text_embeddings(_encode_local_clip(clip_path, self.prompts))

    @torch.no_grad()
    def set_text_embeddings(self, embeddings: Tensor) -> None:
        t = _normalize_text(embeddings, self.text_dim)
        self.text_embeddings.copy_(t.to(self.text_embeddings))
        self.text_ready.fill_(True)

    def load_text_cache(self, path: str | Path) -> None:
        cache = torch.load(Path(path).expanduser(), map_location='cpu', weights_only=True)
        if not isinstance(cache, dict) or cache.get('format') != TEXT_FORMAT:
            raise ValueError('Unsupported text cache. Use prepare_clip_text_cache()')
        if cache.get('prompts') != list(self.prompts):
            raise ValueError('Text cache prompts differ from constructor prompts; do not silently reuse')
        self.set_text_embeddings(cache['embeddings'])

    @staticmethod
    def _require_full(state: str | Tensor, batch: int) -> None:
        if isinstance(state, str):
            if state == 'full':
                return
            raise ValueError('Full-only fusion: Missing uses a separate CT-to-bank retrieval path')
        if not isinstance(state, Tensor) or state.ndim != 1 or state.numel() != batch:
            raise ValueError('state must be "full" or a one-dimensional all-ones availability tensor')
        if not bool((state == 1).all().item()):
            raise ValueError('Full-only fusion: split mixed batches and route Missing to CT-to-bank retrieval')

    def forward(
        self, ct_features: Sequence[Tensor], pet_features: Sequence[Tensor] | None,
        state: str | Tensor = 'full', *, return_diagnostics: bool = False,
    ) -> list[Tensor] | tuple[list[Tensor], list[dict[str, Any]]]:
        if not isinstance(ct_features, (list, tuple)) or len(ct_features) != 4:
            raise ValueError('ct_features must be a list/tuple of four BCHW tensors')
        if not isinstance(ct_features[0], Tensor) or ct_features[0].ndim != 4:
            raise ValueError('CT S1 must be BCHW')
        batch = ct_features[0].shape[0]
        self._require_full(state, batch)  # BEFORE inspecting PET or calling any feature layer.
        if not isinstance(pet_features, (list, tuple)) or len(pet_features) != 4:
            raise ValueError('Full requires four real-PET feature tensors; None/zero masking is not Missing support')
        if batch < 1:
            raise ValueError('Empty Full subset: skip this module in the caller')
        device = ct_features[0].device
        weight = self.scales[0].fusion_output.weight
        if device != weight.device:
            raise ValueError('Move the module and input features to the same device')
        for i, (c, p, expected, scale) in enumerate(zip(ct_features, pet_features, self.channels, self.scales)):
            if not isinstance(c, Tensor) or not isinstance(p, Tensor) or c.ndim != 4 or p.ndim != 4:
                raise ValueError(f'S{i+1}: CT and PET must both be BCHW tensors')
            if c.shape != p.shape or c.shape[0] != batch or c.shape[1] != expected:
                raise ValueError(f'S{i+1}: expected matching [B,{expected},H,W], got {c.shape} and {p.shape}; '
                                 'align CT channels and spatial coordinates upstream')
            if min(c.shape[-2:]) < 1 or not c.is_floating_point() or not p.is_floating_point():
                raise ValueError(f'S{i+1}: nonempty floating-point features are required')
            if c.device != device or p.device != device:
                raise ValueError('All input scales/modalities must be on the same device')
            scale.pool.grid(*c.shape[-2:])
            if self.check_finite and (not torch.isfinite(c).all() or not torch.isfinite(p).all()):
                raise FloatingPointError(f'S{i+1}: input contains NaN/Inf')
        text = None
        if self.use_text:
            if not bool(self.text_ready.item()):
                raise RuntimeError('use_text=True but no valid text cache is loaded; initialize text or load a checkpoint')
            text = self.text_embeddings
        fused, diagnostics = [], []
        for scale, ct, pet in zip(self.scales, ct_features, pet_features):
            pet = pet.to(dtype=ct.dtype)  # Grad-preserving; no detach or masked input construction.
            y, d = scale(ct, pet, text, return_diagnostics)
            if self.check_finite and not torch.isfinite(y).all():
                raise FloatingPointError('Fusion output contains NaN/Inf; no silent nan_to_num repair')
            fused.append(y)
            if d is not None:
                diagnostics.append(d)
        return (fused, diagnostics) if return_diagnostics else fused


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prepare-text', action='store_true')
    parser.add_argument('--clip-path', type=str)
    parser.add_argument('--text-cache', type=str)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--use-text', action='store_true', help='Smoke test with real cached/local text')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--checkpoint-attention', action='store_true')
    args = parser.parse_args()
    if args.prepare_text:
        if not args.clip_path or not args.text_cache:
            parser.error('--prepare-text requires --clip-path and --text-cache')
        print(prepare_clip_text_cache(args.clip_path, args.text_cache))
        if not args.smoke:
            return
    if not args.smoke:
        parser.error('Choose --prepare-text or --smoke')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.use_text and not (args.text_cache or args.clip_path):
        parser.error('--use-text requires --text-cache or --clip-path; synthetic text is never substituted')
    device = torch.device(args.device)
    torch.manual_seed(23)
    if device.type == 'cpu':
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    kwargs = {}
    if args.use_text:
        kwargs = {'text_cache_path': args.text_cache} if args.text_cache else {'clip_path': args.clip_path}
    model = PETCTPairedGlobalFusion(use_text=args.use_text, check_finite=True,
                                  checkpoint_attention=args.checkpoint_attention, **kwargs).to(device).train()
    ct = [torch.randn(args.batch_size, c, h, h, device=device, requires_grad=True)
          for c, h in zip(model.channels, (128, 64, 32, 16))]
    pet = [torch.randn_like(x, requires_grad=True) for x in ct]
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    amp_dtype = torch.float16 if device.type == 'cuda' else torch.bfloat16
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
        output, diagnostics = model(ct, pet, return_diagnostics=True)
        loss = sum(x.float().square().mean() for x in output)
    # Smoke only. Real CUDA training should use GradScaler when appropriate.
    loss.backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in ct + pet)
    assert all(torch.isfinite(x).all() for x in output)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    report = {'torch': torch.__version__, 'device': str(device), 'batch_size': args.batch_size,
              'amp': args.amp, 'use_text': args.use_text,
              'parameters': sum(p.numel() for p in model.parameters()),
              'output_shapes': [list(x.shape) for x in output],
              'attention_shapes': [d['attention_shape'] for d in diagnostics],
              'loss': float(loss.detach()), 'forward_backward_finite': True}
    if device.type == 'cuda':
        report['module_peak_cuda_allocated_mib'] = torch.cuda.max_memory_allocated(device) / 2**20
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    _main()
