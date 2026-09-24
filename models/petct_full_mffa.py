"""Full-only PET/CT MFFA residual fusion; standalone, PyTorch >= 2.0.

Destination: models/petct_full_mffa.py
Baseline: AilinLiuilearn/new-train, e1-api-masked-baseline-mix-full-missing,
          76c9387b5776a3e075b5165711d68f3fa81ae182.
Reference: Jie-1203/WFANet, net_torch.py, Attention / S_MWiT / L_MWiT,
           fe2e5ae80ffbc67c5660e89220301467db041851 (AAAI 2025).
           https://github.com/Jie-1203/WFANet

Contract
--------
Inputs: aligned CT and REAL PET feature lists, S1 -> S4, normally
        (64,128,320,512) channels at (128,64,32,16) spatial sizes.
        pet_available is explicit: 1=Full, 0=Missing. A mixed batch is
        selected BEFORE any trainable PET/fusion operation in this module.
Full: Q_i <- CT band i; K_i <- CT LL; V_i <- fuse(CT LL, PET LL).
      Four independent global attentions, as in the reference. Reconstruct
      a spatial delta; output CT + delta. PET high bands are unused.
Missing: exact CT identity, zero delta; no MFFA call. PET may be None when
         every row is Missing. PET zeros must NEVER be run through MFFA.
Output: list[Tensor] with input CT shapes/device/dtype. return_delta=True
        returns (fused_features, spatial_deltas). Deltas keep their graph;
        future bank collection must explicitly detach, on TRAIN data only.

Source fidelity and adaptations
-------------------------------
* Preserve source Q/K/V MLPs, per-band parameters, scaling sqrt(head_dim),
  projected-V residual, and two output MLP residuals.
* sdpa implements the SAME global attention. No windows, pooling of K/V,
  top-k, linear attention or attention-map export. math is a small-tensor
  reference backend. Fused CUDA dispatch depends on the actual environment.
* Use two convolutions for Figure 4's two-input value mixer:
  conv_mix(CT_LL + conv_pet(PET_LL)). No pyramid back_img or mixing scalars.
* Match source Haar FORWARD scaling: analysis coefficients use 1/4;
  synthesis uses +/-1. This is invertible, NOT an orthonormal energy basis.
  Native autograd deliberately replaces the source custom backward, whose
  DWT multiplies gradients by 4 and IDWT divides them by 4. Those factors
  are not the derivatives of the individual forward operators.
* 1x1 projections adapt task channels to the source's default width 32,
  then restore task channels. The output projection has no bias and
  commutes with IDWT. Spatial resolutions and global token coverage remain.
* CT outer residual is the task adaptation. No lambda, PET bypass, gates,
  text, losses, encoders, decoder or prototype bank live in this file.
* Standard nonzero PyTorch initialization, NOT zero output initialization.
  Full initially differs from both CT and CT+PET. PET and MFFA get gradient
  from the first Full backward. Only Missing has exact CT equivalence.
* These are CT-conditioned fusion deltas, not guaranteed PET-specific
  information. Their usefulness and PET dependence need task evidence.

Usage
-----
    fusion = PETCTFullMFFA(channels=(64,128,320,512))  # register before AdamW
    fused = fusion(ct_feats, real_pet_feats, pet_available)  # Tensor[B]
    fused, delta = fusion.forward_full(ct_feats, real_pet_feats, return_delta=True)
    fused = fusion.forward_missing(ct_feats)  # no PET needed

Mixed integration must encode only REAL PET Full rows upstream, then
scatter those feature rows into zero placeholders. Passing the full state
mask is still mandatory. This module does not itself call an encoder.
Do not directly replace AddFusion without fixing its legacy third arg None.

    python models/petct_full_mffa.py --smoke-test
    python models/petct_full_mffa.py --profile --device cuda --batch-size 16 \
        --mode mixed --amp --checkpoint-attention --flash-only
The profile measures ONLY this module + input features + AdamW, not encoders
or decoder. Default invocation prints the config and exact parameter count.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ["PETCTFullMFFA", "MFFAScale", "GlobalFrequencyAttention",
           "haar_dwt2", "haar_ll", "haar_idwt2"]
BAND_NAMES = ("ll", "lh", "hl", "hh")
Bands = Tuple[Tensor, Tensor, Tensor, Tensor]
State = Union[bool, int, Tensor]
FusionReturn = Union[List[Tensor], Tuple[List[Tensor], List[Tensor]]]


def _image(x: Tensor, name: str) -> None:
    if not isinstance(x, Tensor) or x.ndim != 4 or not x.is_floating_point():
        raise ValueError(f"{name} must be floating Tensor[B,C,H,W]")
    if min(x.shape) <= 0 or x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError(f"{name} requires positive sizes and even H,W; got {tuple(x.shape)}")


def haar_dwt2(x: Tensor) -> Bands:
    """Source-compatible average Haar, LH varies along rows, HL along columns."""
    _image(x, "Haar input")
    a, b = x[..., 0::2, 0::2] * 0.25, x[..., 0::2, 1::2] * 0.25
    c, d = x[..., 1::2, 0::2] * 0.25, x[..., 1::2, 1::2] * 0.25
    return ((a + b) + (c + d), (a + b) - (c + d),
            (a - b) + (c - d), (a - b) - (c - d))


def haar_ll(x: Tensor) -> Tensor:
    """Exactly the LL of haar_dwt2, without allocating unused PET high bands."""
    _image(x, "PET Haar input")
    return ((x[..., 0::2, 0::2] * 0.25 + x[..., 0::2, 1::2] * 0.25)
            + (x[..., 1::2, 0::2] * 0.25 + x[..., 1::2, 1::2] * 0.25))


def haar_idwt2(ll: Tensor, lh: Tensor, hl: Tensor, hh: Tensor) -> Tensor:
    """Inverse of average Haar. Band H,W can be odd; output is always even."""
    if not isinstance(ll, Tensor) or ll.ndim != 4 or not ll.is_floating_point() or min(ll.shape) <= 0:
        raise ValueError("IDWT expects four positive floating 4D bands")
    for band in (lh, hl, hh):
        if not isinstance(band, Tensor) or band.shape != ll.shape or band.dtype != ll.dtype or band.device != ll.device:
            raise ValueError("IDWT bands must have identical shape, dtype and device")
    a, b = (ll + lh) + (hl + hh), (ll + lh) - (hl + hh)
    c, d = (ll - lh) + (hl - hh), (ll - lh) - (hl - hh)
    top = torch.stack((a, b), dim=-1).flatten(-2, -1)
    bottom = torch.stack((c, d), dim=-1).flatten(-2, -1)
    return torch.stack((top, bottom), dim=-2).flatten(-3, -2)


def _mlp(dim: int, norm: bool) -> nn.Sequential:
    layers = [nn.LayerNorm(dim)] if norm else []
    return nn.Sequential(*layers, nn.Linear(dim, 2 * dim), nn.LeakyReLU(0.01),
                         nn.Linear(2 * dim, dim), nn.Dropout(0.0))


class GlobalFrequencyAttention(nn.Module):
    """Reference Attention algebra; channels adapted before entry, no windows."""
    def __init__(self, dim: int = 32, num_heads: int = 4, backend: str = "sdpa"):
        super().__init__()
        if type(dim) is not int or type(num_heads) is not int or dim <= 0 or num_heads <= 0 or dim % num_heads:
            raise ValueError("positive dim must be divisible by positive num_heads")
        if backend not in ("sdpa", "math"):
            raise ValueError("backend must be sdpa or math")
        self.dim, self.num_heads, self.backend = dim, num_heads, backend
        self.head_dim = dim // num_heads
        self.q, self.k, self.v = (_mlp(dim, True) for _ in range(3))
        self.mlp_1, self.mlp_2 = _mlp(dim, False), _mlp(dim, False)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[1] != self.dim:
            raise ValueError("attention expects equally shaped [B,dim,H,W] inputs")
        b, _, h, w = q.shape
        tokens = lambda x: x.flatten(2).transpose(1, 2)
        split = lambda x: x.reshape(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        qt, kt, vt = self.q(tokens(q)), self.k(tokens(k)), self.v(tokens(v))
        qh, kh, vh = split(qt), split(kt), split(vt)
        if self.backend == "sdpa":
            attended = F.scaled_dot_product_attention(qh, kh, vh, dropout_p=0.0, is_causal=False)
        else:
            # Deliberately explicit O(N^2) reference, not the production default.
            dtype = torch.float64 if qh.dtype == torch.float64 else torch.float32
            with torch.autocast(device_type=qh.device.type, enabled=False):
                logits = (qh.to(dtype) @ kh.to(dtype).transpose(-2, -1)) / math.sqrt(self.head_dim)
                attended = (logits.softmax(dim=-1) @ vh.to(dtype)).to(vh.dtype)
        attended = attended.transpose(1, 2).reshape(b, h * w, self.dim)
        # These are the reference projected-Value and MLP residuals, NOT Q skips.
        first = vt + self.mlp_1(attended)
        result = first + self.mlp_2(first)
        return result.transpose(1, 2).reshape(b, self.dim, h, w)


class MFFAScale(nn.Module):
    """One Full-only scale: return spatial delta, with exactly global attention."""
    def __init__(self, channels: int, embed_dim: int = 32, num_heads: int = 4,
                 attention_backend: str = "sdpa", checkpoint_attention: bool = False):
        super().__init__()
        if type(channels) is not int or channels <= 0:
            raise ValueError("channels must be a positive integer")
        if type(checkpoint_attention) is not bool:
            raise TypeError("checkpoint_attention must be bool")
        self.channels = channels
        self.checkpoint_attention = checkpoint_attention
        self.ct_projection = nn.Conv2d(channels, embed_dim, 1, bias=False)
        self.pet_projection = nn.Conv2d(channels, embed_dim, 1, bias=False)
        self.pet_value_conv = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.value_mix_conv = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.attentions = nn.ModuleDict({
            band: GlobalFrequencyAttention(embed_dim, num_heads, attention_backend)
            for band in BAND_NAMES
        })
        self.output_projection = nn.Conv2d(embed_dim, channels, 1, bias=False)

    def forward(self, ct: Tensor, pet: Tensor) -> Tensor:
        _image(ct, "CT")
        _image(pet, "PET")
        if ct.shape != pet.shape or ct.shape[1] != self.channels or ct.device != pet.device or ct.dtype != pet.dtype:
            raise ValueError("Full CT/PET must match configured channels, shape, device and dtype")
        cb = tuple(self.ct_projection(band) for band in haar_dwt2(ct))
        pl = self.pet_projection(haar_ll(pet))
        value = self.value_mix_conv(cb[0] + self.pet_value_conv(pl))
        reconstructed = []
        for name, query in zip(BAND_NAMES, cb):
            block = self.attentions[name]
            if self.checkpoint_attention and self.training and torch.is_grad_enabled():
                result = checkpoint(block, query, cb[0], value, use_reentrant=False)
            else:
                result = block(query, cb[0], value)
            reconstructed.append(result)
        # bias-free linear channel projection commutes with this IDWT.
        return self.output_projection(haar_idwt2(*reconstructed)).to(dtype=ct.dtype)


def _availability(state: State, batch: int, device: torch.device) -> Tensor:
    if isinstance(state, bool):
        return torch.full((batch,), state, dtype=torch.bool, device=device)
    if type(state) is int and state in (0, 1):
        return torch.full((batch,), bool(state), dtype=torch.bool, device=device)
    if not isinstance(state, Tensor):
        raise TypeError("explicit pet_available required: bool, 0/1, or bool/integer Tensor[B]")
    valid_types = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if state.dtype not in valid_types or state.ndim != 1 or state.numel() != batch:
        raise ValueError("pet_available must be bool/integer Tensor[B], never floating gates")
    if not bool(((state == 0) | (state == 1)).all()):
        raise ValueError("pet_available values must be 0=Missing or 1=Full")
    return state.to(device=device, dtype=torch.bool)


class PETCTFullMFFA(nn.Module):
    """Four-scale adapter, with exact CT-only Missing and row-wise Full selection."""
    def __init__(self, channels: Sequence[int] = (64, 128, 320, 512),
                 embed_dim: int = 32, num_heads: int = 4,
                 attention_backend: str = "sdpa", checkpoint_attention: bool = False):
        super().__init__()
        if not channels or any(type(c) is not int or c <= 0 for c in channels):
            raise ValueError("channels must be a nonempty sequence of positive integers")
        if type(embed_dim) is not int or type(num_heads) is not int or embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        self.channels = tuple(channels)
        self.config = dict(channels=list(channels), embed_dim=embed_dim, num_heads=num_heads,
                           attention_backend=attention_backend, checkpoint_attention=checkpoint_attention)
        self.scales = nn.ModuleList([
            MFFAScale(c, embed_dim, num_heads, attention_backend, checkpoint_attention)
            for c in channels
        ])

    def forward(self, ct_feats: Sequence[Tensor], pet_feats: Optional[Sequence[Tensor]],
                pet_available: State, *, return_delta: bool = False) -> FusionReturn:
        if type(return_delta) is not bool:
            raise TypeError("return_delta must be bool")
        if not isinstance(ct_feats, (list, tuple)) or len(ct_feats) != len(self.scales):
            raise ValueError("CT must contain configured scales, ordered S1 -> S4")
        first = ct_feats[0]
        _image(first, "CT[0]")
        for i, (ct, block) in enumerate(zip(ct_feats, self.scales)):
            _image(ct, f"CT[{i}]")
            if ct.shape[0] != first.shape[0] or ct.shape[1] != block.channels or ct.dtype != first.dtype or ct.device != first.device:
                raise ValueError("CT scales must share B, dtype, device and configured channels")
            if block.ct_projection.weight.device != ct.device:
                raise ValueError("move module and features to the same device")
        mask = _availability(pet_available, first.shape[0], first.device)
        full = mask.nonzero(as_tuple=True)[0]
        if full.numel() == 0:
            # Do not inspect, normalize, multiply by zero, or encode supplied PET.
            fused = list(ct_feats)
            return (fused, [torch.zeros_like(c) for c in ct_feats]) if return_delta else fused
        if not isinstance(pet_feats, (list, tuple)) or len(pet_feats) != len(ct_feats):
            raise ValueError("real PET feature list required when any row is Full")
        for ct, pet in zip(ct_feats, pet_feats):
            if not isinstance(pet, Tensor) or pet.shape != ct.shape or pet.dtype != ct.dtype or pet.device != ct.device:
                raise ValueError("PET feature scales must exactly match CT; align upstream")
        fused, deltas = [], []
        all_full = full.numel() == first.shape[0]
        for ct, pet, block in zip(ct_feats, pet_feats, self.scales):
            if all_full:
                delta = block(ct, pet)
                out = ct + delta
            else:
                ct_full = ct.index_select(0, full)
                delta_full = block(ct_full, pet.index_select(0, full))
                out = ct.index_copy(0, full, ct_full + delta_full)
                delta = torch.zeros_like(ct).index_copy(0, full, delta_full) if return_delta else None
            fused.append(out)
            if return_delta:
                deltas.append(delta)
        return (fused, deltas) if return_delta else fused

    def forward_full(self, ct_feats: Sequence[Tensor], pet_feats: Sequence[Tensor],
                     *, return_delta: bool = False) -> FusionReturn:
        return self(ct_feats, pet_feats, True, return_delta=return_delta)

    def forward_missing(self, ct_feats: Sequence[Tensor], *, return_delta: bool = False) -> FusionReturn:
        return self(ct_feats, None, False, return_delta=return_delta)

    def parameter_report(self) -> dict:
        return {"total": sum(p.numel() for p in self.parameters()),
                "per_scale": [sum(p.numel() for p in m.parameters()) for m in self.scales]}


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mode", choices=("mixed", "full", "missing"), default="mixed")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpoint-attention", action="store_true")
    parser.add_argument("--flash-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or (args.mode == "mixed" and args.batch_size < 2):
        parser.error("positive batch size required; mixed needs >=2")
    if args.smoke_test and args.profile:
        parser.error("choose smoke-test or profile")
    torch.manual_seed(2023)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; no CUDA memory result can be produced")
    tiny = args.smoke_test
    channels = (8, 16, 24, 32) if tiny else (64, 128, 320, 512)
    sizes = (16, 8, 4, 2) if tiny else (128, 64, 32, 16)
    model = PETCTFullMFFA(channels, embed_dim=8 if tiny else 32,
                         num_heads=2 if tiny else 4,
                         checkpoint_attention=args.checkpoint_attention).to(device)
    report = {"config": model.config, "parameters": model.parameter_report()}
    if not (args.smoke_test or args.profile):
        print(json.dumps(report, indent=2))
        return
    # Avoid excessive CPU thread launch overhead in a portable smoke test.
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    batch = args.batch_size
    state = torch.ones(batch, device=device, dtype=torch.long)
    if args.mode == "missing":
        state.zero_()
    elif args.mode == "mixed":
        state[1::2] = 0
    ct = [torch.randn(batch, c, s, s, device=device, requires_grad=True) for c, s in zip(channels, sizes)]
    pet = [torch.randn_like(c, requires_grad=True) for c in ct]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    fused_backend = nullcontext()
    if args.flash_only:
        if device.type != "cuda":
            parser.error("flash-only requires CUDA")
        from torch.nn.attention import SDPBackend, sdpa_kernel
        fused_backend = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with fused_backend:
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            for x in ct + pet:
                x.grad = None
            amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
            with torch.autocast(device.type, dtype=amp_dtype, enabled=args.amp):
                outputs, delta = model(ct, pet, state, return_delta=True)
                loss = sum(x.float().square().mean() for x in outputs)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite smoke loss")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("nonfinite parameter gradient")
            optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    missing = state.eq(0)
    assert all(torch.equal(o[missing], c[missing]) for o, c in zip(outputs, ct))
    assert all(torch.count_nonzero(d[missing]) == 0 for d in delta)
    assert all(p.grad is None or torch.count_nonzero(p.grad[missing]) == 0 for p in pet)
    report.update(mode=args.mode, num_full=int(state.sum()), loss=float(loss.detach()),
                  output_shapes=[list(x.shape) for x in outputs],
                  seconds=time.perf_counter() - start, torch_version=torch.__version__,
                  device=str(device), scope="module + features + AdamW only; NOT whole model")
    if device.type == "cuda":
        report["peak_allocated_MiB"] = torch.cuda.max_memory_allocated(device) / 2**20
        report["peak_reserved_MiB"] = torch.cuda.max_memory_reserved(device) / 2**20
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    _main()
