"""PET/CT 状态感知小波融合模块（无文本、无专家、无额外损失）。

建议位置：models/petct_state_wavelet_fusion.py
依赖：Python >= 3.9，PyTorch >= 2.0；不依赖 pywt、timm 或其他项目文件。

模块边界
========
输入是已经对齐的四尺度 CT 特征，以及已经按状态组装的 PET 特征：
Full 行为真实 PET，Missing 行为 Module-1 的 P_comp（冷启动可为零）。
本模块不调用编码器、原型库、仿射补偿、标签、损失或解码器，不 detach 输入。
它无法从特征值判断 PET 的来源；调用者必须正确组装并显式提供状态。
输入顺序 S1 -> S4；默认通道 (64,128,320,512)，空间 (128,64,32,16)。
输出为相同长度、顺序、形状、device、dtype 的 list[Tensor]。

最新确认设计
============
* CT 基础投影跨状态共享；真实/补偿 PET 投影及低频交互分别独立。
* 每个状态只有一个 MFFA 风格低频单元：Q/K 来自 CT LL，V 含 CT/PET LL。
* Full 高频采用 TEWF 公式(5)的交叉 Q/K、同侧 V，两方向平均；三子带复用参数。
* Full = C + P_real + IDWT(delta_LL, delta_LH, delta_HL, delta_HH)。
* Missing = C + IDWT(delta_LL, 0, 0, 0)；不计算 PET 高频或 Full 交互。
* 原始 C/P 只走未归一化的基础路径；Norm 仅存在于交互分支。
* 所有增量输出层置零，step-0 精确回到 Full=C+P、Missing=C。
  第一轮反向时内部交互梯度为零是零输出初始化的预期行为，输出层更新后打开。
* 低频窗口 8、高频窗口 4；无全空间 HW x HW 注意力。
* 固定正交 Haar 用原生运算与 autograd，不复制参考实现的自定义 backward。

参考与适配（不是整网复现）
========================
1. WFANet, AAAI 2025, DOI:10.1609/aaai.v39i4.32381
   https://ojs.aaai.org/index.php/AAAI/article/view/32381
   https://github.com/Jie-1203/WFANet/blob/master/net_torch.py
   参考 Attention / v_ll_attn 的 Q/K/V 来源、Value 残差及两层 MLP 残差。
   原全空间 attention 改为带 padding mask 的窗口 attention；只选用 LL 路径。
   两模态 Value 混合器、通道压缩、状态分支及外部 C/P 残差是本任务适配。
2. Frequency-Aligned Cross-Modal Learning with Top-K Wavelet Fusion and
   Dynamic Expert Routing for Enhanced Retinal Disease Diagnosis, AAAI 2026
   https://ojs.aaai.org/index.php/AAAI/article/view/37635/41597
   依据论文式(5)-(8)，不是已验证的作者源码移植（未确认该文官方代码）。
   保留 softmax(Qc Kp^T) Vc / softmax(Qp Kc^T) Vp 的原文配对及平均。
   因两路子带严格配准且窗口 token 数相同，此配对形状合法；不擅自替换为另一公式。
   去除专家/路由/图像级分类结构；增加双模态基础残差及零初始化输出投影。

接口与接入
==========
    fusion = PETCTWaveletFusion()
    fused = fusion(ct_feats, pet_for_fusion, pet_available)  # [B], 1=Full,0=Missing
    fused = fusion.forward_full(ct_feats, pet_real_feats)
    fused = fusion.forward_missing(ct_feats, pet_comp_feats)

也可用关键字 is_full=bool_tensor，不能同时传 pet_available 和 is_full。
状态可为 Python bool（整批）、0/1 Python int（整批）、bool/int Tensor[B]。
禁用 None 默认值，以防将 Missing 误当 Full。不要传 missing_mask，除非先取反。
构造一次并注册到 nn.Module，再创建 optimizer；不能在 forward 中临时构造。

检查过的历史基线：AilinLiuilearn/new-train
e1-pspi-ct-affine-smoothl1-add @ 8c39080e95de0c0d24683ed9cf55082685abee1c
该提交 AddFusion 返回四尺度 list，但融合处传入第三参 None：接入时必须改为
Full=True / Missing=False / mixed=pet_available。只替换类名而不传状态会报错。
已确认 feature-level 接口；用户后续 perscale BG8/FG6 改动未在该提交核实。
不改变 Module-1 的 collect/retrieve/affine/建库生命周期与原有训练损失。
DDP 若按整批交替 Full/Missing，会有未使用的分支参数：需 find_unused_parameters=True。

验证与资源测量
==============
    python petct_state_wavelet_fusion.py --self-test
    python petct_state_wavelet_fusion.py --smoke-test --batch-size 2
    python petct_state_wavelet_fusion.py --profile --device cuda --batch-size 16 --amp
    python petct_state_wavelet_fusion.py --profile --device cuda --batch-size 16 --amp --checkpoint
无参数运行仅显示配置与参数量。profile 只测本模块+输入+AdamW，不代表整个训练模型。
数据形状必须已对齐、空间尺寸为正偶数；窗口不整除时自动 padding 并屏蔽无效 key。
验证过的事实应以自测结果为准；无任何 Dice 增益或整网显存保证。
"""

from __future__ import annotations

import argparse
import io
import json
import math
import time
import unittest
from contextlib import nullcontext
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ["PETCTWaveletFusion", "WaveletFusionScale", "haar_dwt2", "haar_idwt2"]
State = Union[bool, int, Tensor]
Bands = Tuple[Tensor, Tensor, Tensor, Tensor]


def _check_image(x: Tensor) -> None:
    if not isinstance(x, Tensor) or x.ndim != 4 or not x.is_floating_point():
        raise ValueError("Expected floating Tensor[B,C,H,W].")
    if any(n <= 0 for n in x.shape) or x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError("B,C,H,W must be positive; Haar requires even H and W.")


def _haar_ll(x: Tensor) -> Tensor:
    # Same LL as haar_dwt2, without materializing discarded high-frequency bands.
    a, b = x[..., 0::2, 0::2] * 0.5, x[..., 0::2, 1::2] * 0.5
    c, d = x[..., 1::2, 0::2] * 0.5, x[..., 1::2, 1::2] * 0.5
    return (a + b) + (c + d)


def haar_dwt2(x: Tensor) -> Bands:
    """Orthonormal Haar. Axis convention: LH varies along rows, HL along columns."""
    _check_image(x)
    a, b = x[..., 0::2, 0::2] * 0.5, x[..., 0::2, 1::2] * 0.5
    c, d = x[..., 1::2, 0::2] * 0.5, x[..., 1::2, 1::2] * 0.5
    return (a + b) + (c + d), (a + b) - (c + d), (a - b) + (c - d), (a - b) - (c - d)


def haar_idwt2(ll: Tensor, lh: Tensor, hl: Tensor, hh: Tensor) -> Tensor:
    """Inverse of haar_dwt2; native autograd, no trainable filters or custom backward."""
    if not isinstance(ll, Tensor) or ll.ndim != 4 or not ll.is_floating_point():
        raise ValueError("Expected floating 4D Haar bands.")
    for x in (lh, hl, hh):
        if x.shape != ll.shape or x.dtype != ll.dtype or x.device != ll.device:
            raise ValueError("All four bands must have identical shapes, dtype and device.")
    ll, lh, hl, hh = ll * 0.5, lh * 0.5, hl * 0.5, hh * 0.5
    a = (ll + lh) + (hl + hh)
    b = (ll + lh) - (hl + hh)
    c = (ll - lh) + (hl - hh)
    d = (ll - lh) - (hl - hh)
    top = torch.stack((a, b), dim=-1).flatten(-2, -1)
    bottom = torch.stack((c, d), dim=-1).flatten(-2, -1)
    return torch.stack((top, bottom), dim=-2).flatten(-3, -2)


def _low_inverse(ll: Tensor) -> Tensor:
    # IDWT(ll, 0, 0, 0), saving three zero tensors and three additions.
    return (ll * 0.5).repeat_interleave(2, -2).repeat_interleave(2, -1)


def _state_mask(state: Optional[State], batch: int, device: torch.device) -> Tensor:
    if isinstance(state, bool):
        return torch.full((batch,), state, dtype=torch.bool, device=device)
    if isinstance(state, int) and state in (0, 1):
        return torch.full((batch,), bool(state), dtype=torch.bool, device=device)
    if not isinstance(state, Tensor):
        raise ValueError("Explicit state required: True/False or bool/integer Tensor[B], 1=Full.")
    integer_types = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if state.dtype not in integer_types or state.ndim != 1 or state.numel() != batch:
        raise ValueError("State must be bool/integer Tensor[B]; floating gates are not states.")
    if not bool(((state == 0) | (state == 1)).all()):
        raise ValueError("State values must be 0=Missing or 1=Full.")
    return state.to(device=device, dtype=torch.bool)


def _windows(x: Tensor, size: int) -> Tuple[Tensor, Optional[Tensor]]:
    b, c, h, w = x.shape
    ph, pw = (-h) % size, (-w) % size
    padded = F.pad(x, (0, pw, 0, ph))
    hp, wp = h + ph, w + pw
    tokens = padded.reshape(b, c, hp // size, size, wp // size, size)
    tokens = tokens.permute(0, 2, 4, 3, 5, 1).reshape(-1, size * size, c)
    valid = None
    if ph or pw:
        mask = torch.ones((b, 1, h, w), dtype=torch.bool, device=x.device)
        mask = F.pad(mask, (0, pw, 0, ph), value=False)
        valid = mask.reshape(b, 1, hp // size, size, wp // size, size)
        valid = valid.permute(0, 2, 4, 3, 5, 1).reshape(-1, size * size)
    return tokens, valid


def _unwindows(tokens: Tensor, size: int, shape: torch.Size) -> Tensor:
    b, c, h, w = shape
    hp, wp = h + (-h) % size, w + (-w) % size
    x = tokens.reshape(b, hp // size, wp // size, size, size, c)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, c, hp, wp)
    return x[..., :h, :w].contiguous()


def _attention(q: Tensor, k: Tensor, v: Tensor, heads: int,
               valid: Optional[Tensor], backend: str) -> Tensor:
    # Inputs [batch_windows,N,d], output same. No batch or cross-window mixing.
    bw, n, d = q.shape
    dh = d // heads
    q, k, v = [x.reshape(bw, n, heads, dh).transpose(1, 2) for x in (q, k, v)]
    mask = None if valid is None else valid[:, None, None, :]
    if backend == "sdpa":
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
    else:
        # Explicit reference backend, FP32 softmax for FP16/BF16. Preserve double for gradcheck.
        work_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=q.device.type, enabled=False):
            logits = (q.to(work_dtype) @ k.to(work_dtype).transpose(-2, -1)) / math.sqrt(dh)
            if mask is not None:
                logits = logits.masked_fill(~mask, float("-inf"))
            result = (logits.softmax(-1) @ v.to(work_dtype)).to(v.dtype)
    return result.transpose(1, 2).reshape(bw, n, d)


def _mlp(dim: int, norm: bool = False) -> nn.Sequential:
    layers = [nn.LayerNorm(dim)] if norm else []
    return nn.Sequential(*layers, nn.Linear(dim, 2 * dim), nn.LeakyReLU(0.01),
                         nn.Linear(2 * dim, dim))


class _LowFrequencyInteraction(nn.Module):
    """Single LL-only MFFA-style unit, with state-specific parameters."""
    def __init__(self, dim: int, out_channels: int, heads: int, window: int, backend: str):
        super().__init__()
        self.heads, self.window, self.backend = heads, window, backend
        self.value_mix = nn.Sequential(nn.Conv2d(2 * dim, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1))
        self.q, self.k, self.v = _mlp(dim, True), _mlp(dim, True), _mlp(dim, True)
        self.mlp_1, self.mlp_2 = _mlp(dim), _mlp(dim)
        self.out = nn.Conv2d(dim, out_channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, ct: Tensor, pet: Tensor) -> Tensor:
        content = self.value_mix(torch.cat((ct, pet), dim=1))
        x, valid = _windows(ct, self.window)
        z, _ = _windows(content, self.window)
        q, k, v = self.q(x), self.k(x), self.v(z)
        a = _attention(q, k, v, self.heads, valid, self.backend)
        t = v + self.mlp_1(a)
        t = t + self.mlp_2(t)
        return self.out(_unwindows(t, self.window, ct.shape))


class _HighFrequencyInteraction(nn.Module):
    """TEWF-equation-inspired shared unit, called separately on LH, HL, HH."""
    def __init__(self, dim: int, hidden: int, out_channels: int, heads: int,
                 window: int, backend: str):
        super().__init__()
        self.heads, self.window, self.backend = heads, window, backend
        self.reduce_ct, self.reduce_pet = nn.Conv2d(dim, hidden, 1), nn.Conv2d(dim, hidden, 1)
        self.norm_ct, self.norm_pet = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.qkv_ct, self.qkv_pet = nn.Linear(hidden, 3 * hidden), nn.Linear(hidden, 3 * hidden)
        self.post_ct, self.post_pet = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.out = nn.Conv2d(2 * hidden, out_channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, ct: Tensor, pet: Tensor) -> Tensor:
        ct, pet = self.reduce_ct(ct), self.reduce_pet(pet)
        xc, valid = _windows(ct, self.window)
        xp, _ = _windows(pet, self.window)
        qc, kc, vc = self.qkv_ct(self.norm_ct(xc)).chunk(3, dim=-1)
        qp, kp, vp = self.qkv_pet(self.norm_pet(xp)).chunk(3, dim=-1)
        # Preserve TEWF Eq.(5): cross Q/K, same-side V, NOT textbook swapped V.
        rc = self.post_ct(_attention(qc, kp, vc, self.heads, valid, self.backend))
        rp = self.post_pet(_attention(qp, kc, vp, self.heads, valid, self.backend))
        r = (rc + rp) * 0.5
        tc = _unwindows(xc + r, self.window, ct.shape)
        tp = _unwindows(xp + r, self.window, pet.shape)
        return self.out(torch.cat((tc, tp), dim=1))


class WaveletFusionScale(nn.Module):
    """One aligned scale. Public input state must be explicit, including eval mode."""
    def __init__(self, channels: int, dim: int, high_dim: int, low_heads: int,
                 high_heads: int, low_window: int = 8, high_window: int = 4,
                 checkpoint_enabled: bool = False, enable_full_high: bool = True,
                 attention_backend: str = "sdpa"):
        super().__init__()
        values = (channels, dim, high_dim, low_heads, high_heads, low_window, high_window)
        if any(type(v) is not int or v <= 0 for v in values):
            raise ValueError("Channels, dimensions, heads and windows must be positive integers.")
        if dim % low_heads or high_dim % high_heads:
            raise ValueError("Attention dimensions must be divisible by head counts.")
        if attention_backend not in ("sdpa", "manual"):
            raise ValueError("attention_backend must be 'sdpa' or 'manual'.")
        if type(checkpoint_enabled) is not bool or type(enable_full_high) is not bool:
            raise ValueError("Checkpoint and high-frequency flags must be bool.")
        self.channels = channels
        self.checkpoint_enabled = checkpoint_enabled
        self.enable_full_high = enable_full_high
        self.ct_project = nn.Conv2d(channels, dim, 1, bias=False)
        self.pet_full_project = nn.Conv2d(channels, dim, 1, bias=False)
        self.pet_missing_project = nn.Conv2d(channels, dim, 1, bias=False)
        self.low_full = _LowFrequencyInteraction(dim, channels, low_heads, low_window, attention_backend)
        self.low_missing = _LowFrequencyInteraction(dim, channels, low_heads, low_window, attention_backend)
        self.high_full = (_HighFrequencyInteraction(dim, high_dim, channels, high_heads,
                                                   high_window, attention_backend)
                          if enable_full_high else None)

    def _full_delta(self, ct: Tensor, pet: Tensor) -> Tensor:
        if self.high_full is None:
            cl, pl = _haar_ll(ct), _haar_ll(pet)
            dl = self.low_full(self.ct_project(cl), self.pet_full_project(pl))
            return _low_inverse(dl)
        cb, pb = haar_dwt2(ct), haar_dwt2(pet)
        dl = self.low_full(self.ct_project(cb[0]), self.pet_full_project(pb[0]))
        dh = [self.high_full(self.ct_project(c), self.pet_full_project(p))
              for c, p in zip(cb[1:], pb[1:])]
        return haar_idwt2(dl, *dh)

    def _missing_delta(self, ct: Tensor, pet: Tensor) -> Tensor:
        cl, pl = _haar_ll(ct), _haar_ll(pet)
        dl = self.low_missing(self.ct_project(cl), self.pet_missing_project(pl))
        return _low_inverse(dl)

    def _run_group(self, ct: Tensor, pet: Tensor, full: bool) -> Tensor:
        operation = self._full_delta if full else self._missing_delta
        if self.checkpoint_enabled and self.training and torch.is_grad_enabled():
            delta = checkpoint(operation, ct, pet, use_reentrant=False)
        else:
            delta = operation(ct, pet)
        # Keep feature contract dtype even under autocast. Original residual untouched.
        base = ct + pet if full else ct
        return base + delta.to(dtype=ct.dtype)

    def _validate(self, ct: Tensor, pet: Tensor) -> None:
        _check_image(ct)
        _check_image(pet)
        if ct.shape != pet.shape or ct.shape[1] != self.channels:
            raise ValueError("CT/PET must have identical aligned shapes and configured channel count.")
        if ct.device != pet.device or ct.dtype != pet.dtype:
            raise ValueError("CT/PET must have identical dtype and device; align upstream explicitly.")
        if self.ct_project.weight.device != ct.device:
            raise ValueError("Move the fusion module and features to the same device.")

    def _forward_rows(self, ct: Tensor, pet: Tensor, full_rows: Tensor, missing_rows: Tensor) -> Tensor:
        if missing_rows.numel() == 0:
            return self._run_group(ct, pet, True)
        if full_rows.numel() == 0:
            return self._run_group(ct, pet, False)
        full = self._run_group(ct.index_select(0, full_rows), pet.index_select(0, full_rows), True)
        missing = self._run_group(ct.index_select(0, missing_rows), pet.index_select(0, missing_rows), False)
        # No cross-patient mixing; index_copy is out of place and preserves gradients.
        result = torch.zeros_like(ct).index_copy(0, full_rows, full)
        return result.index_copy(0, missing_rows, missing)

    def forward(self, ct: Tensor, pet: Tensor, is_full: State) -> Tensor:
        self._validate(ct, pet)
        mask = _state_mask(is_full, ct.shape[0], ct.device)
        return self._forward_rows(ct, pet, mask.nonzero(as_tuple=True)[0], (~mask).nonzero(as_tuple=True)[0])


class PETCTWaveletFusion(nn.Module):
    """Feature-pyramid fusion. Returns a list ready for the existing shared decoder.

    Defaults implement the approved four-scale design. Parameter initialization
    follows torch's global RNG; torch.manual_seed(...) before construction makes
    it reproducible without this module changing the caller's seed.
    """
    def __init__(self, channels: Sequence[int] = (64, 128, 320, 512),
                 low_dims: Sequence[int] = (32, 64, 96, 128),
                 high_dims: Sequence[int] = (16, 32, 48, 64),
                 low_heads: Sequence[int] = (2, 4, 4, 4),
                 high_heads: Sequence[int] = (1, 2, 2, 2),
                 low_window: int = 8, high_window: int = 4,
                 checkpoint_scales: Union[bool, Sequence[bool]] = False,
                 enable_full_high: bool = True, attention_backend: str = "sdpa"):
        super().__init__()
        self.channels = tuple(channels)
        n = len(self.channels)
        if n == 0 or any(len(x) != n for x in (low_dims, high_dims, low_heads, high_heads)):
            raise ValueError("All per-scale settings must have the same nonzero length.")
        cp = (checkpoint_scales,) * n if isinstance(checkpoint_scales, bool) else tuple(checkpoint_scales)
        if len(cp) != n:
            raise ValueError("checkpoint_scales must be bool or one bool per scale.")
        self.config = dict(channels=list(channels), low_dims=list(low_dims), high_dims=list(high_dims),
                           low_heads=list(low_heads), high_heads=list(high_heads), low_window=low_window,
                           high_window=high_window, checkpoint_scales=list(cp),
                           enable_full_high=enable_full_high, attention_backend=attention_backend)
        self.scales = nn.ModuleList([
            WaveletFusionScale(c, d, dh, hl, hh, low_window, high_window, ck,
                               enable_full_high, attention_backend)
            for c, d, dh, hl, hh, ck in zip(channels, low_dims, high_dims, low_heads, high_heads, cp)
        ])

    def forward(self, ct_feats: Sequence[Tensor], pet_feats: Sequence[Tensor],
                pet_available: Optional[State] = None, *, is_full: Optional[State] = None) -> List[Tensor]:
        if is_full is not None and pet_available is not None:
            raise ValueError("Pass only one of pet_available and is_full.")
        state = pet_available if is_full is None else is_full
        if not isinstance(ct_feats, (list, tuple)) or not isinstance(pet_feats, (list, tuple)):
            raise ValueError("Features must be a list/tuple ordered S1 -> S4.")
        if len(ct_feats) != len(self.scales) or len(pet_feats) != len(self.scales):
            raise ValueError("Feature count must match configured scales.")
        # Validate everything before doing any expensive computation.
        for block, c, p in zip(self.scales, ct_feats, pet_feats):
            block._validate(c, p)
            if c.shape[0] != ct_feats[0].shape[0] or c.device != ct_feats[0].device or c.dtype != ct_feats[0].dtype:
                raise ValueError("All scales must share batch size, dtype and device.")
        mask = _state_mask(state, ct_feats[0].shape[0], ct_feats[0].device)
        full_rows, missing_rows = mask.nonzero(as_tuple=True)[0], (~mask).nonzero(as_tuple=True)[0]
        return [block._forward_rows(c, p, full_rows, missing_rows)
                for block, c, p in zip(self.scales, ct_feats, pet_feats)]

    def forward_full(self, ct_feats: Sequence[Tensor], pet_real_feats: Sequence[Tensor]) -> List[Tensor]:
        return self(ct_feats, pet_real_feats, True)

    def forward_missing(self, ct_feats: Sequence[Tensor], pet_comp_feats: Sequence[Tensor]) -> List[Tensor]:
        return self(ct_feats, pet_comp_feats, False)

    def parameter_report(self) -> dict:
        return {"total": sum(p.numel() for p in self.parameters()),
                "per_scale": [sum(p.numel() for p in s.parameters()) for s in self.scales]}


# ---------------------------------------------------------------------------
# Contract tests: embedded to keep delivery a single, independently runnable file.
# Helpers below are never called on import or in a normal model forward.
# ---------------------------------------------------------------------------
def _tiny(checkpoint_on: bool = False, backend: str = "sdpa") -> PETCTWaveletFusion:
    return PETCTWaveletFusion((8, 12, 16, 24), (8, 8, 12, 16), (4, 4, 6, 8),
                             (2, 2, 3, 4), (1, 1, 2, 2), low_window=4, high_window=2,
                             checkpoint_scales=checkpoint_on, attention_backend=backend)


def _open_outputs(model: nn.Module) -> None:
    # Tests only: emulate trained, nonzero increment heads to avoid vacuous zero-init tests.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (_LowFrequencyInteraction, _HighFrequencyInteraction)):
                nn.init.normal_(module.out.weight, std=0.025)
                nn.init.normal_(module.out.bias, std=0.01)


def _features(batch: int = 4, requires_grad: bool = False) -> Tuple[List[Tensor], List[Tensor]]:
    # Includes window padding and nonsquare inputs, but preserves the even Haar contract.
    shapes = [(8, 18, 14), (12, 10, 8), (16, 6, 6), (24, 2, 4)]
    c = [torch.randn(batch, ch, h, w, requires_grad=requires_grad) for ch, h, w in shapes]
    p = [torch.randn_like(x, requires_grad=requires_grad) for x in c]
    return c, p


class _ContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2023)

    def assertFinite(self, tensors):
        for x in tensors:
            self.assertIsNotNone(x)
            self.assertTrue(bool(torch.isfinite(x).all()))

    def test_haar_reconstruction_and_gradients(self):
        x = torch.randn(1, 2, 4, 6, dtype=torch.float64, requires_grad=True)
        torch.testing.assert_close(haar_idwt2(*haar_dwt2(x)), x, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(_haar_ll(x), haar_dwt2(x)[0])
        self.assertTrue(torch.autograd.gradcheck(haar_dwt2, (x,), fast_mode=True))
        bands = tuple(t.detach().requires_grad_() for t in haar_dwt2(x))
        self.assertTrue(torch.autograd.gradcheck(haar_idwt2, bands, fast_mode=True))
        zeros = torch.zeros_like(bands[0])
        torch.testing.assert_close(_low_inverse(bands[0]), haar_idwt2(bands[0], zeros, zeros, zeros))

    def test_zero_init_all_states_and_scales(self):
        model = _tiny()
        c, p = _features()
        mask = torch.tensor([True, False, True, False])
        for state in (True, False, mask):
            outputs = model(c, p, state)
            flags = _state_mask(state, 4, c[0].device)
            for out, x, y in zip(outputs, c, p):
                expected = torch.where(flags[:, None, None, None], x + y, x)
                torch.testing.assert_close(out, expected, atol=0, rtol=0)

    def test_mixed_matches_split_and_permutation(self):
        model = _tiny().eval()
        _open_outputs(model)
        c, p = _features()
        mask = torch.tensor([False, True, True, False])
        mixed = model(c, p, is_full=mask)
        full = model.forward_full([x[mask] for x in c], [x[mask] for x in p])
        missing = model.forward_missing([x[~mask] for x in c], [x[~mask] for x in p])
        order = torch.tensor([2, 0, 3, 1])
        perm = model([x[order] for x in c], [x[order] for x in p], mask[order])
        for z, f, m, shuffled in zip(mixed, full, missing, perm):
            torch.testing.assert_close(z[mask], f)
            torch.testing.assert_close(z[~mask], m)
            torch.testing.assert_close(z[order], shuffled)

    def test_nonzero_forward_backward_all_scales(self):
        model = _tiny()
        _open_outputs(model)
        c, p = _features(requires_grad=True)
        outputs = model(c, p, torch.tensor([1, 0, 0, 1]))
        sum(x.square().mean() for x in outputs).backward()
        self.assertFinite(outputs + [x.grad for x in c + p])
        for scale, pet in zip(model.scales, p):
            self.assertGreater(float(pet.grad[[1, 2]].abs().sum()), 0)
            for prefix in ("ct_project", "pet_full_project", "pet_missing_project", "low_full", "low_missing", "high_full"):
                grads = [v.grad for k, v in scale.named_parameters() if k.startswith(prefix)]
                self.assertFinite(grads)
                self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)

    def test_forbidden_branch_calls_and_inactive_gradients(self):
        model = _tiny()
        _open_outputs(model)
        c, p = _features(requires_grad=True)
        def forbidden(*args):
            raise AssertionError("Inactive state branch was executed")
        for full_state in (False, True):
            model.zero_grad(set_to_none=True)
            handles = []
            for s in model.scales:
                modules = [s.low_missing, s.pet_missing_project] if full_state else [s.low_full, s.high_full, s.pet_full_project]
                handles.extend(m.register_forward_pre_hook(forbidden) for m in modules)
            try:
                sum(t.square().mean() for t in model(c, p, full_state)).backward()
            finally:
                for h in handles:
                    h.remove()
            for name, param in model.named_parameters():
                inactive = ("missing" in name) if full_state else ("full" in name)
                if inactive:
                    self.assertIsNone(param.grad, name)

    def test_missing_ignores_pet_high_frequencies_after_training(self):
        model = _tiny().eval()
        _open_outputs(model)
        c, p = _features(batch=2)
        perturbed = []
        for x in p:
            ll = _haar_ll(x)
            noise = haar_idwt2(torch.zeros_like(ll), torch.randn_like(ll),
                              torch.randn_like(ll), torch.randn_like(ll))
            perturbed.append(x + noise)
        a, b = model(c, p, False), model(c, perturbed, False)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-6)
        # Also ensure LL is actually used, not merely ignored by zero output heads.
        changed = model(c, [x + torch.randn_like(_haar_ll(x)).repeat_interleave(2, -2).repeat_interleave(2, -1) for x in p], False)
        self.assertGreater(sum(float((x - y).abs().sum().detach()) for x, y in zip(a, changed)), 1e-5)

    def test_upstream_real_pet_isolation_and_affine_gradient(self):
        model = _tiny()
        _open_outputs(model)
        c, real = _features(requires_grad=True)
        beta = [torch.randn_like(x, requires_grad=True) for x in c]
        mask = torch.tensor([True, False, True, False])
        # Minimal differentiable stand-in for Module-1's CT-conditioned P_comp.
        assembled = [torch.where(mask[:, None, None, None], r, 0.25 * x + b)
                     for x, r, b in zip(c, real, beta)]
        outputs = model(c, assembled, mask)
        sum(x[~mask].square().mean() for x in outputs).backward()
        for r, b in zip(real, beta):
            torch.testing.assert_close(r.grad, torch.zeros_like(r), atol=0, rtol=0)
            self.assertGreater(float(b.grad[~mask].abs().sum()), 0)
            torch.testing.assert_close(b.grad[mask], torch.zeros_like(b.grad[mask]), atol=0, rtol=0)
        altered = [torch.where(mask[:, None, None, None], r.detach() * 10, a.detach()) for r, a in zip(real, assembled)]
        second = model([x.detach() for x in c], altered, mask)
        for a, b in zip(outputs, second):
            torch.testing.assert_close(a[~mask], b[~mask])

    def test_checkpoint_forward_backward_equivalence(self):
        a, b = _tiny(), _tiny(True)
        _open_outputs(a)
        b.load_state_dict(a.state_dict())
        c, p = _features(requires_grad=True)
        c2 = [x.detach().clone().requires_grad_() for x in c]
        p2 = [x.detach().clone().requires_grad_() for x in p]
        state = torch.tensor([1, 0, 1, 0])
        x, y = a(c, p, state), b(c2, p2, state)
        sum(v.square().mean() for v in x).backward()
        sum(v.square().mean() for v in y).backward()
        for v, w in zip(x + [t.grad for t in c + p], y + [t.grad for t in c2 + p2]):
            torch.testing.assert_close(v, w)
        for v, w in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(v.grad, w.grad)

    def test_sdpa_matches_manual_including_padding(self):
        a, b = _tiny(), _tiny(backend="manual")
        _open_outputs(a)
        b.load_state_dict(a.state_dict())
        c, p = _features()
        for x, y in zip(a(c, p, torch.tensor([1, 0, 1, 0])), b(c, p, torch.tensor([1, 0, 1, 0]))):
            torch.testing.assert_close(x, y, atol=3e-6, rtol=3e-6)

    def test_state_dict_roundtrip(self):
        a = _tiny().eval()
        _open_outputs(a)
        stream = io.BytesIO()
        torch.save(a.state_dict(), stream)
        stream.seek(0)
        b = PETCTWaveletFusion(**a.config).eval()
        b.load_state_dict(torch.load(stream, weights_only=True))
        c, p = _features()
        for x, y in zip(a(c, p, False), b(c, p, False)):
            torch.testing.assert_close(x, y, atol=0, rtol=0)

    def test_validation(self):
        model = _tiny()
        c, p = _features()
        bad = (None, 2, torch.tensor([1., 0., 1., 0.]), torch.tensor([1, 0]), torch.tensor([1, 2, 0, 0]))
        for state in bad:
            with self.assertRaises(ValueError):
                model(c, p, state)
        with self.assertRaises(ValueError):
            model(c, p, True, is_full=False)
        with self.assertRaises(ValueError):
            model(c[:-1], p, False)
        with self.assertRaises(ValueError):
            model([c[0][..., :-1, :]] + c[1:], p, False)
        with self.assertRaises(ValueError):
            WaveletFusionScale(8, 7, 4, 2, 1)
        with self.assertRaises(ValueError):
            WaveletFusionScale(8, 8, 4, 2, 1, attention_backend="invalid")

    def test_zero_heads_learn_then_upstream_opens(self):
        model = _tiny()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
        c, p = _features()
        target = [torch.randn_like(x) for x in c]
        state = torch.tensor([1, 0, 1, 0])
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            outputs = model(c, p, state)
            sum(F.mse_loss(x, y) for x, y in zip(outputs, target)).backward()
            self.assertFinite([v.grad for v in model.parameters()])
            for s in model.scales:
                self.assertGreater(float(s.low_missing.out.weight.grad.abs().sum()), 0)
                internal = float(s.low_missing.q[1].weight.grad.abs().sum())
                if step == 0:
                    self.assertEqual(internal, 0)
                else:
                    self.assertGreater(internal, 0)
            opt.step()

    def test_cpu_bfloat16_autocast(self):
        model = _tiny()
        _open_outputs(model)
        c, p = _features(requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            outputs = model(c, p, torch.tensor([1, 0, 1, 0]))
            loss = sum(x.square().mean() for x in outputs)
        loss.backward()
        self.assertFinite(outputs + [x.grad for x in c + p] + [x.grad for x in model.parameters()])
        self.assertTrue(all(x.dtype == torch.float32 for x in outputs))

    def test_bfloat16_feature_input_preserves_dtype(self):
        model = _tiny()
        _open_outputs(model)
        c, p = _features(batch=2)
        c = [x.to(torch.bfloat16).requires_grad_() for x in c]
        p = [x.to(torch.bfloat16).requires_grad_() for x in p]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            outputs = model(c, p, torch.tensor([1, 0]))
            loss = sum(x.float().square().mean() for x in outputs)
        loss.backward()
        self.assertTrue(all(x.dtype == torch.bfloat16 for x in outputs))
        self.assertFinite(outputs + [x.grad for x in c + p])

    def test_high_ablation_preserves_full_pet_base(self):
        model = PETCTWaveletFusion(**dict(_tiny().config, enable_full_high=False))
        c, p = _features()
        for x, ct, pet in zip(model(c, p, True), c, p):
            torch.testing.assert_close(x, ct + pet, atol=0, rtol=0)
        self.assertTrue(all(s.high_full is None for s in model.scales))


def _run_smoke_or_profile(args) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no GPU result can be reported.")
    torch.manual_seed(2023)
    model = PETCTWaveletFusion(checkpoint_scales=args.checkpoint).to(device).train()
    _open_outputs(model)  # Exercise trained paths; not an initialization-equivalence benchmark.
    c = [torch.randn(args.batch_size, ch, s, s, device=device, requires_grad=True)
         for ch, s in zip(model.channels, (128, 64, 32, 16))]
    p = [torch.randn_like(x, requires_grad=True) for x in c]
    state = (torch.arange(args.batch_size, device=device) % 2 == 0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    use_scaler = args.amp and device.type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    else:  # Compatibility with torch 2.0/2.1.
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    def check_gradients(loss):
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite smoke/profile loss")
        for param in model.parameters():
            if param.grad is not None and not bool(torch.isfinite(param.grad).all()):
                raise RuntimeError("Non-finite parameter gradient in smoke/profile")
        for source in c + p:
            if source.grad is None or not bool(torch.isfinite(source.grad).all()):
                raise RuntimeError("Missing/non-finite feature gradient in smoke/profile")

    def step(validate=False):
        opt.zero_grad(set_to_none=True)
        for x in c + p:
            x.grad = None
        context = torch.autocast(device.type, dtype=amp_dtype) if args.amp else nullcontext()
        with context:
            y = model(c, p, state)
            loss = sum(t.float().square().mean() for t in y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if validate:
            check_gradients(loss)
            for x, source in zip(y, c):
                if x.shape != source.shape or not bool(torch.isfinite(x).all()):
                    raise RuntimeError("Invalid output in smoke/profile")
        scaler.step(opt)
        scaler.update()
        return loss.detach()

    step(validate=True)  # Validate and allocate Adam states before measuring.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    steps = args.steps if args.profile else 1
    for _ in range(steps):
        loss = step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peaks = {} if device.type != "cuda" else dict(
        peak_allocated_MiB=torch.cuda.max_memory_allocated(device) / 2**20,
        peak_reserved_MiB=torch.cuda.max_memory_reserved(device) / 2**20)
    check_gradients(loss)  # Keep per-parameter CPU/GPU synchronization out of timing.
    result = {"scope": "module + synthetic feature inputs + AdamW; NOT whole segmentation model",
              "torch": torch.__version__, "device": str(device), "batch": args.batch_size,
              "full_rows": int(state.sum()), "missing_rows": int((~state).sum()),
              "amp": args.amp, "checkpoint": args.checkpoint, "loss": float(loss),
              "mean_step_seconds": elapsed / steps,
              "parameters": model.parameter_report()}
    result.update(peaks)
    return result


def _main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--profile", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.steps, args.threads) <= 0:
        parser.error("batch-size, steps, threads must be positive")
    torch.set_num_threads(args.threads)
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(_ContractTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.smoke_test or args.profile:
        print(json.dumps(_run_smoke_or_profile(args), ensure_ascii=False, indent=2))
    else:
        model = PETCTWaveletFusion()
        print(json.dumps({"config": model.config, "parameters": model.parameter_report()}, indent=2))


if __name__ == "__main__":
    _main()
