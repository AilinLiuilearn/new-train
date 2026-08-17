"""
TRDF: Text-guided Reliability-aware Dual-path Fusion
====================================================

A standalone, drop-in four-scale PET-CT fusion module designed to sit AFTER
the existing PET calibration / compensation module.

Main input contract
-------------------
ct_feats  : list[Tensor], four CT feature maps
pet_feats : list[Tensor], four calibrated PET feature maps
            - Full:    calibrated real PET
            - Missing: calibrated compensated/proxy PET
mode      : "full", "missing", or "auto"
pet_available : optional [B] tensor for mode="auto" (1=real PET, 0=proxy PET)

Main output contract
--------------------
By default:
    fused_feats : list[Tensor], same shapes as ct_feats / pet_feats

With return_aux=True:
    fused_feats, aux

The module is intentionally independent from CPPI internals.  It does not
consume prototype IDs, retrieval scores, CT-reference features, or calibration
intermediate variables.

Design
------
1) PET reliability estimation on ORIGINAL channels:
   - PET self uncertainty
   - CT-conditioned PET uncertainty
   - visual confidence generation

2) Optional source-aware text calibration:
   - real PET prompt
   - compensated PET prompt
   Text is only a source prior; it does NOT create a spatial reliability map.
   The last text-calibration layer is zero-initialized, so enabling text starts
   as an identity calibration.

3) Reliability is used ONCE:
       P_rel = R * P

4) Dual-path fusion:
   - reliable direct path: C + P_rel
   - low-memory synergistic path:
       linear cross-attention(Q=C, K=P_rel, V=P_rel)
   - final:
       F_out = C + P_rel + A

The linear attention never forms an H*W by H*W attention matrix.

Text integration
----------------
Two local/offline options are supported:

A. Precomputed embeddings (recommended for maximum independence)
   Save a .pt file such as:
       {
           "real":  Tensor[D] or Tensor[1,D],
           "proxy": Tensor[D] or Tensor[1,D],
       }

B. Local HuggingFace-compatible text encoder directory
   Put it anywhere (for example parallel to ./models) and pass its path.
   The encoder is loaded on CPU once during initialization, the two prompts are
   encoded, then the encoder is released. No text model stays in the training
   graph.

If use_text_prior=False, no text dependency is loaded at all.

Reference inspiration
---------------------
- UMFNet (CVPR 2026): Gaussian uncertainty heads + confidence generation.
  This implementation keeps reliability estimation at the original feature
  channels and removes stochastic reparameterization.
- The final direct + synergistic fusion is adapted to the PET-CT setting and
  uses linear cross-attention for low memory.

This file contains no dependency on the user's repository and can be executed:
    python trdf_fusion.py

For a text-off ablation (default), the demo has no external dependency beyond
PyTorch.
"""

from __future__ import annotations

import argparse
import gc
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


REAL_PET_PROMPT = (
    "This is real PET, providing patient-specific lesion metabolic information "
    "for tumor segmentation."
)

PROXY_PET_PROMPT = (
    "This is compensated PET for missing PET, providing smoother metabolic information "
    "and shared lesion priors for tumor segmentation."
)


def _sanitize(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


def _check_4d(name: str, x: torch.Tensor) -> None:
    if not torch.is_tensor(x) or x.ndim != 4:
        raise ValueError(f"{name} must be a 4D tensor [B,C,H,W], got {type(x)} / {getattr(x, 'shape', None)}")


def _check_pair(name: str, ct: torch.Tensor, pet: torch.Tensor) -> None:
    _check_4d(f"{name}.ct", ct)
    _check_4d(f"{name}.pet", pet)
    if ct.shape != pet.shape:
        raise ValueError(f"{name}: CT/PET must have identical shape, got ct={tuple(ct.shape)} pet={tuple(pet.shape)}")


class BottleneckConvMlp(nn.Module):
    """Lightweight 1x1 bottleneck MLP that preserves the input/output channel count."""

    def __init__(self, channels: int, ratio: int = 4, min_hidden: int = 16):
        super().__init__()
        hidden = max(channels // ratio, min_hidden)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class PETSelfGaussianHead(nn.Module):
    """
    PET intrinsic uncertainty head.

    Input/output stay at the ORIGINAL scale channel count C.
    No C -> d reliability projection is used.
    """

    def __init__(
        self,
        channels: int,
        logvar_clamp: Tuple[float, float] = (-8.0, 8.0),
    ):
        super().__init__()
        self.logvar_clamp = tuple(float(v) for v in logvar_clamp)

        self.local = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GELU(),
        )
        self.mu_norm = nn.BatchNorm2d(channels)
        self.lv_norm = nn.BatchNorm2d(channels)
        self.mu_head = BottleneckConvMlp(channels)
        self.lv_head = BottleneckConvMlp(channels)

    def forward(self, pet: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = pet + self.local(pet)

        # Mean representation is residual to preserve the original PET evidence.
        mu = pet + self.mu_head(self.mu_norm(h))
        logvar = self.lv_head(self.lv_norm(h))
        logvar = torch.clamp(logvar, self.logvar_clamp[0], self.logvar_clamp[1])
        return mu, logvar


class CTConditionedPETGaussianHead(nn.Module):
    """
    Learn PET uncertainty under the current CT context.

    Important:
    - no cosine similarity
    - no |CT-PET|
    - no CT*PET hand-crafted descriptor
    - no 2C concatenation tensor

    CT and PET are independently mapped at C channels and added before the
    local context block, letting the network learn useful cross-modal relations.
    """

    def __init__(
        self,
        channels: int,
        logvar_clamp: Tuple[float, float] = (-8.0, 8.0),
    ):
        super().__init__()
        self.logvar_clamp = tuple(float(v) for v in logvar_clamp)

        self.ct_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.pet_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.pre_norm = nn.BatchNorm2d(channels)

        self.local = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GELU(),
        )

        self.mu_norm = nn.BatchNorm2d(channels)
        self.lv_norm = nn.BatchNorm2d(channels)
        self.mu_head = BottleneckConvMlp(channels)
        self.lv_head = BottleneckConvMlp(channels)

    def forward(
        self,
        ct: torch.Tensor,
        pet_mu: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.ct_proj(ct) + self.pet_proj(pet_mu)
        h = self.pre_norm(h)
        h = h + self.local(h)

        mu_ctx = pet_mu + self.mu_head(self.mu_norm(h))
        logvar_ctx = self.lv_head(self.lv_norm(h))
        logvar_ctx = torch.clamp(
            logvar_ctx,
            self.logvar_clamp[0],
            self.logvar_clamp[1],
        )
        return mu_ctx, logvar_ctx


class PETConfidenceGenerator(nn.Module):
    """
    Convert PET self/context uncertainty into a spatial visual reliability map.

    Both uncertainty maps have shape [B,1,H,W].
    Output:
        r_vis in (0,1), shape [B,1,H,W]
    """

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(2, hidden, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )

    @staticmethod
    def confidence_from_logvar(logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Stable positive uncertainty.  We intentionally avoid exp(exp(.)).
        uncertainty = F.softplus(logvar).mean(dim=1, keepdim=True)
        confidence = torch.exp(-uncertainty)
        return confidence, uncertainty

    def forward(
        self,
        self_logvar: torch.Tensor,
        ctx_logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        conf_self, u_self = self.confidence_from_logvar(self_logvar)
        conf_ctx, u_ctx = self.confidence_from_logvar(ctx_logvar)

        x = torch.cat([conf_self, conf_ctx], dim=1)
        r_vis = torch.sigmoid(self.head(x))

        aux = {
            "self_uncertainty": u_self,
            "context_uncertainty": u_ctx,
            "self_confidence": conf_self,
            "context_confidence": conf_ctx,
        }
        return r_vis, aux


class PETReliabilityEstimator(nn.Module):
    """UMFNet-inspired PET reliability estimator for one feature scale."""

    def __init__(
        self,
        channels: int,
        logvar_clamp: Tuple[float, float] = (-8.0, 8.0),
        confidence_hidden: int = 8,
    ):
        super().__init__()
        self.self_head = PETSelfGaussianHead(channels, logvar_clamp)
        self.context_head = CTConditionedPETGaussianHead(channels, logvar_clamp)
        self.confidence = PETConfidenceGenerator(confidence_hidden)

    def forward(
        self,
        ct: torch.Tensor,
        pet: torch.Tensor,
        return_aux: bool = False,
    ):
        _check_pair("PETReliabilityEstimator", ct, pet)

        pet_mu, pet_logvar = self.self_head(pet)
        _, ctx_logvar = self.context_head(ct, pet_mu)
        r_vis, conf_aux = self.confidence(pet_logvar, ctx_logvar)

        if not return_aux:
            return r_vis

        aux = dict(conf_aux)
        aux["visual_reliability"] = r_vis
        return r_vis, aux


class StaticPETTextPrior(nn.Module):
    """
    Offline/static text-prior provider.

    The two prompt embeddings are computed ONCE at initialization and stored as
    non-trainable buffers.  No language model stays in the training graph.

    Supported backends:
        - "precomputed": load two embeddings from a .pt file
        - "hf_local":    encode the prompts with a local HF-compatible model
    """

    def __init__(
        self,
        backend: str,
        embedding_path: Optional[str] = None,
        text_model_path: Optional[str] = None,
        real_prompt: str = REAL_PET_PROMPT,
        proxy_prompt: str = PROXY_PET_PROMPT,
        trust_remote_code: bool = True,
    ):
        super().__init__()
        backend = str(backend).lower().strip()
        self.backend = backend
        self.real_prompt = real_prompt
        self.proxy_prompt = proxy_prompt

        if backend == "precomputed":
            real, proxy = self._load_precomputed(embedding_path)
        elif backend == "hf_local":
            real, proxy = self._encode_hf_local(
                text_model_path=text_model_path,
                real_prompt=real_prompt,
                proxy_prompt=proxy_prompt,
                trust_remote_code=trust_remote_code,
            )
        else:
            raise ValueError(
                f"Unsupported text backend={backend!r}. "
                "Use 'precomputed' or 'hf_local'."
            )

        real = self._normalize_embedding(real)
        proxy = self._normalize_embedding(proxy)

        if real.shape != proxy.shape:
            raise ValueError(
                f"Real/proxy text embeddings must have the same shape, "
                f"got {tuple(real.shape)} and {tuple(proxy.shape)}"
            )

        self.register_buffer("real_embedding", real, persistent=True)
        self.register_buffer("proxy_embedding", proxy, persistent=True)
        self.output_dim = int(real.shape[-1])

    @staticmethod
    def _normalize_embedding(x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.detach().float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[0] != 1:
            raise ValueError(
                f"Each static prompt embedding must have shape [D] or [1,D], got {tuple(x.shape)}"
            )
        return F.normalize(x, p=2, dim=-1, eps=1e-6)

    @staticmethod
    def _load_precomputed(
        embedding_path: Optional[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not embedding_path:
            raise ValueError("embedding_path is required for backend='precomputed'")
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Text embedding file not found: {embedding_path}")

        data = torch.load(embedding_path, map_location="cpu")
        if not isinstance(data, dict):
            raise ValueError(
                "Precomputed embedding file must be a dict containing real/proxy embeddings."
            )

        real = data.get("real", data.get("real_text_embedding"))
        proxy = data.get("proxy", data.get("proxy_text_embedding"))
        if real is None or proxy is None:
            raise KeyError(
                "Expected keys {'real','proxy'} or "
                "{'real_text_embedding','proxy_text_embedding'} in the .pt file."
            )
        return real, proxy

    @staticmethod
    @torch.no_grad()
    def _encode_hf_local(
        text_model_path: Optional[str],
        real_prompt: str,
        proxy_prompt: str,
        trust_remote_code: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not text_model_path:
            raise ValueError("text_model_path is required for backend='hf_local'")
        if not os.path.exists(text_model_path):
            raise FileNotFoundError(f"Local text model path not found: {text_model_path}")

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "backend='hf_local' requires transformers. "
                "Install it or use backend='precomputed'."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(
            text_model_path,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModel.from_pretrained(
            text_model_path,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        model.eval()
        model.to("cpu")

        encoded = tokenizer(
            [real_prompt, proxy_prompt],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        outputs = model(**encoded)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            hidden = outputs.last_hidden_state
            mask = encoded.get("attention_mask")
            if mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask = mask.unsqueeze(-1).to(dtype=hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        pooled = pooled.detach().float().cpu()
        real, proxy = pooled[0:1], pooled[1:2]

        # Do not retain the language model in memory.
        del model, tokenizer, outputs, encoded, pooled
        gc.collect()
        return real, proxy

    def get(
        self,
        batch_size: int,
        mode: str,
        device: torch.device,
        dtype: torch.dtype,
        pet_available: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mode = str(mode).lower().strip()

        real = self.real_embedding.to(device=device, dtype=dtype)
        proxy = self.proxy_embedding.to(device=device, dtype=dtype)

        if mode in ("full", "real"):
            return real.expand(batch_size, -1)
        if mode in ("missing", "proxy"):
            return proxy.expand(batch_size, -1)
        if mode != "auto":
            raise ValueError(f"Unsupported mode={mode!r}")

        if pet_available is None:
            raise ValueError("pet_available is required when mode='auto'")
        availability = pet_available.to(device=device).view(-1)
        if availability.numel() != batch_size:
            raise ValueError(
                f"pet_available must have B={batch_size} entries, got {availability.numel()}"
            )
        if not torch.all((availability == 0) | (availability == 1)):
            raise ValueError("pet_available values must be 0 or 1")

        availability = availability.to(dtype=dtype).unsqueeze(1)
        return availability * real + (1.0 - availability) * proxy


class TextReliabilityCalibrator(nn.Module):
    """
    Source-text calibration of visual reliability.

    One shared MLP outputs (a_s, b_s) for every scale.
    Final layer is zero-initialized:
        gamma = 1 + tanh(a_s) = 1
        beta  = tanh(b_s)     = 0
    at initialization, so R = R_vis initially.
    """

    def __init__(
        self,
        text_dim: int,
        num_scales: int = 4,
        hidden_dim: int = 128,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_scales = int(num_scales)
        self.eps = float(eps)

        hidden_dim = min(int(hidden_dim), max(32, int(text_dim)))
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * self.num_scales),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        r_vis: torch.Tensor,
        text_embedding: torch.Tensor,
        scale_idx: int,
    ) -> torch.Tensor:
        if not (0 <= scale_idx < self.num_scales):
            raise IndexError(f"scale_idx={scale_idx} out of range")

        params = self.mlp(text_embedding)
        params = params.view(text_embedding.shape[0], self.num_scales, 2)
        a = params[:, scale_idx, 0].view(-1, 1, 1, 1)
        b = params[:, scale_idx, 1].view(-1, 1, 1, 1)

        gamma = 1.0 + torch.tanh(a)  # (0,2)
        beta = torch.tanh(b)         # (-1,1)

        r = r_vis.clamp(self.eps, 1.0 - self.eps)
        logits = torch.log(r) - torch.log1p(-r)
        calibrated = torch.sigmoid(gamma * logits + beta)
        return calibrated


class LinearCrossAttention2d(nn.Module):
    """
    Low-memory CT->PET linear cross-attention.

    Q comes from CT.
    K/V come from already reliability-filtered PET.

    It never creates [B, N, N].  The core relation matrix is [B, d, d].

    Zero-input safety:
    - all Q/K/V/output projections use bias=False
    - V local depthwise conv uses bias=False
    Therefore, when reliable PET is exactly zero, the synergy output is zero.
    """

    def __init__(
        self,
        in_channels: int,
        attn_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.attn_dim = int(attn_dim)
        self.eps = float(eps)

        self.q_proj = nn.Conv2d(in_channels, attn_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(in_channels, attn_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(in_channels, attn_dim, 1, bias=False)

        self.v_local = nn.Conv2d(
            attn_dim,
            attn_dim,
            kernel_size=3,
            padding=1,
            groups=attn_dim,
            bias=False,
        )
        self.out_proj = nn.Conv2d(attn_dim, in_channels, 1, bias=False)

        # Small residual initialization: the new synergy path starts close to
        # the reliable direct-fusion baseline without blocking gradients.
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-3)

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(
        self,
        ct: torch.Tensor,
        pet_reliable: torch.Tensor,
    ) -> torch.Tensor:
        _check_pair("LinearCrossAttention2d", ct, pet_reliable)

        b, _, h, w = ct.shape
        n = h * w

        q = self.q_proj(ct)
        k = self.k_proj(pet_reliable)
        v = self.v_local(self.v_proj(pet_reliable))

        # [B,d,H,W] -> [B,N,d]
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)

        q_phi = self._phi(q)
        k_phi = self._phi(k)

        # Core relation: [B,d,N] @ [B,N,d] -> [B,d,d]
        kv = torch.bmm(k_phi.transpose(1, 2), v)

        # Linear-attention normalizer.
        k_sum = k_phi.sum(dim=1)  # [B,d]
        denom = torch.bmm(q_phi, k_sum.unsqueeze(-1)).clamp_min(self.eps)

        out = torch.bmm(q_phi, kv) / denom
        out = out.transpose(1, 2).reshape(b, self.attn_dim, h, w)
        out = self.out_proj(out)
        return _sanitize(out)


class TRDFScale(nn.Module):
    """One-scale TRDF block."""

    def __init__(
        self,
        channels: int,
        attn_dim: int,
        logvar_clamp: Tuple[float, float] = (-8.0, 8.0),
        confidence_hidden: int = 8,
        attention_eps: float = 1e-6,
    ):
        super().__init__()
        self.reliability = PETReliabilityEstimator(
            channels=channels,
            logvar_clamp=logvar_clamp,
            confidence_hidden=confidence_hidden,
        )
        self.synergy = LinearCrossAttention2d(
            in_channels=channels,
            attn_dim=attn_dim,
            eps=attention_eps,
        )

    def forward(
        self,
        ct: torch.Tensor,
        pet: torch.Tensor,
        text_embedding: Optional[torch.Tensor],
        text_calibrator: Optional[TextReliabilityCalibrator],
        scale_idx: int,
        return_aux: bool = False,
    ):
        if return_aux:
            r_vis, rel_aux = self.reliability(ct, pet, return_aux=True)
        else:
            r_vis = self.reliability(ct, pet, return_aux=False)
            rel_aux = None

        if text_calibrator is not None:
            if text_embedding is None:
                raise ValueError("text_embedding is required when text calibration is enabled")
            r = text_calibrator(r_vis, text_embedding, scale_idx)
        else:
            r = r_vis

        # IMPORTANT: reliability is used once, here.
        pet_reliable = r * pet

        # Path 1: reliable direct fusion.
        direct = ct + pet_reliable

        # Path 2: low-memory synergistic interaction.
        synergy = self.synergy(ct, pet_reliable)

        # No second reliability multiplication and no extra gate.
        out = _sanitize(direct + synergy)

        if not return_aux:
            return out

        aux = dict(rel_aux)
        aux.update(
            {
                "reliability": r,
                "reliable_pet": pet_reliable,
                "synergy": synergy,
            }
        )
        return out, aux


@dataclass
class TRDFConfig:
    channels: Tuple[int, ...] = (64, 128, 320, 512)
    attention_dims: Optional[Tuple[int, ...]] = None

    use_text_prior: bool = False
    text_backend: str = "precomputed"  # "precomputed" or "hf_local"
    text_embedding_path: Optional[str] = None
    text_model_path: Optional[str] = None
    text_hidden_dim: int = 128

    real_prompt: str = REAL_PET_PROMPT
    proxy_prompt: str = PROXY_PET_PROMPT

    logvar_clamp: Tuple[float, float] = (-8.0, 8.0)
    confidence_hidden: int = 8
    attention_eps: float = 1e-6


class TRDFFusion(nn.Module):
    """
    Four-scale standalone TRDF fusion module.

    Drop-in-style signature intentionally resembles the current
    StateAwareWeightedAddFusion:
        fusion(ct_feats, pet_feats, mode='full')
        fusion(ct_feats, pet_feats, mode='missing')
        fusion(ct_feats, pet_feats, mode='auto', pet_available=...)

    By default it returns only the fused feature list, so existing decoder
    calling code can remain unchanged.
    """

    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 320, 512),
        attention_dims: Optional[Sequence[int]] = None,
        use_text_prior: bool = False,
        text_backend: str = "precomputed",
        text_embedding_path: Optional[str] = None,
        text_model_path: Optional[str] = None,
        text_hidden_dim: int = 128,
        real_prompt: str = REAL_PET_PROMPT,
        proxy_prompt: str = PROXY_PET_PROMPT,
        logvar_clamp: Tuple[float, float] = (-8.0, 8.0),
        confidence_hidden: int = 8,
        attention_eps: float = 1e-6,
    ):
        super().__init__()

        self.channels = tuple(int(c) for c in channels)
        self.num_scales = len(self.channels)
        self.use_text_prior = bool(use_text_prior)

        if attention_dims is None:
            # Attention only is compressed. Reliability stays at original C.
            attention_dims = tuple(min(max(c // 4, 16), 32) for c in self.channels)
        else:
            attention_dims = tuple(int(d) for d in attention_dims)

        if len(attention_dims) != self.num_scales:
            raise ValueError(
                f"attention_dims must have {self.num_scales} values, got {attention_dims}"
            )
        self.attention_dims = attention_dims

        self.scales = nn.ModuleList(
            [
                TRDFScale(
                    channels=c,
                    attn_dim=d,
                    logvar_clamp=logvar_clamp,
                    confidence_hidden=confidence_hidden,
                    attention_eps=attention_eps,
                )
                for c, d in zip(self.channels, self.attention_dims)
            ]
        )

        if self.use_text_prior:
            self.text_prior = StaticPETTextPrior(
                backend=text_backend,
                embedding_path=text_embedding_path,
                text_model_path=text_model_path,
                real_prompt=real_prompt,
                proxy_prompt=proxy_prompt,
            )
            self.text_calibrator = TextReliabilityCalibrator(
                text_dim=self.text_prior.output_dim,
                num_scales=self.num_scales,
                hidden_dim=text_hidden_dim,
            )
        else:
            self.text_prior = None
            self.text_calibrator = None

    def _validate_feature_lists(
        self,
        ct_feats: Sequence[torch.Tensor],
        pet_feats: Sequence[torch.Tensor],
    ) -> None:
        if len(ct_feats) != self.num_scales or len(pet_feats) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} scales, "
                f"got CT={len(ct_feats)} PET={len(pet_feats)}"
            )
        for i, (ct, pet, c) in enumerate(zip(ct_feats, pet_feats, self.channels)):
            _check_pair(f"scale_{i+1}", ct, pet)
            if ct.shape[1] != c:
                raise ValueError(
                    f"scale_{i+1}: expected C={c}, got C={ct.shape[1]}"
                )

    def forward(
        self,
        ct_feats: Sequence[torch.Tensor],
        pet_feats: Sequence[torch.Tensor],
        mode: str,
        pet_available: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        self._validate_feature_lists(ct_feats, pet_feats)
        mode = str(mode).lower().strip()

        batch_size = int(ct_feats[0].shape[0])
        device = ct_feats[0].device
        dtype = ct_feats[0].dtype

        if self.use_text_prior:
            text_embedding = self.text_prior.get(
                batch_size=batch_size,
                mode=mode,
                device=device,
                dtype=dtype,
                pet_available=pet_available,
            )
        else:
            # Still validate mode for interface consistency.
            if mode not in ("full", "real", "missing", "proxy", "auto"):
                raise ValueError(f"Unsupported mode={mode!r}")
            if mode == "auto":
                if pet_available is None:
                    raise ValueError("pet_available is required when mode='auto'")
            text_embedding = None

        outputs: List[torch.Tensor] = []
        aux_scales: List[Dict[str, torch.Tensor]] = []

        for scale_idx, (block, ct, pet) in enumerate(
            zip(self.scales, ct_feats, pet_feats)
        ):
            if return_aux:
                out, aux = block(
                    ct=ct,
                    pet=pet,
                    text_embedding=text_embedding,
                    text_calibrator=self.text_calibrator,
                    scale_idx=scale_idx,
                    return_aux=True,
                )
                outputs.append(out)
                aux_scales.append(aux)
            else:
                out = block(
                    ct=ct,
                    pet=pet,
                    text_embedding=text_embedding,
                    text_calibrator=self.text_calibrator,
                    scale_idx=scale_idx,
                    return_aux=False,
                )
                outputs.append(out)

        if not return_aux:
            return outputs

        return outputs, {
            "scales": aux_scales,
            "mode": mode,
            "use_text_prior": self.use_text_prior,
        }

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, attention_dims={self.attention_dims}, "
            f"use_text_prior={self.use_text_prior}"
        )


def build_trdf_fusion_from_config(config: TRDFConfig) -> TRDFFusion:
    """Convenience builder for later repository integration."""
    return TRDFFusion(
        channels=config.channels,
        attention_dims=config.attention_dims,
        use_text_prior=config.use_text_prior,
        text_backend=config.text_backend,
        text_embedding_path=config.text_embedding_path,
        text_model_path=config.text_model_path,
        text_hidden_dim=config.text_hidden_dim,
        real_prompt=config.real_prompt,
        proxy_prompt=config.proxy_prompt,
        logvar_clamp=config.logvar_clamp,
        confidence_hidden=config.confidence_hidden,
        attention_eps=config.attention_eps,
    )


def _count_params(module: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def demo(
    batch_size: int = 1,
    use_text_prior: bool = False,
    text_backend: str = "precomputed",
    text_embedding_path: Optional[str] = None,
    text_model_path: Optional[str] = None,
    run_backward: bool = True,
) -> None:
    """
    Standalone smoke test using the actual expected MiT-B1-aligned four scales.

    Default text is OFF so:
        python trdf_fusion.py
    works with PyTorch only.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = (64, 128, 320, 512)
    spatial = ((128, 128), (64, 64), (32, 32), (16, 16))

    model = TRDFFusion(
        channels=channels,
        use_text_prior=use_text_prior,
        text_backend=text_backend,
        text_embedding_path=text_embedding_path,
        text_model_path=text_model_path,
    ).to(device)

    model.train()
    total, trainable = _count_params(model)

    print("=" * 80)
    print("TRDF standalone demo")
    print(f"device={device}")
    print(f"model={model}")
    print(f"params_total={total:,} | trainable={trainable:,}")
    print("=" * 80)

    ct_feats = [
        torch.randn(batch_size, c, h, w, device=device, requires_grad=True)
        for c, (h, w) in zip(channels, spatial)
    ]
    pet_feats = [
        torch.randn(batch_size, c, h, w, device=device, requires_grad=True)
        for c, (h, w) in zip(channels, spatial)
    ]

    fused, aux = model(
        ct_feats,
        pet_feats,
        mode="full",
        return_aux=True,
    )

    for i, (ct, pet, out, a) in enumerate(
        zip(ct_feats, pet_feats, fused, aux["scales"]), start=1
    ):
        r = a["reliability"]
        syn = a["synergy"]
        print(
            f"S{i}: CT={tuple(ct.shape)} PET={tuple(pet.shape)} "
            f"R={tuple(r.shape)} synergy={tuple(syn.shape)} OUT={tuple(out.shape)} "
            f"R_mean={float(r.detach().mean()):.4f}"
        )
        assert out.shape == ct.shape == pet.shape
        assert r.shape == (batch_size, 1, ct.shape[-2], ct.shape[-1])

    if run_backward:
        loss = sum(x.float().square().mean() for x in fused)
        loss.backward()
        print(f"backward_ok=True | dummy_loss={float(loss.detach()):.6f}")

    # Explicit zero-PET limit check in eval mode:
    # P=0 does not guarantee R=0, but if reliable PET is manually zeroed inside
    # the attention primitive, synergy must be exactly zero (up to numerical noise).
    model.eval()
    with torch.no_grad():
        block = model.scales[0].synergy
        zero_pet = torch.zeros_like(ct_feats[0])
        zero_synergy = block(ct_feats[0].detach(), zero_pet)
        max_abs = float(zero_synergy.abs().max())
        print(f"zero_reliable_pet_synergy_max_abs={max_abs:.8f}")

    print("TRDF demo completed successfully.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone TRDF fusion smoke test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-backward", action="store_true")

    parser.add_argument("--use-text-prior", action="store_true")
    parser.add_argument(
        "--text-backend",
        type=str,
        default="precomputed",
        choices=("precomputed", "hf_local"),
    )
    parser.add_argument("--text-embedding-path", type=str, default=None)
    parser.add_argument("--text-model-path", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    demo(
        batch_size=args.batch_size,
        use_text_prior=args.use_text_prior,
        text_backend=args.text_backend,
        text_embedding_path=args.text_embedding_path,
        text_model_path=args.text_model_path,
        run_backward=not args.no_backward,
    )