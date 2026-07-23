"""
APSF: Asymmetric PET-Surrogate Fusion
======================================

三步结构（与设计图一致）
------------------------
1. 路径专用可靠性 R_s（连续、全 token）：
   Full:    R = max(r_pet, r_task)
   Missing: R = r_ct * r_agree
   R 控制辅助信息的全部进入，而不是只缩放一条残差。

2. 把辅助信息迁移到 CT 表征空间：
   Q = W_q LN(C),  K = W_k LN(A),  V = W_v LN(A)
   M = Softmax( (√R Q)^T (√R K) )   # [B, d, d]
   Z = V M^T                        # [B, N, d]

3. 辅助证据生成 CT 状态修正（无 +A 直通）：
   [γ, β] = W_m(Z)
   U = tanh(γ) ⊙ Q + β
   F = C + W_o( R ⊙ U )

初始化：W_o 全零 ⇒ F ≡ C；无 +A 直通，PET/proxy 只经可靠性→跨协方差→仿射路径影响。
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

FeaturePyramid = Sequence[Tensor]
DebugInfo = Dict[str, Tensor]


def standardize_scores(scores: Tensor) -> Tensor:
    """按样本在空间 token 维度做无可调参数的标准化（FP32 均值/方差，无排序）。"""
    scores_fp32 = scores.float()
    mean = scores_fp32.mean(dim=-1, keepdim=True)
    variance = (scores_fp32 - mean).square().mean(dim=-1, keepdim=True)
    # Larger than finfo.eps: under AMP, near-zero spatial variance used to explode.
    return ((scores_fp32 - mean) * torch.rsqrt(variance + 1e-5)).to(scores.dtype)


def map_to_tokens(feature: Tensor) -> Tensor:
    """[B, C, H, W] -> [B, H*W, C]."""
    return feature.flatten(2).transpose(1, 2).contiguous()


def tokens_to_map(tokens: Tensor, height: int, width: int) -> Tensor:
    """[B, H*W, C] -> [B, C, H, W]."""
    batch, token_count, channels = tokens.shape
    if token_count != height * width:
        raise ValueError(
            f"Token count mismatch: got {token_count}, expected {height * width}."
        )
    return tokens.transpose(1, 2).reshape(batch, channels, height, width)


def zero_init_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class APSFStage(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        self.channels = int(channels)
        self.inner_dim = min(64, max(8, self.channels // 4))
        self.attention_scale = self.channels ** -0.5

        self.task_token = nn.Parameter(torch.empty(1, self.channels))
        self.ct_norm = nn.LayerNorm(self.channels)
        self.real_pet_norm = nn.LayerNorm(self.channels)
        self.proxy_norm = nn.LayerNorm(self.channels)
        self.query_norm = nn.LayerNorm(self.channels)

        # Step 2 projections
        self.q_proj = nn.Linear(self.channels, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(self.channels, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(self.channels, self.inner_dim, bias=False)
        # Step 3: low-dim affine from Z → [γ, β]
        self.mod_proj = nn.Linear(self.inner_dim, 2 * self.inner_dim, bias=True)
        # Step 3: map back to channel space (zero-init → CT identity at start)
        self.out_proj = nn.Linear(self.inner_dim, self.channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.task_token, std=0.02)
        for norm in (
            self.ct_norm,
            self.real_pet_norm,
            self.proxy_norm,
            self.query_norm,
        ):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)
        for projection in (self.q_proj, self.k_proj, self.v_proj, self.mod_proj):
            nn.init.xavier_uniform_(projection.weight)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)
        zero_init_linear(self.out_proj)

    @staticmethod
    def _validate_pair(ct: Tensor, auxiliary: Tensor) -> None:
        if ct.ndim != 4 or auxiliary.ndim != 4 or ct.shape != auxiliary.shape:
            raise ValueError(
                "CT and auxiliary features must be aligned 4D maps, "
                f"got {tuple(ct.shape)} and {tuple(auxiliary.shape)}."
            )

    def _ct_conditioned_query(self, ct_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        normalized_ct = self.ct_norm(ct_tokens)
        task = self.task_token.expand(normalized_ct.shape[0], -1)
        attn = (task[:, None, :] * normalized_ct).sum(dim=-1) * self.attention_scale
        attn = attn.softmax(dim=-1)
        ct_context = torch.einsum("bn,bnd->bd", attn, normalized_ct)
        return self.query_norm(task + ct_context), normalized_ct

    def _full_frontend(
        self,
        ct_tokens: Tensor,
        pet_tokens: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """R_full = max(r_pet, r_task); 同时返回 LN(C), LN(A)."""
        query, normalized_ct = self._ct_conditioned_query(ct_tokens)
        normalized_pet = self.real_pet_norm(pet_tokens)
        pet_global = normalized_pet.mean(dim=1)
        self_scores = (
            torch.einsum("bd,bnd->bn", pet_global, normalized_pet)
            * self.attention_scale
        )
        task_scores = (
            torch.einsum("bd,bnd->bn", query, normalized_pet) * self.attention_scale
        )
        self_relevance = torch.sigmoid(standardize_scores(self_scores))
        task_relevance = torch.sigmoid(standardize_scores(task_scores))
        relevance = torch.maximum(self_relevance, task_relevance)
        return normalized_ct, normalized_pet, relevance

    def _missing_frontend(
        self,
        ct_tokens: Tensor,
        proxy_tokens: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """R_miss = r_ct * r_agree; 同时返回 LN(C), LN(A)."""
        query, normalized_ct = self._ct_conditioned_query(ct_tokens)
        normalized_proxy = self.proxy_norm(proxy_tokens)
        ct_task_scores = (
            torch.einsum("bd,bnd->bn", query, normalized_ct) * self.attention_scale
        )
        agreement_scores = (
            normalized_ct * normalized_proxy
        ).sum(dim=-1) * self.attention_scale
        ct_relevance = torch.sigmoid(standardize_scores(ct_task_scores))
        agreement_relevance = torch.sigmoid(standardize_scores(agreement_scores))
        relevance = ct_relevance * agreement_relevance
        return normalized_ct, normalized_proxy, relevance

    def _reliability_xcov_ct_update(
        self,
        ct_tokens: Tensor,
        normalized_ct: Tensor,
        normalized_auxiliary: Tensor,
        relevance: Tensor,
    ) -> Tensor:
        """
        Step2–3:
          Q,K,V ∈ [B,N,d]
          M = Softmax((√R Q)^T (√R K)) ∈ [B,d,d]  (FP32; L2-norm over N for AMP stability)
          Z = V M^T
          [γ,β] = W_m(Z)
          F = C + W_o( R ⊙ (tanh(γ) ⊙ Q + β) )
        初始化时 W_o=0 ⇒ F ≡ C（无辅助直通）。
        """
        # Keep structure F=C+W_o(R⊙U). Compute the low-dim xcov path in FP32 so
        # that N≈H*W matmuls under AMP cannot produce Inf before softmax.
        q = self.q_proj(normalized_ct).float()
        k = self.k_proj(normalized_auxiliary).float()
        v = self.v_proj(normalized_auxiliary).float()
        relevance_f = relevance.float().clamp(0.0, 1.0)

        # √R 约束：控制辅助信息进入跨协方差
        root_r = relevance_f.clamp_min(1e-6).sqrt().unsqueeze(-1)  # [B,N,1]
        q_w = q * root_r
        k_w = k * root_r

        # Normalize over spatial tokens so M stays O(1) when N is large (e.g. 128^2).
        # Shapes remain [B,d,N] / [B,d,d] — never form [B,N,N].
        q_channel = F.normalize(q_w.transpose(1, 2), p=2, dim=-1, eps=1e-6)
        k_channel = F.normalize(k_w.transpose(1, 2), p=2, dim=-1, eps=1e-6)
        channel_relation = torch.matmul(q_channel, k_channel.transpose(-2, -1))
        channel_relation = channel_relation.softmax(dim=-1)

        # Z = V M^T → [B,N,d]
        z = torch.matmul(v, channel_relation.transpose(-2, -1))

        # [γ, β] = W_m(Z)
        gamma, beta = self.mod_proj(z).chunk(2, dim=-1)
        # Soft clamp affine params so residual cannot explode after out_proj grows.
        gamma = gamma.clamp(-20.0, 20.0)
        beta = beta.clamp(-50.0, 50.0)
        # U = tanh(γ) ⊙ Q + β
        u = torch.tanh(gamma) * q + beta

        residual = self.out_proj(relevance_f.unsqueeze(-1) * u)
        residual = residual.to(dtype=ct_tokens.dtype)
        residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
        # No auxiliary passthrough: F = C + W_o(R ⊙ U)
        return ct_tokens + residual

    def _forward_stage(
        self,
        ct: Tensor,
        aux: Tensor,
        mode: str,
    ) -> Tuple[Tensor, Tensor]:
        self._validate_pair(ct, aux)
        height, width = ct.shape[-2:]
        ct_tokens = map_to_tokens(ct)
        aux_tokens = map_to_tokens(aux)

        if mode == "full":
            norm_ct, norm_aux, relevance = self._full_frontend(ct_tokens, aux_tokens)
        elif mode == "missing":
            norm_ct, norm_aux, relevance = self._missing_frontend(ct_tokens, aux_tokens)
        else:
            raise ValueError(f"Unsupported mode={mode!r}")

        fused_tokens = self._reliability_xcov_ct_update(
            ct_tokens,
            norm_ct,
            norm_aux,
            relevance,
        )
        fused = tokens_to_map(fused_tokens, height, width)
        return fused, relevance

    def forward_full(
        self,
        ct: Tensor,
        pet: Tensor,
        return_debug: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, DebugInfo]]:
        if ct.shape[1] != self.channels:
            raise ValueError(
                f"Stage was built for {self.channels} channels, but received {ct.shape[1]}."
            )
        fused, relevance = self._forward_stage(ct, pet, "full")
        if not return_debug:
            return fused
        reliability_map = tokens_to_map(
            relevance.unsqueeze(-1),
            ct.shape[-2],
            ct.shape[-1],
        )
        debug = {
            "reliability_map": reliability_map,
            "selection_map": reliability_map,
            "support_count": relevance.float().sum(dim=-1),
        }
        return fused, debug

    def forward_missing(
        self,
        ct: Tensor,
        proxy: Tensor,
        return_debug: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, DebugInfo]]:
        if ct.shape[1] != self.channels:
            raise ValueError(
                f"Stage was built for {self.channels} channels, but received {ct.shape[1]}."
            )
        fused, relevance = self._forward_stage(ct, proxy, "missing")
        if not return_debug:
            return fused
        reliability_map = tokens_to_map(
            relevance.unsqueeze(-1),
            ct.shape[-2],
            ct.shape[-1],
        )
        debug = {
            "reliability_map": reliability_map,
            "selection_map": reliability_map,
            "support_count": relevance.float().sum(dim=-1),
        }
        return fused, debug

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, inner_dim={self.inner_dim}, "
            f"selection=continuous, fusion=xcov_ct_update"
        )


class APSF(nn.Module):
    def __init__(self, channels: Sequence[int] = (64, 128, 320, 512)) -> None:
        super().__init__()
        self.channels = tuple(int(v) for v in channels)
        self.stages = nn.ModuleList([APSFStage(v) for v in self.channels])

    def _run(self, fn, ct_features, aux_features, return_debug=False):
        if len(ct_features) != len(self.stages):
            raise ValueError(
                f"Expected {len(self.stages)} CT scales, got {len(ct_features)}."
            )
        if len(aux_features) != len(self.stages):
            raise ValueError(
                f"Expected {len(self.stages)} aux scales, got {len(aux_features)}."
            )

        outputs = []
        debugs = []
        for stage, ct, aux in zip(self.stages, ct_features, aux_features):
            result = fn(stage, ct, aux, return_debug)
            if return_debug:
                fused, debug = result
                outputs.append(fused)
                debugs.append(debug)
            else:
                outputs.append(result)
        if return_debug:
            return outputs, debugs
        return outputs

    def forward_full(self, ct_features, pet_features, return_debug=False):
        return self._run(
            lambda stage, ct, aux, rd: stage.forward_full(ct, aux, rd),
            ct_features,
            pet_features,
            return_debug,
        )

    def forward_missing(self, ct_features, proxy_features, return_debug=False):
        return self._run(
            lambda stage, ct, aux, rd: stage.forward_missing(ct, aux, rd),
            ct_features,
            proxy_features,
            return_debug,
        )

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(7)
    module = APSF(channels=(64, 128, 320, 512))
    print("APSF three-step CT-update self-test")
    print(f"  trainable parameters: {module.trainable_parameter_count():,}")
    print(f"  inner_dims: {[s.inner_dim for s in module.stages]}")
    ct = [
        torch.randn(2, c, h, w)
        for c, (h, w) in zip(
            (64, 128, 320, 512),
            ((32, 32), (16, 16), (8, 8), (4, 4)),
        )
    ]
    pet = [t.clone() for t in ct]
    full = module.forward_full(ct, pet)
    for o, a in zip(full, ct):
        torch.testing.assert_close(o, a, rtol=1e-6, atol=1e-6)
    print("  init F≡C (W_o=0, no +A): OK")
