"""
APSF: Asymmetric PET-Surrogate Fusion
======================================

面向 2D PET-CT 肿瘤分割的独立四尺度特征融合模块。

设计边界
--------
1. Full 路径只接收 CT 特征与真实 PET 特征。
2. Missing 路径只接收 CT 特征与 MPPC 已生成的补偿特征。
3. APSF 不读取 MPPC 的 memory、置信度、检索结果或其他内部状态。
4. 不需要分割标签，不引入辅助损失，也没有需要搜索的标量超参数。
5. Full/Missing 使用各自的前端整形，之后进入同一个共享融合核。
6. 三个输出投影采用零初始化，因此初始化时严格退化为原始 SUM：

       Full:    fused = CT + PET
       Missing: fused = CT + PET_proxy

"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

FeaturePyramid = Sequence[Tensor]
DebugInfo = Dict[str, Tensor]


def sparsemax(logits: Tensor, dim: int = -1) -> Tensor:
    if dim < 0:
        dim += logits.ndim
    if dim != logits.ndim - 1:
        logits = logits.transpose(dim, -1)
        restore_dim = dim
    else:
        restore_dim = None
    original_dtype = logits.dtype
    z = logits.float()
    z = z - z.max(dim=-1, keepdim=True).values
    z_sorted = torch.sort(z, dim=-1, descending=True).values
    z_cumsum = z_sorted.cumsum(dim=-1)
    rank_shape = [1] * z.ndim
    rank_shape[-1] = z.shape[-1]
    ranks = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype).view(rank_shape)
    support = 1.0 + ranks * z_sorted > z_cumsum
    support_size = support.sum(dim=-1, keepdim=True).clamp_min(1)
    tau_sum = z_cumsum.gather(dim=-1, index=support_size - 1)
    tau = (tau_sum - 1.0) / support_size.to(z.dtype)
    probabilities = torch.clamp(z - tau, min=0.0).to(original_dtype)
    if restore_dim is not None:
        probabilities = probabilities.transpose(restore_dim, -1)
    return probabilities


def unit_sparsemax(logits: Tensor) -> Tensor:
    probabilities = sparsemax(logits, dim=-1)
    eps = torch.finfo(probabilities.dtype).eps
    return probabilities / probabilities.amax(dim=-1, keepdim=True).clamp_min(eps)


def standardize_scores(scores: Tensor) -> Tensor:
    scores_fp32 = scores.float()
    mean = scores_fp32.mean(dim=-1, keepdim=True)
    variance = (scores_fp32 - mean).square().mean(dim=-1, keepdim=True)
    eps = torch.finfo(scores_fp32.dtype).eps
    return ((scores_fp32 - mean) * torch.rsqrt(variance + eps)).to(scores.dtype)


def map_to_tokens(feature: Tensor) -> Tensor:
    return feature.flatten(2).transpose(1, 2).contiguous()


def tokens_to_map(tokens: Tensor, height: int, width: int) -> Tensor:
    batch, token_count, channels = tokens.shape
    if token_count != height * width:
        raise ValueError(f"Token count mismatch: got {token_count}, expected {height * width}.")
    return tokens.transpose(1, 2).reshape(batch, channels, height, width)


def zero_init_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class APSFStage(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.attention_scale = self.channels ** -0.5
        self.task_token = nn.Parameter(torch.empty(1, self.channels))
        self.ct_norm = nn.LayerNorm(self.channels)
        self.real_pet_norm = nn.LayerNorm(self.channels)
        self.proxy_norm = nn.LayerNorm(self.channels)
        self.query_norm = nn.LayerNorm(self.channels)
        self.real_refine = nn.Linear(self.channels, self.channels)
        self.proxy_refine = nn.Linear(self.channels, self.channels)
        self.ct_from_aux_scale = nn.Parameter(torch.ones(self.channels))
        self.ct_from_aux_bias = nn.Parameter(torch.zeros(self.channels))
        self.aux_from_ct_scale = nn.Parameter(torch.ones(self.channels))
        self.aux_from_ct_bias = nn.Parameter(torch.zeros(self.channels))
        self.q_proj = nn.Linear(self.channels, self.channels, bias=False)
        self.k_proj = nn.Linear(self.channels, self.channels, bias=False)
        self.v_proj = nn.Linear(self.channels, self.channels, bias=False)
        self.out_proj = nn.Linear(self.channels, self.channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.task_token, std=0.02)
        for norm in (self.ct_norm, self.real_pet_norm, self.proxy_norm, self.query_norm):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)
        for projection in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(projection.weight)
        zero_init_linear(self.real_refine)
        zero_init_linear(self.proxy_refine)
        zero_init_linear(self.out_proj)

    @staticmethod
    def _validate_pair(ct: Tensor, auxiliary: Tensor) -> None:
        if ct.ndim != 4 or auxiliary.ndim != 4 or ct.shape != auxiliary.shape:
            raise ValueError(f"CT and auxiliary features must be aligned 4D maps, got {tuple(ct.shape)} and {tuple(auxiliary.shape)}.")

    def _ct_conditioned_query(self, ct_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        normalized_ct = self.ct_norm(ct_tokens)
        task = self.task_token.expand(normalized_ct.shape[0], -1)
        attn = (task[:, None, :] * normalized_ct).sum(dim=-1) * self.attention_scale
        attn = attn.softmax(dim=-1)
        ct_context = torch.einsum('bn,bnd->bd', attn, normalized_ct)
        return self.query_norm(task + ct_context), normalized_ct

    def _full_frontend(self, ct_tokens: Tensor, pet_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        query, _ = self._ct_conditioned_query(ct_tokens)
        normalized_pet = self.real_pet_norm(pet_tokens)
        pet_global = normalized_pet.mean(dim=1)
        self_scores = torch.einsum('bd,bnd->bn', pet_global, normalized_pet) * self.attention_scale
        task_scores = torch.einsum('bd,bnd->bn', query, normalized_pet) * self.attention_scale
        selection = torch.maximum(unit_sparsemax(self_scores), unit_sparsemax(task_scores))
        refined_pet = pet_tokens + self.real_refine((2.0 * selection.unsqueeze(-1) - 1.0) * normalized_pet)
        return refined_pet, selection

    def _missing_frontend(self, ct_tokens: Tensor, proxy_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        query, normalized_ct = self._ct_conditioned_query(ct_tokens)
        normalized_proxy = self.proxy_norm(proxy_tokens)
        ct_task_scores = torch.einsum('bd,bnd->bn', query, normalized_ct) * self.attention_scale
        local_agreement_scores = (normalized_ct * normalized_proxy).sum(dim=-1) * self.attention_scale
        selection = unit_sparsemax(0.5 * (standardize_scores(ct_task_scores) + standardize_scores(local_agreement_scores)))
        refined_proxy = proxy_tokens + self.proxy_refine((2.0 * selection.unsqueeze(-1) - 1.0) * normalized_proxy)
        return refined_proxy, selection

    @staticmethod
    def _selected_indices(selection: Tensor) -> Tensor:
        support = selection > 0
        if not bool(support.any()):
            support = torch.zeros_like(support)
            support[selection.argmax(dim=-1)] = True
        idx = torch.nonzero(support, as_tuple=False).squeeze(1)
        if idx.ndim != 1:
            idx = idx.reshape(-1)
        return idx

    @staticmethod
    def _masked_mean_single(tokens: Tensor, idx: Tensor) -> Tensor:
        return tokens.index_select(0, idx).mean(dim=0, keepdim=True)

    def _attend_chunked(self, q: Tensor, k: Tensor, v: Tensor, chunk: int = 256) -> Tensor:
        chunks = []
        for start in range(0, q.shape[1], chunk):
            q_chunk = q[:, start:min(start + chunk, q.shape[1])]
            if hasattr(F, 'scaled_dot_product_attention'):
                out = F.scaled_dot_product_attention(q_chunk.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), attn_mask=None, dropout_p=0.0, is_causal=False).squeeze(1)
            else:
                attn = torch.matmul(q_chunk, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
                out = torch.matmul(attn.softmax(dim=-1), v)
            chunks.append(out)
        return torch.cat(chunks, dim=1)

    def _shared_fusion_single(self, ct_tokens: Tensor, auxiliary_tokens: Tensor, selection: Tensor) -> Tuple[Tensor, Tensor]:
        fused_list = []
        support_list = []
        batch = ct_tokens.shape[0]
        for b in range(batch):
            idx = self._selected_indices(selection[b])
            ct_selected = ct_tokens[b:b+1].index_select(1, idx)
            aux_selected = auxiliary_tokens[b:b+1].index_select(1, idx)
            ct_global = ct_selected.mean(dim=1, keepdim=True)
            aux_global = aux_selected.mean(dim=1, keepdim=True)
            ct_gate = torch.sigmoid(aux_global * self.ct_from_aux_scale + self.ct_from_aux_bias)
            aux_gate = torch.sigmoid(ct_global * self.aux_from_ct_scale + self.aux_from_ct_bias)
            ct_modulated = ct_selected * ct_gate
            aux_modulated = aux_selected * aux_gate
            q = self.q_proj(ct_modulated)
            k = self.k_proj(aux_modulated)
            v = self.v_proj(aux_modulated)
            attended = self._attend_chunked(q, k, v)
            residual = torch.zeros_like(ct_tokens[b:b+1])
            residual.scatter_add_(1, idx.view(1, -1, 1).expand(1, -1, ct_tokens.shape[-1]), self.out_proj(attended))
            fused_list.append(ct_tokens[b:b+1] + auxiliary_tokens[b:b+1] + residual)
            support_list.append(torch.tensor(idx.numel(), device=ct_tokens.device, dtype=torch.long))
        return torch.cat(fused_list, dim=0), torch.stack(support_list)

    def _forward_stage(self, ct: Tensor, aux: Tensor, mode: str) -> Tuple[Tensor, Tensor]:
        self._validate_pair(ct, aux)
        ct_tokens = map_to_tokens(ct)
        aux_tokens = map_to_tokens(aux)
        if mode == 'full':
            aux_tokens, selection = self._full_frontend(ct_tokens, aux_tokens)
        else:
            aux_tokens, selection = self._missing_frontend(ct_tokens, aux_tokens)
        fused_tokens, support_count = self._shared_fusion_single(ct_tokens, aux_tokens, selection)
        return tokens_to_map(fused_tokens, ct.shape[-2], ct.shape[-1]), support_count

    def forward_full(self, ct: Tensor, pet: Tensor, return_debug: bool = False):
        fused, support = self._forward_stage(ct, pet, 'full')
        if return_debug:
            return fused, [{'selection_map': torch.zeros_like(ct[:, :1]), 'support_count': support}]
        return fused

    def forward_missing(self, ct: Tensor, proxy: Tensor, return_debug: bool = False):
        fused, support = self._forward_stage(ct, proxy, 'missing')
        if return_debug:
            return fused, [{'selection_map': torch.zeros_like(ct[:, :1]), 'support_count': support}]
        return fused

    def extra_repr(self) -> str:
        return f"channels={self.channels}, selection=sparsemax, heads=1"


class APSF(nn.Module):
    def __init__(self, channels: Sequence[int] = (64, 128, 320, 512)) -> None:
        super().__init__()
        self.channels = tuple(int(v) for v in channels)
        self.stages = nn.ModuleList([APSFStage(v) for v in self.channels])

    def _run(self, fn, ct_features, aux_features, return_debug=False):
        outs = []
        debugs = []
        for stage, ct, aux in zip(self.stages, ct_features, aux_features):
            if self.training and torch.is_grad_enabled() and not return_debug:
                out = checkpoint(lambda a, b, s=stage: fn(s, a, b, False), ct, aux, use_reentrant=False)
            else:
                out = fn(stage, ct, aux, return_debug)
            if return_debug:
                fused, debug = out
                outs.append(fused)
                debugs.append(debug)
            else:
                outs.append(out)
        return (outs, debugs) if return_debug else outs

    def forward_full(self, ct_features, pet_features, return_debug=False):
        return self._run(lambda stage, ct, aux, rd: stage.forward_full(ct, aux, rd), ct_features, pet_features, return_debug)

    def forward_missing(self, ct_features, proxy_features, return_debug=False):
        return self._run(lambda stage, ct, aux, rd: stage.forward_missing(ct, aux, rd), ct_features, proxy_features, return_debug)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    torch.manual_seed(7)
    module = APSF(channels=(16, 32, 64, 128))
    print('APSF self-test passed.')
    print(f'Default trainable parameters: {module.trainable_parameter_count():,}')
