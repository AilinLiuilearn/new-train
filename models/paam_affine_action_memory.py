#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAAM: PET-induced Affine Action Memory
======================================

独立 PyTorch 模块，用于 PET 缺失鲁棒的多尺度 PET-CT 分割。

核心逻辑
--------
1. 每个尺度独立建立仿射作用原型库。
2. Full 路径：由当前 CT/PET 生成真实仿射作用 (Δγ*, Δβ*)。
3. Missing 路径：仅由 CT 查询上一轮冻结记忆，得到 (Δγ_hat, Δβ_hat)。
4. Full 与 Missing 共用同一仿射执行器：
       T = C + Δγ ⊙ Norm(C) + Δβ
       F = (1-W) ⊙ C + W ⊙ T
5. 训练时，Full 与模拟 Missing 样本都可利用配对 PET 生成真实仿射作用，
   但真实作用只用于写入下一轮记忆；绝不会进入当前 Missing 融合。
6. 当前版本不包含辅助损失。唯一需要人工调整的模型超参数是 K。

记忆内容
--------
每个尺度第 j 个槽只保存：
    K_ct[j]  : CT 检索地址
    Gamma[j] : PET 对 CT 的通道缩放作用原型
    Beta[j]  : PET 对 CT 的通道偏移作用原型

原型按仿射作用 [Gamma, Beta] 聚类，而不是按 CT 外观聚类；
CT 只负责为每类仿射作用建立可检索地址。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().float()
    if x.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "abs_mean": 0.0}
    flat = x.reshape(-1)
    return {
        "mean": _to_float(flat.mean()),
        "std": _to_float(flat.std(unbiased=False)),
        "min": _to_float(flat.min()),
        "max": _to_float(flat.max()),
        "abs_mean": _to_float(flat.abs().mean()),
    }


def _safe_entropy(prob: torch.Tensor, dim: int = -1) -> torch.Tensor:
    p = prob.clamp_min(1e-8)
    return -(p * p.log()).sum(dim=dim)


def _pairwise_cosine_mean(x: torch.Tensor) -> float:
    if x.ndim != 2 or x.shape[0] < 2:
        return 0.0
    x = F.normalize(x.float(), dim=1)
    sim = x @ x.t()
    mask = ~torch.eye(x.shape[0], dtype=torch.bool, device=x.device)
    return _to_float(sim[mask].mean())


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return _to_float(obj)
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


class RunningMean:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def mean(self) -> float:
        return self.total / max(self.count, 1)

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class AffineActionWriter(nn.Module):
    """由真实 CT/PET 关系生成真实仿射作用。"""

    def __init__(self, channels: int, relation_dim: int = 32) -> None:
        super().__init__()
        self.ct_norm = LayerNorm2d(channels)
        self.pet_norm = LayerNorm2d(channels)
        self.ct_proj = nn.Conv2d(channels, relation_dim, 1, bias=False)
        self.pet_proj = nn.Conv2d(channels, relation_dim, 1, bias=False)
        self.relation = nn.Sequential(
            nn.Conv2d(relation_dim, relation_dim, 1, bias=True),
            nn.GELU(),
        )
        self.gamma_head = nn.Conv2d(relation_dim, channels, 1, bias=True)
        self.beta_head = nn.Conv2d(relation_dim, channels, 1, bias=True)

        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, ct: torch.Tensor, pet: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ct_r = self.ct_proj(self.ct_norm(ct))
        pet_r = self.pet_proj(self.pet_norm(pet))
        relation = self.relation(pet_r - ct_r)
        delta_gamma = torch.tanh(self.gamma_head(relation))
        delta_beta = torch.tanh(self.beta_head(relation))
        return delta_gamma, delta_beta


class SharedAffineExecutor(nn.Module):
    """Full/Missing 共用同一仿射执行器。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.ct_norm = LayerNorm2d(channels)
        self.gate = nn.Conv2d(channels * 2, 1, 1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        ct: torch.Tensor,
        delta_gamma: torch.Tensor,
        delta_beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ct_n = self.ct_norm(ct)
        correction = delta_gamma * ct_n + delta_beta
        corrected_ct = ct + correction
        gate = torch.sigmoid(self.gate(torch.cat([ct_n, correction], dim=1)))
        fused = (1.0 - gate) * ct + gate * corrected_ct

        ct_l2 = ct.float().flatten(1).norm(dim=1).clamp_min(1e-6)
        corr_l2 = correction.float().flatten(1).norm(dim=1)
        return fused, {
            "gate": gate,
            "correction": correction,
            "corrected_ct": corrected_ct,
            "correction_ratio": corr_l2 / ct_l2,
        }


@dataclass
class RetrievalOutput:
    gamma: torch.Tensor
    beta: torch.Tensor
    info: Dict[str, torch.Tensor]


class AffineActionMemoryScale(nn.Module):
    """单尺度仿射作用记忆。K 是唯一对外调节的模型超参数。"""

    QUERY_DIM = 32
    CACHE_CAPACITY = 12000
    KMEANS_ITERS = 20
    MAX_CANDIDATES_PER_IMAGE = 64

    def __init__(self, channels: int, K: int, scale_index: int) -> None:
        super().__init__()
        if K < 1:
            raise ValueError(f"K 必须 >= 1，当前为 {K}")

        self.channels = int(channels)
        self.K = int(K)
        self.scale_index = int(scale_index)

        self.query_norm = LayerNorm2d(channels)
        self.query_proj = nn.Conv2d(channels, self.QUERY_DIM, 1, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))

        self.register_buffer("keys", torch.zeros(K, self.QUERY_DIM))
        self.register_buffer("gamma_proto", torch.zeros(K, channels))
        self.register_buffer("beta_proto", torch.zeros(K, channels))
        self.register_buffer("slot_counts", torch.zeros(K))
        self.register_buffer("memory_ready", torch.tensor(False, dtype=torch.bool))

        self._cache_q: List[torch.Tensor] = []
        self._cache_gamma: List[torch.Tensor] = []
        self._cache_beta: List[torch.Tensor] = []
        self._cache_strength: List[torch.Tensor] = []

        self._retrieval_slot_hits = torch.zeros(K, dtype=torch.long)
        self._stats: Dict[str, RunningMean] = {
            "retrieval_entropy": RunningMean(),
            "retrieval_max_similarity": RunningMean(),
            "retrieval_top1_weight": RunningMean(),
            "retrieved_gamma_abs_mean": RunningMean(),
            "retrieved_beta_abs_mean": RunningMean(),
            "candidate_strength": RunningMean(),
            "candidate_count": RunningMean(),
        }
        self._last_maps: Dict[str, torch.Tensor] = {}

    def make_query(self, ct: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(self.query_norm(ct))
        return F.normalize(q, dim=1, eps=1e-6)

    @torch.no_grad()
    def collect(
        self,
        query_map: torch.Tensor,
        gamma_star: torch.Tensor,
        beta_star: torch.Tensor,
        gate_star: torch.Tensor,
    ) -> Dict[str, Any]:
        b, _, h, w = query_map.shape
        strength_map = gate_star.detach().squeeze(1) * (
            gamma_star.detach().abs().mean(dim=1)
            + beta_star.detach().abs().mean(dim=1)
        )

        per_image = min(self.MAX_CANDIDATES_PER_IMAGE, max(1, (h * w) // 16))
        qs: List[torch.Tensor] = []
        gs: List[torch.Tensor] = []
        bs: List[torch.Tensor] = []
        ss: List[torch.Tensor] = []

        for bi in range(b):
            flat_strength = strength_map[bi].reshape(-1)
            topk = min(per_image, flat_strength.numel())
            _, idx = torch.topk(flat_strength, k=topk, largest=True, sorted=False)

            q_flat = query_map[bi].permute(1, 2, 0).reshape(-1, self.QUERY_DIM)
            g_flat = gamma_star[bi].permute(1, 2, 0).reshape(-1, self.channels)
            b_flat = beta_star[bi].permute(1, 2, 0).reshape(-1, self.channels)

            qs.append(q_flat[idx].detach().float().cpu())
            gs.append(g_flat[idx].detach().float().cpu())
            bs.append(b_flat[idx].detach().float().cpu())
            ss.append(flat_strength[idx].detach().float().cpu())

        q = torch.cat(qs, dim=0)
        gamma = torch.cat(gs, dim=0)
        beta = torch.cat(bs, dim=0)
        strength = torch.cat(ss, dim=0)

        self._cache_q.append(q)
        self._cache_gamma.append(gamma)
        self._cache_beta.append(beta)
        self._cache_strength.append(strength)
        self._trim_cache_if_needed()

        self._stats["candidate_strength"].update(_to_float(strength.mean()), n=int(strength.numel()))
        self._stats["candidate_count"].update(float(strength.numel()), n=1)

        return {
            "selected_count": int(strength.numel()),
            "strength_mean": _to_float(strength.mean()),
            "strength_max": _to_float(strength.max()),
        }

    @torch.no_grad()
    def _trim_cache_if_needed(self) -> None:
        total = sum(x.shape[0] for x in self._cache_strength)
        if total <= self.CACHE_CAPACITY:
            return

        q = torch.cat(self._cache_q, dim=0)
        gamma = torch.cat(self._cache_gamma, dim=0)
        beta = torch.cat(self._cache_beta, dim=0)
        strength = torch.cat(self._cache_strength, dim=0)
        keep = min(self.CACHE_CAPACITY, strength.numel())
        _, idx = torch.topk(strength, k=keep, largest=True, sorted=False)
        self._cache_q = [q[idx]]
        self._cache_gamma = [gamma[idx]]
        self._cache_beta = [beta[idx]]
        self._cache_strength = [strength[idx]]

    @torch.no_grad()
    def finalize_memory(self) -> Dict[str, Any]:
        if not self._cache_q:
            return {
                "scale": self.scale_index,
                "updated": False,
                "reason": "empty_cache",
                "memory_ready": bool(self.memory_ready.item()),
            }

        q = torch.cat(self._cache_q, dim=0).float()
        gamma = torch.cat(self._cache_gamma, dim=0).float()
        beta = torch.cat(self._cache_beta, dim=0).float()
        strength = torch.cat(self._cache_strength, dim=0).float().clamp_min(1e-8)

        n = q.shape[0]
        if n < self.K:
            repeat = math.ceil(self.K / max(n, 1))
            q = q.repeat(repeat, 1)[: self.K]
            gamma = gamma.repeat(repeat, 1)[: self.K]
            beta = beta.repeat(repeat, 1)[: self.K]
            strength = strength.repeat(repeat)[: self.K]
            n = q.shape[0]

        # 按仿射作用聚类，而不是按 CT 查询聚类。
        action = torch.cat([gamma, beta], dim=1)
        action_n = F.normalize(action, dim=1, eps=1e-6)
        centers = self._deterministic_farthest_init(action_n, strength, self.K)

        assignment = torch.zeros(n, dtype=torch.long)
        for _ in range(self.KMEANS_ITERS):
            sim = action_n @ centers.t()
            assignment = sim.argmax(dim=1)
            new_centers: List[torch.Tensor] = []
            for j in range(self.K):
                mask = assignment == j
                if mask.any():
                    weighted = action_n[mask] * strength[mask, None]
                    center = weighted.sum(dim=0) / strength[mask].sum().clamp_min(1e-8)
                    center = F.normalize(center, dim=0, eps=1e-6)
                else:
                    center = action_n[sim.max(dim=1).values.argmin()]
                new_centers.append(center)
            new_centers_t = torch.stack(new_centers, dim=0)
            if torch.allclose(centers, new_centers_t, atol=1e-5, rtol=1e-4):
                centers = new_centers_t
                break
            centers = new_centers_t

        assignment = (action_n @ centers.t()).argmax(dim=1)
        keys: List[torch.Tensor] = []
        gammas: List[torch.Tensor] = []
        betas: List[torch.Tensor] = []
        counts: List[float] = []

        for j in range(self.K):
            mask = assignment == j
            if not mask.any():
                fallback_idx = int(strength.argmax().item())
                mask = torch.zeros_like(assignment, dtype=torch.bool)
                mask[fallback_idx] = True

            wj = strength[mask]
            denom = wj.sum().clamp_min(1e-8)
            key = F.normalize((q[mask] * wj[:, None]).sum(dim=0) / denom, dim=0, eps=1e-6)
            gamma_j = (gamma[mask] * wj[:, None]).sum(dim=0) / denom
            beta_j = (beta[mask] * wj[:, None]).sum(dim=0) / denom

            keys.append(key)
            gammas.append(gamma_j)
            betas.append(beta_j)
            counts.append(float(mask.sum().item()))

        device = self.keys.device
        self.keys.copy_(torch.stack(keys).to(device=device, dtype=self.keys.dtype))
        self.gamma_proto.copy_(torch.stack(gammas).to(device=device, dtype=self.gamma_proto.dtype))
        self.beta_proto.copy_(torch.stack(betas).to(device=device, dtype=self.beta_proto.dtype))
        self.slot_counts.copy_(torch.tensor(counts, device=device, dtype=self.slot_counts.dtype))
        self.memory_ready.fill_(True)

        report = {
            "scale": self.scale_index,
            "updated": True,
            "candidate_count": int(n),
            "slot_counts": counts,
            "slot_count_min": float(min(counts)),
            "slot_count_max": float(max(counts)),
            "key_pairwise_cosine_mean": _pairwise_cosine_mean(self.keys.detach().cpu()),
            "gamma_proto_abs_mean": _to_float(self.gamma_proto.abs().mean()),
            "beta_proto_abs_mean": _to_float(self.beta_proto.abs().mean()),
            "memory_ready": True,
        }
        self.clear_cache()
        return report

    @staticmethod
    @torch.no_grad()
    def _deterministic_farthest_init(
        action_n: torch.Tensor,
        strength: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        first = int(strength.argmax().item())
        chosen = [first]
        centers = [action_n[first]]
        while len(centers) < K:
            current = torch.stack(centers, dim=0)
            max_sim = (action_n @ current.t()).max(dim=1).values
            max_sim[torch.tensor(chosen, dtype=torch.long)] = 1.0
            next_idx = int(max_sim.argmin().item())
            chosen.append(next_idx)
            centers.append(action_n[next_idx])
        return torch.stack(centers, dim=0)

    def clear_cache(self) -> None:
        self._cache_q.clear()
        self._cache_gamma.clear()
        self._cache_beta.clear()
        self._cache_strength.clear()

    def retrieve(self, query_map: torch.Tensor, capture_visuals: bool = False) -> RetrievalOutput:
        b, _, h, w = query_map.shape
        if not bool(self.memory_ready.item()):
            gamma = query_map.new_zeros((b, self.channels, h, w))
            beta = query_map.new_zeros((b, self.channels, h, w))
            info = {
                "memory_ready": torch.tensor(False, device=query_map.device),
                "entropy": query_map.new_zeros((b, h, w)),
                "max_similarity": query_map.new_zeros((b, h, w)),
                "top1_weight": query_map.new_zeros((b, h, w)),
                "top1_slot": torch.zeros((b, h, w), device=query_map.device, dtype=torch.long),
            }
            if capture_visuals:
                self._store_last_maps(info, gamma, beta)
            return RetrievalOutput(gamma=gamma, beta=beta, info=info)

        q = query_map.permute(0, 2, 3, 1).reshape(-1, self.QUERY_DIM)
        q = F.normalize(q, dim=1, eps=1e-6)
        keys = F.normalize(self.keys, dim=1, eps=1e-6)
        scale = self.logit_scale.exp().clamp(1.0, 100.0)

        similarity = q @ keys.t()
        weights = torch.softmax(similarity * scale, dim=1)
        gamma_flat = weights @ self.gamma_proto
        beta_flat = weights @ self.beta_proto

        gamma = gamma_flat.reshape(b, h, w, self.channels).permute(0, 3, 1, 2)
        beta = beta_flat.reshape(b, h, w, self.channels).permute(0, 3, 1, 2)

        entropy = _safe_entropy(weights, dim=1).reshape(b, h, w)
        max_similarity = similarity.max(dim=1).values.reshape(b, h, w)
        top1_weight, top1_slot = weights.max(dim=1)
        top1_weight = top1_weight.reshape(b, h, w)
        top1_slot = top1_slot.reshape(b, h, w)

        with torch.no_grad():
            hits = torch.bincount(top1_slot.detach().cpu().reshape(-1), minlength=self.K)
            self._retrieval_slot_hits += hits
            self._stats["retrieval_entropy"].update(_to_float(entropy.mean()), n=int(entropy.numel()))
            self._stats["retrieval_max_similarity"].update(_to_float(max_similarity.mean()), n=int(max_similarity.numel()))
            self._stats["retrieval_top1_weight"].update(_to_float(top1_weight.mean()), n=int(top1_weight.numel()))
            self._stats["retrieved_gamma_abs_mean"].update(_to_float(gamma.abs().mean()), n=int(gamma.numel()))
            self._stats["retrieved_beta_abs_mean"].update(_to_float(beta.abs().mean()), n=int(beta.numel()))

        info = {
            "memory_ready": torch.tensor(True, device=query_map.device),
            "entropy": entropy,
            "max_similarity": max_similarity,
            "top1_weight": top1_weight,
            "top1_slot": top1_slot,
            "logit_scale": scale.detach(),
        }
        if capture_visuals:
            self._store_last_maps(info, gamma, beta)
        return RetrievalOutput(gamma=gamma, beta=beta, info=info)

    @torch.no_grad()
    def _store_last_maps(
        self,
        info: Dict[str, torch.Tensor],
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> None:
        self._last_maps["top1_slot"] = info["top1_slot"][0].detach().cpu()
        self._last_maps["entropy"] = info["entropy"][0].detach().cpu()
        self._last_maps["max_similarity"] = info["max_similarity"][0].detach().cpu()
        self._last_maps["top1_weight"] = info["top1_weight"][0].detach().cpu()
        self._last_maps["retrieved_gamma_norm"] = gamma[0].float().norm(dim=0).detach().cpu()
        self._last_maps["retrieved_beta_norm"] = beta[0].float().norm(dim=0).detach().cpu()

    def reset_epoch_stats(self) -> None:
        self._retrieval_slot_hits.zero_()
        for meter in self._stats.values():
            meter.reset()
        self._last_maps.clear()

    def diagnostics(self) -> Dict[str, Any]:
        slot_hits = self._retrieval_slot_hits.tolist()
        total_hits = max(int(self._retrieval_slot_hits.sum().item()), 1)
        utilization = [float(v) / total_hits for v in slot_hits]
        cache_count = sum(x.shape[0] for x in self._cache_strength)
        return {
            "scale": self.scale_index,
            "channels": self.channels,
            "K": self.K,
            "memory_ready": bool(self.memory_ready.item()),
            "cache_count": int(cache_count),
            "slot_counts_build": self.slot_counts.detach().cpu().tolist(),
            "slot_hits_retrieval": slot_hits,
            "slot_utilization_retrieval": utilization,
            "active_slots_retrieval": int(sum(v > 0 for v in slot_hits)),
            "key_pairwise_cosine_mean": _pairwise_cosine_mean(self.keys.detach().cpu())
            if bool(self.memory_ready.item()) else 0.0,
            "gamma_prototype": _tensor_stats(self.gamma_proto),
            "beta_prototype": _tensor_stats(self.beta_proto),
            "logit_scale": _to_float(self.logit_scale.exp().clamp(1.0, 100.0)),
            "running": {name: meter.mean for name, meter in self._stats.items()},
        }


class PETAffineActionMemory(nn.Module):
    """四尺度 PAAM。唯一人工调整的模型超参数为 K。"""

    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 320, 512),
        K: int = 8,
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("当前设计要求四尺度输入。")
        self.channels = tuple(int(c) for c in channels)
        self.K = int(K)

        self.writers = nn.ModuleList([
            AffineActionWriter(c, relation_dim=AffineActionMemoryScale.QUERY_DIM)
            for c in self.channels
        ])
        self.executors = nn.ModuleList([SharedAffineExecutor(c) for c in self.channels])
        self.memories = nn.ModuleList([
            AffineActionMemoryScale(c, K=self.K, scale_index=i + 1)
            for i, c in enumerate(self.channels)
        ])

        self.current_epoch = 0
        self.route_counts = {"full": 0, "missing": 0}
        self.paired_pet_memory_write_count = 0
        self._last_forward: Dict[str, Any] = {}
        self._last_visual_maps: Dict[str, Dict[str, torch.Tensor]] = {}

    def begin_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        self.route_counts = {"full": 0, "missing": 0}
        self.paired_pet_memory_write_count = 0
        self._last_forward = {}
        self._last_visual_maps = {}
        for memory in self.memories:
            memory.reset_epoch_stats()

    def forward(
        self,
        ct_features: Sequence[torch.Tensor],
        pet_features: Optional[Sequence[torch.Tensor]] = None,
        route: str = "full",
        update_memory: bool = False,
        capture_visuals: bool = False,
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        if route not in {"full", "missing"}:
            raise ValueError(f"route 必须为 full 或 missing，当前为 {route}")
        if len(ct_features) != 4:
            raise ValueError("ct_features 必须包含四尺度。")
        if pet_features is not None and len(pet_features) != 4:
            raise ValueError("pet_features 必须包含四尺度。")
        if route == "full" and pet_features is None:
            raise ValueError("Full 路径必须提供 pet_features。")

        self.route_counts[route] += 1
        fused_features: List[torch.Tensor] = []
        per_scale_info: List[Dict[str, Any]] = []
        visual_maps: Dict[str, Dict[str, torch.Tensor]] = {}

        for idx, (ct, writer, executor, memory) in enumerate(
            zip(ct_features, self.writers, self.executors, self.memories)
        ):
            scale_name = f"s{idx + 1}"
            query = memory.make_query(ct)
            gamma_star: Optional[torch.Tensor] = None
            beta_star: Optional[torch.Tensor] = None
            true_exec: Optional[Dict[str, torch.Tensor]] = None
            write_report: Optional[Dict[str, Any]] = None

            # 即使 route=missing，也可计算真实 PET 仿射作用用于未来记忆，
            # 但不会进入当前 Missing 融合。
            if pet_features is not None:
                if route == "missing":
                    with torch.no_grad():
                        gamma_star, beta_star = writer(ct.detach(), pet_features[idx].detach())
                        _, true_exec = executor(ct.detach(), gamma_star, beta_star)
                        if update_memory:
                            write_report = memory.collect(
                                query_map=query.detach(),
                                gamma_star=gamma_star,
                                beta_star=beta_star,
                                gate_star=true_exec["gate"],
                            )
                else:
                    gamma_star, beta_star = writer(ct, pet_features[idx])
                    _, true_exec = executor(ct, gamma_star, beta_star)
                    if update_memory:
                        write_report = memory.collect(
                            query_map=query,
                            gamma_star=gamma_star,
                            beta_star=beta_star,
                            gate_star=true_exec["gate"],
                        )

            if route == "full":
                assert gamma_star is not None and beta_star is not None
                used_gamma = gamma_star
                used_beta = beta_star
                retrieval_info: Optional[Dict[str, torch.Tensor]] = None
                used_source = "current_real_pet_affine"
            else:
                retrieval = memory.retrieve(query, capture_visuals=capture_visuals)
                used_gamma = retrieval.gamma
                used_beta = retrieval.beta
                retrieval_info = retrieval.info
                used_source = "delayed_memory_retrieval"

            fused, exec_info = executor(ct, used_gamma, used_beta)
            fused_features.append(fused)

            with torch.no_grad():
                summary: Dict[str, Any] = {
                    "scale": idx + 1,
                    "route": route,
                    "used_affine_source": used_source,
                    "memory_ready": bool(memory.memory_ready.item()),
                    "ct_shape": list(ct.shape),
                    "used_gamma_abs_mean": _to_float(used_gamma.abs().mean()),
                    "used_beta_abs_mean": _to_float(used_beta.abs().mean()),
                    "gate_mean": _to_float(exec_info["gate"].mean()),
                    "correction_abs_mean": _to_float(exec_info["correction"].abs().mean()),
                    "correction_to_ct_l2_ratio": _to_float(exec_info["correction_ratio"].mean()),
                    "paired_pet_available": pet_features is not None,
                    "paired_pet_used_for_current_fusion": route == "full",
                    "paired_pet_used_for_future_memory": bool(update_memory and pet_features is not None),
                }
                if gamma_star is not None and beta_star is not None:
                    summary.update({
                        "true_gamma_abs_mean": _to_float(gamma_star.abs().mean()),
                        "true_beta_abs_mean": _to_float(beta_star.abs().mean()),
                        "true_gate_mean": _to_float(true_exec["gate"].mean()) if true_exec is not None else 0.0,
                    })
                if write_report is not None:
                    summary["memory_write"] = write_report
                if retrieval_info is not None:
                    summary.update({
                        "retrieval_entropy_mean": _to_float(retrieval_info["entropy"].mean()),
                        "retrieval_max_similarity_mean": _to_float(retrieval_info["max_similarity"].mean()),
                        "retrieval_top1_weight_mean": _to_float(retrieval_info["top1_weight"].mean()),
                    })
                per_scale_info.append(summary)

                if capture_visuals:
                    visual_maps[scale_name] = {
                        "gate": exec_info["gate"][0, 0].detach().cpu(),
                        "correction_norm": exec_info["correction"][0].float().norm(dim=0).detach().cpu(),
                        "used_gamma_norm": used_gamma[0].float().norm(dim=0).detach().cpu(),
                        "used_beta_norm": used_beta[0].float().norm(dim=0).detach().cpu(),
                    }
                    if gamma_star is not None and beta_star is not None:
                        visual_maps[scale_name]["true_gamma_norm"] = gamma_star[0].float().norm(dim=0).detach().cpu()
                        visual_maps[scale_name]["true_beta_norm"] = beta_star[0].float().norm(dim=0).detach().cpu()
                        visual_maps[scale_name]["true_action_strength"] = (
                            true_exec["gate"][0, 0]
                            * (gamma_star[0].abs().mean(dim=0) + beta_star[0].abs().mean(dim=0))
                        ).detach().cpu()
                    if route == "missing":
                        visual_maps[scale_name].update(memory._last_maps)

        if update_memory and pet_features is not None:
            self.paired_pet_memory_write_count += 1

        forward_info = {
            "epoch": self.current_epoch,
            "route": route,
            "K": self.K,
            "paired_pet_available": pet_features is not None,
            "paired_pet_used_for_current_fusion": route == "full",
            "paired_pet_used_for_future_memory": bool(update_memory and pet_features is not None),
            "leakage_guard": (
                "PASS: Missing 当前融合只使用上一轮冻结记忆。"
                if route == "missing"
                else "N/A: Full 使用当前真实 PET 仿射作用。"
            ),
            "scales": per_scale_info,
        }
        self._last_forward = forward_info
        if capture_visuals:
            self._last_visual_maps = visual_maps
        return fused_features, forward_info

    @torch.no_grad()
    def finalize_epoch_memory(self) -> Dict[str, Any]:
        reports = [memory.finalize_memory() for memory in self.memories]
        return {
            "epoch": self.current_epoch,
            "K": self.K,
            "delayed_update": True,
            "scales": reports,
        }

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "module": "PAAM",
            "full_name": "PET-induced Affine Action Memory",
            "epoch": self.current_epoch,
            "config": {
                "channels": list(self.channels),
                "K": self.K,
                "only_tunable_model_hyperparameter": "K",
                "auxiliary_loss_enabled": False,
                "per_scale_independent_memory": True,
                "memory_content": ["CT retrieval key", "gamma affine prototype", "beta affine prototype"],
                "prototype_clustering_space": "affine action [gamma, beta]",
                "full_missing_shared_executor": True,
                "memory_update": "epoch-wise delayed read/write",
            },
            "route_counts": dict(self.route_counts),
            "paired_pet_memory_write_batches": self.paired_pet_memory_write_count,
            "last_forward": self._last_forward,
            "scales": [memory.diagnostics() for memory in self.memories],
        }

    def print_diagnostics(self) -> None:
        d = self.diagnostics()
        print("=" * 88)
        print("PAAM 诊断摘要")
        print(
            f"Epoch={d['epoch']} | K={self.K} | Full batches={self.route_counts['full']} "
            f"| Missing batches={self.route_counts['missing']}"
        )
        print("原则：Full/Missing 共用同一仿射执行器；Missing 当前融合不读取当前真实 PET。")
        print("-" * 88)
        for sd in d["scales"]:
            running = sd["running"]
            print(
                f"S{sd['scale']} | C={sd['channels']} | ready={sd['memory_ready']} "
                f"| cache={sd['cache_count']} | active_slots={sd['active_slots_retrieval']}/{self.K}"
            )
            print(
                f"  retrieval: entropy={running['retrieval_entropy']:.4f}, "
                f"top1_w={running['retrieval_top1_weight']:.4f}, "
                f"max_sim={running['retrieval_max_similarity']:.4f}, "
                f"logit_scale={sd['logit_scale']:.3f}"
            )
            print(
                f"  prototypes: key_pair_cos={sd['key_pairwise_cosine_mean']:.4f}, "
                f"|gamma|={sd['gamma_prototype']['abs_mean']:.6f}, "
                f"|beta|={sd['beta_prototype']['abs_mean']:.6f}"
            )
            print(f"  slot_counts={sd['slot_counts_build']}")
        print("=" * 88)

    def export_diagnostics(
        self,
        output_dir: str | os.PathLike[str],
        epoch: Optional[int] = None,
        split: str = "train",
    ) -> Dict[str, str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        epoch_id = self.current_epoch if epoch is None else int(epoch)
        stem = f"epoch_{epoch_id:03d}_{split}"

        diagnostics = _json_ready(self.diagnostics())
        json_path = out / f"{stem}_paam_diagnostics.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(diagnostics, f, ensure_ascii=False, indent=2)

        saved: Dict[str, str] = {"json": str(json_path)}
        if self._last_visual_maps:
            bigfig_path = out / f"{stem}_paam_visual_summary.png"
            self._save_big_figure(bigfig_path, stem)
            saved["bigfig"] = str(bigfig_path)

            arrays: Dict[str, np.ndarray] = {}
            for scale_name, maps in self._last_visual_maps.items():
                for map_name, tensor in maps.items():
                    arrays[f"{scale_name}_{map_name}"] = tensor.detach().cpu().numpy()
            npz_path = out / f"{stem}_paam_maps.npz"
            np.savez_compressed(npz_path, **arrays)
            saved["npz"] = str(npz_path)

        manifest_path = out / f"{stem}_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(saved, f, ensure_ascii=False, indent=2)
        saved["manifest"] = str(manifest_path)

        print(f"[PAAM] 诊断已保存：{json_path}")
        print(f"[PAAM] 可视化目录：{out}")
        return saved

    def _save_big_figure(self, path: Path, stem: str) -> None:
        scales = list(self._last_visual_maps.keys())
        metric_order = [
            ("gate", "Gate"),
            ("correction_norm", "Correction Norm"),
            ("used_gamma_norm", "Used Gamma Norm"),
            ("used_beta_norm", "Used Beta Norm"),
            ("reliability", "Reliability"),
            ("effective_slots", "Effective Slots"),
            ("raw_gamma_norm", "Raw Gamma Norm"),
            ("safe_gamma_norm", "Safe Gamma Norm"),
            ("raw_beta_norm", "Raw Beta Norm"),
            ("safe_beta_norm", "Safe Beta Norm"),
        ]
        rows = len(scales)
        cols = len(metric_order)
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows))
        if rows == 1:
            axes = np.expand_dims(axes, axis=0)
        for r, scale_name in enumerate(scales):
            maps = self._last_visual_maps[scale_name]
            for c, (key, title) in enumerate(metric_order):
                ax = axes[r, c]
                if key in maps:
                    arr = maps[key].detach().float().cpu().numpy()
                    im = ax.imshow(arr)
                    ax.set_title(f"{scale_name.upper()} {title}")
                    ax.axis("off")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.axis("off")
                    ax.set_title(f"{scale_name.upper()} {title} (NA)")
        fig.suptitle(f"PAAM Visual Summary | {stem}", fontsize=16)
        plt.tight_layout()
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def _seed_everything(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dummy_features(
    batch_size: int,
    channels: Sequence[int],
    spatial_sizes: Sequence[Tuple[int, int]],
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    ct: List[torch.Tensor] = []
    pet: List[torch.Tensor] = []
    for c, (h, w) in zip(channels, spatial_sizes):
        ct_s = torch.randn(batch_size, c, h, w, device=device)
        pet_s = 0.45 * ct_s + 0.55 * torch.randn_like(ct_s)
        ct.append(ct_s)
        pet.append(pet_s)
    return ct, pet


def run_self_test(K: int, output_dir: str, device_name: str) -> None:
    _seed_everything(2026)
    device = torch.device(device_name)
    channels = (64, 128, 320, 512)
    spatial_sizes = ((32, 32), (16, 16), (8, 8), (4, 4))

    model = PETAffineActionMemory(channels=channels, K=K).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.begin_epoch(epoch=1)
    for step, route in enumerate(["full", "missing", "full", "missing"]):
        ct, pet = _dummy_features(2, channels, spatial_sizes, device)
        fused, info = model(
            ct,
            pet_features=pet,
            route=route,
            update_memory=True,
            capture_visuals=False,
        )
        pseudo_loss = sum(x.square().mean() for x in fused)
        optimizer.zero_grad(set_to_none=True)
        pseudo_loss.backward()
        optimizer.step()
        print(
            f"[Self-test] epoch=1 step={step} route={route} "
            f"loss={pseudo_loss.item():.6f} "
            f"paired_pet_used_for_current_fusion={info['paired_pet_used_for_current_fusion']}"
        )

    build_report = model.finalize_epoch_memory()
    print("[Self-test] Epoch 1 memory build:")
    print(json.dumps(_json_ready(build_report), ensure_ascii=False, indent=2))

    model.begin_epoch(epoch=2)
    ct, _ = _dummy_features(2, channels, spatial_sizes, device)
    _, missing_info = model(
        ct,
        pet_features=None,
        route="missing",
        update_memory=False,
        capture_visuals=True,
    )
    print("[Self-test] Missing leakage guard:", missing_info["leakage_guard"])
    model.print_diagnostics()
    saved = model.export_diagnostics(output_dir, epoch=2, split="selftest")
    print("[Self-test] Saved files:")
    print(json.dumps(saved, ensure_ascii=False, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PAAM 独立模块与自检。K 是唯一模型超参数。")
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="每个尺度的仿射作用原型数量；唯一需要调整的模型超参数。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./paam_debug",
        help="诊断 JSON 和可视化保存目录。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行设备；不是模型超参数。",
    )
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_self_test(K=args.k, output_dir=args.output_dir, device_name=args.device)