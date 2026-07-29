#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPBDM: CT-Conditioned PET Benefit Distribution Memory
=====================================================

独立 PyTorch 模块，用于 PET 缺失条件下的任务级纠偏。

核心思想
--------
1. 不恢复 PET 图像、PET 特征或仿射参数。
2. 对同一配对病例比较 Full 与 masked CT-only：Δp = p_full - p_ct。
3. 仅把 Full 确实降低局部损失的改变视为 PET 有益纠偏。
4. 只在测试可观察的 CT 任务状态空间中建立 Key。
5. 每个 Key 保存零/正/负纠偏概率和正负纠偏幅度分布。
6. Missing 时读取条件期望纠偏：
       E[Δp|q] = Σ α_k (π_k^+ μ_k^+ + π_k^- μ_k^-)
7. 不使用多重乘法门控；Memory 通过无梯度统计更新。

推荐集成位置
------------
Key  : 最后一级 CT-only decoder feature + CT probability + entropy + boundary
Value: PET 有益概率纠偏分布
作用 : 最终 segmentation logits 之后、loss 之前

运行示例
--------
python cpbdm_distribution_memory.py --output_dir ./cpbdm_demo_outputs --device cpu
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 基础工具
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_float(x: Any) -> float:
    return float(x.detach().cpu().item()) if isinstance(x, torch.Tensor) else float(x)


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().float()
    if x.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "abs_mean": 0.0}
    return {
        "mean": to_float(x.mean()),
        "std": to_float(x.std(unbiased=False)),
        "min": to_float(x.min()),
        "max": to_float(x.max()),
        "abs_mean": to_float(x.abs().mean()),
    }


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return to_float(obj) if obj.numel() == 1 else obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def pairwise_cosine_mean(x: torch.Tensor) -> float:
    if x.ndim != 2 or x.shape[0] < 2:
        return 0.0
    x = F.normalize(x.float(), dim=1, eps=1e-6)
    sim = x @ x.t()
    mask = ~torch.eye(x.shape[0], dtype=torch.bool, device=x.device)
    return to_float(sim[mask].mean())


def binary_entropy(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = prob.clamp(eps, 1.0 - eps)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


def probability_boundary(prob: torch.Tensor) -> torch.Tensor:
    dx = F.pad(prob[..., :, 1:] - prob[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(prob[..., 1:, :] - prob[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-8)


def deterministic_take(indices: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0 or indices.numel() == 0:
        return indices[:0]
    if indices.numel() <= count:
        return indices
    pos = torch.linspace(0, indices.numel() - 1, steps=count, device=indices.device).round().long()
    return indices[pos]


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


# -----------------------------------------------------------------------------
# CT 任务状态 Query
# -----------------------------------------------------------------------------

class QueryBuilder(nn.Module):
    """由 decoder feature、CT概率、熵和边界响应构造局部 Query。"""

    def __init__(self, decoder_channels: int, query_dim: int) -> None:
        super().__init__()
        hidden = max(query_dim, 16)
        self.proj = nn.Sequential(
            nn.Conv2d(decoder_channels + 3, hidden, 1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, query_dim, 1, bias=False),
        )

    def forward(
        self,
        decoder_feature: torch.Tensor,
        ct_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if decoder_feature.ndim != 4 or ct_logits.ndim != 4:
            raise ValueError("decoder_feature 和 ct_logits 必须为 BCHW。")
        if ct_logits.shape[1] != 1:
            raise ValueError("当前脚本针对二分类分割，ct_logits 通道必须为 1。")

        if decoder_feature.shape[-2:] != ct_logits.shape[-2:]:
            decoder_feature = F.interpolate(
                decoder_feature,
                size=ct_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        prob = torch.sigmoid(ct_logits)
        entropy = binary_entropy(prob)
        boundary = probability_boundary(prob)
        state = torch.cat([decoder_feature, prob, entropy, boundary], dim=1)
        raw = self.proj(state)

        # 去除每个病例的公共空间方向，缓解 Key 高度同向。
        centered = raw - raw.mean(dim=(2, 3), keepdim=True)
        query = F.normalize(centered, dim=1, eps=1e-6)
        return query, {
            "probability": prob,
            "entropy": entropy,
            "boundary": boundary,
            "query_raw": raw,
            "query_centered": centered,
        }


# -----------------------------------------------------------------------------
# CPBDM
# -----------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    delta_probability: torch.Tensor
    corrected_probability: torch.Tensor
    corrected_logits: torch.Tensor
    info: Dict[str, torch.Tensor]


class CTConditionedPETBenefitDistributionMemory(nn.Module):
    """
    每个槽保存：
      key, π0, π+, π-, μ+, σ+, μ-, σ-, count
    """

    EVENT_ZERO = 0
    EVENT_POS = 1
    EVENT_NEG = -1

    def __init__(
        self,
        decoder_channels: int,
        query_dim: int = 32,
        K: int = 8,
        fit_cache_capacity: int = 30000,
        stat_cache_capacity: int = 80000,
        max_fit_samples_per_image: int = 384,
        max_stat_samples_per_image: int = 1024,
        kmeans_iters: int = 25,
    ) -> None:
        super().__init__()
        if K < 1:
            raise ValueError("K 必须 >= 1。")

        self.decoder_channels = int(decoder_channels)
        self.query_dim = int(query_dim)
        self.K = int(K)
        self.fit_cache_capacity = int(fit_cache_capacity)
        self.stat_cache_capacity = int(stat_cache_capacity)
        self.max_fit_samples_per_image = int(max_fit_samples_per_image)
        self.max_stat_samples_per_image = int(max_stat_samples_per_image)
        self.kmeans_iters = int(kmeans_iters)

        self.query_builder = QueryBuilder(self.decoder_channels, self.query_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))

        self.register_buffer("keys", torch.zeros(self.K, self.query_dim))
        self.register_buffer("pi_zero", torch.ones(self.K))
        self.register_buffer("pi_pos", torch.zeros(self.K))
        self.register_buffer("pi_neg", torch.zeros(self.K))
        self.register_buffer("mu_pos", torch.zeros(self.K))
        self.register_buffer("sigma_pos", torch.zeros(self.K))
        self.register_buffer("mu_neg", torch.zeros(self.K))
        self.register_buffer("sigma_neg", torch.zeros(self.K))
        self.register_buffer("slot_counts", torch.zeros(self.K))
        self.register_buffer("slot_zero_counts", torch.zeros(self.K))
        self.register_buffer("slot_pos_counts", torch.zeros(self.K))
        self.register_buffer("slot_neg_counts", torch.zeros(self.K))
        self.register_buffer("memory_ready", torch.tensor(False, dtype=torch.bool))

        # 平衡 Key 拟合缓存。
        self._fit_q: List[torch.Tensor] = []
        # 真实比例统计缓存。
        self._stat_q: List[torch.Tensor] = []
        self._stat_event: List[torch.Tensor] = []
        self._stat_delta: List[torch.Tensor] = []
        self._stat_benefit: List[torch.Tensor] = []
        self._stat_weight: List[torch.Tensor] = []

        self._retrieval_stats = {
            "entropy": RunningMean(),
            "effective_slots": RunningMean(),
            "top1_weight": RunningMean(),
            "max_similarity": RunningMean(),
            "delta_abs_mean": RunningMean(),
        }
        self._last_maps: Dict[str, torch.Tensor] = {}

    @staticmethod
    def compute_paired_events(
        ct_logits: torch.Tensor,
        full_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """比较同一病例 Full 与 masked CT-only 的局部收益。"""
        if target.shape[-2:] != ct_logits.shape[-2:]:
            target = F.interpolate(target.float(), size=ct_logits.shape[-2:], mode="nearest")
        target = target.float()

        ct_loss = F.binary_cross_entropy_with_logits(ct_logits, target, reduction="none")
        full_loss = F.binary_cross_entropy_with_logits(full_logits, target, reduction="none")
        benefit = ct_loss - full_loss

        ct_prob = torch.sigmoid(ct_logits)
        full_prob = torch.sigmoid(full_logits)
        raw_delta = full_prob - ct_prob
        beneficial = benefit > 0

        event = torch.zeros_like(raw_delta, dtype=torch.int8)
        event[beneficial & (raw_delta > 0)] = CTConditionedPETBenefitDistributionMemory.EVENT_POS
        event[beneficial & (raw_delta < 0)] = CTConditionedPETBenefitDistributionMemory.EVENT_NEG
        delta_star = torch.where(beneficial, raw_delta, torch.zeros_like(raw_delta))

        return {
            "ct_probability": ct_prob,
            "full_probability": full_prob,
            "raw_delta": raw_delta,
            "delta_star": delta_star,
            "benefit": benefit,
            "event": event,
            "target": target,
        }

    @staticmethod
    def _trim_synced_caches(caches: Sequence[List[torch.Tensor]], capacity: int) -> None:
        if not caches or not caches[0]:
            return
        total = sum(int(x.shape[0]) for x in caches[0])
        if total <= capacity:
            return
        cats = [torch.cat(c, dim=0) for c in caches]
        idx = torch.linspace(0, cats[0].shape[0] - 1, steps=capacity).round().long()
        for cache, cat in zip(caches, cats):
            cache[:] = [cat[idx]]

    @torch.no_grad()
    def collect_from_pair(
        self,
        ct_decoder_feature: torch.Tensor,
        ct_logits: torch.Tensor,
        full_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Epoch 末在 eval + no_grad 下调用。
        同一病例：m=1 得 Full；m=0 得 masked CT-only。
        """
        query, q_aux = self.query_builder(ct_decoder_feature, ct_logits)
        events = self.compute_paired_events(ct_logits, full_logits, target)

        b, _, h, w = query.shape
        q_flat = query.permute(0, 2, 3, 1).reshape(b, h * w, self.query_dim)
        e_flat = events["event"].reshape(b, h * w)
        d_flat = events["delta_star"].reshape(b, h * w)
        g_flat = events["benefit"].reshape(b, h * w)
        entropy_flat = q_aux["entropy"].reshape(b, h * w)

        fit_selected = 0
        stat_selected = 0
        raw_counts = {"zero": 0, "positive": 0, "negative": 0}

        for bi in range(b):
            all_idx = torch.arange(h * w, device=query.device)
            zero_idx = all_idx[e_flat[bi] == self.EVENT_ZERO]
            pos_idx = all_idx[e_flat[bi] == self.EVENT_POS]
            neg_idx = all_idx[e_flat[bi] == self.EVENT_NEG]

            raw_counts["zero"] += int(zero_idx.numel())
            raw_counts["positive"] += int(pos_idx.numel())
            raw_counts["negative"] += int(neg_idx.numel())

            # A. 三类平衡采样拟合 CT Key。
            quota = max(1, self.max_fit_samples_per_image // 3)
            if pos_idx.numel() > quota:
                pos_idx = pos_idx[torch.topk(g_flat[bi, pos_idx], k=quota).indices]
            if neg_idx.numel() > quota:
                neg_idx = neg_idx[torch.topk(g_flat[bi, neg_idx], k=quota).indices]
            if zero_idx.numel() > quota:
                # 零事件优先覆盖 CT 不确定区域，避免全是普通背景。
                zero_idx = zero_idx[torch.topk(entropy_flat[bi, zero_idx], k=quota).indices]

            fit_idx = torch.cat([zero_idx, pos_idx, neg_idx], dim=0)
            if fit_idx.numel() > 0:
                self._fit_q.append(q_flat[bi, fit_idx].float().cpu())
                fit_selected += int(fit_idx.numel())

            # B. 近似均匀采样，估计真实零/正/负比例。
                total_budget = self.max_stat_samples_per_image
            pos_quota = total_budget // 3
            neg_quota = total_budget // 3
            zero_quota = total_budget - pos_quota - neg_quota

            sampled_zero = deterministic_take(zero_idx, min(zero_quota, int(zero_idx.numel())))
            sampled_pos = deterministic_take(pos_idx, min(pos_quota, int(pos_idx.numel())))
            sampled_neg = deterministic_take(neg_idx, min(neg_quota, int(neg_idx.numel())))

            stat_idx = torch.cat([sampled_zero, sampled_pos, sampled_neg], dim=0)
            zero_sample_weight = zero_idx.numel() / max(sampled_zero.numel(), 1)
            pos_sample_weight = pos_idx.numel() / max(sampled_pos.numel(), 1)
            neg_sample_weight = neg_idx.numel() / max(sampled_neg.numel(), 1)
            zero_weights = torch.full((sampled_zero.numel(),), float(zero_sample_weight), device=query.device, dtype=torch.float32)
            pos_weights = torch.full((sampled_pos.numel(),), float(pos_sample_weight), device=query.device, dtype=torch.float32)
            neg_weights = torch.full((sampled_neg.numel(),), float(neg_sample_weight), device=query.device, dtype=torch.float32)
            stat_weight = torch.cat([zero_weights, pos_weights, neg_weights], dim=0)
            self._stat_q.append(q_flat[bi, stat_idx].float().cpu())
            self._stat_event.append(e_flat[bi, stat_idx].cpu())
            self._stat_delta.append(d_flat[bi, stat_idx].float().cpu())
            self._stat_benefit.append(g_flat[bi, stat_idx].float().cpu())
            self._stat_weight.append(stat_weight.detach().float().cpu())
            assert self._stat_q[-1].shape[0] == self._stat_event[-1].shape[0] == self._stat_delta[-1].shape[0] == self._stat_benefit[-1].shape[0] == self._stat_weight[-1].shape[0]
            stat_selected += int(stat_idx.numel())

        self._trim_synced_caches([self._fit_q], self.fit_cache_capacity)
        self._trim_synced_caches(
            [self._stat_q, self._stat_event, self._stat_delta, self._stat_benefit, self._stat_weight],
            self.stat_cache_capacity,
        )

        return {
            "fit_selected": fit_selected,
            "stat_selected": stat_selected,
            "raw_event_counts": raw_counts,
            "query_raw_abs_mean": to_float(q_aux["query_raw"].abs().mean()),
            "query_centered_abs_mean": to_float(q_aux["query_centered"].abs().mean()),
            "query_spatial_variance": to_float(
                q_aux["query_centered"].var(dim=(2, 3), unbiased=False).mean()
            ),
            "benefit_mean": to_float(events["benefit"].mean()),
            "delta_star_abs_mean": to_float(events["delta_star"].abs().mean()),
        }

    @torch.no_grad()
    def _spherical_kmeans(self, q: torch.Tensor) -> torch.Tensor:
        q = F.normalize(q.float(), dim=1, eps=1e-6)
        if q.shape[0] < self.K:
            q = q.repeat(math.ceil(self.K / max(q.shape[0], 1)), 1)[: self.K]

        chosen = [0]
        centers = [q[0]]
        while len(centers) < self.K:
            current = torch.stack(centers, dim=0)
            max_sim = (q @ current.t()).max(dim=1).values
            max_sim[torch.tensor(chosen, dtype=torch.long)] = 1.0
            nxt = int(max_sim.argmin().item())
            chosen.append(nxt)
            centers.append(q[nxt])
        centers_t = torch.stack(centers, dim=0)

        for _ in range(self.kmeans_iters):
            assignment = (q @ centers_t.t()).argmax(dim=1)
            updated = []
            for k in range(self.K):
                mask = assignment == k
                if mask.any():
                    center = F.normalize(q[mask].mean(dim=0), dim=0, eps=1e-6)
                else:
                    max_sim = (q @ centers_t.t()).max(dim=1).values
                    center = q[max_sim.argmin()]
                updated.append(center)
            updated_t = torch.stack(updated, dim=0)
            if torch.allclose(centers_t, updated_t, atol=1e-5, rtol=1e-4):
                centers_t = updated_t
                break
            centers_t = updated_t
        return F.normalize(centers_t, dim=1, eps=1e-6)

    @torch.no_grad()
    def finalize_memory(self) -> Dict[str, Any]:
        """每个 epoch 根据当前模型重新建库，直接替换旧 Memory，不做 EMA。"""
        if not self._fit_q or not self._stat_q:
            return {"updated": False, "reason": "empty_cache", "memory_ready": bool(self.memory_ready)}

        fit_q = torch.cat(self._fit_q, dim=0).float()
        stat_q = torch.cat(self._stat_q, dim=0).float()
        stat_event = torch.cat(self._stat_event, dim=0).long()
        stat_delta = torch.cat(self._stat_delta, dim=0).float()
        stat_benefit = torch.cat(self._stat_benefit, dim=0).float()
        stat_weight = torch.cat(self._stat_weight, dim=0).float()

        centers = self._spherical_kmeans(fit_q)
        assignment = (F.normalize(stat_q, dim=1, eps=1e-6) @ centers.t()).argmax(dim=1)

        values = {name: torch.zeros(self.K) for name in [
            "pi_zero", "pi_pos", "pi_neg", "mu_pos", "sigma_pos",
            "mu_neg", "sigma_neg", "count", "zero_count", "pos_count", "neg_count", "weighted_mass"
        ]}

        for k in range(self.K):
            slot = assignment == k
            n = int(slot.sum().item())
            values["count"][k] = n
            if n == 0:
                values["pi_zero"][k] = 1.0
                continue

            e = stat_event[slot]
            d = stat_delta[slot]
            w = stat_weight[slot]
            zero, pos, neg = e == self.EVENT_ZERO, e == self.EVENT_POS, e == self.EVENT_NEG
            values["zero_count"][k] = zero.sum()
            values["pos_count"][k] = pos.sum()
            values["neg_count"][k] = neg.sum()
            zero_mass = w[zero].sum()
            pos_mass = w[pos].sum()
            neg_mass = w[neg].sum()
            total_mass = zero_mass + pos_mass + neg_mass
            values["weighted_mass"][k] = total_mass
            if total_mass > 0:
                values["pi_zero"][k] = zero_mass / total_mass
                values["pi_pos"][k] = pos_mass / total_mass
                values["pi_neg"][k] = neg_mass / total_mass
            else:
                values["pi_zero"][k] = 1.0
            if pos.any() and pos_mass > 0:
                pos_w = w[pos]
                pos_d = d[pos]
                values["mu_pos"][k] = (pos_w * pos_d).sum() / pos_w.sum()
                var_pos = (pos_w * (pos_d - values["mu_pos"][k]).square()).sum() / pos_w.sum()
                values["sigma_pos"][k] = torch.sqrt(var_pos.clamp_min(0.0))
            if neg.any() and neg_mass > 0:
                neg_w = w[neg]
                neg_d = d[neg]
                values["mu_neg"][k] = (neg_w * neg_d).sum() / neg_w.sum()
                var_neg = (neg_w * (neg_d - values["mu_neg"][k]).square()).sum() / neg_w.sum()
                values["sigma_neg"][k] = torch.sqrt(var_neg.clamp_min(0.0))

        device = self.keys.device
        self.keys.copy_(centers.to(device))
        self.pi_zero.copy_(values["pi_zero"].to(device))
        self.pi_pos.copy_(values["pi_pos"].to(device))
        self.pi_neg.copy_(values["pi_neg"].to(device))
        self.mu_pos.copy_(values["mu_pos"].to(device))
        self.sigma_pos.copy_(values["sigma_pos"].to(device))
        self.mu_neg.copy_(values["mu_neg"].to(device))
        self.sigma_neg.copy_(values["sigma_neg"].to(device))
        self.slot_counts.copy_(values["count"].to(device))
        self.slot_zero_counts.copy_(values["zero_count"].to(device))
        self.slot_pos_counts.copy_(values["pos_count"].to(device))
        self.slot_neg_counts.copy_(values["neg_count"].to(device))
        self.memory_ready.fill_(True)

        report = {
            "updated": True,
            "memory_ready": True,
            "K": self.K,
            "fit_candidate_count": int(fit_q.shape[0]),
            "stat_candidate_count": int(stat_q.shape[0]),
            "key_pairwise_cosine_mean": pairwise_cosine_mean(centers),
            "slot_counts": values["count"].tolist(),
            "slot_zero_counts": values["zero_count"].tolist(),
            "slot_positive_counts": values["pos_count"].tolist(),
            "slot_negative_counts": values["neg_count"].tolist(),
            "slot_weighted_mass": values["weighted_mass"].tolist(),
            "pi_zero": values["pi_zero"].tolist(),
            "pi_positive": values["pi_pos"].tolist(),
            "pi_negative": values["pi_neg"].tolist(),
            "mu_positive": values["mu_pos"].tolist(),
            "sigma_positive": values["sigma_pos"].tolist(),
            "mu_negative": values["mu_neg"].tolist(),
            "sigma_negative": values["sigma_neg"].tolist(),
            "sampled_event_count": {
                "zero": int((stat_event == self.EVENT_ZERO).sum()),
                "positive": int((stat_event == self.EVENT_POS).sum()),
                "negative": int((stat_event == self.EVENT_NEG).sum()),
            },
            "weighted_event_mass": {
                "zero": float((stat_weight * (stat_event == self.EVENT_ZERO).float()).sum()),
                "positive": float((stat_weight * (stat_event == self.EVENT_POS).float()).sum()),
                "negative": float((stat_weight * (stat_event == self.EVENT_NEG).float()).sum()),
            },
            "weighted_event_ratio": {
                "zero": float((stat_weight * (stat_event == self.EVENT_ZERO).float()).sum() / stat_weight.sum().clamp_min(1.0)),
                "positive": float((stat_weight * (stat_event == self.EVENT_POS).float()).sum() / stat_weight.sum().clamp_min(1.0)),
                "negative": float((stat_weight * (stat_event == self.EVENT_NEG).float()).sum() / stat_weight.sum().clamp_min(1.0)),
            },
            "global_event_counts": {
                "zero": int((stat_event == self.EVENT_ZERO).sum()),
                "positive": int((stat_event == self.EVENT_POS).sum()),
                "negative": int((stat_event == self.EVENT_NEG).sum()),
            },
            "global_benefit_stats": tensor_stats(stat_benefit),
            "global_delta_stats": tensor_stats(stat_delta),
        }
        self.clear_cache()
        return report

    def clear_cache(self) -> None:
        self._fit_q.clear()
        self._stat_q.clear()
        self._stat_event.clear()
        self._stat_delta.clear()
        self._stat_benefit.clear()

    def slot_expected_delta(self) -> torch.Tensor:
        # 零纠偏概率乘以 0，因此无需额外门控。
        return self.pi_pos * self.mu_pos + self.pi_neg * self.mu_neg

    def retrieve(
        self,
        ct_decoder_feature: torch.Tensor,
        ct_logits: torch.Tensor,
        capture_maps: bool = False,
    ) -> RetrievalResult:
        query, q_aux = self.query_builder(ct_decoder_feature, ct_logits)
        b, _, h, w = query.shape
        ct_prob = torch.sigmoid(ct_logits.float()).to(ct_logits.dtype)

        if not bool(self.memory_ready.item()):
            info = {
                "memory_ready": False,
                "retrieval_entropy_mean": 0.0,
                "effective_slots_mean": 1.0,
                "top1_weight_mean": 0.0,
                "max_similarity_mean": 0.0,
                "delta_abs_mean": 0.0,
            }
            return RetrievalResult(torch.zeros_like(ct_logits), ct_prob, ct_logits, info)

        q = F.normalize(query.float().permute(0, 2, 3, 1).reshape(-1, self.query_dim), dim=1, eps=1e-6)
        keys = F.normalize(self.keys.float(), dim=1, eps=1e-6)
        similarity = q @ keys.t()
        scale = self.logit_scale.exp().clamp(1.0, 20.0)
        weights = torch.softmax(similarity * scale, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

        delta_flat = weights @ self.slot_expected_delta().float()
        delta = delta_flat.reshape(b, 1, h, w).to(ct_logits.dtype)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.25, neginf=-0.25).clamp(-0.25, 0.25)
        corrected_prob = (ct_prob + delta).clamp(1e-4, 1.0 - 1e-4)
        corrected_logits = torch.logit(corrected_prob.float()).to(ct_logits.dtype)
        corrected_logits = torch.nan_to_num(corrected_logits, nan=0.0, posinf=10.0, neginf=-10.0)

        p = weights.clamp_min(1e-8)
        entropy = -(p * p.log()).sum(dim=1)
        effective = entropy.exp()
        top1_weight, top1_slot = weights.max(dim=1)
        max_similarity = similarity.max(dim=1).values
        with torch.no_grad():
            n = int(entropy.numel())
            for name, value in [("entropy", entropy.mean()), ("effective_slots", effective.mean()), ("top1_weight", top1_weight.mean()), ("max_similarity", max_similarity.mean()), ("delta_abs_mean", delta.abs().mean())]:
                self._retrieval_stats[name].update(to_float(value), n=n)
            if capture_maps:
                entropy_map = entropy.reshape(b, 1, h, w)
                effective_map = effective.reshape(b, 1, h, w)
                top1_weight_map = top1_weight.reshape(b, 1, h, w)
                top1_slot_map = top1_slot.reshape(b, 1, h, w)
                max_similarity_map = max_similarity.reshape(b, 1, h, w)
                self._last_maps = {
                    "ct_probability": ct_prob[0, 0].cpu(),
                    "ct_entropy": q_aux["entropy"][0, 0].cpu(),
                    "ct_boundary": q_aux["boundary"][0, 0].cpu(),
                    "delta_probability": delta[0, 0].cpu(),
                    "corrected_probability": corrected_prob[0, 0].cpu(),
                    "retrieval_entropy": entropy_map[0, 0].cpu(),
                    "effective_slots": effective_map[0, 0].cpu(),
                    "top1_weight": top1_weight_map[0, 0].cpu(),
                    "top1_slot": top1_slot_map[0, 0].cpu(),
                    "max_similarity": max_similarity_map[0, 0].cpu(),
                }
        info = {"memory_ready": True, "retrieval_entropy_mean": to_float(entropy.mean()), "effective_slots_mean": to_float(effective.mean()), "top1_weight_mean": to_float(top1_weight.mean()), "max_similarity_mean": to_float(max_similarity.mean()), "delta_abs_mean": to_float(delta.abs().mean())}
        return RetrievalResult(delta, corrected_prob, corrected_logits, info)

    def forward(
        self,
        ct_decoder_feature: torch.Tensor,
        ct_logits: torch.Tensor,
        capture_maps: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        result = self.retrieve(ct_decoder_feature, ct_logits, capture_maps)
        info = dict(result.info)
        info["delta_probability"] = result.delta_probability
        info["corrected_probability"] = result.corrected_probability
        return result.corrected_logits, info

    def reset_retrieval_stats(self) -> None:
        for meter in self._retrieval_stats.values():
            meter.reset()
        self._last_maps.clear()

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "module": "CPBDM",
            "decoder_channels": self.decoder_channels,
            "query_dim": self.query_dim,
            "K": self.K,
            "memory_ready": bool(self.memory_ready.item()),
            "key_pairwise_cosine_mean": pairwise_cosine_mean(self.keys.cpu()) if self.memory_ready else 0.0,
            "logit_scale": to_float(self.logit_scale.exp().clamp(1.0, 100.0)),
            "slot_counts": self.slot_counts.cpu().tolist(),
            "slot_zero_counts": self.slot_zero_counts.cpu().tolist(),
            "slot_positive_counts": self.slot_pos_counts.cpu().tolist(),
            "slot_negative_counts": self.slot_neg_counts.cpu().tolist(),
            "pi_zero": self.pi_zero.cpu().tolist(),
            "pi_positive": self.pi_pos.cpu().tolist(),
            "pi_negative": self.pi_neg.cpu().tolist(),
            "mu_positive": self.mu_pos.cpu().tolist(),
            "sigma_positive": self.sigma_pos.cpu().tolist(),
            "mu_negative": self.mu_neg.cpu().tolist(),
            "sigma_negative": self.sigma_neg.cpu().tolist(),
            "slot_expected_delta": self.slot_expected_delta().detach().cpu().tolist(),
            "retrieval_running": {k: v.mean for k, v in self._retrieval_stats.items()},
        }

    def print_diagnostics(self) -> None:
        d = self.diagnostics()
        print("=" * 94)
        print("CPBDM 诊断摘要")
        print(
            f"ready={d['memory_ready']} | K={d['K']} | query_dim={d['query_dim']} | "
            f"key_pair_cos={d['key_pairwise_cosine_mean']:.4f} | logit_scale={d['logit_scale']:.3f}"
        )
        print("-" * 94)
        for k in range(self.K):
            print(
                f"Slot {k:02d} | n={d['slot_counts'][k]:7.0f} | "
                f"π0={d['pi_zero'][k]:.3f} π+={d['pi_positive'][k]:.3f} π-={d['pi_negative'][k]:.3f} | "
                f"μ+={d['mu_positive'][k]:+.4f} μ-={d['mu_negative'][k]:+.4f} | "
                f"EΔp={d['slot_expected_delta'][k]:+.4f}"
            )
        r = d["retrieval_running"]
        print("-" * 94)
        print(
            f"retrieval | entropy={r['entropy']:.4f} | effective_slots={r['effective_slots']:.4f} | "
            f"top1_weight={r['top1_weight']:.4f} | max_similarity={r['max_similarity']:.4f} | "
            f"|Δp|={r['delta_abs_mean']:.6f}"
        )
        print("=" * 94)

    def export_json(
        self,
        output_dir: str | os.PathLike[str],
        tag: str,
        build_report: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{tag}_cpbdm_diagnostics.json"
        payload = {"tag": tag, "diagnostics": self.diagnostics(), "build_report": build_report or {}, "extra": extra or {}}
        path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[CPBDM] JSON 已保存：{path}")
        return path

    # ------------------------------ 单张组合图 ------------------------------
    @staticmethod
    def _render_map(array: np.ndarray, title: str, vmin=None, vmax=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        fig = plt.figure(figsize=(3.1, 2.9), dpi=120)
        ax = fig.add_axes([0.08, 0.12, 0.84, 0.78])
        im = ax.imshow(array, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig); buf.seek(0)
        return Image.open(buf).convert('RGB')

    @staticmethod
    def _render_bar(values: Sequence[float], title: str, ylabel: str):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        fig = plt.figure(figsize=(3.1, 2.9), dpi=120)
        ax = fig.add_axes([0.18, 0.18, 0.76, 0.70])
        x = np.arange(len(values))
        ax.bar(x, np.asarray(values))
        ax.set_title(title)
        ax.set_xlabel('Memory slot')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig); buf.seek(0)
        return Image.open(buf).convert('RGB')

    @staticmethod
    def _compose(images: Sequence[Image.Image], rows: int, cols: int, path: Path) -> None:
        width = max(im.width for im in images)
        height = max(im.height for im in images)
        canvas = Image.new("RGB", (cols * width, rows * height), "white")
        for i, im in enumerate(images):
            r, c = divmod(i, cols)
            canvas.paste(im, (c * width + (width - im.width) // 2, r * height + (height - im.height) // 2))
        canvas.save(path)

    @torch.no_grad()
    def save_composite_visualization(self, output_dir, tag, target=None, full_probability=None, benefit=None, event=None):
        try:
            if not self._last_maps:
                raise RuntimeError('请先使用 capture_maps=True 执行 Missing forward。')
            import matplotlib; matplotlib.use('Agg')
            from PIL import Image
            m = self._last_maps; zeros = np.zeros_like(m['ct_probability'].numpy())
            panels = [self._render_map(m['ct_probability'].numpy(), 'Masked CT-only probability', 0, 1), self._render_map(m['corrected_probability'].numpy(), 'CPBDM corrected probability', 0, 1), self._render_map(m['delta_probability'].numpy(), 'Retrieved probability correction', -1, 1), self._render_map(m['ct_entropy'].numpy(), 'CT prediction entropy'), self._render_map(m['ct_boundary'].numpy(), 'CT probability boundary'), self._render_map(m['retrieval_entropy'].numpy(), 'Retrieval entropy'), self._render_map(m['effective_slots'].numpy(), 'Effective retrieved slots'), self._render_map(m['top1_slot'].numpy(), 'Top-1 memory slot'), self._render_map(target[0, 0].cpu().numpy() if target is not None else zeros, 'Ground truth', 0, 1), self._render_map(full_probability[0, 0].cpu().numpy() if full_probability is not None else zeros, 'Full CT+PET probability', 0, 1), self._render_map(benefit[0, 0].cpu().numpy() if benefit is not None else zeros, 'Full-vs-CT local benefit'), self._render_map(event[0, 0].cpu().numpy() if event is not None else zeros, 'Event map: -1 / 0 / +1', -1, 1)]
            output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f'{tag}_cpbdm_composite.png'; self._compose(panels, rows=3, cols=4, path=path)
            print(f'[CPBDM] 单张组合可视化已保存：{path}')
            return path
        except Exception as e:
            print(f'[CPBDM WARNING] composite visualization failed: {e}')
            return Path(output_dir) / f'{tag}_cpbdm_composite.png'


# -----------------------------------------------------------------------------
# Synthetic self-test
# -----------------------------------------------------------------------------

def coordinate_grid(batch: int, h: int, w: int, device: torch.device):
    yy = torch.linspace(-1, 1, h, device=device)
    xx = torch.linspace(-1, 1, w, device=device)
    y, x = torch.meshgrid(yy, xx, indexing="ij")
    return x.expand(batch, 1, h, w), y.expand(batch, 1, h, w)


def make_synthetic_batch(batch: int, channels: int, h: int, w: int, device: torch.device):
    x, y = coordinate_grid(batch, h, w, device)
    targets, cts, fulls, feats = [], [], [], []

    for bi in range(batch):
        cx = float(torch.empty(1).uniform_(-0.45, 0.45))
        cy = float(torch.empty(1).uniform_(-0.45, 0.45))
        radius = float(torch.empty(1).uniform_(0.12, 0.27))
        lesion = (((x[bi:bi+1]-cx)**2 + (y[bi:bi+1]-cy)**2) <= radius**2).float()

        ct = -3.0 + 5.0 * lesion + 0.55 * torch.randn_like(lesion)
        miss_cx = cx + float(torch.empty(1).uniform_(-0.08, 0.08))
        miss_cy = cy + float(torch.empty(1).uniform_(-0.08, 0.08))
        miss_r = radius * float(torch.empty(1).uniform_(0.30, 0.55))
        miss = (((x[bi:bi+1]-miss_cx)**2 + (y[bi:bi+1]-miss_cy)**2) <= miss_r**2).float() * lesion
        ct = ct - 3.0 * miss

        fp_cx = float(torch.empty(1).uniform_(-0.75, 0.75))
        fp_cy = float(torch.empty(1).uniform_(-0.75, 0.75))
        fp_r = float(torch.empty(1).uniform_(0.06, 0.13))
        fp = (((x[bi:bi+1]-fp_cx)**2 + (y[bi:bi+1]-fp_cy)**2) <= fp_r**2).float() * (1-lesion)
        ct = ct + 4.0 * fp

        full = ct + 2.8 * miss - 3.2 * fp
        # 少量有害 PET 干扰，应该被归入零纠偏。
        hx = float(torch.empty(1).uniform_(-0.8, 0.8))
        hy = float(torch.empty(1).uniform_(-0.8, 0.8))
        hr = float(torch.empty(1).uniform_(0.04, 0.09))
        harmful = (((x[bi:bi+1]-hx)**2 + (y[bi:bi+1]-hy)**2) <= hr**2).float() * (1-lesion)
        full = full + 1.2 * harmful

        p = torch.sigmoid(ct)
        base = torch.cat([
            p, binary_entropy(p), probability_boundary(p),
            x[bi:bi+1], y[bi:bi+1], miss, fp, torch.randn_like(p) * 0.1,
        ], dim=1)
        if channels < base.shape[1]:
            base = base[:, :channels]
        elif channels > base.shape[1]:
            base = torch.cat([base, torch.randn(1, channels-base.shape[1], h, w, device=device)*0.1], dim=1)

        targets.append(lesion)
        cts.append(ct)
        fulls.append(full)
        feats.append(base)

    return {
        "decoder_feature": torch.cat(feats),
        "ct_logits": torch.cat(cts),
        "full_logits": torch.cat(fulls),
        "target": torch.cat(targets),
    }


def dice(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) >= 0.5).float()
    inter = (pred * target).sum()
    return to_float((2 * inter + 1e-6) / (pred.sum() + target.sum() + 1e-6))


def run_demo(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device)
    set_seed(args.seed)
    model = CTConditionedPETBenefitDistributionMemory(
        decoder_channels=args.decoder_channels,
        query_dim=args.query_dim,
        K=args.K,
        fit_cache_capacity=args.fit_cache_capacity,
        stat_cache_capacity=args.stat_cache_capacity,
        max_fit_samples_per_image=args.max_fit_samples_per_image,
        max_stat_samples_per_image=args.max_stat_samples_per_image,
        kmeans_iters=args.kmeans_iters,
    ).to(device).eval()

    reports = []
    with torch.no_grad():
        for step in range(args.demo_steps):
            batch = make_synthetic_batch(args.batch_size, args.decoder_channels, args.image_size, args.image_size, device)
            report = model.collect_from_pair(
                batch["decoder_feature"], batch["ct_logits"], batch["full_logits"], batch["target"]
            )
            reports.append(report)
            if (step + 1) % max(1, args.demo_steps // 4) == 0:
                print(
                    f"[collect {step+1:03d}/{args.demo_steps:03d}] fit={report['fit_selected']} "
                    f"stat={report['stat_selected']} events={report['raw_event_counts']} "
                    f"|Δp*|={report['delta_star_abs_mean']:.6f}"
                )

    build_report = model.finalize_memory()
    if not build_report.get("updated"):
        raise RuntimeError(f"Memory 构建失败：{build_report}")

    test = make_synthetic_batch(1, args.decoder_channels, args.image_size, args.image_size, device)
    model.reset_retrieval_stats()
    with torch.no_grad():
        corrected_logits, info = model(test["decoder_feature"], test["ct_logits"], capture_maps=True)
        paired = model.compute_paired_events(test["ct_logits"], test["full_logits"], test["target"])

    metrics = {
        "ct_only_dice": dice(test["ct_logits"], test["target"]),
        "full_oracle_dice": dice(test["full_logits"], test["target"]),
        "cpbdm_corrected_dice": dice(corrected_logits, test["target"]),
        "retrieved_delta_stats": tensor_stats(info["delta_probability"]),
        "corrected_probability_stats": tensor_stats(torch.sigmoid(corrected_logits)),
    }

    model.print_diagnostics()
    print("[CPBDM] Demo 指标：")
    print(f"  CT-only Dice     = {metrics['ct_only_dice']:.4f}")
    print(f"  Full oracle Dice = {metrics['full_oracle_dice']:.4f}")
    print(f"  CPBDM corrected  = {metrics['cpbdm_corrected_dice']:.4f}")
    print("说明：synthetic demo 只验证代码闭环，不代表真实数据一定提升。")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = model.export_json(
        out, "demo", build_report,
        extra={"metrics": metrics, "last_collect_report": reports[-1], "config": vars(args)},
    )
    vis_path = model.save_composite_visualization(
        out, "demo", target=test["target"], full_probability=paired["full_probability"],
        benefit=paired["benefit"], event=paired["event"],
    )
    state_path = out / "demo_cpbdm_state.pt"
    torch.save(model.state_dict(), state_path)
    print(f"[CPBDM] state_dict 已保存：{state_path}")
    return {
        "json_path": str(json_path),
        "visualization_path": str(vis_path),
        "state_path": str(state_path),
        "metrics": metrics,
        "build_report": build_report,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("CPBDM independent module")
    p.add_argument("--output_dir", type=str, default="./cpbdm_demo_outputs")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--query_dim", type=int, default=16)
    p.add_argument("--decoder_channels", type=int, default=8)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--demo_steps", type=int, default=16)
    p.add_argument("--kmeans_iters", type=int, default=20)
    p.add_argument("--fit_cache_capacity", type=int, default=24000)
    p.add_argument("--stat_cache_capacity", type=int, default=48000)
    p.add_argument("--max_fit_samples_per_image", type=int, default=384)
    p.add_argument("--max_stat_samples_per_image", type=int, default=768)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main() -> None:
    args = build_parser().parse_args()
    result = run_demo(args)
    print("[CPBDM] 完成：")
    print(json.dumps(json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()