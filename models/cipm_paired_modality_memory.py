"""
cipm_paired_modality_memory.py

CIPM: CT-Indexed Paired Modality Memory
=======================================

独立的多尺度 CT-Key / PET-Value 配对记忆模块。

核心规则
--------
1. 每个尺度维护独立的 K 个槽，四尺度只共享槽数量 K，不共享 Key/Value。
2. CT 决定槽位；同位置 PET 特征跟随 CT 写入相同槽。
3. Full 与训练时模拟 Missing batch 都可收集真实 CT/PET 配对候选。
4. 当前 Missing 前向只能读取上一轮记忆，当前候选仅用于 epoch 结束后更新下一轮。
5. 第一个 epoch 记忆未就绪时，Missing 返回全零 PET proxy，即退回 CT-only。
6. 支持：球面 K-means 初始化、5% 配对离群过滤、槽利用率统计、查询统计、
   空间检索可视化、PCA 聚类可视化和利用率柱状图。

典型集成
--------
    memory = CTIndexedPairedModalityMemory(
        channels=[64, 128, 320, 512],
        num_slots=16,
    )

    # Full batch
    pet_for_fusion = memory(
        ct_features,
        pet_features,
        mode="full",
        collect=True,
        mask=mask,
    )
    fused = memory.fuse(ct_features, pet_for_fusion)

    # 模拟 Missing batch：pet_features 只用于下一轮记忆收集
    pet_proxy = memory(
        ct_features,
        pet_features,
        mode="missing",
        collect=True,
        mask=mask,
    )
    fused = memory.fuse(ct_features, pet_proxy)

    # epoch 结束
    memory.finalize_epoch()
    memory.print_memory_report()

依赖
----
必需：torch
可视化：matplotlib
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_EPS = 1e-8


@dataclass
class ScaleMemoryReport:
    scale_index: int
    channels: int
    num_slots: int
    ready: bool
    active_slots: int
    slot_counts: List[int]
    slot_utilization: List[float]
    slot_tumor_fraction: List[float]
    mean_key_pairwise_cosine: float
    mean_pet_value_norm: float
    query_counts: List[int]
    query_utilization: List[float]
    mean_query_max_similarity: float
    mean_query_entropy: float

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


def _l2_normalize(x: Tensor, dim: int = -1) -> Tensor:
    return F.normalize(x, p=2.0, dim=dim, eps=_EPS)


def _entropy(prob: Tensor, dim: int = -1) -> Tensor:
    p = prob.clamp_min(_EPS)
    return -(p * p.log()).sum(dim=dim)


def _ensure_feature_list(
    features: Sequence[Tensor], expected_scales: int, name: str
) -> List[Tensor]:
    if not isinstance(features, (list, tuple)):
        raise TypeError(f"{name} 必须是 list/tuple。")
    if len(features) != expected_scales:
        raise ValueError(
            f"{name} 尺度数量错误：期望 {expected_scales}，实际 {len(features)}。"
        )
    return list(features)


def _resize_mask(mask: Tensor, size: Tuple[int, int]) -> Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(
            f"mask 应为 [B,H,W] 或 [B,1,H,W]，实际 {tuple(mask.shape)}。"
        )
    return F.interpolate(mask.float(), size=size, mode="area").clamp_(0.0, 1.0)


class PairedPrototypeMemoryScale(nn.Module):
    """单尺度 CT-Key / PET-Value 配对记忆。"""

    def __init__(
        self,
        channels: int,
        num_slots: int = 16,
        *,
        max_tokens_per_batch: int = 4096,
        max_cached_tokens: int = 50000,
        positive_fraction: float = 0.5,
        mask_threshold: float = 0.5,
        outlier_fraction: float = 0.05,
        init_kmeans_iters: int = 20,
        update_kmeans_iters: int = 3,
        seed: int = 2026,
    ) -> None:
        super().__init__()

        if channels <= 0 or num_slots <= 0:
            raise ValueError("channels 和 num_slots 必须大于 0。")
        if max_tokens_per_batch <= 0 or max_cached_tokens <= 0:
            raise ValueError("缓存上限必须大于 0。")
        if not 0.0 <= positive_fraction <= 1.0:
            raise ValueError("positive_fraction 必须位于 [0,1]。")
        if not 0.0 <= outlier_fraction < 1.0:
            raise ValueError("outlier_fraction 必须位于 [0,1)。")

        self.channels = int(channels)
        self.num_slots = int(num_slots)
        self.max_tokens_per_batch = int(max_tokens_per_batch)
        self.max_cached_tokens = int(max_cached_tokens)
        self.positive_fraction = float(positive_fraction)
        self.mask_threshold = float(mask_threshold)
        self.outlier_fraction = float(outlier_fraction)
        self.init_kmeans_iters = int(init_kmeans_iters)
        self.update_kmeans_iters = int(update_kmeans_iters)
        self.seed = int(seed)

        self.register_buffer(
            "ct_keys", torch.zeros(self.num_slots, self.channels, dtype=torch.float32)
        )
        self.register_buffer(
            "pet_values", torch.zeros(self.num_slots, self.channels, dtype=torch.float32)
        )
        self.register_buffer("slot_counts", torch.zeros(self.num_slots, dtype=torch.long))
        self.register_buffer(
            "slot_tumor_fraction",
            torch.full((self.num_slots,), -1.0, dtype=torch.float32),
        )
        self.register_buffer("memory_ready", torch.tensor(False, dtype=torch.bool))

        self.register_buffer(
            "query_slot_counts", torch.zeros(self.num_slots, dtype=torch.long)
        )
        self.register_buffer("query_token_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer(
            "query_max_similarity_sum", torch.tensor(0.0, dtype=torch.float64)
        )
        self.register_buffer("query_entropy_sum", torch.tensor(0.0, dtype=torch.float64))

        self._ct_cache: List[Tensor] = []
        self._pet_cache: List[Tensor] = []
        self._label_cache: List[Tensor] = []
        self._cached_count = 0
        self._cache_compactions = 0

        self._last_vis_ct: Optional[Tensor] = None
        self._last_vis_assignments: Optional[Tensor] = None
        self._last_vis_labels: Optional[Tensor] = None

    @property
    def ready(self) -> bool:
        return bool(self.memory_ready.item())

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, num_slots={self.num_slots}, "
            f"ready={self.ready}, max_cached_tokens={self.max_cached_tokens}"
        )

    @staticmethod
    def _sample_indices(indices: Tensor, num_samples: int) -> Tensor:
        if num_samples <= 0 or indices.numel() == 0:
            return indices[:0]
        if indices.numel() <= num_samples:
            return indices
        perm = torch.randperm(indices.numel(), device=indices.device)[:num_samples]
        return indices[perm]

    def _balanced_sample_indices(
        self,
        mask_flat: Optional[Tensor],
        batch_size: int,
        tokens_per_sample: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        selected_all: List[Tensor] = []
        labels_all: List[Tensor] = []
        quota = max(1, self.max_tokens_per_batch // max(batch_size, 1))

        for b in range(batch_size):
            offset = b * tokens_per_sample
            local_all = torch.arange(tokens_per_sample, device=device)

            if mask_flat is None:
                chosen = self._sample_indices(local_all, quota)
                labels = torch.full(
                    (chosen.numel(),), -1, dtype=torch.long, device=device
                )
            else:
                current_mask = mask_flat[b]
                pos = torch.nonzero(
                    current_mask >= self.mask_threshold, as_tuple=False
                ).flatten()
                neg = torch.nonzero(
                    current_mask < self.mask_threshold, as_tuple=False
                ).flatten()

                target_pos = int(round(quota * self.positive_fraction))
                target_neg = quota - target_pos
                pos_chosen = self._sample_indices(pos, target_pos)
                neg_chosen = self._sample_indices(neg, target_neg)

                # 某一类不足时，从另一类补齐。
                remaining = quota - pos_chosen.numel() - neg_chosen.numel()
                if remaining > 0:
                    if neg.numel() > neg_chosen.numel():
                        extra = self._sample_indices(neg, remaining)
                        neg_chosen = torch.unique(torch.cat([neg_chosen, extra]))
                    elif pos.numel() > pos_chosen.numel():
                        extra = self._sample_indices(pos, remaining)
                        pos_chosen = torch.unique(torch.cat([pos_chosen, extra]))

                chosen = torch.cat([pos_chosen, neg_chosen], dim=0)
                labels = torch.cat(
                    [
                        torch.ones(pos_chosen.numel(), dtype=torch.long, device=device),
                        torch.zeros(neg_chosen.numel(), dtype=torch.long, device=device),
                    ],
                    dim=0,
                )
                if chosen.numel() > 1:
                    order = torch.randperm(chosen.numel(), device=device)
                    chosen, labels = chosen[order], labels[order]

            selected_all.append(chosen + offset)
            labels_all.append(labels)

        return torch.cat(selected_all), torch.cat(labels_all)

    @torch.no_grad()
    def collect(
        self,
        ct_feature: Tensor,
        pet_feature: Tensor,
        mask: Optional[Tensor] = None,
    ) -> int:
        if ct_feature.ndim != 4 or pet_feature.ndim != 4:
            raise ValueError("ct_feature 和 pet_feature 必须是 [B,D,H,W]。")
        if ct_feature.shape != pet_feature.shape:
            raise ValueError(
                "CT/PET 特征形状必须一致："
                f"CT={tuple(ct_feature.shape)}, PET={tuple(pet_feature.shape)}。"
            )

        b, d, h, w = ct_feature.shape
        if d != self.channels:
            raise ValueError(f"当前尺度通道应为 {self.channels}，实际为 {d}。")

        ct_tokens = (
            ct_feature.detach().float().permute(0, 2, 3, 1).reshape(b * h * w, d)
        )
        pet_tokens = (
            pet_feature.detach().float().permute(0, 2, 3, 1).reshape(b * h * w, d)
        )

        mask_flat: Optional[Tensor] = None
        if mask is not None:
            mask_s = _resize_mask(mask.to(ct_feature.device), (h, w))
            mask_flat = mask_s[:, 0].reshape(b, h * w)

        selected, labels = self._balanced_sample_indices(
            mask_flat=mask_flat,
            batch_size=b,
            tokens_per_sample=h * w,
            device=ct_feature.device,
        )

        ct_selected = ct_tokens[selected].cpu()
        pet_selected = pet_tokens[selected].cpu()
        labels = labels.cpu()

        self._ct_cache.append(ct_selected)
        self._pet_cache.append(pet_selected)
        self._label_cache.append(labels)
        self._cached_count += int(ct_selected.shape[0])
        self._compact_cache_if_needed()
        return int(ct_selected.shape[0])

    @torch.no_grad()
    def _compact_cache_if_needed(self) -> None:
        if self._cached_count <= self.max_cached_tokens:
            return

        ct = torch.cat(self._ct_cache, dim=0)
        pet = torch.cat(self._pet_cache, dim=0)
        labels = torch.cat(self._label_cache, dim=0)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self._cache_compactions)
        indices = torch.randperm(ct.shape[0], generator=generator)[
            : self.max_cached_tokens
        ]

        self._ct_cache = [ct[indices].contiguous()]
        self._pet_cache = [pet[indices].contiguous()]
        self._label_cache = [labels[indices].contiguous()]
        self._cached_count = self.max_cached_tokens
        self._cache_compactions += 1

    @torch.no_grad()
    def clear_epoch_cache(self) -> None:
        self._ct_cache.clear()
        self._pet_cache.clear()
        self._label_cache.clear()
        self._cached_count = 0

    def cached_candidate_count(self) -> int:
        return self._cached_count

    def _farthest_point_init(self, x_norm: Tensor) -> Tensor:
        n = x_norm.shape[0]
        if n < self.num_slots:
            raise ValueError(f"候选数 {n} 小于槽数 {self.num_slots}。")

        global_center = _l2_normalize(x_norm.mean(dim=0, keepdim=True))[0]
        first_idx = torch.argmax(x_norm @ global_center)
        keys = [x_norm[first_idx]]
        min_distance = 1.0 - (x_norm @ keys[0])

        for _ in range(1, self.num_slots):
            next_idx = torch.argmax(min_distance)
            new_key = x_norm[next_idx]
            keys.append(new_key)
            min_distance = torch.minimum(min_distance, 1.0 - (x_norm @ new_key))

        return torch.stack(keys, dim=0)

    @staticmethod
    def _assign(x_norm: Tensor, keys_norm: Tensor) -> Tensor:
        return torch.argmax(x_norm @ keys_norm.t(), dim=1)

    def _recompute_keys(
        self, x_norm: Tensor, assignments: Tensor, current_keys: Tensor
    ) -> Tensor:
        new_keys = current_keys.clone()
        nonempty_slots: List[int] = []
        empty_slots: List[int] = []

        for j in range(self.num_slots):
            idx = torch.nonzero(assignments == j, as_tuple=False).flatten()
            if idx.numel() > 0:
                new_keys[j] = _l2_normalize(
                    x_norm[idx].mean(dim=0, keepdim=True)
                )[0]
                nonempty_slots.append(j)
            else:
                empty_slots.append(j)

        if empty_slots:
            if nonempty_slots:
                covered_sim = (x_norm @ new_keys[nonempty_slots].t()).max(dim=1).values
            else:
                covered_sim = torch.full((x_norm.shape[0],), -1.0)

            num_needed = min(len(empty_slots), x_norm.shape[0])
            candidate_indices = torch.topk(
                -covered_sim, k=num_needed, largest=True
            ).indices
            for slot, candidate_idx in zip(empty_slots, candidate_indices):
                new_keys[slot] = x_norm[candidate_idx]

        return _l2_normalize(new_keys)

    def _run_spherical_kmeans(
        self, x_norm: Tensor, init_keys: Tensor, max_iters: int
    ) -> Tuple[Tensor, Tensor]:
        keys = _l2_normalize(init_keys)
        previous_assignments: Optional[Tensor] = None

        for _ in range(max(1, max_iters)):
            assignments = self._assign(x_norm, keys)
            keys = self._recompute_keys(x_norm, assignments, keys)
            if previous_assignments is not None and torch.equal(
                assignments, previous_assignments
            ):
                break
            previous_assignments = assignments

        assignments = self._assign(x_norm, keys)
        return keys, assignments

    @staticmethod
    def _distributed_gather_candidates(
        ct: Tensor, pet: Tensor, labels: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if not (dist.is_available() and dist.is_initialized()):
            return ct, pet, labels

        world_size = dist.get_world_size()
        gathered: List[Optional[Tuple[Tensor, Tensor, Tensor]]] = [
            None for _ in range(world_size)
        ]
        dist.all_gather_object(gathered, (ct.cpu(), pet.cpu(), labels.cpu()))
        valid = [x for x in gathered if x is not None]
        return (
            torch.cat([x[0] for x in valid], dim=0),
            torch.cat([x[1] for x in valid], dim=0),
            torch.cat([x[2] for x in valid], dim=0),
        )

    @torch.no_grad()
    def finalize_epoch(
        self,
        *,
        sync_distributed: bool = False,
        visualization_sample_size: int = 3000,
    ) -> Dict[str, object]:
        if self._cached_count == 0:
            return {
                "updated": False,
                "reason": "当前尺度没有缓存候选。",
                "ready": self.ready,
            }

        ct = torch.cat(self._ct_cache, dim=0).float()
        pet = torch.cat(self._pet_cache, dim=0).float()
        labels = torch.cat(self._label_cache, dim=0).long()

        if sync_distributed:
            ct, pet, labels = self._distributed_gather_candidates(ct, pet, labels)

        if ct.shape[0] < self.num_slots:
            raise RuntimeError(
                f"候选数 {ct.shape[0]} 小于槽数 {self.num_slots}；"
                "请降低 num_slots 或提高采样量。"
            )

        x_norm = _l2_normalize(ct)

        if not self.ready:
            init_keys = self._farthest_point_init(x_norm)
            keys, assignments = self._run_spherical_kmeans(
                x_norm, init_keys, self.init_kmeans_iters
            )
            update_type = "initialization"
        else:
            init_keys = self.ct_keys.detach().cpu().float()
            keys, assignments = self._run_spherical_kmeans(
                x_norm, init_keys, self.update_kmeans_iters
            )
            update_type = "update"

        final_keys = torch.empty_like(keys)
        final_values = torch.empty(self.num_slots, self.channels, dtype=pet.dtype)
        final_counts = torch.zeros(self.num_slots, dtype=torch.long)
        final_tumor_fraction = torch.full(
            (self.num_slots,), -1.0, dtype=torch.float32
        )

        for j in range(self.num_slots):
            idx = torch.nonzero(assignments == j, as_tuple=False).flatten()

            if idx.numel() == 0:
                final_keys[j] = keys[j]
                if self.ready:
                    final_values[j] = self.pet_values[j].detach().cpu()
                else:
                    final_values[j].zero_()
                continue

            slot_ct = x_norm[idx]
            center = _l2_normalize(slot_ct.mean(dim=0, keepdim=True))[0]
            distances = torch.linalg.vector_norm(slot_ct - center, dim=1)
            keep_count = max(
                1,
                int(math.ceil(idx.numel() * (1.0 - self.outlier_fraction))),
            )
            keep_local = torch.argsort(distances)[:keep_count]
            keep_idx = idx[keep_local]

            kept_ct = x_norm[keep_idx]
            kept_pet = pet[keep_idx]
            kept_labels = labels[keep_idx]

            final_keys[j] = _l2_normalize(
                kept_ct.mean(dim=0, keepdim=True)
            )[0]
            final_values[j] = kept_pet.mean(dim=0)
            final_counts[j] = keep_count

            valid_labels = kept_labels[kept_labels >= 0]
            if valid_labels.numel() > 0:
                final_tumor_fraction[j] = valid_labels.float().mean()

        self.ct_keys.copy_(final_keys.to(self.ct_keys.device))
        self.pet_values.copy_(final_values.to(self.pet_values.device))
        self.slot_counts.copy_(final_counts.to(self.slot_counts.device))
        self.slot_tumor_fraction.copy_(
            final_tumor_fraction.to(self.slot_tumor_fraction.device)
        )
        self.memory_ready.fill_(True)

        vis_n = min(int(visualization_sample_size), x_norm.shape[0])
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + 999)
        vis_idx = torch.randperm(x_norm.shape[0], generator=generator)[:vis_n]
        self._last_vis_ct = x_norm[vis_idx].cpu()
        self._last_vis_assignments = assignments[vis_idx].cpu()
        self._last_vis_labels = labels[vis_idx].cpu()

        total_before = int(ct.shape[0])
        total_after = int(final_counts.sum().item())
        self.clear_epoch_cache()

        return {
            "updated": True,
            "update_type": update_type,
            "ready": True,
            "total_candidates_before_filter": total_before,
            "total_candidates_after_filter": total_after,
            "active_slots": int((final_counts > 0).sum().item()),
            "slot_counts": final_counts.tolist(),
            "slot_tumor_fraction": final_tumor_fraction.tolist(),
        }

    def reset_query_stats(self) -> None:
        with torch.no_grad():
            self.query_slot_counts.zero_()
            self.query_token_count.zero_()
            self.query_max_similarity_sum.zero_()
            self.query_entropy_sum.zero_()

    def retrieve(
        self, ct_feature: Tensor, *, return_maps: bool = False
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if ct_feature.ndim != 4:
            raise ValueError("ct_feature 必须是 [B,D,H,W]。")

        b, d, h, w = ct_feature.shape
        if d != self.channels:
            raise ValueError(f"当前尺度通道应为 {self.channels}，实际为 {d}。")

        if not self.ready:
            zeros = torch.zeros_like(ct_feature)
            diagnostics: Dict[str, Tensor] = {
                "ready": torch.tensor(False, device=ct_feature.device)
            }
            if return_maps:
                diagnostics.update(
                    {
                        "slot_index": torch.full(
                            (b, h, w), -1, dtype=torch.long, device=ct_feature.device
                        ),
                        "max_similarity": torch.zeros(b, h, w, device=ct_feature.device),
                        "entropy": torch.zeros(b, h, w, device=ct_feature.device),
                        "compensation_norm": torch.zeros(
                            b, h, w, device=ct_feature.device
                        ),
                    }
                )
            return zeros, diagnostics

        ct_flat = ct_feature.permute(0, 2, 3, 1).reshape(b, h * w, d)
        query = _l2_normalize(ct_flat)
        keys = _l2_normalize(self.ct_keys.float()).to(
            device=ct_feature.device, dtype=query.dtype
        )
        values = self.pet_values.to(
            device=ct_feature.device, dtype=ct_feature.dtype
        )

        similarity = query @ keys.t()
        weights = torch.softmax(similarity, dim=-1)
        pet_flat = weights @ values
        pet_proxy = (
            pet_flat.reshape(b, h, w, d).permute(0, 3, 1, 2).contiguous()
        )

        top_similarity, top_slot = similarity.max(dim=-1)
        entropy_map = _entropy(weights, dim=-1)
        compensation_norm = torch.linalg.vector_norm(pet_flat.float(), dim=-1)

        with torch.no_grad():
            counts = torch.bincount(
                top_slot.reshape(-1), minlength=self.num_slots
            )
            self.query_slot_counts.add_(counts.to(self.query_slot_counts.device))
            token_count = top_slot.numel()
            self.query_token_count.add_(
                torch.tensor(
                    token_count,
                    dtype=self.query_token_count.dtype,
                    device=self.query_token_count.device,
                )
            )
            self.query_max_similarity_sum.add_(
                top_similarity.double().sum().to(
                    self.query_max_similarity_sum.device
                )
            )
            self.query_entropy_sum.add_(
                entropy_map.double().sum().to(self.query_entropy_sum.device)
            )

        diagnostics = {
            "ready": torch.tensor(True, device=ct_feature.device),
            "mean_max_similarity": top_similarity.mean(),
            "mean_entropy": entropy_map.mean(),
            "mean_compensation_norm": compensation_norm.mean(),
        }
        if return_maps:
            diagnostics.update(
                {
                    "slot_index": top_slot.reshape(b, h, w),
                    "max_similarity": top_similarity.reshape(b, h, w),
                    "entropy": entropy_map.reshape(b, h, w),
                    "compensation_norm": compensation_norm.reshape(b, h, w),
                    "weights": weights.reshape(b, h, w, self.num_slots),
                }
            )

        return pet_proxy, diagnostics

    def build_report(self, scale_index: int) -> ScaleMemoryReport:
        counts = self.slot_counts.detach().cpu()
        count_total = int(counts.sum().item())
        utilization = (
            (counts.float() / count_total).tolist()
            if count_total > 0
            else [0.0] * self.num_slots
        )

        query_counts = self.query_slot_counts.detach().cpu()
        query_total = int(self.query_token_count.item())
        if query_total > 0:
            query_utilization = (
                query_counts.float() / float(query_total)
            ).tolist()
            mean_query_max_similarity = float(
                self.query_max_similarity_sum.item() / query_total
            )
            mean_query_entropy = float(self.query_entropy_sum.item() / query_total)
        else:
            query_utilization = [0.0] * self.num_slots
            mean_query_max_similarity = 0.0
            mean_query_entropy = 0.0

        if self.ready and self.num_slots > 1:
            keys = _l2_normalize(self.ct_keys.detach().float().cpu())
            cosine = keys @ keys.t()
            mask = ~torch.eye(self.num_slots, dtype=torch.bool)
            mean_pairwise = float(cosine[mask].mean().item())
        else:
            mean_pairwise = 0.0

        mean_pet_norm = (
            float(
                torch.linalg.vector_norm(
                    self.pet_values.detach().float().cpu(), dim=1
                ).mean().item()
            )
            if self.ready
            else 0.0
        )

        return ScaleMemoryReport(
            scale_index=scale_index,
            channels=self.channels,
            num_slots=self.num_slots,
            ready=self.ready,
            active_slots=int((counts > 0).sum().item()),
            slot_counts=[int(x) for x in counts.tolist()],
            slot_utilization=[float(x) for x in utilization],
            slot_tumor_fraction=[
                float(x) for x in self.slot_tumor_fraction.detach().cpu().tolist()
            ],
            mean_key_pairwise_cosine=mean_pairwise,
            mean_pet_value_norm=mean_pet_norm,
            query_counts=[int(x) for x in query_counts.tolist()],
            query_utilization=[float(x) for x in query_utilization],
            mean_query_max_similarity=mean_query_max_similarity,
            mean_query_entropy=mean_query_entropy,
        )


class CTIndexedPairedModalityMemory(nn.Module):
    """多尺度 CIPM 总模块。"""

    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 320, 512),
        num_slots: int = 16,
        *,
        max_tokens_per_batch: int = 4096,
        max_cached_tokens: int = 50000,
        positive_fraction: float = 0.5,
        mask_threshold: float = 0.5,
        outlier_fraction: float = 0.05,
        init_kmeans_iters: int = 20,
        update_kmeans_iters: int = 3,
        seed: int = 2026,
    ) -> None:
        super().__init__()
        if len(channels) == 0:
            raise ValueError("channels 至少包含一个尺度。")

        self.channels = tuple(int(c) for c in channels)
        self.num_scales = len(self.channels)
        self.num_slots = int(num_slots)
        self.memories = nn.ModuleList(
            [
                PairedPrototypeMemoryScale(
                    channels=c,
                    num_slots=num_slots,
                    max_tokens_per_batch=max_tokens_per_batch,
                    max_cached_tokens=max_cached_tokens,
                    positive_fraction=positive_fraction,
                    mask_threshold=mask_threshold,
                    outlier_fraction=outlier_fraction,
                    init_kmeans_iters=init_kmeans_iters,
                    update_kmeans_iters=update_kmeans_iters,
                    seed=seed + i * 1000,
                )
                for i, c in enumerate(self.channels)
            ]
        )

    @property
    def ready(self) -> bool:
        return all(memory.ready for memory in self.memories)

    @torch.no_grad()
    def collect(
        self,
        ct_features: Sequence[Tensor],
        pet_features: Sequence[Tensor],
        mask: Optional[Tensor] = None,
    ) -> List[int]:
        ct_list = _ensure_feature_list(ct_features, self.num_scales, "ct_features")
        pet_list = _ensure_feature_list(
            pet_features, self.num_scales, "pet_features"
        )
        return [
            memory.collect(ct, pet, mask=mask)
            for memory, ct, pet in zip(self.memories, ct_list, pet_list)
        ]

    def retrieve(
        self,
        ct_features: Sequence[Tensor],
        *,
        return_maps: bool = False,
    ) -> Tuple[List[Tensor], List[Dict[str, Tensor]]]:
        ct_list = _ensure_feature_list(ct_features, self.num_scales, "ct_features")
        proxies: List[Tensor] = []
        diagnostics: List[Dict[str, Tensor]] = []
        for memory, ct in zip(self.memories, ct_list):
            proxy, info = memory.retrieve(ct, return_maps=return_maps)
            proxies.append(proxy)
            diagnostics.append(info)
        return proxies, diagnostics

    def forward(
        self,
        ct_features: Sequence[Tensor],
        pet_features: Optional[Sequence[Tensor]] = None,
        *,
        mode: str,
        collect: bool = False,
        mask: Optional[Tensor] = None,
        return_diagnostics: bool = False,
        return_maps: bool = False,
    ) -> Union[List[Tensor], Tuple[List[Tensor], List[Dict[str, Tensor]]]]:
        mode = mode.lower().strip()
        if mode not in {"full", "missing"}:
            raise ValueError("mode 只能是 'full' 或 'missing'。")

        ct_list = _ensure_feature_list(ct_features, self.num_scales, "ct_features")
        pet_list: Optional[List[Tensor]] = None
        if pet_features is not None:
            pet_list = _ensure_feature_list(
                pet_features, self.num_scales, "pet_features"
            )

        if collect:
            if pet_list is None:
                raise ValueError("collect=True 时必须提供真实 pet_features。")
            self.collect(ct_list, pet_list, mask=mask)

        if mode == "full":
            if pet_list is None:
                raise ValueError("Full 模式必须提供 pet_features。")
            output = pet_list
            diagnostics = [
                {"ready": torch.tensor(memory.ready, device=ct.device)}
                for memory, ct in zip(self.memories, ct_list)
            ]
        else:
            output, diagnostics = self.retrieve(
                ct_list, return_maps=return_maps
            )

        if return_diagnostics:
            return output, diagnostics
        return output

    @staticmethod
    def fuse(
        ct_features: Sequence[Tensor],
        pet_or_proxy_features: Sequence[Tensor],
    ) -> List[Tensor]:
        if len(ct_features) != len(pet_or_proxy_features):
            raise ValueError("CT 与 PET/proxy 的尺度数量必须一致。")
        fused: List[Tensor] = []
        for i, (ct, pet) in enumerate(zip(ct_features, pet_or_proxy_features)):
            if ct.shape != pet.shape:
                raise ValueError(
                    f"尺度 {i + 1} 形状不一致："
                    f"CT={tuple(ct.shape)}, PET/proxy={tuple(pet.shape)}。"
                )
            fused.append(ct + pet)
        return fused

    @torch.no_grad()
    def finalize_epoch(
        self,
        *,
        sync_distributed: bool = False,
        visualization_sample_size: int = 3000,
    ) -> List[Dict[str, object]]:
        reports: List[Dict[str, object]] = []
        for i, memory in enumerate(self.memories):
            report = memory.finalize_epoch(
                sync_distributed=sync_distributed,
                visualization_sample_size=visualization_sample_size,
            )
            report["scale_index"] = i
            report["channels"] = memory.channels
            reports.append(report)
        return reports

    def reset_query_stats(self) -> None:
        for memory in self.memories:
            memory.reset_query_stats()

    def clear_epoch_cache(self) -> None:
        for memory in self.memories:
            memory.clear_epoch_cache()

    def build_memory_reports(self) -> List[ScaleMemoryReport]:
        return [
            memory.build_report(scale_index=i)
            for i, memory in enumerate(self.memories)
        ]

    def print_memory_report(self, *, print_per_slot: bool = True) -> None:
        reports = self.build_memory_reports()
        print("\n" + "=" * 92)
        print("CIPM MEMORY REPORT")
        print("=" * 92)

        for report in reports:
            print(
                f"\n[Scale {report.scale_index + 1}] "
                f"channels={report.channels}, slots={report.num_slots}, "
                f"ready={report.ready}, active={report.active_slots}/{report.num_slots}"
            )
            print(
                "  mean_key_pairwise_cosine="
                f"{report.mean_key_pairwise_cosine:.4f} | "
                "mean_pet_value_norm="
                f"{report.mean_pet_value_norm:.4f}"
            )
            print(
                "  mean_query_max_similarity="
                f"{report.mean_query_max_similarity:.4f} | "
                "mean_query_entropy="
                f"{report.mean_query_entropy:.4f}"
            )

            if print_per_slot:
                print("  " + "-" * 86)
                print(
                    "  slot | build_count | build_util | tumor_frac | "
                    "query_count | query_util"
                )
                print("  " + "-" * 86)
                for j in range(report.num_slots):
                    tumor_fraction = report.slot_tumor_fraction[j]
                    tumor_text = (
                        f"{tumor_fraction:10.4f}"
                        if tumor_fraction >= 0
                        else "       N/A"
                    )
                    print(
                        f"  {j:4d} | "
                        f"{report.slot_counts[j]:11d} | "
                        f"{report.slot_utilization[j] * 100:9.3f}% | "
                        f"{tumor_text} | "
                        f"{report.query_counts[j]:11d} | "
                        f"{report.query_utilization[j] * 100:9.3f}%"
                    )

        print("\n" + "=" * 92 + "\n")

    def visualize_retrieval(
        self,
        ct_features: Sequence[Tensor],
        *,
        mask: Optional[Tensor] = None,
        sample_index: int = 0,
        save_dir: Union[str, os.PathLike] = "./cipm_visualization",
        show: bool = False,
    ) -> List[Path]:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("可视化需要 matplotlib。") from exc

        ct_list = _ensure_feature_list(ct_features, self.num_scales, "ct_features")
        _, diagnostics = self.retrieve(ct_list, return_maps=True)
        save_root = Path(save_dir)
        save_root.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []

        for scale_idx, (ct, info) in enumerate(zip(ct_list, diagnostics)):
            if sample_index >= ct.shape[0]:
                raise IndexError("sample_index 超过 batch size。")

            h, w = ct.shape[-2:]
            num_cols = 5 if mask is not None else 4
            fig, axes = plt.subplots(1, num_cols, figsize=(4.2 * num_cols, 4.0))
            axes = list(axes) if hasattr(axes, "__len__") else [axes]

            images = [
                (
                    info["slot_index"][sample_index].detach().cpu(),
                    "Top-1 slot index",
                    "tab20",
                ),
                (
                    info["max_similarity"][sample_index].detach().cpu(),
                    "Max CT-Key similarity",
                    "viridis",
                ),
                (
                    info["entropy"][sample_index].detach().cpu(),
                    "Retrieval entropy",
                    "magma",
                ),
                (
                    info["compensation_norm"][sample_index].detach().cpu(),
                    "PET proxy norm",
                    "inferno",
                ),
            ]

            for ax, (image, title, cmap) in zip(axes, images):
                im = ax.imshow(image.numpy(), cmap=cmap)
                ax.set_title(title)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if mask is not None:
                mask_s = _resize_mask(
                    mask[sample_index : sample_index + 1].to(ct.device), (h, w)
                )[0, 0].detach().cpu()
                im = axes[-1].imshow(mask_s.numpy(), cmap="gray")
                axes[-1].set_title("GT mask")
                axes[-1].axis("off")
                fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)

            fig.suptitle(f"CIPM retrieval - Scale {scale_idx + 1}")
            fig.tight_layout()
            path = save_root / f"scale_{scale_idx + 1}_retrieval.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            saved.append(path)
            if show:
                plt.show()
            plt.close(fig)

        return saved

    def visualize_cluster_pca(
        self,
        *,
        save_dir: Union[str, os.PathLike] = "./cipm_visualization",
        show: bool = False,
    ) -> List[Path]:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("可视化需要 matplotlib。") from exc

        save_root = Path(save_dir)
        save_root.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []

        for scale_idx, memory in enumerate(self.memories):
            if (
                memory._last_vis_ct is None
                or memory._last_vis_assignments is None
                or not memory.ready
            ):
                continue

            samples = memory._last_vis_ct.float()
            assignments = memory._last_vis_assignments.long()
            labels = memory._last_vis_labels
            keys = _l2_normalize(memory.ct_keys.detach().float().cpu())

            all_points = torch.cat([samples, keys], dim=0)
            centered = all_points - all_points.mean(dim=0, keepdim=True)
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            projected = centered @ vh[:2].t()
            sample_xy = projected[: samples.shape[0]]
            key_xy = projected[samples.shape[0] :]

            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(
                sample_xy[:, 0].numpy(),
                sample_xy[:, 1].numpy(),
                c=assignments.numpy(),
                cmap="tab20",
                s=10,
                alpha=0.55,
            )

            if labels is not None:
                tumor_idx = torch.nonzero(labels == 1, as_tuple=False).flatten()
                if tumor_idx.numel() > 0:
                    ax.scatter(
                        sample_xy[tumor_idx, 0].numpy(),
                        sample_xy[tumor_idx, 1].numpy(),
                        facecolors="none",
                        edgecolors="black",
                        s=24,
                        linewidths=0.6,
                        label="Tumor candidate",
                    )

            ax.scatter(
                key_xy[:, 0].numpy(),
                key_xy[:, 1].numpy(),
                c=torch.arange(memory.num_slots).numpy(),
                cmap="tab20",
                marker="X",
                s=150,
                edgecolors="black",
                linewidths=0.8,
                label="CT Key",
            )
            ax.set_title(f"CIPM CT clustering - Scale {scale_idx + 1}")
            ax.set_xlabel("PCA component 1")
            ax.set_ylabel("PCA component 2")
            ax.grid(alpha=0.2)
            ax.legend(loc="best")
            fig.colorbar(scatter, ax=ax, label="Slot index")
            fig.tight_layout()

            path = save_root / f"scale_{scale_idx + 1}_cluster_pca.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            saved.append(path)
            if show:
                plt.show()
            plt.close(fig)

        return saved

    def visualize_slot_utilization(
        self,
        *,
        save_path: Union[str, os.PathLike] = (
            "./cipm_visualization/slot_utilization.png"
        ),
        show: bool = False,
    ) -> Path:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("可视化需要 matplotlib。") from exc

        reports = self.build_memory_reports()
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(
            self.num_scales,
            1,
            figsize=(12, max(3.2 * self.num_scales, 4.0)),
            squeeze=False,
        )
        x = torch.arange(self.num_slots).numpy()
        width = 0.38

        for scale_idx, report in enumerate(reports):
            ax = axes[scale_idx, 0]
            build = torch.tensor(report.slot_utilization).numpy() * 100.0
            query = torch.tensor(report.query_utilization).numpy() * 100.0
            ax.bar(x - width / 2, build, width=width, label="Build utilization")
            ax.bar(x + width / 2, query, width=width, label="Query utilization")
            ax.set_title(
                f"Scale {scale_idx + 1} slot utilization "
                f"(channels={report.channels})"
            )
            ax.set_xlabel("Slot index")
            ax.set_ylabel("Utilization (%)")
            ax.set_xticks(x)
            ax.grid(axis="y", alpha=0.25)
            ax.legend()

        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return output_path


def _demo() -> None:
    """运行脚本时的最小自检和可视化示例。"""
    torch.manual_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = [16, 32, 64, 128]
    spatial_sizes = [(32, 32), (16, 16), (8, 8), (4, 4)]

    memory = CTIndexedPairedModalityMemory(
        channels=channels,
        num_slots=4,
        max_tokens_per_batch=512,
        max_cached_tokens=5000,
        init_kmeans_iters=15,
        update_kmeans_iters=3,
    ).to(device)

    print(f"[Demo] device={device}")

    for _ in range(5):
        ct_features: List[Tensor] = []
        pet_features: List[Tensor] = []
        for c, (h, w) in zip(channels, spatial_sizes):
            ct = torch.randn(2, c, h, w, device=device)
            pet = 0.65 * ct + 0.25 * torch.sin(ct) + 0.10 * torch.randn_like(ct)
            ct_features.append(ct)
            pet_features.append(pet)

        mask = torch.zeros(2, 1, 128, 128, device=device)
        mask[:, :, 45:75, 50:82] = 1.0
        pet_proxy, diagnostics = memory(
            ct_features,
            pet_features,
            mode="missing",
            collect=True,
            mask=mask,
            return_diagnostics=True,
        )
        assert all(torch.allclose(x, torch.zeros_like(x)) for x in pet_proxy)
        assert all(not bool(item["ready"].item()) for item in diagnostics)

    print("[Demo] Finalizing epoch 1 memory...")
    memory.finalize_epoch()
    memory.print_memory_report()

    ct_features = []
    pet_features = []
    for c, (h, w) in zip(channels, spatial_sizes):
        ct = torch.randn(2, c, h, w, device=device)
        pet = 0.65 * ct + 0.25 * torch.sin(ct) + 0.10 * torch.randn_like(ct)
        ct_features.append(ct)
        pet_features.append(pet)

    mask = torch.zeros(2, 1, 128, 128, device=device)
    mask[:, :, 45:75, 50:82] = 1.0
    pet_proxy, diagnostics = memory(
        ct_features,
        pet_features,
        mode="missing",
        collect=True,
        mask=mask,
        return_diagnostics=True,
        return_maps=True,
    )
    fused = memory.fuse(ct_features, pet_proxy)

    for i, (ct, proxy, fusion) in enumerate(zip(ct_features, pet_proxy, fused)):
        assert ct.shape == proxy.shape == fusion.shape
        print(
            f"[Demo][Scale {i + 1}] shape={tuple(proxy.shape)}, "
            f"mean_proxy_norm="
            f"{diagnostics[i]['mean_compensation_norm'].item():.4f}"
        )

    output_dir = Path("./cipm_demo_outputs")
    memory.visualize_retrieval(ct_features, mask=mask, save_dir=output_dir)
    memory.visualize_cluster_pca(save_dir=output_dir)
    memory.visualize_slot_utilization(
        save_path=output_dir / "slot_utilization.png"
    )
    memory.print_memory_report()
    print(f"[Demo] Visualization saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    _demo()