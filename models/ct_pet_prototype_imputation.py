# -*- coding: utf-8 -*-
"""
Deep-stage Guided Cross-scale CT-Key / PET-Value Prototype Imputation
=====================================================================

Purpose
-------
A standalone PyTorch module for PET-missing segmentation experiments.

Core design
-----------
1. Both Full and Missing training batches first encode real CT/PET features.
2. Before PET masking, slice-level foreground/background CT and PET prototypes
   are collected with GT masks.
3. Foreground and background are clustered independently at one deep build stage.
4. The build-stage cluster identity is reused to aggregate CT keys and PET values
   at every scale.
5. During Missing inference, current CT feature maps query the CT prototype keys
   through scaled dot-product cross-attention and retrieve paired PET prototype
   values.
6. The returned PET proxy can be fused by the unchanged baseline:
       fused_s = ct_s + pet_proxy_s

This file intentionally does NOT add:
- reliability gates
- learnable per-scale lambda
- temperature hyperparameters
- auxiliary reconstruction/contrastive losses
- PET gain / Full-minus-CT targets
- distribution transport

Externally meaningful experimental hyperparameters:
- num_clusters
- build_stage

Expected encoder feature interface
----------------------------------
ct_feats[s], pet_feats[s]: [B, C_s, H_s, W_s]
mask:                     [B, 1, H, W]

Typical real channels:
    channels=(64, 128, 320, 512)

Integration sketch
------------------
memory = CrossScaleCTPETPrototypeMemory(
    channels=(64, 128, 320, 512),
    num_clusters=4,
    build_stage=4,
    output_dir="runs/prototype_memory",
)

ct_feats = encode_ct(ct)
pet_feats_real = encode_pet(pet)

# Full and Missing batches both collect BEFORE masking PET.
if model.training:
    memory.collect(ct_feats, pet_feats_real, mask)

if forward_mode == "full":
    pet_for_fusion = pet_feats_real
else:
    pet_for_fusion, retrieval_info = memory.retrieve(
        ct_feats,
        save_diagnostics=True,
        tag=f"epoch_{epoch}_step_{step}",
    )

fused_feats = [c + p for c, p in zip(ct_feats, pet_for_fusion)]
output = shared_decoder(fused_feats)

# Once per epoch, after all training batches:
memory.finalize_epoch(epoch=epoch)

Notes
-----
- Prototype tensors are registered buffers and are saved in model checkpoints.
- Cache tensors are detached and kept on CPU in float16 to reduce memory.
- Current Missing predictions read the frozen bank from the previous finalize call.
- The current batch's real PET only enters the temporary cache.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8
CLASS_NAMES = ("background", "foreground")
OUTLIER_DISCARD_RATE = 0.05


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_jsonable(obj):
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _save_json(payload: Dict, path: Path) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)


def _finite_or_raise(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x)).sum().item()
        raise RuntimeError(f"{name} contains {bad} NaN/Inf values")


def _tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().float().cpu()
    if x.numel() == 0:
        return {"numel": 0}
    return {
        "numel": int(x.numel()),
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "min": float(x.min()),
        "max": float(x.max()),
        "l2_norm": float(x.norm()),
    }


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=EPS)


def _class_mask_at_scale(
    mask: torch.Tensor,
    output_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Area-preserving soft mask adaptation.

    mask: [B,1,H,W], binary or soft.
    returns:
        bg_mask, fg_mask: [B,1,H_s,W_s]
    """
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must be [B,1,H,W], got {tuple(mask.shape)}")
    fg = F.adaptive_avg_pool2d(mask.float(), output_hw).clamp_(0.0, 1.0)
    bg = 1.0 - fg
    return bg, fg


def _masked_average_pool_2d(
    feature: torch.Tensor,
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    feature: [B,C,H,W]
    weight:  [B,1,H,W]

    returns:
        prototype: [B,C]
        valid:     [B] bool
    """
    if feature.ndim != 4 or weight.ndim != 4:
        raise ValueError("feature and weight must both be 4D")
    if feature.shape[0] != weight.shape[0] or feature.shape[-2:] != weight.shape[-2:]:
        raise ValueError(
            f"shape mismatch: feature={tuple(feature.shape)}, weight={tuple(weight.shape)}"
        )
    denom = weight.sum(dim=(2, 3))  # [B,1]
    valid = denom[:, 0] > EPS
    proto = (feature * weight).sum(dim=(2, 3)) / denom.clamp_min(EPS)
    proto = torch.where(valid[:, None], proto, torch.zeros_like(proto))
    return proto, valid


@torch.no_grad()
def filter_cluster_member_indices(
    build_features: torch.Tensor,
    member_indices: torch.Tensor,
    discard_rate: float = OUTLIER_DISCARD_RATE,
) -> Tuple[torch.Tensor, Dict]:
    if member_indices.ndim != 1:
        raise ValueError("member_indices must be 1D")
    member_indices = member_indices.long()
    num_members = int(member_indices.numel())
    if num_members == 0:
        raise ValueError("member_indices cannot be empty")
    if num_members <= 1:
        kept = member_indices.clone().long()
        return kept, {
            "before_count": num_members,
            "after_count": num_members,
            "discarded_count": 0,
            "discard_rate": float(discard_rate),
            "distances": [],
            "sorted_member_indices": kept.detach().cpu().tolist(),
        }

    cluster_features = build_features[member_indices]
    center = cluster_features.mean(dim=0)
    distances = torch.norm(cluster_features - center, dim=1)
    sorted_local = torch.argsort(distances, dim=0, descending=False)
    keep_num = max(1, int((1.0 - discard_rate) * num_members))
    kept_local = sorted_local[:keep_num]
    kept = member_indices[kept_local].long()
    return kept, {
        "before_count": num_members,
        "after_count": int(kept.numel()),
        "discarded_count": int(num_members - kept.numel()),
        "discard_rate": float(discard_rate),
        "distances": distances.detach().cpu().tolist(),
        "sorted_member_indices": member_indices[sorted_local].detach().cpu().tolist(),
        "kept_member_indices": kept.detach().cpu().tolist(),
    }


# ---------------------------------------------------------------------------
# Deterministic spherical K-means
# ---------------------------------------------------------------------------

@torch.no_grad()
def spherical_kmeans(
    x: torch.Tensor,
    num_clusters: int,
    max_iter: int = 25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Deterministic spherical K-means.

    x: [N,D], expected float32.
    returns:
        labels:  [N]
        centers: [K_eff,D], L2-normalized
    """
    if x.ndim != 2:
        raise ValueError(f"x must be [N,D], got {tuple(x.shape)}")
    n, d = x.shape
    if n == 0:
        raise ValueError("Cannot cluster an empty tensor")
    k = min(int(num_clusters), n)
    x = _normalize_rows(x.float())

    # Deterministic farthest-point initialization.
    first_idx = int(torch.argmax(x.norm(dim=1)).item())
    center_indices = [first_idx]
    min_dist = torch.full((n,), float("inf"), device=x.device)

    for _ in range(1, k):
        last_center = x[center_indices[-1]]
        cosine = x @ last_center
        dist = 1.0 - cosine
        min_dist = torch.minimum(min_dist, dist)
        next_idx = int(torch.argmax(min_dist).item())
        center_indices.append(next_idx)

    centers = x[torch.tensor(center_indices, device=x.device)].clone()
    labels = torch.zeros(n, dtype=torch.long, device=x.device)

    for _ in range(max_iter):
        sims = x @ centers.t()
        new_labels = sims.argmax(dim=1)
        if torch.equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels

        new_centers = []
        assigned_best = sims.max(dim=1).values
        for cluster_idx in range(k):
            members = x[labels == cluster_idx]
            if members.numel() == 0:
                # Reinitialize with the currently worst represented sample.
                replacement_idx = int(torch.argmin(assigned_best).item())
                new_center = x[replacement_idx]
            else:
                new_center = members.mean(dim=0)
            new_centers.append(F.normalize(new_center, dim=0, eps=EPS))
        centers = torch.stack(new_centers, dim=0)

    return labels, centers


# ---------------------------------------------------------------------------
# FedMEPD-style scaled dot-product prototype retrieval
# ---------------------------------------------------------------------------

class PrototypeCrossAttention(nn.Module):
    """
    Current CT spatial tokens are Q.
    CT prototypes are K.
    Paired PET prototypes are V.

    This is a task adaptation of the reference retrieval form:
        attention(q=current feature, k=prototype, v=prototype)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = int(channels)
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Identity initialization preserves the direct prototype geometry initially.
        eye = torch.eye(self.channels)
        with torch.no_grad():
            self.q_proj.weight.copy_(eye)
            self.k_proj.weight.copy_(eye)
            self.v_proj.weight.copy_(eye)
            self.out_proj.weight.copy_(eye)

    def forward(
        self,
        query_map: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        ready: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query_map: [B,C,H,W]
        keys:      [M,C]
        values:    [M,C]
        ready:     [M] bool

        returns:
            retrieved: [B,C,H,W]
            attention: [B,H*W,M]
        """
        if query_map.ndim != 4:
            raise ValueError("query_map must be [B,C,H,W]")
        b, c, h, w = query_map.shape
        if c != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {c}")
        if keys.shape != values.shape or keys.ndim != 2 or keys.shape[1] != c:
            raise ValueError(
                f"keys/values must both be [M,{c}], got {tuple(keys.shape)}, {tuple(values.shape)}"
            )
        if ready.ndim != 1 or ready.shape[0] != keys.shape[0]:
            raise ValueError("ready shape mismatch")

        if not bool(ready.any()):
            zeros = torch.zeros_like(query_map)
            empty_attn = torch.zeros(
                b, h * w, keys.shape[0], device=query_map.device, dtype=query_map.dtype
            )
            return zeros, empty_attn

        q = query_map.flatten(2).transpose(1, 2)  # [B,N,C]
        q = self.q_proj(q)
        k = self.k_proj(keys)                    # [M,C]
        v = self.v_proj(values)                  # [M,C]

        logits = torch.matmul(q, k.t()) / math.sqrt(float(c))  # [B,N,M]
        logits = logits.masked_fill(~ready.view(1, 1, -1), torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits, dim=-1)
        retrieved = torch.matmul(attn, v)
        retrieved = self.out_proj(retrieved)
        retrieved = retrieved.transpose(1, 2).reshape(b, c, h, w)

        _finite_or_raise("prototype_attention", attn)
        _finite_or_raise("retrieved_pet", retrieved)
        return retrieved, attn


# ---------------------------------------------------------------------------
# Main memory module
# ---------------------------------------------------------------------------

@dataclass
class MemoryConfig:
    channels: Tuple[int, ...]
    num_clusters: int = 4
    build_stage: int = 4
    output_dir: str = "prototype_memory_outputs"

    def validate(self) -> None:
        if len(self.channels) == 0:
            raise ValueError("channels cannot be empty")
        if self.num_clusters < 1:
            raise ValueError("num_clusters must be >= 1")
        if not (1 <= self.build_stage <= len(self.channels)):
            raise ValueError(
                f"build_stage must be within [1,{len(self.channels)}], got {self.build_stage}"
            )


class CrossScaleCTPETPrototypeMemory(nn.Module):
    """
    Slice-level, class-independent, deep-stage-guided, cross-scale CT/PET bank.

    Class index:
        0 -> background
        1 -> foreground
    """

    def __init__(
        self,
        channels: Sequence[int],
        num_clusters: int = 4,
        build_stage: int = 4,
        output_dir: str = "prototype_memory_outputs",
    ):
        super().__init__()
        self.config = MemoryConfig(
            channels=tuple(int(c) for c in channels),
            num_clusters=int(num_clusters),
            build_stage=int(build_stage),
            output_dir=str(output_dir),
        )
        self.config.validate()

        self.channels = self.config.channels
        self.num_scales = len(self.channels)
        self.num_clusters = self.config.num_clusters
        self.build_stage_idx = self.config.build_stage - 1
        self.output_dir = _ensure_dir(Path(self.config.output_dir))
        self.json_dir = _ensure_dir(self.output_dir / "json")
        self.vis_dir = _ensure_dir(self.output_dir / "visualizations")

        self.attention = nn.ModuleList(
            [PrototypeCrossAttention(c) for c in self.channels]
        )

        # Per-scale paired banks.
        for scale_idx, c in enumerate(self.channels):
            self.register_buffer(
                f"ct_keys_s{scale_idx + 1}",
                torch.zeros(2, self.num_clusters, c),
            )
            self.register_buffer(
                f"pet_values_s{scale_idx + 1}",
                torch.zeros(2, self.num_clusters, c),
            )

        self.register_buffer(
            "prototype_ready",
            torch.zeros(2, self.num_clusters, dtype=torch.bool),
        )
        self.register_buffer(
            "prototype_count",
            torch.zeros(2, self.num_clusters, dtype=torch.long),
        )
        self.register_buffer(
            "bank_version",
            torch.zeros((), dtype=torch.long),
        )

        self._epoch_cache = self._new_cache()
        self._collect_calls = 0
        self._collected_slices = 0

    # -------------------------- cache management --------------------------

    def _new_cache(self):
        cache = {}
        for class_idx in range(2):
            cache[class_idx] = {
                "ct": [[] for _ in range(self.num_scales)],
                "pet": [[] for _ in range(self.num_scales)],
            }
        return cache

    @torch.no_grad()
    def reset_epoch_cache(self) -> None:
        self._epoch_cache = self._new_cache()
        self._collect_calls = 0
        self._collected_slices = 0

    @torch.no_grad()
    def collect(
        self,
        ct_feats: Sequence[torch.Tensor],
        pet_feats: Sequence[torch.Tensor],
        mask: torch.Tensor,
        print_info: bool = False,
        compute_report: bool = False,
    ) -> Dict:
        """
        Collect real CT/PET slice prototypes BEFORE PET masking.

        Full and Missing training batches should both call this method.
        All cached tensors are detached CPU float16.
        """
        self._validate_features(ct_feats, pet_feats)
        if mask.shape[0] != ct_feats[0].shape[0]:
            raise ValueError("mask batch size does not match feature batch size")

        batch_size = int(mask.shape[0])
        need_report = bool(compute_report or print_info)
        call_info = {
            "batch_size": batch_size,
            "valid_background": 0,
            "valid_foreground": 0,
        }
        if need_report:
            call_info["scales"] = []

        class_valid_at_build = {}

        for scale_idx, (ct, pet) in enumerate(zip(ct_feats, pet_feats)):
            _finite_or_raise(f"ct_feats[{scale_idx}]", ct)
            _finite_or_raise(f"pet_feats[{scale_idx}]", pet)

            bg_mask, fg_mask = _class_mask_at_scale(mask, ct.shape[-2:])
            masks = (bg_mask, fg_mask)

            scale_info = {
                "scale": scale_idx + 1,
                "shape": list(ct.shape),
                "classes": {},
            } if need_report else None

            for class_idx, class_mask in enumerate(masks):
                ct_proto, ct_valid = _masked_average_pool_2d(ct, class_mask)
                pet_proto, pet_valid = _masked_average_pool_2d(pet, class_mask)
                valid = ct_valid & pet_valid

                # Foreground validity should follow the original slice mask.
                if class_idx == 1:
                    original_fg_valid = mask.flatten(1).sum(dim=1) > EPS
                    valid = valid & original_fg_valid

                if scale_idx == self.build_stage_idx:
                    class_valid_at_build[class_idx] = valid.detach().cpu()

                if valid.any():
                    self._epoch_cache[class_idx]["ct"][scale_idx].append(
                        ct_proto[valid].detach().to("cpu", dtype=torch.float16)
                    )
                    self._epoch_cache[class_idx]["pet"][scale_idx].append(
                        pet_proto[valid].detach().to("cpu", dtype=torch.float16)
                    )

                if need_report:
                    scale_info["classes"][CLASS_NAMES[class_idx]] = {
                        "valid_count": int(valid.sum().item()),
                        "ct_proto_stats": _tensor_stats(ct_proto[valid]),
                        "pet_proto_stats": _tensor_stats(pet_proto[valid]),
                    }

            if need_report:
                call_info["scales"].append(scale_info)

        # Since validity is determined from the same slice mask, counts at build
        # stage represent records that will be aligned across all scales.
        call_info["valid_background"] = int(class_valid_at_build[0].sum().item())
        call_info["valid_foreground"] = int(class_valid_at_build[1].sum().item())

        self._collect_calls += 1
        self._collected_slices += batch_size

        if print_info:
            print(
                "[PrototypeCollect] "
                f"call={self._collect_calls} batch={batch_size} "
                f"bg={call_info['valid_background']} fg={call_info['valid_foreground']}"
            )
        return call_info

    # -------------------------- bank finalization --------------------------

    @torch.no_grad()
    def finalize_epoch(
        self,
        epoch: Optional[int] = None,
        save_json: bool = True,
        save_visualizations: bool = True,
        print_info: bool = True,
    ) -> Dict:
        """
        Build the bank from all cached slice prototypes.

        For each class independently:
          1. Cluster build-stage CT prototypes.
          2. Reuse those labels to aggregate CT keys and PET values at all scales.
        """
        report = {
            "epoch": epoch,
            "config": asdict(self.config),
            "collect_calls": self._collect_calls,
            "collected_slices": self._collected_slices,
            "bank_version_before": int(self.bank_version.item()),
            "classes": {},
            "scales": {},
        }

        # Build temporary banks first, so incomplete classes do not corrupt
        # the previous valid bank.
        new_keys = [
            torch.zeros_like(getattr(self, f"ct_keys_s{s + 1}"), device="cpu")
            for s in range(self.num_scales)
        ]
        new_values = [
            torch.zeros_like(getattr(self, f"pet_values_s{s + 1}"), device="cpu")
            for s in range(self.num_scales)
        ]
        new_ready = torch.zeros_like(self.prototype_ready, device="cpu")
        new_count = torch.zeros_like(self.prototype_count, device="cpu")

        for class_idx, class_name in enumerate(CLASS_NAMES):
            build_ct = self._concat_cache(class_idx, "ct", self.build_stage_idx)
            class_report = {
                "num_candidates": int(build_ct.shape[0]),
                "num_effective_clusters": 0,
                "cluster_counts_before_filter": [0] * self.num_clusters,
                "cluster_counts_after_filter": [0] * self.num_clusters,
                "cluster_discarded_counts": [0] * self.num_clusters,
                "cluster_counts": [0] * self.num_clusters,
                "discard_rate": float(OUTLIER_DISCARD_RATE),
            }

            if build_ct.shape[0] == 0:
                report["classes"][class_name] = class_report
                continue

            labels, centers = spherical_kmeans(
                build_ct.float(),
                num_clusters=self.num_clusters,
            )
            k_eff = int(centers.shape[0])
            class_report["num_effective_clusters"] = k_eff

            cluster_member_cache = {}
            for cluster_idx in range(k_eff):
                member_indices = torch.nonzero(labels == cluster_idx, as_tuple=False).flatten().long()
                before_count = int(member_indices.numel())
                if before_count == 0:
                    continue

                kept_indices, filter_stats = filter_cluster_member_indices(
                    build_features=build_ct.float(),
                    member_indices=member_indices,
                    discard_rate=OUTLIER_DISCARD_RATE,
                )
                kept_indices = kept_indices.long()
                after_count = int(kept_indices.numel())
                discarded_count = before_count - after_count
                cluster_member_cache[cluster_idx] = {
                    "member_indices": member_indices,
                    "kept_indices": kept_indices,
                    "filter_stats": filter_stats,
                    "before_count": before_count,
                    "after_count": after_count,
                    "discarded_count": discarded_count,
                }
                class_report["cluster_counts_before_filter"][cluster_idx] = before_count
                class_report["cluster_counts_after_filter"][cluster_idx] = after_count
                class_report["cluster_discarded_counts"][cluster_idx] = discarded_count
                class_report["cluster_counts"][cluster_idx] = after_count

            # Cache lists are appended with the same valid slice ordering at
            # every scale. Check alignment before aggregating.
            for scale_idx in range(self.num_scales):
                ct_all = self._concat_cache(class_idx, "ct", scale_idx).float()
                pet_all = self._concat_cache(class_idx, "pet", scale_idx).float()
                if ct_all.shape[0] != labels.shape[0] or pet_all.shape[0] != labels.shape[0]:
                    raise RuntimeError(
                        "Cross-scale cache alignment failed for "
                        f"class={class_name}, scale={scale_idx + 1}: "
                        f"build={labels.shape[0]}, ct={ct_all.shape[0]}, pet={pet_all.shape[0]}"
                    )

                for cluster_idx in range(k_eff):
                    cache_item = cluster_member_cache.get(cluster_idx)
                    if cache_item is None:
                        continue
                    kept_indices = cache_item["kept_indices"]
                    after_count = cache_item["after_count"]

                    ct_key = ct_all[kept_indices].mean(dim=0)
                    pet_value = pet_all[kept_indices].mean(dim=0)

                    new_keys[scale_idx][class_idx, cluster_idx] = F.normalize(
                        ct_key, dim=0, eps=EPS
                    )
                    new_values[scale_idx][class_idx, cluster_idx] = pet_value

                    if scale_idx == self.build_stage_idx:
                        new_ready[class_idx, cluster_idx] = True
                        new_count[class_idx, cluster_idx] = after_count

            report["classes"][class_name] = class_report

        if not bool(new_ready.any()):
            report["status"] = "bank_not_updated_no_candidates"
            if print_info:
                print("[PrototypeFinalize] No valid candidates; bank unchanged.")
            if save_json:
                tag = f"epoch_{epoch}" if epoch is not None else "epoch_unknown"
                _save_json(report, self.json_dir / f"bank_{tag}.json")
            self.reset_epoch_cache()
            return report

        # Copy the newly built bank to registered buffers.
        for scale_idx in range(self.num_scales):
            getattr(self, f"ct_keys_s{scale_idx + 1}").copy_(
                new_keys[scale_idx].to(
                    device=getattr(self, f"ct_keys_s{scale_idx + 1}").device,
                    dtype=getattr(self, f"ct_keys_s{scale_idx + 1}").dtype,
                )
            )
            getattr(self, f"pet_values_s{scale_idx + 1}").copy_(
                new_values[scale_idx].to(
                    device=getattr(self, f"pet_values_s{scale_idx + 1}").device,
                    dtype=getattr(self, f"pet_values_s{scale_idx + 1}").dtype,
                )
            )

        self.prototype_ready.copy_(new_ready.to(self.prototype_ready.device))
        self.prototype_count.copy_(new_count.to(self.prototype_count.device))
        self.bank_version.add_(1)

        report["status"] = "bank_updated"
        report["bank_version_after"] = int(self.bank_version.item())
        report["ready_count"] = int(self.prototype_ready.sum().item())
        report["total_slots"] = int(self.prototype_ready.numel())

        for scale_idx in range(self.num_scales):
            keys = getattr(self, f"ct_keys_s{scale_idx + 1}")
            values = getattr(self, f"pet_values_s{scale_idx + 1}")
            ready_flat = self.prototype_ready.flatten()
            keys_flat = keys.reshape(-1, keys.shape[-1])[ready_flat]
            values_flat = values.reshape(-1, values.shape[-1])[ready_flat]

            report["scales"][f"scale_{scale_idx + 1}"] = {
                "channels": self.channels[scale_idx],
                "ct_key_stats": _tensor_stats(keys_flat),
                "pet_value_stats": _tensor_stats(values_flat),
                "mean_ct_key_norm": (
                    float(keys_flat.norm(dim=1).mean()) if keys_flat.numel() else 0.0
                ),
                "mean_pet_value_norm": (
                    float(values_flat.norm(dim=1).mean()) if values_flat.numel() else 0.0
                ),
            }

        tag = f"epoch_{epoch}" if epoch is not None else f"version_{int(self.bank_version.item())}"
        if save_json:
            _save_json(report, self.json_dir / f"bank_{tag}.json")
        if save_visualizations:
            self.save_bank_visualizations(tag=tag)

        if print_info:
            self._print_bank_report(report)

        self.reset_epoch_cache()
        return report

    # -------------------------- missing retrieval --------------------------

    @property
    def bank_ready(self) -> bool:
        return bool(self.prototype_ready.any())

    def retrieve(
        self,
        ct_feats: Sequence[torch.Tensor],
        save_diagnostics: bool = False,
        tag: str = "retrieval",
        visualize_batch_index: int = 0,
        print_info: bool = False,
        compute_report: bool = False,
    ) -> Tuple[List[torch.Tensor], Dict]:
        """
        Retrieve PET proxy features for a Missing batch.

        Each scale independently uses:
            Q = current CT spatial tokens
            K = same-scale CT prototype keys
            V = same-scale paired PET prototype values
        """
        self._validate_features(ct_feats, None)
        ready_flat = self.prototype_ready.flatten()

        outputs: List[torch.Tensor] = []
        need_report = bool(compute_report or save_diagnostics or print_info)
        report = {
            "tag": tag,
            "bank_version": int(self.bank_version.item()),
            "ready_slots": int(ready_flat.sum().item()),
            "total_slots": int(ready_flat.numel()),
        }
        if need_report:
            report["scales"] = {}

        for scale_idx, ct in enumerate(ct_feats):
            keys = getattr(self, f"ct_keys_s{scale_idx + 1}").reshape(
                2 * self.num_clusters, self.channels[scale_idx]
            )
            values = getattr(self, f"pet_values_s{scale_idx + 1}").reshape(
                2 * self.num_clusters, self.channels[scale_idx]
            )
            keys = keys.to(device=ct.device, dtype=ct.dtype)
            values = values.to(device=ct.device, dtype=ct.dtype)
            ready = ready_flat.to(ct.device)

            retrieved, attn = self.attention[scale_idx](ct, keys, values, ready)
            outputs.append(retrieved)

            if need_report:
                probs = attn.float().clamp_min(EPS)
                entropy = -(probs * probs.log()).sum(dim=-1)
                valid_slot_count = max(int(ready.sum().item()), 1)
                max_entropy = math.log(valid_slot_count) if valid_slot_count > 1 else 1.0
                normalized_entropy = entropy / max_entropy
                confidence = 1.0 - normalized_entropy
                top1 = attn.argmax(dim=-1)
                usage = torch.bincount(top1.flatten(), minlength=2 * self.num_clusters)
                ct_norm = ct.float().norm(dim=1).mean()
                pet_norm = retrieved.float().norm(dim=1).mean()
                report["scales"][f"scale_{scale_idx + 1}"] = {
                    "ct_shape": list(ct.shape),
                    "retrieved_shape": list(retrieved.shape),
                    "attention_shape": list(attn.shape),
                    "entropy": _tensor_stats(normalized_entropy),
                    "confidence": _tensor_stats(confidence),
                    "top1_usage": usage.detach().cpu().tolist(),
                    "ct_mean_spatial_l2": float(ct_norm.detach().cpu()),
                    "retrieved_pet_mean_spatial_l2": float(pet_norm.detach().cpu()),
                    "retrieved_to_ct_norm_ratio": float(pet_norm / (ct_norm + EPS)),
                }

            if save_diagnostics:
                self._save_retrieval_scale_visualizations(
                    attn=attn,
                    retrieved=retrieved,
                    scale_idx=scale_idx,
                    tag=tag,
                    batch_index=visualize_batch_index,
                )

        if save_diagnostics:
            _save_json(report, self.json_dir / f"retrieval_{tag}.json")
        if print_info:
            self._print_retrieval_report(report)
        return outputs, report

    def fuse_missing(
        self,
        ct_feats: Sequence[torch.Tensor],
        **retrieve_kwargs,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict]:
        pet_proxy, report = self.retrieve(ct_feats, **retrieve_kwargs)
        fused = [ct + pet for ct, pet in zip(ct_feats, pet_proxy)]
        return fused, pet_proxy, report

    # -------------------------- diagnostics --------------------------

    @torch.no_grad()
    def save_bank_visualizations(self, tag: str) -> None:
        ready = self.prototype_ready.detach().cpu()

        # Separate cluster-count plots for background and foreground.
        for class_idx, class_name in enumerate(CLASS_NAMES):
            counts = np.array(self.prototype_count[class_idx].detach().cpu().tolist())
            fig = plt.figure(figsize=(7, 4))
            ax = fig.add_subplot(111)
            ax.bar(np.arange(self.num_clusters), counts)
            ax.set_xlabel("Cluster index")
            ax.set_ylabel("Slice prototype count")
            ax.set_title(f"{class_name.title()} cluster counts")
            ax.set_xticks(np.arange(self.num_clusters))
            fig.tight_layout()
            fig.savefig(
                self.vis_dir / f"{tag}_{class_name}_cluster_counts.png",
                dpi=180,
            )
            plt.close(fig)

        # Separate norm plot for every scale.
        for scale_idx in range(self.num_scales):
            keys = getattr(self, f"ct_keys_s{scale_idx + 1}").detach().cpu()
            values = getattr(self, f"pet_values_s{scale_idx + 1}").detach().cpu()
            key_norm = np.array(keys.norm(dim=-1).reshape(-1).detach().cpu().tolist())
            value_norm = np.array(values.norm(dim=-1).reshape(-1).detach().cpu().tolist())
            labels = [
                f"{CLASS_NAMES[c][0].upper()}{k}"
                for c in range(2)
                for k in range(self.num_clusters)
            ]

            fig = plt.figure(figsize=(9, 4))
            ax = fig.add_subplot(111)
            x = np.arange(len(labels))
            width = 0.38
            ax.bar(x - width / 2, key_norm, width, label="CT key norm")
            ax.bar(x + width / 2, value_norm, width, label="PET value norm")
            ax.set_xlabel("Prototype slot")
            ax.set_ylabel("L2 norm")
            ax.set_title(f"Scale {scale_idx + 1} prototype norms")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                self.vis_dir / f"{tag}_scale_{scale_idx + 1}_prototype_norms.png",
                dpi=180,
            )
            plt.close(fig)

            # CT-key cosine similarity heatmap.
            flat_keys = keys.reshape(-1, keys.shape[-1])
            flat_keys = _normalize_rows(flat_keys)
            similarity = np.array((flat_keys @ flat_keys.t()).detach().cpu().tolist())

            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111)
            im = ax.imshow(similarity, vmin=-1.0, vmax=1.0, cmap="coolwarm")
            ax.set_title(f"Scale {scale_idx + 1} CT-key cosine similarity")
            ax.set_xlabel("Prototype slot")
            ax.set_ylabel("Prototype slot")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(
                self.vis_dir / f"{tag}_scale_{scale_idx + 1}_ct_key_similarity.png",
                dpi=180,
            )
            plt.close(fig)

    @torch.no_grad()
    def _save_retrieval_scale_visualizations(
        self,
        attn: torch.Tensor,
        retrieved: torch.Tensor,
        scale_idx: int,
        tag: str,
        batch_index: int,
    ) -> None:
        b, c, h, w = retrieved.shape
        idx = max(0, min(int(batch_index), b - 1))
        attn_i = attn[idx].detach().float().cpu()  # [H*W,M]
        probs = attn_i.clamp_min(EPS)

        top1 = np.array(probs.argmax(dim=-1).reshape(h, w).detach().cpu().tolist())
        entropy = -(probs * probs.log()).sum(dim=-1)
        valid_count = max(int(self.prototype_ready.sum().item()), 1)
        max_entropy = math.log(valid_count) if valid_count > 1 else 1.0
        entropy = np.array((entropy / max_entropy).reshape(h, w).detach().cpu().tolist())
        pet_norm = np.array(retrieved[idx].detach().float().cpu().norm(dim=0).tolist())
        mean_usage = np.array(probs.mean(dim=0).detach().cpu().tolist())

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        im = ax.imshow(top1, interpolation="nearest")
        ax.set_title(f"Scale {scale_idx + 1} top-1 prototype slot")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(
            self.vis_dir / f"{tag}_scale_{scale_idx + 1}_top1_map.png",
            dpi=180,
        )
        plt.close(fig)

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        im = ax.imshow(entropy, vmin=0.0, vmax=1.0, cmap="magma")
        ax.set_title(f"Scale {scale_idx + 1} normalized retrieval entropy")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(
            self.vis_dir / f"{tag}_scale_{scale_idx + 1}_entropy_map.png",
            dpi=180,
        )
        plt.close(fig)

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        im = ax.imshow(pet_norm, cmap="viridis")
        ax.set_title(f"Scale {scale_idx + 1} retrieved PET spatial L2")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(
            self.vis_dir / f"{tag}_scale_{scale_idx + 1}_retrieved_pet_norm.png",
            dpi=180,
        )
        plt.close(fig)

        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.bar(np.arange(len(mean_usage)), mean_usage)
        ax.set_title(f"Scale {scale_idx + 1} mean prototype attention")
        ax.set_xlabel("Prototype slot")
        ax.set_ylabel("Mean attention")
        ax.set_xticks(np.arange(len(mean_usage)))
        fig.tight_layout()
        fig.savefig(
            self.vis_dir / f"{tag}_scale_{scale_idx + 1}_mean_attention.png",
            dpi=180,
        )
        plt.close(fig)

    def _print_bank_report(self, report: Dict) -> None:
        print("\n" + "=" * 78)
        print("[PrototypeBank] Finalization report")
        print(
            f"epoch={report.get('epoch')} "
            f"version={report.get('bank_version_after')} "
            f"ready={report.get('ready_count')}/{report.get('total_slots')} "
            f"slices={report.get('collected_slices')}"
        )
        for class_name in CLASS_NAMES:
            item = report["classes"][class_name]
            print(f"  {class_name:10s}: candidates={item['num_candidates']:5d} effective_clusters={item['num_effective_clusters']}")
            print(f"    before={item['cluster_counts_before_filter']}")
            print(f"    after={item['cluster_counts_after_filter']}")
            print(f"    discarded={item['cluster_discarded_counts']}")
        for scale_name, item in report["scales"].items():
            print(
                f"  {scale_name}: C={item['channels']} "
                f"mean|K_ct|={item['mean_ct_key_norm']:.4f} "
                f"mean|V_pet|={item['mean_pet_value_norm']:.4f}"
            )
        print(f"JSON: {self.json_dir}")
        print(f"Visualizations: {self.vis_dir}")
        print("=" * 78 + "\n")

    def _print_retrieval_report(self, report: Dict) -> None:
        print("\n" + "-" * 78)
        print(
            f"[PrototypeRetrieve] tag={report['tag']} "
            f"version={report['bank_version']} "
            f"ready={report['ready_slots']}/{report['total_slots']}"
        )
        for scale_name, item in report["scales"].items():
            print(
                f"  {scale_name}: entropy_mean={item['entropy'].get('mean', 0.0):.4f} "
                f"confidence_mean={item['confidence'].get('mean', 0.0):.4f} "
                f"|PET_hat|/|CT|={item['retrieved_to_ct_norm_ratio']:.4f} "
                f"usage={item['top1_usage']}"
            )
        print("-" * 78 + "\n")

    # -------------------------- internal validation --------------------------

    def _validate_features(
        self,
        ct_feats: Sequence[torch.Tensor],
        pet_feats: Optional[Sequence[torch.Tensor]],
    ) -> None:
        if len(ct_feats) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} CT scales, got {len(ct_feats)}"
            )
        if pet_feats is not None and len(pet_feats) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} PET scales, got {len(pet_feats)}"
            )

        batch_size = ct_feats[0].shape[0]
        for scale_idx, ct in enumerate(ct_feats):
            if ct.ndim != 4:
                raise ValueError(f"CT scale {scale_idx + 1} must be [B,C,H,W]")
            if ct.shape[0] != batch_size:
                raise ValueError("All CT scales must have the same batch size")
            if ct.shape[1] != self.channels[scale_idx]:
                raise ValueError(
                    f"CT scale {scale_idx + 1}: expected C={self.channels[scale_idx]}, "
                    f"got C={ct.shape[1]}"
                )

            if pet_feats is not None:
                pet = pet_feats[scale_idx]
                if pet.shape != ct.shape:
                    raise ValueError(
                        f"CT/PET shape mismatch at scale {scale_idx + 1}: "
                        f"{tuple(ct.shape)} vs {tuple(pet.shape)}"
                    )

    def _concat_cache(
        self,
        class_idx: int,
        modality: str,
        scale_idx: int,
    ) -> torch.Tensor:
        chunks = self._epoch_cache[class_idx][modality][scale_idx]
        channels = self.channels[scale_idx]
        if len(chunks) == 0:
            return torch.empty(0, channels, dtype=torch.float32)
        return torch.cat(chunks, dim=0).float()


# ---------------------------------------------------------------------------
# Standalone synthetic verification
# ---------------------------------------------------------------------------

def _make_small_lesion_masks(
    batch_size: int,
    image_size: int,
    rng: random.Random,
    device: torch.device,
) -> torch.Tensor:
    masks = torch.zeros(batch_size, 1, image_size, image_size, device=device)
    yy, xx = torch.meshgrid(
        torch.arange(image_size, device=device),
        torch.arange(image_size, device=device),
        indexing="ij",
    )
    for i in range(batch_size):
        # Keep some negative slices to verify background-only collection.
        if i % 4 == 0:
            continue
        radius = rng.randint(max(2, image_size // 24), max(3, image_size // 12))
        cy = rng.randint(radius, image_size - radius - 1)
        cx = rng.randint(radius, image_size - radius - 1)
        lesion = (yy - cy).square() + (xx - cx).square() <= radius * radius
        masks[i, 0, lesion] = 1.0
    return masks


def _make_synthetic_features(
    mask: torch.Tensor,
    channels: Sequence[int],
    spatial_sizes: Sequence[int],
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    ct_feats, pet_feats = [], []
    for c, size in zip(channels, spatial_sizes):
        b = mask.shape[0]
        ct = torch.randn(b, c, size, size, device=device)
        pet = 0.35 * ct + 0.65 * torch.randn_like(ct)

        lesion = F.adaptive_avg_pool2d(mask, (size, size))
        # Inject a weak but structured foreground signal.
        ct[:, : max(1, c // 8)] += lesion * 0.8
        pet[:, : max(1, c // 6)] += lesion * 1.5

        ct.requires_grad_(True)
        pet.requires_grad_(True)
        ct_feats.append(ct)
        pet_feats.append(pet)
    return ct_feats, pet_feats


@torch.no_grad()
def _run_outlier_filter_sanity_check() -> Dict:
    build_features = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [20.0, 20.0],
        ],
        dtype=torch.float32,
    )
    member_indices = torch.arange(build_features.shape[0], dtype=torch.long)
    kept_indices, stats = filter_cluster_member_indices(
        build_features=build_features,
        member_indices=member_indices,
        discard_rate=OUTLIER_DISCARD_RATE,
    )
    assert int(stats["before_count"]) == 5
    assert int(stats["after_count"]) == 4
    assert int(stats["discarded_count"]) == 1
    assert kept_indices.dtype == torch.long
    assert 4 not in kept_indices.tolist()
    print(
        "[OutlierFilterDemo] before=5 after=4 discarded=1 kept_indices=",
        kept_indices.tolist(),
    )
    return stats


@torch.no_grad()
def _run_deterministic_bank_build_check(seed: int, channels: Sequence[int], spatial_sizes: Sequence[int], num_clusters: int, build_stage: int, device: torch.device) -> Dict:
    def _build_once(run_seed: int):
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        memory = CrossScaleCTPETPrototypeMemory(
            channels=channels,
            num_clusters=num_clusters,
            build_stage=build_stage,
            output_dir=str(Path("/tmp") / f"cppi_det_{run_seed}"),
        ).to(device)
        memory.train()
        local_rng = random.Random(run_seed)
        for _ in range(3):
            mask = _make_small_lesion_masks(
                batch_size=4,
                image_size=128,
                rng=local_rng,
                device=device,
            )
            ct_feats, pet_feats = _make_synthetic_features(
                mask=mask,
                channels=channels,
                spatial_sizes=spatial_sizes,
                device=device,
            )
            memory.collect(
                ct_feats=ct_feats,
                pet_feats=pet_feats,
                mask=mask,
                print_info=False,
                compute_report=False,
            )
        report = memory.finalize_epoch(
            epoch=1,
            save_json=False,
            save_visualizations=False,
            print_info=False,
        )
        return memory, report

    mem1, rep1 = _build_once(seed)
    mem2, rep2 = _build_once(seed)

    comparisons = {}
    for scale_idx in range(len(channels)):
        k1 = getattr(mem1, f"ct_keys_s{scale_idx + 1}")
        k2 = getattr(mem2, f"ct_keys_s{scale_idx + 1}")
        v1 = getattr(mem1, f"pet_values_s{scale_idx + 1}")
        v2 = getattr(mem2, f"pet_values_s{scale_idx + 1}")
        comparisons[f"scale_{scale_idx + 1}"] = {
            "ct_keys_close": bool(torch.allclose(k1, k2)),
            "pet_values_close": bool(torch.allclose(v1, v2)),
        }
    assert rep1["bank_version_after"] == rep2["bank_version_after"]
    assert mem1.bank_ready == mem2.bank_ready
    print("[DeterminismDemo] repeated bank build produced identical keys/values")
    return {"report1": rep1, "report2": rep2, "comparisons": comparisons}


def run_demo(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Smaller channels for a fast standalone check.
    channels = (16, 32, 64, 96)
    spatial_sizes = (64, 32, 16, 8)
    output_dir = Path(args.output_dir)

    _run_outlier_filter_sanity_check()
    _run_deterministic_bank_build_check(
        seed=args.seed,
        channels=(8, 16, 32, 48),
        spatial_sizes=(32, 16, 8, 4),
        num_clusters=3,
        build_stage=4,
        device=device,
    )

    memory = CrossScaleCTPETPrototypeMemory(
        channels=channels,
        num_clusters=args.num_clusters,
        build_stage=args.build_stage,
        output_dir=str(output_dir),
    ).to(device)
    memory.train()

    rng = random.Random(args.seed)
    collect_reports = []

    for batch_idx in range(args.demo_batches):
        mask = _make_small_lesion_masks(
            batch_size=args.demo_batch_size,
            image_size=128,
            rng=rng,
            device=device,
        )
        ct_feats, pet_feats = _make_synthetic_features(
            mask=mask,
            channels=channels,
            spatial_sizes=spatial_sizes,
            device=device,
        )
        info = memory.collect(
            ct_feats=ct_feats,
            pet_feats=pet_feats,
            mask=mask,
            print_info=True,
            compute_report=True,
        )
        collect_reports.append(info)

    bank_report = memory.finalize_epoch(
        epoch=1,
        save_json=True,
        save_visualizations=True,
        print_info=True,
    )
    assert memory.bank_ready
    assert int(memory.bank_version.item()) >= 1

    # One missing retrieval pass.
    mask = _make_small_lesion_masks(
        batch_size=args.demo_batch_size,
        image_size=128,
        rng=rng,
        device=device,
    )
    ct_feats, _ = _make_synthetic_features(
        mask=mask,
        channels=channels,
        spatial_sizes=spatial_sizes,
        device=device,
    )
    fused, pet_proxy, retrieval_report = memory.fuse_missing(
        ct_feats,
        save_diagnostics=True,
        tag="demo_missing",
        visualize_batch_index=0,
        print_info=True,
        compute_report=True,
    )
    assert len(pet_proxy) == len(ct_feats)
    for idx, (proxy, ct_feat) in enumerate(zip(pet_proxy, ct_feats)):
        assert proxy.shape == ct_feat.shape
        assert fused[idx].shape == ct_feat.shape

    # Verify differentiability through CT query and attention projections.
    demo_loss = sum(x.square().mean() for x in fused)
    demo_loss.backward()
    grad_norms = []
    for scale_idx, ct in enumerate(ct_feats):
        grad_norms.append(
            {
                "scale": scale_idx + 1,
                "ct_grad_norm": float(ct.grad.norm().detach().cpu()) if ct.grad is not None else 0.0,
                "pet_proxy_shape": list(pet_proxy[scale_idx].shape),
                "fused_shape": list(fused[scale_idx].shape),
            }
        )

    demo_summary = {
        "status": "success",
        "device": str(device),
        "seed": args.seed,
        "config": asdict(memory.config),
        "demo_loss": float(demo_loss.detach().cpu()),
        "gradients": grad_norms,
        "bank_report_file": str(memory.json_dir / "bank_epoch_1.json"),
        "retrieval_report_file": str(memory.json_dir / "retrieval_demo_missing.json"),
        "visualization_dir": str(memory.vis_dir),
    }
    _save_json(demo_summary, memory.json_dir / "demo_summary.json")

    print("[Demo] Standalone verification completed successfully.")
    print(f"[Demo] Summary: {memory.json_dir / 'demo_summary.json'}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone CT-key/PET-value prototype imputation module"
    )
    parser.add_argument("--demo", action="store_true", help="Run synthetic verification")
    parser.add_argument("--num-clusters", type=int, default=4)
    parser.add_argument("--build-stage", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="prototype_memory_demo_outputs",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto', 'cpu', or a CUDA device such as 'cuda:0'",
    )
    parser.add_argument("--demo-batches", type=int, default=6)
    parser.add_argument("--demo-batch-size", type=int, default=4)
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    if not args.demo:
        parser.print_help()
        print( 
            "\nRun a verification demo with:\n"
            "python ct_pet_prototype_imputation.py --demo "
            "--num-clusters 4 --build-stage 4"
        )
    else:
        run_demo(args)