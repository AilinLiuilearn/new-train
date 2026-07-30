#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDTM: Paired Distribution Transport Memory
===========================================

独立验证脚本，用于实现：

    paired Full / CT-only task features
        -> Gaussian distribution modeling
        -> closed-form Bures-Wasserstein transport operator
        -> compact transport memory
        -> missing-PET retrieval and CT-only distribution calibration

核心对象
--------
Memory Key:
    CT-only source Gaussian distribution N(mu_ct, Sigma_ct)

Memory Value:
    paired CT-only -> Full transport operator (delta_mu, A)

Missing inference:
    1) estimate current CT-only distribution;
    2) retrieve nearest memory source distribution by W2 distance;
    3) apply the paired transport operator to CT-only features;
    4) output transported feature map for the downstream segmentation head.

输入文件
--------
.pt / .pth，内容可为：
    - Tensor [B, C, H, W]
    - dict containing one of: features / ct_features / full_features / tensor

运行示例
--------
1) 内置合成数据自检：
    python pdtm_standalone.py --mode demo --output-dir ./pdtm_demo

2) 用配对训练特征建库：
    python pdtm_standalone.py \
        --mode build \
        --ct-features ./ct_only_train.pt \
        --full-features ./full_train.pt \
        --slots 8 \
        --output-dir ./pdtm_memory

3) PET 缺失时检索并调整 CT-only 特征：
    python pdtm_standalone.py \
        --mode infer \
        --memory-path ./pdtm_memory/pdtm_memory.pt \
        --ct-features ./ct_only_test.pt \
        --full-features ./full_test.pt \
        --output-dir ./pdtm_infer

说明
----
- 本脚本不包含编码器、解码器和分割头，只验证 PDTM 核心机制。
- --full-features 在 infer 模式中是可选的，仅用于评估运输前后的分布距离。
- paired_bures_wasserstein_loss() 可直接作为训练时唯一的可选 OT 辅助损失。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# =============================================================================
# Utilities
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(payload: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().double()
    return {
        "mean": float(x.mean().cpu()),
        "std": float(x.std(unbiased=False).cpu()),
        "min": float(x.min().cpu()),
        "max": float(x.max().cpu()),
        "abs_mean": float(x.abs().mean().cpu()),
        "l2_norm": float(torch.linalg.vector_norm(x).cpu()),
    }


def load_feature_tensor(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        tensor = obj
    elif isinstance(obj, dict):
        tensor = None
        for key in ("features", "ct_features", "full_features", "tensor"):
            if isinstance(obj.get(key), torch.Tensor):
                tensor = obj[key]
                break
        if tensor is None:
            raise ValueError(
                f"{path} is a dict but no supported tensor key was found."
            )
    else:
        raise TypeError(f"Unsupported feature file content: {type(obj)!r}")

    if tensor.ndim != 4:
        raise ValueError(
            f"Expected [B,C,H,W], got shape={tuple(tensor.shape)} from {path}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Non-finite values found in {path}")
    return tensor.float().contiguous()


def flatten_spatial(feature: torch.Tensor) -> torch.Tensor:
    """[C,H,W] -> [H*W,C]."""
    if feature.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(feature.shape)}")
    c, h, w = feature.shape
    return feature.permute(1, 2, 0).reshape(h * w, c)


# =============================================================================
# Symmetric positive-definite matrix operations
# =============================================================================


def symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def spd_eigh(
    matrix: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    matrix = symmetrize(matrix.double())
    values, vectors = torch.linalg.eigh(matrix)
    values = values.clamp_min(eps)
    return values, vectors


def spd_matrix_power(
    matrix: torch.Tensor,
    power: float,
    eps: float,
) -> torch.Tensor:
    values, vectors = spd_eigh(matrix, eps)
    powered = values.pow(power)
    result = (vectors * powered.unsqueeze(-2)) @ vectors.transpose(-1, -2)
    return symmetrize(result)


def spd_sqrt(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    return spd_matrix_power(matrix, 0.5, eps)


def spd_inv_sqrt(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    return spd_matrix_power(matrix, -0.5, eps)


def matrix_condition_number(matrix: torch.Tensor, eps: float) -> float:
    values, _ = spd_eigh(matrix, eps)
    return float((values.max() / values.min()).cpu())


# =============================================================================
# Gaussian feature distribution and Bures-Wasserstein OT
# =============================================================================


@dataclass
class GaussianDistribution:
    mean: torch.Tensor
    covariance: torch.Tensor
    sample_count: int

    def cpu(self) -> "GaussianDistribution":
        return GaussianDistribution(
            mean=self.mean.cpu(),
            covariance=self.covariance.cpu(),
            sample_count=self.sample_count,
        )


def estimate_gaussian(
    samples: torch.Tensor,
    eps: float,
) -> GaussianDistribution:
    """Estimate N(mean, covariance) from [N,C] samples."""
    if samples.ndim != 2:
        raise ValueError(f"Expected [N,C], got {tuple(samples.shape)}")
    if samples.shape[0] < 2:
        raise ValueError("At least two samples are required for covariance.")

    x = samples.double()
    mean = x.mean(dim=0)
    centered = x - mean
    covariance = centered.transpose(0, 1) @ centered / float(x.shape[0])
    covariance = symmetrize(covariance)
    covariance = covariance + eps * torch.eye(
        covariance.shape[0], dtype=covariance.dtype, device=covariance.device
    )
    return GaussianDistribution(mean, covariance, int(x.shape[0]))


def gaussian_from_feature(
    feature: torch.Tensor,
    eps: float,
) -> GaussianDistribution:
    return estimate_gaussian(flatten_spatial(feature), eps)


def bures_wasserstein_squared(
    source: GaussianDistribution,
    target: GaussianDistribution,
    eps: float,
) -> torch.Tensor:
    """Squared W2 distance between Gaussian distributions."""
    source_mean = source.mean.double()
    target_mean = target.mean.double()
    source_cov = source.covariance.double()
    target_cov = target.covariance.double()

    mean_term = (source_mean - target_mean).square().sum()
    source_sqrt = spd_sqrt(source_cov, eps)
    middle = source_sqrt @ target_cov @ source_sqrt
    middle_sqrt = spd_sqrt(middle, eps)
    covariance_term = torch.trace(source_cov + target_cov - 2.0 * middle_sqrt)
    return (mean_term + covariance_term).clamp_min(0.0)


def gaussian_ot_operator(
    source: GaussianDistribution,
    target: GaussianDistribution,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Closed-form OT map from Gaussian source to target:

        A = Sigma_s^{-1/2}
            (Sigma_s^{1/2} Sigma_t Sigma_s^{1/2})^{1/2}
            Sigma_s^{-1/2}
        delta_mean = mean_t - mean_s
    """
    source_cov = source.covariance.double()
    target_cov = target.covariance.double()
    source_sqrt = spd_sqrt(source_cov, eps)
    source_inv_sqrt = spd_inv_sqrt(source_cov, eps)
    middle = source_sqrt @ target_cov @ source_sqrt
    operator = source_inv_sqrt @ spd_sqrt(middle, eps) @ source_inv_sqrt
    operator = symmetrize(operator)
    delta_mean = target.mean.double() - source.mean.double()
    return delta_mean, operator


def apply_transport(
    feature: torch.Tensor,
    current_distribution: GaussianDistribution,
    delta_mean: torch.Tensor,
    operator: torch.Tensor,
) -> torch.Tensor:
    """Apply the retrieved paired transport operator to [C,H,W]."""
    samples = flatten_spatial(feature).double()
    centered = samples - current_distribution.mean.double()
    transported = (
        current_distribution.mean.double()
        + delta_mean.double()
        + centered @ operator.double().transpose(0, 1)
    )
    c, h, w = feature.shape
    return (
        transported.reshape(h, w, c)
        .permute(2, 0, 1)
        .to(feature.dtype)
        .contiguous()
    )


def paired_bures_wasserstein_loss(
    ct_features: torch.Tensor,
    full_features: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Optional paired OT auxiliary loss for [B,C,H,W] tensors."""
    if ct_features.shape != full_features.shape:
        raise ValueError(
            f"Shape mismatch: {ct_features.shape} vs {full_features.shape}"
        )
    losses = []
    for index in range(ct_features.shape[0]):
        ct_dist = gaussian_from_feature(ct_features[index], eps)
        full_dist = gaussian_from_feature(full_features[index], eps)
        losses.append(bures_wasserstein_squared(ct_dist, full_dist, eps))
    return torch.stack(losses).mean().to(ct_features.dtype)


# =============================================================================
# Paired distribution transport memory
# =============================================================================


@dataclass
class TransportEntry:
    case_id: str
    source_mean: torch.Tensor
    source_covariance: torch.Tensor
    target_mean: torch.Tensor
    target_covariance: torch.Tensor
    delta_mean: torch.Tensor
    operator: torch.Tensor
    paired_w2: float
    operator_mean_error: float
    operator_covariance_error: float

    def source_distribution(self) -> GaussianDistribution:
        return GaussianDistribution(
            self.source_mean,
            self.source_covariance,
            sample_count=0,
        )


class PairedDistributionTransportMemory:
    def __init__(self, eps: float = 1e-5) -> None:
        self.eps = float(eps)
        self.channel_count: Optional[int] = None
        self.entries: List[TransportEntry] = []
        self.build_report: Dict[str, Any] = {}

    @property
    def ready(self) -> bool:
        return len(self.entries) > 0

    def _make_entry(
        self,
        ct_feature: torch.Tensor,
        full_feature: torch.Tensor,
        case_id: str,
    ) -> TransportEntry:
        source = gaussian_from_feature(ct_feature, self.eps)
        target = gaussian_from_feature(full_feature, self.eps)
        delta_mean, operator = gaussian_ot_operator(source, target, self.eps)

        mapped_mean = source.mean + delta_mean
        mapped_covariance = symmetrize(
            operator @ source.covariance @ operator.transpose(0, 1)
        )
        mean_error = torch.linalg.vector_norm(mapped_mean - target.mean)
        covariance_error = torch.linalg.matrix_norm(
            mapped_covariance - target.covariance, ord="fro"
        )
        paired_w2 = bures_wasserstein_squared(source, target, self.eps)

        return TransportEntry(
            case_id=case_id,
            source_mean=source.mean.cpu(),
            source_covariance=source.covariance.cpu(),
            target_mean=target.mean.cpu(),
            target_covariance=target.covariance.cpu(),
            delta_mean=delta_mean.cpu(),
            operator=operator.cpu(),
            paired_w2=float(paired_w2.cpu()),
            operator_mean_error=float(mean_error.cpu()),
            operator_covariance_error=float(covariance_error.cpu()),
        )

    def _source_distance_matrix(
        self,
        entries: Sequence[TransportEntry],
    ) -> torch.Tensor:
        n = len(entries)
        matrix = torch.zeros(n, n, dtype=torch.float64)
        for i in range(n):
            source_i = entries[i].source_distribution()
            for j in range(i + 1, n):
                distance = bures_wasserstein_squared(
                    source_i,
                    entries[j].source_distribution(),
                    self.eps,
                )
                matrix[i, j] = distance
                matrix[j, i] = distance
        return matrix

    @staticmethod
    def _farthest_first_initialization(
        distance_matrix: torch.Tensor,
        k: int,
    ) -> List[int]:
        first = int(distance_matrix.sum(dim=1).argmin().item())
        medoids = [first]
        while len(medoids) < k:
            nearest = distance_matrix[:, medoids].min(dim=1).values.clone()
            nearest[torch.tensor(medoids)] = -1.0
            medoids.append(int(nearest.argmax().item()))
        return medoids

    @classmethod
    def _k_medoids(
        cls,
        distance_matrix: torch.Tensor,
        k: int,
        max_iters: int,
    ) -> Tuple[List[int], torch.Tensor]:
        n = int(distance_matrix.shape[0])
        if not 1 <= k <= n:
            raise ValueError(f"k must be in [1,{n}], got {k}")

        medoids = cls._farthest_first_initialization(distance_matrix, k)
        for _ in range(max_iters):
            assignment = distance_matrix[:, medoids].argmin(dim=1)
            updated: List[int] = []

            for cluster_id in range(k):
                members = torch.where(assignment == cluster_id)[0]
                if members.numel() == 0:
                    nearest = distance_matrix[:, medoids].min(dim=1).values.clone()
                    blocked = list(set(medoids + updated))
                    nearest[torch.tensor(blocked)] = -1.0
                    updated.append(int(nearest.argmax().item()))
                    continue

                intra = distance_matrix[members][:, members]
                best_local = int(intra.sum(dim=1).argmin().item())
                updated.append(int(members[best_local].item()))

            if updated == medoids:
                break
            medoids = updated

        assignment = distance_matrix[:, medoids].argmin(dim=1)
        return medoids, assignment

    def build(
        self,
        ct_features: torch.Tensor,
        full_features: torch.Tensor,
        slots: int,
        case_ids: Optional[Sequence[str]] = None,
        max_kmedoids_iters: int = 30,
    ) -> Dict[str, Any]:
        if ct_features.shape != full_features.shape:
            raise ValueError(
                f"Shape mismatch: {ct_features.shape} vs {full_features.shape}"
            )
        if ct_features.ndim != 4:
            raise ValueError("Expected [B,C,H,W] tensors.")
        if not torch.isfinite(ct_features).all() or not torch.isfinite(full_features).all():
            raise ValueError("Non-finite feature values detected.")

        batch, channels, height, width = ct_features.shape
        case_ids = list(case_ids) if case_ids is not None else [
            f"case_{i:04d}" for i in range(batch)
        ]
        if len(case_ids) != batch:
            raise ValueError("case_ids length must equal batch size.")

        print("=" * 92)
        print("PDTM BUILD: paired CT-only -> Full distribution transport memory")
        print(
            f"pairs={batch} | feature_shape=[{channels},{height},{width}] | "
            f"requested_slots={slots} | eps={self.eps:g}"
        )
        print("=" * 92)

        all_entries = []
        for index in range(batch):
            entry = self._make_entry(
                ct_features[index].cpu(),
                full_features[index].cpu(),
                case_ids[index],
            )
            all_entries.append(entry)
            print(
                f"[Pair {index:03d}] {entry.case_id} | "
                f"W2^2={entry.paired_w2:.6f} | "
                f"mean_map_err={entry.operator_mean_error:.3e} | "
                f"cov_map_err={entry.operator_covariance_error:.3e}"
            )

        source_distances = self._source_distance_matrix(all_entries)
        effective_slots = min(int(slots), batch)
        medoid_indices, assignment = self._k_medoids(
            source_distances,
            effective_slots,
            max_kmedoids_iters,
        )

        self.entries = [all_entries[index] for index in medoid_indices]
        self.channel_count = int(channels)
        cluster_sizes = [
            int((assignment == cluster_id).sum().item())
            for cluster_id in range(effective_slots)
        ]

        self.build_report = {
            "ready": True,
            "pair_count": int(batch),
            "channel_count": int(channels),
            "height": int(height),
            "width": int(width),
            "requested_slots": int(slots),
            "effective_slots": int(effective_slots),
            "selected_medoid_indices": medoid_indices,
            "selected_case_ids": [entry.case_id for entry in self.entries],
            "cluster_sizes": cluster_sizes,
            "paired_w2_values": [entry.paired_w2 for entry in all_entries],
            "paired_w2_mean": float(
                np.mean([entry.paired_w2 for entry in all_entries])
            ),
            "operator_mean_error_mean": float(
                np.mean([entry.operator_mean_error for entry in all_entries])
            ),
            "operator_covariance_error_mean": float(
                np.mean([entry.operator_covariance_error for entry in all_entries])
            ),
            "source_distance_matrix": source_distances,
        }

        print("-" * 92)
        print(f"Memory ready with {effective_slots} slots")
        for slot, (entry, size) in enumerate(zip(self.entries, cluster_sizes)):
            print(
                f"[Slot {slot:02d}] source={entry.case_id} | "
                f"cluster_size={size} | paired_W2^2={entry.paired_w2:.6f}"
            )
        print("=" * 92)
        return self.build_report

    def retrieve(
        self,
        ct_feature: torch.Tensor,
    ) -> Tuple[int, torch.Tensor, GaussianDistribution]:
        if not self.ready:
            raise RuntimeError("PDTM memory is not ready.")
        if ct_feature.ndim != 3:
            raise ValueError("Expected one [C,H,W] feature map.")
        if self.channel_count is not None and ct_feature.shape[0] != self.channel_count:
            raise ValueError(
                f"Channel mismatch: memory={self.channel_count}, input={ct_feature.shape[0]}"
            )

        current = gaussian_from_feature(ct_feature.cpu(), self.eps)
        distances = torch.stack(
            [
                bures_wasserstein_squared(
                    current,
                    entry.source_distribution(),
                    self.eps,
                )
                for entry in self.entries
            ]
        )
        slot = int(distances.argmin().item())
        return slot, distances, current

    def transform_batch(
        self,
        ct_features: torch.Tensor,
        full_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        if ct_features.ndim != 4:
            raise ValueError("Expected [B,C,H,W] CT-only features.")
        if full_features is not None and full_features.shape != ct_features.shape:
            raise ValueError("full_features must match ct_features shape.")

        print("=" * 92)
        print("PDTM INFER: retrieve paired transport and calibrate CT-only features")
        print("=" * 92)

        outputs: List[torch.Tensor] = []
        reports: List[Dict[str, Any]] = []

        for index in range(ct_features.shape[0]):
            ct_feature = ct_features[index].cpu()
            slot, distances, current = self.retrieve(ct_feature)
            entry = self.entries[slot]

            transported = apply_transport(
                ct_feature,
                current,
                entry.delta_mean,
                entry.operator,
            )
            transported_dist = gaussian_from_feature(transported, self.eps)

            sorted_distances = torch.sort(distances).values
            margin = (
                float((sorted_distances[1] - sorted_distances[0]).cpu())
                if sorted_distances.numel() > 1
                else float("inf")
            )

            report: Dict[str, Any] = {
                "sample_index": int(index),
                "selected_slot": int(slot),
                "selected_case_id": entry.case_id,
                "retrieval_distances": distances,
                "nearest_distance": float(distances[slot].cpu()),
                "retrieval_margin": margin,
                "ct_feature": tensor_stats(ct_feature),
                "transported_feature": tensor_stats(transported),
                "current_mean": tensor_stats(current.mean),
                "retrieved_delta_mean": tensor_stats(entry.delta_mean),
                "retrieved_operator": tensor_stats(entry.operator),
                "ct_covariance_trace": float(torch.trace(current.covariance).cpu()),
                "transported_covariance_trace": float(
                    torch.trace(transported_dist.covariance).cpu()
                ),
                "source_condition_number": matrix_condition_number(
                    current.covariance, self.eps
                ),
                "transported_condition_number": matrix_condition_number(
                    transported_dist.covariance, self.eps
                ),
            }

            if full_features is not None:
                full_dist = gaussian_from_feature(full_features[index].cpu(), self.eps)
                before = bures_wasserstein_squared(current, full_dist, self.eps)
                after = bures_wasserstein_squared(
                    transported_dist,
                    full_dist,
                    self.eps,
                )
                report.update(
                    {
                        "w2_to_full_before": float(before.cpu()),
                        "w2_to_full_after": float(after.cpu()),
                        "w2_improvement": float((before - after).cpu()),
                        "w2_relative_reduction": float(
                            ((before - after) / before.clamp_min(self.eps)).cpu()
                        ),
                    }
                )

            print(
                f"[Sample {index:03d}] slot={slot:02d} ({entry.case_id}) | "
                f"nearest_W2^2={report['nearest_distance']:.6f} | margin={margin:.6f}"
            )
            if full_features is not None:
                print(
                    f"             W2^2 to Full: "
                    f"{report['w2_to_full_before']:.6f} -> "
                    f"{report['w2_to_full_after']:.6f} | "
                    f"improvement={report['w2_improvement']:+.6f}"
                )

            outputs.append(transported)
            reports.append(report)

        print("=" * 92)
        return torch.stack(outputs, dim=0), reports

    def memory_summary(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "eps": self.eps,
            "channel_count": self.channel_count,
            "slot_count": len(self.entries),
            "slots": [
                {
                    "slot": slot,
                    "case_id": entry.case_id,
                    "paired_w2": entry.paired_w2,
                    "source_mean": tensor_stats(entry.source_mean),
                    "target_mean": tensor_stats(entry.target_mean),
                    "delta_mean": tensor_stats(entry.delta_mean),
                    "operator": tensor_stats(entry.operator),
                    "source_covariance_trace": float(
                        torch.trace(entry.source_covariance).cpu()
                    ),
                    "target_covariance_trace": float(
                        torch.trace(entry.target_covariance).cpu()
                    ),
                    "source_condition_number": matrix_condition_number(
                        entry.source_covariance, self.eps
                    ),
                    "target_condition_number": matrix_condition_number(
                        entry.target_covariance, self.eps
                    ),
                }
                for slot, entry in enumerate(self.entries)
            ],
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "eps": self.eps,
            "channel_count": self.channel_count,
            "entries": [
                {
                    "case_id": entry.case_id,
                    "source_mean": entry.source_mean,
                    "source_covariance": entry.source_covariance,
                    "target_mean": entry.target_mean,
                    "target_covariance": entry.target_covariance,
                    "delta_mean": entry.delta_mean,
                    "operator": entry.operator,
                    "paired_w2": entry.paired_w2,
                    "operator_mean_error": entry.operator_mean_error,
                    "operator_covariance_error": entry.operator_covariance_error,
                }
                for entry in self.entries
            ],
            "build_report": self.build_report,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.eps = float(state["eps"])
        self.channel_count = state.get("channel_count")
        self.entries = [
            TransportEntry(
                case_id=item["case_id"],
                source_mean=item["source_mean"].double(),
                source_covariance=item["source_covariance"].double(),
                target_mean=item["target_mean"].double(),
                target_covariance=item["target_covariance"].double(),
                delta_mean=item["delta_mean"].double(),
                operator=item["operator"].double(),
                paired_w2=float(item["paired_w2"]),
                operator_mean_error=float(item["operator_mean_error"]),
                operator_covariance_error=float(item["operator_covariance_error"]),
            )
            for item in state["entries"]
        ]
        self.build_report = state.get("build_report", {})

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PairedDistributionTransportMemory":
        state = torch.load(path, map_location="cpu")
        memory = cls(eps=float(state["eps"]))
        memory.load_state_dict(state)
        return memory


# =============================================================================
# Visualizations
# =============================================================================


def pca_project(arrays: Sequence[np.ndarray]) -> List[np.ndarray]:
    lengths = [array.shape[0] for array in arrays]
    merged = np.concatenate(arrays, axis=0).astype(np.float64)
    centered = merged - merged.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2].T
    projected = centered @ basis

    outputs: List[np.ndarray] = []
    start = 0
    for length in lengths:
        outputs.append(projected[start : start + length])
        start += length
    return outputs


def plot_feature_distribution_pca(
    ct_feature: torch.Tensor,
    transported_feature: torch.Tensor,
    output_path: Path,
    full_feature: Optional[torch.Tensor] = None,
) -> Path:
    arrays = [flatten_spatial(ct_feature).cpu().numpy()]
    labels = ["CT-only"]
    if full_feature is not None:
        arrays.append(flatten_spatial(full_feature).cpu().numpy())
        labels.append("Full")
    arrays.append(flatten_spatial(transported_feature).cpu().numpy())
    labels.append("Transported")

    projected = pca_project(arrays)
    fig = plt.figure(figsize=(8.2, 6.2), dpi=150)
    ax = fig.add_subplot(111)
    for points, label in zip(projected, labels):
        ax.scatter(points[:, 0], points[:, 1], s=9, alpha=0.45, label=label)
    ax.set_title("PDTM feature distributions (PCA projection)")
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_covariance_comparison(
    ct_feature: torch.Tensor,
    transported_feature: torch.Tensor,
    eps: float,
    output_path: Path,
    full_feature: Optional[torch.Tensor] = None,
) -> Path:
    covariances = [gaussian_from_feature(ct_feature, eps).covariance.numpy()]
    labels = ["CT-only covariance"]
    if full_feature is not None:
        covariances.append(gaussian_from_feature(full_feature, eps).covariance.numpy())
        labels.append("Full covariance")
    covariances.append(
        gaussian_from_feature(transported_feature, eps).covariance.numpy()
    )
    labels.append("Transported covariance")

    vmin = min(float(cov.min()) for cov in covariances)
    vmax = max(float(cov.max()) for cov in covariances)
    fig, axes = plt.subplots(
        1,
        len(covariances),
        figsize=(4.8 * len(covariances), 4.2),
        dpi=150,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    for ax, covariance, label in zip(axes, covariances, labels):
        image = ax.imshow(covariance, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(label)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Channel")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_retrieval_distances(
    distances: Sequence[float],
    selected_slot: int,
    output_path: Path,
) -> Path:
    values = np.asarray(distances, dtype=np.float64)
    fig = plt.figure(figsize=(7.2, 4.8), dpi=150)
    ax = fig.add_subplot(111)
    positions = np.arange(values.size)
    bars = ax.bar(positions, values)
    if 0 <= selected_slot < len(bars):
        bars[selected_slot].set_hatch("//")
    ax.set_title("Bures-Wasserstein retrieval distances")
    ax.set_xlabel("Memory slot")
    ax.set_ylabel("W2 squared")
    ax.set_xticks(positions)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_w2_before_after(
    infer_reports: Sequence[Dict[str, Any]],
    output_path: Path,
) -> Optional[Path]:
    valid = [
        report
        for report in infer_reports
        if "w2_to_full_before" in report and "w2_to_full_after" in report
    ]
    if not valid:
        return None

    indices = np.arange(len(valid))
    before = np.asarray([r["w2_to_full_before"] for r in valid])
    after = np.asarray([r["w2_to_full_after"] for r in valid])

    fig = plt.figure(figsize=(8.2, 4.8), dpi=150)
    ax = fig.add_subplot(111)
    width = 0.38
    ax.bar(indices - width / 2, before, width=width, label="Before")
    ax.bar(indices + width / 2, after, width=width, label="After")
    ax.set_title("W2 distance to paired Full distribution")
    ax.set_xlabel("Test sample")
    ax.set_ylabel("W2 squared")
    ax.set_xticks(indices)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_visualizations(
    ct_features: torch.Tensor,
    transported_features: torch.Tensor,
    infer_reports: Sequence[Dict[str, Any]],
    output_dir: Path,
    eps: float,
    sample_index: int,
    full_features: Optional[torch.Tensor] = None,
) -> List[Path]:
    if not 0 <= sample_index < ct_features.shape[0]:
        raise IndexError(
            f"visualize_sample_index={sample_index} is outside batch range."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    full_feature = None if full_features is None else full_features[sample_index]
    paths = [
        plot_feature_distribution_pca(
            ct_features[sample_index],
            transported_features[sample_index],
            output_dir / "feature_distribution_pca.png",
            full_feature,
        ),
        plot_covariance_comparison(
            ct_features[sample_index],
            transported_features[sample_index],
            eps,
            output_dir / "covariance_comparison.png",
            full_feature,
        ),
        plot_retrieval_distances(
            infer_reports[sample_index]["retrieval_distances"],
            infer_reports[sample_index]["selected_slot"],
            output_dir / "retrieval_distances.png",
        ),
    ]
    before_after = plot_w2_before_after(
        infer_reports,
        output_dir / "w2_before_after.png",
    )
    if before_after is not None:
        paths.append(before_after)
    return paths


# =============================================================================
# Synthetic demo data
# =============================================================================


def make_synthetic_pairs(
    pair_count: int,
    channels: int,
    height: int,
    width: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate paired CT-only and Full features with several transport families."""
    generator = torch.Generator().manual_seed(seed)
    spatial_count = height * width
    family_count = max(2, min(4, pair_count // 3))

    family_operators = []
    family_shifts = []
    family_source_covariances = []
    family_source_means = []
    for family in range(family_count):
        random_matrix = torch.randn(channels, channels, generator=generator)
        q, _ = torch.linalg.qr(random_matrix)
        scales = torch.linspace(
            0.78 + 0.06 * family,
            1.22 + 0.05 * family,
            channels,
        )
        family_operators.append(q @ torch.diag(scales) @ q.transpose(0, 1))
        family_shifts.append(
            torch.sin(torch.linspace(0, math.pi, channels) + family)
            * (0.10 + 0.025 * family)
        )

        # Demo 中让 CT-only 源分布与传输族相关，使“按源分布检索传输”
        # 这一核心假设可以被直接验证，而不是由完全随机源分布掩盖。
        source_mix = torch.randn(channels, channels, generator=generator)
        source_cov = (
            source_mix @ source_mix.transpose(0, 1) / channels
            + (0.55 + 0.12 * family) * torch.eye(channels)
        )
        source_mean = (
            0.20 * family
            + 0.06 * torch.sin(torch.linspace(0, math.pi, channels) + family)
        )
        family_source_covariances.append(source_cov)
        family_source_means.append(source_mean)

    ct_batch: List[torch.Tensor] = []
    full_batch: List[torch.Tensor] = []

    for index in range(pair_count):
        family = index % family_count
        latent = torch.randn(spatial_count, channels, generator=generator)
        source_cov = (
            family_source_covariances[family]
            + 0.015 * torch.eye(channels)
        )
        source_mean = (
            family_source_means[family]
            + 0.015 * torch.randn(channels, generator=generator)
        )
        ct_samples = latent @ spd_sqrt(source_cov, 1e-5).float().transpose(0, 1)
        ct_samples = ct_samples + source_mean

        centered = ct_samples - ct_samples.mean(dim=0, keepdim=True)
        full_samples = (
            centered @ family_operators[family].transpose(0, 1)
            + ct_samples.mean(dim=0, keepdim=True)
            + family_shifts[family]
            + 0.01 * torch.randn(ct_samples.shape, generator=generator)
        )

        ct_batch.append(
            ct_samples.reshape(height, width, channels)
            .permute(2, 0, 1)
            .contiguous()
        )
        full_batch.append(
            full_samples.reshape(height, width, channels)
            .permute(2, 0, 1)
            .contiguous()
        )

    return torch.stack(ct_batch), torch.stack(full_batch)


# =============================================================================
# CLI modes
# =============================================================================


def aggregate_inference_reports(
    reports: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {
        "sample_count": len(reports),
        "selected_slots": [report["selected_slot"] for report in reports],
        "nearest_distance_mean": float(
            np.mean([report["nearest_distance"] for report in reports])
        ),
        "retrieval_margin_mean": float(
            np.mean(
                [
                    report["retrieval_margin"]
                    for report in reports
                    if math.isfinite(report["retrieval_margin"])
                ]
                or [0.0]
            )
        ),
    }
    if reports and "w2_to_full_before" in reports[0]:
        before = np.asarray([r["w2_to_full_before"] for r in reports])
        after = np.asarray([r["w2_to_full_after"] for r in reports])
        aggregate.update(
            {
                "w2_to_full_before_mean": float(before.mean()),
                "w2_to_full_after_mean": float(after.mean()),
                "w2_improvement_mean": float((before - after).mean()),
                "w2_relative_reduction_mean": float(
                    np.mean(
                        [r["w2_relative_reduction"] for r in reports]
                    )
                ),
                "improved_sample_ratio": float(np.mean(after < before)),
            }
        )
    return aggregate


def run_build(args: argparse.Namespace) -> Dict[str, Path]:
    ct_features = load_feature_tensor(args.ct_features)
    full_features = load_feature_tensor(args.full_features)

    memory = PairedDistributionTransportMemory(eps=args.eps)
    build_report = memory.build(
        ct_features,
        full_features,
        slots=args.slots,
        max_kmedoids_iters=args.max_kmedoids_iters,
    )

    output_dir = Path(args.output_dir)
    memory_path = memory.save(output_dir / "pdtm_memory.pt")
    json_path = save_json(
        {
            "mode": "build",
            "config": vars(args),
            "memory": memory.memory_summary(),
            "build_report": build_report,
        },
        output_dir / "pdtm_build_report.json",
    )
    return {"memory": memory_path, "json": json_path}


def run_infer(args: argparse.Namespace) -> Dict[str, Path]:
    memory = PairedDistributionTransportMemory.load(args.memory_path)
    ct_features = load_feature_tensor(args.ct_features)
    full_features = (
        load_feature_tensor(args.full_features) if args.full_features else None
    )

    transported, infer_reports = memory.transform_batch(
        ct_features,
        full_features,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transported_path = output_dir / "transported_features.pt"
    torch.save(transported, transported_path)

    visualization_paths = save_visualizations(
        ct_features,
        transported,
        infer_reports,
        output_dir,
        memory.eps,
        args.visualize_sample_index,
        full_features,
    )

    json_path = save_json(
        {
            "mode": "infer",
            "config": vars(args),
            "memory": memory.memory_summary(),
            "aggregate": aggregate_inference_reports(infer_reports),
            "inference": infer_reports,
            "visualizations": visualization_paths,
        },
        output_dir / "pdtm_infer_report.json",
    )

    outputs = {
        "transported_features": transported_path,
        "json": json_path,
    }
    for index, path in enumerate(visualization_paths, start=1):
        outputs[f"visualization_{index}"] = path
    return outputs


def run_demo(args: argparse.Namespace) -> Dict[str, Path]:
    ct_all, full_all = make_synthetic_pairs(
        pair_count=args.demo_pairs,
        channels=args.demo_channels,
        height=args.demo_height,
        width=args.demo_width,
        seed=args.seed,
    )

    train_count = max(args.slots, int(round(args.demo_pairs * 0.7)))
    train_count = min(train_count, args.demo_pairs - 1)
    ct_train, ct_test = ct_all[:train_count], ct_all[train_count:]
    full_train, full_test = full_all[:train_count], full_all[train_count:]

    memory = PairedDistributionTransportMemory(eps=args.eps)
    build_report = memory.build(
        ct_train,
        full_train,
        slots=min(args.slots, ct_train.shape[0]),
        max_kmedoids_iters=args.max_kmedoids_iters,
    )
    transported, infer_reports = memory.transform_batch(ct_test, full_test)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_path = memory.save(output_dir / "pdtm_memory.pt")
    transported_path = output_dir / "transported_features.pt"
    torch.save(transported, transported_path)

    sample_index = min(args.visualize_sample_index, ct_test.shape[0] - 1)
    visualization_paths = save_visualizations(
        ct_test,
        transported,
        infer_reports,
        output_dir,
        memory.eps,
        sample_index,
        full_test,
    )

    aggregate = aggregate_inference_reports(infer_reports)
    json_path = save_json(
        {
            "mode": "demo",
            "config": vars(args),
            "data": {
                "train_pair_count": int(ct_train.shape[0]),
                "test_pair_count": int(ct_test.shape[0]),
                "feature_shape": list(ct_all.shape[1:]),
            },
            "memory": memory.memory_summary(),
            "build_report": build_report,
            "aggregate": aggregate,
            "inference": infer_reports,
            "visualizations": visualization_paths,
        },
        output_dir / "pdtm_demo_report.json",
    )

    print("\nDemo aggregate")
    print(
        f"mean W2^2 to Full: {aggregate['w2_to_full_before_mean']:.6f} -> "
        f"{aggregate['w2_to_full_after_mean']:.6f}"
    )
    print(f"mean improvement: {aggregate['w2_improvement_mean']:+.6f}")
    print(f"improved sample ratio: {100.0 * aggregate['improved_sample_ratio']:.1f}%")

    outputs = {
        "memory": memory_path,
        "transported_features": transported_path,
        "json": json_path,
    }
    for index, path in enumerate(visualization_paths, start=1):
        outputs[f"visualization_{index}"] = path
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone Paired Distribution Transport Memory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("demo", "build", "infer"), default="demo")
    parser.add_argument("--output-dir", type=str, default="./pdtm_outputs")
    parser.add_argument("--ct-features", type=str, default=None)
    parser.add_argument("--full-features", type=str, default=None)
    parser.add_argument("--memory-path", type=str, default=None)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--max-kmedoids-iters", type=int, default=30)
    parser.add_argument("--visualize-sample-index", type=int, default=0)

    parser.add_argument("--demo-pairs", type=int, default=16)
    parser.add_argument("--demo-channels", type=int, default=8)
    parser.add_argument("--demo-height", type=int, default=16)
    parser.add_argument("--demo-width", type=int, default=16)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.slots < 1:
        raise ValueError("--slots must be >= 1")
    if args.eps <= 0:
        raise ValueError("--eps must be > 0")
    if args.visualize_sample_index < 0:
        raise ValueError("--visualize-sample-index must be >= 0")

    if args.mode == "build" and (not args.ct_features or not args.full_features):
        raise ValueError("build mode requires --ct-features and --full-features")
    if args.mode == "infer" and (not args.memory_path or not args.ct_features):
        raise ValueError("infer mode requires --memory-path and --ct-features")
    if args.mode == "demo":
        if args.demo_pairs < 3:
            raise ValueError("--demo-pairs must be >= 3")
        if args.demo_channels < 2:
            raise ValueError("--demo-channels must be >= 2")
        if args.demo_height * args.demo_width < 2:
            raise ValueError("demo spatial sample count must be >= 2")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    set_seed(args.seed)

    if args.mode == "build":
        outputs = run_build(args)
    elif args.mode == "infer":
        outputs = run_infer(args)
    else:
        outputs = run_demo(args)

    print("\nGenerated files")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()