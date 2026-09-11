# -*- coding: utf-8 -*-
"""
Paired Semantic Multi-Prototype Imputation (Module-1)  —  API-style refactored
==========================================================================

Module-1 is ONLY the missing-PET compensation path:

    CT feats -> CT-key/PET-value paired prototype retrieval (cosine soft)
             -> population PET prototypes
             -> CT-conditioned spatial affine (gamma * proto + beta)
             -> compensated PET feats

Baseline invariants kept: deterministic spherical K-means (S4 CT descriptors),
cosine top-5% outlier filter, paired cross-scale member reuse, direct bank
buffers, Full/Missing strict AddFusion boundary.

Removed vs old PSPI: Stage-1.5 bootstrap, PASSION semantic relation loss,
ct_reference discrepancy personalization, zero-init/tanh gamma-beta.

Added (API-style): PET multi-positive prototype contrastive loss (grad ->
PET encoder) and FG/BG-balanced reconstruction loss (grad -> retrieval +
spatial personalization only). Both follow the API convention: all real
features are encoded first, detached features build candidates, the real PET
path keeps gradient for contrastive supervision and is detached as the
reconstruction target.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8
CLASS_NAMES = ("background", "foreground")


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _finite_or_raise(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        bad = int((~torch.isfinite(x)).sum().item())
        raise RuntimeError(f"{name} contains {bad} NaN/Inf values")


def _sanitize(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1, eps=EPS)


def _stage_to_index(stage: int, num_scales: int) -> int:
    stage = int(stage)
    if not 1 <= stage <= num_scales:
        raise ValueError(f"stage must be in [1,{num_scales}], got {stage}")
    return stage - 1


def _class_masks_at_scale(
    mask: torch.Tensor,
    output_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return soft BG/FG masks at a feature-map resolution."""
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must be [B,1,H,W], got {tuple(mask.shape)}")
    fg = F.adaptive_avg_pool2d(mask.float(), output_hw).clamp(0.0, 1.0)
    bg = 1.0 - fg
    return bg, fg


def _masked_average_pool_2d(
    feature: torch.Tensor,
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    feature: [B,C,H,W]
    weight:  [B,1,H,W]
    """
    if feature.ndim != 4 or weight.ndim != 4:
        raise ValueError("feature and weight must both be 4D")
    if feature.shape[0] != weight.shape[0] or feature.shape[-2:] != weight.shape[-2:]:
        raise ValueError(
            f"shape mismatch: feature={tuple(feature.shape)}, weight={tuple(weight.shape)}"
        )
    denom = weight.sum(dim=(2, 3))  # [B,1]
    valid = denom[:, 0] > EPS
    descriptor = (feature * weight).sum(dim=(2, 3)) / denom.clamp_min(EPS)
    descriptor = torch.where(valid[:, None], descriptor, torch.zeros_like(descriptor))
    return descriptor, valid


def _pairwise_cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine distance matrix: [N,D] x [M,D] -> [N,M]."""
    a_n = _normalize_rows(a)
    b_n = _normalize_rows(b)
    return 1.0 - a_n @ b_n.t()


def _zero_loss_result(ref: torch.Tensor) -> Dict:
    """Exact scalar 0 on the same device/dtype as ref."""
    zero = ref.new_zeros(())
    return {"loss": zero, "weighted_loss": zero, "num_terms": 0, "per_scale": {}, "details": {}}


# -----------------------------------------------------------------------------
# Deterministic spherical K-means + geometrically consistent outlier filtering
# -----------------------------------------------------------------------------


@torch.no_grad()
def deterministic_spherical_kmeans(
    x: torch.Tensor,
    num_clusters: int,
    max_iter: int = 25,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    if x.ndim != 2:
        raise ValueError(f"x must be [N,D], got {tuple(x.shape)}")
    n, _ = x.shape
    if n == 0:
        raise ValueError("Cannot cluster an empty tensor")
    if num_clusters < 1:
        raise ValueError("num_clusters must be >= 1")

    k = min(int(num_clusters), int(n))
    # L2 normalization BEFORE clustering: spherical/cosine geometry.
    x_n = _normalize_rows(x)

    mean_vec = x_n.mean(dim=0)
    mean_norm = float(mean_vec.norm().item())
    if mean_norm <= EPS:
        first_idx = 0
        mean_dir = x_n[first_idx]
    else:
        mean_dir = F.normalize(mean_vec, dim=0, eps=EPS)
        first_idx = int(torch.argmax(x_n @ mean_dir).item())

    center_indices: List[int] = [first_idx]

    while len(center_indices) < k:
        centers_now = x_n[
            torch.tensor(center_indices, device=x_n.device, dtype=torch.long)
        ]
        nearest_similarity = (x_n @ centers_now.t()).max(dim=1).values
        nearest_distance = 1.0 - nearest_similarity
        nearest_distance[
            torch.tensor(center_indices, device=x_n.device, dtype=torch.long)
        ] = -float("inf")
        next_idx = int(torch.argmax(nearest_distance).item())
        center_indices.append(next_idx)

    centers = x_n[
        torch.tensor(center_indices, device=x_n.device, dtype=torch.long)
    ].clone()
    labels = torch.full((n,), -1, dtype=torch.long, device=x_n.device)

    converged_iter = max_iter
    for iteration in range(max_iter):
        similarities = x_n @ centers.t()
        new_labels = similarities.argmax(dim=1)
        if torch.equal(new_labels, labels):
            labels = new_labels
            converged_iter = iteration
            break
        labels = new_labels

        new_centers: List[torch.Tensor] = []
        represented_similarity = similarities.max(dim=1).values
        used_replacements = set()

        for cluster_idx in range(k):
            members = x_n[labels == cluster_idx]
            if members.numel() > 0:
                center = F.normalize(members.mean(dim=0), dim=0, eps=EPS)
            else:
                scores = represented_similarity.clone()
                for idx in used_replacements:
                    scores[idx] = float("inf")
                replacement_idx = int(torch.argmin(scores).item())
                used_replacements.add(replacement_idx)
                center = x_n[replacement_idx]
            new_centers.append(center)

        centers = torch.stack(new_centers, dim=0)

    cluster_counts = [int((labels == j).sum().item()) for j in range(k)]
    report = {
        "num_candidates": int(n),
        "num_effective_clusters": int(k),
        "first_center_index": int(first_idx),
        "initial_center_indices": [int(v) for v in center_indices],
        "iterations": int(converged_iter),
        "cluster_counts": cluster_counts,
    }
    return labels, centers, report


@torch.no_grad()
def cosine_cluster_outlier_filter(
    build_features: torch.Tensor,
    labels: torch.Tensor,
    num_clusters: int,
    discard_rate: float,
) -> Tuple[Dict[int, torch.Tensor], Dict]:
    if not 0.0 <= float(discard_rate) < 1.0:
        raise ValueError("discard_rate must be in [0,1)")
    if build_features.ndim != 2 or labels.ndim != 1:
        raise ValueError("build_features must be [N,D] and labels must be [N]")
    if build_features.shape[0] != labels.shape[0]:
        raise ValueError("build_features / labels length mismatch")

    x_n = _normalize_rows(build_features)
    kept: Dict[int, torch.Tensor] = {}
    report: Dict[str, Dict] = {}
    singleton_warning = False

    for cluster_idx in range(int(num_clusters)):
        member_indices = torch.nonzero(
            labels == cluster_idx, as_tuple=False
        ).flatten().long()
        n = int(member_indices.numel())
        if n == 0:
            continue

        members = x_n[member_indices]
        center = F.normalize(members.mean(dim=0), dim=0, eps=EPS)
        distances = 1.0 - members @ center
        order = torch.argsort(distances, descending=False, stable=True)
        discard_n = int(math.floor(float(discard_rate) * n))
        keep_n = max(1, n - discard_n)
        kept_indices = member_indices[order[:keep_n]]
        kept[cluster_idx] = kept_indices

        if n == 1:
            singleton_warning = True

        report[str(cluster_idx)] = {
            "before_count": n,
            "after_count": int(keep_n),
            "discarded_count": int(n - keep_n),
            "mean_cosine_distance": float(distances.mean().item()),
            "max_cosine_distance": float(distances.max().item()),
            "kept_indices": kept_indices.detach().cpu().tolist(),
        }

    report["singleton_cluster_warning"] = singleton_warning
    return kept, report


# -----------------------------------------------------------------------------
# Prototype retrieval attention (cosine soft, temperature)
# -----------------------------------------------------------------------------


class PrototypeCrossAttention(nn.Module):
    """
    Q = current CT feature tokens (detached INSIDE Module-1)
    K = CT prototype keys   (L2 normalized after k_proj)
    V = paired PET prototype values (NOT normalized)

    logits = Normalize(W_q C_det) @ Normalize(W_k K)^T / temperature
    A = softmax(logits) over ready slots; P_proto = A @ W_v V.
    """

    def __init__(self, channels: int, retrieval_temperature: float = 0.1):
        super().__init__()
        self.channels = int(channels)
        self.temperature = float(retrieval_temperature)
        if self.temperature <= 0:
            raise ValueError("retrieval_temperature must be > 0")
        self.q_proj = nn.Linear(self.channels, self.channels, bias=False)
        self.k_proj = nn.Linear(self.channels, self.channels, bias=False)
        self.v_proj = nn.Linear(self.channels, self.channels, bias=False)
        self.out_proj = nn.Linear(self.channels, self.channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Projections stay identity-initialized (NOT the API affine init).
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
        if query_map.ndim != 4:
            raise ValueError("query_map must be [B,C,H,W]")
        b, c, h, w = query_map.shape
        if c != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {c}")
        if keys.ndim != 2 or values.ndim != 2 or keys.shape != values.shape:
            raise ValueError("keys and values must have the same [M,C] shape")
        if keys.shape[1] != c:
            raise ValueError("prototype channel mismatch")
        if ready.ndim != 1 or ready.shape[0] != keys.shape[0]:
            raise ValueError("ready must be [M]")

        if not bool(ready.any()):
            return (
                torch.zeros_like(query_map),
                torch.zeros(
                    b,
                    h * w,
                    keys.shape[0],
                    device=query_map.device,
                    dtype=query_map.dtype,
                ),
            )

        # CT detach happens here, inside Module-1.
        q = self.q_proj(query_map.detach().flatten(2).transpose(1, 2))  # [B,N,C]
        k = self.k_proj(keys)
        v = self.v_proj(values)

        # Cosine geometry: q/k normalized, v keeps its magnitude.
        q = F.normalize(q.float(), p=2, dim=-1, eps=EPS)
        k = F.normalize(k.float(), p=2, dim=-1, eps=EPS)

        logits = torch.matmul(q, k.t()) / float(self.temperature)
        logits = logits.masked_fill(
            ~ready.view(1, 1, -1).to(logits.device),
            torch.finfo(logits.dtype).min,
        )
        attention = torch.softmax(logits, dim=-1).to(dtype=query_map.dtype)
        retrieved = torch.matmul(attention, v.to(dtype=attention.dtype))
        retrieved = self.out_proj(retrieved.to(dtype=query_map.dtype))
        retrieved = retrieved.transpose(1, 2).reshape(b, c, h, w)

        _finite_or_raise("prototype_attention", attention)
        _finite_or_raise("retrieved_pet", retrieved)
        return _sanitize(retrieved), attention


# -----------------------------------------------------------------------------
# API-style spatial affine personalization
# -----------------------------------------------------------------------------


class SpatialPrototypePersonalization(nn.Module):
    """
    API-style CT spatial affine personalization.

        ct_condition = Normalize(detached CT, dim=1)     (per-pixel, channel dim)
        trunk: 1x1 Conv(C->hidden) -> GELU -> 3x3 Depthwise Conv -> GELU
        gamma = gamma_head(trunk);  beta = beta_head(trunk)
        P_comp = gamma * P_proto + beta                  (direct, elementwise)

    All Conv2d weights use Xavier uniform init with zero bias. The output
    heads have NO activation (no sigmoid/tanh) and are NOT zero-initialized:
    the initial output is not required to equal the prototype.
    """

    def __init__(self, channels: Sequence[int]):
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        self.trunks = nn.ModuleList()
        self.gamma_heads = nn.ModuleList()
        self.beta_heads = nn.ModuleList()
        for c in self.channels:
            hidden = max(c // 4, 16)
            trunk = nn.Sequential(
                nn.Conv2d(c, hidden, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=True),
                nn.GELU(),
            )
            gamma_head = nn.Conv2d(hidden, c, kernel_size=1, bias=True)
            beta_head = nn.Conv2d(hidden, c, kernel_size=1, bias=True)
            for m in trunk.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            nn.init.xavier_uniform_(gamma_head.weight)
            nn.init.zeros_(gamma_head.bias)
            nn.init.xavier_uniform_(beta_head.weight)
            nn.init.zeros_(beta_head.bias)
            self.trunks.append(trunk)
            self.gamma_heads.append(gamma_head)
            self.beta_heads.append(beta_head)

    def _scale_stats(self, gamma: torch.Tensor, beta: torch.Tensor,
                     pet_proto: torch.Tensor, pet_comp: torch.Tensor, tag: str) -> Dict[str, float]:
        with torch.no_grad():
            g, b = gamma.detach().float(), beta.detach().float()
            return {
                f"gamma_mean{tag}": float(g.mean().item()),
                f"gamma_std{tag}": float(g.std().item()) if g.numel() > 1 else 0.0,
                f"gamma_abs_mean{tag}": float(g.abs().mean().item()),
                f"beta_mean{tag}": float(b.mean().item()),
                f"beta_std{tag}": float(b.std().item()) if b.numel() > 1 else 0.0,
                f"beta_abs_mean{tag}": float(b.abs().mean().item()),
                f"pet_proto_norm{tag}": float(pet_proto.detach().float().pow(2).mean().sqrt().item()),
                f"pet_comp_norm{tag}": float(pet_comp.detach().float().pow(2).mean().sqrt().item()),
            }

    def forward(
        self,
        ct_feats: Sequence[torch.Tensor],
        pet_proto_feats: Sequence[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], Dict]:
        if len(ct_feats) != len(pet_proto_feats):
            raise ValueError("CT and PET prototype must have the same scale count")
        pet_comp: List[torch.Tensor] = []
        stats: Dict[str, float] = {}
        for scale_idx, (ct, pet_proto) in enumerate(zip(ct_feats, pet_proto_feats)):
            if pet_proto.shape[-2:] != ct.shape[-2:]:
                pet_proto = F.interpolate(
                    pet_proto, size=ct.shape[-2:], mode="bilinear", align_corners=False,
                )
            # CT condition: channel-normalized, detached (available-modality
            # conditioning is detached per API convention).
            ct_condition = F.normalize(
                ct.detach().float(), p=2, dim=1, eps=EPS,
            ).to(dtype=ct.dtype)
            feat = self.trunks[scale_idx](ct_condition)
            gamma = self.gamma_heads[scale_idx](feat)  # [B,C,H,W]
            beta = self.beta_heads[scale_idx](feat)    # [B,C,H,W]
            out = gamma * pet_proto + beta             # direct affine
            _finite_or_raise(f"personalization_gamma_s{scale_idx+1}", gamma)
            _finite_or_raise(f"personalization_beta_s{scale_idx+1}", beta)
            _finite_or_raise(f"pet_comp_s{scale_idx+1}", out)
            pet_comp.append(_sanitize(out))
            stats.update(self._scale_stats(gamma, beta, pet_proto, out, f"_s{scale_idx+1}"))
        # Aggregates over the provided scales (detached scalars only).
        n_scales = len(ct_feats)
        with torch.no_grad():
            for key in ("gamma_mean", "gamma_std", "gamma_abs_mean",
                        "beta_mean", "beta_std", "beta_abs_mean",
                        "pet_proto_norm", "pet_comp_norm"):
                vals = [stats[f"{key}_s{i+1}"] for i in range(n_scales)]
                stats[key] = float(sum(vals) / len(vals))
        return pet_comp, stats


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class Module1Config:
    channels: Tuple[int, ...]
    num_clusters: int = 6
    build_stage: int = 4
    cluster_max_iter: int = 25
    outlier_discard_rate: float = 0.05
    bank_update_mode: str = "direct"
    ema_momentum: float = 0.999
    retrieval_temperature: float = 0.1
    proto_contrastive_weight: float = 0.01
    proto_temperature: float = 0.02
    reconstruction_weight: float = 0.1
    spatial_affine: bool = True
    collect_candidates_during_training: bool = True

    def validate(self) -> None:
        if len(self.channels) == 0:
            raise ValueError("channels cannot be empty")
        if self.num_clusters < 1:
            raise ValueError("num_clusters must be >= 1")
        _stage_to_index(self.build_stage, len(self.channels))
        if self.cluster_max_iter < 1:
            raise ValueError("cluster_max_iter must be >= 1")
        if not 0.0 <= self.outlier_discard_rate < 1.0:
            raise ValueError("outlier_discard_rate must be in [0,1)")
        valid_modes = {"direct", "matched_ema", "fedmepd_ema"}
        if self.bank_update_mode not in valid_modes:
            raise ValueError(f"bank_update_mode must be one of {valid_modes}, got {self.bank_update_mode!r}")
        if not 0.0 <= self.ema_momentum < 1.0:
            raise ValueError("ema_momentum must be in [0,1)")
        if float(self.retrieval_temperature) <= 0:
            raise ValueError("retrieval_temperature must be > 0")
        if float(self.proto_temperature) <= 0:
            raise ValueError("proto_temperature must be > 0")
        if float(self.proto_contrastive_weight) < 0:
            raise ValueError("proto_contrastive_weight must be >= 0")
        if float(self.reconstruction_weight) < 0:
            raise ValueError("reconstruction_weight must be >= 0")


# -----------------------------------------------------------------------------
# Main standalone Module-1  (missing-only, API-style)
# -----------------------------------------------------------------------------


class PairedSemanticPrototypeImputation(nn.Module):
    """
    Missing-only Module-1.

    Public contracts
    ----------------
    * Candidate collection:  collect_candidates(ct_feats, pet_feats_real, mask)
    * Bank update:           finalize_epoch(epoch)
    * Missing prediction:    recover_missing(ct_feats)  -- no PET argument
    * Contrastive loss:      compute_pet_prototype_contrastive_loss(pet_real, mask)
    * Reconstruction loss:   compute_balanced_reconstruction_loss(pet_comp, pet_real, mask)
    """

    def __init__(
        self,
        channels: Sequence[int],
        num_clusters: int = 6,
        build_stage: int = 4,
        cluster_max_iter: int = 25,
        outlier_discard_rate: float = 0.05,
        bank_update_mode: str = "direct",
        ema_momentum: float = 0.999,
        retrieval_temperature: float = 0.1,
        proto_contrastive_weight: float = 0.01,
        proto_temperature: float = 0.02,
        reconstruction_weight: float = 0.1,
        spatial_affine: bool = True,
        collect_candidates_during_training: bool = True,
        **kwargs,
    ):
        super().__init__()
        # Drop removed-option kwargs so stale builder/checkpoint code fails soft.
        for _legacy in (
            "prototype_loss_type", "prototype_loss_weight",
            "prototype_temperature", "prototype_loss_stages",
            "use_affine_calibration", "use_pet_contribution_gate",
            "use_retrieval_reliability",
            "semantic_loss_weight", "pspi_semantic_loss_weight",
            "bootstrap_bank", "pspi_bootstrap_bank",
        ):
            kwargs.pop(_legacy, None)
        if kwargs:
            raise TypeError(f"Unexpected kwargs for PairedSemanticPrototypeImputation: {list(kwargs)}")

        self.config = Module1Config(
            channels=tuple(int(c) for c in channels),
            num_clusters=int(num_clusters),
            build_stage=int(build_stage),
            cluster_max_iter=int(cluster_max_iter),
            outlier_discard_rate=float(outlier_discard_rate),
            bank_update_mode=str(bank_update_mode),
            ema_momentum=float(ema_momentum),
            retrieval_temperature=float(retrieval_temperature),
            proto_contrastive_weight=float(proto_contrastive_weight),
            proto_temperature=float(proto_temperature),
            reconstruction_weight=float(reconstruction_weight),
            spatial_affine=bool(spatial_affine),
            collect_candidates_during_training=bool(collect_candidates_during_training),
        )
        self.config.validate()

        self.channels = self.config.channels
        self.num_scales = len(self.channels)
        self.num_clusters = self.config.num_clusters
        self.build_stage_idx = _stage_to_index(
            self.config.build_stage, self.num_scales
        )

        self.attention = nn.ModuleList(
            [
                PrototypeCrossAttention(c, retrieval_temperature=self.config.retrieval_temperature)
                for c in self.channels
            ]
        )
        self.personalization = SpatialPrototypePersonalization(self.channels)

        for scale_idx, c in enumerate(self.channels):
            self.register_buffer(
                f"ct_keys_s{scale_idx + 1}",
                torch.zeros(2, self.num_clusters, c, dtype=torch.float32),
            )
            self.register_buffer(
                f"pet_values_s{scale_idx + 1}",
                torch.zeros(2, self.num_clusters, c, dtype=torch.float32),
            )

        self.register_buffer(
            "prototype_ready",
            torch.zeros(2, self.num_clusters, dtype=torch.bool),
        )
        self.register_buffer(
            "prototype_count",
            torch.zeros(2, self.num_clusters, dtype=torch.long),
        )
        self.register_buffer("bank_version", torch.zeros((), dtype=torch.long))

        self._epoch_cache = self._new_cache()
        self._collect_calls = 0
        self._collected_records = 0

    @property
    def bank_ready(self) -> bool:
        return bool(self.prototype_ready.any())

    def export_config(self) -> Dict:
        return asdict(self.config)

    # ------------------------------------------------------------------
    # Validation / cache
    # ------------------------------------------------------------------

    def _validate_features(
        self,
        ct_feats: Sequence[torch.Tensor],
        pet_feats: Optional[Sequence[torch.Tensor]],
    ) -> None:
        if len(ct_feats) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} CT scales, got {len(ct_feats)}")
        if pet_feats is not None and len(pet_feats) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} PET scales, got {len(pet_feats)}")
        batch_size = int(ct_feats[0].shape[0])
        for s, ct in enumerate(ct_feats):
            if ct.ndim != 4:
                raise ValueError(f"ct_feats[{s}] must be 4D")
            if ct.shape[0] != batch_size or ct.shape[1] != self.channels[s]:
                raise ValueError(
                    f"ct_feats[{s}] expected [B,{self.channels[s]},H,W], got {tuple(ct.shape)}"
                )
            _finite_or_raise(f"ct_feats[{s}]", ct)
            if pet_feats is not None:
                pet = pet_feats[s]
                if pet.ndim != 4:
                    raise ValueError(f"pet_feats[{s}] must be 4D")
                if pet.shape != ct.shape:
                    raise ValueError(
                        f"Aligned CT/PET shape mismatch at scale {s+1}: "
                        f"ct={tuple(ct.shape)} pet={tuple(pet.shape)}"
                    )
                _finite_or_raise(f"pet_feats[{s}]", pet)

    def _new_cache(self) -> Dict:
        cache: Dict[int, Dict[str, List[List[torch.Tensor]]]] = {}
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
        self._collected_records = 0

    def _concat_cache(self, class_idx: int, modality: str, scale_idx: int) -> torch.Tensor:
        chunks = self._epoch_cache[class_idx][modality][scale_idx]
        if not chunks:
            return torch.empty(0, self.channels[scale_idx], dtype=torch.float32, device="cpu")
        return torch.cat(chunks, dim=0).float().contiguous()

    # ------------------------------------------------------------------
    # Candidate collection (detached, cross-scale aligned)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect_candidates(
        self,
        ct_feats: Sequence[torch.Tensor],
        pet_feats_real: Sequence[torch.Tensor],
        mask: torch.Tensor,
    ) -> Dict:
        self._validate_features(ct_feats, pet_feats_real)
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("mask must be [B,1,H,W]")
        if mask.shape[0] != ct_feats[0].shape[0]:
            raise ValueError("mask batch size mismatch")

        b = int(mask.shape[0])
        descriptors: Dict[int, Dict[str, List[torch.Tensor]]] = {
            c: {"ct": [], "pet": []} for c in range(2)
        }
        valids: Dict[int, List[torch.Tensor]] = {0: [], 1: []}

        for scale_idx, (ct, pet) in enumerate(zip(ct_feats, pet_feats_real)):
            bg_mask, fg_mask = _class_masks_at_scale(mask, ct.shape[-2:])
            for class_idx, class_mask in enumerate((bg_mask, fg_mask)):
                ct_desc, ct_valid = _masked_average_pool_2d(ct, class_mask)
                pet_desc, pet_valid = _masked_average_pool_2d(pet, class_mask)
                descriptors[class_idx]["ct"].append(ct_desc)
                descriptors[class_idx]["pet"].append(pet_desc)
                valids[class_idx].append(ct_valid & pet_valid)

        fg_present = mask.flatten(1).sum(dim=1) > EPS
        bg_present = (1.0 - mask.float()).flatten(1).sum(dim=1) > EPS
        semantic_presence = {0: bg_present, 1: fg_present}

        accepted = {}
        for class_idx in range(2):
            common_valid = semantic_presence[class_idx].clone()
            for v in valids[class_idx]:
                common_valid = common_valid & v
            accepted_count = int(common_valid.sum().item())
            accepted[CLASS_NAMES[class_idx]] = accepted_count
            if accepted_count == 0:
                continue
            for scale_idx in range(self.num_scales):
                self._epoch_cache[class_idx]["ct"][scale_idx].append(
                    descriptors[class_idx]["ct"][scale_idx][common_valid]
                    .detach().to(device="cpu", dtype=torch.float32).contiguous()
                )
                self._epoch_cache[class_idx]["pet"][scale_idx].append(
                    descriptors[class_idx]["pet"][scale_idx][common_valid]
                    .detach().to(device="cpu", dtype=torch.float32).contiguous()
                )
            self._collected_records += accepted_count
        self._collect_calls += 1
        return {
            "batch_size": b,
            "accepted_background": int(accepted.get("background", 0)),
            "accepted_foreground": int(accepted.get("foreground", 0)),
            "collect_calls": int(self._collect_calls),
            "collected_records": int(self._collected_records),
        }

    # ------------------------------------------------------------------
    # Matching helpers for optional paired EMA
    # ------------------------------------------------------------------

    @staticmethod
    def _optimal_pairs(cost: torch.Tensor) -> List[Tuple[int, int]]:
        if cost.ndim != 2:
            raise ValueError("cost must be 2D")
        n_rows, n_cols = cost.shape
        if n_rows == 0 or n_cols == 0:
            return []
        transposed = False
        work = cost
        if n_rows > n_cols:
            work = cost.t()
            transposed = True
        r, c = work.shape
        if c > 16:
            pairs = []
            used_cols = set()
            for i in range(r):
                candidates = [j for j in range(c) if j not in used_cols]
                j = min(candidates, key=lambda z: (float(work[i, z]), z))
                used_cols.add(j)
                pairs.append((i, j))
        else:
            states: Dict[int, Tuple[float, Tuple[int, ...]]] = {0: (0.0, tuple())}
            for i in range(r):
                next_states: Dict[int, Tuple[float, Tuple[int, ...]]] = {}
                for mask, (acc_cost, chosen) in states.items():
                    for j in range(c):
                        if mask & (1 << j):
                            continue
                        new_mask = mask | (1 << j)
                        new_cost = acc_cost + float(work[i, j].item())
                        new_chosen = chosen + (j,)
                        old = next_states.get(new_mask)
                        if old is None or (new_cost, new_chosen) < old:
                            next_states[new_mask] = (new_cost, new_chosen)
                states = next_states
            _, best_cols = min(states.values(), key=lambda item: (item[0], item[1]))
            pairs = [(i, int(best_cols[i])) for i in range(r)]
        if transposed:
            return [(j, i) for i, j in pairs]
        return pairs

    @torch.no_grad()
    def _apply_direct_update(self, new_keys, new_values, new_ready, new_count) -> Dict:
        for s in range(self.num_scales):
            key_buf = getattr(self, f"ct_keys_s{s + 1}")
            val_buf = getattr(self, f"pet_values_s{s + 1}")
            key_buf.copy_(new_keys[s].to(key_buf.device, dtype=key_buf.dtype))
            val_buf.copy_(new_values[s].to(val_buf.device, dtype=val_buf.dtype))
        self.prototype_ready.copy_(new_ready.to(self.prototype_ready.device))
        self.prototype_count.copy_(new_count.to(self.prototype_count.device))
        return {"mode": "direct", "matches": {}}

    @torch.no_grad()
    def _apply_matched_ema_update(self, new_keys, new_values, new_ready, new_count) -> Dict:
        momentum = float(self.config.ema_momentum)
        report = {"mode": "matched_ema", "matches": {}}
        out_keys = [getattr(self, f"ct_keys_s{s + 1}").detach().float().cpu().clone() for s in range(self.num_scales)]
        out_values = [getattr(self, f"pet_values_s{s + 1}").detach().float().cpu().clone() for s in range(self.num_scales)]
        out_ready = self.prototype_ready.detach().cpu().clone()
        out_count = self.prototype_count.detach().cpu().clone()
        old_build = out_keys[self.build_stage_idx]
        new_build = new_keys[self.build_stage_idx]
        for class_idx, class_name in enumerate(CLASS_NAMES):
            old_slots = torch.nonzero(out_ready[class_idx], as_tuple=False).flatten().long()
            new_slots = torch.nonzero(new_ready[class_idx], as_tuple=False).flatten().long()
            if new_slots.numel() == 0:
                report["matches"][class_name] = []
                continue
            class_pairs = []
            used_new = set()
            used_old = set()
            if old_slots.numel() > 0:
                cost = _pairwise_cosine_distance(old_build[class_idx, old_slots], new_build[class_idx, new_slots])
                local_pairs = self._optimal_pairs(cost)
                for old_local, new_local in local_pairs:
                    old_slot = int(old_slots[old_local].item())
                    new_slot = int(new_slots[new_local].item())
                    used_old.add(old_slot)
                    used_new.add(new_slot)
                    for s in range(self.num_scales):
                        mixed_key = momentum * out_keys[s][class_idx, old_slot] + (1.0 - momentum) * new_keys[s][class_idx, new_slot]
                        out_keys[s][class_idx, old_slot] = F.normalize(mixed_key, dim=0, eps=EPS)
                        out_values[s][class_idx, old_slot] = momentum * out_values[s][class_idx, old_slot] + (1.0 - momentum) * new_values[s][class_idx, new_slot]
                    out_ready[class_idx, old_slot] = True
                    out_count[class_idx, old_slot] = new_count[class_idx, new_slot]
                    class_pairs.append({"old_slot": old_slot, "new_slot": new_slot, "cosine_distance": float(cost[old_local, new_local])})
            free_slots = [s for s in range(self.num_clusters) if s not in used_old and (not bool(out_ready[class_idx, s]) or s in used_old)]
            if len(free_slots) < (len(new_slots) - len(used_new)):
                free_slots.extend(s for s in range(self.num_clusters) if s not in used_old and s not in free_slots)
            for new_slot in [int(v.item()) for v in new_slots if int(v.item()) not in used_new]:
                if not free_slots:
                    break
                target_slot = free_slots.pop(0)
                for s in range(self.num_scales):
                    out_keys[s][class_idx, target_slot] = new_keys[s][class_idx, new_slot]
                    out_values[s][class_idx, target_slot] = new_values[s][class_idx, new_slot]
                out_ready[class_idx, target_slot] = True
                out_count[class_idx, target_slot] = new_count[class_idx, new_slot]
                class_pairs.append({"old_slot": None, "new_slot": new_slot, "assigned_slot": target_slot, "cosine_distance": None})
            report["matches"][class_name] = class_pairs
        for s in range(self.num_scales):
            key_buf = getattr(self, f"ct_keys_s{s + 1}")
            val_buf = getattr(self, f"pet_values_s{s + 1}")
            key_buf.copy_(out_keys[s].to(key_buf.device, dtype=key_buf.dtype))
            val_buf.copy_(out_values[s].to(val_buf.device, dtype=val_buf.dtype))
        self.prototype_ready.copy_(out_ready.to(self.prototype_ready.device))
        self.prototype_count.copy_(out_count.to(self.prototype_count.device))
        return report

    @torch.no_grad()
    def _apply_fedmepd_ema_update(self, new_keys, new_values, new_ready, new_count) -> Dict:
        momentum = float(self.config.ema_momentum)
        # First-time bank init: no persistent ready slot -> direct init, never 0.999*0+0.001*current
        if not self.prototype_ready.any():
            # direct copy, but report as fedmepd_ema_init
            for s in range(self.num_scales):
                key_buf = getattr(self, f"ct_keys_s{s + 1}")
                val_buf = getattr(self, f"pet_values_s{s + 1}")
                key_buf.copy_(new_keys[s].to(key_buf.device, dtype=key_buf.dtype))
                val_buf.copy_(new_values[s].to(val_buf.device, dtype=val_buf.dtype))
            self.prototype_ready.copy_(new_ready.to(self.prototype_ready.device))
            self.prototype_count.copy_(new_count.to(self.prototype_count.device))
            # compute diversities after init
            diversities = {}
            for class_idx, class_name in enumerate(CLASS_NAMES):
                s4_keys = getattr(self, f"ct_keys_s{self.build_stage_idx + 1}")[class_idx]  # [K,C]
                ready = self.prototype_ready[class_idx]
                n_ready = int(ready.sum().item())
                if n_ready < 2:
                    div = 0.0
                else:
                    keys_ready = s4_keys[ready].float()
                    # already L2 normalized, but re-normalize for safety
                    keys_ready = F.normalize(keys_ready, p=2, dim=1, eps=EPS)
                    cos_mat = keys_ready @ keys_ready.t()
                    dist_mat = 1.0 - cos_mat
                    # sum upper triangle
                    triu = torch.triu(dist_mat, diagonal=1)
                    div = float(triu.sum().item() / (n_ready * (n_ready - 1) / 2))
                diversities[class_name] = div
            return {
                "mode": "fedmepd_ema_init",
                "momentum": momentum,
                "matches": {"background": [], "foreground": []},
                "mean_matching_cosine_distance": 0.0,
                "max_matching_cosine_distance": 0.0,
                "duplicate_current_match_count": 0,
                "ct_key_update_norm": 0.0,
                "pet_value_update_norm": 0.0,
                "prototype_diversity_background": diversities.get("background", 0.0),
                "prototype_diversity_foreground": diversities.get("foreground", 0.0),
            }

        out_keys = [getattr(self, f"ct_keys_s{s + 1}").detach().float().cpu().clone() for s in range(self.num_scales)]
        out_values = [getattr(self, f"pet_values_s{s + 1}").detach().float().cpu().clone() for s in range(self.num_scales)]
        out_ready = self.prototype_ready.detach().cpu().clone()
        out_count = self.prototype_count.detach().cpu().clone()
        old_build = out_keys[self.build_stage_idx]
        new_build = new_keys[self.build_stage_idx]

        report_matches: Dict[str, List[Dict]] = {"background": [], "foreground": []}
        all_distances: List[float] = []
        ct_update_norms: List[float] = []
        pet_update_norms: List[float] = []
        duplicate_current_match_count = 0

        for class_idx, class_name in enumerate(CLASS_NAMES):
            old_slots = torch.nonzero(out_ready[class_idx], as_tuple=False).flatten().long()
            new_slots = torch.nonzero(new_ready[class_idx], as_tuple=False).flatten().long()
            if new_slots.numel() == 0:
                report_matches[class_name] = []
                # keep optional flag for no_current logging
                continue
            if old_slots.numel() == 0:
                # No old anchor for this class yet -> direct init empty slots from current centroids
                free_slots = [int(s) for s in range(self.num_clusters) if not bool(out_ready[class_idx, s])]
                # use current centroids in order
                for idx, new_slot in enumerate([int(v.item()) for v in new_slots]):
                    if idx >= len(free_slots):
                        break
                    target = free_slots[idx]
                    for s in range(self.num_scales):
                        out_keys[s][class_idx, target] = new_keys[s][class_idx, new_slot]
                        out_values[s][class_idx, target] = new_values[s][class_idx, new_slot]
                    out_ready[class_idx, target] = True
                    out_count[class_idx, target] = new_count[class_idx, new_slot]
                report_matches[class_name] = []
                continue

            # S4 cosine distance matrix old x current
            old_keys = old_build[class_idx, old_slots]  # [n_old, C]
            cur_keys = new_build[class_idx, new_slots]  # [n_new, C]
            cost = _pairwise_cosine_distance(old_keys, cur_keys)  # [n_old, n_new]
            nearest_local = cost.argmin(dim=1)  # [n_old]
            # track chosen current slot values for duplicate count and for free-slot reuse
            chosen_current_slots: List[int] = []
            for i in range(old_slots.numel()):
                old_slot = int(old_slots[i].item())
                new_local = int(nearest_local[i].item())
                new_slot = int(new_slots[new_local].item())
                dist = float(cost[i, new_local].item())
                chosen_current_slots.append(new_slot)
                all_distances.append(dist)
                # snapshot old for norm computation
                old_key_norms_before = [out_keys[s][class_idx, old_slot].clone() for s in range(self.num_scales)]
                old_val_before = [out_values[s][class_idx, old_slot].clone() for s in range(self.num_scales)]
                # Paired EMA across all 4 scales with same mapping
                for s in range(self.num_scales):
                    mixed_key = momentum * out_keys[s][class_idx, old_slot] + (1.0 - momentum) * new_keys[s][class_idx, new_slot]
                    out_keys[s][class_idx, old_slot] = F.normalize(mixed_key, dim=0, eps=EPS)
                    out_values[s][class_idx, old_slot] = momentum * out_values[s][class_idx, old_slot] + (1.0 - momentum) * new_values[s][class_idx, new_slot]
                out_count[class_idx, old_slot] = new_count[class_idx, new_slot]
                # update norms (use S4 as representative, but compute per scale mean)
                # CT
                for s in range(self.num_scales):
                    ct_delta = float((out_keys[s][class_idx, old_slot] - old_key_norms_before[s]).norm().item())
                    pet_delta = float((out_values[s][class_idx, old_slot] - old_val_before[s]).norm().item())
                    ct_update_norms.append(ct_delta)
                    pet_update_norms.append(pet_delta)
                report_matches[class_name].append({
                    "old_slot": old_slot,
                    "current_slot": new_slot,
                    "cosine_distance": dist,
                })

            # duplicate count: number of extra matches beyond unique current
            unique_chosen = len(set(chosen_current_slots))
            duplicate_current_match_count += len(chosen_current_slots) - unique_chosen

            # Fill remaining empty slots with still-unused current centroids (no EMA, direct init)
            free_slots = [int(s) for s in range(self.num_clusters) if not bool(out_ready[class_idx, s])]
            # unused = new_slots not in chosen_current_slots
            chosen_set = set(chosen_current_slots)
            unused_new = [int(v.item()) for v in new_slots if int(v.item()) not in chosen_set]
            for target, new_slot in zip(free_slots, unused_new):
                for s in range(self.num_scales):
                    out_keys[s][class_idx, target] = new_keys[s][class_idx, new_slot]
                    out_values[s][class_idx, target] = new_values[s][class_idx, new_slot]
                out_ready[class_idx, target] = True
                out_count[class_idx, target] = new_count[class_idx, new_slot]

        # commit buffers
        for s in range(self.num_scales):
            key_buf = getattr(self, f"ct_keys_s{s + 1}")
            val_buf = getattr(self, f"pet_values_s{s + 1}")
            key_buf.copy_(out_keys[s].to(key_buf.device, dtype=key_buf.dtype))
            val_buf.copy_(out_values[s].to(val_buf.device, dtype=val_buf.dtype))
        self.prototype_ready.copy_(out_ready.to(self.prototype_ready.device))
        self.prototype_count.copy_(out_count.to(self.prototype_count.device))

        # diversity after update
        diversities = {}
        for class_idx, class_name in enumerate(CLASS_NAMES):
            s4_keys = getattr(self, f"ct_keys_s{self.build_stage_idx + 1}")[class_idx]
            ready = self.prototype_ready[class_idx]
            n_ready = int(ready.sum().item())
            if n_ready < 2:
                div = 0.0
            else:
                keys_ready = s4_keys[ready].float()
                keys_ready = F.normalize(keys_ready, p=2, dim=1, eps=EPS)
                cos_mat = keys_ready @ keys_ready.t()
                dist_mat = 1.0 - cos_mat
                triu = torch.triu(dist_mat, diagonal=1)
                div = float(triu.sum().item() / (n_ready * (n_ready - 1) / 2))
            diversities[class_name] = div

        mean_dist = float(sum(all_distances) / len(all_distances)) if all_distances else 0.0
        max_dist = float(max(all_distances)) if all_distances else 0.0
        ct_norm = float(sum(ct_update_norms) / len(ct_update_norms)) if ct_update_norms else 0.0
        pet_norm = float(sum(pet_update_norms) / len(pet_update_norms)) if pet_update_norms else 0.0

        return {
            "mode": "fedmepd_ema",
            "momentum": momentum,
            "matches": report_matches,
            "mean_matching_cosine_distance": mean_dist,
            "max_matching_cosine_distance": max_dist,
            "duplicate_current_match_count": int(duplicate_current_match_count),
            "ct_key_update_norm": ct_norm,
            "pet_value_update_norm": pet_norm,
            "prototype_diversity_background": diversities.get("background", 0.0),
            "prototype_diversity_foreground": diversities.get("foreground", 0.0),
        }

    # ------------------------------------------------------------------
    # Epoch bank finalization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def finalize_epoch(self, epoch: Optional[int] = None) -> Dict:
        report: Dict = {
            "epoch": None if epoch is None else int(epoch),
            "config": self.export_config(),
            "collect_calls": int(self._collect_calls),
            "collected_records": int(self._collected_records),
            "bank_version_before": int(self.bank_version.item()),
            "classes": {},
        }
        new_keys = [torch.zeros(2, self.num_clusters, c, dtype=torch.float32) for c in self.channels]
        new_values = [torch.zeros(2, self.num_clusters, c, dtype=torch.float32) for c in self.channels]
        new_ready = torch.zeros(2, self.num_clusters, dtype=torch.bool)
        new_count = torch.zeros(2, self.num_clusters, dtype=torch.long)
        any_candidate = False
        for class_idx, class_name in enumerate(CLASS_NAMES):
            build_ct_raw = self._concat_cache(class_idx, "ct", self.build_stage_idx)
            class_report: Dict = {
                "num_candidates": int(build_ct_raw.shape[0]),
                "build_stage": int(self.config.build_stage),
                "clustering": None,
                "filtering": None,
            }
            if build_ct_raw.shape[0] == 0:
                report["classes"][class_name] = class_report
                continue

            # Pre-filter: NaN / Inf / zero-norm S4 CT descriptors. The SAME
            # valid indices are applied to S1-S4 CT AND PET descriptors so the
            # cross-scale row alignment is preserved.
            valid_mask = (
                torch.isfinite(build_ct_raw).all(dim=1)
                & (build_ct_raw.norm(dim=1) > EPS)
            )
            num_filtered = int((~valid_mask).sum().item())
            if num_filtered > 0:
                class_report["prefilter_discarded"] = num_filtered
            if not bool(valid_mask.any()):
                class_report["status"] = "all_candidates_filtered"
                report["classes"][class_name] = class_report
                continue
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten().long()
            build_ct = build_ct_raw[valid_indices]
            filtered_caches: Dict[int, Dict[str, torch.Tensor]] = {}
            for s in range(self.num_scales):
                ct_all_raw = self._concat_cache(class_idx, "ct", s)
                pet_all_raw = self._concat_cache(class_idx, "pet", s)
                if ct_all_raw.shape[0] != build_ct_raw.shape[0] or pet_all_raw.shape[0] != build_ct_raw.shape[0]:
                    raise RuntimeError(
                        f"Cross-scale paired candidate misalignment: class={class_name}, "
                        f"scale={s+1}, build={build_ct_raw.shape[0]}, "
                        f"ct={ct_all_raw.shape[0]}, pet={pet_all_raw.shape[0]}"
                    )
                filtered_caches[s] = {
                    "ct": ct_all_raw[valid_indices],
                    "pet": pet_all_raw[valid_indices],
                }

            any_candidate = True
            labels, centers, kmeans_report = deterministic_spherical_kmeans(
                build_ct, num_clusters=self.num_clusters, max_iter=self.config.cluster_max_iter,
            )
            k_eff = int(centers.shape[0])
            kept_by_cluster, filter_report = cosine_cluster_outlier_filter(
                build_ct, labels, num_clusters=k_eff, discard_rate=self.config.outlier_discard_rate,
            )
            singleton_warning = bool(filter_report.get("singleton_cluster_warning", False))

            # Paired cross-scale prototype build: identical kept members for
            # CT keys (L2 normalized) and PET values (raw magnitude).
            for s in range(self.num_scales):
                ct_all = filtered_caches[s]["ct"]
                pet_all = filtered_caches[s]["pet"]
                for cluster_idx in range(k_eff):
                    kept = kept_by_cluster.get(cluster_idx)
                    if kept is None or kept.numel() == 0:
                        continue
                    ct_key = ct_all[kept].mean(dim=0)
                    pet_value = pet_all[kept].mean(dim=0)
                    new_keys[s][class_idx, cluster_idx] = F.normalize(ct_key.float(), dim=0, eps=EPS)
                    new_values[s][class_idx, cluster_idx] = pet_value.float()
                    if s == self.build_stage_idx:
                        new_ready[class_idx, cluster_idx] = True
                        new_count[class_idx, cluster_idx] = int(kept.numel())

            class_report["clustering"] = kmeans_report
            class_report["filtering"] = filter_report
            class_report["effective_clusters"] = int(k_eff)
            class_report["cluster_count_before_filter"] = list(kmeans_report["cluster_counts"])
            class_report["cluster_count_after_filter"] = [int(new_count[class_idx, j].item()) for j in range(self.num_clusters)]
            class_report["singleton_cluster_warning"] = singleton_warning
            report["classes"][class_name] = class_report

        if not any_candidate or not bool(new_ready.any()):
            report["status"] = "bank_unchanged_no_valid_candidates"
            self.reset_epoch_cache()
            return report
        if self.config.bank_update_mode == "direct":
            update_report = self._apply_direct_update(new_keys, new_values, new_ready, new_count)
        elif self.config.bank_update_mode == "matched_ema":
            update_report = self._apply_matched_ema_update(new_keys, new_values, new_ready, new_count)
        elif self.config.bank_update_mode == "fedmepd_ema":
            update_report = self._apply_fedmepd_ema_update(new_keys, new_values, new_ready, new_count)
        else:
            raise RuntimeError(f"Unsupported bank_update_mode={self.config.bank_update_mode!r}")
        self.bank_version.add_(1)
        report["status"] = "bank_updated"
        report["update"] = update_report
        report["bank_version_after"] = int(self.bank_version.item())
        report["ready_count"] = int(self.prototype_ready.sum().item())
        report["total_slots"] = int(self.prototype_ready.numel())
        report["prototype_count"] = self.prototype_count.detach().cpu().tolist()
        # Expose FedMEPD monitoring fields at top level when present
        if isinstance(update_report, dict):
            for extra_key in (
                "prototype_diversity_background",
                "prototype_diversity_foreground",
                "mean_matching_cosine_distance",
                "max_matching_cosine_distance",
                "duplicate_current_match_count",
                "ct_key_update_norm",
                "pet_value_update_norm",
            ):
                if extra_key in update_report:
                    report[extra_key] = update_report[extra_key]
        self.reset_epoch_cache()
        return report

    # ------------------------------------------------------------------
    # Retrieval (cosine soft; CT detached inside; entropy stats)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        ct_feats: Sequence[torch.Tensor],
        return_attention: bool = False,
    ) -> Dict:
        self._validate_features(ct_feats, None)
        ready_flat = self.prototype_ready.flatten()
        pet_proxy: List[torch.Tensor] = []
        attentions: List[torch.Tensor] = []
        entropy_list: List[float] = []
        norm_entropy_list: List[float] = []
        any_ready = bool(ready_flat.any())
        for s, ct in enumerate(ct_feats):
            keys = getattr(self, f"ct_keys_s{s + 1}").reshape(2 * self.num_clusters, self.channels[s])
            values = getattr(self, f"pet_values_s{s + 1}").reshape(2 * self.num_clusters, self.channels[s])
            keys = keys.to(device=ct.device, dtype=ct.dtype)
            values = values.to(device=ct.device, dtype=ct.dtype)
            ready = ready_flat.to(device=ct.device)
            retrieved, attention = self.attention[s](ct, keys, values, ready)
            pet_proxy.append(retrieved)
            if return_attention:
                attentions.append(attention)
            with torch.no_grad():
                if not any_ready:
                    entropy_list.append(0.0)
                    norm_entropy_list.append(0.0)
                else:
                    p = attention.detach().float().clamp_min(1e-12)
                    ent = float(-(p * p.log()).sum(dim=-1).mean().item())
                    k_ready = int(ready.sum().item())
                    max_ent = math.log(k_ready) if k_ready > 1 else 0.0
                    if max_ent <= 0.0:
                        norm_ent = 0.0
                    else:
                        norm_ent = min(1.0, max(0.0, ent / max_ent))
                    entropy_list.append(ent)
                    norm_entropy_list.append(norm_ent)
        return {
            "pet_proxy": pet_proxy,
            "attention": attentions if return_attention else None,
            "bank_ready": self.bank_ready,
            "bank_version": int(self.bank_version.item()),
            "attention_entropy": entropy_list,
            "normalized_attention_entropy": norm_entropy_list,
            # Kept for interface compatibility only; never enters any compute.
            "ct_reference": None,
        }

    # ------------------------------------------------------------------
    # PET multi-positive prototype contrastive loss
    # ------------------------------------------------------------------

    def compute_pet_prototype_contrastive_loss(
        self,
        pet_real_feats: Sequence[torch.Tensor],
        mask: torch.Tensor,
    ) -> Dict:
        """
        Multi-positive prototype contrastive loss on real PET descriptors.

        Gradient boundary: pet_real_feats keep grad (PET encoder updates);
        the prototype bank is detached; CT/retrieval/personalization get none.
        """
        self._validate_features(pet_real_feats, None)
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("mask must be [B,1,H,W]")
        if not self.bank_ready:
            return _zero_loss_result(pet_real_feats[0])
        tau = float(self.config.proto_temperature)
        weight = float(self.config.proto_contrastive_weight)

        fg_present = mask.flatten(1).sum(dim=1) > EPS
        bg_present = (1.0 - mask.float()).flatten(1).sum(dim=1) > EPS
        semantic_presence = {0: bg_present, 1: fg_present}

        group_losses: List[torch.Tensor] = []
        per_scale: Dict[str, float] = {}
        details: Dict[str, int] = {}

        for s, pet_feat in enumerate(pet_real_feats):
            bg_mask, fg_mask = _class_masks_at_scale(mask, pet_feat.shape[-2:])
            for class_idx, class_mask in enumerate((bg_mask, fg_mask)):
                target_ready = self.prototype_ready[class_idx]
                other_ready = self.prototype_ready[1 - class_idx]
                if not bool(target_ready.any()) or not bool(other_ready.any()):
                    continue
                descriptors, desc_valid = _masked_average_pool_2d(pet_feat, class_mask)
                valid = (
                    desc_valid
                    & semantic_presence[class_idx].to(pet_feat.device)
                    & torch.isfinite(descriptors).all(dim=1)
                    & (descriptors.norm(dim=1) > EPS)
                )
                if not bool(valid.any()):
                    continue

                pet_bank = getattr(self, f"pet_values_s{s + 1}").detach()  # [2,K,C]
                target_protos = pet_bank[class_idx][target_ready]          # [Kt,C]
                negative_protos = pet_bank[1 - class_idx][other_ready]     # [Kn,C]
                target_protos_n = F.normalize(target_protos.float(), p=2, dim=-1, eps=EPS)
                negative_protos_n = F.normalize(negative_protos.float(), p=2, dim=-1, eps=EPS)

                sample_terms: List[torch.Tensor] = []
                valid_indices = torch.nonzero(valid, as_tuple=False).flatten().long()
                for idx in valid_indices:
                    z = descriptors[idx : idx + 1]  # [1,C], grad to PET encoder
                    z_n = F.normalize(z.float(), p=2, dim=-1, eps=EPS)
                    cos_target = (z_n @ target_protos_n.t()).squeeze(0)    # [Kt]
                    cos_negative = (z_n @ negative_protos_n.t()).squeeze(0)  # [Kn]
                    s_c = torch.logsumexp(cos_target / tau, dim=0)
                    s_o = torch.logsumexp(cos_negative / tau, dim=0)
                    lse = torch.logsumexp(torch.stack([s_c, s_o]), dim=0)
                    loss = lse - s_c
                    _finite_or_raise(f"proto_contrastive_term_s{s+1}", loss)
                    sample_terms.append(loss)
                if not sample_terms:
                    continue
                group_loss = torch.stack(sample_terms).mean()
                _finite_or_raise(f"proto_contrastive_s{s+1}_{CLASS_NAMES[class_idx]}", group_loss)
                group_losses.append(group_loss)
                per_scale[f"s{s+1}_{CLASS_NAMES[class_idx]}"] = float(group_loss.item())
                details[f"s{s+1}_{CLASS_NAMES[class_idx]}_terms"] = len(sample_terms)

        if not group_losses:
            return _zero_loss_result(pet_real_feats[0])
        loss = torch.stack(group_losses).mean()
        weighted = weight * loss
        _finite_or_raise("proto_contrastive_loss", loss)
        return {
            "loss": loss,
            "weighted_loss": weighted,
            "num_terms": len(group_losses),
            "per_scale": per_scale,
            "details": details,
        }

    # ------------------------------------------------------------------
    # FG/BG-balanced reconstruction loss
    # ------------------------------------------------------------------

    def compute_balanced_reconstruction_loss(
        self,
        pet_comp_feats: Sequence[torch.Tensor],
        pet_real_feats: Sequence[torch.Tensor],
        mask: torch.Tensor,
    ) -> Dict:
        """
        Balanced PET reconstruction: per scale, FG and BG masked MSE are
        computed separately (FG skipped when the batch has no foreground),
        averaged within the scale, then averaged over valid scales.
        Target = detached real PET. Grad flows only to retrieval +
        personalization through pet_comp.
        """
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("mask must be [B,1,H,W]")
        self._validate_features(pet_comp_feats, None)
        self._validate_features(pet_real_feats, None)
        if not self.bank_ready:
            return _zero_loss_result(pet_comp_feats[0])
        if len(pet_comp_feats) != len(pet_real_feats):
            raise ValueError("pet_comp/pet_real scale count mismatch")

        fg_present = bool((mask.flatten(1).sum(dim=1) > EPS).any().item())
        scale_losses: List[torch.Tensor] = []
        per_scale: Dict[str, float] = {}
        for s, (pet_comp, pet_real) in enumerate(zip(pet_comp_feats, pet_real_feats)):
            if pet_comp.shape != pet_real.shape:
                raise ValueError(f"pet_comp/pet_real shape mismatch at scale {s+1}")
            _finite_or_raise(f"pet_comp_s{s+1}", pet_comp)
            _finite_or_raise(f"pet_real_s{s+1}", pet_real)
            target = pet_real.detach().float()
            diff2 = (pet_comp.float() - target).pow(2)  # [B,C,H,W]
            bg_mask, fg_mask = _class_masks_at_scale(mask, pet_comp.shape[-2:])
            class_terms: List[torch.Tensor] = []
            for class_idx, class_mask in enumerate((bg_mask, fg_mask)):
                if class_idx == 1 and not fg_present:
                    continue  # no foreground in this batch: skip FG term
                mask_sum = float(class_mask.sum().item())
                if mask_sum <= EPS:
                    continue
                c = pet_comp.shape[1]
                masked = diff2 * class_mask.float()  # broadcast [B,C,H,W]
                loss_cls = masked.sum() / (c * mask_sum + EPS)
                _finite_or_raise(f"recon_s{s+1}_{CLASS_NAMES[class_idx]}", loss_cls)
                class_terms.append(loss_cls)
            if not class_terms:
                continue
            scale_loss = torch.stack(class_terms).mean()
            scale_losses.append(scale_loss)
            per_scale[f"s{s+1}"] = float(scale_loss.item())
        if not scale_losses:
            return _zero_loss_result(pet_comp_feats[0])
        loss = torch.stack(scale_losses).mean()
        weighted = float(self.config.reconstruction_weight) * loss
        _finite_or_raise("recon_loss", loss)
        return {
            "loss": loss,
            "weighted_loss": weighted,
            "num_terms": len(scale_losses),
            "per_scale": per_scale,
            "details": {},
        }

    # ------------------------------------------------------------------
    # Missing prediction (strict, no PET argument)
    # ------------------------------------------------------------------

    def recover_missing(
        self,
        ct_feats: Sequence[torch.Tensor],
        return_attention: bool = False,
    ) -> Tuple[List[torch.Tensor], Dict]:
        retrieval = self.retrieve(ct_feats, return_attention=return_attention)
        pet_proxy = retrieval["pet_proxy"]
        zero_stats = {
            "gamma_mean": 0.0, "gamma_std": 0.0, "gamma_abs_mean": 0.0,
            "beta_mean": 0.0, "beta_std": 0.0, "beta_abs_mean": 0.0,
            "pet_proto_norm": 0.0, "pet_comp_norm": 0.0,
        }
        if not self.bank_ready:
            # Epoch-1 cold start: strictly zero compensated PET.
            pet_comp = [torch.zeros_like(c) for c in ct_feats]
            for t in pet_comp:
                _finite_or_raise("pet_comp_fallback", t)
            aux = {
                "pet_proxy": pet_proxy,
                "ct_reference": None,
                "attention": retrieval["attention"],
                "bank_ready": False,
                "bank_version": int(self.bank_version.item()),
                "attention_entropy": retrieval["attention_entropy"],
                "normalized_attention_entropy": retrieval["normalized_attention_entropy"],
                **zero_stats,
            }
            return pet_comp, aux

        if bool(self.config.spatial_affine):
            pet_comp, stats = self.personalization(ct_feats, pet_proxy)
        else:
            # Personalization ablation: P_comp = P_proto.
            pet_comp = list(pet_proxy)
            with torch.no_grad():
                stats = dict(zero_stats)
                stats["pet_proto_norm"] = float(
                    sum(p.detach().float().pow(2).mean().sqrt().item() for p in pet_proxy) / len(pet_proxy)
                )
                stats["pet_comp_norm"] = stats["pet_proto_norm"]

        for t in pet_comp:
            _finite_or_raise("pet_comp", t)

        aux = {
            "pet_proxy": pet_proxy,
            "ct_reference": None,
            "attention": retrieval["attention"],
            "bank_ready": True,
            "bank_version": int(self.bank_version.item()),
            "attention_entropy": retrieval["attention_entropy"],
            "normalized_attention_entropy": retrieval["normalized_attention_entropy"],
            **stats,
        }
        return pet_comp, aux

    def forward(
        self,
        ct_feats: Sequence[torch.Tensor],
        pet_feats_real: Optional[Sequence[torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        mode: str = "missing",
        collect_candidates: Optional[bool] = None,
        return_attention: bool = False,
    ) -> Dict:
        """Compat wrapper; the joint model calls the granular APIs directly."""
        if mode not in {"full", "missing"}:
            raise ValueError("mode must be 'full' or 'missing'")
        self._validate_features(ct_feats, pet_feats_real if pet_feats_real is not None else None)

        should_collect = (
            self.config.collect_candidates_during_training
            if collect_candidates is None
            else bool(collect_candidates)
        )
        collect_report = None
        if self.training and should_collect and pet_feats_real is not None and mask is not None:
            collect_report = self.collect_candidates(ct_feats, pet_feats_real, mask)

        zero = _zero_loss_result(ct_feats[0])
        if mode == "full":
            proto = zero
            recon = zero
            if self.training and pet_feats_real is not None and mask is not None:
                proto = self.compute_pet_prototype_contrastive_loss(pet_feats_real, mask)
            pred = {
                "pet_output": None,
                "pet_proxy": None,
                "ct_reference": None,
                "attention": None,
                "bank_ready": self.bank_ready,
                "bank_version": int(self.bank_version.item()),
                "prototype_contrastive_loss": proto["loss"],
                "prototype_contrastive_loss_weighted": proto["weighted_loss"],
                "prototype_contrastive_num_terms": proto["num_terms"],
                "reconstruction_loss": recon["loss"],
                "reconstruction_loss_weighted": recon["weighted_loss"],
                "reconstruction_num_terms": recon["num_terms"],
                "collect_report": collect_report,
            }
            return pred

        pet_comp, aux = self.recover_missing(ct_feats, return_attention=return_attention)
        proto = zero
        recon = zero
        if self.training and pet_feats_real is not None and mask is not None:
            proto = self.compute_pet_prototype_contrastive_loss(pet_feats_real, mask)
            recon = self.compute_balanced_reconstruction_loss(pet_comp, pet_feats_real, mask)
        pred = {
            "pet_output": pet_comp,
            **aux,
            "prototype_contrastive_loss": proto["loss"],
            "prototype_contrastive_loss_weighted": proto["weighted_loss"],
            "prototype_contrastive_num_terms": proto["num_terms"],
            "reconstruction_loss": recon["loss"],
            "reconstruction_loss_weighted": recon["weighted_loss"],
            "reconstruction_num_terms": recon["num_terms"],
            "collect_report": collect_report,
        }
        return pred


# -----------------------------------------------------------------------------
# Minimal self-check
# -----------------------------------------------------------------------------


def _self_check() -> None:
    torch.manual_seed(7)
    channels = (8, 12, 16, 20)
    shapes = ((16, 16), (8, 8), (4, 4), (2, 2))
    module = PairedSemanticPrototypeImputation(
        channels=channels, num_clusters=3, build_stage=4, bank_update_mode="direct",
        retrieval_temperature=0.1, proto_contrastive_weight=0.01,
        proto_temperature=0.02, reconstruction_weight=0.1,
    )
    assert not module.bank_ready
    assert int(module.bank_version.item()) == 0
    module.train()
    for _ in range(3):
        ct = [torch.randn(4, c, h, w) for c, (h, w) in zip(channels, shapes)]
        pet = [torch.randn(4, c, h, w) for c, (h, w) in zip(channels, shapes)]
        mask = torch.zeros(4, 1, 64, 64)
        mask[:, :, 18:46, 20:44] = 1.0
        module.collect_candidates(ct, pet, mask)
    report = module.finalize_epoch(epoch=1)
    assert report["status"] == "bank_updated"
    assert module.bank_ready

    b = 2
    ct = [torch.randn(b, c, h, w, requires_grad=True) for c, (h, w) in zip(channels, shapes)]
    pet_a = [torch.randn(b, c, h, w, requires_grad=True) for c, (h, w) in zip(channels, shapes)]
    mask = torch.zeros(b, 1, 64, 64)
    mask[:, :, 16:48, 16:48] = 1.0

    # Not-ready fallback (fresh module)
    empty = PairedSemanticPrototypeImputation(channels=channels, num_clusters=3, build_stage=4)
    empty.eval()
    pet_zero, aux_zero = empty.recover_missing(ct)
    for s in range(4):
        assert bool((pet_zero[s] == 0).all())
    assert aux_zero["bank_ready"] is False
    assert float(empty.compute_pet_prototype_contrastive_loss(pet_a, mask)["loss"]) == 0.0
    assert float(empty.compute_balanced_reconstruction_loss(pet_zero, pet_a, mask)["loss"]) == 0.0

    # Ready: shapes + default no attention map
    module.eval()
    pet_comp, aux = module.recover_missing(ct)
    for s in range(4):
        assert pet_comp[s].shape == ct[s].shape
        assert bool(torch.isfinite(pet_comp[s]).all())
    assert aux["attention"] is None
    pet_comp2, aux2 = module.recover_missing(ct, return_attention=True)
    assert aux2["attention"] is not None and len(aux2["attention"]) == 4
    for a in aux2["attention"]:
        assert bool(torch.isfinite(a).all())
        assert torch.allclose(a.float().sum(dim=-1), torch.ones(a.shape[0], a.shape[1]), atol=1e-5)
    for v in aux2["normalized_attention_entropy"]:
        assert 0.0 <= v <= 1.0

    # CT detach: no grad to CT through Module-1
    module.train()
    ct_g = [c.clone().detach().requires_grad_(True) for c in ct]
    pet_comp_g, _ = module.recover_missing(ct_g)
    sum(x.float().pow(2).mean() for x in pet_comp_g).backward()
    assert all(c.grad is None for c in ct_g), "CT must be detached inside Module-1"

    # Direct affine formula: P_comp == gamma * P_proto + beta
    module.eval()
    ct1 = [torch.randn(1, c, h, w) for c, (h, w) in zip(channels, shapes)]
    pet_proxy = module.retrieve(ct1)["pet_proxy"]
    pet_comp3, _ = module.recover_missing(ct1)
    cond = F.normalize(ct1[0].detach().float(), p=2, dim=1, eps=EPS).to(ct1[0].dtype)
    trunk_out = module.personalization.trunks[0](cond)
    gamma = module.personalization.gamma_heads[0](trunk_out)
    beta = module.personalization.beta_heads[0](trunk_out)
    assert torch.allclose(pet_comp3[0], gamma * pet_proxy[0] + beta, atol=1e-5)

    # Contrastive + reconstruction losses finite, >= 0
    module.train()
    proto = module.compute_pet_prototype_contrastive_loss(pet_a, mask)
    assert proto["num_terms"] > 0 and float(proto["loss"].item()) >= 0.0
    recon = module.compute_balanced_reconstruction_loss(pet_comp, pet_a, mask)
    assert recon["num_terms"] > 0 and float(recon["loss"].item()) >= 0.0

    # Gradient boundaries
    module.zero_grad(set_to_none=True)
    for p in pet_a:
        p.grad = None
    proto["loss"].backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in pet_a)
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in ct)

    print("[SELF-CHECK] passed")


if __name__ == "__main__":
    _self_check()
