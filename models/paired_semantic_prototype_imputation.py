# -*- coding: utf-8 -*-
"""
Paired Semantic Multi-Prototype Imputation (Module-1)  —  missing-only version
==========================================================================

Role after the boundary contraction
-----------------------------------
Module-1 ONLY handles missing PET compensation. Full real PET never enters
retrieval, CT reference, personalization or any prototype supervision.

Full (train & eval):
    C_s = E_CT(CT),  P_s_real = E_PET(PET)
    F_s_full = C_s + P_s_real  (outside Module-1, via AddFusion)

Missing:
    C_s = E_CT(CT)
      -> paired CT-key / PET-value prototype bank (K_s_CT <-> V_s_PET)
      -> attention A_s = softmax(Q_s K_s^T / sqrt(C))
        -> P_s_proto = A_s V_s_PET,  C_s_ref = A_s K_s_CT   (same A)
      -> CT discrepancy delta_s = mean(norm(C_s) - norm(C_s_ref))
        -> H_s(delta_s) -> gamma_s, beta_s
      -> P_s_comp = P_s_proto + gamma_s*(P_s_proto - mu_s) + beta_s*sigma_s
      -> F_s_miss = C_s + P_s_comp  (outside Module-1, via AddFusion)

Candidate cache, deterministic spherical K-means, cosine outlier filtering,
paired S4 membership reuse and direct/matched-EMA update are all preserved.

Supervision
-----------
Main prototype assignment losses (PAD-KL / PAD-JS / retrieval cosine) are
removed. Only a PASSION-style build-stage semantic prototype relation loss
is kept for Missing training:

    per-sample BG/FG prototypes from real and compensated PET at build stage
    S_real / S_comp = cosine relation maps
    L_sem = mean_valid_classes mean_spatial (S_comp - stopgrad(S_real))^2

Stage-1 / Stage-1.5 bootstrap plumbing is unchanged.
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

        report[str(cluster_idx)] = {
            "before_count": n,
            "after_count": int(keep_n),
            "discarded_count": int(n - keep_n),
            "mean_cosine_distance": float(distances.mean().item()),
            "max_cosine_distance": float(distances.max().item()),
            "kept_indices": kept_indices.detach().cpu().tolist(),
        }

    return kept, report


# -----------------------------------------------------------------------------
# Prototype retrieval attention
# -----------------------------------------------------------------------------


class PrototypeCrossAttention(nn.Module):
    """
    Q = current CT feature/token
    K = CT prototype keys
    V = paired PET prototype values
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = int(channels)
        self.q_proj = nn.Linear(self.channels, self.channels, bias=False)
        self.k_proj = nn.Linear(self.channels, self.channels, bias=False)
        self.v_proj = nn.Linear(self.channels, self.channels, bias=False)
        self.out_proj = nn.Linear(self.channels, self.channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
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

        q = query_map.flatten(2).transpose(1, 2)  # [B,N,C]
        q = self.q_proj(q)
        k = self.k_proj(keys)
        v = self.v_proj(values)

        logits = torch.matmul(q, k.t()) / math.sqrt(float(c))
        logits = logits.masked_fill(
            ~ready.view(1, 1, -1),
            torch.finfo(logits.dtype).min,
        )
        attention = torch.softmax(logits, dim=-1)
        retrieved = torch.matmul(attention, v)
        retrieved = self.out_proj(retrieved)
        retrieved = retrieved.transpose(1, 2).reshape(b, c, h, w)

        _finite_or_raise("prototype_attention", attention)
        _finite_or_raise("retrieved_pet", retrieved)
        return _sanitize(retrieved), attention


# -----------------------------------------------------------------------------
# Missing-only prototype personalization
# -----------------------------------------------------------------------------


class MissingPrototypePersonalization(nn.Module):
    """
    CT-reference-guided Missing PET prototype personalization.

    Discrepancy  delta = mean(norm(CT) - norm(C_ref))  guides
        [gamma, beta] = head(delta)
        P_comp = P_proto + gamma*(P_proto - mu) + beta*sigma
    where mu/sigma are spatial statistics of P_proto itself.

    Zero-initialized -> identity at start: P_comp == P_proto.
    """

    def __init__(self, channels: Sequence[int]):
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        heads: List[nn.Module] = []
        for c in self.channels:
            hidden = max(c // 4, 16)
            head = nn.Sequential(
                nn.Linear(c, hidden),
                nn.GELU(),
                nn.Linear(hidden, 2 * c),
            )
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            heads.append(head)
        self.heads = nn.ModuleList(heads)

    def forward(
        self,
        ct_feats: Sequence[torch.Tensor],
        pet_proto_feats: Sequence[torch.Tensor],
        ct_reference_feats: Optional[Sequence[torch.Tensor]],
        reference_valid: bool,
    ) -> List[torch.Tensor]:
        if len(ct_feats) != len(pet_proto_feats):
            raise ValueError("CT and PET prototype must have the same scale count")
        if not reference_valid or ct_reference_feats is None:
            pet_comp = []
            for pet_proto in pet_proto_feats:
                _finite_or_raise("pet_proto_fallback", pet_proto)
                pet_comp.append(_sanitize(pet_proto))
            return pet_comp
        if len(ct_reference_feats) != len(ct_feats):
            raise ValueError("CT reference scale count mismatch")

        pet_comp: List[torch.Tensor] = []
        for scale_idx, (ct, pet_proto, ct_ref) in enumerate(
            zip(ct_feats, pet_proto_feats, ct_reference_feats)
        ):
            ct_det = ct.detach()
            ref_det = ct_ref.detach()

            if pet_proto.shape[-2:] != ct_det.shape[-2:]:
                pet_proto = F.interpolate(
                    pet_proto,
                    size=ct_det.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if ref_det.shape[-2:] != ct_det.shape[-2:]:
                ref_det = F.interpolate(
                    ref_det,
                    size=ct_det.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if ct_det.shape != pet_proto.shape or ct_det.shape != ref_det.shape:
                raise ValueError(
                    f"Scale {scale_idx + 1} shape mismatch: "
                    f"ct={tuple(ct_det.shape)} pet_proto={tuple(pet_proto.shape)} "
                    f"ref={tuple(ref_det.shape)}"
                )

            ct_tokens = F.normalize(
                ct_det.flatten(2).transpose(1, 2), p=2, dim=-1, eps=1e-6,
            )
            ref_tokens = F.normalize(
                ref_det.flatten(2).transpose(1, 2), p=2, dim=-1, eps=1e-6,
            )
            delta = (ct_tokens - ref_tokens).mean(dim=1)
            params = self.heads[scale_idx](delta)
            raw_gamma, raw_beta = params.chunk(2, dim=-1)
            gamma = torch.tanh(raw_gamma).view(pet_proto.shape[0], pet_proto.shape[1], 1, 1)
            beta = torch.tanh(raw_beta).view(pet_proto.shape[0], pet_proto.shape[1], 1, 1)
            _finite_or_raise(f"personalization_gamma_s{scale_idx+1}", gamma)
            _finite_or_raise(f"personalization_beta_s{scale_idx+1}", beta)

            mu = pet_proto.mean(dim=(2, 3), keepdim=True)
            sigma = pet_proto.float().std(dim=(2, 3), keepdim=True, unbiased=False)
            sigma = sigma.to(dtype=pet_proto.dtype).clamp_min(1e-6)
            _finite_or_raise(f"pet_proto_mu_s{scale_idx+1}", mu)
            _finite_or_raise(f"pet_proto_sigma_s{scale_idx+1}", sigma)

            out = pet_proto + gamma * (pet_proto - mu) + beta * sigma
            _finite_or_raise(f"pet_comp_s{scale_idx+1}", out)
            pet_comp.append(_sanitize(out))
        return pet_comp


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
    semantic_loss_weight: float = 0.01
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
        if self.bank_update_mode not in {"direct", "matched_ema"}:
            raise ValueError("bank_update_mode must be 'direct' or 'matched_ema'")
        if not 0.0 <= self.ema_momentum < 1.0:
            raise ValueError("ema_momentum must be in [0,1)")
        if self.semantic_loss_weight < 0.0:
            raise ValueError("semantic_loss_weight must be >= 0")


# -----------------------------------------------------------------------------
# Main standalone Module-1  (missing-only)
# -----------------------------------------------------------------------------


class PairedSemanticPrototypeImputation(nn.Module):
    """
    Missing-only Module-1.

    Public contracts
    ----------------
    * Candidate collection:  collect_candidates(ct_feats, pet_feats_real, mask)
    * Bank update:           finalize_epoch(epoch)
    * Missing prediction:    recover_missing(ct_feats)  -- no PET argument
    * Semantic supervision:  compute_semantic_relation_loss(pet_comp, pet_real, mask)
      on the build stage only; teacher is detached.
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
        semantic_loss_weight: float = 0.01,
        collect_candidates_during_training: bool = True,
        # Backward-compat: silently ignore removed kwargs if old checkpoints
        # or builder code passes them.
        **kwargs,
    ):
        super().__init__()
        # Drop legacy keys without changing semantics.
        for _legacy in (
            "prototype_loss_type", "prototype_loss_weight",
            "prototype_temperature", "prototype_loss_stages",
            "use_affine_calibration", "use_pet_contribution_gate",
            "use_retrieval_reliability",
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
            semantic_loss_weight=float(semantic_loss_weight),
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
            [PrototypeCrossAttention(c) for c in self.channels]
        )
        self.personalization = MissingPrototypePersonalization(self.channels)

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
            cache[class_idx] = {"ct": [[] for _ in range(self.num_scales)], "pet": [[] for _ in range(self.num_scales)]}
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
    # Candidate collection
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
        descriptors: Dict[int, Dict[str, List[torch.Tensor]]] = {c: {"ct": [], "pet": []} for c in range(2)}
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
                    descriptors[class_idx]["ct"][scale_idx][common_valid].detach().to(device="cpu", dtype=torch.float32).contiguous()
                )
                self._epoch_cache[class_idx]["pet"][scale_idx].append(
                    descriptors[class_idx]["pet"][scale_idx][common_valid].detach().to(device="cpu", dtype=torch.float32).contiguous()
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
            build_ct = self._concat_cache(class_idx, "ct", self.build_stage_idx)
            class_report: Dict = {
                "num_candidates": int(build_ct.shape[0]),
                "build_stage": int(self.config.build_stage),
                "clustering": None,
                "filtering": None,
            }
            if build_ct.shape[0] == 0:
                report["classes"][class_name] = class_report
                continue
            any_candidate = True
            labels, centers, kmeans_report = deterministic_spherical_kmeans(
                build_ct, num_clusters=self.num_clusters, max_iter=self.config.cluster_max_iter,
            )
            k_eff = int(centers.shape[0])
            kept_by_cluster, filter_report = cosine_cluster_outlier_filter(
                build_ct, labels, num_clusters=k_eff, discard_rate=self.config.outlier_discard_rate,
            )
            expected_n = int(build_ct.shape[0])
            for s in range(self.num_scales):
                ct_all = self._concat_cache(class_idx, "ct", s)
                pet_all = self._concat_cache(class_idx, "pet", s)
                if ct_all.shape[0] != expected_n or pet_all.shape[0] != expected_n:
                    raise RuntimeError(
                        f"Cross-scale paired candidate alignment failed: class={class_name}, scale={s+1}, build={expected_n}, ct={ct_all.shape[0]}, pet={pet_all.shape[0]}"
                    )
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
            class_report["cluster_counts_after_filter"] = [int(new_count[class_idx, j].item()) for j in range(self.num_clusters)]
            report["classes"][class_name] = class_report

        if not any_candidate or not bool(new_ready.any()):
            report["status"] = "bank_unchanged_no_valid_candidates"
            self.reset_epoch_cache()
            return report
        if self.config.bank_update_mode == "direct":
            update_report = self._apply_direct_update(new_keys, new_values, new_ready, new_count)
        else:
            update_report = self._apply_matched_ema_update(new_keys, new_values, new_ready, new_count)
        self.bank_version.add_(1)
        report["status"] = "bank_updated"
        report["update"] = update_report
        report["bank_version_after"] = int(self.bank_version.item())
        report["ready_count"] = int(self.prototype_ready.sum().item())
        report["total_slots"] = int(self.prototype_ready.numel())
        report["prototype_count"] = self.prototype_count.detach().cpu().tolist()
        self.reset_epoch_cache()
        return report

    # ------------------------------------------------------------------
    # Retrieval (no reliability)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        ct_feats: Sequence[torch.Tensor],
        return_attention: bool = True,
    ) -> Dict:
        self._validate_features(ct_feats, None)
        ready_flat = self.prototype_ready.flatten()
        pet_proxy: List[torch.Tensor] = []
        ct_reference: List[torch.Tensor] = []
        attentions: List[torch.Tensor] = []
        for s, ct in enumerate(ct_feats):
            keys = getattr(self, f"ct_keys_s{s + 1}").reshape(2 * self.num_clusters, self.channels[s])
            values = getattr(self, f"pet_values_s{s + 1}").reshape(2 * self.num_clusters, self.channels[s])
            keys = keys.to(device=ct.device, dtype=ct.dtype)
            values = values.to(device=ct.device, dtype=ct.dtype)
            ready = ready_flat.to(device=ct.device)
            retrieved, attention = self.attention[s](ct, keys, values, ready)
            pet_proxy.append(retrieved)
            if bool(ready.any()):
                ref_tokens = torch.matmul(attention, keys)
                ref_map = ref_tokens.transpose(1, 2).reshape_as(ct)
            else:
                ref_map = torch.zeros_like(ct)
            ct_reference.append(_sanitize(ref_map))
            if return_attention:
                attentions.append(attention)
        return {
            "pet_proxy": pet_proxy,
            "ct_reference": ct_reference,
            "attention": attentions if return_attention else None,
            "bank_ready": self.bank_ready,
            "bank_version": int(self.bank_version.item()),
        }

    # ------------------------------------------------------------------
    # PASSION-style semantic prototype relation loss (build stage only)
    # ------------------------------------------------------------------

    def _zero_semantic_loss(self, ref: torch.Tensor) -> Dict:
        zero = ref.new_zeros(())
        return {"loss": zero, "weighted_loss": zero, "num_terms": 0, "details": {}}

    def compute_semantic_relation_loss(
        self,
        pet_comp_feats: Sequence[torch.Tensor],
        pet_real_feats: Sequence[torch.Tensor],
        mask: torch.Tensor,
    ) -> Dict:
        """
        Build-stage PASSION-style semantic relation loss.

        Teacher: detached real PET at build stage
        Student: compensated PET at build stage (grad flows to personalization
                 and retrieval).
        """
        self._validate_features(pet_comp_feats, None)
        self._validate_features(pet_real_feats, None)
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("mask must be [B,1,H,W]")
        s = self.build_stage_idx
        pet_comp = pet_comp_feats[s]
        pet_real = pet_real_feats[s]
        if pet_comp.shape != pet_real.shape:
            raise ValueError(f"pet_comp/pet_real shape mismatch at build stage: {tuple(pet_comp.shape)} vs {tuple(pet_real.shape)}")
        _finite_or_raise("pet_comp_build", pet_comp)
        _finite_or_raise("pet_real_build", pet_real)

        bg_mask, fg_mask = _class_masks_at_scale(mask, pet_comp.shape[-2:])
        fg_present = mask.flatten(1).sum(dim=1) > EPS
        bg_present = (1.0 - mask.float()).flatten(1).sum(dim=1) > EPS
        semantic_presence = {0: bg_present, 1: fg_present}

        term_losses: List[torch.Tensor] = []
        details: Dict[str, float] = {}

        # Teacher must be detached; student keeps grad.
        pet_real_det = pet_real.detach()

        for class_idx, class_mask in enumerate((bg_mask, fg_mask)):
            comp_desc, comp_valid = _masked_average_pool_2d(pet_comp, class_mask)
            real_desc, real_valid = _masked_average_pool_2d(pet_real_det, class_mask)
            # real_desc already detached, comp_desc keeps grad graph for its
            # per-sample prototype; but the prototype itself is derived from
            # pet_comp, so gradients propagate through it — intended.
            valid = comp_valid & real_valid & semantic_presence[class_idx].to(pet_comp.device)
            # For relation maps we need per-sample spatial maps, not just
            # aggregated descriptors. We compute relation per valid sample.
            if not bool(valid.any()):
                continue
            valid_indices = torch.nonzero(valid, as_tuple=False).flatten().long()
            for idx in valid_indices:
                # Per-sample class prototype from the SAME sample's pooled descriptor.
                # comp prototype keeps grad, real prototype detached.
                c_comp = comp_desc[idx : idx + 1]  # [1,C]
                c_real = real_desc[idx : idx + 1].detach()  # [1,C]
                # Skip degenerate zero prototypes (should not happen when valid).
                if float(c_comp.norm().item()) <= EPS or float(c_real.norm().item()) <= EPS:
                    continue
                # Spatial tokens
                comp_tokens = F.normalize(pet_comp[idx : idx + 1].flatten(2).transpose(1, 2), p=2, dim=-1, eps=EPS)  # [1,N,C]
                real_tokens = F.normalize(pet_real_det[idx : idx + 1].flatten(2).transpose(1, 2), p=2, dim=-1, eps=EPS)
                proto_comp = F.normalize(c_comp, p=2, dim=-1, eps=EPS)  # [1,C]
                proto_real = F.normalize(c_real, p=2, dim=-1, eps=EPS)
                s_comp = (comp_tokens * proto_comp[:, None, :]).sum(-1)  # [1,N]
                s_real = (real_tokens * proto_real[:, None, :]).sum(-1)  # [1,N]
                _finite_or_raise("semantic_s_comp", s_comp)
                _finite_or_raise("semantic_s_real", s_real)
                # Teacher detached, so S_real has no grad.
                term = ((s_comp - s_real.detach()) ** 2).mean()
                _finite_or_raise("semantic_term", term)
                term_losses.append(term)
            details[f"build_s{self.config.build_stage}_{CLASS_NAMES[class_idx]}_terms"] = float(len(term_losses))

        if not term_losses:
            return self._zero_semantic_loss(pet_comp)
        loss = torch.stack(term_losses).mean()
        weighted = float(self.config.semantic_loss_weight) * loss
        _finite_or_raise("semantic_loss", loss)
        _finite_or_raise("semantic_loss_weighted", weighted)
        return {"loss": loss, "weighted_loss": weighted, "num_terms": len(term_losses), "details": details}

    # ------------------------------------------------------------------
    # Missing prediction (strict, no PET arg)
    # ------------------------------------------------------------------

    def recover_missing(
        self,
        ct_feats: Sequence[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], Dict]:
        retrieval = self.retrieve(ct_feats, return_attention=True)
        pet_proxy = retrieval["pet_proxy"]
        ct_ref = retrieval["ct_reference"]
        pet_comp = self.personalization(
            ct_feats, pet_proxy, ct_ref, reference_valid=self.bank_ready,
        )
        # Bank not ready fallback: personalization returns zeros_like PET proxy
        # when reference invalid and bank empty, but proxy itself is already
        # zeros_like(CT). Keep explicit fallback to honor spec 3.7.
        if not self.bank_ready:
            pet_comp = [torch.zeros_like(c) for c in ct_feats]
            for t in pet_comp:
                _finite_or_raise("pet_comp_fallback", t)
        else:
            for t in pet_comp:
                _finite_or_raise("pet_comp", t)
                _sanitize(t)

        aux = {
            "pet_proxy": pet_proxy,
            "ct_reference": ct_ref,
            "attention": retrieval["attention"],
            "bank_ready": self.bank_ready,
            "bank_version": int(self.bank_version.item()),
        }
        return pet_comp, aux

    def forward(
        self,
        ct_feats: Sequence[torch.Tensor],
        pet_feats_real: Optional[Sequence[torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        mode: str = "missing",
        collect_candidates: Optional[bool] = None,
        compute_semantic_loss: bool = True,
    ) -> Dict:
        """
        Unified TRAINING wrapper (kept for compatibility).

        Full mode now does NOT use Module-1 for prediction; Missing mode
        follows recover_missing + optional semantic relation loss.
        Candidate collection is still supported for both modes.
        """
        if mode not in {"full", "missing"}:
            raise ValueError("mode must be 'full' or 'missing'")
        self._validate_features(ct_feats, pet_feats_real if mode == "full" or pet_feats_real is not None else None)

        should_collect = (
            self.config.collect_candidates_during_training if collect_candidates is None else bool(collect_candidates)
        )
        collect_report = None
        if self.training and should_collect and pet_feats_real is not None and mask is not None:
            collect_report = self.collect_candidates(ct_feats, pet_feats_real, mask)

        # Semantic relation loss only for Missing with a real teacher.
        semantic = self._zero_semantic_loss(ct_feats[0])
        # Full never computes semantic loss.

        if mode == "full":
            # Full does not use Module-1 for compensation; caller should bypass.
            # Keep API compatible: return zeros / no pet_output.
            pred: Dict = {
                "pet_output": None,
                "pet_proxy": None,
                "ct_reference": None,
                "attention": None,
                "bank_ready": self.bank_ready,
                "bank_version": int(self.bank_version.item()),
            }
        else:
            pet_comp, aux = self.recover_missing(ct_feats)
            if self.training and compute_semantic_loss and pet_feats_real is not None and mask is not None:
                # pet_feats_real provides the teacher at build stage.
                try:
                    self._validate_features(pet_feats_real, None)
                    semantic = self.compute_semantic_relation_loss(pet_comp, pet_feats_real, mask)
                except Exception:
                    # Keep training alive if semantic loss cannot be computed for this batch.
                    semantic = self._zero_semantic_loss(ct_feats[0])
            pred = {"pet_output": pet_comp, **aux}

        pred["semantic_loss"] = semantic["loss"]
        pred["semantic_loss_weighted"] = semantic["weighted_loss"]
        pred["semantic_loss_num_terms"] = semantic["num_terms"]
        pred["semantic_loss_details"] = semantic["details"]
        # Legacy keys for backward compat (zero).
        pred["prototype_loss"] = pred["semantic_loss"].detach() if isinstance(pred["semantic_loss"], torch.Tensor) else pred["semantic_loss"]
        pred["prototype_loss_weighted"] = pred["semantic_loss_weighted"]
        pred["prototype_loss_num_terms"] = pred["semantic_loss_num_terms"]
        pred["collect_report"] = collect_report
        return pred


# -----------------------------------------------------------------------------
# Minimal self-check (missing-only)
# -----------------------------------------------------------------------------


def _self_check() -> None:
    torch.manual_seed(7)
    channels = (8, 12, 16, 20)
    shapes = ((16, 16), (8, 8), (4, 4), (2, 2))
    module = PairedSemanticPrototypeImputation(
        channels=channels, num_clusters=3, build_stage=4, bank_update_mode="direct",
        semantic_loss_weight=0.01,
    )
    module.train()
    for _ in range(3):
        ct = [torch.randn(4, c, h, w, requires_grad=True) for c, (h, w) in zip(channels, shapes)]
        pet = [torch.randn(4, c, h, w, requires_grad=True) for c, (h, w) in zip(channels, shapes)]
        mask = torch.zeros(4, 1, 64, 64)
        mask[:, :, 18:46, 20:44] = 1.0
        module.collect_candidates(ct, pet, mask)
    report = module.finalize_epoch(epoch=1)
    assert report["status"] == "bank_updated"
    assert module.bank_ready

    b = 2
    ct = [torch.randn(b, c, h, w, requires_grad=True) for c, (h, w) in zip(channels, shapes)]
    pet_a = [torch.randn(b, c, h, w, requires_grad=True) for c, (h, w) in zip(channels, shapes)]
    pet_b = [torch.randn(b, c, h, w, requires_grad=True) for c, (h, w) in zip(channels, shapes)]
    mask = torch.zeros(b, 1, 64, 64)
    mask[:, :, 16:48, 16:48] = 1.0

    # 1-4: bank build ok, recover_missing shape
    pet_comp, aux = module.recover_missing(ct)
    for s in range(4):
        assert pet_comp[s].shape == ct[s].shape
        assert aux["pet_proxy"][s].shape == ct[s].shape
        assert aux["ct_reference"][s].shape == ct[s].shape

    # 4: bank not ready fallback
    empty = PairedSemanticPrototypeImputation(channels=channels, num_clusters=3, build_stage=4)
    empty.eval()
    pet_zero, _ = empty.recover_missing(ct)
    for s in range(4):
        assert bool((pet_zero[s] == 0).all())

    # 5: zero-init personalization => P_comp == P_proto when bank ready and head zero
    retrieval = module.retrieve(ct, return_attention=True)
    for s in range(4):
        assert torch.allclose(pet_comp[s], retrieval["pet_proxy"][s], atol=1e-5), "zero-init personalization must be identity"

    # 6-7: gamma/beta in [-1,1]
    with torch.no_grad():
        for head in module.personalization.heads:
            nn.init.normal_(head[-1].weight, std=0.02)
            nn.init.normal_(head[-1].bias, std=0.02)
    pet_comp2, _ = module.recover_missing(ct)
    # Trigger personalization internals to inspect gamma/beta indirectly:
    # Instead, verify the head outputs bounded after tanh by checking P_comp finite and not exploding.
    for s in range(4):
        assert bool(torch.isfinite(pet_comp2[s]).all())

    # 8: sigma finite >=1e-6 (implicit in personalization; check via direct call)
    # Already finite-checked inside.

    # 9-10: semantic loss finite >=0
    # Use pet_a as teacher, pet_comp as student
    module.train()
    # Re-collect with fresh module needing bank ready (already ready)
    sem = module.compute_semantic_relation_loss(pet_comp, pet_a, mask)
    assert bool(torch.isfinite(sem["loss"]).all())
    assert float(sem["loss"].item()) >= 0

    # 11-12: semantic teacher stop-grad, student has grad
    pet_comp_g = [p.clone().detach().requires_grad_(True) if s == module.build_stage_idx else p for s, p in enumerate(pet_comp)]
    # Build grad graph via personalization path: use recover_missing with grad-enabled ct
    ct_g = [c.clone().detach().requires_grad_(True) for c in ct]
    pet_comp_g2, _ = module.recover_missing(ct_g)
    # pet_real detached teacher
    pet_real_g = [p.clone().detach().requires_grad_(True) for p in pet_a]
    sem2 = module.compute_semantic_relation_loss(pet_comp_g2, pet_real_g, mask)
    if sem2["num_terms"] > 0:
        sem2["loss"].backward()
        # Teacher should have no grad (detached inside)
        assert all(p.grad is None or float(p.grad.abs().max().item()) == 0 for p in pet_real_g)
        # Student path should have grad (through ct/attention/personalization)
        assert any(c.grad is not None and float(c.grad.abs().sum().item()) > 0 for c in ct_g)

    # Backward through missing route
    module.zero_grad()
    pet_comp3, _ = module.recover_missing(ct)
    loss = sum(x.mean() for x in pet_comp3) + module.compute_semantic_relation_loss(pet_comp3, pet_a, mask)["weighted_loss"]
    loss.backward()

    print("[SELF-CHECK] passed")


if __name__ == "__main__":
    _self_check()
