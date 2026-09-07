# -*- coding: utf-8 -*-
"""
Paired Semantic Multi-Prototype Imputation (Module-1)
=====================================================

Standalone feature-level Module-1 for CT/PET missing-modality segmentation.

Design intent
-------------
This module keeps the original CPPI -> affine-calibration logic, but makes the
prototype construction, geometry, update rule, loss, and public interface
explicit and independently testable.

Core assumptions
----------------
1) Training data are paired CT/PET. Missing PET is simulated during training.
2) Real PET may be used as privileged TRAINING supervision / prototype source.
3) Missing prediction itself must depend only on current CT + the training-data
   prototype bank. `recover_missing()` therefore accepts no real PET argument.
4) Population-level PET prototype retrieval is intentionally common/average;
   a shared CT-reference-guided affine calibration then restores instance-level
   deviation for BOTH Full and Missing routes.

Module flow
-----------
Training candidate construction:
    CT/PET features + GT mask
      -> FG/BG masked average pooling at every scale
      -> FP32 paired candidate cache
      -> deterministic spherical K-means on a configurable build stage (S4 by
         default)
      -> cluster-wise cosine outlier filtering
      -> reuse the SAME build-stage membership/kept indices at all scales
      -> paired bank: CT keys <-> PET values
      -> epoch-wise direct replacement (default) OR matched paired EMA (ablation)

Prototype retrieval loss (default: PAD-KL):
    real PET descriptor -> PET-prototype soft assignment (teacher, stop-grad)
    current CT descriptor -> paired CT-key assignment (student)
    KL(teacher || student)
The loss is class-wise (FG and BG independently), and can be computed on Full
AND Missing training batches because real PET is privileged supervision only.

Missing prediction:
    current CT spatial tokens (Q)
      -> CT prototype keys (K)
      -> paired PET prototype values (V)
      -> population PET prior
      -> retrieval reliability R (normalized attention entropy, detached)
      -> CT-current vs CT-reference discrepancy
      -> shared affine calibration (gamma / beta)
      -> PET contribution gate G = 2 * sigmoid(raw_gate) in (0, 2), init 1
      -> P_out = G * R * calibrated prototype PET
      -> simple_fused = CT + P_out

Full prediction:
    real PET features
      -> the SAME shared CT-reference-guided affine calibration
      -> the SAME PET contribution gate (retrieval reliability NOT applied)
      -> P_out = G * calibrated real PET
      -> simple_fused = CT + P_out

Outputs
-------
`pet_output` is the final quality-controlled PET feature (Module-2 input);
`simple_fused = CT + pet_output` is the Module-1-only decoder input.
The module implements the simple residual fusion itself, so downstream code
must NOT apply an external AddFusion on top of `pet_output`.

Recommended first controlled configuration
------------------------------------------
    channels=(64, 128, 320, 512)
    num_clusters=6
    build_stage=4
    outlier_discard_rate=0.05
    bank_update_mode="direct"
    prototype_loss_type="pad_kl"
    prototype_loss_weight=0.01
    prototype_temperature=0.1
    prototype_loss_stages=None   # => build_stage only
    use_affine_calibration=True
    use_pet_contribution_gate=True
    use_retrieval_reliability=True

Optional ablations exposed as switches
--------------------------------------
- build_stage: 1/2/3/4
- num_clusters: K
- prototype_loss_type: none / pad_kl / pad_js / retrieval_cosine
- prototype_loss_stages: any subset of stages, e.g. (4,) or (1,2,3,4)
- bank_update_mode: direct / matched_ema
- use_affine_calibration: True / False
- use_pet_contribution_gate: True / False
- use_retrieval_reliability: True / False

Notes on reproducibility
------------------------
- Candidate cache is detached CPU float32 (never float16).
- Spherical K-means is deterministic:
    first center = real sample most aligned with global mean direction;
    later centers = deterministic cosine farthest-point initialization.
- Clustering and outlier filtering use the SAME cosine geometry.
- The training entry script should still enable global PyTorch/CUDA strict
  deterministic settings; this module does not modify global backend state.
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

    Returns
    -------
    descriptor: [B,C]
    valid:      [B] bool
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
    """
    Deterministic spherical K-means on cosine geometry.

    Key normalization change vs. the old implementation:
    - `x` is L2 normalized first.
    - first center is NOT argmax(norm(x)) (all norms are ~1 after normalize).
    - first center is the real sample most aligned with the global mean direction.
    - subsequent centers use deterministic cosine farthest-point initialization.

    Parameters
    ----------
    x : [N,D]
        CPU/GPU tensor; internally converted to float32.
    num_clusters : int
    max_iter : int

    Returns
    -------
    labels  : [N] long
    centers : [K_eff,D] normalized
    report  : diagnostic dict
    """
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
        # distance to nearest already-selected center
        nearest_similarity = (x_n @ centers_now.t()).max(dim=1).values
        nearest_distance = 1.0 - nearest_similarity
        # Prevent re-selecting an already chosen sample.
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
                # Deterministic rescue for an empty cluster: choose the currently
                # worst represented sample, avoiding duplicate rescue samples.
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
    """
    Filter outliers *inside each discovered phenotype cluster* using the same
    cosine geometry as spherical K-means.

    The number discarded is floor(discard_rate * cluster_size), so small
    clusters are not accidentally over-pruned (e.g. 1/6 != 5%).
    """
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
        # Deterministic ascending sort; ties keep the lower source index order.
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

    Linear projections are identity-initialized to preserve prototype geometry at
    initialization while still allowing task-driven adaptation.
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

    def projected_ct_similarity(
        self,
        descriptors: torch.Tensor,
        keys: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        q = F.normalize(self.q_proj(descriptors), p=2, dim=-1, eps=EPS)
        k = F.normalize(self.k_proj(keys), p=2, dim=-1, eps=EPS)
        return (q @ k.t()) / float(temperature)

    def projected_pet_similarity(
        self,
        descriptors: torch.Tensor,
        values: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        p = F.normalize(self.v_proj(descriptors), p=2, dim=-1, eps=EPS)
        v = F.normalize(self.v_proj(values), p=2, dim=-1, eps=EPS)
        return (p @ v.t()) / float(temperature)


# -----------------------------------------------------------------------------
# Shared Full/Missing affine calibration
# -----------------------------------------------------------------------------


class SharedPrototypeReferencedAffineCalibration(nn.Module):
    """
    Shared CT-reference-guided PET affine calibration.

    Population prototype retrieval gives a common PET prior. The discrepancy
    between current CT and its retrieved CT reference provides an instance cue:

        delta = mean_t( normalize(CT_current_t) - normalize(CT_reference_t) )
        [gamma, beta] = head(delta)
        PET_cal = PET + gamma * (PET - spatial_mean(PET)) + beta

    The SAME parameters are used for:
      - Full:    real PET -> calibrated real PET
      - Missing: retrieved PET prior -> individualized compensated PET

    CT/current-reference inputs are detached here, matching the original design:
    calibration learns how to use the discrepancy without using this path to
    destabilize the CT representation itself.
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
                nn.Linear(hidden, 3 * c),
            )
            # Identity behavior at initialization: gamma=beta=0, gate=1.
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            heads.append(head)
        self.heads = nn.ModuleList(heads)

    def forward(
        self,
        ct_feats: Sequence[torch.Tensor],
        pet_evidence_feats: Sequence[torch.Tensor],
        ct_reference_feats: Optional[Sequence[torch.Tensor]],
        reference_valid: bool,
        use_pet_contribution_gate: bool = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if len(ct_feats) != len(pet_evidence_feats):
            raise ValueError("CT and PET evidence must have the same scale count")
        if not reference_valid or ct_reference_feats is None:
            calibrated = [_sanitize(pet) for pet in pet_evidence_feats]
            gates = [
                torch.ones_like(pet[:, :1, :, :])
                for pet in pet_evidence_feats
            ]
            return calibrated, gates
        if len(ct_reference_feats) != len(ct_feats):
            raise ValueError("CT reference scale count mismatch")

        calibrated: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []
        for scale_idx, (ct, pet, ct_ref) in enumerate(
            zip(ct_feats, pet_evidence_feats, ct_reference_feats)
        ):
            ct_det = ct.detach()
            ref_det = ct_ref.detach()

            if pet.shape[-2:] != ct_det.shape[-2:]:
                pet = F.interpolate(
                    pet,
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
            if ct_det.shape != pet.shape or ct_det.shape != ref_det.shape:
                raise ValueError(
                    f"Scale {scale_idx + 1} shape mismatch: "
                    f"ct={tuple(ct_det.shape)} pet={tuple(pet.shape)} "
                    f"ref={tuple(ref_det.shape)}"
                )

            ct_tokens = F.normalize(
                ct_det.flatten(2).transpose(1, 2),
                p=2,
                dim=-1,
                eps=1e-6,
            )
            ref_tokens = F.normalize(
                ref_det.flatten(2).transpose(1, 2),
                p=2,
                dim=-1,
                eps=1e-6,
            )
            delta = (ct_tokens - ref_tokens).mean(dim=1)
            affine = self.heads[scale_idx](delta)
            raw_gamma, raw_beta, raw_gate = affine.chunk(3, dim=-1)
            gamma = torch.tanh(raw_gamma).view(pet.shape[0], pet.shape[1], 1, 1)
            beta = torch.tanh(raw_beta).view(pet.shape[0], pet.shape[1], 1, 1)
            # Gate range (0, 2); raw_gate=0 at init gives gate=1 so Full PET is
            # not scaled down before learning starts.
            gate = 2.0 * torch.sigmoid(raw_gate).view(
                pet.shape[0], pet.shape[1], 1, 1
            )
            if not use_pet_contribution_gate:
                gate = torch.ones_like(gate)

            pet_centered = pet - pet.mean(dim=(2, 3), keepdim=True)
            pet_cal = pet + gamma * pet_centered + beta
            _finite_or_raise("pet_gate", gate)
            calibrated.append(_sanitize(pet_cal))
            gates.append(gate)
        return calibrated, gates


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

    # direct = current-epoch bank replaces previous bank (recommended first)
    # matched_ema = optional FedMEPD-style stabilization ablation; CT determines
    #               slot correspondence and paired PET values follow the same map.
    bank_update_mode: str = "direct"
    ema_momentum: float = 0.999

    # Prototype supervision switches.
    prototype_loss_type: str = "pad_kl"
    prototype_loss_weight: float = 0.01
    prototype_temperature: float = 0.1
    prototype_loss_stages: Optional[Tuple[int, ...]] = None

    use_affine_calibration: bool = True
    use_pet_contribution_gate: bool = True
    use_retrieval_reliability: bool = True
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
        if self.prototype_loss_type not in {
            "none",
            "pad_kl",
            "pad_js",
            "retrieval_cosine",
        }:
            raise ValueError(
                "prototype_loss_type must be one of: "
                "none, pad_kl, pad_js, retrieval_cosine"
            )
        if self.prototype_loss_weight < 0.0:
            raise ValueError("prototype_loss_weight must be >= 0")
        if self.prototype_temperature <= 0.0:
            raise ValueError("prototype_temperature must be > 0")
        if self.prototype_loss_stages is not None:
            for stage in self.prototype_loss_stages:
                _stage_to_index(stage, len(self.channels))

    @property
    def resolved_loss_stages(self) -> Tuple[int, ...]:
        if self.prototype_loss_stages is None:
            return (int(self.build_stage),)
        # Stable de-duplication.
        seen = set()
        stages = []
        for s in self.prototype_loss_stages:
            s = int(s)
            if s not in seen:
                stages.append(s)
                seen.add(s)
        return tuple(stages)


# -----------------------------------------------------------------------------
# Main standalone Module-1
# -----------------------------------------------------------------------------


class PairedSemanticPrototypeImputation(nn.Module):
    """
    Independent Module-1.

    Public integration contract
    ---------------------------
    Inputs are already-encoded, channel-aligned CT/PET multi-scale features:
        ct_feats[s], pet_feats_real[s]: [B,C_s,H_s,W_s]

    Recommended use inside the baseline training loop:

        out = module1(
            ct_feats=ct_feats,
            pet_feats_real=pet_feats_real,  # privileged train-only teacher/source
            mask=mask,
            mode=route,                     # 'full' or 'missing'
        )
        fused_feats = out['simple_fused']   # CT + P_out (Module-1 simple fusion)
        pet_output  = out['pet_output']     # quality-controlled PET for Module-2
        proto_loss = out['prototype_loss_weighted']

    End of every epoch:
        report = module1.finalize_epoch(epoch)

    Strict Missing inference:
        # No current-patient PET is accepted by this method.
        pet_output, aux = module1.recover_missing(ct_feats)
        fused_feats = aux['simple_fused']
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
        prototype_loss_type: str = "pad_kl",
        prototype_loss_weight: float = 0.01,
        prototype_temperature: float = 0.1,
        prototype_loss_stages: Optional[Sequence[int]] = None,
        use_affine_calibration: bool = True,
        use_pet_contribution_gate: bool = True,
        use_retrieval_reliability: bool = True,
        collect_candidates_during_training: bool = True,
    ):
        super().__init__()
        self.config = Module1Config(
            channels=tuple(int(c) for c in channels),
            num_clusters=int(num_clusters),
            build_stage=int(build_stage),
            cluster_max_iter=int(cluster_max_iter),
            outlier_discard_rate=float(outlier_discard_rate),
            bank_update_mode=str(bank_update_mode),
            ema_momentum=float(ema_momentum),
            prototype_loss_type=str(prototype_loss_type),
            prototype_loss_weight=float(prototype_loss_weight),
            prototype_temperature=float(prototype_temperature),
            prototype_loss_stages=(
                None
                if prototype_loss_stages is None
                else tuple(int(s) for s in prototype_loss_stages)
            ),
            use_affine_calibration=bool(use_affine_calibration),
            use_pet_contribution_gate=bool(use_pet_contribution_gate),
            use_retrieval_reliability=bool(use_retrieval_reliability),
            collect_candidates_during_training=bool(
                collect_candidates_during_training
            ),
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
        self.affine_calibration = SharedPrototypeReferencedAffineCalibration(
            self.channels
        )

        # Paired per-scale bank. Semantic class dimension: 0=BG, 1=FG.
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
        self.register_buffer(
            "bank_version", torch.zeros((), dtype=torch.long)
        )

        self._epoch_cache = self._new_cache()
        self._collect_calls = 0
        self._collected_records = 0

    # ------------------------------------------------------------------
    # Introspection / switches
    # ------------------------------------------------------------------

    @property
    def bank_ready(self) -> bool:
        return bool(self.prototype_ready.any())

    def export_config(self) -> Dict:
        cfg = asdict(self.config)
        cfg["resolved_loss_stages"] = list(self.config.resolved_loss_stages)
        return cfg

    def set_prototype_loss(
        self,
        loss_type: Optional[str] = None,
        weight: Optional[float] = None,
        stages: Optional[Sequence[int]] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """Runtime ablation switch without reconstructing the module."""
        if loss_type is not None:
            self.config.prototype_loss_type = str(loss_type)
        if weight is not None:
            self.config.prototype_loss_weight = float(weight)
        if stages is not None:
            self.config.prototype_loss_stages = tuple(int(s) for s in stages)
        if temperature is not None:
            self.config.prototype_temperature = float(temperature)
        self.config.validate()

    # ------------------------------------------------------------------
    # Validation / cache
    # ------------------------------------------------------------------

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

        batch_size = int(ct_feats[0].shape[0])
        for s, ct in enumerate(ct_feats):
            if ct.ndim != 4:
                raise ValueError(f"ct_feats[{s}] must be 4D")
            if ct.shape[0] != batch_size or ct.shape[1] != self.channels[s]:
                raise ValueError(
                    f"ct_feats[{s}] expected [B,{self.channels[s]},H,W], "
                    f"got {tuple(ct.shape)}"
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

    def _concat_cache(
        self,
        class_idx: int,
        modality: str,
        scale_idx: int,
    ) -> torch.Tensor:
        chunks = self._epoch_cache[class_idx][modality][scale_idx]
        if not chunks:
            return torch.empty(
                0, self.channels[scale_idx], dtype=torch.float32, device="cpu"
            )
        return torch.cat(chunks, dim=0).float().contiguous()

    # ------------------------------------------------------------------
    # Candidate collection: paired + FP32 + exact cross-scale alignment
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect_candidates(
        self,
        ct_feats: Sequence[torch.Tensor],
        pet_feats_real: Sequence[torch.Tensor],
        mask: torch.Tensor,
    ) -> Dict:
        """
        Collect paired CT/PET FG/BG descriptors from REAL training modalities.

        Crucial implementation invariant:
        a record is accepted only if it is valid at *every* scale for both CT
        and PET. Therefore candidate index i refers to exactly the same sample
        and semantic region across S1..S4, which makes build-stage membership
        reuse mathematically well-defined.
        """
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

        # Original-resolution semantic presence makes FG/BG validity explicit.
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
                # CPU float32 cache: no irreversible fp16 quantization.
                self._epoch_cache[class_idx]["ct"][scale_idx].append(
                    descriptors[class_idx]["ct"][scale_idx][common_valid]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                )
                self._epoch_cache[class_idx]["pet"][scale_idx].append(
                    descriptors[class_idx]["pet"][scale_idx][common_valid]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
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
    # Exact small-K one-to-one matching for optional paired EMA
    # ------------------------------------------------------------------

    @staticmethod
    def _optimal_pairs(cost: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Deterministic minimum-cost one-to-one matching without SciPy.

        Expected K is small (e.g. 4/6/8). Dynamic programming is exact for the
        smaller side and avoids an external dependency. Returned indices refer
        to rows and columns of `cost`.
        """
        if cost.ndim != 2:
            raise ValueError("cost must be 2D")
        n_rows, n_cols = cost.shape
        if n_rows == 0 or n_cols == 0:
            return []

        # Make rows the smaller/equal side for bitmask DP.
        transposed = False
        work = cost
        if n_rows > n_cols:
            work = cost.t()
            transposed = True
        r, c = work.shape

        # For unusually large K, use deterministic greedy to avoid exponential
        # state growth. This branch is not expected for the intended K<=8 use.
        if c > 16:
            pairs = []
            used_cols = set()
            for i in range(r):
                candidates = [j for j in range(c) if j not in used_cols]
                j = min(candidates, key=lambda z: (float(work[i, z]), z))
                used_cols.add(j)
                pairs.append((i, j))
        else:
            # state: used-column mask -> (cost, tuple(columns chosen so far))
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
    def _apply_direct_update(
        self,
        new_keys: Sequence[torch.Tensor],
        new_values: Sequence[torch.Tensor],
        new_ready: torch.Tensor,
        new_count: torch.Tensor,
    ) -> Dict:
        for s in range(self.num_scales):
            key_buf = getattr(self, f"ct_keys_s{s + 1}")
            val_buf = getattr(self, f"pet_values_s{s + 1}")
            key_buf.copy_(new_keys[s].to(key_buf.device, dtype=key_buf.dtype))
            val_buf.copy_(new_values[s].to(val_buf.device, dtype=val_buf.dtype))
        self.prototype_ready.copy_(new_ready.to(self.prototype_ready.device))
        self.prototype_count.copy_(new_count.to(self.prototype_count.device))
        return {"mode": "direct", "matches": {}}

    @torch.no_grad()
    def _apply_matched_ema_update(
        self,
        new_keys: Sequence[torch.Tensor],
        new_values: Sequence[torch.Tensor],
        new_ready: torch.Tensor,
        new_count: torch.Tensor,
    ) -> Dict:
        """
        Optional FedMEPD-style stabilization adapted to a paired CT-key/PET-value
        bank. CT build-stage keys determine slot correspondence; PET values use
        exactly the same correspondence and EMA coefficient.
        """
        momentum = float(self.config.ema_momentum)
        report = {"mode": "matched_ema", "matches": {}}

        # Work on CPU float32 copies to make the update explicit and stable.
        out_keys = [
            getattr(self, f"ct_keys_s{s + 1}").detach().float().cpu().clone()
            for s in range(self.num_scales)
        ]
        out_values = [
            getattr(self, f"pet_values_s{s + 1}").detach().float().cpu().clone()
            for s in range(self.num_scales)
        ]
        out_ready = self.prototype_ready.detach().cpu().clone()
        out_count = self.prototype_count.detach().cpu().clone()

        old_build = out_keys[self.build_stage_idx]
        new_build = new_keys[self.build_stage_idx]

        for class_idx, class_name in enumerate(CLASS_NAMES):
            old_slots = torch.nonzero(
                out_ready[class_idx], as_tuple=False
            ).flatten().long()
            new_slots = torch.nonzero(
                new_ready[class_idx], as_tuple=False
            ).flatten().long()

            if new_slots.numel() == 0:
                report["matches"][class_name] = []
                continue

            class_pairs = []
            used_new = set()
            used_old = set()

            if old_slots.numel() > 0:
                cost = _pairwise_cosine_distance(
                    old_build[class_idx, old_slots],
                    new_build[class_idx, new_slots],
                )
                local_pairs = self._optimal_pairs(cost)
                for old_local, new_local in local_pairs:
                    old_slot = int(old_slots[old_local].item())
                    new_slot = int(new_slots[new_local].item())
                    used_old.add(old_slot)
                    used_new.add(new_slot)

                    for s in range(self.num_scales):
                        mixed_key = (
                            momentum * out_keys[s][class_idx, old_slot]
                            + (1.0 - momentum) * new_keys[s][class_idx, new_slot]
                        )
                        out_keys[s][class_idx, old_slot] = F.normalize(
                            mixed_key, dim=0, eps=EPS
                        )
                        out_values[s][class_idx, old_slot] = (
                            momentum * out_values[s][class_idx, old_slot]
                            + (1.0 - momentum)
                            * new_values[s][class_idx, new_slot]
                        )
                    out_ready[class_idx, old_slot] = True
                    out_count[class_idx, old_slot] = new_count[
                        class_idx, new_slot
                    ]
                    class_pairs.append(
                        {
                            "old_slot": old_slot,
                            "new_slot": new_slot,
                            "cosine_distance": float(cost[old_local, new_local]),
                        }
                    )

            # New clusters with no old match occupy free/unmatched bank slots.
            free_slots = [
                s
                for s in range(self.num_clusters)
                if s not in used_old
                and (not bool(out_ready[class_idx, s]) or s in used_old)
            ]
            # If every slot was previously ready but there are unmatched new
            # clusters (possible when old/new counts differ), use old slots not
            # consumed by a match, deterministically by slot index.
            if len(free_slots) < (len(new_slots) - len(used_new)):
                free_slots.extend(
                    s
                    for s in range(self.num_clusters)
                    if s not in used_old and s not in free_slots
                )

            for new_slot in [int(v.item()) for v in new_slots if int(v.item()) not in used_new]:
                if not free_slots:
                    break
                target_slot = free_slots.pop(0)
                for s in range(self.num_scales):
                    out_keys[s][class_idx, target_slot] = new_keys[s][
                        class_idx, new_slot
                    ]
                    out_values[s][class_idx, target_slot] = new_values[s][
                        class_idx, new_slot
                    ]
                out_ready[class_idx, target_slot] = True
                out_count[class_idx, target_slot] = new_count[
                    class_idx, new_slot
                ]
                class_pairs.append(
                    {
                        "old_slot": None,
                        "new_slot": new_slot,
                        "assigned_slot": target_slot,
                        "cosine_distance": None,
                    }
                )

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
        """
        Build/update the paired bank once per epoch from the current epoch cache.

        Build-stage CT descriptors determine K-means membership. The same kept
        candidate indices are then applied to CT and PET descriptors at every
        scale, preserving exact CT-key <-> PET-value pairing.
        """
        report: Dict = {
            "epoch": None if epoch is None else int(epoch),
            "config": self.export_config(),
            "collect_calls": int(self._collect_calls),
            "collected_records": int(self._collected_records),
            "bank_version_before": int(self.bank_version.item()),
            "classes": {},
        }

        new_keys = [
            torch.zeros(2, self.num_clusters, c, dtype=torch.float32)
            for c in self.channels
        ]
        new_values = [
            torch.zeros(2, self.num_clusters, c, dtype=torch.float32)
            for c in self.channels
        ]
        new_ready = torch.zeros(2, self.num_clusters, dtype=torch.bool)
        new_count = torch.zeros(2, self.num_clusters, dtype=torch.long)

        any_candidate = False

        for class_idx, class_name in enumerate(CLASS_NAMES):
            build_ct = self._concat_cache(
                class_idx, "ct", self.build_stage_idx
            )
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
                build_ct,
                num_clusters=self.num_clusters,
                max_iter=self.config.cluster_max_iter,
            )
            k_eff = int(centers.shape[0])
            kept_by_cluster, filter_report = cosine_cluster_outlier_filter(
                build_ct,
                labels,
                num_clusters=k_eff,
                discard_rate=self.config.outlier_discard_rate,
            )

            # Strong alignment check before cross-scale aggregation.
            expected_n = int(build_ct.shape[0])
            for s in range(self.num_scales):
                ct_all = self._concat_cache(class_idx, "ct", s)
                pet_all = self._concat_cache(class_idx, "pet", s)
                if ct_all.shape[0] != expected_n or pet_all.shape[0] != expected_n:
                    raise RuntimeError(
                        "Cross-scale paired candidate alignment failed: "
                        f"class={class_name}, scale={s+1}, "
                        f"build={expected_n}, ct={ct_all.shape[0]}, "
                        f"pet={pet_all.shape[0]}"
                    )

                for cluster_idx in range(k_eff):
                    kept = kept_by_cluster.get(cluster_idx)
                    if kept is None or kept.numel() == 0:
                        continue

                    ct_key = ct_all[kept].mean(dim=0)
                    pet_value = pet_all[kept].mean(dim=0)
                    new_keys[s][class_idx, cluster_idx] = F.normalize(
                        ct_key.float(), dim=0, eps=EPS
                    )
                    # PET value keeps magnitude information; normalize only when
                    # computing cosine teacher assignments.
                    new_values[s][class_idx, cluster_idx] = pet_value.float()

                    if s == self.build_stage_idx:
                        new_ready[class_idx, cluster_idx] = True
                        new_count[class_idx, cluster_idx] = int(kept.numel())

            class_report["clustering"] = kmeans_report
            class_report["filtering"] = filter_report
            class_report["effective_clusters"] = int(k_eff)
            class_report["cluster_counts_after_filter"] = [
                int(new_count[class_idx, j].item())
                for j in range(self.num_clusters)
            ]
            report["classes"][class_name] = class_report

        if not any_candidate or not bool(new_ready.any()):
            report["status"] = "bank_unchanged_no_valid_candidates"
            self.reset_epoch_cache()
            return report

        if self.config.bank_update_mode == "direct":
            update_report = self._apply_direct_update(
                new_keys, new_values, new_ready, new_count
            )
        else:
            update_report = self._apply_matched_ema_update(
                new_keys, new_values, new_ready, new_count
            )

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
    # Retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_retrieval_reliability(
        attention: torch.Tensor,
        ready: torch.Tensor,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-pixel retrieval reliability from the *same* spatial prototype
        attention used for PET retrieval:

            H(x) = -sum_j A(x,j) log A(x,j)
            R(x) = 1 - H(x) / log(M),  M = number of ready prototypes

        M is the number of prototypes that actually participate in the softmax
        (the masked non-ready slots carry zero probability). The reliability
        branch is detached: it only measures how decisive retrieval already is
        and must not let segmentation loss push the attention toward a
        degenerate high-confidence solution.
        """
        b, n, _ = attention.shape
        num_ready = int(ready.sum().item())
        h, w = ref.shape[-2], ref.shape[-1]
        if num_ready == 0:
            return torch.zeros(
                b, 1, h, w, device=ref.device, dtype=ref.dtype
            )
        if num_ready == 1:
            return torch.ones(
                b, 1, h, w, device=ref.device, dtype=ref.dtype
            )
        attn_fp32 = attention.float()
        entropy = -(attn_fp32 * attn_fp32.clamp_min(EPS).log()).sum(dim=-1)
        reliability = 1.0 - entropy / math.log(float(num_ready))
        reliability = reliability.clamp(0.0, 1.0).detach()
        reliability = reliability.view(b, 1, h, w).to(dtype=ref.dtype)
        _finite_or_raise("retrieval_reliability", reliability)
        return reliability

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
        reliabilities: List[torch.Tensor] = []

        for s, ct in enumerate(ct_feats):
            keys = getattr(self, f"ct_keys_s{s + 1}").reshape(
                2 * self.num_clusters, self.channels[s]
            )
            values = getattr(self, f"pet_values_s{s + 1}").reshape(
                2 * self.num_clusters, self.channels[s]
            )
            keys = keys.to(device=ct.device, dtype=ct.dtype)
            values = values.to(device=ct.device, dtype=ct.dtype)
            ready = ready_flat.to(device=ct.device)

            retrieved, attention = self.attention[s](
                ct, keys, values, ready
            )
            pet_proxy.append(retrieved)

            # Reliability comes from the SAME attention; no second attention.
            reliabilities.append(
                self._compute_retrieval_reliability(attention, ready, ct)
            )

            if bool(ready.any()):
                # Use the same attention weights to reconstruct the paired CT
                # prototype reference; no independent assignment is introduced.
                projected_keys = self.attention[s].k_proj(keys)
                # Calibration discrepancy should live in CT feature geometry.
                # Reconstruct from raw normalized CT keys using the SAME weights,
                # matching the original CPPI-affine design.
                del projected_keys
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
            "retrieval_reliability": reliabilities,
            "bank_ready": self.bank_ready,
            "bank_version": int(self.bank_version.item()),
        }

    # ------------------------------------------------------------------
    # Prototype supervision
    # ------------------------------------------------------------------

    def _zero_loss_dict(self, ref: torch.Tensor) -> Dict:
        zero = ref.new_zeros(())
        return {
            "loss": zero,
            "weighted_loss": zero,
            "num_terms": 0,
            "details": {},
        }

    def compute_prototype_loss(
        self,
        ct_feats: Sequence[torch.Tensor],
        pet_feats_real: Sequence[torch.Tensor],
        mask: torch.Tensor,
        loss_type: Optional[str] = None,
        stages: Optional[Sequence[int]] = None,
    ) -> Dict:
        """
        Compute privileged real-PET prototype supervision.

        Default PAD-KL for each selected stage and semantic class c:

          teacher = softmax(cos(v_proj(P_real), v_proj(PET_values_c)) / T)
          student = softmax(cos(q_proj(CT),     k_proj(CT_keys_c))   / T)
          L = KL(stopgrad(teacher) || student)

        The teacher distribution is detached. Prototype buffers are registered
        buffers and are never optimized by this loss.
        """
        self._validate_features(ct_feats, pet_feats_real)
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("mask must be [B,1,H,W]")

        current_loss_type = (
            self.config.prototype_loss_type if loss_type is None else str(loss_type)
        )
        if current_loss_type == "none" or not self.bank_ready:
            return self._zero_loss_dict(ct_feats[0])

        if current_loss_type not in {
            "pad_kl",
            "pad_js",
            "retrieval_cosine",
        }:
            raise ValueError(f"Unsupported prototype loss: {current_loss_type}")

        selected_stages = (
            self.config.resolved_loss_stages
            if stages is None
            else tuple(int(s) for s in stages)
        )
        stage_indices = [_stage_to_index(s, self.num_scales) for s in selected_stages]
        temperature = float(self.config.prototype_temperature)

        term_losses: List[torch.Tensor] = []
        details: Dict[str, float] = {}

        fg_present = mask.flatten(1).sum(dim=1) > EPS
        bg_present = (1.0 - mask.float()).flatten(1).sum(dim=1) > EPS
        semantic_presence = {0: bg_present, 1: fg_present}

        for s in stage_indices:
            ct = ct_feats[s]
            pet = pet_feats_real[s]
            bg_mask, fg_mask = _class_masks_at_scale(mask, ct.shape[-2:])

            for class_idx, class_mask in enumerate((bg_mask, fg_mask)):
                ct_desc, ct_valid = _masked_average_pool_2d(ct, class_mask)
                pet_desc, pet_valid = _masked_average_pool_2d(pet, class_mask)
                valid = ct_valid & pet_valid & semantic_presence[class_idx].to(ct.device)
                if not bool(valid.any()):
                    continue

                ready = self.prototype_ready[class_idx]
                ready_idx = torch.nonzero(ready, as_tuple=False).flatten().long()
                # With one prototype the assignment distribution is trivially 1,
                # so PAD gives no learning signal. Retrieval-cosine can still work.
                if current_loss_type in {"pad_kl", "pad_js"} and ready_idx.numel() <= 1:
                    continue
                if ready_idx.numel() == 0:
                    continue

                keys = getattr(self, f"ct_keys_s{s + 1}")[class_idx, ready_idx]
                values = getattr(self, f"pet_values_s{s + 1}")[class_idx, ready_idx]
                keys = keys.to(device=ct.device, dtype=ct.dtype)
                values = values.to(device=ct.device, dtype=ct.dtype)

                ct_v = ct_desc[valid]
                pet_v = pet_desc[valid]

                student_logits = self.attention[s].projected_ct_similarity(
                    ct_v, keys, temperature
                )
                teacher_logits = self.attention[s].projected_pet_similarity(
                    pet_v, values, temperature
                )
                teacher_prob = torch.softmax(teacher_logits, dim=-1).detach()
                student_log_prob = torch.log_softmax(student_logits, dim=-1)
                student_prob = student_log_prob.exp()

                if current_loss_type == "pad_kl":
                    # Exact KL(teacher || student), averaged over valid descriptors.
                    term = (
                        teacher_prob
                        * (
                            torch.log(teacher_prob.clamp_min(EPS))
                            - student_log_prob
                        )
                    ).sum(dim=-1).mean()
                elif current_loss_type == "pad_js":
                    midpoint = 0.5 * (teacher_prob + student_prob)
                    kl_t = (
                        teacher_prob
                        * (
                            torch.log(teacher_prob.clamp_min(EPS))
                            - torch.log(midpoint.clamp_min(EPS))
                        )
                    ).sum(dim=-1)
                    kl_s = (
                        student_prob
                        * (
                            torch.log(student_prob.clamp_min(EPS))
                            - torch.log(midpoint.clamp_min(EPS))
                        )
                    ).sum(dim=-1)
                    term = 0.5 * (kl_t + kl_s).mean()
                else:  # retrieval_cosine
                    weights = student_prob
                    retrieved_desc = torch.matmul(weights, values)
                    term = (
                        1.0
                        - F.cosine_similarity(
                            retrieved_desc.float(),
                            pet_v.float(),
                            dim=-1,
                            eps=EPS,
                        )
                    ).mean()

                term_losses.append(term)
                details[
                    f"s{s+1}_{CLASS_NAMES[class_idx]}_{current_loss_type}"
                ] = float(term.detach().cpu())

        if not term_losses:
            return self._zero_loss_dict(ct_feats[0])

        loss = torch.stack(term_losses).mean()
        weighted = float(self.config.prototype_loss_weight) * loss
        return {
            "loss": loss,
            "weighted_loss": weighted,
            "num_terms": len(term_losses),
            "details": details,
        }

    # ------------------------------------------------------------------
    # Full / Missing PET preparation
    # ------------------------------------------------------------------

    def prepare_full(
        self,
        ct_feats: Sequence[torch.Tensor],
        pet_feats_real: Sequence[torch.Tensor],
    ) -> Dict:
        """
        Full route: shared calibration + contribution gate of REAL PET.

        Flow: retrieve CT reference -> CT-current/reference discrepancy ->
        shared head (gamma/beta/gate) -> calibrate real PET -> gate real PET ->
        P_out -> simple_fused = CT + P_out.

        The retrieval reliability is NOT applied to real PET (real PET is not
        produced by retrieval, so retrieval confidence must not suppress it).
        When the bank is not ready, calibration is identity, gate = 1, and
        P_out = real PET, keeping epoch-1 behavior close to the baseline.
        """
        self._validate_features(ct_feats, pet_feats_real)
        if self.bank_ready:
            retrieval = self.retrieve(ct_feats, return_attention=True)
            ct_ref = retrieval["ct_reference"]
            reliability = retrieval["retrieval_reliability"]
            reference_valid = True
        else:
            retrieval = {
                "pet_proxy": None,
                "ct_reference": None,
                "attention": None,
                "retrieval_reliability": None,
                "bank_ready": False,
                "bank_version": int(self.bank_version.item()),
            }
            ct_ref = None
            reliability = None
            reference_valid = False

        if self.config.use_affine_calibration:
            calibrated, gates = self.affine_calibration(
                ct_feats,
                pet_feats_real,
                ct_ref,
                reference_valid=reference_valid,
                use_pet_contribution_gate=self.config.use_pet_contribution_gate,
            )
        else:
            calibrated = [_sanitize(p) for p in pet_feats_real]
            gates = [
                torch.ones(
                    p.shape[0],
                    p.shape[1],
                    1,
                    1,
                    device=p.device,
                    dtype=p.dtype,
                )
                for p in pet_feats_real
            ]

        pet_output: List[torch.Tensor] = []
        simple_fused: List[torch.Tensor] = []
        for s, (ct, pet_cal, gate) in enumerate(
            zip(ct_feats, calibrated, gates)
        ):
            pet_out = gate * pet_cal
            _finite_or_raise(f"pet_output_s{s + 1}", pet_out)
            pet_out = _sanitize(pet_out)
            fused = ct + pet_out
            _finite_or_raise(f"simple_fused_s{s + 1}", fused)
            simple_fused.append(_sanitize(fused))
            pet_output.append(pet_out)

        return {
            "pet_output": pet_output,
            "simple_fused": simple_fused,
            "pet_gate": gates,
            "pet_proxy": None,
            "ct_reference": ct_ref,
            "attention": retrieval["attention"],
            "retrieval_reliability": reliability,
            "bank_ready": self.bank_ready,
            "bank_version": int(self.bank_version.item()),
        }

    def recover_missing(
        self,
        ct_feats: Sequence[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], Dict]:
        """
        Strict Missing prediction primitive.

        IMPORTANT: this method accepts NO real PET input. Therefore its output is
        structurally independent of the current patient's PET file.

        Flow: retrieve P_proto / C_ref / R -> shared adaptation -> P_cal_proto
        -> gate -> P_out = gate * reliability * P_cal_proto ->
        simple_fused = CT + P_out.
        """
        retrieval = self.retrieve(ct_feats, return_attention=True)
        pet_proxy = retrieval["pet_proxy"]
        ct_ref = retrieval["ct_reference"]
        reliability = retrieval["retrieval_reliability"]

        if self.config.use_affine_calibration:
            calibrated, gates = self.affine_calibration(
                ct_feats,
                pet_proxy,
                ct_ref,
                reference_valid=self.bank_ready,
                use_pet_contribution_gate=self.config.use_pet_contribution_gate,
            )
        else:
            calibrated = [_sanitize(p) for p in pet_proxy]
            gates = [
                torch.ones(
                    p.shape[0],
                    p.shape[1],
                    1,
                    1,
                    device=p.device,
                    dtype=p.dtype,
                )
                for p in pet_proxy
            ]

        pet_output: List[torch.Tensor] = []
        simple_fused: List[torch.Tensor] = []
        for s, (ct, pet_cal, gate, rel) in enumerate(
            zip(ct_feats, calibrated, gates, reliability)
        ):
            pet_out = gate * pet_cal
            if self.config.use_retrieval_reliability:
                pet_out = pet_out * rel
            _finite_or_raise(f"pet_output_s{s + 1}", pet_out)
            pet_out = _sanitize(pet_out)
            fused = ct + pet_out
            _finite_or_raise(f"simple_fused_s{s + 1}", fused)
            simple_fused.append(_sanitize(fused))
            pet_output.append(pet_out)

        reliability_mean = float(
            torch.stack([r.float().mean() for r in reliability])
            .mean()
            .item()
       #这个是为了保存文件，所以故意做出的修改     
        )
        gate_mean = float(
            torch.stack([g.float().mean() for g in gates]).mean().item()
        )

        aux = {
            "pet_proxy": pet_proxy,
            "ct_reference": ct_ref,
            "attention": retrieval["attention"],
            "pet_gate": gates,
            "simple_fused": simple_fused,
            "retrieval_reliability_mean": reliability_mean,
            "pet_gate_mean": gate_mean,
            "bank_ready": self.bank_ready,
            "bank_version": int(self.bank_version.item()),
        }
        return pet_output, aux

    def forward(
        self,
        ct_feats: Sequence[torch.Tensor],
        pet_feats_real: Optional[Sequence[torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        mode: str = "missing",
        collect_candidates: Optional[bool] = None,
        compute_prototype_loss: bool = True,
    ) -> Dict:
        """
        Unified TRAINING convenience wrapper.

        For `mode='missing'`, real PET (when provided) is used ONLY for:
          a) detached prototype candidate collection, and
          b) privileged prototype supervision.
        The prediction path itself calls `recover_missing(ct_feats)`, which has
        no PET argument.
        """
        if mode not in {"full", "missing"}:
            raise ValueError("mode must be 'full' or 'missing'")
        self._validate_features(ct_feats, pet_feats_real)

        should_collect = (
            self.config.collect_candidates_during_training
            if collect_candidates is None
            else bool(collect_candidates)
        )
        collect_report = None
        if (
            self.training
            and should_collect
            and pet_feats_real is not None
            and mask is not None
        ):
            collect_report = self.collect_candidates(
                ct_feats, pet_feats_real, mask
            )

        if (
            self.training
            and compute_prototype_loss
            and pet_feats_real is not None
            and mask is not None
        ):
            proto_loss = self.compute_prototype_loss(
                ct_feats, pet_feats_real, mask
            )
        else:
            proto_loss = self._zero_loss_dict(ct_feats[0])

        if mode == "full":
            if pet_feats_real is None:
                raise ValueError("Full route requires real PET features")
            pred = self.prepare_full(ct_feats, pet_feats_real)
        else:
            pet_output, aux = self.recover_missing(ct_feats)
            pred = {
                "pet_output": pet_output,
                **aux,
            }

        pred["prototype_loss"] = proto_loss["loss"]
        pred["prototype_loss_weighted"] = proto_loss["weighted_loss"]
        pred["prototype_loss_num_terms"] = proto_loss["num_terms"]
        pred["prototype_loss_details"] = proto_loss["details"]
        pred["collect_report"] = collect_report
        return pred


# -----------------------------------------------------------------------------
# Minimal self-check
# -----------------------------------------------------------------------------


def _self_check() -> None:
    """Small CPU smoke test for shape/API/backward/leakage sanity."""
    torch.manual_seed(7)
    channels = (8, 12, 16, 20)
    shapes = ((16, 16), (8, 8), (4, 4), (2, 2))
    module = PairedSemanticPrototypeImputation(
        channels=channels,
        num_clusters=3,
        build_stage=4,
        prototype_loss_type="pad_kl",
        prototype_loss_weight=0.01,
        prototype_temperature=0.1,
        bank_update_mode="direct",
        use_affine_calibration=True,
        use_pet_contribution_gate=True,
        use_retrieval_reliability=True,
    )
    module.train()

    # Collect enough candidates to build a bank.
    for _ in range(3):
        ct = [
            torch.randn(4, c, h, w, requires_grad=True)
            for c, (h, w) in zip(channels, shapes)
        ]
        pet = [
            torch.randn(4, c, h, w, requires_grad=True)
            for c, (h, w) in zip(channels, shapes)
        ]
        mask = torch.zeros(4, 1, 64, 64)
        mask[:, :, 18:46, 20:44] = 1.0
        module.collect_candidates(ct, pet, mask)

    report = module.finalize_epoch(epoch=1)
    assert report["status"] == "bank_updated"  # check 1
    assert module.bank_ready  # check 2

    b = 2
    ct = [
        torch.randn(b, c, h, w, requires_grad=True)
        for c, (h, w) in zip(channels, shapes)
    ]
    pet_a = [
        torch.randn(b, c, h, w, requires_grad=True)
        for c, (h, w) in zip(channels, shapes)
    ]
    pet_b = [
        torch.randn(b, c, h, w, requires_grad=True)
        for c, (h, w) in zip(channels, shapes)
    ]
    mask = torch.zeros(b, 1, 64, 64)
    mask[:, :, 16:48, 16:48] = 1.0

    full = module(ct, pet_a, mask, mode="full", collect_candidates=False)
    missing = module(ct, pet_a, mask, mode="missing", collect_candidates=False)
    missing_alt = module(
        ct, pet_b, mask, mode="missing",
        collect_candidates=False, compute_prototype_loss=False,
    )

    for s in range(4):
        # checks 3-6: pet_output / simple_fused shapes match CT at all scales
        assert full["pet_output"][s].shape == ct[s].shape
        assert missing["pet_output"][s].shape == ct[s].shape
        assert full["simple_fused"][s].shape == ct[s].shape
        assert missing["simple_fused"][s].shape == ct[s].shape
        # check 7: gates finite
        for out in (full, missing):
            for gate in out["pet_gate"] if "pet_gate" in out else []:
                assert bool(torch.isfinite(gate).all())
        for gate in full["pet_gate"]:
            assert bool(torch.isfinite(gate).all())  # check 7
            assert bool((gate > 0.0).all() and (gate < 2.0).all())  # check 10
        for gate in missing["pet_gate"] + missing_alt["pet_gate"]:
            assert bool(torch.isfinite(gate).all())
            assert bool((gate > 0.0).all() and (gate < 2.0).all())

    retrieval = module.retrieve(ct, return_attention=False)
    for rel in retrieval["retrieval_reliability"]:
        assert bool(torch.isfinite(rel).all())
        assert bool((rel >= 0.0).all() and (rel <= 1.0).all())

    # Zero-init identity: gamma=beta=0 and gate=1, so the Full route passes real
    # PET through unchanged and the Missing route is R * prototype PET.
    for s in range(4):
        assert torch.allclose(
            full["pet_output"][s], pet_a[s].detach(), atol=1e-5
        )
        proxy = retrieval["pet_proxy"][s].detach()
        rel = retrieval["retrieval_reliability"][s].detach()
        expected = rel * proxy
        assert torch.allclose(
            missing["pet_output"][s], expected, atol=1e-5
        )

    # check 13 (Module-1 level): the missing prediction path is independent of
    # the real PET that is passed for candidate/teacher purposes only.
    for s in range(4):
        assert torch.allclose(
            missing["pet_output"][s],
            missing_alt["pet_output"][s],
            atol=1e-6,
        )
        assert torch.allclose(
            missing["simple_fused"][s],
            missing_alt["simple_fused"][s],
            atol=1e-6,
        )
    assert missing["retrieval_reliability_mean"] == (
        missing_alt["retrieval_reliability_mean"]
    )
    assert missing["pet_gate_mean"] == missing_alt["pet_gate_mean"]

    # check 12: recover_missing() takes no PET argument at all.
    pet_out_no_pet, aux_no_pet = module.recover_missing(ct)
    assert len(pet_out_no_pet) == 4
    assert "simple_fused" in aux_no_pet
    for s in range(4):
        assert torch.allclose(
            pet_out_no_pet[s], missing["pet_output"][s], atol=1e-6
        )
        assert torch.allclose(
            aux_no_pet["simple_fused"][s], missing["simple_fused"][s],
            atol=1e-6,
        )

    # check 11: backward works through both routes and the gate parameters.
    loss = (
        full["prototype_loss_weighted"]
        + missing["prototype_loss_weighted"]
        + sum(x.mean() for x in full["pet_output"])
        + sum(x.mean() for x in missing["pet_output"])
    )
    loss.backward()

    # Equivalence smoke test: with gate/reliability switches off, Module-1 must
    # reproduce the previous PSPI behavior exactly:
    #   pet_output == calibrated PET, simple_fused == CT + calibrated PET.
    equiv = PairedSemanticPrototypeImputation(
        channels=channels,
        num_clusters=3,
        build_stage=4,
        prototype_loss_type="pad_kl",
        prototype_loss_weight=0.01,
        prototype_temperature=0.1,
        bank_update_mode="direct",
        use_affine_calibration=True,
        use_pet_contribution_gate=False,
        use_retrieval_reliability=False,
    )
    equiv.load_state_dict(module.state_dict())
    # Non-trivial head weights so identity shortcuts cannot hide a wiring bug.
    with torch.no_grad():
        for head in equiv.affine_calibration.heads:
            nn.init.normal_(head[-1].weight, std=0.02)
            nn.init.normal_(head[-1].bias, std=0.02)
    equiv.train()

    full_eq = equiv(ct, pet_a, mask, mode="full", collect_candidates=False)
    ref_cal_full, _ = equiv.affine_calibration(
        ct,
        pet_a,
        full_eq["ct_reference"],
        reference_valid=True,
        use_pet_contribution_gate=False,
    )
    for s in range(4):
        assert torch.allclose(
            full_eq["pet_output"][s], ref_cal_full[s], atol=1e-5
        )
        assert torch.allclose(
            full_eq["simple_fused"][s], ct[s] + ref_cal_full[s], atol=1e-5
        )

    miss_eq, miss_eq_aux = equiv.recover_missing(ct)
    retrieval_eq = equiv.retrieve(ct, return_attention=False)
    ref_cal_miss, _ = equiv.affine_calibration(
        ct,
        retrieval_eq["pet_proxy"],
        retrieval_eq["ct_reference"],
        reference_valid=True,
        use_pet_contribution_gate=False,
    )
    for s in range(4):
        assert torch.allclose(
            miss_eq[s], ref_cal_miss[s], atol=1e-5
        )
        assert torch.allclose(
            miss_eq_aux["simple_fused"][s], ct[s] + ref_cal_miss[s], atol=1e-5
        )

    print("[SELF-CHECK] passed")


if __name__ == "__main__":
    _self_check()
