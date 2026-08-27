"""Per-scale Full-Anchored Affine Gradient Alignment (ANGA-adapted).

Applies the official ANGA three-branch cone projection only to
``PrototypeReferencedPETAffineCalibration.heads[0:4]``.

Task-specific rule (beyond vanilla ANGA on all params):
if the Full anchor norm is ~0, Missing affine grads are zeroed because there
is no reliable Full direction to anchor against.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

EPS = 1e-12


def get_affine_head_param_groups(pet_calibration: nn.Module) -> List[List[nn.Parameter]]:
    """Return four mutually exclusive parameter lists, one per affine head."""
    heads = getattr(pet_calibration, "heads", None)
    if heads is None or len(heads) != 4:
        raise ValueError(
            f"Expected pet_calibration.heads with 4 scales, got {type(heads)} "
            f"len={0 if heads is None else len(heads)}"
        )
    groups: List[List[nn.Parameter]] = []
    for head in heads:
        params = [p for p in head.parameters() if p.requires_grad]
        groups.append(params)
    return groups


def _zeros_like_param(p: nn.Parameter) -> torch.Tensor:
    return torch.zeros_like(p)


def snapshot_param_grads(params: Sequence[nn.Parameter]) -> List[torch.Tensor]:
    """Clone grads; None -> zeros_like(param) to keep positional alignment."""
    out: List[torch.Tensor] = []
    for p in params:
        if p.grad is None:
            out.append(_zeros_like_param(p).detach())
        else:
            out.append(p.grad.detach().clone())
    return out


def clear_param_grads(params: Sequence[nn.Parameter]) -> None:
    for p in params:
        p.grad = None


def write_param_grads(params: Sequence[nn.Parameter], grads: Sequence[torch.Tensor]) -> None:
    if len(params) != len(grads):
        raise ValueError(f"params/grads length mismatch: {len(params)} vs {len(grads)}")
    for p, g in zip(params, grads):
        if g is None:
            p.grad = torch.zeros_like(p)
        else:
            p.grad = g.to(device=p.device, dtype=p.dtype)


def _flat_float(grads: Sequence[torch.Tensor]) -> torch.Tensor:
    if not grads:
        return torch.zeros((), dtype=torch.float32)
    flats = [g.detach().float().reshape(-1) for g in grads]
    return torch.cat(flats, dim=0) if flats else torch.zeros(1, dtype=torch.float32)


def project_missing_onto_full_cone(
    g_full: Sequence[torch.Tensor],
    g_missing: Sequence[torch.Tensor],
    tau: float = 0.7,
    eps: float = EPS,
) -> tuple[List[torch.Tensor], Dict[str, float]]:
    """ANGA cone projection for one affine-head parameter group.

    Returns (g_missing_aligned, stats).
    """
    if not (0.0 < float(tau) < 1.0):
        raise ValueError(f"tau must satisfy 0 < tau < 1, got {tau}")
    if len(g_full) != len(g_missing):
        raise ValueError("g_full/g_missing length mismatch")

    device = g_full[0].device if g_full else torch.device("cpu")
    gF = [g.detach().float() for g in g_full]
    gM = [g.detach().float() for g in g_missing]

    gF_sq = torch.zeros((), device=device, dtype=torch.float32)
    for g in gF:
        gF_sq = gF_sq + (g * g).sum()
    gF_norm = gF_sq.sqrt()

    gM_sq = torch.zeros((), device=device, dtype=torch.float32)
    for g in gM:
        gM_sq = gM_sq + (g * g).sum()
    gM_norm = gM_sq.sqrt()

    stats = {
        "cosine": 0.0,
        "full_grad_norm": float(gF_norm.item()),
        "missing_grad_norm": float(gM_norm.item()),
        "aligned_missing_norm": 0.0,
        "zero": 0.0,
        "project": 0.0,
        "inside": 0.0,
    }

    # No reliable Full anchor -> forbid Missing-only affine motion.
    if float(gF_norm.item()) <= float(eps):
        aligned = [torch.zeros_like(g) for g in g_missing]
        stats["zero"] = 1.0
        stats["aligned_missing_norm"] = 0.0
        stats["cosine"] = 0.0
        return aligned, stats

    a = [g / (gF_norm + eps) for g in gF]

    dot = torch.zeros((), device=device, dtype=torch.float32)
    for gm, ah in zip(gM, a):
        dot = dot + (gm * ah).sum()
    cos_val = (dot / (gM_norm + eps)).clamp(-1.0, 1.0)
    stats["cosine"] = float(cos_val.item())

    if float(cos_val.item()) <= 0.0:
        aligned_f = [torch.zeros_like(g) for g in gM]
        stats["zero"] = 1.0
    elif float(cos_val.item()) >= float(tau):
        aligned_f = gM
        stats["inside"] = 1.0
    else:
        g_par = [dot * ah for ah in a]
        g_perp = [gm - gp for gm, gp in zip(gM, g_par)]
        t_max = torch.sqrt(torch.tensor(1.0, device=device, dtype=torch.float32) - float(tau) ** 2) / (
            float(tau) + eps
        )
        gpar_norm = torch.abs(dot)
        gperp_sq = torch.zeros((), device=device, dtype=torch.float32)
        for gp in g_perp:
            gperp_sq = gperp_sq + (gp * gp).sum()
        gperp_norm = gperp_sq.sqrt().clamp_min(eps)
        scale = t_max * gpar_norm / (gperp_norm + eps)
        aligned_f = [gp + scale * gq for gp, gq in zip(g_par, g_perp)]
        stats["project"] = 1.0

    aligned_sq = torch.zeros((), device=device, dtype=torch.float32)
    for g in aligned_f:
        aligned_sq = aligned_sq + (g * g).sum()
    stats["aligned_missing_norm"] = float(aligned_sq.sqrt().item())

    # Restore original dtype/device of the Missing snapshot tensors.
    aligned = [
        g.to(device=src.device, dtype=src.dtype) for g, src in zip(aligned_f, g_missing)
    ]
    return aligned, stats


def merge_affine_grads_per_scale(
    head_groups: Sequence[Sequence[nn.Parameter]],
    grads_full: Sequence[Sequence[torch.Tensor]],
    grads_missing: Sequence[Sequence[torch.Tensor]],
    mode: str,
    tau: float = 0.7,
    eps: float = EPS,
) -> Dict[str, float]:
    """Write ``gF + aligned(gM)`` (or ``gF+gM``) back into each head group.

    ``mode``:
      - ``joint``: affine.grad = gF + gM
      - ``anga``:  affine.grad = gF + ANGA(gM | gF)
    """
    if len(head_groups) != 4 or len(grads_full) != 4 or len(grads_missing) != 4:
        raise ValueError("Expected 4 affine scales")
    mode = str(mode).strip().lower()
    if mode not in ("joint", "anga"):
        raise ValueError(f"Unsupported merge mode={mode!r}")

    summary: Dict[str, float] = {}
    zero_sum = project_sum = inside_sum = 0.0
    cos_sum = 0.0

    for scale_idx, (params, gF, gM) in enumerate(
        zip(head_groups, grads_full, grads_missing), start=1
    ):
        if mode == "joint":
            aligned = list(gM)
            # Diagnostics still report raw cosine / norms.
            gF_flat = _flat_float(gF)
            gM_flat = _flat_float(gM)
            gF_norm = float(gF_flat.norm().item())
            gM_norm = float(gM_flat.norm().item())
            cos = float(
                torch.dot(gF_flat, gM_flat)
                / (gF_flat.norm() * gM_flat.norm() + eps)
            ) if gF_norm > eps and gM_norm > eps else 0.0
            branch = {"zero": 0.0, "project": 0.0, "inside": 1.0}
            aligned_norm = gM_norm
            stats = {
                "cosine": cos,
                "full_grad_norm": gF_norm,
                "missing_grad_norm": gM_norm,
                "aligned_missing_norm": aligned_norm,
                **branch,
            }
        else:
            aligned, stats = project_missing_onto_full_cone(gF, gM, tau=tau, eps=eps)

        merged = [
            gf.to(dtype=gf.dtype, device=gf.device) + gm.to(dtype=gf.dtype, device=gf.device)
            for gf, gm in zip(gF, aligned)
        ]
        write_param_grads(params, merged)

        prefix = f"affine_s{scale_idx}"
        summary[f"{prefix}_cosine"] = float(stats["cosine"])
        summary[f"{prefix}_full_grad_norm"] = float(stats["full_grad_norm"])
        summary[f"{prefix}_missing_grad_norm"] = float(stats["missing_grad_norm"])
        summary[f"{prefix}_aligned_missing_norm"] = float(stats["aligned_missing_norm"])
        summary[f"{prefix}_zero_ratio"] = float(stats["zero"])
        summary[f"{prefix}_project_ratio"] = float(stats["project"])
        summary[f"{prefix}_inside_ratio"] = float(stats["inside"])

        zero_sum += float(stats["zero"])
        project_sum += float(stats["project"])
        inside_sum += float(stats["inside"])
        cos_sum += float(stats["cosine"])

    summary["affine_mean_cosine"] = cos_sum / 4.0
    summary["affine_mean_zero_ratio"] = zero_sum / 4.0
    summary["affine_mean_project_ratio"] = project_sum / 4.0
    summary["affine_mean_inside_ratio"] = inside_sum / 4.0
    # Conflict proxy: fraction of scales that were not "inside" the cone.
    summary["affine_mean_conflict_ratio"] = 1.0 - summary["affine_mean_inside_ratio"]
    return summary


def snapshot_all_affine_grads(head_groups: Sequence[Sequence[nn.Parameter]]) -> List[List[torch.Tensor]]:
    return [snapshot_param_grads(group) for group in head_groups]


def clear_all_affine_grads(head_groups: Sequence[Sequence[nn.Parameter]]) -> None:
    for group in head_groups:
        clear_param_grads(group)
