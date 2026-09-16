# -*- coding: utf-8 -*-
"""Foreground/background-balanced multi-scale Smooth-L1 PET feature reconstruction.

The ONLY auxiliary objective of the CT-affine experiment. The reconstruction
target is the *same sample's* detached real PET feature
(`pet_real.detach()`); the prediction is the post-affine `pet_comp` that is
actually consumed by the Missing fusion path.

Fixed definition (per scale l, over the current Missing sub-batch):
    target     = pet_real_missing.detach().float()          # [B_m,C,H,W]
    prediction = pet_comp_missing.float()                    # keeps grad
    scale      = target.square().mean().sqrt().clamp_min(1e-3).detach()
    error_map  = smooth_l1_loss(pred/scale, target/scale, beta=1.0,
                                reduction='none').mean(dim=1, keepdim=True)
    fg         = adaptive_avg_pool2d(mask_missing, (H,W)).clamp(0,1)
    bg         = 1 - fg
    L_fg       = (error_map*fg).sum() / fg.sum().clamp_min(1e-8)
    L_bg       = (error_map*bg).sum() / bg.sum().clamp_min(1e-8)

Per Missing sample: valid classes share 1/2 each; a single valid class takes
the full weight. Then average over samples, then equal-weight across the 4
scales. Class denominators are NEVER pooled across the batch, so small
lesions keep their proportional influence.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


EPS = 1e-8


def _finite_or_raise(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        bad = int((~torch.isfinite(x)).sum().item())
        raise RuntimeError(f"[PETReconstruction] {name} contains {bad} NaN/Inf values")


def balanced_multi_scale_smooth_l1_reconstruction(
    pet_comp: Sequence[torch.Tensor],
    pet_real: Sequence[torch.Tensor],
    mask_missing: torch.Tensor,
) -> Dict:
    """Compute the single auxiliary reconstruction loss on Missing rows only.

    Parameters
    ----------
    pet_comp:      4-scale post-affine PET features (grad kept), [B_m,C,H,W] each.
    pet_real:      4-scale real PET features of the SAME samples. Detached by the
                   helper itself; any pre-supplied graph is intentionally cut.
    mask_missing:  [B_m,1,H,W] segmentation GT of the Missing rows (soft-safe).

    Returns
    -------
    dict with:
      loss:         FP32 scalar (0.0-valued tensor on ref device when inactive)
      active:       bool; True iff there was at least one valid term
      num_samples:  int; number of Missing samples that contributed terms
      per_scale:    {"s{l}_fg": float, "s{l}_bg": float, "s{l}_rms": float, ...}
    """
    pet_comp = list(pet_comp)
    pet_real = list(pet_real)
    if len(pet_comp) != 4 or len(pet_real) != 4:
        raise ValueError(
            f"expected 4 scales each, got pet_comp={len(pet_comp)} pet_real={len(pet_real)}"
        )
    if mask_missing.ndim != 4 or mask_missing.shape[1] != 1:
        raise ValueError(f"mask_missing must be [B,1,H,W], got {tuple(mask_missing.shape)}")

    ref = pet_comp[0]
    b_m = int(mask_missing.shape[0])
    for s, (comp, real) in enumerate(zip(pet_comp, pet_real)):
        if comp.ndim != 4 or real.ndim != 4:
            raise ValueError(
                f"[Recon][S{s + 1}] features must be [B,C,H,W]: "
                f"comp={tuple(comp.shape)} real={tuple(real.shape)}"
            )
        if tuple(comp.shape) != tuple(real.shape):
            raise ValueError(
                f"[Recon][S{s + 1}] shape mismatch: comp={tuple(comp.shape)} real={tuple(real.shape)}"
            )
        if int(comp.shape[0]) != b_m:
            raise ValueError(
                f"[Recon][S{s + 1}] batch mismatch: comp={int(comp.shape[0])} mask={b_m}"
            )
        _finite_or_raise(f"pet_comp_s{s + 1}", comp)
        _finite_or_raise(f"pet_real_s{s + 1}", real)

    scale_terms: List[torch.Tensor] = []
    per_scale: Dict[str, float] = {}
    contributing_samples = 0

    if b_m == 0:
        zero = ref.new_zeros((), dtype=torch.float32)
        return {"loss": zero, "active": False, "num_samples": 0, "per_scale": {}}

    for s, (comp, real) in enumerate(zip(pet_comp, pet_real)):
        target = real.detach().float()
        prediction = comp.float()
        if not prediction.requires_grad:
            raise RuntimeError(
                f"[Recon][S{s + 1}] pet_comp has no grad_fn; the reconstruction "
                "loss must be applied to the post-affine Missing features"
            )
        # One non-learnable scalar per scale for loss-unit normalization only.
        scale = target.square().mean().sqrt().clamp_min(1e-3).detach()
        error_map = F.smooth_l1_loss(
            prediction / scale,
            target / scale,
            beta=1.0,
            reduction="none",
        ).mean(dim=1, keepdim=True)
        _finite_or_raise(f"error_map_s{s + 1}", error_map)

        fg = F.adaptive_avg_pool2d(
            mask_missing.to(device=comp.device).float(), prediction.shape[-2:]
        ).clamp(0.0, 1.0)
        bg = 1.0 - fg

        per_sample_terms: List[torch.Tensor] = []
        for i in range(b_m):
            err_i = error_map[i]
            fg_i = fg[i]
            bg_i = bg[i]
            fg_sum = fg_i.sum()
            bg_sum = bg_i.sum()
            class_terms: List[torch.Tensor] = []
            if float(fg_sum.item()) > EPS:
                class_terms.append((err_i * fg_i).sum() / fg_sum.clamp_min(1e-8))
            if float(bg_sum.item()) > EPS:
                class_terms.append((err_i * bg_i).sum() / bg_sum.clamp_min(1e-8))
            if not class_terms:
                continue
            # Valid classes share 1/2 each; a single valid class takes all.
            per_sample_terms.append(torch.stack(class_terms).mean())
        if not per_sample_terms:
            per_scale[f"s{s + 1}_fg"] = 0.0
            per_scale[f"s{s + 1}_bg"] = 0.0
            per_scale[f"s{s + 1}_rms"] = float(scale.item())
            per_scale[f"s{s + 1}_terms"] = 0
            continue
        contributing_samples = max(contributing_samples, len(per_sample_terms))
        scale_loss = torch.stack(per_sample_terms).mean()
        _finite_or_raise(f"scale_loss_s{s + 1}", scale_loss)
        scale_terms.append(scale_loss)

        with torch.no_grad():
            err_fg = float(((error_map * fg).sum() / fg.sum().clamp_min(1e-8)).item())
            err_bg = float(((error_map * bg).sum() / bg.sum().clamp_min(1e-8)).item())
        per_scale[f"s{s + 1}_fg"] = err_fg
        per_scale[f"s{s + 1}_bg"] = err_bg
        per_scale[f"s{s + 1}_rms"] = float(scale.item())
        per_scale[f"s{s + 1}_terms"] = len(per_sample_terms)

    if not scale_terms:
        zero = ref.new_zeros((), dtype=torch.float32)
        return {"loss": zero, "active": False, "num_samples": 0, "per_scale": per_scale}

    # Equal weight across the 4 scales; Missing-subset mean already applied.
    loss = torch.stack(scale_terms).mean()
    _finite_or_raise("reconstruction_loss", loss)
    return {
        "loss": loss,
        "active": True,
        "num_samples": contributing_samples,
        "per_scale": per_scale,
    }
