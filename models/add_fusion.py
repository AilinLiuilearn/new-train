# -*- coding: utf-8 -*-
"""Pure per-scale addition fusion.

This is the explicit Full-path fusion seam (§九): a future fusion module only
needs to replace this call for the Full path. Missing never enters it; the
Missing path feeds ``aligned_CT`` straight to the shared decoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AddFusion(nn.Module):
    """Elementwise CT + PET per scale with strict shape validation.

    Spatial sizes are bilinearly aligned when they differ (necessary
    alignment, loudly applied per scale). Anything else raises: no silent
    ``zip`` truncation, no NaN masking.
    """

    def forward(self, ct_feats, pet_feats):
        if len(ct_feats) != len(pet_feats):
            raise ValueError(
                f'AddFusion scale count mismatch: ct={len(ct_feats)} pet={len(pet_feats)}')
        if len(ct_feats) != 4:
            raise ValueError(f'AddFusion expects 4 scales, got {len(ct_feats)}')
        fused = []
        for i, (ct_feat, pet_feat) in enumerate(zip(ct_feats, pet_feats)):
            if ct_feat.shape[1] != pet_feat.shape[1]:
                raise ValueError(
                    f'AddFusion scale {i} channel mismatch: '
                    f'ct={ct_feat.shape[1]} pet={pet_feat.shape[1]}')
            if pet_feat.shape[-2:] != ct_feat.shape[-2:]:
                pet_feat = F.interpolate(pet_feat, size=ct_feat.shape[-2:],
                                         mode='bilinear', align_corners=False)
            out = ct_feat + pet_feat
            if not torch.isfinite(out).all():
                raise RuntimeError(f'[NaN/Inf] AddFusion output scale {i} is non-finite')
            fused.append(out)
        return fused