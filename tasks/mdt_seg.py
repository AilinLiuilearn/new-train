# -*- coding: utf-8 -*-
import math
import os
import torch
import torch.nn.functional as F
from utils.metrics_seg import SegmentationMetricsCIPA
from utils.optimization import get_optimizer
from utils.seg_losses import BCEDiceLoss

try:
    from models.pet_prompted_ct_decoder import PET_PROMPT_LOG_KEYS
except ImportError:
    PET_PROMPT_LOG_KEYS = []

try:
    from models.pet_lap_hgl_prior import PET_LAP_HGL_LOG_KEYS
except ImportError:
    PET_LAP_HGL_LOG_KEYS = []

try:
    from models.pet_mrp_gsa import PET_MRP_GSA_LOG_KEYS
except ImportError:
    PET_MRP_GSA_LOG_KEYS = []


def mask_to_boundary(mask):
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = (mask > 0.5).float()
    kernel = torch.ones((1, 1, 3, 3), dtype=mask.dtype, device=mask.device)
    neighbor_sum = F.conv2d(mask, kernel, padding=1)
    return ((neighbor_sum > 0) & (neighbor_sum < 9)).float()


def _forward(nets, ct, pet, target_size, pet_available=None, return_aux=False, forward_mode=None, mask=None):
    kwargs = {'target_size': target_size}
    if pet_available is not None:
        kwargs['pet_available'] = pet_available
    if forward_mode is not None:
        kwargs['forward_mode'] = forward_mode
    if return_aux:
        kwargs['return_aux'] = return_aux
    if mask is not None:
        import inspect
        try:
            if 'mask' in inspect.signature(nets['model'].forward).parameters:
                kwargs['mask'] = mask
        except (TypeError, ValueError):
            pass
    return nets['model'](ct, pet, **kwargs)

# ... rest unchanged ...
