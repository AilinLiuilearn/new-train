# -*- coding: utf-8 -*-
"""Full-only model wiring the background-referenced region fusion.

Subclasses :class:`DualSharedAddPETCTBaseline` so encoders (ConvNeXtV2-Nano
CT, MiT-B1 PET), pretrained loading, ``ct_align``/``pet_align``, the shared
GroupNorm decoder and training state are inherited unchanged. Only
``self.fusion`` is replaced, before any optimizer/EMA exists.

Full-only: ``pet`` is required; ``missing``/``auto`` modes and any
non-Full ``pet_available`` state raise instead of falling back to the old
addition path. ``pet_raw`` is recovered from the same augmented/normalized
PET tensor via :func:`pet_to_unit_interval` with the real ``norm_mode``.
"""
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from models.background_referenced_region_fusion import (
    BackgroundReferencedRegionFusion,
    FusionResult,
    pet_to_unit_interval,
)
from models.dual_shared_add_baseline import (
    DECODER_INPUT_CHANNELS,
    DualSharedAddPETCTBaseline,
)

MODEL_ARCH = 'full_background_referenced'


class FullBackgroundReferencedPETCTModel(DualSharedAddPETCTBaseline):
    def __init__(self, *args, norm_mode: str = 'imagenet',
                 fusion_kwargs: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if norm_mode not in ('imagenet', 'cipa', 'unit'):
            raise ValueError(f'unsupported norm_mode={norm_mode!r}')
        self.norm_mode = norm_mode
        fusion_kwargs = dict(fusion_kwargs or {})
        fusion_kwargs.setdefault('channels', tuple(DECODER_INPUT_CHANNELS))
        if tuple(fusion_kwargs['channels']) != tuple(DECODER_INPUT_CHANNELS):
            raise ValueError('fusion channels must be the aligned '
                             f'{tuple(DECODER_INPUT_CHANNELS)}')
        self.fusion = BackgroundReferencedRegionFusion(**fusion_kwargs)

    def fusion_config(self) -> Dict[str, Any]:
        return {'model_arch': MODEL_ARCH, 'norm_mode': self.norm_mode,
                **self.fusion.get_config()}

    @staticmethod
    def _require_all_full(pet_available: Any, batch: int, device: torch.device) -> None:
        if pet_available is None:
            return
        state = torch.as_tensor(pet_available, device=device)
        if state.ndim != 1 or state.numel() != batch or state.is_floating_point():
            raise ValueError('pet_available must be bool/0-1 integer [B]')
        if not bool((state == 1).all()):
            raise ValueError('Full-only model requires all-Full pet_available')

    def forward(self, ct: Tensor, pet: Optional[Tensor] = None,
                pet_available: Any = None, target_size: Any = None,
                forward_mode: str = 'full', *,
                background_train_mask: Optional[Tensor] = None,
                return_diagnostics: bool = False) -> Dict[str, Any]:
        if forward_mode != 'full':
            raise ValueError(f'Full-only model rejects forward_mode={forward_mode!r}')
        if pet is None:
            raise ValueError('Full path requires a valid PET tensor, got pet=None')
        self._require_all_full(pet_available, ct.shape[0], ct.device)
        if target_size is None:
            target_size = tuple(ct.shape[-2:])
        ct_features = self._encode_ct(ct)
        pet_features = self._encode_pet(pet)
        pet_raw = pet_to_unit_interval(pet, mode=self.norm_mode)
        result: FusionResult = self.fusion(
            ct_features, pet_features, pet_raw,
            background_train_mask=background_train_mask,
            return_diagnostics=bool(return_diagnostics),
        )
        outputs = self._decode(result.fused, target_size)
        outputs['background_loss'] = result.background_loss
        if return_diagnostics:
            outputs['fusion_diagnostics'] = result.diagnostics
        return outputs


def build_full_background_referenced_model(cfg):
    """Task-compatible builder: ``{'model': model}``, same encoder rules."""
    def _get(name, default):
        return getattr(cfg, name, default)

    fusion_kwargs: Dict[str, Any] = {
        'channels': tuple(DECODER_INPUT_CHANNELS),
        'descriptor_dim': int(_get('brlc_descriptor_dim', 32)),
        'region_size': int(_get('brlc_region_size', 8)),
        'region_stride': int(_get('brlc_region_stride', 4)),
        'window_chunk': int(_get('brlc_window_chunk', 256)),
        'evidence_pool': str(_get('brlc_evidence_pool', 'signed_peak')),
        'bg_kernels': tuple(_get('brlc_bg_kernels', (9, 17))),
        'bg_hole': int(_get('brlc_bg_hole', 5)),
        'bg_ring_channels': int(_get('brlc_bg_ring_channels', 4)),
        'bg_hidden': int(_get('brlc_bg_hidden', 16)),
        'sigma_min': float(_get('brlc_sigma_min', 0.01)),
        'sigma_max': float(_get('brlc_sigma_max', 0.5)),
        'temperature_min': float(_get('brlc_temperature_min', 0.1)),
        'temperature_max': float(_get('brlc_temperature_max', 10.0)),
        'eta_max': float(_get('brlc_eta_max', 4.0)),
        'prior_logit_limit': float(_get('brlc_prior_logit_limit', 6.0)),
        'loo_min_mass': float(_get('brlc_loo_min_mass', 1e-4)),
        'checkpoint_windows': bool(_get('brlc_checkpoint_windows', True)),
        'checkpoint_background': bool(_get('brlc_checkpoint_background', True)),
        'validate_values': bool(_get('brlc_validate_values', True)),
    }
    model = FullBackgroundReferencedPETCTModel(
        ct_backbone=_get('ct_backbone', 'convnextv2_nano'),
        pet_backbone=_get('pet_backbone', 'mit_b1'),
        ct_pretrained_path=_get('ct_pretrained_path', None),
        pet_pretrained_path=_get('pet_pretrained_path', None),
        in_channels=3,
        out_channels=1,
        decoder_channels=tuple(_get('decoder_channels', (512, 256, 128, 64))),
        use_deep_supervision=bool(_get('use_deep_supervision', False)
                                  or _get('deep_supervision', False)),
        pretrained=bool(_get('pretrained', True)),
        norm_mode=str(_get('norm_mode', 'imagenet')),
        fusion_kwargs=fusion_kwargs,
    )
    if not isinstance(model.fusion, BackgroundReferencedRegionFusion):
        raise RuntimeError('expected BackgroundReferencedRegionFusion, '
                           f'got {type(model.fusion).__name__}')
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f'[background_referenced] fusion_class={type(model.fusion).__name__} '
          f'channels={tuple(DECODER_INPUT_CHANNELS)} '
          f'fusion_params={fusion_params} params_total={total_params}', flush=True)
    return {'model': model}
