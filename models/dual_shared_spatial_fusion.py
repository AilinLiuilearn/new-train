# -*- coding: utf-8 -*-
"""Dual-modal baseline with the spatial bidirectional fusion on the Full path.

:class:`DualSharedSpatialFusionPETCTModel` subclasses the clean
:class:`DualSharedAddPETCTBaseline` so encoders, channel alignment and the
shared decoder are constructed identically; only ``self.fusion`` is swapped
for :class:`MultiScaleSpatialBidirectionalFusion` before any optimizer or EMA
is built. Forward signature and the full/missing/auto dataflow are inherited
unchanged:

- Full / auto-Full rows: fused by the spatial module (replaces AddFusion).
- Missing: parent bypass — PET is never encoded, the fusion is never called,
  aligned CT goes straight to the shared decoder.

``fusion_enabled=False`` keeps the original AddFusion for equivalence checks.
"""
from models.components.add_fusion import AddFusion
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.spatial_bidirectional_fusion import MultiScaleSpatialBidirectionalFusion

MODEL_ARCH = 'dual_shared_spatial_fusion'

DEFAULT_FUSION_KWARGS = dict(
    channels=(64, 128, 320, 512),
    attention_dim=64,
    num_heads=4,
    local_kernel_size=5,
    max_axis_length=128,
    axis_chunk_size=32,
    mode='full',
    use_checkpoint=True,
    check_finite=True,
    resize_pet=True,
)


class DualSharedSpatialFusionPETCTModel(DualSharedAddPETCTBaseline):
    def __init__(self, *args, fusion_enabled=True, fusion_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fusion_enabled = bool(fusion_enabled)
        self.fusion_kwargs = dict(DEFAULT_FUSION_KWARGS)
        if fusion_kwargs:
            self.fusion_kwargs.update(fusion_kwargs)
        if self.fusion_kwargs.get('mode', 'full') not in ('full', 'local', 'axial', 'add'):
            raise ValueError(
                f"fusion mode must be full/local/axial/add, "
                f"got {self.fusion_kwargs.get('mode')!r}")
        if self.fusion_enabled:
            self.fusion = MultiScaleSpatialBidirectionalFusion(**self.fusion_kwargs)
        else:
            self.fusion = AddFusion()
        self.fusion_mode = self.fusion_kwargs.get('mode', 'full')

    def fusion_config(self):
        """Record of the fusion actually built (for checkpoint reproducibility)."""
        return {
            'model_arch': MODEL_ARCH,
            'fusion_enabled': self.fusion_enabled,
            'fusion_class': type(self.fusion).__name__,
            **{k: (list(v) if isinstance(v, (list, tuple)) else v)
                for k, v in self.fusion_kwargs.items()},
        }