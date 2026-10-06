# -*- coding: utf-8 -*-
"""Dual-modal baseline with LocalContrastFusionPyramid on the Full path.

Subclasses :class:`DualSharedAddPETCTBaseline` so encoders, pretrained
loading, channel alignment, decoder, output dict and routing are inherited
unchanged; only ``self.fusion`` is swapped during construction (before any
optimizer or EMA is built).

Inherited dataflow:
- Full / auto-Full rows: ``self.fusion(ct_feats, pet_feats)`` (four-scale
  list, AddFusion-compatible) -> shared decoder.
- Missing: parent bypass — PET never encoded, fusion never called, aligned
  CT straight to the decoder.

``forward_features`` triplets are never passed to the decoder and never
stored; training uses the plain four-scale ``forward`` list only.
"""
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.local_contrast_bidirectional_fusion import LocalContrastFusionPyramid

MODEL_ARCH = 'dual_shared_local_contrast_fusion'

DEFAULT_FUSION_KWARGS = dict(
    channels=(64, 128, 320, 512),
    dim=32,
    heads=4,
    window=5,
    chunk_rows=16,
    checkpoint_chunks=True,
)


class DualSharedLocalContrastFusionModel(DualSharedAddPETCTBaseline):
    def __init__(self, *args, fusion_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fusion_kwargs = dict(DEFAULT_FUSION_KWARGS)
        if fusion_kwargs:
            self.fusion_kwargs.update(fusion_kwargs)
        self.fusion = LocalContrastFusionPyramid(**self.fusion_kwargs)

    def fusion_config(self):
        """Record of the fusion actually built (for checkpoint reproducibility)."""
        return {
            'model_arch': MODEL_ARCH,
            'fusion_class': type(self.fusion).__name__,
            **{k: (list(v) if isinstance(v, (list, tuple)) else v)
                for k, v in self.fusion_kwargs.items()},
        }