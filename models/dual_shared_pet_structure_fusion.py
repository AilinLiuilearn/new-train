# -*- coding: utf-8 -*-
"""Dual-modal baseline with MultiScalePETGuidedStructureFusion on Full path.

Subclasses :class:`DualSharedAddPETCTBaseline` so encoders, pretrained
loading, channel alignment, decoder, routing and the output dict are
inherited unchanged. Only ``self.fusion`` is swapped after the parent
constructor completes (hence after the decoder is built, preserving the
baseline RNG consumption order), and before any task/optimizer/EMA exists.

- ``fusion_type='add'`` keeps the original ``AddFusion`` untouched: no
  candidate module is constructed, no extra RNG consumed, same parameters.
- ``fusion_type='pet_structure'`` installs the candidate pyramid.
- Full / auto-Full rows: ``self.fusion(ct_feats, pet_feats)`` (four-scale
  list, AddFusion-compatible) -> shared decoder.
- Missing: parent bypass — PET never encoded, fusion never called.

The default forward list is used; ``ct_enhanced``/``pet_enhanced`` are never
stored in ``aux``.
"""
from models.components.add_fusion import AddFusion
from models.dual_shared_add_baseline import (
    DECODER_INPUT_CHANNELS,
    DualSharedAddPETCTBaseline,
)
from models.pet_guided_structure_fusion import MultiScalePETGuidedStructureFusion

MODEL_ARCH = 'dual_shared_pet_structure_fusion'

DEFAULT_FUSION_KWARGS = dict(
    inner_channels=32,
    heads=4,
    kernel_size=5,
    chunk_rows=16,
    checkpoint_chunks=True,
    beta_init=0.0,
    structure_strength_init=0.1,
    use_pet_guidance=True,
    use_structure_constraint=True,
    ct_update_type='difference',
    check_finite=True,
)


class DualSharedPETStructureFusionModel(DualSharedAddPETCTBaseline):
    def __init__(self, *args, fusion_type='pet_structure', fusion_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        if fusion_type not in ('add', 'pet_structure'):
            raise ValueError(
                f"fusion_type must be 'add' or 'pet_structure', got {fusion_type!r}")
        self.fusion_type = fusion_type
        self.fusion_kwargs = dict(DEFAULT_FUSION_KWARGS)
        if fusion_kwargs:
            self.fusion_kwargs.update(fusion_kwargs)
        if fusion_type == 'pet_structure':
            self.fusion = MultiScalePETGuidedStructureFusion(
                channels=DECODER_INPUT_CHANNELS, **self.fusion_kwargs)
        else:
            assert isinstance(self.fusion, AddFusion)

    def fusion_config(self):
        """Record of the fusion actually built (for checkpoint reproducibility)."""
        return {
            'model_arch': MODEL_ARCH,
            'fusion_type': self.fusion_type,
            'fusion_class': type(self.fusion).__name__,
            **{k: (list(v) if isinstance(v, (list, tuple)) else v)
                for k, v in self.fusion_kwargs.items()},
        }