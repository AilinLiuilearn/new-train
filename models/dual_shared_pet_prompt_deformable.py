# -*- coding: utf-8 -*-
"""Dual-modal baseline with MultiScalePETPromptDeformableFusion on Full path.

Subclasses :class:`DualSharedAddPETCTBaseline` so encoders, pretrained
loading, channel alignment, decoder, routing and the output dict are
inherited unchanged. Only ``self.fusion`` is swapped after the parent
constructor completes (hence after the decoder is built, preserving the
baseline RNG consumption order), and before any task/optimizer/EMA exists.

- ``fusion_type='pet_prompt'`` installs the candidate pyramid. Its
  initialization runs inside ``torch.random.fork_rng(devices=[])`` so later
  data sampling RNG state is preserved.
- ``fusion_type='add'`` keeps the original ``AddFusion`` untouched: no
  candidate module is constructed, no extra RNG consumed, same parameters.
- Full / auto-Full rows: ``self.fusion(ct_feats, pet_feats)`` (four-scale
  list, AddFusion-compatible) -> shared decoder.
- Missing: parent bypass — PET never encoded, fusion never called.
"""
import torch

from models.components.add_fusion import AddFusion
from models.dual_shared_add_baseline import (
    DECODER_INPUT_CHANNELS,
    DualSharedAddPETCTBaseline,
)
from models.pet_prompt_deformable_fusion import MultiScalePETPromptDeformableFusion

MODEL_ARCH = 'dual_shared_pet_prompt_deformable'

DEFAULT_FUSION_KWARGS = dict(
    inner_channels=32,
    heads=4,
    offset_radius=2.0,
    chunk_rows=16,
    checkpoint_chunks=True,
    use_gaussian=True,
    use_pet_prompt=True,
    use_deformable=True,
    use_ct_correction=True,
    check_finite=True,
)


class DualSharedPETPromptDeformableModel(DualSharedAddPETCTBaseline):
    def __init__(self, *args, fusion_type='pet_prompt', fusion_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        if fusion_type not in ('add', 'pet_prompt'):
            raise ValueError(
                f"fusion_type must be 'add' or 'pet_prompt', got {fusion_type!r}")
        self.fusion_type = fusion_type
        self.fusion_kwargs = dict(DEFAULT_FUSION_KWARGS)
        if fusion_kwargs:
            self.fusion_kwargs.update(fusion_kwargs)
        if fusion_type == 'pet_prompt':
            with torch.random.fork_rng(devices=[]):
                self.fusion = MultiScalePETPromptDeformableFusion(
                    channels=DECODER_INPUT_CHANNELS, **self.fusion_kwargs)
        else:
            assert isinstance(self.fusion, AddFusion)

    def forward_with_fusion_diagnostics(self, ct, pet, target_size=None):
        """Full-only diagnostics: (outputs_dict, per-scale diagnostics list).

        Uses the same encode/fuse/decode path as training; diagnostics are
        detached inside the fusion. ``add`` mode has no soft prompts.
        """
        if self.fusion_type != 'pet_prompt':
            raise ValueError('Diagnostics require fusion_type=pet_prompt, '
                             f'got {self.fusion_type!r}')
        ct_feats = self._encode_ct(ct)
        pet_feats = self._encode_pet(pet)
        fused, diagnostics = self.fusion.forward_with_diagnostics(ct_feats, pet_feats)
        outputs = self._decode(
            fused, ct.shape[-2:] if target_size is None else target_size)
        return outputs, diagnostics

    def fusion_config(self):
        """Record of the fusion actually built (for checkpoint reproducibility)."""
        return {
            'model_arch': MODEL_ARCH,
            'fusion_type': self.fusion_type,
            'fusion_class': type(self.fusion).__name__,
            **{k: (list(v) if isinstance(v, (list, tuple)) else v)
                for k, v in self.fusion_kwargs.items()},
        }
