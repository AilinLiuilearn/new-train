# -*- coding: utf-8 -*-
"""Full-only experiment; reuse the exact Add baseline encoders/align/decoder."""
import torch

from models.components.amdanet_fusion import AMDANetFusion
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline, _validate_state


UPSTREAM_COMMIT = '88e4562eacc41c6d61377d1a34da894decc5a994'
BASELINE_COMMIT = '663f9786122365d8b2774e4497df92bc06b715d5'


class FullAMDANetPETCT(DualSharedAddPETCTBaseline):
    def __init__(self, **kwargs):
        # Construct the baseline first so, for the same seed, every common
        # parameter is initialized identically before the new fusion is built.
        super().__init__(**kwargs)
        self.fusion = AMDANetFusion()

    def forward(self, ct, pet=None, pet_available=None, target_size=None,
                forward_mode='full'):
        # Fail before encoding, rather than accidentally evaluating Missing
        # with an untrained CT-only route on this Full-only experimental model.
        if forward_mode not in ('full', 'auto'):
            raise ValueError('FullAMDANetPETCT supports Full only; Missing is not trained')
        if pet_available is not None:
            state = _validate_state(pet_available, ct.shape[0])
            if not bool(torch.all(state == 1)):
                raise ValueError('FullAMDANetPETCT requires real PET for every sample')
        if pet is None or pet.shape != ct.shape:
            raise ValueError('FullAMDANetPETCT requires matching CT/PET input tensors')
        return super().forward(ct, pet=pet, target_size=target_size, forward_mode='full')


def build_full_amdanet_model(cfg):
    model = FullAMDANetPETCT(
        ct_backbone=getattr(cfg, 'ct_backbone', 'convnextv2_nano'),
        pet_backbone=getattr(cfg, 'pet_backbone', 'mit_b1'),
        ct_pretrained_path=getattr(cfg, 'ct_pretrained_path', None),
        pet_pretrained_path=getattr(cfg, 'pet_pretrained_path', None),
        in_channels=3, out_channels=1,
        decoder_channels=tuple(getattr(cfg, 'decoder_channels', (512, 256, 128, 64))),
        use_deep_supervision=bool(getattr(cfg, 'use_deep_supervision', False)
                                  or getattr(cfg, 'deep_supervision', False)),
        pretrained=bool(getattr(cfg, 'pretrained', True)),
    )
    print('[full_amdanet] fusion=CT+PET+FDAM(CT,PET) '
          f'ct={getattr(cfg, "ct_backbone", "convnextv2_nano")} '
          f'pet={getattr(cfg, "pet_backbone", "mit_b1")} '
          f'upstream={UPSTREAM_COMMIT} '
          f'params_fusion={sum(p.numel() for p in model.fusion.parameters())} '
          f'params_total={sum(p.numel() for p in model.parameters())}', flush=True)
    return {'model': model}
