"""Full-only LSConv experiment, derived from the aligned Add baseline.

No missing-PET compensation is introduced in this Stage-1 experiment.
All original encoder/aligner/decoder state keys and initialization are retained.
"""
import torch

from models.components.lsconv_fusion import LSConvFusion
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline, _validate_state

BASELINE_COMMIT = '663f9786122365d8b2774e4497df92bc06b715d5'
UPSTREAM_COMMIT = 'cbe737c92b7c43ecf02d08545a07f03f1010177c'
CONFIG_FIELDS = ('lsconv_enabled', 'lsconv_backend')


class FullLSConvPETCT(DualSharedAddPETCTBaseline):
    def __init__(self, lsconv_enabled=True, lsconv_backend='auto', **kwargs):
        # Build all baseline weights first, before drawing new fusion weights.
        super().__init__(**kwargs)
        # Extra initialization must not shift subsequent loader/dropout RNG.
        with torch.random.fork_rng(devices=[]):
            self.fusion = LSConvFusion(lsconv_enabled, lsconv_backend)

    def forward(self, ct, pet=None, pet_available=None, target_size=None,
                forward_mode='full'):
        if forward_mode not in ('full', 'auto'):
            raise ValueError('LSConv experiment is Full-only; Missing is not trained')
        if pet_available is not None:
            state = _validate_state(pet_available, ct.shape[0])
            if not bool(torch.all(state == 1)):
                raise ValueError('Real PET is required for every sample')
        if pet is None or pet.shape != ct.shape:
            raise ValueError('Matching CT and real PET tensors are required')
        return super().forward(ct, pet=pet, target_size=target_size, forward_mode='full')


def build_full_lsconv_model(cfg):
    missing = [name for name in CONFIG_FIELDS if not hasattr(cfg, name)]
    if missing:
        raise ValueError(f'Explicit saved LSConv flags are required: {missing}')
    model = FullLSConvPETCT(
        **{name: getattr(cfg, name) for name in CONFIG_FIELDS},
        ct_backbone=getattr(cfg, 'ct_backbone', 'convnextv2_nano'),
        pet_backbone=getattr(cfg, 'pet_backbone', 'mit_b1'),
        ct_pretrained_path=getattr(cfg, 'ct_pretrained_path', None),
        pet_pretrained_path=getattr(cfg, 'pet_pretrained_path', None),
        in_channels=3, out_channels=1,
        decoder_channels=tuple(getattr(cfg, 'decoder_channels', (512, 256, 128, 64))),
        use_deep_supervision=bool(getattr(cfg, 'use_deep_supervision', False)
                                  or getattr(cfg, 'deep_supervision', False)),
        pretrained=bool(getattr(cfg, 'pretrained', True)))
    print(f'[full_lsconv] enabled={cfg.lsconv_enabled} backend={cfg.lsconv_backend} '
          f'scales=S1,S2,S3,S4 lks=7 sks=3 groups=8 final_bn_init=zeros '
          f'params_fusion={sum(p.numel() for p in model.fusion.parameters())} '
          f'params_total={sum(p.numel() for p in model.parameters())}', flush=True)
    return {'model': model}
