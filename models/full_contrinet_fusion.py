"""Full-only official ConTriNet MFM on the aligned PET-CT Add baseline."""
import torch

from models.components.contrinet_fusion import ConTriNetMFMFusion
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline, _validate_state

BASELINE_COMMIT = '663f9786122365d8b2774e4497df92bc06b715d5'
UPSTREAM_COMMIT = '9af315d4a93d897cf0b9abb11e194d1ed1072ff9'
CONFIG_FIELDS = ('mfm_enabled', 'mfm_cross_scale', 'mfm_output_mode')


class FullConTriNetPETCT(DualSharedAddPETCTBaseline):
    def __init__(self, mfm_enabled=True, mfm_cross_scale=True,
                 mfm_output_mode='residual', **kwargs):
        # Initialize encoders/aligners/decoder identically to baseline first.
        super().__init__(**kwargs)
        self.fusion = ConTriNetMFMFusion(mfm_enabled, mfm_cross_scale, mfm_output_mode)

    def forward(self, ct, pet=None, pet_available=None, target_size=None,
                forward_mode='full'):
        if forward_mode not in ('full', 'auto'):
            raise ValueError('ConTriNet experiment is Full-only; Missing is not trained')
        if pet_available is not None:
            state = _validate_state(pet_available, ct.shape[0])
            if not bool(torch.all(state == 1)):
                raise ValueError('Real PET is required for every sample')
        if pet is None or pet.shape != ct.shape:
            raise ValueError('Matching CT and real PET tensors are required')
        return super().forward(ct, pet=pet, target_size=target_size, forward_mode='full')


def build_full_contrinet_model(cfg):
    missing = [name for name in CONFIG_FIELDS if not hasattr(cfg, name)]
    if missing:
        raise ValueError(f'Explicit saved ConTriNet flags are required: {missing}')
    model = FullConTriNetPETCT(
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
    print(f'[full_contrinet] MFM={cfg.mfm_enabled} cross_scale={cfg.mfm_cross_scale} '
          f'output={cfg.mfm_output_mode} '
          f'params_fusion={sum(p.numel() for p in model.fusion.parameters())} '
          f'params_total={sum(p.numel() for p in model.parameters())}', flush=True)
    return {'model': model}
