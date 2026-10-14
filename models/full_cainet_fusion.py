"""Full-only CAINet fusion on the unchanged aligned Add baseline modules."""
import torch

from models.components.cainet_fusion import CAINetFusion
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline, _validate_state

BASELINE_COMMIT = '663f9786122365d8b2774e4497df92bc06b715d5'
UPSTREAM_COMMIT = '4fff413041f1776ce1e588eb4763fe46fca2677a'


class FullCAINetPETCT(DualSharedAddPETCTBaseline):
    def __init__(self, cacr_enabled=True, gcm_enabled=True,
                 cacr_stages=(1, 2, 3, 4), **kwargs):
        # All shared weights receive the exact baseline initialization first.
        super().__init__(**kwargs)
        self.fusion = CAINetFusion(cacr_enabled, gcm_enabled, cacr_stages)

    def forward(self, ct, pet=None, pet_available=None, target_size=None,
                forward_mode='full'):
        if forward_mode not in ('full', 'auto'):
            raise ValueError('FullCAINetPETCT supports Full only; Missing is not trained')
        if pet_available is not None:
            state = _validate_state(pet_available, ct.shape[0])
            if not bool(torch.all(state == 1)):
                raise ValueError('Real PET is required for every sample')
        if pet is None or pet.shape != ct.shape:
            raise ValueError('Matching CT and real PET tensors are required')
        return super().forward(ct, pet=pet, target_size=target_size, forward_mode='full')


def build_full_cainet_model(cfg):
    # Flags must be present: checkpoint evaluation must not silently enable a
    # module that was disabled during training.
    if not hasattr(cfg, 'cacr_enabled') or not hasattr(cfg, 'gcm_enabled'):
        raise ValueError('Explicit saved cacr_enabled/gcm_enabled flags are required')
    model = FullCAINetPETCT(
        cacr_enabled=cfg.cacr_enabled, gcm_enabled=cfg.gcm_enabled,
        cacr_stages=getattr(cfg, 'cacr_stages', (1, 2, 3, 4)),
        ct_backbone=getattr(cfg, 'ct_backbone', 'convnextv2_nano'),
        pet_backbone=getattr(cfg, 'pet_backbone', 'mit_b1'),
        ct_pretrained_path=getattr(cfg, 'ct_pretrained_path', None),
        pet_pretrained_path=getattr(cfg, 'pet_pretrained_path', None),
        in_channels=3, out_channels=1,
        decoder_channels=tuple(getattr(cfg, 'decoder_channels', (512, 256, 128, 64))),
        use_deep_supervision=bool(getattr(cfg, 'use_deep_supervision', False)
                                  or getattr(cfg, 'deep_supervision', False)),
        pretrained=bool(getattr(cfg, 'pretrained', True)))
    print(f'[full_cainet] CACR={cfg.cacr_enabled} stages={model.fusion.cacr_stages} '
          f'GCM={cfg.gcm_enabled} '
          f'params_fusion={sum(p.numel() for p in model.fusion.parameters())} '
          f'params_total={sum(p.numel() for p in model.parameters())}', flush=True)
    return {'model': model}
