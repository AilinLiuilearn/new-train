"""Full-only official-CMX experiment with unchanged baseline backbones/decoder."""
import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline, _validate_state
from models.components.cmx_fusion import CMXFusion

BASELINE_COMMIT = '663f9786122365d8b2774e4497df92bc06b715d5'
CONFIG_FIELDS = ('cmx_ffm_enabled', 'cmx_frm_enabled')


class FullCMXPETCT(DualSharedAddPETCTBaseline):
    def __init__(self, cmx_ffm_enabled=True, cmx_frm_enabled=False, **kwargs):
        if kwargs.get('use_deep_supervision', False):
            raise ValueError('This Full CMX experiment does not use decoder deep supervision')
        super().__init__(**kwargs)
        # Preserve shared baseline initialization and subsequent CPU RNG.
        with torch.random.fork_rng(devices=[]):
            self.fusion = CMXFusion(cmx_ffm_enabled, cmx_frm_enabled)

    def forward(self, ct, pet=None, pet_available=None, target_size=None, forward_mode='full'):
        if forward_mode not in ('full', 'auto'):
            raise ValueError('CMX experiment is Full-only; Missing is not connected')
        if pet_available is not None:
            state = _validate_state(pet_available, ct.shape[0])
            if not bool(torch.all(state == 1)):
                raise ValueError('CMX Full experiment requires real PET for every row')
        if pet is None or pet.shape != ct.shape or pet.device != ct.device:
            raise ValueError('Matching real CT/PET tensors are required')
        return super().forward(ct, pet=pet, target_size=target_size, forward_mode='full')


def build_full_cmx_model(cfg):
    missing = [name for name in CONFIG_FIELDS if not hasattr(cfg, name)]
    if missing:
        raise ValueError(f'Explicit saved CMX flags are required: {missing}')
    model = FullCMXPETCT(
        **{name: getattr(cfg, name) for name in CONFIG_FIELDS},
        ct_backbone=cfg.ct_backbone, pet_backbone=cfg.pet_backbone,
        ct_pretrained_path=cfg.ct_pretrained_path, pet_pretrained_path=cfg.pet_pretrained_path,
        pretrained=bool(cfg.pretrained), decoder_channels=tuple(cfg.decoder_channels),
        use_deep_supervision=bool(cfg.use_deep_supervision or cfg.deep_supervision))
    print(f'[cmx] ffm={cfg.cmx_ffm_enabled} frm={cfg.cmx_frm_enabled} '
          'scales=S1,S2,S3,S4 channels=64,128,320,512 heads=1,2,5,8 reduction=1 '
          'norm=official initialization=official output=replace_add '
          f'params_fusion={sum(p.numel() for p in model.fusion.parameters())} '
          f'params_total={sum(p.numel() for p in model.parameters())}', flush=True)
    return {'model': model}
